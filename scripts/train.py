import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import h5py as h5
import utils
from ABCNet import ABCNet, SWD
import gc
tf.random.set_seed(1)


if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')



    parser = argparse.ArgumentParser()
        
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/PU/vertex_info', help='Folder containing data and MC files')
    # parser.add_argument('--data_folder', default='/global/cscratch1/sd/vmikuni/PU', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/PU/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--sup', action='store_true', default=False,help='Do a supervised training')
    flags = parser.parse_args()
    dataset_config = utils.LoadJson(flags.config)
    preprocessing = utils.LoadJson(dataset_config['PREPFILE'])

    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(dataset_config['CHECKPOINT_NAME'])
    if hvd.rank()==0:
        backup = '../checkpoints_{}'.format(dataset_config['CHECKPOINT_NAME'])
        if not os.path.exists(backup):
            os.makedirs(backup)
        os.system('cp ABCNet.py {}'.format(backup)) # bkp of model def
        os.system('cp {} {}'.format(flags.config,backup)) # bkp of config file
        #model.save_weights('{}/{}'.format(checkpoint_folder,'checkpoint'),save_format='tf')

    
    #NSWD = dataset_config['NSWD'] #SWD is calculated considering only NSWD features
    NPART = dataset_config['NPART'] #SWD is calculated considering only NSWD features
    for iset, dataset in enumerate(dataset_config['FILES']):
        data_,label_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),flags.nevts)

        if iset ==0:
            data = data_[:,:NPART]
            label = label_[:,:NPART]
        else:
            data = np.concatenate((data,data_[:,:NPART]),0)
            label=np.concatenate((label,label_[:,:NPART]),0)

    
    #utils.CalcPrep(dataset_config['PREPFILE'],data)
    data = utils.ApplyPrep(preprocessing,data)
    data_size = data.shape[0]
    
    if not flags.sup:
        label = utils.ApplyPrep(preprocessing,label)            
        # data_label = data[:,:,:NSWD].copy()
        # label = label[:,:,:NSWD]
        data_label = data[:,:,:8].copy()
        label = label[:,:,:8]
        label = np.concatenate([data_label,label],-1)
        loss = SWD
        del data_label
    else:
        label = data[:,:,8]
        #loss = tf.losses.BinaryCrossentropy(from_logits=False)
        loss = tf.losses.MeanSquaredError()
        
        
    dataset = tf.data.Dataset.from_tensor_slices((data,label))

        
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, data, label
    gc.collect()
    
    BATCH_SIZE = dataset_config['BATCH']
    LR = dataset_config['LR']
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    EARLY_STOP = dataset_config['EARLYSTOP']
    inputs,outputs = ABCNet(npoint=NPART,nfeat=dataset_config['SHAPE'][2])
    model = Model(inputs=inputs,outputs=outputs)

    if flags.load:
        model.load_weights(checkpoint_folder)


    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=LR*hvd.size(),
        decay_steps=NUM_EPOCHS*int(data_size*flags.frac/BATCH_SIZE)
    )
    
    #opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)    
    opt = keras.optimizers.Adam(learning_rate=LR)
    
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)


    
    model.compile(loss=loss,
                  #run_eagerly=True,
                  optimizer=opt,experimental_run_tf_function=False)

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        # ReduceLROnPlateau(patience=10, factor=0.5,
        #                   min_lr=1e-8,verbose=hvd.rank()==0),
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
    ]

    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,save_best_only=True,mode='auto',
                                     period=1,save_weights_only=True)
        
        callbacks.append(checkpoint)
        print(model.summary())

    # print(int(data_size*flags.frac/BATCH_SIZE),int(data_size*(1-flags.frac)/BATCH_SIZE))
    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        # steps_per_epoch=1,
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )




