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
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/PU/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
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

    
    NSWD = dataset_config['NSWD'] #SWD is calculated considering only NSWD features

    train_data = [utils.DataLoader(os.path.join(flags.data_folder,'train_'+dataset),flags.nevts)[0] for dataset in dataset_config['FILES']]
    train_data = np.concatenate(train_data)
    train_label = [utils.DataLoader(os.path.join(flags.data_folder,'train_'+dataset),flags.nevts)[1] for dataset in dataset_config['FILES']]
    train_label = np.concatenate(train_label)


    val_data = [utils.DataLoader(os.path.join(flags.data_folder,'val_'+dataset),flags.nevts)[0] for dataset in dataset_config['FILES']]
    val_data = np.concatenate(val_data)
    val_label = [utils.DataLoader(os.path.join(flags.data_folder,'val_'+dataset),flags.nevts)[1] for dataset in dataset_config['FILES']]
    val_label = np.concatenate(val_label)
                
    train_data = utils.ApplyPrep(preprocessing,train_data)
    val_data = utils.ApplyPrep(preprocessing,val_data)
    train_label = utils.ApplyPrep(preprocessing,train_label)
    val_label = utils.ApplyPrep(preprocessing,val_label)

    BATCH_SIZE = dataset_config['BATCH']
    train = tf.data.Dataset.from_tensor_slices((train_data,np.concatenate(
        [train_data[:,:,:NSWD],train_label[:,:,:NSWD]],-1))).cache().shuffle(50*BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val = tf.data.Dataset.from_tensor_slices((val_data,np.concatenate(
        [val_data[:,:,:NSWD],val_label[:,:,:NSWD]],-1))).cache().shuffle(50*BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    del train_data, val_data, train_label, val_label
    gc.collect()
    

    LR = float(dataset_config['LR'])*np.sqrt(hvd.size())
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    EARLY_STOP = dataset_config['EARLYSTOP']
    inputs,outputs = ABCNet(nfeat=dataset_config['SHAPE'][2])
    model = Model(inputs=inputs,outputs=outputs)
    #opt = keras.optimizers.Adam(learning_rate=LR)
    opt = keras.optimizers.Lion(learning_rate=LR,beta_1=0.95)

    
    opt = hvd.DistributedOptimizer(opt)
    model.compile(loss=SWD,
                  #run_eagerly=True,
                  optimizer=opt,experimental_run_tf_function=False)
    if flags.load:
        model.load_weights(checkpoint_folder)

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        ReduceLROnPlateau(patience=10, min_lr=1e-7,verbose=hvd.rank()==0),
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
    ]

    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,save_best_only=True,mode='auto',
                                     period=1,save_weights_only=True)
        
        callbacks.append(checkpoint)
        print(model.summary())


    history = model.fit(
        train,
        epochs=NUM_EPOCHS,
        # steps_per_epoch=1,
        validation_data=val,
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )




