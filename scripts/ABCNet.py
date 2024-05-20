import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow.keras.backend as K




def get_neighbors(points,features,projection_dim,K):
    drij = pairwise_distance(points)  # (N, P, P)
    _, indices = tf.nn.top_k(-drij, k=K + 1)  # (N, P, K+1)
    indices = indices[:, :, 1:]  # (N, P, K)
    knn_fts = knn(tf.shape(points)[1], K, indices, features)  # (N, P, K, C)
    knn_fts_center = tf.broadcast_to(tf.expand_dims(features, 2), tf.shape(knn_fts))
    local = tf.concat([knn_fts-knn_fts_center,knn_fts_center],-1)
    local = layers.Dense(2*projection_dim,activation='gelu')(local)
    local = layers.Dense(projection_dim,activation='gelu')(local)    
    return local


def pairwise_distance(point_cloud):
    r = tf.reduce_sum(point_cloud * point_cloud, axis=2, keepdims=True)
    m = tf.matmul(point_cloud, point_cloud, transpose_b = True)
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1)) + 1e-5
    return D


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)    
    batch_size = tf.shape(features)[0]

    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, num_points, k))
    indices = tf.stack([batch_indices, topk_indices], axis=-1)
    return tf.gather_nd(features, indices)


def get_encoding(x,projection_dim,use_bias=True):
    x = layers.Dense(2*projection_dim,use_bias=use_bias,activation='gelu')(x)
    x = layers.Dense(projection_dim,use_bias=use_bias,activation='gelu')(x)
    return x


def ABCNet(nfeat=1,
           k = 20,
           projection_dim = 64,
           nlayers = 3):
    inputs = Input(shape=(None,nfeat))
    mask = tf.cast(inputs[:,:,2,None]!=0, dtype='float32')

    encoded = get_encoding(inputs,projection_dim)
    coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0.0), dtype='float32'))
    edges = get_neighbors(coord_shift+inputs[:,:,:2],inputs,projection_dim,k)

    skip_connection = encoded
    for i in range(nlayers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)        
        update_particles, edges = GAP(x1,edges,projection_dim)

        x2 = layers.Add()([update_particles,encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        encoded = layers.Add()([x3,x2])*mask

    encoded = encoded + skip_connection
    outputs = layers.Dense(1,activation='sigmoid')(encoded)
    return inputs,outputs


def GAP(points,
        edges,
        projection_dim    
):

    updates_points = layers.Dense(2*projection_dim,activation='gelu')(points)
    updates_points = layers.LayerNormalization(epsilon=1e-6)(updates_points)
    updates_points = layers.Dense(1)(updates_points)
    
    updates_edges = layers.Dense(2*projection_dim,activation='gelu')(edges)
    updates_edges = layers.LayerNormalization(epsilon=1e-6)(updates_edges)
    updates_edges = layers.Dense(1)(updates_edges)

    logits = updates_points[:,:,None] + updates_edges
    logits = tf.transpose(logits, [0, 1, 3, 2])
    att_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits,alpha=0.01))
    updates = tf.linalg.matmul(att_coefs, updates_edges)[:,:,0] #removing useless dimension
    
    return updates, updates_edges
    

def SWD(y_true, y_pred,nprojections=128,use_charge=True):
    pu_pfs = y_true[:,:,:y_true.shape[2]//2]
    nopu_pfs = y_true[:,:,y_true.shape[2]//2:]


    if use_charge:
        #Calculate the loss separately for charged and neutral particles
        charge_pu_mask = tf.cast(tf.expand_dims(tf.abs(pu_pfs[:,:,-1])>0,-1),tf.float32)
        charge_nopu_mask = tf.cast(tf.expand_dims(tf.abs(nopu_pfs[:,:,-1])>0,-1),tf.float32)

    #First 4 features are only the event kinematics
    nopu_pfs = nopu_pfs[:,:,:4]
    pu_pfs = pu_pfs[:,:,:4]*y_pred


    def _getSWD(pu_pf,nopu_pf):    
        proj = tf.random.normal(shape=[tf.shape(pu_pf)[0],tf.shape(pu_pf)[2], nprojections])
        proj *= tf.math.rsqrt(tf.reduce_sum(tf.square(proj), 1, keepdims=True))

        p1 = tf.matmul(nopu_pf, proj) #BxNxNPROJ
        p2 = tf.matmul(pu_pf, proj) #BxNxNPROJ
        p1 = sort_rows(p1, tf.shape(pu_pf)[1])
        p2 = sort_rows(p2, tf.shape(pu_pf)[1])
        
        wdist = tf.reduce_mean(tf.square(p1 - p2),-1)
        return wdist
    
    def _getMET(particles):
        px = tf.abs(particles[:,:,2])*tf.math.cos(particles[:,:,1])
        py = tf.abs(particles[:,:,2])*tf.math.sin(particles[:,:,1])
        met = tf.stack([px,py],-1)
        return met


    met_pu = tf.reduce_sum(_getMET(pu_pfs)*y_pred,1)
    met_nopu = tf.reduce_sum(_getMET(nopu_pfs),1)
    met_mse = tf.reduce_sum(tf.square(met_pu[:,:2] - met_nopu[:,:2]),-1)


    if use_charge:
        wdist_charge = _getSWD(pu_pfs*charge_pu_mask,nopu_pfs*charge_nopu_mask)
        wdist_neutral = _getSWD(pu_pfs*tf.cast(charge_pu_mask==0,tf.float32),
                                nopu_pfs*tf.cast(charge_nopu_mask==0,tf.float32))
        wdist = wdist_charge + wdist_neutral
    else:
        wdist = _getSWD(pu_pfs,nopu_pfs)
        
    notzero = tf.reduce_sum(tf.where(wdist>0,tf.ones_like(wdist),tf.zeros_like(wdist)))    
    return 1e3*tf.reduce_sum(wdist)/tf.reduce_sum(notzero) + tf.reduce_mean(met_mse)

    
def sort_rows(matrix, num_rows):
    matrix_T = tf.transpose(matrix, [0,2,1])
    sorted_matrix_T,index_matrix = tf.math.top_k(matrix_T, num_rows)    
    return tf.transpose(sorted_matrix_T, [0,2, 1])
