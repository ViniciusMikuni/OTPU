import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def pairwise_distance(point_cloud): 
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape()[0]
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)  # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1,
                                       keepdims=True)  # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def pairwise_distanceR(point_cloud, mask):
    """Compute pairwise distance in the eta-phi plane for the point cloud.
    Uses the third dimension to find the zero-padded terms
    Args:
      point_cloud: tensor (batch_size, num_points, 2)
      IMPORTANT: The order should be (eta, phi)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape()[0]
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud = point_cloud[:, :, :2]  # Only use eta and phi, BxNx2
    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) #Bx2xN
    point_cloud_phi = point_cloud_transpose[:, 1:, :] #Bx1xN
    point_cloud_phi = tf.tile(point_cloud_phi, [1, point_cloud_phi.get_shape()[2], 1]) #BxNxN
    point_cloud_phi_transpose = tf.transpose(point_cloud_phi, perm=[0, 2, 1]) #BxNxN
    point_cloud_phi = tf.math.abs(point_cloud_phi - point_cloud_phi_transpose) #compute distance in the phi space for all the particles in the cloud (more precisely, its abs)
    is_biggerpi = tf.greater_equal(tf.abs(point_cloud_phi), np.pi) #is the abs greater than pi?
    point_cloud_phi_corr = tf.where(is_biggerpi, 2 * np.pi - point_cloud_phi, point_cloud_phi) #Correct if bigger than pi
    #build matrix of pairwise DeltaRs between particles
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)  # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1,
                                       keepdims=True)  # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    deltaR_matrix = point_cloud_square + point_cloud_square_transpose + point_cloud_inner #this matrix contains the squared, pairwise DRs between particles in the cloud
    deltaR_matrix = deltaR_matrix - tf.square(point_cloud_phi) #subtract non-corrected delta_phi squared part
    deltaR_matrix = deltaR_matrix + tf.square(point_cloud_phi_corr) #add corrected delta_phi squared part
    #Move zero-padded away
    point_shift = 1000*tf.expand_dims(mask,-1) #BxNx1
    point_shift_transpose = tf.transpose(point_shift,perm=[0, 2, 1]) #Bx1xN
    zero_mask = point_shift_transpose + point_shift #when adding tensors having a dimension equal to 1, tf tiles them to make them compatible for the sum
    zero_mask = tf.where(tf.equal(zero_mask, 2000), tf.zeros_like(zero_mask), zero_mask)
    return deltaR_matrix + zero_mask

def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    distances, nn_idx = tf.math.top_k(neg_adj, k=k)  # values, indices
    return nn_idx,distances


def get_neighbors(point_cloud, nn_idx, k=20):
    """Construct neighbors feature for each point
      Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int
      Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """    
    point_cloud = tf.squeeze(point_cloud, axis=-2)
    point_cloud_shape = tf.shape(point_cloud)
    point_cloud_shape_int = point_cloud.get_shape()
    batch_size = point_cloud_shape[0]
    num_points = point_cloud_shape[1]
    num_dims = point_cloud_shape_int[2]

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    
    return point_cloud_neighbors


class AttnFeat(layers.Layer):

    def __init__(self, k=10, filters = 32,                 
                 activation_NoBias=tf.keras.activations.relu,
                 activation=tf.nn.leaky_relu,
                 expand_dims=True, name='AttnFeat', **kwargs): 
        super(AttnFeat, self).__init__(name=name, **kwargs)
        self.k = k
        self.activation = activation
        self.filters = filters
        self.activation_NoBias = activation_NoBias
        self.expand_dims = expand_dims

        #self.BatchNormNoBias = layers.BatchNormalization(momentum=self.momentum)
        self.BatchNormNoBias = tfa.layers.GroupNormalization(groups=4)

        self.Conv2DEdgeFeat = layers.Conv2D(filters=self.filters, kernel_size=1)

        #self.BatchNormEdgeFeat = layers.BatchNormalization(momentum=self.momentum)
        self.BatchNormEdgeFeat = tfa.layers.GroupNormalization(groups=4)

        self.Conv2DSelfAtt = layers.Conv2D(filters=1, kernel_size=1)
        self.BatchNormSelfAtt = layers.BatchNormalization(center=False, scale=False)
        

        self.Conv2DNeighAtt = layers.Conv2D(filters=1, kernel_size=1)
        self.BatchNormNeighAtt = layers.BatchNormalization(center=False, scale=False)
        
        
        self.Conv2DNoBias = layers.Conv2D(filters=self.filters,
                                          use_bias=False,kernel_size=1,
                                          activation=self.activation_NoBias) 

    def call(self, inputs, training=None, **kwargs): 
        
        #Implement the operations described in the ABCNet paper (https://link.springer.com/article/10.1140/epjp/s13360-020-00497-3)

        nn_idx = kwargs['nn_idx']
        mask = kwargs['mask']        
        if self.expand_dims:
            inputs = tf.expand_dims(inputs, axis=-2)

        mask_neighbors = get_neighbors(tf.reshape(mask,(-1,tf.shape(mask)[1],1,1)),
                                       nn_idx=nn_idx,
                                       k=self.k)  # Group up the neighbors using the index passed on the arguments
        
        mask_neighbors = -10000*tf.transpose(mask_neighbors,(0,1,3,2))
        
        neighbors = get_neighbors(inputs, nn_idx=nn_idx,k=self.k)  


        inputs_tiled = tf.tile(inputs, [1, 1, self.k, 1])
        edge_feature_pre = inputs_tiled - neighbors  # Make the edge features yij

        if 'deltaR' in kwargs:
            deltaR = kwargs['deltaR']
            edge_feature_pre = tf.concat([edge_feature_pre,
                                          tf.expand_dims(
                                              tf.math.divide_no_nan(inputs_tiled[:,:,:,2],deltaR),-1),
                                          tf.expand_dims(deltaR,-1)],-1)

        new_feature = self.Conv2DNoBias(inputs)
        new_feature = self.BatchNormNoBias(new_feature)

        edge_feature = self.Conv2DEdgeFeat(edge_feature_pre)
        edge_feature = self.BatchNormEdgeFeat(edge_feature)

        self_attention = self.Conv2DSelfAtt(new_feature)
        self_attention = self.BatchNormSelfAtt(self_attention)

        neighbor_attention = self.Conv2DNeighAtt(edge_feature)
        neighbor_attention = self.BatchNormNeighAtt(neighbor_attention)
        
        logits = self_attention + neighbor_attention
        logits = tf.transpose(logits, [0, 1, 3, 2])
        
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits)+mask_neighbors)
                
        vals = tf.linalg.matmul(coefs, edge_feature)

        outputs = self.activation(vals)
        return outputs, edge_feature


class GAPBlock(layers.Layer):

    def __init__(self, nheads=1, k=10, filters=32,
                 activation_NoBias=tf.keras.activations.relu,
                 activation=tf.keras.activations.relu,
                 expand_dims=True,
                 name='GAPBlock', **kwargs):  
        super(GAPBlock, self).__init__(name=name, **kwargs)
        self.k = k
        self.nheads = nheads
        self.attn_feat_layers = []
        self.Name=name
        for i in range(nheads):
            self.attn_feat_layers.append(
                AttnFeat(k=self.k, filters=filters, 
                         expand_dims=expand_dims,
                         activation_NoBias=activation_NoBias,
                         activation=activation)
            )

    def call(self, inputs, training=None, **kwargs): 
        nn_idx, layer_input,  mask = inputs 
        attns = []
        local_features = []
        for i in range(self.nheads):
            out, edge_feat = self.attn_feat_layers[i](inputs=layer_input,nn_idx=nn_idx, mask=mask,**kwargs)
            attns.append(out)  # This is the edge feature * att. coeff. activated by Leaky RELU, one per particle
            local_features.append(edge_feat)  # Those are the yij

        neighbors_features = tf.reduce_mean(tf.concat(attns, axis=-1),-2)
        locals_transform = tf.reduce_mean(tf.concat(local_features, axis=-1), axis=-2)
        return neighbors_features, locals_transform




def ABCNet(npoint,nfeat=1):
    # Define the shapes of the multidimensionanl inputs for the pointcloud (per particle variables)
    # Always leave out the batchsize when specifying the shape
    inputs = Input(shape=(npoint,nfeat))
    k = 20 
    mask = inputs[:,:,-1]
    masked_inputs = layers.Masking(mask_value=0.0)(inputs[:,:,:-1])
    adj_1 = pairwise_distanceR(inputs[:,:,:3], mask)
    nn_idx,dist = knn(adj_1, k=k)
    
    idx_list = list(range(nfeat))

    chs = tf.where((masked_inputs[:,:,2]!=0)&(masked_inputs[:,:,7]==0),
                   -tf.ones_like(masked_inputs[:,:,7]),
                   tf.zeros_like(masked_inputs[:,:,7]))

    chs = chs + tf.cast(masked_inputs[:,:,8]>0,tf.float32)*tf.abs(masked_inputs[:,:,7])    
    chs = tf.expand_dims(chs,-1)
    
    idx_list.pop(8) #hardfrac is a truth label available only in delphes
    idx_list.pop(6) #delete puppi from the list
    idx_list.pop(1) #delete phi from the feature list after calculating the distances

    #x = tf.gather(masked_inputs,idx_list,axis=-1)
    x = layers.Dense(128,activation=None)(tf.gather(tf.concat([masked_inputs,chs],-1),idx_list,axis=-1))
    x = layers.LeakyReLU(alpha=0.01)(layers.Dense(64,activation=None)(x))
    input_x = x
    num_layers = 2
    for i in range(num_layers):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        if i ==0:
            x = GAPBlock(k=k, filters=64, name='Gap{}'.format(i))((nn_idx,x,mask),deltaR=dist)[0] + x
        else:
            x = GAPBlock(k=k, filters=64, name='Gap{}'.format(i))((nn_idx,x,mask))[0] + x
        x = layers.Dense(128,activation='gelu')(x)
        x = layers.Dense(64,activation='gelu')(x)
        
        # adj = pairwise_distance(x)
        # nn_idx,dist = knn(adj, k=k)


    #perform aggregation. Aggregation is a concat tf.operation    
    # x = tf.concat(to_combine, axis = -1)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)    
    # x = layers.Dense(512, activation='relu')(x)        
    # x_prime = x    
    # x = tf.reduce_mean(x, axis=1,keepdims=True)    
    # expand=tf.tile(x, [1, npoint, 1]) #after pooling, recover x tensor's second dimension by tiling
    # x = tf.concat(values = [expand, x_prime], axis=-1)

        
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.LeakyReLU(alpha=0.01)(layers.Dense(256,activation=None)(x + input_x))
    x = layers.LeakyReLU(alpha=0.01)(layers.Dense(128,activation=None)(x))
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return inputs,outputs



    # neighbors_features_1, graph_features_1, attention_features_1 = GAPBlock(k=k, filters_C2DNB=32, padding_C2DNB = 'valid', name='Gap1')((nn_idx,x,mask),deltaR=dist)
    # x = layers.Conv1D(filters = 64, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(neighbors_features_1)
    # x = layers.Conv1D(filters = 32, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = layers.BatchNormalization(momentum=momentum)(x)
    # x01=x
    
        
    # neighbors_features_2, graph_features_2, attention_features_2 = GAPBlock(k=k, momentum=momentum, filters_C2DNB=64, padding_C2DNB = 'valid', name='Gap2')((nn_idx,x, mask))
    # x = layers.Conv1D(filters = 128, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(neighbors_features_2)        
    # x = layers.Conv1D(filters = 64, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = layers.BatchNormalization(momentum=momentum)(x)
    # x11 = x
    
    # #perform aggregation. Aggregation is a concat tf.operation
    # x = tf.concat([x01, x11, graph_features_1, graph_features_2], axis = -1)

    # x = layers.Conv1D(filters = 512, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)        
    # x = layers.BatchNormalization(momentum=momentum)(x)

    # x_prime = x
    
    # #Perform AveragePooling
    # x = tf.reduce_mean(x, axis=1,keepdims=True)
    
    # expand=tf.tile(x, [1, npoint, 1]) #after pooling, recover x tensor's second dimension by tiling

    # x = tf.concat(values = [expand, x_prime], axis=-1)
    # #x = expand + x_prime


    # x = layers.Conv1D(filters = 256, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = layers.Conv1D(filters = 64, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    # outputs = layers.Conv1D(filters = 1, kernel_size = 1, strides = 1,  padding='valid', kernel_initializer='glorot_uniform', activation='sigmoid')(x)

    # return inputs,outputs


#@tf.function
def SWD(y_true, y_pred,nprojections=128,NSWD=4):
    pu_pfs = y_true[:,:,:y_true.shape[2]//2]
    nopu_pfs = y_true[:,:,y_true.shape[2]//2:]

    charge_pu_mask = tf.cast(tf.expand_dims(tf.abs(pu_pfs[:,:,-1])>0,-1),tf.float32)
    charge_nopu_mask = tf.cast(tf.expand_dims(tf.abs(nopu_pfs[:,:,-1])>0,-1),tf.float32)

    def _get_cartesian(particles):
      mask = tf.cast(particles[:,:,2]!=0,tf.float32)
      px = tf.exp(particles[:,:,2])*tf.math.cos(particles[:,:,1])*mask
      py = tf.exp(particles[:,:,2])*tf.math.sin(particles[:,:,1])*mask
      pz = tf.exp(particles[:,:,2])*tf.math.sinh(particles[:,:,0])*mask
      
      vec = tf.stack([px,py,pz],-1)
      return vec


    
    def _getSWD(pu_pf,nopu_pf):    
        proj = tf.random.normal(shape=[tf.shape(pu_pf)[0],tf.shape(pu_pf)[2], nprojections])
        proj *= tf.math.rsqrt(tf.reduce_sum(tf.square(proj), 1, keepdims=True))

        p1 = tf.matmul(nopu_pf, proj) #BxNxNPROJ
        p2 = tf.matmul(pu_pf, proj) #BxNxNPROJ
        p1 = sort_rows(p1, tf.shape(pu_pf)[1])
        p2 = sort_rows(p2, tf.shape(pu_pf)[1])
        
        wdist = tf.reduce_mean(tf.square(p1 - p2),-1)
        return wdist
    
    
    #get back the particles in cartesian coordinates
    # met_pu = tf.reduce_sum(_get_cartesian(pu_pfs)[:,:,:2]*y_pred,1)
    # met_nopu = tf.reduce_sum(_get_cartesian(nopu_pfs)[:,:,:2],1)
    # met_mse = tf.reduce_sum(tf.square(met_pu - met_nopu),-1)

    pu_pfs = pu_pfs[:,:,:NSWD]*y_pred
    nopu_pfs = nopu_pfs[:,:,:NSWD]

    # pu_pfs = _get_cartesian(pu_pfs[:,:,:NSWD])*y_pred
    # nopu_pfs = _get_cartesian(nopu_pfs[:,:,:NSWD])
    
    wdist = _getSWD(pu_pfs,nopu_pfs)
    
    return  1e3*tf.reduce_mean(wdist)

    # 1e3*tf.reduce_mean(puppi_loss) +
    
    #     wdist = _getSWD(pu_pfs,nopu_pfs)
    #     notzero = tf.reduce_sum(tf.where(wdist>0,tf.ones_like(wdist),tf.zeros_like(wdist)))    
    #     return 1e3*tf.reduce_sum(wdist)/tf.reduce_sum(notzero)
    # # #+tf.reduce_mean(met_mse)

    wdist_charge = _getSWD(pu_pfs*charge_pu_mask,nopu_pfs*charge_nopu_mask)
    wdist_neutral = _getSWD(pu_pfs*tf.cast(charge_pu_mask==0,tf.float32),
                            nopu_pfs*tf.cast(charge_nopu_mask==0,tf.float32))

    return 1e3*tf.reduce_mean(wdist_charge) + 1e3*tf.reduce_mean(wdist_neutral)



#+ tf.reduce_mean(met_mse)/1e2

    
def sort_rows(matrix, num_rows):
    matrix_T = tf.transpose(matrix, [0,2,1])
    sorted_matrix_T,index_matrix = tf.math.top_k(matrix_T, num_rows)    
    return tf.transpose(sorted_matrix_T, [0,2, 1])
