import tensorflow as tf
import numpy as np

class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, width, activation, kernel_initializer, bias_initializer):
        super(MaskLayer, self).__init__()
        self.act = tf.keras.activations.get(activation)
        self.w = tf.Variable(
            initial_value=kernel_initializer(shape=(input_dim, width), dtype="float32"),
            trainable=True)
        
        self.b = tf.Variable(
            initial_value=bias_initializer(shape=(width,), dtype="float32"), trainable=True)
        
        self.mask = tf.Variable(
            initial_value=tf.ones((input_dim,width)), trainable=False)
        
    def call(self, inputs):
        return self.act(tf.matmul(inputs, tf.multiply(self.w, self.mask)) + self.b)
        
    def pruneLowMagnitude(self, sparsity):
        sparsityPercent = sparsity*100
        print('PRUNING %s percent of weights' %(sparsityPercent))
        numTotal = self.w.shape[0]*self.w.shape[1] #Total number of weights in layer
        numPrune = np.rint(sparsity*numTotal) #Number of weights we want to prune 
        
        #Finds the lowest numPrune weights and their indices 
        result = tf.math.top_k(tf.negative(tf.reshape(self.w, [-1])), k=numPrune)
        idx = tf.reshape(result.indices, [tf.Variable(numPrune, dtype='int32'),1])
        
        #Flatten mask for manipulation
        flatMask = tf.reshape(self.mask, [-1])
        fm = tf.reshape(flatMask, [len(flatMask), 1])
        
        #Update the mask
        num_updates, index_depth = idx.shape.as_list()
        fm = tf.tensor_scatter_nd_update(fm, idx, tf.zeros([num_updates,1], tf.float32))
        
        #Set self.mask to updated mask 
        fm = tf.reshape(fm, self.mask.shape)
        self.mask = fm
        
    def pruneShapeNet(self, sparsity, si_dim, n_sx, l_sx, so_dim):
        sparsityPercent = sparsity*100
        print('PRUNING %s percent of weights of ShapeNet' %(sparsityPercent))
        
        '''
        ShapeNet is completely determined by the last layer of ParameterNet. The weight 
        matrix to prune is latent_dim by po_dim (ParameterNet output dimension or number
        of weights and biases in ShapeNet).
        '''
        sn_weights = self.w[:,:si_dim*n_sx + l_sx*n_sx**2 + so_dim*n_sx]  
        avgw = tf.reduce_mean(sn_weights, axis=0) #Take average of weights because we want to prune entire node from ShapeNet
        
        numTotal = avgw.shape[0] #Total number of weights in layer
        numPrune = np.rint(sparsity*numTotal) #Number of weights we want to prune 
        
        #Finds the lowest numPrune weights and their indices 
        result = tf.math.top_k(tf.negative(avgw), k=numPrune)
        idx = tf.reshape(result.indices, [tf.Variable(numPrune, dtype='int32'),1])
        
        #Flatten mask for manipulation
        flatMask = self.mask[0,:]
        fm = tf.reshape(flatMask, [len(flatMask), 1])
        
        #Update the mask
        num_updates, index_depth = idx.shape.as_list()
        fm = tf.transpose(tf.tensor_scatter_nd_update(fm, idx, tf.zeros([num_updates,1], tf.float32)))
        
        self.mask = tf.repeat(fm, [self.w.shape[0]], axis=0)
        print(self.mask[:,0:300])
        
    def pruneOtherWay(self):
        pass
        