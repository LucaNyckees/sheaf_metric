from difftda import Cubical
import tensorflow as tf
import numpy as np
import gudhi as gd

class CubicalModel_ISM(tf.keras.Model):
    def __init__(self, p, I1, I2, dim=1, card=50):
        super(CubicalModel_ISM, self).__init__()
        self.p = p
        self.I1 = I1
        self.I2 = I2
        self.dim = dim
        self.card = card
        
    def call(self):

        Xp = tf.reshape(tf.tensordot(self.I1,self.p[0],1),shape=[28,28])
        Yp = tf.reshape(tf.tensordot(self.I2,self.p[1],1),shape=[28,28])

        d, c, D = self.dim, self.card, len(Xp.shape)
        XX = tf.reshape(Xp, [1, Xp.shape[0], Xp.shape[1]])
        YY = tf.reshape(Yp, [1, Yp.shape[0], Yp.shape[1]])
        
        # Turn numpy function into tensorflow function
        CbTF1 = lambda Xp: tf.numpy_function(Cubical, [Xp, d, c], [tf.int32 for _ in range(2*D*c)])
        CbTF2 = lambda Yp: tf.numpy_function(Cubical, [Yp, d, c], [tf.int32 for _ in range(2*D*c)])
        
        # Compute pixels associated to positive and negative simplices 
        # Don't compute gradient for this operation
        
        inds1 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF1,XX,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))
        inds2 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF2,YY,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))

        
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = tf.reshape(tf.gather_nd(Xp, tf.reshape(inds1, [-1,D])), [-1,2])
        dgm2 = tf.reshape(tf.gather_nd(Yp, tf.reshape(inds2, [-1,D])), [-1,2])
        return dgm1, dgm2


