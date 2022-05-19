from difftda import SimplexTree
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from LISM import LISM_optimization


class SimplexTreeModel_ISM(tf.keras.Model):

    def __init__(self, p, F, G, stbase="simplextree.txt", dim=0, card=50):
        super(SimplexTreeModel_ISM, self).__init__()
        self.p = p
        self.F = F
        self.G = G
        self.dim = dim
        self.card = card
        self.st = stbase
        
    def call(self):
        d, c = self.dim, self.card
        st, fct1, fct2 = self.st, tf.tensordot(self.F,self.p,1), tf.tensordot(self.G,self.p,1)

        # Turn STPers into a numpy function
        SimplexTreeTF = lambda fct: tf.numpy_function(SimplexTree, 
        [np.array([st], dtype=str), fct, d, c], [tf.int32 for _ in range(2*c)])
        
        # Don't try to compute gradients for the vertex pairs
        fcts1 = tf.reshape(fct1, [1, fct1.shape[0]])
        fcts2 = tf.reshape(fct2, [1, fct2.shape[0]])
        inds1 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(SimplexTreeTF, 
                                                                 fcts1, dtype=[tf.int32 for _ in range(2*c)]))
        inds2 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(SimplexTreeTF, 
                                                                 fcts2, dtype=[tf.int32 for _ in range(2*c)]))
        
        # Get persistence diagram
        dgm1 = tf.reshape(tf.gather_nd(fct1, inds1), [c,2]) 
        dgm2 = tf.reshape(tf.gather_nd(fct2, inds2), [c,2]) 
        return dgm1, dgm2



def adversarial_pipeline(network, data):

    """
    This function encodes the pipeline of the adversarial attack detection.
    Arguments :
        network : a simplicial complex (gudhi.simplexTree) representing the neural network
        data : a tensor np.array of shape (n_inputs, n_nodes, n_features), where n_inputs 
            is the number of images that we classify as normal input or adv. input, n_nodes
            is the number of nodes in the neural network, and n_features is the number of 
            features considered for each node in the simplicial complex. 
    Returns : 
        distance matrix, a np.array of shape (n_inputs,n_inputs))
    """

    N = data.shape[0]

    D = np.zeros((N,N))

    for i, multifilt1 in tqdm(enumerate(data)):
        for j, multifilt2 in enumerate(data):
            if i<j:
                # random intial projection 
                theta = np.random.rand()*np.pi/2
                p_ = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32).reshape(2,1)
                p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)

                # converting multifiltrations to tensorflow non-trainable variables
                m1 = tf.Variable(initial_value=multifilt1, trainable=False, dtype=tf.float32)
                m2 = tf.Variable(initial_value=multifilt2, trainable=False, dtype=tf.float32)

                # initializing the input-model
                model = SimplexTreeModel_ISM(p, m1, m2, stbase = network, dim=1)
                dist = LISM_optimization(model, fast=True)
                D[i,j] = dist

    D += np.transpose(D)



