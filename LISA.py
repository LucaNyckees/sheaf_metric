import tensorflow as tf
import numpy as np

# I : put in argument, not global variable
# import the Cubical method (?)

class CubicalModel_ISM_norm(tf.keras.Model):
    def __init__(self, p, dim=1, card=50):
        super(CubicalModel_ISM_norm, self).__init__()
        self.p = p
        self.dim = dim
        self.card = card
        
    def call(self):

        Ip = tf.reshape(tf.tensordot(I,self.p,1),shape=[28,28])

        d, c, D = self.dim, self.card, len(Ip.shape)
        XX = tf.reshape(Ip, [1, Ip.shape[0], Ip.shape[1]])
        
        # Turn numpy function into tensorflow function
        CbTF = lambda Ip: tf.numpy_function(Cubical, [Ip, d, c], [tf.int32 for _ in range(2*D*c)])

        # Compute pixels associated to positive and negative simplices 
        # Don't compute gradient for this operation
        inds = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,XX,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))
        
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm = tf.reshape(tf.gather_nd(Ip, tf.reshape(inds, [-1,D])), [-1,2])

        return dgm


def ISM_norm(I, use_reg=True, more_info = False):

    p_ = tf.constant([0.5, 0.5], shape=[2, 1])
    p = tf.Variable(initial_value=np.array(p_, dtype=np.float32), trainable=True)

    model = CubicalModel_ISM_norm(p, dim=1, card=256)

    lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-3, decay_steps=10, decay_rate=.01)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    sigma = 0.001
    lambda_ = 100

    amplitudes = [] 
    losses = []
    projections = []
    alpha = 10.

    optimization = {'losses':[],
                             'amplitudes':[],
                             'projections':[],
                             'diagrams':[]
    }

    # reduce the nb of epochs to 30+1 for faster computations 
    for epoch in range(10+1):
        
        with tf.GradientTape() as tape:
            
            dgm = model.call()
            #amplitude = alpha * tf.square(wasserstein_distance(dgm, [], order=2, enable_autodiff=True))
            if use_reg:
                amplitude = - alpha * tf.math.reduce_sum(tf.abs(dgm[:,1]-dgm[:,0]))
                loss = amplitude + lambda_*(tf.abs(tf.norm(p)-1))
            else:
                amplitude = - alpha * tf.math.reduce_sum(tf.abs(dgm[:,1]-dgm[:,0]))
                loss = amplitude
            #print(amplitude)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        
        np.random.seed(epoch)
        # gradients = [tf.convert_to_tensor(gradients[0])]
        gradients[0] = gradients[0] + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        optimization['losses'].append(loss.numpy())
        optimization['amplitudes'].append(amplitude.numpy())
        optimization['projections'].append(p.numpy())
        optimization['diagrams'].append(dgm)

    return amplitude.numpy(), optimization