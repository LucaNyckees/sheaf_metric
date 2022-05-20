import tensorflow as tf
import numpy as np
import gudhi as gd

# I : put in argument, not global variable
# import the Cubical method (?)

def Cubical(X, dim, card):
    # Parameters: X (image),
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    cc = gd.CubicalComplex(dimensions=X.shape, top_dimensional_cells=X.flatten())
    cc.persistence()
    try:
        cof = cc.cofaces_of_persistence_pairs()[0][dim]
    except IndexError:
        cof = np.array([])
        
    Xs = X.shape

    if len(cof) > 0:
        # Sort points with distance-to-diagonal
        pers = [X[np.unravel_index(cof[idx,1], Xs)] - X[np.unravel_index(cof[idx,0], Xs)] for idx in range(len(cof))]
        perm = np.argsort(pers)
        cof = cof[perm[::-1]]
    
    # Retrieve and ouput image indices/pixels corresponding to positive and negative simplices
    D = len(Xs)
    ocof = np.array([0 for _ in range(D*card*2)])
    count = 0
    for idx in range(0,min(2*card, 2*cof.shape[0]),2):
        ocof[D*idx:D*(idx+1)]     = np.unravel_index(cof[count,0], Xs)
        ocof[D*(idx+1):D*(idx+2)] = np.unravel_index(cof[count,1], Xs)
        count += 1
    return list(np.array(ocof, dtype=np.int32))









class CubicalModel_ISM(tf.keras.Model):
    def __init__(self, p, I, dim=1, card=50):
        super(CubicalModel_ISM, self).__init__()
        self.p = p
        self.I = I
        self.dim = dim
        self.card = card
        
    def call(self):

        Xp = tf.reshape(tf.tensordot(X,p,1),shape=[28,28])
        Yp = tf.reshape(tf.tensordot(Y,p,1),shape=[28,28])

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



















##################
#### Amplitudes
##################



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
