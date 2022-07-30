from difftda import SimplexTreeModel_ISM, CubicalModel_ISM
import tensorflow as tf
import numpy as np
import gudhi as gd
from torch.autograd import Function
import math
from gudhi.wasserstein                    import wasserstein_distance
import plotly.graph_objects as go
from tqdm import tqdm
from difftda import Cubical
import n_sphere
import copy

# In this file, we implement the functions that make the topological optimization necessary to compute
# a linear integral sheaf metric (LISM), and present a class pipeline() that allows to compute a distance
# matrix encoding the LISMs between a given data set of observations (e.g. images or weighted graphs).


class pipeline():

    # class made for easily computing a LISM distance matrix between observations like images or weighted graphs.

    def __init__(self, object, meta_data, card, dims=[0]):

        """
        Args:

            object (int): 0 for simplicial complex, 1 for cubical complex

            meta_data : 
                (1) if object==0, a tuple (data, network) where 
                network is path to a .txt file encoding a simplicial 
                complex and data consists of N multifiltrations (data
                is a np.array of shape [N, n_nodes, n_features]).

                (2) if object==1, meta_data=[data], a np.array of shape
                [N, n_pixels, p_pixels, n_features] encoding N images. 

            card (int): cardinality of persistence diagram (e.g. nb_nodes)

            dims (list of int): a list of homological dimensions
        """

        super(pipeline, self).__init__()
        self.object = object
        self.meta_data = meta_data
        self.card = card
        self.dims = dims
       
        
    def simp_persistence(self, p, F, G, dim):

        network = self.meta_data[1]

        model = SimplexTreeModel_ISM(p, F, G, stbase = network, dim=dim, card=self.card)

        return model


    def cub_persistence(self, p, I, J, dim):

        model = CubicalModel_ISM(p, I, J, dim=dim, card=self.card)

        return model

    def optim(self, model, fast=False, use_reg=True, alpha=1, lambda_=1, sigma=0.001):

        lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.5, decay_steps=10, decay_rate=.01)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        optimization = {'losses':[],
                                    'amplitudes':[],
                                    'projections':[model.p.numpy()],
                                    'diagrams':[]
        }

        for epoch in range(50+1):


            if epoch > 5:
                diff = tf.abs(tf.norm(optimization['projections'][epoch],ord=1)-tf.norm(optimization['projections'][epoch-1],ord=1))
                    
                if diff < 0.00005 or epoch==50: #or any(tf.math.sign(model.p)<0):

                    print(epoch)
                        
                    p_opt = model.p/tf.norm(model.p,ord=1)

                    model.p = p_opt

                    ################
                    p_copy = copy.copy(p_opt).numpy()
            
                    for j in range(p_copy.shape[0]):

                        if np.sign(p_copy[j])<0:

                            p_copy[j]=0

                    I_copy = model.I.numpy()
                    J_copy = model.J.numpy()

                    Ip = np.tensordot(I_copy,p_copy,1)
                    Jp = np.tensordot(J_copy,p_copy,1)

                    cc = gd.CubicalComplex(dimensions=Ip.shape, top_dimensional_cells=Ip.flatten())
                    pers = cc.persistence()
                    cc_ = gd.CubicalComplex(dimensions=Jp.shape, top_dimensional_cells=Jp.flatten())
                    pers_ = cc_.persistence()

                    pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==model.dim and tuple[1][1]!=np.inf]
                    pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==model.dim and tuple[1][1]!=np.inf]

                    dgm1 = np.array(pers1).reshape(len(pers1),2)
                    dgm2 = np.array(pers2).reshape(len(pers2),2)

                    ################

                    #dgm1, dgm2 = model.call()

                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude + lambda_*(tf.norm(model.p,ord=1)-1)**2

                    #optimization['projections'].append(model.p.numpy())
                    optimization['projections'].append(p_copy)
                    optimization['losses'].append(loss.numpy())
                    optimization['amplitudes'].append(amplitude)
                    optimization['diagrams'].append((dgm1,dgm2))
                    
                    break

            with tf.GradientTape() as tape:
                
                dgm1, dgm2 = model.call()

                #s = 0

                #for j in range(model.p.shape[0]):

                    #s += 1000*tf.math.sign(model.p[j])
                
                if use_reg:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude + lambda_*(tf.norm(model.p,ord=1)-1)**2 #- s
                else:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude 


                
            gradients = tape.gradient(loss, model.trainable_variables)
            
            np.random.seed(epoch)
            # gradients = [tf.convert_to_tensor(gradients[0])]
            gradients[0] = gradients[0] + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            optimization['projections'].append(model.p.numpy())
            optimization['losses'].append(loss.numpy())
            optimization['amplitudes'].append(amplitude.numpy())
            optimization['diagrams'].append((dgm1,dgm2))

        if fast:
            return amplitude.numpy()
        else:
            return optimization

    def fast_optim(self, model, use_reg=True, alpha=1, lambda_=1, sigma=0.001):

        # optimization with SGD that stores a minimal amount of information.

        lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.7, decay_steps=10, decay_rate=.1)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        projections = [] 

        for epoch in range(50+1):

            if epoch > 5:
                diff = tf.abs(tf.norm(projections[epoch],ord=1)-tf.norm(projections[epoch-1],ord=1))
                    
                if diff < 0.005 or any(tf.math.sign(model.p)<0):
                        
                    p_opt = model.p/tf.norm(model.p,ord=1)

                    model.p = p_opt

                    ################
                    p_copy = copy.copy(p_opt).numpy()
            
                    for j in range(p_copy.shape[0]):

                        if np.sign(p_copy[j])<0:

                            p_copy[j]=0

                    I_copy = model.I.numpy()
                    J_copy = model.J.numpy()

                    Ip = np.tensordot(I_copy,p_copy,1)
                    Jp = np.tensordot(J_copy,p_copy,1)

                    cc = gd.CubicalComplex(dimensions=Ip.shape, top_dimensional_cells=Ip.flatten())
                    pers = cc.persistence()
                    cc_ = gd.CubicalComplex(dimensions=Jp.shape, top_dimensional_cells=Jp.flatten())
                    pers_ = cc_.persistence()

                    pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==model.dim and tuple[1][1]!=np.inf]
                    pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==model.dim and tuple[1][1]!=np.inf]

                    dgm1 = np.array(pers1).reshape(len(pers1),2)
                    dgm2 = np.array(pers2).reshape(len(pers2),2)

                    ################

                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)

                    return amplitude

            with tf.GradientTape() as tape:
                
                dgm1, dgm2 = model.call()
                
                if use_reg:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude + lambda_*(tf.norm(model.p,ord=1)-1)**2
                else:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude 
                
            gradients = tape.gradient(loss, model.trainable_variables)
            
            np.random.seed(epoch)
            # gradients = [tf.convert_to_tensor(gradients[0])]
            gradients[0] = gradients[0] + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            projections.append(model.p.numpy())

        return amplitude



    def distance_matrix(self, dim, fast=True):

        # computes a distance matrix of size NxN for a given set of N observations

        N = self.meta_data[0].shape[0]
        data = self.meta_data[0]
        n_features = data.shape[-1]

        if self.object==0:

            method = self.simp_persistence

        elif self.object==1:

            method = self.cub_persistence

        D = np.zeros((N,N))

        multifilt1 = data[0]
        multifilt2 = data[0]
        m1 = tf.Variable(initial_value=multifilt1, trainable=False, dtype=tf.float32)
        m2 = tf.Variable(initial_value=multifilt2, trainable=False, dtype=tf.float32)

        # random initial projection 
        x = np.hstack((np.array([1]),np.random.rand(n_features-2)*np.pi,np.random.rand(1)*np.pi*2))
        p_ = np.abs(n_sphere.convert_rectangular(x).reshape(n_features,1))
        p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)

        model = method(p, m1, m2, dim)

        for i in tqdm(range(N)):
            for j in range(N):
                if i<j:
                    
                    multifilt1 = data[i]
                    multifilt2 = data[j]
                    # random initial projection 
                    x = np.hstack((np.array([1]),np.random.rand(n_features-2)*np.pi,np.random.rand(1)*np.pi*2))
                    p_ = np.abs(n_sphere.convert_rectangular(x).reshape(n_features,1))
                    p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)
                    # converting multifiltrations to tensorflow non-trainable variables
                    m1 = tf.Variable(initial_value=multifilt1, trainable=False, dtype=tf.float32)
                    m2 = tf.Variable(initial_value=multifilt2, trainable=False, dtype=tf.float32)
                    
                    model.I = m1
                    model.J = m2
                    model.p = p

                    if not fast:
                        dist = self.optim(model, fast=True)
                    elif fast:
                        dist = self.fast_optim(model)
                    D[i,j] = dist

        D += np.transpose(D)

        return D

        

    def single_distance(self, dim=0, p_init = 0):

        # computes a single LISM between two observations.

        data = self.meta_data[0]
        N = data.shape[0]
        n_features = data.shape[-1]

        if N != 2:
            print("Please enter data consisting of only two multi-filtrations. None is returned.")
            return None

        if self.object==0:

            method = self.simp_persistence

        elif self.object==1:

            method = self.cub_persistence

        if p_init==0:
            # random intial projection 
            x = np.hstack((np.array([1]),np.random.rand(n_features-2)*np.pi,np.random.rand(1)*np.pi*2))
            p_ = np.abs(n_sphere.convert_rectangular(x).reshape(n_features,1))
            p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)

        else:

            p = tf.Variable(initial_value=np.array(p_init).reshape(n_features,1), trainable=True, dtype = tf.float32)
            #tf.keras.constraints.non_neg.__call__(self, p)
        
        # converting multifiltrations to tensorflow non-trainable variables
        m1 = tf.Variable(initial_value=data[0], trainable=False, dtype=tf.float32)
        m2 = tf.Variable(initial_value=data[1], trainable=False, dtype=tf.float32)

        # initializing the input-model
        model = method(p, m1, m2, dim)

        optimization = self.optim(model, fast=False)

        return optimization



def grid_search(I, J, step=0.01, plotting=False, dim=1, card=50):

    # computes a single LISM with a grid search (expensive in high dimension,
    # but is used to proofcheck the results obtained with topological optimization).
    
    params = np.arange(0, 1, step)
    linear_forms = [np.array([t,1-t]).reshape(2,1) for t in params ]

    distances = []

    # Turn numpy function into tensorflow function
    CbTF = lambda X: tf.numpy_function(Cubical, [X, d, c], [tf.int32 for _ in range(2*D*c)])

    for p_ in linear_forms:

        p = tf.Variable(initial_value=p_, dtype=np.float32)


        Ip = tf.reshape(tf.tensordot(I,p,1),shape=[28,28])
        Jp = tf.reshape(tf.tensordot(J,p,1),shape=[28,28])

        d, c, D = dim, card, len(Ip.shape)
        XX = tf.reshape(Ip, [1, Ip.shape[0], Ip.shape[1]])
        YY = tf.reshape(Jp, [1, Jp.shape[0], Jp.shape[1]])
        
        # Compute pixels associated to positive and negative simplices 
        # Don't compute gradient for this operation
        
        inds1 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,XX,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))
        inds2 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,YY,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))

        
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = tf.reshape(tf.gather_nd(Ip, tf.reshape(inds1, [-1,D])), [-1,2])
        dgm2 = tf.reshape(tf.gather_nd(Jp, tf.reshape(inds2, [-1,D])), [-1,2])
        
        distances.append(wasserstein_distance(dgm1, dgm2, order=2))

    dist = max(distances)
    index = distances.index(dist)
    angle = params[index]

    if plotting:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=params,y=distances))
        fig.add_trace(go.Scatter(x=[angle,angle],y=[0,dist],line = dict(width=1, dash='dot'),showlegend=False))
        fig.add_trace(go.Scatter(x=[0,angle],y=[dist,dist],line = dict(width=1, dash='dot'),showlegend=False))
        fig.add_trace(go.Scatter(x=[angle],y=[dist],mode='markers',name="maximum"))
        fig.update_layout(
            title="Computing the LISM with a grid search",
            height=400,
            width=600,
            xaxis_title="angle of projection",
            yaxis_title="sqrt of Wasserstein distance",
            legend_title="Curves"
        )

        return dist, angle, fig
    
    return dist, angle



def fast_grid_search(I, J, step = 0.01, dim=1):

    if I.shape[-1]!=2:
        print("The fast grid search method should only be applied for the case of bi-filtrations (n_features=2). Here, n_features={}.".format(I.shape[-1]))
        return None
    
    params = np.arange(0, 1, step)
    linear_forms = [np.array([t,1-t]).reshape(2,1) for t in params ]

    distances = []

    for p_ in linear_forms:

        Ip = np.tensordot(I,p_,1).reshape(28,28)
        Jp = np.tensordot(J,p_,1).reshape(28,28)

        cc = gd.CubicalComplex(dimensions=Ip.shape, top_dimensional_cells=Ip.flatten())
        pers = cc.persistence()
        cc_ = gd.CubicalComplex(dimensions=Jp.shape, top_dimensional_cells=Jp.flatten())
        pers_ = cc_.persistence()

        pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==dim and tuple[1][1]!=np.inf]
        pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==dim and tuple[1][1]!=np.inf]
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = np.array(pers1).reshape(len(pers1),2)
        dgm2 = np.array(pers2).reshape(len(pers2),2)
        
        distances.append(wasserstein_distance(dgm1, dgm2, order=2))

    dist = max(distances)
    
    return dist









def fast_optim(model, use_reg=True, alpha=1, lambda_=1, sigma=0.001):

        lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.7, decay_steps=10, decay_rate=.1)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        # init_lr = 0.5
        # decay_steps = 10
        # decay_rate = .01
        projections = [] 

        for epoch in range(50+1):

            if epoch > 5:
                diff = tf.abs(tf.norm(projections[epoch-1],ord=1)-tf.norm(projections[epoch-2],ord=1))
                    
                if diff < 0.005 or any(tf.math.sign(model.p)<0):
                        
                    p_opt = model.p/tf.norm(model.p,ord=1)

                    model.p = p_opt

                    ################
                    p_copy = copy.copy(p_opt).numpy()
            
                    for j in range(p_copy.shape[0]):

                        if np.sign(p_copy[j])<0:

                            p_copy[j]=0

                    I_copy = copy.copy(model.I)
                    J_copy = copy.copy(model.J)

                    Ip = np.tensordot(I_copy,p_copy,1)
                    Jp = np.tensordot(J_copy,p_copy,1)

                    cc = gd.CubicalComplex(dimensions=Ip.shape, top_dimensional_cells=Ip.flatten())
                    pers = cc.persistence()
                    cc_ = gd.CubicalComplex(dimensions=Jp.shape, top_dimensional_cells=Jp.flatten())
                    pers_ = cc_.persistence()

                    pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==model.dim and tuple[1][1]!=np.inf]
                    pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==model.dim and tuple[1][1]!=np.inf]

                    dgm1 = np.array(pers1).reshape(len(pers1),2)
                    dgm2 = np.array(pers2).reshape(len(pers2),2)

                    ################

                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                                    
                    return amplitude

            with tf.GradientTape() as tape:
                
                dgm1, dgm2 = model.call()
                
                if use_reg:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude + lambda_*(tf.norm(model.p,ord=1)-1)**2
                else:
                    amplitude = alpha * wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True)
                    loss = - amplitude 
                
            gradients = tape.gradient(loss, model.trainable_variables)
            
            np.random.seed(epoch)
            # gradients = [tf.convert_to_tensor(gradients[0])]
            gradients[0] = gradients[0] + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            projections.append(model.p.numpy())

        return amplitude