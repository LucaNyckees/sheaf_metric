#from turtle import forward
from difftda import Cubical
import tensorflow as tf
import numpy as np
import gudhi as gd
import torch
from torch.autograd import Function
import math
from gudhi.wasserstein                    import wasserstein_distance
import plotly.graph_objects as go
from tqdm import tqdm



class CubicalModel_ISM(tf.keras.Model):
    def __init__(self, p, I, J, dim=1, card=50):
        super(CubicalModel_ISM, self).__init__()
        self.p = p
        self.I = I
        self.J = J
        self.dim = dim
        self.card = card
        
    def call(self):

        Xp = tf.reshape(tf.tensordot(self.I,self.p,1),shape=[28,28])
        Yp = tf.reshape(tf.tensordot(self.J,self.p,1),shape=[28,28])

        d, c, D = self.dim, self.card, len(Xp.shape)
        XX = tf.reshape(Xp, [1, Xp.shape[0], Xp.shape[1]])
        YY = tf.reshape(Yp, [1, Yp.shape[0], Yp.shape[1]])
        
        # Turn numpy function into tensorflow function
        CbTF = lambda X: tf.numpy_function(Cubical, [X, d, c], [tf.int32 for _ in range(2*D*c)])
        
        # Compute pixels associated to positive and negative simplices 
        # Don't compute gradient for this operation
        
        inds1 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,XX,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))
        inds2 = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,YY,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))

        
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = tf.reshape(tf.gather_nd(Xp, tf.reshape(inds1, [-1,D])), [-1,2])
        dgm2 = tf.reshape(tf.gather_nd(Yp, tf.reshape(inds2, [-1,D])), [-1,2])
        return dgm1, dgm2

def LISM_optimization(model, fast=False, use_reg=True, alpha=1, lambda_=1, sigma=0.001):

    lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.5, decay_steps=10, decay_rate=.01)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    optimization = {'losses':[],
                                'amplitudes':[],
                                'projections':[model.p.numpy()],
                                'diagrams':[]
    }

    for epoch in tqdm(range(50+1)):
        
        with tf.GradientTape() as tape:
            
            dgm1, dgm2 = model.call()
            
            if use_reg:
                amplitude = alpha * tf.sqrt(wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True))
                loss = - amplitude + lambda_*(tf.norm(model.p)-1)**2
            else:
                amplitude = alpha * tf.sqrt(wasserstein_distance(dgm1, dgm2, order=2, enable_autodiff=True))
                loss = - amplitude 
            
        gradients = tape.gradient(loss, model.trainable_variables)
        
        np.random.seed(epoch)
        # gradients = [tf.convert_to_tensor(gradients[0])]
        gradients[0] = gradients[0] + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if not fast:
        
            optimization['losses'].append(loss.numpy())
            optimization['amplitudes'].append(amplitude.numpy())
            optimization['projections'].append(model.p.numpy())
            optimization['diagrams'].append((dgm1,dgm2))

            if epoch > 5:
                diff = tf.abs(tf.norm(optimization['projections'][epoch])-tf.norm(optimization['projections'][epoch-1]))
                if diff < 0.0005:
                    break
    if fast:
        return amplitude.numpy()
    else:
        return optimization


def grid_search(I, J, plotting=False, dim=1, card=50):
    
    angles = np.arange(0, math.pi/2, 0.01)
    linear_forms = [np.array([math.cos(theta), math.sin(theta)]).reshape(2,1) for theta in angles ]

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
        
        distances.append(tf.sqrt(wasserstein_distance(dgm1, dgm2, order=2)))

    dist = max(distances)
    index = distances.index(dist)
    angle = angles[index]

    if plotting:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=angles,y=distances))
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



def grid_search_amp(I, dim=1, card=50):
    
    angles = np.arange(0, math.pi/2, 0.01)
    linear_forms = [np.array([math.cos(theta), math.sin(theta)]).reshape(2,1) for theta in angles ]

    amplitudes = []

    # Turn numpy function into tensorflow function
    CbTF = lambda X: tf.numpy_function(Cubical, [X, d, c], [tf.int32 for _ in range(2*D*c)])

    for p_ in linear_forms:

        p = tf.Variable(initial_value=p_, dtype=np.float32)

        Ip = tf.reshape(tf.tensordot(I,p,1),shape=[28,28])

        d, c, D = dim, card, len(Ip.shape)
        XX = tf.reshape(Ip, [1, Ip.shape[0], Ip.shape[1]])
        
        inds = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(CbTF,XX,fn_output_signature=[tf.int32 for _ in range(2*D*c)]))
        dgm = tf.reshape(tf.gather_nd(Ip, tf.reshape(inds, [-1,D])), [-1,2])
        
        amplitudes.append(tf.sqrt(wasserstein_distance(dgm, [], order=2)))

    amp = max(amplitudes)
    
    return amp


        

class CubicalModel_ISM_torch(Function):

    @staticmethod    
    def forward(p, I, J, dim=1, card=256):

        # first image I = (I1,I2)
        # second image J = (J1,J2)

        Xp = torch.tensordot(I,p,1).reshape([28,28])
        Yp = torch.tensor(J,p,1).reshape([28,28])

        d, c, D = dim, card, len(Xp.shape)
        XX = torch.tensor(Xp).reshape([1, Xp.shape[0], Xp.shape[1]])
        YY = torch.tensor(Yp).reshape([1, Yp.shape[0], Yp.shape[1]])

        
        def CbTF(X,d1,c1):
           X_numpy = X.detach().numpy()
           return torch.tensor(Cubical(X_numpy,d1,c1), dtype = torch.int8).reshape(2*D*c)
     
       # Compute pixels associated to positive and negative simplices
       # Don't compute gradient for this operation
      
        inds1 = CbTF(XX, d, c)
        inds2 = CbTF(YY, d, c)
    
        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = torch.tensor(tf.gather_nd(Xp, torch.tensor(inds1).reshape([-1,D]))).reshape([-1,2])
        dgm2 = torch.tensor(tf.gather_nd(Yp, torch.tensor(inds2).reshape([-1,D]))).reshape([-1,2])
        return dgm1, dgm2

    
    @staticmethod    
    def backward(p, I, J, dim=1, card=256):

        dgm1, dgm2 = forward(p, I, J, dim, card)

        return dgm1.grad, dgm2.grad





def fast_grid_search(I, J, dim=1, card=50):
    
    angles = np.arange(0, math.pi/2, 0.01)
    linear_forms = [np.array([math.cos(theta), math.sin(theta)]).reshape(2,1) for theta in angles ]

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
        
        distances.append(np.sqrt(wasserstein_distance(dgm1, dgm2, order=2)))

    dist = max(distances)
    
    return dist
