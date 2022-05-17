import numpy as np
from sklearn.preprocessing import binarize
from tqdm import tqdm
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import math
from gtda.images import Binarizer
from gtda.images import RadialFiltration
from gtda.images import HeightFiltration
from sklearn.cluster                      import KMeans
import plotly.figure_factory as ff
from plotly.colors import n_colors


def plot_optim(optimization, show_diagrams=False):

    amplitudes = optimization['amplitudes']
    n_epochs = len(amplitudes)

    cmap1 = n_colors('rgb(160, 100, 255)', 'rgb(50, 140, 140)', n_epochs, colortype = 'rgb')
    cmap2 = n_colors('rgb(200, 240, 50)', 'rgb(160, 100, 255)', n_epochs+1, colortype = 'rgb')

    if show_diagrams:
        nb_cols = 4
    else:
        nb_cols = 3

    fig = make_subplots(1,nb_cols)

    
    epochs = list(range(n_epochs))
    losses = optimization['losses']
    projections = optimization['projections']
    proj1 = [float(p[0]) for p in projections]
    proj2 = [float(p[1]) for p in projections]

    norms = [tf.norm(p) for p in projections]
    angles = np.arange(0, math.pi/2, 0.01)
    cosines = list(map(np.cos,angles))
    sines = list(map(np.sin,angles))

    fig.add_trace(go.Scatter(x=epochs,y=losses,mode="lines",name="loss",marker=dict(
            color='skyblue'
                )),row=1,col=1)
    fig.add_trace(go.Scatter(x=epochs,y=amplitudes,mode="lines",name="amplitude",marker=dict(
            color='lightcoral'
                )),row=1,col=1)
    fig.add_trace(go.Scatter(x=cosines,y=sines,showlegend=False,marker=dict(
            color='powderblue'
                )),row=1,col=2)
    #fig.add_trace(go.Scatter(x=proj1,y=proj2, mode="markers",name="projection"),row=1,col=2)
    for i, p in enumerate(projections):
            fig.add_trace(go.Scatter(x=[float(p[0])],y=[float(p[1])],mode="markers",name="projection",showlegend=False,
            marker=dict(
            color=cmap2[i]
                )),row=1,col=2)
    #fig.add_trace(go.Scatter(x=epochs,y=proj1,mode="lines",name="p1"),row=1,col=2)
    #fig.add_trace(go.Scatter(x=epochs,y=proj2,mode="lines",name="p2"),row=1,col=2)
    fig.add_trace(go.Scatter(x=epochs,y=norms,mode="lines",name="norm of p",marker=dict(
            color='plum'
                )),row=1,col=3)

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="p1", row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=3)
    fig.update_yaxes(title_text="amplitude", row=1, col=1)

    if show_diagrams:

        dgms = optimization['diagrams']
        b = min([min(tuple[0][:,0]) for tuple in dgms])
        d = max([max(tuple[0][:,1]) for tuple in dgms])
        e = (d-b)/10

        fig.add_trace(go.Scatter(x=[b-e,d+e],y=[b-e,d+e],mode="lines",name="diagonal",showlegend=False),row=1,col=4)
        fig.update_xaxes(title_text="birth", row=1, col=4)
        for i, tuple in enumerate(dgms):
            fig.add_trace(go.Scatter(x=tuple[0][:,0],y=tuple[0][:,1],mode="markers",name="(b,d)",showlegend=False,
            marker=dict(
            color=cmap1[i]
                )),row=1,col=4)

    fig.update_layout(
        title="Optimization summary for integral sheaf metric computation",
        height=400,
        width=1000,
        legend_title="Curves"
    )
    fig.show()



def bifiltration(img, center=[13,13], direction=[0,1]):

    binarizer = Binarizer(threshold=0.4)
    radial_filtration = RadialFiltration(center=np.array(center))
    height_filtration = HeightFiltration(direction=np.array(direction))

    I_bin = binarizer.fit_transform(img)
    I_rad = radial_filtration.fit_transform(I_bin.reshape(1,28,28)).reshape(28,28)
    I_height = height_filtration.fit_transform(I_bin.reshape(1,28,28)).reshape(28,28)
    I_rad = I_rad/np.amax(I_rad)
    I_height = I_height/np.amax(I_height)
    
    I_ = np.array([M.reshape(2,28).T for M in np.split(np.stack((I_rad,I_height),axis=1),28)])
    I = tf.Variable(initial_value=np.array(I_, dtype=np.float32), trainable=False)

    plotting_tuple = (img, I_bin, I_rad, I_height)

    return I, plotting_tuple




def radial_filtration(I, center=[6,13]):

    binarizer = Binarizer(threshold=0.4)
    radial_filtration = RadialFiltration(center=np.array(center))
    I_bin = binarizer.fit_transform(I)
    I_rad = radial_filtration.fit_transform(I_bin.reshape(1,28,28)).reshape(28,28)

    return I_rad


def height_filtration(I, v = [1,0]):

    binarizer = Binarizer(threshold=0.4)
    height_filtration = HeightFiltration(direction=np.array(v))
    I_bin = binarizer.fit_transform(I)
    I_height = height_filtration.fit_transform(I_bin.reshape(1,28,28)).reshape(28,28)

    return I_height


def wasserstein_matrix(images, dim, filtration, param):

    N = len(images)

    D = np.zeros((N,N))

    for i, img1 in tqdm(enumerate(images)):
        for j, img2 in enumerate(images):
            if i<j:
                if filtration=="height":
                    I = height_filtration(img1, param)
                    J = height_filtration(img2, param)
                elif filtration=="radial":
                    I = radial_filtration(img1, param)
                    J = radial_filtration(img2, param)

                cc = gd.CubicalComplex(dimensions=I.shape, top_dimensional_cells=I.flatten())
                pers = cc.persistence()
                cc_ = gd.CubicalComplex(dimensions=J.shape, top_dimensional_cells=J.flatten())
                pers_ = cc_.persistence()

                pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==dim and tuple[1][1]!=np.inf]
                pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==dim and tuple[1][1]!=np.inf]
        
                dgm1 = np.array(pers1).reshape(len(pers1),2)
                dgm2 = np.array(pers2).reshape(len(pers2),2)
                
                D[i,j] = np.sqrt(wasserstein_distance(dgm1, dgm2, order=2))

    D += np.transpose(D)

    return D

    

def wass_amplitudes(images, alpha, dim=1):

    A = []

    for I in tqdm(images):

        cc = gd.CubicalComplex(dimensions=I.shape, top_dimensional_cells=I.flatten())
        dgm = np.array([p[1] for p in cc.persistence() if p[0]==dim])

        A.append(alpha * np.square(wasserstein_distance(dgm, [], order=2)))

    return np.array(A)

def get_accuracy(D, N, train_y):

    kmeans = KMeans(n_clusters=2, random_state=0).fit(D)
    pred = kmeans.labels_
    truth = train_y[:N]
    res = pred - truth

    nb = np.count_nonzero(res==0) + np.count_nonzero(res==-7)
    nb_correct = max(nb, N-nb)

    return nb_correct/N


def dist_matrix_hist(D, N, train_y, method):

    dists_same = []
    dists_diff = []

    methods = ['LISM', 'Wasserstein w/ radial filtration', 'Wasserstein w/ height filtration']

    for i in range(N):
        for j in range(N):
            if i<j:

                if train_y[:N][i]==train_y[:N][j]:
                    dists_same.append(D[i,j])
                else:
                    dists_diff.append(D[i,j])

    fig = make_subplots(1,2)

    # Group data together
    hist_data = [dists_same, dists_diff]

    group_labels = ['same digit', 'diff digits']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[.01, .01])
    fig.update_layout(height=400, width=800, title_text= methods[method] +" distance" + " matrix distribution")
 
    fig.show()


def plot_optim_pc(optimization,X,Y):

    m = max(max(Y[:, 0]),max(Y[:, 1]))
    n = X.shape[0]

    fig = make_subplots(2,2)
    fig.add_trace(go.Scatter(x = X[:, 0], y = X[:, 1], mode='markers', name = 'circle', marker=dict(size=2,color='mediumorchid')),row=1,col=1)
    fig.add_trace(go.Scatter(x = Y[:, 0], y = Y[:, 1], mode='markers', name = 'ellipse', marker=dict(size=2, color='royalblue')),row=1,col=1)
    fig.update_layout(
            title="Optimization of point-cloud projection",
            height=700,
            width=800,
            legend_title="Point-clouds",
            xaxis_range=[-m,m],
            yaxis_range=[-m,m]
        )
    projections = optimization['projections']
    n_epochs = len(projections)

    cmap2 = n_colors('rgb(160, 100, 255)', 'rgb(50, 150, 200)', n_epochs, colortype = 'rgb')

    for i, p in enumerate(projections):

        p = np.array(p).reshape(2,1)
        Xp = np.tensordot(X,p,1).reshape(n)
        Yp = np.tensordot(Y,p,1).reshape(n)

        if i == 0:
            show = True
        else:
            show = False
        fig.add_trace(go.Scatter(x=[0,m],y=[0,float(p[1])/float(p[0])*m], mode="lines",name="projection",showlegend=show,
        marker=dict(
        color=cmap2[i]
            )),row=1,col=1)
        fig.add_trace(go.Scatter(x=Xp,y=Yp, mode="markers",name="Yp=f(Xp)",showlegend=show,
        marker=dict(
        color=cmap2[i],
        size=4
            )),row=1,col=2)
        """fig.add_trace(go.Scatter(x=[i for j in range(n_epochs)],y=Xp, mode="markers",name="Xp=f(t)",showlegend=show,
        marker=dict(
        color=cmap2[i],
        size=4
            )),row=1,col=3)
        fig.add_trace(go.Scatter(x=[i for j in range(n_epochs)],y=Yp, mode="markers",name="Yp=f(t)",showlegend=show,
        marker=dict(
        color=cmap2[i],
        size=4
            )),row=1,col=4)"""
        fig.add_trace(go.Scatter(x=X[:,0],y=Xp, mode="markers",name="Xp=f(x)",showlegend=show,
        marker=dict(
        color=cmap2[i],
        size=4
            )),row=2,col=1)
        fig.add_trace(go.Scatter(x=Y[:,0],y=Yp, mode="markers",name="Yp=f(x)",showlegend=show,
        marker=dict(
        color=cmap2[i],
        size=4
            )),row=2,col=2)
    fig.update_layout(xaxis2 = dict(range=[-m-1, m+1]))
    fig.update_layout(xaxis3 = dict(range=[-2, 2]))
    fig.update_layout(xaxis4 = dict(range=[-m-1, m+1]))
    fig.update_layout(yaxis2 = dict(range=[-m-1, m+1]))
    fig.update_layout(yaxis3 = dict(range=[-2, 2]))
    fig.update_layout(yaxis4 = dict(range=[-m-1, m+1]))
    fig.show()

