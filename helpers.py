from turtle import position
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
import plotly.express as px
import networkx as nx


############################
# PLOTS AND MULTIFILTRATIONS
############################

def plot_bifiltration(plot1,plot2):

    imgs_ = np.vstack((plot1,plot2)).reshape(8,28,28)
    fig = px.imshow(imgs_, facet_col=0, facet_col_wrap=4)
    fig.update_layout(height=400, width=800,title="Illustrating the filtrations")
    names = ["original image", "binarized filtration", "radial filtration", "height filtration"]
    for i, name in enumerate(names):
        fig.layout.annotations[i]['text'] = name
    fig.show()

def plot_optim_filtration(optimization, I, J):

    p_opt = optimization['projections'][-1]
    I_opt = tf.reshape(tf.tensordot(I,p_opt,1),shape=[28,28])
    J_opt = tf.reshape(tf.tensordot(J,p_opt,1),shape=[28,28])
    imgs_opt  = np.vstack((I_opt,J_opt)).reshape(2,28,28)
    fig = px.imshow(imgs_opt, facet_col=0, facet_col_wrap=2)
    fig.update_layout(height=290, width=600, title="Optimal LISM arguments (optimized bifiltrations)")
    fig.show()


def plot_optim(optimization, show_diagrams=False):

    amplitudes = optimization['amplitudes']
    n_epochs = len(amplitudes)

    cmap2 = n_colors('rgb(80, 208, 255)', 'rgb(160, 32, 255)', n_epochs+1, colortype = 'rgb')
    cmap1 = n_colors('rgb(80, 208, 255)', 'rgb(160, 32, 255)', n_epochs, colortype = 'rgb')

    if show_diagrams:
        nb_cols = 4
    else:
        nb_cols = 3

    fig = make_subplots(1,nb_cols)

    
    epochs = list(range(n_epochs))
    losses = optimization['losses']
    projections = optimization['projections']

    norms = [tf.norm(p) for p in projections]
    angles = np.arange(0, math.pi/2, 0.01)
    cosines = list(map(np.cos,angles))
    sines = list(map(np.sin,angles))

    fig.add_trace(go.Scatter(x=epochs,y=losses,mode="lines",name="loss",marker=dict(
            color='skyblue'
                )),row=1,col=1)
    fig.add_trace(go.Scatter(x=epochs,y=amplitudes,mode="lines",name="amplitude",marker=dict(
            color='rgb(160, 32, 255)'
                )),row=1,col=1)
    fig.add_trace(go.Scatter(x=cosines,y=sines,showlegend=False,marker=dict(
            color='powderblue'
                )),row=1,col=2)
    for i, p in enumerate(projections):
            a = float(p[0])/np.linalg.norm(p)
            b = float(p[1])/np.linalg.norm(p)
            fig.add_trace(go.Scatter(x=[a],y=[b],mode="markers",name="projection",showlegend=False,
            marker=dict(
            color=cmap2[i]
                )),row=1,col=2)
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

        fig.add_trace(go.Scatter(x=[b-e,d+e],y=[b-e,d+e],mode="lines",name="diagonal",showlegend=False, marker=dict(
            color='rgb(50, 129, 255)'
                )),row=1,col=4)

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


def wasserstein_matrix(images, train_y, dim, filtration, param):

    N = len(images)
    labels = train_y[:N]
    l = [{'image':images[i], 'label':labels[i]} for i in range(N)]
    l_sorted = sorted(l, key=lambda d: d['label']) 
    images_bb = np.array([l_sorted[i]['image'] for i in range(N)])
    labels_bb = np.array([l_sorted[i]['label'] for i in range(N)])

    D = np.zeros((N,N))

    for i, img1 in tqdm(enumerate(images_bb)):
        for j, img2 in enumerate(images_bb):
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

    fig = px.imshow(D,height=470, width=470)

    return D, fig, images_bb, labels_bb

    

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

    "Point-cloud optimization visual summary."

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



def is_center_of_3_path(G, node, layers):

    count1 = 0
    count2 = 0
    l = layers[node]

    if l>0 and l<4:

        for j in G.neighbors(node):

            if layers[j]==l-1:
                count1+=1
            if layers[j]==l+1:
                count2+=1
            if count1>0 and count2>0:
                return True

    elif l==0:

        for j in G.neighbors(node):

            if layers[j]==l+1:
                count2+=1
            if count2>0:
                return True

    elif l==4:

        for j in G.neighbors(node):

            if layers[j]==l-1:
                count1+=1
            if count1>0:
                return True


    return False


def connected_geometric_network(simplex_list, layers):
    
    E = [tuple(s) for s in simplex_list if len(s)==2]

    H = nx.Graph()
    H.add_edges_from(E)

    for _ in range(5):

        nodes = [node for node in H.nodes() if is_center_of_3_path(H, node, layers)]
        H = nx.Graph()
        H.add_nodes_from(nodes)
        H.add_edges_from([edge for edge in E if (edge[0] in nodes and edge[1] in nodes)])
        
    interesting_nodes = list(H.nodes())

    n0 = len([node for node in interesting_nodes if layers[node]==0])
    n1 = len([node for node in interesting_nodes if layers[node]==1])
    n2 = len([node for node in interesting_nodes if layers[node]==2])
    n3 = len([node for node in interesting_nodes if layers[node]==3])
    n4 = len([node for node in interesting_nodes if layers[node]==4])

    heights = np.hstack((np.linspace(-1,1,n0), np.linspace(-1,1,n1), np.linspace(-1,1,n2), np.linspace(-1,1,n3), np.linspace(-1,1,n4)))

    nodes = [{'node':node,'layer':layers[node]} for node in interesting_nodes]

    sorted_nodes = sorted(nodes, key=lambda d: d['layer']) 

    positional_nodes = [(node['node'],{'pos':[node['layer'],heights[sorted_nodes.index(node)]]}) for node in sorted_nodes]
    ouee = [node[0] for node in positional_nodes]

    edges = [edge for edge in E if (edge[0] in ouee and edge[1] in ouee)]

    G = nx.Graph()
    G.add_nodes_from(positional_nodes)
    G.add_edges_from(edges)

    return G


    
def plotly_geometric_network(G):
        
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Blues',
        reversescale=True,
        color=[],
        size=6,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=0.3))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                title='<br>Neural network structure summary',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text='',
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()

