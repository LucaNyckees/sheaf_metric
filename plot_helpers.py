import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import math
from plotly.colors import n_colors
import plotly.express as px
import n_sphere
from sklearn.manifold import MDS


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


def plot_optim(optimization):

    amplitudes = optimization['amplitudes']
    n_epochs = len(amplitudes)-1
    projections = optimization['projections']
    n_features = projections[-1].shape[0]

    p_opt = projections.pop()
    amp_opt = amplitudes.pop()

    cmap2 = n_colors('rgb(80, 208, 255)', 'rgb(160, 32, 255)', n_epochs+1, colortype = 'rgb')
    cmap1 = n_colors('rgb(80, 208, 255)', 'rgb(160, 32, 255)', n_epochs, colortype = 'rgb')

    nb_cols = 4

    if n_features == 3:
        
        fig = make_subplots(rows=1,cols=nb_cols, subplot_titles=("loss and amplitude", "projection", "dgm1", "dgm2"),
                        specs=[[{"type": 'xy'}, {"type": 'scene'}, {"type": "xy"}, {"type": 'xy'}]])

    else:

        fig = make_subplots(rows=1,cols=nb_cols,subplot_titles=("projected distance", "projection", "dgm1", "dgm2"))

    
    epochs = list(range(n_epochs))
    losses = optimization['losses']

    norms = [tf.norm(p) for p in projections]
    angles = np.arange(0, math.pi/2, 0.01)
    cosines = list(map(np.cos,angles))
    sines = list(map(np.sin,angles))

    """fig.add_trace(go.Scatter(x=epochs,y=losses,mode="lines",name="loss",marker=dict(
            color='skyblue'
                )),row=1,col=1)"""
    fig.add_trace(go.Scatter(x=epochs,y=amplitudes,mode="lines",name="amplitude",marker=dict(
            color='rgb(160, 32, 255)'
                )),row=1,col=1)
    if n_features == 2:

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
    elif n_features == 3:

        phis = np.arange(0, math.pi/2, 0.1)
        thetas = np.arange(0, math.pi/2, 0.1)

        for phi in phis:
            for theta in thetas:

                x = np.array([1,phi,theta])
                v = n_sphere.convert_rectangular(x)
                fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode="markers", showlegend=False, marker=dict(
                color='blue',
                size=1
                    )),row=1,col=2)
        
        for i, p in enumerate(projections):
                a = float(p[0])/np.linalg.norm(p)
                b = float(p[1])/np.linalg.norm(p)
                c = float(p[2])/np.linalg.norm(p)
                fig.add_trace(go.Scatter3d(x=[a], y=[b], z=[c],
                                   mode='markers',name="projection",showlegend=False,
                marker=dict(
                color=cmap2[i],
                size=3
                    )),row=1,col=2)
    
    """fig.add_trace(go.Scatter(x=epochs,y=norms,mode="lines",name="norm of p",marker=dict(
            color='plum'
                )),row=1,col=3)"""

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="p1", row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=3)
    fig.update_yaxes(title_text="amplitude", row=1, col=1)

    dgms = optimization['diagrams']
    dgms_opt = dgms.pop()
    b = min([min(tuple[0][:,0]) for tuple in dgms])
    d = max([max(tuple[0][:,1]) for tuple in dgms])
    b_ = min([min(tuple[1][:,0]) for tuple in dgms])
    d_ = max([max(tuple[1][:,1]) for tuple in dgms])
    e = (d-b)/10
    e_ = (d_-b_)/10

    fig.add_trace(go.Scatter(x=[b-e,d+e],y=[b-e,d+e],mode="lines",name="diagonal",showlegend=False, marker=dict(
        color='rgb(50, 129, 255)'
            )),row=1,col=3)

    fig.update_xaxes(title_text="birth", row=1, col=3)
    for i, tuple in enumerate(dgms):
        fig.add_trace(go.Scatter(x=tuple[0][:,0],y=tuple[0][:,1],mode="markers",name="(b,d)",showlegend=False,
        marker=dict(
        color=cmap1[i]
            )),row=1,col=3)

    fig.add_trace(go.Scatter(x=[b_-e_,d_+e_],y=[b_-e_,d_+e_],mode="lines",name="diagonal",showlegend=False, marker=dict(
        color='rgb(50, 129, 255)'
            )),row=1,col=4)

    fig.update_xaxes(title_text="birth", row=1, col=4)
    for i, tuple in enumerate(dgms):
        fig.add_trace(go.Scatter(x=tuple[1][:,0],y=tuple[1][:,1],mode="markers",name="(b,d)",showlegend=False,
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


def MDS_analysis(matrices, labels_bb):

    n = len(matrices)
    m = len(set(labels_bb))

    fig = make_subplots(rows=n,cols=1)

    colors = ['mediumslateblue', 'mediumturquoise', 'yellowgreen', 'hotpink']
    c = colors[:m]
    legends = [True] + [False for _ in range(n-1)]

    for j, D in enumerate(matrices):
        
        embedding = MDS(n_components=2, dissimilarity='precomputed')

        MDS_ = embedding.fit_transform(D)

        for i, digit in enumerate(set(labels_bb)):

            mds = MDS_[labels_bb==digit]

            fig.add_trace(go.Scatter(x=mds[:,0], y=mds[:,1], 
                                     mode='markers', 
                                     name=str(digit), 
                                     showlegend=legends[j],
                                     marker=dict(
                                         color=c[i]
                                    )), row=j+1, col=1)
            

    fig.update_layout(
            title="Multi-dimensional scaling (MDS) of distance matrices",
            legend_title="Digits",
            height=900
    )

    fig.show()

def heatmaps(matrices_tup):

    m = matrices_tup[0].shape[0]
    n = matrices_tup[0].shape[1]


    imgs_ = np.vstack(matrices_tup).reshape(len(matrices_tup),m,n)
    fig = px.imshow(imgs_, facet_col=0, facet_col_wrap=len(matrices_tup), color_continuous_scale=[[0.0, "rgb(165,0,38)"],
                    [0.1111111111111111, "rgb(215,48,39)"],
                    [0.2222222222222222, "rgb(244,109,67)"],
                    [0.3333333333333333, "rgb(253,174,97)"],
                    [0.4444444444444444, "rgb(254,224,144)"],
                    [0.5555555555555556, "rgb(224,243,248)"],
                    [0.6666666666666666, "rgb(171,217,233)"],
                    [0.7777777777777778, "rgb(116,173,209)"],
                    [0.8888888888888888, "rgb(69,117,180)"],
                    [1.0, "rgb(49,54,149)"]])
    fig.update_layout(height=400, width=800,title="Heat maps of distance matrices")
    fig.show()
