import numpy as np
from tqdm import tqdm
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import math
from gtda.images import Binarizer
from sklearn.cluster                      import KMeans
import plotly.express as px
import networkx as nx
from LISM import fast_grid_search, pipeline, fast_optim
from difftda import CubicalModel_ISM, SimplexTreeModel_ISM_K1K2
import tensorflow as tf
import n_sphere
import copy
from itertools import permutations


####################
# MULTIFILTRATIONS
####################


def multifiltration(img, filtrations):

    n_features = len(filtrations)
    n_pixels = img.shape[0]

    binarizer = Binarizer(threshold=0.4)

    I_bin = binarizer.fit_transform(img)
    filters = []
    
    for filt in filtrations:

        filter = filt.fit_transform(I_bin.reshape(1,n_pixels,n_pixels)).reshape(n_pixels,n_pixels)
        filter = filter/np.amax(filter)
        filters.append(filter)
    
    I_ = np.array([M.reshape(n_features,28).T for M in np.split(np.stack(tuple(filters),axis=1),28)],dtype=np.float32)

    #plotting_tuple = (img, I_bin, I_rad, I_height)
    
    return I_

####################
# DISTANCE MATRICES
####################


def compute_distance_cub(I, J, dim, step=0.05):

    if I.shape[-1]==2 and len(I.shape)==3:
        
        return fast_grid_search(I, J, step, dim)

    elif I.shape[-1]>2 and len(I.shape)==3:

        card=I.shape[0]*I.shape[0]//2

        # random initial projection 
        x = np.hstack((np.array([1]),np.random.rand(I.shape[-1]-2)*np.pi,np.random.rand(1)*np.pi*2))
        p_ = np.abs(n_sphere.convert_rectangular(x).reshape(I.shape[-1],1))
        p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)

        model = CubicalModel_ISM(p, I, J, dim=dim, card=card)

        return fast_optim(model).numpy()

    elif len(I.shape)==2 or (len(I.shape)==3 and I.shape[-1]==1):

        cc = gd.CubicalComplex(dimensions=I.shape, top_dimensional_cells=I.flatten())
        pers = cc.persistence()
        cc_ = gd.CubicalComplex(dimensions=J.shape, top_dimensional_cells=J.flatten())
        pers_ = cc_.persistence()

        pers1 = [[tuple[1][0],tuple[1][1]] for tuple in pers if tuple[0]==dim and tuple[1][1]!=np.inf]
        pers2 = [[tuple[1][0],tuple[1][1]] for tuple in pers_ if tuple[0]==dim and tuple[1][1]!=np.inf]

        dgm1 = np.array(pers1).reshape(len(pers1),2)
        dgm2 = np.array(pers2).reshape(len(pers2),2)

        return wasserstein_distance(dgm1, dgm2, order=2)



def distance_matrix_cub(images, train_y, dim, filtrations, step=0.05):

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

                I = multifiltration(img1, filtrations)
                J = multifiltration(img2, filtrations)
                
                D[i,j] = compute_distance_cub(I, J, dim, step)

    D += np.transpose(D)

    fig = px.imshow(D,height=470, width=470)

    return D, fig, images_bb, labels_bb


###############################
# NEURAL NETWORKS CONNECTIVITY
###############################



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

#################################
# RUNNING TEST FOR MNIST CLASSIF
#################################


def run_experiment_cub(filtrations, images, labels, step=0.05, no_multi=False):

    matrices = []

    for i, filt in enumerate(filtrations):

        print("Computing Wasserstein matrix for filtration no.{}...".format(i+1))

        D_1, fig_1, images_bb, labels_bb = distance_matrix_cub(images, labels, 1, [filt])
        D_0, fig_0, images_bb, labels_bb = distance_matrix_cub(images, labels, 0, [filt])
        D = np.maximum(D_1, D_0)

        matrices.append(D)

    if not no_multi:

        print("Computing LISM matrix...")
        
        D_sheaf_1, fig_1, images_bb, labels_bb = distance_matrix_cub(images, labels, 1, filtrations, step)
        D_sheaf_0, fig_0, images_bb, labels_bb = distance_matrix_cub(images, labels, 0, filtrations, step)

        D_sheaf = np.maximum(D_sheaf_1, D_sheaf_0)

        return tuple(matrices), D_sheaf, labels_bb

    return tuple(matrices), labels_bb

def run_experiment_simp(data, step=0.05, filt=0):

    matrices = []

    data_ = copy.copy(data)

    n_filts = len(data_[0]['Multifiltration'])

    datas = [data_ for _ in range(n_filts)]

    for i in range(n_filts):

        for j, G in enumerate(data_):

            print(i, j)
            
            d=[G['Multifiltration'][i]]
            datas[i][j]['Multifiltration']=d

        print("Computing Wasserstein matrix for filtration no.{}...".format(i+1))

        D_1, fig_1, images_bb, labels_bb = distance_matrix_simp(datas[i], dim=1, step=step)
        D_0, fig_0, images_bb, labels_bb = distance_matrix_simp(datas[i], dim=0, step=step)
        D = np.maximum(D_1, D_0)

        matrices.append(D)

        print("ok")

    print("Computing LISM matrix...")
    
    D_sheaf_1, fig_1, images_bb, labels_bb = distance_matrix_simp(data_, dim=1, step=step, num_filt=filt)
    D_sheaf_0, fig_0, images_bb, labels_bb = distance_matrix_simp(data_, dim=0, step=step, num_filt=filt)

    D_sheaf = np.maximum(D_sheaf_1, D_sheaf_0)

    return tuple(matrices), D_sheaf, labels_bb


def accuracy(matrices_tuple, labels_bb, n_clusters):

    N = matrices_tuple[0].shape[0]

    accuracies = []

    for D in matrices_tuple:
        acc = best_accuracy(D, labels_bb, n_clusters)
        accuracies.append(acc)

    return accuracies



def best_accuracy(D, labels_bb, n_clusters):

    N = labels_bb.shape[0]

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(D)
    pred = kmeans.labels_
    pred

    n_classes = len(set(labels_bb))

    perm = permutations(list(set(labels_bb)))

    errors_list = []

    for i, p in enumerate(list(perm)):

        L = copy.copy(labels_bb)

        where_0 = np.where(L == p[0])
        where_1 = np.where(L == p[1])
        where_2 = np.where(L == p[2])
        where_3 = np.where(L == p[3])

        L[where_0] = 0
        L[where_1] = 1
        L[where_2] = 2
        L[where_3] = 3

        C = np.count_nonzero(L-pred)

        errors_list.append({'error':C, 'permutation':p})

    sort = sorted(errors_list, key=lambda x: x['error'])

    return (N-sort[0]['error'])/N




###############################
# K1, K2 ARE TWO SIMPLEX TREES
###############################


def compute_distance_simp(G1, G2, dim, step=0.05, num_filt=0):

    st1 = G1['SimplexTree']
    st2 = G2['SimplexTree']

    fct1 = G1['Multifiltration']
    fct2 = G2['Multifiltration']

    if len(fct1)==2:

        F1 = np.stack(tuple(fct1),axis=1)
        F2 = np.stack(tuple(fct2),axis=1)
        
        return fast_grid_search_K1K2(F1, F2, st1, st2, step, dim)

    elif len(fct1)>2:

        # random initial projection 
        #x = np.hstack((np.array([1]),np.random.rand(len(fct1)-2)*np.pi,np.random.rand(1)*np.pi*2))
        #x = np.hstack((np.array([1]),np.random.rand(0)*np.pi,np.random.rand(1)*np.pi*2))
        #p_ = np.abs(n_sphere.convert_rectangular(x).reshape(len(fct1),1))
        #p_ = np.abs(n_sphere.convert_rectangular(x).reshape(2,1))
        #p = tf.Variable(initial_value=p_, trainable=True, dtype = tf.float32)

        F1 = np.stack(tuple(fct1),axis=1)[:,num_filt:num_filt+2]
        F2 = np.stack(tuple(fct2),axis=1)[:,num_filt:num_filt+2]
        return fast_grid_search_K1K2(F1, F2, st1, st2, step, dim)

        #model = SimplexTreeModel_ISM_K1K2(p, F1, F2, st1, st2, dim=dim, card=50)

        #return fast_optim(model).numpy()

    elif len(fct1)==1:

        F1 = fct1[0]
        F2 = fct2[0]

        # Assign new filtration values

        for v in range(st1.num_vertices()):
            st1.assign_filtration([v], F1[v])
        st1.make_filtration_non_decreasing()

        for v in range(st2.num_vertices()):
            st2.assign_filtration([v], F2[v])
        st2.make_filtration_non_decreasing()

        dgm1 = st1.persistence()
        dgm2 = st2.persistence()

        dgm1_ = np.array([[tuple[1][0], tuple[1][1]] for tuple in dgm1 if tuple[0]==dim and tuple[1][1]!=np.inf])
        dgm2_ = np.array([[tuple[1][0], tuple[1][1]] for tuple in dgm2 if tuple[0]==dim and tuple[1][1]!=np.inf])
        
        return wasserstein_distance(dgm1_, dgm2_, order=2)


def distance_matrix_simp(data, dim, step=0.05, num_filt=0):

    N = len(data)

    l_sorted = sorted(data, key=lambda d: d['Label']) 
    data_bb = [l_sorted[i]['Multifiltration'] for i in range(N)]
    labels_bb = np.array([l_sorted[i]['Label'] for i in range(N)])

    D = np.zeros((N,N))

    for i, G1 in tqdm(enumerate(l_sorted)):
        for j, G2 in enumerate(l_sorted):
            if i<j:
                
                D[i,j] = compute_distance_simp(G1, G2, dim, step, num_filt)

    D += np.transpose(D)

    fig = px.imshow(D,height=470, width=470)

    return D, fig, data_bb, labels_bb




def fast_grid_search_K1K2(fct1, fct2, st1, st2, step=0.01, dim=0):

    angles = np.arange(0, math.pi/2, step)
    linear_forms = [np.array([math.cos(theta), math.sin(theta)]).reshape(2,1) for theta in angles ]

    distances = []

    for p in linear_forms:


        fct1p = np.tensordot(fct1,p,1)
        fct2p = np.tensordot(fct2,p,1)

        # Assign new filtration values
        for v in range(st1.num_vertices()):
            st1.assign_filtration([v], fct1p[v])
        st1.make_filtration_non_decreasing()
        for v in range(st2.num_vertices()):
            st2.assign_filtration([v], fct2p[v])
        st2.make_filtration_non_decreasing()
        dgm1 = st1.persistence()
        dgm2 = st2.persistence()
        dgm1_ = np.array([[tuple[1][0], tuple[1][1]] for tuple in dgm1 if tuple[0]==dim and tuple[1][1]!=np.inf])
        dgm2_ = np.array([[tuple[1][0], tuple[1][1]] for tuple in dgm2 if tuple[0]==dim and tuple[1][1]!=np.inf])
        
        distances.append(wasserstein_distance(dgm1_, dgm2_, order=2))

    dist = max(distances)
    
    return dist





