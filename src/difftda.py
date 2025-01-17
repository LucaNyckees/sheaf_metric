import numpy as np
import tensorflow as tf
import gudhi as gd

# In this file, we write functions based on the Gudhi library that compute persistence diagrams associated to
# different filtrations (lower star, Rips, cubical), as well as the corresponding positive and negative
# simplices. We also wrap these functions into Tensorflow models.


#########################################
# Lower star filtration on simplex tree #
#########################################

# The parameters of the model are the vertex function values of the simplex tree.


def SimplexTree(stbase, fct, dim, card):
    # Parameters: stbase (array containing the name of the file where the simplex tree is located)
    #             fct (function values on the vertices of stbase),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Copy stbase in another simplex tree st
    st = gd.SimplexTree()
    f = open(stbase[0], "r")
    for line in f:
        ints = line.split(" ")
        s = [int(v) for v in ints[:-1]]
        st.insert(s, -1e10)
    f.close()

    # Assign new filtration values
    for i in range(st.num_vertices()):
        st.assign_filtration([i], fct[i])
    st.make_filtration_non_decreasing()

    # Compute persistence diagram
    dgm = st.persistence()

    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = st.persistence_pairs()

    # Then, loop over all simplex pairs
    indices, pers = [], []
    for s1, s2 in pairs:
        # Select pairs with good homological dimension and finite lifetime
        if len(s1) == dim + 1 and len(s2) > 0:
            # Get IDs of the vertices corresponding to the filtration values of the simplices
            l1, l2 = np.array(s1), np.array(s2)
            i1 = l1[np.argmax(fct[l1])]
            i2 = l2[np.argmax(fct[l2])]
            indices.append(i1)
            indices.append(i2)
            # Compute lifetime
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort vertex pairs wrt lifetime
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 2])[perm][::-1, :].flatten())

    # Pad vertex pairs
    indices = indices[: 2 * card] + [
        0 for _ in range(0, max(0, 2 * card - len(indices)))
    ]
    return list(np.array(indices, dtype=np.int32))


class SimplexTreeModel(tf.keras.Model):
    def __init__(self, F, stbase="simplextree.txt", dim=0, card=50):
        super(SimplexTreeModel, self).__init__()
        self.F = F
        self.dim = dim
        self.card = card
        self.st = stbase

    def call(self):
        d, c = self.dim, self.card
        st, fct = self.st, self.F

        # Turn STPers into a numpy function
        SimplexTreeTF = lambda fct: tf.numpy_function(
            SimplexTree,
            [np.array([st], dtype=str), fct, d, c],
            [tf.int32 for _ in range(2 * c)],
        )

        # Don't try to compute gradients for the vertex pairs
        fcts = tf.reshape(fct, [1, self.F.shape[0]])
        inds = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(SimplexTreeTF, fcts, dtype=[tf.int32 for _ in range(2 * c)]),
        )

        # Get persistence diagram
        self.dgm = tf.reshape(tf.gather_nd(self.F, inds), [c, 2])
        return self.dgm


################################
##### LISM SIMPLICIAL MODEL ####
################################


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
        st, fct1, fct2 = (
            self.st,
            tf.tensordot(self.F, self.p, 1),
            tf.tensordot(self.G, self.p, 1),
        )

        # Turn STPers into a numpy function
        SimplexTreeTF = lambda fct: tf.numpy_function(
            SimplexTree,
            [np.array([st], dtype=str), fct, d, c],
            [tf.int32 for _ in range(2 * c)],
        )

        # Don't try to compute gradients for the vertex pairs
        fcts1 = tf.reshape(fct1, [1, fct1.shape[0]])
        fcts2 = tf.reshape(fct2, [1, fct2.shape[0]])
        inds1 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(SimplexTreeTF, fcts1, dtype=[tf.int32 for _ in range(2 * c)]),
        )
        inds2 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(SimplexTreeTF, fcts2, dtype=[tf.int32 for _ in range(2 * c)]),
        )

        # Get persistence diagram
        dgm1 = tf.reshape(tf.gather_nd(fct1, inds1), [c, 2])
        dgm2 = tf.reshape(tf.gather_nd(fct2, inds2), [c, 2])
        return dgm1, dgm2


class SimplexTreeModel_ISM_K1K2(tf.keras.Model):
    def __init__(
        self,
        p,
        F,
        G,
        stbase1="simplextree.txt",
        stbase2="simplextree.txt",
        dim=0,
        card=50,
    ):
        super(SimplexTreeModel_ISM_K1K2, self).__init__()
        self.p = p
        self.F = F
        self.G = G
        self.dim = dim
        self.card = card
        self.st1 = stbase1
        self.st2 = stbase2

    def call(self):
        d, c = self.dim, self.card
        st1, st2, fct1, fct2 = (
            self.st1,
            self.st2,
            tf.tensordot(self.F, self.p, 1),
            tf.tensordot(self.G, self.p, 1),
        )

        # Turn STPers into a numpy function
        SimplexTreeTF1 = lambda fct: tf.numpy_function(
            SimplexTree,
            [np.array([st1], dtype=str), fct, d, c],
            [tf.int32 for _ in range(2 * c)],
        )
        SimplexTreeTF2 = lambda fct: tf.numpy_function(
            SimplexTree,
            [np.array([st2], dtype=str), fct, d, c],
            [tf.int32 for _ in range(2 * c)],
        )

        # Don't try to compute gradients for the vertex pairs
        fcts1 = tf.reshape(fct1, [1, fct1.shape[0]])
        fcts2 = tf.reshape(fct2, [1, fct2.shape[0]])
        inds1 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(SimplexTreeTF1, fcts1, dtype=[tf.int32 for _ in range(2 * c)]),
        )
        inds2 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(SimplexTreeTF2, fcts2, dtype=[tf.int32 for _ in range(2 * c)]),
        )

        # Get persistence diagram
        dgm1 = tf.reshape(tf.gather_nd(fct1, inds1), [c, 2])
        dgm2 = tf.reshape(tf.gather_nd(fct2, inds2), [c, 2])
        return dgm1, dgm2


############################
# Vietoris-Rips filtration #
############################

# The parameters of the model are the point coordinates.


def Rips(DX, mel, dim, card):
    # Parameters: DX (distance matrix),
    #             mel (maximum edge length for Rips filtration),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    rc = gd.RipsComplex(distance_matrix=DX, max_edge_length=mel)
    st = rc.create_simplex_tree(max_dimension=dim + 1)
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [
                s1[v]
                for v in np.unravel_index(
                    np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)]
                )
            ]
            i2 = [
                s2[v]
                for v in np.unravel_index(
                    np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)]
                )
            ]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[: 4 * card] + [
        0 for _ in range(0, max(0, 4 * card - len(indices)))
    ]
    return list(np.array(indices, dtype=np.int32))


class RipsModel(tf.keras.Model):
    def __init__(self, X, mel=12, dim=1, card=50):
        super(RipsModel, self).__init__()
        self.X = X
        self.mel = mel
        self.dim = dim
        self.card = card

    def call(self):
        m, d, c = self.mel, self.dim, self.card

        # Compute distance matrix
        DX = tf.math.sqrt(
            tf.reduce_sum(
                (tf.expand_dims(self.X, 1) - tf.expand_dims(self.X, 0)) ** 2, 2
            )
        )
        DXX = tf.reshape(DX, [1, DX.shape[0], DX.shape[1]])

        # Turn numpy function into tensorflow function
        RipsTF = lambda DX: tf.numpy_function(
            Rips, [DX, m, d, c], [tf.int32 for _ in range(4 * c)]
        )

        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        ids = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(RipsTF, DXX, dtype=[tf.int32 for _ in range(4 * c)]),
        )

        # Get persistence diagram by simply picking the corresponding entries in the distance matrix
        if d > 0:
            dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(ids, [2 * c, 2])), [c, 2])
        else:
            ids = tf.reshape(ids, [2 * c, 2])[1::2, :]
            dgm = tf.concat(
                [tf.zeros([c, 1]), tf.reshape(tf.gather_nd(DX, ids), [c, 1])], axis=1
            )

        return dgm


######################
# Cubical filtration #
######################

# The parameters of the model are the pixel values.


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
        pers = [
            X[np.unravel_index(cof[idx, 1], Xs)] - X[np.unravel_index(cof[idx, 0], Xs)]
            for idx in range(len(cof))
        ]
        perm = np.argsort(pers)
        cof = cof[perm[::-1]]

    # Retrieve and ouput image indices/pixels corresponding to positive and negative simplices
    D = len(Xs)
    ocof = np.array([0 for _ in range(D * card * 2)])
    count = 0
    for idx in range(0, min(2 * card, 2 * cof.shape[0]), 2):
        ocof[D * idx : D * (idx + 1)] = np.unravel_index(cof[count, 0], Xs)
        ocof[D * (idx + 1) : D * (idx + 2)] = np.unravel_index(cof[count, 1], Xs)
        count += 1
    return list(np.array(ocof, dtype=np.int32))


class CubicalModel(tf.keras.Model):
    def __init__(self, X, dim=1, card=50):
        super(CubicalModel, self).__init__()
        self.X = X
        self.dim = dim
        self.card = card

    def call(self):
        d, c, D = self.dim, self.card, len(self.X.shape)
        XX = tf.reshape(self.X, [1, self.X.shape[0], self.X.shape[1]])

        # Turn numpy function into tensorflow function
        CbTF = lambda X: tf.numpy_function(
            Cubical, [X, d, c], [tf.int32 for _ in range(2 * D * c)]
        )

        # Compute pixels associated to positive and negative simplices
        # Don't compute gradient for this operation
        inds = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(CbTF, XX, dtype=[tf.int32 for _ in range(2 * D * c)]),
        )

        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm = tf.reshape(tf.gather_nd(self.X, tf.reshape(inds, [-1, D])), [-1, 2])
        return dgm


##########################
##### LISM CUB MODEL #####
##########################


class CubicalModel_ISM(tf.keras.Model):
    def __init__(self, p, I, J, dim=1, card=50):
        super(CubicalModel_ISM, self).__init__()
        self.p = p
        self.I = I
        self.J = J
        self.dim = dim
        self.card = card

    def call(self):
        Xp = tf.reshape(tf.tensordot(self.I, self.p, 1), shape=[28, 28])
        Yp = tf.reshape(tf.tensordot(self.J, self.p, 1), shape=[28, 28])

        d, c, D = self.dim, self.card, len(Xp.shape)
        XX = tf.reshape(Xp, [1, Xp.shape[0], Xp.shape[1]])
        YY = tf.reshape(Yp, [1, Yp.shape[0], Yp.shape[1]])

        # Turn numpy function into tensorflow function
        CbTF = lambda X: tf.numpy_function(
            Cubical, [X, d, c], [tf.int32 for _ in range(2 * D * c)]
        )

        # Compute pixels associated to positive and negative simplices
        # Don't compute gradient for this operation

        inds1 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(
                CbTF, XX, fn_output_signature=[tf.int32 for _ in range(2 * D * c)]
            ),
        )
        inds2 = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(
                CbTF, YY, fn_output_signature=[tf.int32 for _ in range(2 * D * c)]
            ),
        )

        # Get persistence diagram by simply picking the corresponding entries in the image
        dgm1 = tf.reshape(tf.gather_nd(Xp, tf.reshape(inds1, [-1, D])), [-1, 2])
        dgm2 = tf.reshape(tf.gather_nd(Yp, tf.reshape(inds2, [-1, D])), [-1, 2])
        return dgm1, dgm2
