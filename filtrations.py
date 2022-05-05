import numpy as np
from sklearn.preprocessing import binarize

def radial_filt(I, center=[6,13]):

    dists = [0]
    c1 = center[0]
    c2 = center[1]

    for i, col in enumerate(I[:,0]):
        for j,pix in enumerate(I[i]):
        
            if pix == 1:
                dists.append(np.linalg.norm([i-c1,j-c2]))

    R = max(dists)

    def radius(pix,i,j,R):

        if pix == 1:
            return R
        elif pix == 0:
            return np.linalg.norm([i-c1,j-c2])

    # input is a binarized image
    filt = np.zeros_like(I)

    for i, col in enumerate(I[:,0]):
        for j,pix in enumerate(I[i]):
            
            filt[i,j] =  radius(pix,i,j,R)

    return filt


def height_filt(I, v = [1,0]):

    filt = np.zeros_like(I)

    H_max = 1

    for i, col in enumerate(I[:,0]):
        for j,pix in enumerate(I[i]):
        
            if pix == 1:
                
                filt[i,j] = H_max

            elif pix == 0:

                filt[i,j]=np.matmul(v,[i,j])


    return filt

def rad_filt(img):

    return radial_filt(binarize(img))

def hei_filt(img,v=[1,0]):

    return height_filt(binarize(img),v)

def wass_amplitudes(images, alpha, dim=1):

    A = []

    for I in tqdm(images):

        cc = gd.CubicalComplex(dimensions=I.shape, top_dimensional_cells=I.flatten())
        dgm = np.array([p[1] for p in cc.persistence() if p[0]==dim])

        A.append(alpha * np.square(wasserstein_distance(dgm, [], order=2)))

    return np.array(A)