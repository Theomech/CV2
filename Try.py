import cv2
import cv2.cv as cv
from landmarks import landmarks
from ToothPCA import ToothPCA
import numpy as np
import fnmatch
import numpy.linalg as linalg
import matplotlib.pyplot as plt



def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def procloop(firstElem1, ProcArray):
    ProcOriginal = np.zeros((14,8,40,2))

    for j in range(14):
        for i in range(8):
            Y = ProcArray[j, i, :, :]
            [_, Z, _] = procrustes(firstElem1, Y, scaling=True, reflection='best')
            ProcOriginal[j, i, :, :] = Z


    ProcMean = np.mean(ProcOriginal, axis=0,)

    return ProcMean, ProcOriginal

def normalize(X):
    X = X.astype(float) / linalg.norm(X).astype(float)
    return X

def normb(X):
    centroid = np.mean(X)
    scale_factor = np.sqrt(np.power(X - centroid, 2).sum())
    X = X.dot(1. / scale_factor)
    return X

def as_vectors(X):
    mpla = []
    for i in X:
        mpla.append(i.as_vector())
    return np.array(mpla)

def normc(X):
    min = np.amin(X)
    max = np.amax(X)
    scale_factor = max - min
    X = X - min
    X = X.dot(1. / scale_factor)
    return X



"""Import Data
"""
landmarksOriginal, landmarksMirrored = landmarks()

firstElem1 = landmarksOriginal[0,0,:,:]
firstmirror = landmarksMirrored[0,0,:,:]
'''Applying Procrustes
First go of Procrustes'''
[ProcMean, ProcOriginal] = procloop(firstElem1, landmarksOriginal)
[ProcMeanMir, ProcOriginalMir] = procloop(firstmirror, landmarksMirrored)

meanshape = normc(np.mean(ProcMean, axis=0))
meanshapemir = normc(np.mean(ProcMean, axis=0))
'''Looping Procrustes in order to converge more'''
for i in range(80):
    [ProcMean, ProcOriginal] = procloop(meanshape, ProcOriginal)
    [ProcMeanMir, ProcOriginalMir] = procloop(meanshapemir, ProcOriginalMir)
    meanshape = np.mean(ProcMean, axis=0)

plt.figure(1)



'''Applying PCA'''


PCAres = ToothPCA(ProcOriginal,ProcOriginalMir)

k=331
for j in range(8):
   for i in PCAres[j,0,:,:]:
       plt.subplot(k)
       plt.scatter(i[0],i[1])
   k=k+1
plt.show()


