import cv2
import cv2.cv as cv
import os,sys
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


def procloop(firstElem1, firstElem2, firstElem3, firstElem4, ProcArray):
    ProcOriginal = np.zeros((14,4,40,2))

    for j in range(14):
        for i in range(0,4,3):
            Y = ProcArray[j, i, :, :]
            [d,Z,tform] = procrustes(firstElem1, Y, scaling=True, reflection='best')
            ProcOriginal[j, 0, :, :] = Z
            Y = ProcArray[j, i + 4, :, :]
            [d, Z, tform] = procrustes(firstElem3, Y, scaling=True, reflection='best')
            ProcOriginal[j, 1, :, :] = Z

        for i in range(1,3):
            Y = ProcArray[j, i, :, :]
            [d, Z, tform] = procrustes(firstElem2, Y, scaling=True, reflection='best')
            ProcOriginal[j, 2, :, :] = Z
            Y = ProcArray[j, i + 4, :, :]
            [d, Z, tform] = procrustes(firstElem4, Y, scaling=True, reflection='best')
            ProcOriginal[j, 3, :, :] = Z

    ProcMean = np.mean(ProcOriginal, axis=0,)

    return ProcMean, ProcOriginal

def normalize(X):
    X = X.astype(float) / linalg.norm(X).astype(float)
    return X

##Defining PCA
#def pca(X, nb_components=0):
#    [n,d,mmm,mm] = X.shape
#    if (nb_components <= 0) or (nb_components>6):
#        nb_components = n
#    mu = X.mean(axis=0)
#    for i in range(n):
#        X[i,:] -= mu
#
#    Covar = (np.dot(X, X.T) / float(n)) #some use n-1
#
#    eigenvalues, eigenvectors = np.linalg.eigh(Covar)
#
#
#    indx = np.argsort(eigenvalues)[::-1]
#    eigenvalues = eigenvalues[indx]
#    eigenvectors = eigenvectors[:,indx][:,0:nb_components]
#
#    eigenvectors = np.dot(X.T, eigenvectors)
#
#    for i in range(nb_components):
#        eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])
#
#    return (eigenvalues, eigenvectors, mu)

##Performing PCA

#[eigenvaluesOriginal, eigenvectorsOriginal, muOriginal] = pca(landmarksOriginal,nb_components=0)
#[eigenvaluesMirrored, eigenvectorsMirrored, muMirrored] = pca(landmarksMirrored,nb_components=0)



##Des edw ti ginetai...


###normalize
#landmarksOriginal = normalize(landmarksOriginal)
#landmarksMirrored = normalize(landmarksMirrored)


###Specifing directories



dirAll = sys.path[0]+'/_Data/Landmarks'
dirOriginal = sys.path[0]+r"/_Data/Landmarks/original"
dirMirrored = sys.path[0]+r"/_Data/Landmarks/mirrored"

###Declaring the 2 arrays that will have the coordinates
landmarksOriginal = []
landmarksMirrored = []

###Importing data
for landmark in os.listdir(dirOriginal):
    textFile = open(os.path.join(dirOriginal, landmark), 'r')
    coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
    landmarksOriginal.append(coordinates)

###Transforming the initial list for Original elements to array and then reshaping
landmarksOriginal = np.asarray(landmarksOriginal)
landmarksOriginal = np.reshape(landmarksOriginal,(14,8,40,2)).astype(float)

for landmark in os.listdir(dirMirrored):
    textFile = open(os.path.join(dirMirrored, landmark), 'r')
    coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
    landmarksMirrored.append(coordinates)


###Transforming the initial list for Mirrored elements to array and then reshaping
landmarksMirrored = np.asarray(landmarksMirrored)
landmarksMirrored = np.reshape(landmarksMirrored,(14,8,40,2)).astype(float)

###4 Classes of teeth
###1st Class = upper lateral incisors
firstElem1 = landmarksOriginal[0,0,:,:]
###2nd Class = upper central incisors
firstElem2 = landmarksOriginal[0,1,:,:]
###3rd Class = lower lateral incisors
firstElem3 = landmarksOriginal[0,4,:,:]
###4th Class = lower central incisors
firstElem4 = landmarksOriginal[0,5,:,:]

#First go of Procrustes
[ProcMean, ProcOriginal ]= procloop(firstElem1, firstElem2, firstElem3, firstElem4, landmarksOriginal)

for i in range(30):
    [ProcMean, ProcOriginal ]= procloop(ProcMean[0,:,:], ProcMean[1,:,:], ProcMean[2,:,:], ProcMean[3,:,:], landmarksOriginal)


for j in range(4):
    for i in ProcMean[j,:,:]:
        plt.scatter(i[0], i[1])
plt.show()

for j in range(1,2):
    for i in ProcOriginal[0,j,:,:]:
        plt.scatter(i[0], i[1])
plt.show()


