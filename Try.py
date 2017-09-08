import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

from landmarks import loadLandmarks
from pca import pca
from procrustes import procrustes


def procloop(firstElem1, L):
    Procresult = np.zeros(L.shape)
    shape = L.shape
    for j in range(shape[0]):
        for i in range(shape[1]):
            Y = L[j, :]
            [d, Z, tform] = procrustes(firstElem1, Y)
            Procresult[j, :, :] = Z
    ProcMean = np.mean(Procresult, axis=0)

    return ProcMean, Procresult


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

landmarks = loadLandmarks()
#land = np.zeros((28,8,80))
#shape = landmarks.shape
#
#for i in range(shape[0]):
#    for j in range(shape[1]):
#        for o in range(shape[2]):
#            land[i, j, o] = landmarks[i, j, o, 0]
#            land[i, j, o + shape[2]] = landmarks[i, j, o, 1]
#
landmarks = np.reshape(landmarks,(28,8,80))
# landmarksOriginal = np.reshape(landmarksOriginal,(28,8,40,2))
firstElem1 = np.mean(landmarks, axis=0)




'''Applying Procrustes
First go of Procrustes'''
[ProcMean, Procresult] = procloop(firstElem1, landmarks)

meanshape = ProcMean

#Procresult = np.reshape(Procresult, (28, 8, 40, 2))
#ProcMean = np.reshape(ProcMean, (8, 40, 2))
#landmarks = np.reshape(landmarks,(28,8,40,2))
#k=331
#for o in range(9):
#    plt.subplot(k)
#    for j in range(8):
#
#        for i in landmarks[o, j, :]:
#
#            plt.scatter(i[0], i[1])
#
#    k=k+1
#plt.show()



# meanshapemir = normc(np.mean(ProcMean, axis=0))
'''Looping Procrustes in order to converge more'''

for i in range(20):
    [ProcMean, Procresult] = procloop(meanshape, Procresult)
    meanshape = ProcMean

Procresult = np.reshape(Procresult, (28, 8, 40, 2))
ProcMean = np.reshape(ProcMean, (8, 40, 2))

#ProcMean Plotter  1 octuple
#for j in range(8):
#    for i in ProcMean[j, :]:
#
#        plt.scatter(i[0], i[1])
#
#    #k=k+1
#plt.show()

#9 octuples of teeth Plotter
#k=331
#for o in range(9):
#    plt.subplot(k)
#    for j in range(8):
#
#        for i in Procresult[o, j, :]:
#            plt.scatter(i[0], i[1])
#
#    k=k+1
#plt.show()


'''Applying PCA'''

eigval, eigvec, mu = pca(np.reshape(landmarks, (28, 640)), nb_components=6)
#explained = np.cumsum(eigval/np.sum(eigval))
#print(explained)
#Ya = project(eigvec, landmarks[0,:], mu)
#Xa= reconstruct(eigvec, Ya, mu)
#k = 331
#ProcMean = np.reshape(ProcMean, (8, 40, 2))
#Procresult = np.reshape(Procresult, (28, 8, 40, 2))

landmarks = np.reshape(landmarks, (28, 8, 80))
landmarks = np.reshape(landmarks, (28, 8, 40, 2))

for o in range(8):
    for i in landmarks[0, o, :]:
        # plt.subplot(k)
        plt.scatter(i[0], i[1])
# k=k+1
plt.show()

print("str")
