import cv2
import cv2.cv as cv
from landmarks import landmarks
from ToothPCA import ToothPCA
import numpy as np
import fnmatch
from procrustes import procrustes
import numpy.linalg as linalg
import matplotlib.pyplot as plt



def procloop(firstElem1, ProcArray):
    ProcOriginal = np.zeros((14, 8, 80))

    for j in range(14):
        #for i in range(8):
        Y = ProcArray[j, :, :]
        [_, Z, _] = procrustes(firstElem1, Y, scaling=True, reflection='best')
        ProcOriginal[j, :, :] = Z


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

#swsto pca gia tin 8ada dontiwn
#prokrousti gia tin 8ada dontiwn

"""Import Data
"""
landmarksOriginal, landmarksMirrored = landmarks()

#firstElem1 = landmarksOriginal[0,:,:]
##firstElem1 = np.reshape(firstElem1, (8,80))
##landmarksOriginal = np.reshape(landmarksOriginal,(14,8,80))
##firstmirror = landmarksMirrored[0,:,:]
##firstmirror = np.reshape(firstmirror, (8,80))
#
#
#'''Applying Procrustes
#First go of Procrustes'''
#[ProcMean, ProcOriginal] = procloop(firstElem1, landmarksOriginal)
##[ProcMeanMir, ProcOriginalMir] = procloop(firstmirror, landmarksMirrored)
#
#meanshape = ProcMean
#
#
#
##ProcOriginal = np.reshape(ProcOriginal, (14,8,40,2))
##ProcMean = np.reshape(ProcMean,(8,40,2))
##for j in range(8):
##    for i in ProcMean[j,:]:
##       #plt.subplot(k)
##        plt.scatter(i[0],i[1])
###k=k+1
##plt.show()
#
#
#
##meanshapemir = normc(np.mean(ProcMean, axis=0))
#'''Looping Procrustes in order to converge more'''
#
#for i in range(50):
#    [ProcMean, ProcOriginal] = procloop(meanshape, ProcOriginal)
#    #[ProcMeanMir, ProcOriginalMir] = procloop(meanshapemir, ProcOriginalMir)
#    meanshape = ProcMean
#
#plt.figure(1)



'''Applying PCA'''

landmarksOriginal = np.reshape(landmarksOriginal,(14,640))
#landmarksOriginal = np.swapaxes(landmarksOriginal,0,1)
PCAres = ToothPCA(landmarksOriginal)



k=331
ProcMean = np.reshape(ProcMean,(8,40,2))
ProcOriginal = np.reshape(ProcOriginal, (14,8,40,2))
landmarksOriginal = np.reshape(landmarksOriginal, (14,8,40,2))
for j in range(2):
    for i in ProcOriginal[0,j,:]:
       #plt.subplot(k)
        plt.scatter(i[0],i[1])
#k=k+1
plt.show()
sadas

