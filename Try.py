import cv2
import cv2.cv as cv
import os,sys
import numpy as np
import fnmatch
import numpy.linalg as linalg
import matplotlib.pyplot as plt


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
landmarksOriginal = np.reshape(landmarksOriginal,(14,8,40,2))

for landmark in os.listdir(dirMirrored):
    textFile = open(os.path.join(dirMirrored, landmark), 'r')
    coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
    landmarksMirrored.append(coordinates)


###Transforming the initial list for Mirrored elements to array and then reshaping
landmarksMirrored = np.asarray(landmarksMirrored)
landmarksMirrored = np.reshape(landmarksMirrored,(14,8,40,2))


##normalize
landmarksOriginal = landmarksOriginal.astype(float) / linalg.norm(landmarksOriginal).astype(float)
landmarksMirrored = landmarksMirrored.astype(float) / linalg.norm(landmarksMirrored).astype(float)


##Des edw ti ginetai...
for j in range(8):
    for i in landmarksOriginal[0,j,:,:]:
        plt.scatter(i[0], i[1])
plt.show()