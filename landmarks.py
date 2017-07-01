import os,sys
import numpy as np

def landmarks():
    '''Specifing directories'''
    dirAll = sys.path[0] + '/_Data/Landmarks'
    dirOriginal = sys.path[0] + r"/_Data/Landmarks/original"
    dirMirrored = sys.path[0] + r"/_Data/Landmarks/mirrored"

    '''Declaring the 2 lists that will have the coordinates'''
    landmarksOriginal = []
    landmarksMirrored = []

    '''Importing data'''
    for landmark in os.listdir(dirOriginal):
        textFile = open(os.path.join(dirOriginal, landmark), 'r')
        coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
        landmarksOriginal.append(coordinates)

    '''Transforming the initial list for Original elements to array and then reshaping'''
    landmarksOriginal = np.asarray(landmarksOriginal)
    landmarksOriginal = np.reshape(landmarksOriginal, (14, 8, 40, 2)).astype(float)

    for landmark in os.listdir(dirMirrored):
        textFile = open(os.path.join(dirMirrored, landmark), 'r')
        coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
        landmarksMirrored.append(coordinates)

    '''Transforming the initial list for Mirrored elements to array and then reshaping'''
    landmarksMirrored = np.asarray(landmarksMirrored)
    landmarksMirrored = np.reshape(landmarksMirrored, (14, 8, 40, 2)).astype(float)

    textFile.close()
    return landmarksOriginal, landmarksMirrored