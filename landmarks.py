import os, sys
import numpy as np



def landmarks():
    # Specifying directories
    dirAll = sys.path[0] + '/_Data/Landmarks'
    dirOriginal = sys.path[0] + r"/_Data/Landmarks/original"
    dirMirrored = sys.path[0] + r"/_Data/Landmarks/mirrored"

    '''Declaring the 2 lists that will have the coordinates'''
    landmarks = []

    '''Importing data'''
    for landmark in os.listdir(dirOriginal):
        textFile = open(os.path.join(dirOriginal, landmark), 'r')
        coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
        landmarks.append(coordinates)

    for landmark in os.listdir(dirMirrored):
        textFile = open(os.path.join(dirMirrored, landmark), 'r')
        coordinates = np.reshape(textFile.read().splitlines(), (40, 2))
        landmarks.append(coordinates)

    '''Transforming the initial list for Original elements to array and then reshaping'''
    landmarks = np.asarray(landmarks)
    landmarks = np.reshape(landmarks, (28, 8, 40, 2)).astype(float)

    textFile.close()
    return landmarks

print (landmarks())
