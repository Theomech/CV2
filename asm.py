import numpy as np

def initialfit(X):

    X = ( X / np.amax(X)) * 200
    X = np.reshape(X,(320,2))
    length = 1920 / 2
    height = 1080 / 1.6

    X = X + [length, height]

    return X

