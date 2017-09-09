import numpy as np
from pca import pca


def ToothPCA(procOriginal):

    #X = np.reshape(procOriginal, (14,640))
    #XY = np.concatenate((X,Y), axis=0)
    PCA1 = pca(procOriginal,10)
    #PCA2 = pca(Y)
    eigval, eigvec, mu = PCA1
    #for i in eigval[i,:]:
    #    X = procOriginal[:, i, :, :]
    #    Y = procOriginalMir[:, 7 - i, :, :]
    #    X = np.concatenate((X, Y), axis=0)
    #    X = np.reshape(X, (28, 80))
    #    eigval, eigvec, mu = pca(X, 4)
    #    M = []
        #for i in range(0, np.len):
            #M.append(np.sqrt(eigval[i]) * eigvec[:, i] + mu)
        #pcakept = np.reshape(np.array(M).squeeze(), (4, 40, 2))
    explained = np.cumsum(eigval/np.sum(eigval))
    print(explained)
        #PCAresult.append(pcakept)

    #PCAresult = np.reshape(PCAresult, (8, 4, 40, 2))
    return _, eigval, eigvec, mu
