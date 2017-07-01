import numpy as np
from pca import pca


def ToothPCA(ProcOriginal,ProcOriginalMir):
    PCAres = []
    for i in range(8):
        X = ProcOriginal[:, i, :, :]
        Y = ProcOriginalMir[:, 7 - i, :, :]
        X = np.concatenate((X, Y), axis=0)
        X = np.reshape(X, (28, 80))
        eigval, eigvec, mu = pca(X, 4)
        M = []
        for i in range(0, 4):
            M.append(np.sqrt(eigval[i]) * eigvec[:, i] + mu)
        pcakept = np.reshape(np.array(M).squeeze(), (4, 40, 2))
        # explained = np.cumsum(eigval/np.sum(eigval))
        # print(explained)
        PCAres.append(pcakept)

    PCAres = np.reshape(PCAres, (8, 4, 40, 2))
    return PCAres, eigval, eigvec, mu