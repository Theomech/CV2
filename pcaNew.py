import numpy.linalg as linalg
import numpy as np

def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample
    '''
    [n, d] = X.shape
    mean = np.zeros(d)
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    for i in range(n):
        mean += X[i, :]
    mean /= n

    Xm = np.zeros((n, d))
    for i in range(n):
        Xm[i, :] = X[i, :] - mean
    s = np.matmul(Xm, Xm.transpose()) / (n - 1)

    eigenvalues, eigenvectors = linalg.eigh(s)
    index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    eigenvectors = np.matmul(Xm.transpose(), eigenvectors)
    for i in range(nb_components):
        eigenvectors[:, i] = eigenvectors[:, i] / linalg.norm(eigenvectors[:, i])
    return eigenvalues, eigenvectors, mean