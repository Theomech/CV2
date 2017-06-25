import numpy as np

def pca(X, nb_components=0):
    [n, d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    mu = X.mean(axis=0)
    for i in range(n):
        X[i, :] -= mu

    Covar = np.cov(X)

    eigenvalues, eigenvectors = np.linalg.eigh(Covar)


    indx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indx]
    eigenvectors = eigenvectors[:,indx][:,0:nb_components]

    eigenvectors = np.dot(X.T, eigenvectors)

    for i in range(nb_components):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    return (eigenvalues, eigenvectors, mu)