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

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image
    '''
    return np.dot(X-mu,W)

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return np.dot(W,Y) + mu

def pcaLand(Z):
    Z = np.reshape(Z, (28, 640))
    ASMeigenval, ASMeigenvec, ASMmu = pca(Z)
    return ASMeigenval,ASMeigenvec,ASMmu

def pcaRadio(preprocessedRadios):
    radios = np.zeros((14, 2073600))
    for i in range(14):
        radios[i, :] = np.reshape(preprocessedRadios[i, :], 2073600)
    Radioeigenval, Radioeigenvec, Radiomu = pca(radios, nb_components=5)
    return Radioeigenval,Radioeigenvec,Radiomu
