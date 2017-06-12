import cv2
import cv2.cv as cv
import os,sys
import numpy as np
import fnmatch
import numpy.linalg as linalg
import matplotlib.pyplot as plt

dirAll = sys.path[0]+'/_Data/Landmarks'
dirOriginal = sys.path[0]+r"/_Data/Landmarks/original"
dirMirrored = sys.path[0]+r"/_Data/Landmarks/mirrored"
up1, up2, up3, up4, up5, up6, up7, up8, up9, up10, up11, up12, up13, up14, down1, down2, down3, down4, down5, down6, down7, down8, down9, down10, down11, down12, down13, down14 = (
    [] for i in range(28))

for landmarks in os.listdir(dirOriginal):
    textFile = open(os.path.join(dirOriginal, landmarks), 'r')
    coordinates = np.reshape(textFile.read().splitlines(), (40, 2)).tolist()
    # take image number
    part1 = landmarks.split("-")[0]
    imageNo = part1[len(part1) - 2:len(part1)]
    if imageNo[0].isalpha():
        imageNo = imageNo[1]
    # take tooth number
    toothNo = int(landmarks.split("-")[1].split(".")[0])


    if imageNo.__eq__("1"):
        if toothNo > 4:
            for co in coordinates:
                down1.append(co)
        else:
            for co in coordinates:
                up1.append(co)
    if imageNo.__eq__("2"):
        if toothNo > 4:
            for co in coordinates:
                down2.append(co)
        else:
            for co in coordinates:
                up2.append(co)
    if imageNo.__eq__("3"):
        if toothNo > 4:
            for co in coordinates:
                down3.append(co)
        else:
            for co in coordinates:
                up3.append(co)
    if imageNo.__eq__("4"):
        if toothNo > 4:
            for co in coordinates:
                down4.append(co)
        else:
            for co in coordinates:
                up4.append(co)
    if imageNo.__eq__("5"):
        if toothNo > 4:
            for co in coordinates:
                down5.append(co)
        else:
            for co in coordinates:
                up5.append(co)
    if imageNo.__eq__("6"):
        if toothNo > 4:
            for co in coordinates:
                down6.append(co)
        else:
            for co in coordinates:
                up6.append(co)
    if imageNo.__eq__("7"):
        if toothNo > 4:
            for co in coordinates:
                down7.append(co)
        else:
            for co in coordinates:
                up7.append(co)
    if imageNo.__eq__("8"):
        if toothNo > 4:
            for co in coordinates:
                down8.append(co)
        else:
            for co in coordinates:
                up8.append(co)
    if imageNo.__eq__("9"):
        if toothNo > 4:
            for co in coordinates:
                down9.append(co)
        else:
            for co in coordinates:
                up9.append(co)
    if imageNo.__eq__("10"):
        if toothNo > 4:
            for co in coordinates:
                down10.append(co)
        else:
            for co in coordinates:
                up10.append(co)
    if imageNo.__eq__("11"):
        if toothNo > 4:
            for co in coordinates:
                down11.append(co)
        else:
            for co in coordinates:
                up11.append(co)
    if imageNo.__eq__("12"):
        if toothNo > 4:
            for co in coordinates:
                down12.append(co)
        else:
            for co in coordinates:
                up12.append(co)
    if imageNo.__eq__("13"):
        if toothNo > 4:
            for co in coordinates:
                down13.append(co)
        else:
            for co in coordinates:
                up13.append(co)
    if imageNo.__eq__("14"):
        if toothNo > 4:
            for co in coordinates:
                down14.append(co)
        else:
            for co in coordinates:
                up14.append(co)
    textFile.close()
print(len(down8))

for i in down8:
    plt.scatter(i[0], i[1])
plt.show()


# 1.2 norm

def norma(list):
    return (list - np.mean(list)) / np.std(list)


normD1 = norma(np.array(down1).astype(np.float))
normD2 = norma(np.array(down1).astype(np.float))
normD3 = norma(np.array(down1).astype(np.float))
normD4 = norma(np.array(down1).astype(np.float))
normD5 = norma(np.array(down1).astype(np.float))
normD6 = norma(np.array(down1).astype(np.float))
normD7 = norma(np.array(down1).astype(np.float))
normD8 = norma(np.array(down1).astype(np.float))
normD9 = norma(np.array(down1).astype(np.float))
normD10 = norma(np.array(down1).astype(np.float))
normD11 = norma(np.array(down1).astype(np.float))
normD12 = norma(np.array(down1).astype(np.float))
normD13 = norma(np.array(down1).astype(np.float))
normD14 = norma(np.array(down1).astype(np.float))
normU1 = norma(np.array(down1).astype(np.float))
normU2 = norma(np.array(down1).astype(np.float))
normU3 = norma(np.array(down1).astype(np.float))
normU4 = norma(np.array(down1).astype(np.float))
normU5 = norma(np.array(down1).astype(np.float))
normU6 = norma(np.array(down1).astype(np.float))
normU7 = norma(np.array(down1).astype(np.float))
normU8 = norma(np.array(down1).astype(np.float))
normU9 = norma(np.array(down1).astype(np.float))
normU10 = norma(np.array(down1).astype(np.float))
normU11 = norma(np.array(down1).astype(np.float))
normU12 = norma(np.array(down1).astype(np.float))
normU13 = norma(np.array(down1).astype(np.float))
normU14 = norma(np.array(down1).astype(np.float))


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


# 1.3 pca
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
