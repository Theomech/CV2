import numpy as np


def procrustes(X, Y):
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

    traceTA = s.sum()

    # optimum scaling of Y
    b = traceTA * normX / normY

    # standarised distance between X and b*Y*T + c
    d = 1 - traceTA ** 2

    # transformed coords
    Z = normX * traceTA * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def procloop(firstElem1, L):
    Procresult = np.zeros(L.shape)
    shape = L.shape
    for j in range(shape[0]):
        for i in range(shape[1]):
            Y = L[j, :]
            [d, Z, tform] = procrustes(firstElem1, Y)
            Procresult[j, :, :] = Z
    ProcMean = np.mean(Procresult, axis=0)

    # Procresult = np.reshape(Procresult, (28, 8, 40, 2))
    # ProcMean = np.reshape(ProcMean, (320, 2))

    return ProcMean, Procresult

def proc(landmarks):
    lands = np.zeros((28, 320, 2))
    for i in range(28):
        lands[i, :] = landmarks[i, :] - landmarks[i, :].mean(0)
    firstElem = lands[0, :]
    ProcMean, Z = procloop.procloop(firstElem, lands)
    meanshape = ProcMean
    for i in range(20):
        [ProcMean, Z] = procloop.procloop(meanshape, Z)
        meanshape = ProcMean
    return Z
