import numpy as np
import landmarks
import math, plotter


def procrustes(landmarks):
    landmarksReshaped = landmarks.reshape(8960, 2)
    # scaledTranslated = np.array(8960,2)
    # scaledTranslated[:, 0] = (landmarksReshaped[:, 0] - np.mean(landmarksReshaped[:, 0])) / np.std(landmarksReshaped[:, 0])
    # scaledTranslated[:, 1] = (landmarksReshaped[:, 1] - np.mean(landmarksReshaped[:, 1])) / np.std(landmarksReshaped[:, 1])
    scaledTranslated = (landmarksReshaped - np.mean(landmarksReshaped)) / np.std(landmarksReshaped)

    scaledReshaped = scaledTranslated.reshape(28, 320, 2)
    firstOct = scaledReshaped[0, :, :]
    normalized = []
    for i in range(0, 27):
        currentOct = scaledReshaped[i, :, :]
        num = (currentOct[:, 0] * firstOct[:, 1] - currentOct[:, 1] * firstOct[:, 0]).sum()
        denom = (currentOct[:, 0] * firstOct[:, 0] + currentOct[:, 1] * firstOct[:, 1]).sum()
        theta = math.atan2(num, denom)

        r_matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        b2 = np.dot(r_matrix, currentOct.transpose()).transpose()

        normalized.append(math.sqrt(((firstOct - b2) ** 2.0).sum()))
    # compute the error metric
    normalized=np.asarray(normalized)
    normalized.resize(28,320,2)
    return normalized


plotter.plotall(procrustes(landmarks.loadLandmarks()))
