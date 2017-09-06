import numpy as np
import cv2
import os, sys
from pca import pca
#from scipy.ndimage import morphology

dirTrainRadios = sys.path[0] + '/_Data/Radiographs/'
dirTestRadios = sys.path[0] + '/_Data/Radiographs/extra'
preprocessedTrainRadios=[]
preprocessedTestRadios=[]


def load(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images


def resizeRadio(image):
    return cv2.resize(image, (1920, 1080))


'''
def pseudoBinarize(image):
    image[i<120] = 0
    return image
'''


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def clahe(img):
    """Creates a CLAHE object and applies it to the given image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The result of applying CLAHE to the given image.

    """
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)


def print_image(image, name):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(cv2.convertScaleAbs(sobelx), .5, cv2.convertScaleAbs(sobely), .5, 0)

def preprocess2(image):
    # blurs
    image=cv2.GaussianBlur(image,(7,7),0)
    #image=cv2.medianBlur(image,5)
    #image=cv2.bilateralFilter(image,1,50,50)

    image=cv2.equalizeHist(image)

    #edge
    #image=cv2.Canny(image, 10,60)
    #image=cv2.Laplacian(image,cv2.CV_64F)
    image=sobel(image)



    # image=cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, np.ones((300, 300)))
    # image=cv2.morphologyEx(image, cv2.MORPH_TOPHAT, np.ones((300, 300)))
    return image

def preprocess(image):
    # image = image[300:880,660:1260]
    image = cv2.bilateralFilter(image, 9, 175, 175)
    image = clahe(image)
    # mean = np.mean(image)
    # imageind = np.where(image < 50)
    # image[imageind] = 0
    # image = cv2.subtract(image, cv2.Laplacian(image,0))
    # image2 = 0.5*cv2.bilateralFilter(image, 9, 175, 175)

    # image = cv2.subtract(image,image2.astype(np.uint8))
    # image = cv2.bilateralFilter(image, 5, 275, 275)
    kernel = np.ones((400, 400))
    kernel2 = np.ones((100, 100))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel2)
    image = cv2.add(image, tophat)
    image = cv2.subtract(image, blackhat)
    image=sobel(image)
    return image

def getPreprocessedTrainingRadios():
    for image in load(dirTrainRadios):
        preprocessedTrainRadios.append(preprocess2(resizeRadio(image)))
    return preprocessedTrainRadios

def getPreprocessedTestingRadios():
    for image in load(dirTestRadios):
        preprocessedTestRadios.append(preprocess2(resizeRadio(image)))
    return preprocessedTestRadios

# print_image(image, 'Original radiograph')
# print_image(preprocess(resizeRadio(image)), 'Preprocessed radiograph')
# print_image(preprocess2(resizeRadio(image)), 'Preprocessed radiograph Str')
#print_image(pca(preprocess2(resizeRadio(image))),'pca')

