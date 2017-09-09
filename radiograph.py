import os
import sys

import cv2
import numpy as np

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


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def clahe(img):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)


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
        preprocessedTrainRadios.append(preprocess(image))
    return preprocessedTrainRadios

def getPreprocessedTestingRadios():
    for image in load(dirTestRadios):
        preprocessedTestRadios.append(preprocess2(image))
    return preprocessedTestRadios

# plotter.print_image(image, 'Original radiograph')
# plotter.print_image(preprocess(image), 'Preprocessed radiograph')
# plotter.print_image(resizeRadio(getPreprocessedTrainingRadios().__getitem__(1)), 'Preprocessed radiograph Str')
