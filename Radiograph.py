import numpy as np
import cv2
import os, sys
import fnmatch

dirAllRadio = sys.path[0] + '/_Data/Radiographs'


def load(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images


def resizeRadio(image):
    return cv2.resize(image, (1920, 1080))


def pseudoBinarize(image):
    height, width = image.shape
    for i in range(0, height):
        for j in range(0, width):
            if image[i, j] / 256.0 < 0.5:
                image[i, j] = 0
    return image


def print_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)


def preprocess(image):
    # blur
    image = cv2.bilateralFilter(image, 9, 75, 75)

    image = cv2.equalizeHist(image)

    image = sobel(image)

    return image


image = load(dirAllRadio).__getitem__(0)
print(image.shape)
print_image(image, 'Original radiograph')
image = resizeRadio(image)
print(image.shape)
print_image(image, "Resized")
image = preprocess(image)
print(image.shape)
print_image(image, 'Preprocessed radiograph')
