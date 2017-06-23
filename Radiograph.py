import numpy as np
import cv2
import os, sys
import fnmatch

dirAllRadio = sys.path[0] + '/_Data/Radiographs'


def load(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def pseudobinarize(image):
    height, width, depth = image.shape
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, depth):
                if image[i, j, k] / 256.0 < 0.1:
                    image[i, j, k] = 0
    return image


def print_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(image):
    # pseudobinarize
    img = pseudobinarize(image)
    # blur
    cv2.medianBlur(img, 5, 0)
    # Discard values lower with grey scale intensity lower than 0.5

    return img


image = preprocess(load(dirAllRadio).__getitem__(0))
print_image(image, 'Original radiograph')
print_image(preprocess(image), 'Preprocessed radiograph')
print_image(image, '')
