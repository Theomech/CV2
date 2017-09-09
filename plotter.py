import cv2
import matplotlib.pyplot as plt

#Plotting 1 set of Teeth
def plotmean(array):
    for i in array[:]:
        plt.scatter(i[0], i[1])
    plt.show()



#Plotting 1 set of Teeth
def plot1_8(array):
    for i in array[0, :]:
        plt.scatter(i[0], i[1])
    plt.show()


#Plotting 9 sets of Teeth
def plot9_8(array):
    k = 331
    for o in range(9):
        plt.subplot(k)
        for i in array[o, :]:
            plt.scatter(i[0], i[1])
        k = k + 1
    plt.show()


#Plotting 1 specified set of 8Teeth
def spplot1_8(array, sp):
    for i in array[sp, :]:
        plt.scatter(i[0], i[1])
    plt.show()

#Plotting all sets of teeth
def plotall(array):
    for j in range(28):
        for i in array[j, :]:
            plt.scatter(i[0], i[1])
    plt.show()


def print_image(image, name):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
