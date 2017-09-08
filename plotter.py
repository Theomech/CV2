import matplotlib.pyplot as plt


def plot(array):
    for i in range(8):
        for j in array[0,i,:]:
            plt.scatter(j[0], j[1])
    plt.show()
