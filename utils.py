import matplotlib.pyplot as plt
import numpy as np


def show(real, fake, n=16):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(real[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(fake[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def scale(image):
    image = image.astype(np.float32)
    return (image - 127.5) / 127.5


def rescale(image):
    return (image * 127.5 + 127.5).astype(np.uint8)
