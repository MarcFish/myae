import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


class DenoiseAutoencoder(keras.Model):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()

    def build(self, input_shape):
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=1, padding="SAME"),
        ])

    def call(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return decode
