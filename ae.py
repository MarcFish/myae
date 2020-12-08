import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

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


def scale(image):
    image = image.astype(np.float32)
    return (image - 127.5) / 127.5


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]

model = Autoencoder()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5), loss=keras.losses.MeanAbsoluteError())
model.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
