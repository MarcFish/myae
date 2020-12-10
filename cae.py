import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from utils import show, scale, rescale


class ContractiveAutoencoder(keras.Model):
    def __init__(self, lambda_=1e-3):
        self.lambda_ = lambda_
        super(ContractiveAutoencoder, self).__init__()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            encode = self.encoder(x)
            decode = self.decoder(encode)
            loss = self.compiled_loss(y, decode) + tf.reduce_sum(tf.norm(tf.gradients(encode, x)[0])) * self.lambda_
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, decode)
        return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        super(ContractiveAutoencoder, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, padding="SAME"),
        ])

    def call(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return decode


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]

model = ContractiveAutoencoder()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5), loss=keras.losses.MeanSquaredError())
model.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
show(x_test, model(x_test).numpy())
