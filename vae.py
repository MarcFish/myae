import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from utils import show, scale, rescale


class ConvolutionalVariationalAutoencoder(keras.Model):
    def __init__(self, latent_dim=512):
        self.latent_dim = latent_dim
        super(ConvolutionalVariationalAutoencoder, self).__init__()

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            logpz = self.log_normal_pdf(z, 0., 0.)
            logqz_x = self.log_normal_pdf(z, mean, logvar)
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, x_logit)
        return {m.name: m.result() for m in self.metrics}

    def build(self, input_shape):
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(self.latent_dim + self.latent_dim)
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(7*7*16),
            keras.layers.Reshape((7, 7, 16)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, padding="SAME"),
        ])
        super(ConvolutionalVariationalAutoencoder, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.sample(z)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        super(ConvolutionalVariationalAutoencoder, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = (x_train / 255.0)[..., np.newaxis]
x_test = (x_test / 255.0)[..., np.newaxis]

model = ConvolutionalVariationalAutoencoder()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5), loss=keras.losses.MeanSquaredError())
model.build([])
model.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
show(x_test, model(x_test).numpy())
