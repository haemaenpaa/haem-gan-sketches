import time

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Image_GAN:
    #
    # A simple GAN implementation for generating images.
    #
    def __init__(self, real_images, input_size=50, checkpoint_dir=None):
        #
        # Constructor.
        #
        # Parameters:
        #
        # real_images: The training dataset
        # input_size: The size of the random vector
        # checkpoint_dir: directory to save the discriminator and generator every 10 training epochs
        #
        self.real_images = real_images
        self.input_size = input_size
        self.input_population = real_images.shape[0]
        self.checkpoint_dir = checkpoint_dir

        input_height = real_images.shape[1]
        input_width = real_images.shape[2]

        pad_height = int((input_height - 20) / 2)
        pad_width = int((input_width - 20) / 2)

        self.generator = keras.Sequential([
            keras.layers.Dense(400, activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Reshape((20 + input_height % 2, 20 + input_width % 2, 1)),
            keras.layers.ZeroPadding2D(padding=(pad_height, pad_width)),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), padding='same'),
            keras.layers.Conv2DTranspose(filters=16, kernel_size=(4, 4), padding='same'),
            keras.layers.Conv2DTranspose(filters=8, kernel_size=(4, 4), padding='same'),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), padding='same'),
            keras.layers.Reshape(real_images.shape[1:])
        ])
        self.generator.compile(optimizer=tf.train.RMSPropOptimizer(2e-4), loss='mean_squared_error')

        self.discriminator = keras.Sequential([
            keras.layers.Reshape((input_height, input_width, 1)),
            keras.layers.Conv2D(filters=8, kernel_size=(4, 4), activation=tf.nn.relu),
            keras.layers.Conv2D(filters=16, kernel_size=(4, 4), activation=tf.nn.relu),
            keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation=tf.nn.relu),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation=tf.nn.relu)
        ])
        learning_rate = 5e-5

        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.discriminator.compile(optimizer=self.discriminator_optimizer,
                                   loss=tf.keras.losses.mean_absolute_error,
                                   metrics=['accuracy'])

        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=5 * learning_rate)
        self.stacked_model = keras.Sequential(self.generator.layers + self.discriminator.layers)
        self.stacked_model.compile(optimizer=self.generator_optimizer,
                                   loss=tf.keras.losses.mean_absolute_error,
                                   metrics=['accuracy'])

    def train_discriminator(self, epochs=3):
        #
        # Train the discriminator part of the GAN
        #
        start = time.clock()
        print('Generating %i random images' % self.input_population)
        random_data = np.random.rand(self.input_population, self.input_size)
        generated_images = self.generator.predict(random_data)
        duration = time.clock() - start

        print('Generated %i random images in %d seconds' % (generated_images.shape[0], duration))

        training_images = np.concatenate((self.real_images, generated_images), axis=0)
        training_labels = np.concatenate((np.ones((self.input_population, 1)),
                                          np.zeros((self.input_population, 1))), axis=0)
        self.discriminator.fit(training_images, training_labels, epochs=epochs)

    def train_generator(self, epochs=3):
        #
        # Train the generator side of the GAN
        #

        random_data = np.random.rand(self.input_population, self.input_size)

        self.discriminator.trainable = False
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.discriminator.loss,
                                   metrics=self.discriminator.metrics)

        self.stacked_model.compile(optimizer=self.generator_optimizer,
                                   loss=tf.keras.losses.mean_absolute_error,
                                   metrics=['accuracy'])

        self.stacked_model.fit(random_data, np.ones((self.input_population, 1)), epochs=epochs)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.discriminator.loss,
                                   metrics=self.discriminator.metrics)

    def train(self, epochs, sub_epochs=1):
        #
        # Alternate between training the discriminator and generator
        #
        for epoch in range(epochs):
            print('GAN epoch %i/%i' % (epoch+1, epochs))
            print('discriminator:')
            self.train_discriminator(sub_epochs)
            print('generator:')
            self.train_generator(sub_epochs)
            if self.checkpoint_dir and epoch % 10 == 0:
                filename = self.checkpoint_dir + '/epoch-%i-checkpoint-g' % epoch
                self.save_generator(filename)
                filename = self.checkpoint_dir + '/epoch-%i-checkpoint-d' % epoch
                self.save_discriminator(filename)

    def save_generator(self, filename):
        self.generator.save_weights(filename)

    def load_generator(self, path):
        self.generator.load_weights(path)
        self.generator.compile(optimizer=self.generator_optimizer,
                               loss=tf.keras.losses.mean_absolute_error,
                               metrics=['accuracy'])

    def save_discriminator(self, filename):
        self.discriminator.save_weights(filename)

    def load_discriminator(self, path):
        self.discriminator.load_weights(path)
        self.discriminator.compile(optimizer=self.generator_optimizer,
                               loss=tf.keras.losses.mean_absolute_error,
                               metrics=['accuracy'])

    def random_image(self):
        random_data = np.random.rand(1, self.input_size)
        return self.generator.predict(random_data)