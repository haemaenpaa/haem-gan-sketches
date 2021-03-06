# TensorFlow and tf.keras

from tensorflow import keras

# Helper libraries
import matplotlib.pyplot as plt

import sys
import os

from helper_functions import draw_demo_images
from image_gan import Image_GAN

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_data = test_images.astype('float32') / 255.0

gan = Image_GAN(training_data, checkpoint_dir='./checkpoints')


if sys.argv[1] and os.path.exists(sys.argv[1]):
    gan.load_generator(sys.argv[1])

gan.train(300)

if sys.argv[2]:
    gan.save_generator(sys.argv[2])

draw_demo_images(gan)
