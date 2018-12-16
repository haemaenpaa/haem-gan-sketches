
import numpy as np
from tensorflow import keras
import tensorflow as tf
import sys
import pylab
from matplotlib import pyplot as plt

from helper_functions import plot_image

import sys
import os

from helper_functions import draw_demo_images
from image_gan import Image_GAN

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_data = test_images.astype('float32') / 255.0

gan = Image_GAN(training_data, checkpoint_dir='./checkpoints')

gan.load_generator(sys.argv[1])

draw_demo_images(gan)