
from tensorflow import keras
from image_gan import Image_GAN

import sys

from helper_functions import draw_demo_images

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


training_data = train_images.astype('float32') / 255.0

gan = Image_GAN(training_data, checkpoint_dir='./checkpoints')

gan.load_generator(sys.argv[1])
gan.load_discriminator(sys.argv[2])

epochs = 10

outfile=None
if sys.argv[3]:
    outfile = sys.argv[3]

gan.train(epochs=epochs)

if outfile:
    gan.save_generator(outfile)

draw_demo_images(gan)