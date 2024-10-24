#!/usr/bin/env python3
""""""
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # HL Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

