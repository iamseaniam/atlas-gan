#!/usr/bin/env python3
# Set up deep learning framework:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequrntial
from tensorflow.keras.layers import Dense,Flatten
import matplotlib as plt


# Loading in the mnist dataset
(x_train, y_train), (x_Test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing data, values down to a range of 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
