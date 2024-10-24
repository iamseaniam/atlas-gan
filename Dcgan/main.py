#!/usr/bin/env python3
""""""
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float')

X_train /= 255
X_test /= 255

print(X_train.shape)
print(X_test.shape)