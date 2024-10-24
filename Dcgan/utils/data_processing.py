#!/usr/bin/env python3
# Dcgan/utils/data_processing
""" Preparing the MNIST dataset """
import numpy as np
from tensorflow.keras.datasets import mnist


def preprocess_savedata():
    """ Preprocessing data and saving it """
    (x_train, _), (x_test, _) = mnist.load_data()
    # hl: normalize to [0, 1] and expand dimensions to add a channel
    x_train = np.expand_dims(x_train, axis=-1) / 255.0
    x_test = np.expand_dims(x_test, axis=-1) / 255.0