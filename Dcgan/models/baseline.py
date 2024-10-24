#!/usr/bin/env python3
# Dcgan/models/baseline.py
    """ Contains
    Generator, 
    Discriminator,
    DCGAN
    """
import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')
    ])
    return model