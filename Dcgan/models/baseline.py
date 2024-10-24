#!/usr/bin/env python3
# hl: Dcgan/models/baseline.py
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

def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_dcgan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False

    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)

    gan = tf.keras.models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan
