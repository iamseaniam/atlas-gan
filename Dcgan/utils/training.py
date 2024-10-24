#!/usr/bin/env python3
# hl: Dcgan/utils/training.py
    """ functions for training the GAN and saving results """
import tensorflow as tf
from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt


def train_gan(generator, discriminator, gan, data, epochs, batch_size, latent_dim, log_dir)
    half_batch = batch_size // 2
    losses = []

    for epoch in range(epochs):
        # hl: training discrimintor
        real_images = data[np.random.randint(0, data.shape[0], half_batch)]
        fake_images = generator.predict(np.random.randn(half_batch, latent_dim))
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((hatch_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # hl: train generator
    noise = np.random.randn(batch_size, latent_dim)
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

    # hl: save losses and images
    losses.append((d_loss, g_loss))
    if epoch % 100 == 0:
        print(f"{epoch}/{epochs}, d_loss: {d_loss}, g_loss: {g_loss}")
        # hl: Save generated images
        save_images(generator, epoch, log_dir)

    if epoch % 10 == 0:
        save_generated_images(epoch, generator)

    if epoch % 20 == 0:
        generator.save(f'log/model_at_epoch_{epoch}.h5')

    return losses
