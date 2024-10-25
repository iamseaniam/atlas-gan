import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb

# Initialize WandB
wandb.init(project="advanced_gan_project")

# Set up paths and parameters
data_dir = 'PATH< but can download the stupid '
train_data_dir = os.path.join(data_dir, 'img_align_celeba')
image_size = (64, 64)  # Resize images to 64x64
batch_size = 32
num_epochs = 100

# Set up ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True
)

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64 * 64 * 3, activation='tanh'))
    model.add(layers.Reshape((64, 64, 3)))
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(64, 64, 3)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combined model
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training the GAN
for epoch in range(num_epochs):
    for _ in range(batch_size):
        noise = tf.random.normal(shape=(batch_size, 100))

        fake_images = generator(noise)

        real_images = next(train_generator)

        combined_images = tf.concat([real_images, fake_images], axis=0)

        labels = tf.constant([[1.0]] * batch_size + [[0.0]] * batch_size)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        noise = tf.random.normal(shape=(batch_size, 100))

        g_loss = gan.train_on_batch(noise, [[1.0]] * batch_size)

    wandb.log({
        'epoch': epoch,
        'generator_loss': g_loss,
        'discriminator_loss': d_loss[0],
    })

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')

print("Training complete! Models saved.")
