# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:40:00 2020

@author:       Genocs
@description:  In this exercise we are going to use load standard dataset
               stored as images folder
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models

# Import numpy
import numpy as np

# Import Math plot lib
import matplotlib.pyplot as plt

import pathlib

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

"""
Get a public training-set 
"""
data_dir = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))

CLASS_NAMES = np.array(
    [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print('CLASS_NAMES: %s' % CLASS_NAMES)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

BATCH_SIZE = 2048
CHANNELS = 3
IMG_HEIGHT = 32
IMG_WIDTH = 32
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(
                                                         IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(CLASS_NAMES))

train_images, train_labels = next(train_data_gen)

# Convert the one-shoot label vector to classes
# Please find a more elegant way
vector_classes = np.argmax(train_labels, axis=1)

vector_classes_list = []
for v in vector_classes:
    vector_classes_list.append([v])

training_label_classes = np.asarray(vector_classes_list, dtype=np.float32)

# Define the model type
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_images, training_label_classes, epochs=25)

# Evaluate it
predictions = model.predict(train_images[0:2])
print('predicts: %s' % predictions)
print('predict[0]: %s' % CLASS_NAMES[np.argmax(predictions[0])])
print('predict[1]: %s' % CLASS_NAMES[np.argmax(predictions[1])])

plotImages(train_images[:2])
