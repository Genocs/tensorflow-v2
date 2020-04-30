# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:30:00 2020

@author:       Genocs
@description:  In this exercise we are going to use CIFAR-10 dataset
               the model will use real Deep neural network with CNN 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models

# Import numpy
import numpy as np

# Import Math plot lib
import matplotlib.pyplot as plt

import IPython.display as display
from PIL import Image
import os
import pathlib

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

# Load the feature and label, splitted by training_set and test_set
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Simple logdump
#print('train_images: %s' % train_images)
print('train_labels: %s' % train_labels)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


#plotImages(train_images[:5])

# Define the model type
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Fit the model
history = model.fit(train_images, train_labels, epochs=10)

# Evaluate it
print(model.predict(train_images[:2]))
