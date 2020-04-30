# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:00:00 2020

@author:       Genocs
@description:  In this exercise we are going to use the hello-world MNIST dataset 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras
from tensorflow import keras

# Import Keras layers
from tensorflow.keras import layers

# Import numpy
import numpy as np

# Import Math plot lib
import matplotlib.pyplot as plt

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

# Get the dataset
mnist = tf.keras.datasets.fashion_mnist

# Load the feature and label, splitted by training_set and test_set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Simple logdump
print('x_train: %s' % x_train)
print('y_train: %s' % y_train)

# Use the matplotlib to sho the first trainingset sample
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Define the model type
model = tf.keras.Sequential()

# Flatten the input as a vector of 28*28 elements
model.add(layers.Flatten(input_shape=(28, 28)))

# Add a dense layer with a relu activation function
model.add(layers.Dense(128, activation='relu'))

# Add a dropout Dropout. It consists in randomly setting a fraction
# rate of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(layers.Dropout(0.2))

# Add the last output layer
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=5)

# Evaluate it
model.evaluate(x_test, y_test, verbose=2)

# Run the prediction on test_set
predictions = model.predict(x_test)

# Dump the predictions
num_bins = 10
plt.hist(predictions[0], num_bins, facecolor='blue', alpha=0.5)
plt.show()
