# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:13:49 2020

@author: Giovanni
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf
# Import Keras
from tensorflow import keras
# Import Keras layers
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Check the Tensorflow version (should be 2.1.0)
print('Tensorflow version: %s' % tf.__version__)

# Check the GPU device availability
print('Num GPUs Available: %d' %
      len(tf.config.experimental.list_physical_devices('GPU')))


mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print('x_train: %s' % x_train)
print('y_train: %s' % y_train)

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


model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

predictions = model.predict(x_test)

print(predictions[0])
print(predictions[1])
