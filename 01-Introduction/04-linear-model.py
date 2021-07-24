# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:00:00 2020

@author:       Genocs
@description:  Simple linear model y = mx + b
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras layers
from tensorflow.keras import layers

# Import numpy
import numpy as np

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

# Define the feature and the label
model_data = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
model_value = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                       dtype=float)

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise
noise = np.random.normal(0, 1, 100)

with tf.device('/GPU:0'):

    # Define the model type
    model = tf.keras.Sequential()

    # Add a densely-connected layer with 1 unit
    model.add(layers.Dense(units=1, input_shape=[1]))

    # Build the model pipeline
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Display the model structure
    model.summary()                  

    # Instruct the code
    model.fit(model_data, model_value, epochs=50)

    # Evaluate a sample using the model
    print('The predicted result is: %f' % model.predict([70.0]))
