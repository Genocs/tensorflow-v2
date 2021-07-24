# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:20:00 2020

@author:       Genocs
@description:  In this exercise we are going to load the trainingset from a csv file
               using Pandas dataframe
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras layers
from tensorflow.keras import layers

# Import utility libraries
import datetime

# Import pandas to read csv files
import pandas as pd

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)
"""
*** This function return a set of samples and label
"""


def load_trainingset():
    # Load the data using pandas
    samples = pd.read_csv(".\\data\\samples.csv", delimiter=',', header=None)
    labels = pd.read_csv(".\\data\\labels.csv", delimiter=',', header=None)
    samples = samples.to_numpy()
    labels = labels.to_numpy()
    print(samples[0])
    print(labels[0])
    return samples[0], labels[0]


"""
*** This function run the model
"""


def training_the_model():
    # Load the trainingset from csv file
    model_data, model_value = load_trainingset()
    log_path = 'c:\\log\\'

    # Tensorboard trace
    log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        # Write TensorBoard logs to `log_dir` directory
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    with tf.device('/GPU:0'):
        # Define the model type
        model = tf.keras.Sequential()

        # Adds a densely-connected layer with 1 unit
        model.add(layers.Dense(units=1, input_shape=[1]))

        # Build the model pipeline
        model.compile(optimizer='sgd',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        # Display the model structure
        model.summary()

        # Instruct the code
        model.fit(model_data, model_value, callbacks=callbacks, epochs=50)

        # Evaluate a sample using the model
        print('The predicted result is: %f' % model.predict([70.0]))


"""
*** The entrypoint function
"""


def main():
    print("main started")
    training_the_model()
    print("main finish")


if __name__ == '__main__':
    main()
