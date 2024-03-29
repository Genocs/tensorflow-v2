# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:20:00 2020

@author:       Genocs
@description:  In this exercise we are going to load the trainingset from a csv file
               using Pandas dataframe. The model will be saved for future evaluation.
               Whenever running an instance and a model is detected no training is 
               executed and the existing model is loaded.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras layers
from tensorflow.keras import layers

# Import utility libraries
import os
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
*** The entrypoint function
"""


def main(args=""):
    print("main started")

    samples, labels = load_trainingset()

    # Build the model
    model = build_model()
    model = load_or_instruct_model(model, samples, labels)

    # Evaluate a sample using the model
    print(model.predict([3.0]))

    print("main finish")


"""
*** This function return a set of samples and label
"""


def run_training(model, samples, labels):
    log_path = 'c:\\log\\'

    # Tensorboard trace
    log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        # Write TensorBoard logs to `log_dir` directory
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    with tf.device('/GPU:0'):
        # Instruct the model
        model.fit(samples, labels, callbacks=callbacks, epochs=50)
        return model


"""
*** This is the model that we are using
"""


def build_model():

    # It defines the model type
    model = tf.keras.Sequential()

    # Adds a densely-connected layer with 1 unit
    model.add(layers.Dense(units=1, input_shape=[1]))

    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # print the model structure
    model.summary()

    return model


"""
*** This function load the model if exist, otherwise start a new one trainig and the save the weights
"""


def load_or_instruct_model(model, samples, labels):
    model_dir = ".\\model\\"
    model_weights_path = model_dir + "weights.tf"

    if os.path.isdir(model_dir):
        print('Found model.')
        print('Loading weights ...')
        print('NO Training required')
        model.load_weights(model_weights_path)
    else:
        print('Model do not exit!')
        print('A new one training is required!')
        model = run_training(model, samples, labels)
        print('Model weights SAVING... ')
        model.save_weights(model_weights_path)
        # Save the model using the tensorflow library, 
        # so I can use Tensorflow lite Converter. Check #14-convert-totensorflow-lite.py 
        tf.saved_model.save(model, model_weights_path)
        print('Model weights SAVED SUCCESSFULLY')

    return model


if __name__ == '__main__':
    main()
