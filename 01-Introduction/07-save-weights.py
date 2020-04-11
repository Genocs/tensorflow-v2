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

# Import Keras
from tensorflow import keras

# Import Keras layers
from tensorflow.keras import layers

# Import numpy
import numpy as np

# Import pandas to read csv files
import pandas as pd

# Import utility libraries
import os
import pathlib
import datetime

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
    return (samples[0], labels[0])


"""
*** The entrypoint function
"""
def main(args=""):
    print("main started")
    
    samples, labels = load_trainingset()

    # Build the model
    model = build_model()
    model = instruct_model(model, samples, labels)

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
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
    ]

    with tf.device('/GPU:0'):    
       # Instruct the model
        model.fit(samples, labels, callbacks=callbacks, epochs=5)
        return model


"""
*** This is the model that we are using
"""
def build_model():

    # It defines the model type
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 1 unit to the model:
    model.add(layers.Dense(units=1, input_shape=[1]))

    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # print the model structure
    model.summary()

    return model

"""
*** This function loadad the model if exist, otherwise start a new one trainig and the save the weights
"""
def instruct_model(model, samples, labels):
    model_dir = ".\\model\\"

    if os.path.isdir(model_dir):
        print('Found model.')
        print('Loading weights ...')
        print('NO Training required')
        model.load_weights(model_dir + "model.tf")
    else:
        print('Model do not exit!')
        print('A new one training is required!')
        model = run_training(model, samples, labels)
        print('Model weights SAVING... ')
        model.save_weights(model_dir + "model.tf")
        print('Model weights SAVED SUCCESSFULLY')

    return model


if __name__ == '__main__':
    main()
