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

# Import utility libraries
import os
import pathlib
import datetime

# Import pandas to read csv files
import pandas as pd

from linear_model import LinearModel

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
    return (samples[0], labels[0])


"""
*** The entrypoint function
"""
def main(args=""):
    print("main started")
    
    samples, labels = load_trainingset()

    # Build a model using a class, and after the compiling start the training
    model = LinearModel("my_model")
    model.compile_model()
    model.training(samples, labels)

    # Evaluate a sample using the model
    print(model.predict([3.0]))

    print("main finish")


if __name__ == '__main__':
    main()
