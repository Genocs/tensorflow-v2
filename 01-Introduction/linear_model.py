# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:30:00 2020

@author:       Genocs
@description:  Simple linear model y = mx + b. The pipeline logs are written using tensorflow 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras
from tensorflow import keras

# Import Keras layers
from tensorflow.keras import layers

# Import utility libraries
import os
import datetime

"""
This class defines a very simple linear model
"""


class LinearModel(keras.Model):

    def __init__(self, name=None):
        super(LinearModel, self).__init__(name=name)
        self.dense_1 = layers.Dense(units=1, input_shape=[1], name='dense_1')

    def call(self, inputs):
        # x = self.dense_1(inputs)
        # x = self.dense_2(x)
        # return self.pred_layer(x)
        return self.dense_1(inputs)

    def compile_model(self):
        self.compile(optimizer='sgd',
                     loss='mean_squared_error',
                     metrics=['accuracy'])
        return self

    def training(self, model_data, model_value):
        model_dir = ".\\model\\"

        if os.path.isdir(model_dir):
            print('Found model.')
            print('Loading weights ...')
            print('NO Training required')
            self.load_weights(model_dir + "model.tf")
        else:
            print('Model do not exit!')
            print('A new one training is required!')
            log_path = 'c:\\log\\'
            # Tensorboard trace
            log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks = [
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                tf.keras.callbacks.EarlyStopping(
                    patience=2, monitor='accuracy'),
                # Write TensorBoard logs to `./logs` directory
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir, histogram_freq=1)
            ]

            with tf.device('/GPU:0'):
                # Instruct the model
                self.fit(model_data, model_value, callbacks=callbacks, epochs=5)
                self.summary()

            print('Model weights SAVING... ')
            self.save_weights(model_dir + "model.tf")
            print('Model weights SAVED SUCCESSFULLY')

        return self
