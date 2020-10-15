# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:30:00 2020

@author:       Genocs
@description:  This test check-out if tensorflow is installed correctly
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

model_weights_path = ".\\model\\weights.tf"

converter = tf.lite.TFLiteConverter.from_saved_model(model_weights_path)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)