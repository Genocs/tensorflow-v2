# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:30:00 2020

@author:       Genocs
@description:  This test check-out if tensorflow is installed correctly
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import utility libraries
import os
import pathlib


# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

model_dir = ".\\model\\"
model_weights_path = model_dir + "weights.tf"


if os.path.isdir(model_dir):
  print('Found model\n')
  print('Start conversion process\n')
  converter = tf.lite.TFLiteConverter.from_saved_model(model_weights_path)
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
  print('Conversion completed\n')
else:
  print('Missing Model folder\n')
