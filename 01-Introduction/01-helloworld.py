# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:30:00 2020

@author:       Genocs
@description:  This test check if tensorflow is installed correctly
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)
