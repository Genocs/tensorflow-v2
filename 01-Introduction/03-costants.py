# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:40:00 2020

@author:       Genocs
@description:  Simple contants tensor and basic matrix multiply
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Check the Tensorflow version
print('Tensorflow version: %s' % tf.__version__)

# Create a tensor as a vector (one dimension)
vector_a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(vector_a)

# Create a tensor as a matrix (two dimensions) layout: 3x2
matrix_b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(matrix_b)

# Create two matrices
m_a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
m_b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Matrix multiply (m_b * m_a)
m_ba = tf.matmul(m_b, m_a)
print(m_ba)

# Matrix multiply (m_a * m_b) different from the previous ones
m_ab = tf.matmul(m_a, m_b)
print(m_ab)
