# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:50:41 2020

@author: Giovanni
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Check following packages
# TensorFlow
# Pillow
# matplotlib

# Import TensorFlow

import tensorflow as tf


import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import time

# Check the Tensorflow version (should be 2.1.0)
print('*****\nTensorflow version: %s \n*****' % tf.__version__)

default_timeit_steps = 1000

# Load image from local file system 
data_dir = pathlib.Path('E:/Data/UTU/directinvoice_img')

image_count = len(list(data_dir.glob('*/*.jpg')))
print("data_dir: %s" % data_dir)
print("image_count: %d" % image_count)


class_names = np.array([item.name for item in data_dir.glob('*')])
print("class names: %s" % class_names) 

    
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32
IMG_HEIGHT = 400
IMG_WIDTH = 540
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(class_names))



def show_batch(image_batch, label_batch):
  plt.figure(figsize=(40,55))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(class_names[label_batch[n]==1][0].title())
      plt.axis('off')
      
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == class_names  

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
      
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds  

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
  

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(4):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())

