# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:40:00 2020

@author:       Genocs
@description:  In this exercise we are going to use images dataset
               stored in a local drive
               - Check following packages
			   - TensorFlow
               - Pillow
               - matplotlib
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Import Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models

# Import numpy
import numpy as np

# Import Math plot lib
import matplotlib.pyplot as plt

import IPython.display as display
from PIL import Image
import os
import pathlib
import datetime

# Import python openCV
import cv2

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def build_model():

    CHANNELS = 3
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    # Define the  model type
    model = tf.keras.Sequential()

    # Define the graph 
    model.add(layers.Conv2D(32, (3, 3),
                activation='relu',
                input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS),
                data_format="channels_last"))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5))

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    # print the model structure
    model.summary()

    return model

def run_training(model, samples, labels):
    log_path = 'c:\\log\\'

    # tensorboard trace
    log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    # Instruct the model
    history = model.fit(samples,
                        labels,
                        epochs=25,
                        callbacks=[tensorboard_callback])
    model.summary()

    return model 


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



def scan_local(data_path, log_path='c:\\log\\'):
    writer = tf.summary.create_file_writer(log_path)

    with writer.as_default():
        tf.summary.scalar("my_metric", 0.5, step=22)
        writer.flush()

    # Load image from local file system
    data_dir = pathlib.Path(data_path)

    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))

    print('image_count: %d' % image_count)
    print('data_dir: %s' % data_dir)

    CLASS_NAMES = np.array([
        item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"
    ])
    print('CLASS_NAMES: %s' % CLASS_NAMES)

    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    BATCH_SIZE = 256
    CHANNELS = 3
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    train_data_gen = image_generator.flow_from_directory(
                                                        directory=str(data_dir),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes=list(CLASS_NAMES))

    train_images, train_labels = next(train_data_gen)

    #plotImages(train_images[:5])

    # Convert the oneshot label vector to classes vectors
    # PLEASE find a more elegant way
    vector_classes = np.argmax(train_labels, axis=1)

    vector_classes_list = []
    for v in vector_classes:
        vector_classes_list.append([v])

    training_label_classes = np.asarray(vector_classes_list, dtype=np.float32)

    model = build_model()
    model = instruct_model(model, train_images, training_label_classes)

    # Evaluate a sample using the model
    predictions = model.predict(train_images[0:2])

    print('predict[0]: %s' % CLASS_NAMES[np.argmax(predictions[0])])
    print('predict[1]: %s' % CLASS_NAMES[np.argmax(predictions[1])])
    print('labels: %s' % training_label_classes[0:2])

    return model


def main():
    #weightfile = "weights/utu_direct.weights"
    #cfgfile = "cfg/utu_direct.cfg"
    #model_size = (416, 416, 3)
    #num_classes = 5
    model = scan_local(data_path='E:/Data/UTU/directinvoice_img')
    # evaluate model
    #print('predict: %s' % model.predict(train_images[0:2]))
    #print('labels: %s' % training_label_classes[0:2])

    #img_dir = 'E:\\Data\\UTU\\directinvoice_img\\globalblue_f\\12083.027006.5344.44365.jpg'
    #image = cv2.imread(img_dir)
    #image = np.array(image)
    #model.predict(image)
    #plt.figure()
    #plt.imshow(image)
    #plt.colorbar()
    #plt.grid(False)
    #plt.grid(False)
    #plt.show()

if __name__ == '__main__':
    main()
