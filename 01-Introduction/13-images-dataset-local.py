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
               - OpenCV
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

# Import pandas to read csv files
import pandas as pd

import IPython.display as display
from PIL import Image
import os
import pathlib
import datetime

# Import python openCV
import cv2


# Plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
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
    model.add(
        layers.Conv2D(32, (3, 3),
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

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Print the model structure
    model.summary()

    return model


def run_training(model, samples, labels):
    log_path = 'c:\\log\\'

    # Tensorboard trace
    log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    # Instruct the model
    history = model.fit(samples,
                        labels,
                        epochs=25,
                        callbacks=[tensorboard_callback])

    return model


# Check if exists the model weights, based on the directory where the weights are stored
def exist_weights():
    model_dir = ".\\model\\"

    return os.path.isdir(model_dir)


# Load the trainingset from local filesystem
def get_traingset(data_path):

    # Load image from local file system
    data_dir = pathlib.Path(data_path)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))

    # Simple log
    print('data_dir: %s' % data_dir)
    print('image_count: %d' % image_count)

    # Get the class name based on the trainingset folder
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

    # Print the class names
    print('CLASS_NAMES: %s' % CLASS_NAMES)

    BATCH_SIZE = 256
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    # Prepare the image dataset generator
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    train_data_gen = image_generator.flow_from_directory(
        directory=str(data_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes=list(CLASS_NAMES))

    train_images, train_labels = next(train_data_gen)

    # Remoge comment to see some TS images
    #plotImages(train_images[:5])

    # Convert the one shot label vector to classes vectors
    # PLEASE find a more elegant way
    vector_classes = np.argmax(train_labels, axis=1)

    vector_classes_list = []
    for v in vector_classes:
        vector_classes_list.append([v])

    training_label_classes = np.asarray(vector_classes_list, dtype=np.float32)

    print('CLASS_NAMES:')
    print(training_label_classes)

    return train_images, training_label_classes, CLASS_NAMES


# Load the image to inspect from local filesystem
def get_images(data_path):
    # Load image from local file system
    data_dir = pathlib.Path(data_path)
    data_dir = pathlib.Path(data_dir)

    BATCH_SIZE = 256
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    # Prepare the image dataset generator
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    train_data_gen = image_generator.flow_from_directory(
        directory=str(data_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH))

    images, labels = next(train_data_gen)

    return images, labels


def load_image(data_path):
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    # Load image by OpenCV
    img = cv2.imread(data_path)

    # Resize to respect the input_shape
    inp = cv2.resize(img, (IMG_HEIGHT, IMG_HEIGHT))

    # Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Now you can use rgb_tensor to predict label for exemple :

    #P redict label
    return rgb_tensor


def save_result(predictions):
    result_filename = '.\\result.csv'

    CLASS_NAMES = np.array(
        ['globalblue_f', 'globalblue_m', 'globalblue_t', 'others', 'planet'])

    df = pd.DataFrame(columns=['Prediction', 'Scores'])

    if (os.path.exists(result_filename)):
        df = pd.read_csv(result_filename)

    df = df.append(
        {
            'Prediction': CLASS_NAMES[np.argmax(predictions[0])],
            'Scores': predictions[0]
        },
        ignore_index=True)

    df.to_csv(result_filename, index=False)
    print('predict[0]: %s' % CLASS_NAMES[np.argmax(predictions[0])])
    print(predictions)


def scan_local(data_path, log_path='c:/log/'):
    model_weights_path = ".\\model\\weights.tf"

    writer = tf.summary.create_file_writer(log_path)

    with writer.as_default():
        tf.summary.scalar("my_metric", 0.5, step=22)
        writer.flush()

    # Build the model graph
    model = build_model()

    # Check if the weights exist and if dont than go to training the model and save it
    if (exist_weights()):
        print('Found model weights, NO Training required. Loading...')
        model.load_weights(model_weights_path)
        print(
            'Model weights LOADED SUCCESSFULLY...  go straight to evaluate!!')
    else:
        train_images, _labels, _CLASS_NAMES = get_traingset(data_path)
        model = run_training(model, train_images, _labels)
        print('Model weights SAVING... ')
        model.save_weights(model_weights_path)
        # Save the model using the tensorflow library, 
        # so I can use Tensorflow lite Converter. Check the next step
        tf.saved_model.save(model, model_weights_path)
        print('Model weights SAVED SUCCESSFULLY')

    CLASS_NAMES = np.array(
        ['globalblue_f', 'globalblue_m', 'globalblue_t', 'others', 'planet'])

    evaluate_images = load_image('E:\\Data\\UTU\\evaluate\\eval\\1.jpg')
    predictions = model.predict(evaluate_images)

    # Save prediction on file
    save_result(predictions)


def main():
    scan_local(data_path='E:\\Data\\UTU\\directinvoice_img')


if __name__ == '__main__':
    main()
