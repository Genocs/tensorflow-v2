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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
import glob
import shutil
import pathlib
import datetime

# Import python openCV
import cv2

CLASS_NAMES = ['globalblue_f', 'globalblue_m',
               'globalblue_t', 'others', 'planet']

IMG_SIZE = 128
BATCH_SIZE = 256


# Plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def build_model():
    # Model / data parameters
    num_classes = 5
    img_shape = (IMG_SIZE, IMG_SIZE, 3)  # (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

    # Define the model type
    model = tf.keras.Sequential()

    # Define the graph
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                            input_shape=img_shape, data_format="channels_last"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(num_classes))

    model.add(layers.Dense(num_classes, activation='softmax', name='last_dense'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # Print the model structure
    model.summary()

    return model


def run_training(model, train_data, val_data, labels):
    log_path = 'c:\\log\\'
    epochs = 10

    # Tensorboard trace
    log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    # Instruct the model
    history = model.fit(train_data,
                        labels,
                        epochs=epochs,
                        validation_data=val_data,
                        callbacks=[tensorboard_callback])

    plot_history(history, epochs)

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

    # Get the class name based on the trainingset folder
    classes = np.array([item.name for item in data_dir.glob('*')])

    # Simple log
    print('data_dir: %s' % data_dir)
    print('image_count: %d' % image_count)
    print('dir_classes: %s' % classes)

    # Prepare the image dataset generator
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    train_data_gen = image_gen_train.flow_from_directory(directory=os.path.join(data_dir, 'train'),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_SIZE, IMG_SIZE),
                                                         class_mode='sparse')

    train_images, train_labels = next(train_data_gen)

    # Remove comment to see some TS images
    # plotImages(train_images[:5])

    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=os.path.join(data_dir, 'val'),
                                                     target_size=(IMG_SIZE, IMG_SIZE),
                                                     class_mode='sparse')

    return train_images, val_data_gen, train_labels


# Load the image to inspect from local filesystem
def get_images(data_path):
    # Load image from local file system
    data_dir = pathlib.Path(data_path)
    data_dir = pathlib.Path(data_dir)

    # Prepare the image dataset generator
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    train_data_gen = image_generator.flow_from_directory(
        directory=str(data_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_SIZE, IMG_SIZE))

    images, labels = next(train_data_gen)

    return images, labels


def load_image(data_path):
    # Load image by OpenCV
    img = cv2.imread(data_path)

    # Resize to respect the input_shape
    inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Now you can use rgb_tensor to predict label for exemple :

    # P redict label
    return rgb_tensor


def save_result(predictions):
    result_filename = '.\\result.csv'

    df = pd.DataFrame(columns=['Prediction', 'Scores'])

    if os.path.exists(result_filename):
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


def create_dataset(base_dir):
    # Check if exist the training folder
    if os.path.exists(os.path.join(base_dir, 'train')):
        return

    # The class list
    classes = ['globalblue_f', 'globalblue_m',
               'globalblue_t', 'others', 'planet']

    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        num_train = int(round(len(images) * 0.8))
        train, val = images[:num_train], images[num_train:]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))


def scan_local(data_path, log_path='c:/log/'):
    model_weights_path = ".\\model\\weights.tf"

    writer = tf.summary.create_file_writer(log_path)

    with writer.as_default():
        tf.summary.scalar("my_metric", 0.5, step=22)
        writer.flush()

    # Build the model graph
    model = build_model()

    # Check if the weights exist and if don't than go to training the model and save it
    if exist_weights():
        print('Found model weights, NO Training required. Loading...')
        model.load_weights(model_weights_path)
        print(
            'Model weights LOADED SUCCESSFULLY...  go straight to evaluate!!')
    else:
        train_images, val_data, labels = get_traingset(data_path)
        model = run_training(model, train_images, val_data, labels)
        print('Model weights SAVING... ')
        model.save_weights(model_weights_path)
        # Save the model using the tensorflow library,
        # so I can use Tensorflow lite Converter. Check the next step
        tf.saved_model.save(model, model_weights_path)
        print('Model weights SAVED SUCCESSFULLY')

    evaluate_images = load_image(
        'E:\\Data\\UTU\\evaluate\\eval\\1.jpg')
    predictions = model.predict(evaluate_images)

    # Save prediction on file
    save_result(predictions)


def plot_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def main():
    # create_dataset(base_dir='E:\\Data\\UTU\\directinvoice_img')
    scan_local(data_path='E:\\Data\\UTU\\directinvoice_img')


if __name__ == '__main__':
    main()
