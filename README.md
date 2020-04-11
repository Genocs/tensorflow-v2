# tensorflow-v2

This repo is a tensorflow course starting from scratch.

Python and Machine Learning
The aims of the repo is to train myself in order to use python as the prototyping language implementing solution using Machine Learning and Deep Learning algorithms. The applied fields are NLP (Natural Language Processing) and Image Processing.

## Prerequisites

Python environment must be present and correctly configured in order o use these exercises.


Please check:
* [Anaconda](https://www.continuum.io).
* [Nvidia Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
* [Google TensorFlow 2.0](https://www.tensorflow.org/)
* [Keras](https://keras.io/)


## Example and Tutorials

* Git [Aymeric Damien](https://github.com/aymericdamien)


## How Activate Tensorboard

Create a log folder like **c:\log**

Start console prompt and activate the correct conda environment

``` cmd
conda activate tensorflow
(tensorflow) tensorboard --logdir=c:\log --port 6006
```

Open a web browser

localhost:6006

## The exercises
Introduction
- **01-helloworld.py**: It is a simple check to evaluate if Tensorflow is correctly installed.
- **02-helloworld-gpu.py**: It is a simple check to evaluate if Tensorflow could be executed on GPU (Nvidia Cuda ready GPU is required).
- **03-constants.py**: It explain how Tensorflow manages the the tensors. A matmul example is executed as well.
The models
- **04-linear-model.py**: The first linear model.
- **05-tensorboard.py**: The first linear model with Tensorboard.
- **06-linear-model-class.py**: The first linear model with Tensorboard embedded in a Python class.
