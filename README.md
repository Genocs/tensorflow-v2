# tensorflow-v2

This repo is a Tensorflow course starting from scratch. The ***inspiration*** coming from awesome guys to whom I say thanks a lot.
> - [Aymeric Damien](https://github.com/aymericdamien)
> - [YunYang1994](https://github.com/YunYang1994)
> - [zzh8829](https://github.com/zzh8829)


## Prerequisites

Python environment must be present and correctly configured in order o use these exercises.

Please check:
* [Anaconda](https://www.continuum.io).
* [Nvidia Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
* [Google TensorFlow 2.0](https://www.tensorflow.org/)
* [Keras](https://keras.io/)


## Guides
[tflearn](http://tflearn.org)


## Example and Tutorials

* Git Tensorflow
> - [Aymeric Damien](https://github.com/aymericdamien)
> - [YunYang1994 Examples](https://github.com/YunYang1994/TensorFlow2.0-Examples)


* Tensorflow Yolo v3
> - [zzh8829](https://github.com/zzh8829/yolov3-tf2)
> - [YunYang1994 yolov3](https://github.com/YunYang1994/tensorflow-yolov3)


## Contents

#### 1 - Introduction
- **01-helloworld.py**: It is a simple check to evaluate if Tensorflow is correctly installed.
- **02-helloworld-gpu.py**: Same as the item above but checking the GPU support (Nvidia Cuda GPU is required).
- **03-constants.py**: Basic example about Tensorflow tensors. Simple Linear Algebra ops are executed as well.

#### 2 - Keras - The models
- **04-linear-model.py**: The basic linear model.
- **05-tensorboard.py**: How write trace to Tensorboard.
- **06-linear-model-class.py**: The same linear model with Tensorboard but embedded in a Python class.
- **07-save-weights.py**: How to save the training weights and load it when available.
- **08-use-model-class.py**: Complete example to training, save the weights and load it if present.

#### 3 - Standard Datasets
- **09-mnist-dataset.py**: The helloworld MINIST dataset.
- **10-cifar-dataset.py**: The CIFAR-10 dataset.



## Miscellaneous

#### How Activate Tensorboard

Create a log folder like **c:\log**

Start console prompt and activate the correct conda environment

``` cmd
conda activate tensorflow
(tensorflow) tensorboard --logdir=c:\log --port 6006
```

Open a web browser

localhost:6006

# Installed library
- pytesseract https://anaconda.org/auto/pytesseract

pip install pytesseract



# Tips and Tricks
Run following commands before run jupyter at 
https://github.com/tensorflow/models

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

``` cmd
conda install git
then:
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```