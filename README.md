# tensorflow-v2

This repo is a Tensorflow course starting from scratch. The ***inspiration*** coming from awesome guys to whom I say thanks a lot.
> - [AntonMu](https://github.com/AntonMu/TrainYourOwnYOLO)
> - [Aymeric Damien](https://github.com/aymericdamien)
> - [YunYang1994](https://github.com/YunYang1994)
> - [zzh8829](https://github.com/zzh8829)
> - [Anton Mu](https://github.com/AntonMu)



## Prerequisites

Python environment is required and correctly configured in order o use these exercises.

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
- **06-load-traininset.py**: Load the trainingset data from csv file.
- **07-save-weights.py**: How to save the training weights and load it when available.
- **08-use-model-class.py**: Complete example to training, save the weights and load it if present.

#### 3 - Standard Datasets
- **09-mnist-dataset.py**: The helloworld MINIST dataset.
- **10-cifar-dataset.py**: The CIFAR-10 dataset.

#### 4 - Images Datasets
- **11-images-dataset.py**: Run CNN on images classified by folder name.
- **12-images-dataset-cache.py**: Same as the above test!!!
- **13-images-dataset-local.py**: Same as the above but from local folder
- **14-convert-to-tensorflow-lite.py**: Convert the model to tensorflow lite model available to be used on mobile device



## How to install from scratch

Nvidia GPU support
CUDA Toolkit 11.6 Update 1 Downloads cuda_11.6.0_511.23_windows
Download cuDNN v8.3.2 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2 cudnn-windows-x86_64-8.3.2.44_cuda11.5-archive

python-3.9.7.exe (DO not use python-3.8.5.exe)

- Check Nvidia cuda version
``` sh
nvcc --version
```
Checked version:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Fri_Dec_17_18:28:54_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.6, V11.6.55
Build cuda_11.6.r11.6/compiler.30794723_0


## Preliminary checks
``` sh
python --version
conda --version
pip3 --version

# Checked version:
Python 3.9 (3.9.7)
conda 4.10.3
pip 22.0.3
```

- Create a log folder like **c:\log**

- Create the conda environment
``` sh
conda create --name tensorflow python=3.9
conda activate tensorflow
conda install git
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Run tensorboard
``` sh
(tensorflow) tensorboard --logdir=c:\log --port 6006
```

Open a web browser
http://localhost:6006


# Tips and Tricks
Run following commands before run jupyter at 
https://github.com/tensorflow/models

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

``` sh
conda install git
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

# How to use AWS EC2 intances

Connect to AWS
Select the environment 
Update the system and clone the repo 
``` sh
sudo apt-get update
git clone https://github.com/AntonMu/TrainYourOwnYOLO.git
```

