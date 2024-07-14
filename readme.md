# Object Detection and Localization with TensorFlow and MobileNetV2

This project demonstrates how to train an object detection model using TensorFlow and MobileNetV2 on the Caltech Birds 2010 dataset. The model detects bounding boxes around bird species in images and evaluates its performance using Intersection over Union (IoU).

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)

<a name="overview"></a>
## Overview

Object detection is an essential task in computer vision, allowing machines to identify and localize multiple objects within an image. This project utilizes MobileNetV2 as a feature extractor and trains a model to predict bounding boxes around bird species in images from the Caltech Birds 2010 dataset. It includes utilities for visualization, model training, evaluation metrics, and more.

<a name="dataset"></a>
## Dataset

The [Caltech Birds 2010 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) contains images of 200 bird species. It is divided into training and test sets, with bounding box annotations for each image.

<a name="model-architecture"></a>
## Model Architecture

The model architecture consists of:

- **Feature Extractor**: MobileNetV2 pre-trained on ImageNet to extract features from input images.
  
- **Dense Layers**: Global Average Pooling and dense layers for feature processing.
  
- **Bounding Box Regression**: Dense layer predicting bounding box coordinates.

<a name="training"></a>
## Training

The model is trained using TensorFlow's SGD optimizer with MSE loss. Training involves iterating over batches of preprocessed images and their corresponding bounding box annotations.
<br>
### Training Sample
![image](https://github.com/user-attachments/assets/2daf7b1e-b2ad-4ce8-926f-386f30f0c3ba)

<a name="evaluation"></a>
## Evaluation

Model performance is evaluated using Intersection over Union (IoU), comparing predicted bounding boxes with ground truth annotations. Loss metrics and validation curves are plotted to assess training progress.
<br>
### Evaluation Sample
![image](https://github.com/user-attachments/assets/aad9a76a-4457-403b-b43e-50b3d5e6dfd4)

<a name="visualization"></a>
## Visualization

Visualization utilities are provided to display images with predicted and ground truth bounding boxes, highlighting IoU scores for each prediction.





