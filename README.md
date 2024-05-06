# Advanced Recognition of License Plates (ARLP)

[![GitHub issues](https://img.shields.io/github/issues/PrudhviNallagatla/Automated-Recognition-of-License-Plates-ARLP?color=red&label=Issues&style=flat)](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/issues)
[![GitHub license](https://img.shields.io/github/license/PrudhviNallagatla/Automated-Recognition-of-License-Plates-ARLP?color=green&label=License&style=flat)](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/LICENSE)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Introduction

In today's digital era, the demand for intelligent systems capable of understanding and processing visual information is ever-increasing. One critical application domain is the recognition and interpretation of vehicle number plates, which are essential for various purposes ranging from law enforcement to traffic management. This Envision project (I03), as part of IEEE NITK (2024), focuses on developing a robust Vehicle Number Plate Recognition (NPR) system tailored specifically for Indian vehicles.

## Objective

The primary objective of this project is to design and deploy a specialized Number Plate Recognition system for Indian vehicles. The goal is to employ deep learning techniques such as **Inception Resnet v2, CNNs, YOLO, UNets, etc.,** to enable efficient and accurate identification of number plates, considering the unique characteristics and variations of Indian license plates. This NPR system aims to provide a practical solution for automated toll collection, traffic monitoring, and law enforcement applications.

## User Guide

To get started with ARLP, follow these steps:

These codes can be run in Google Colab.
1. Click on the "Open in Colab" button.
2. Then, run all the cells in Google Colab.
3. Follow the on-screen instructions to upload your own image and get the license plate number.

If you want to run it on your local machine or anywhere else, follow these steps:
1. Clone the repository to your local machine.
2. Install the prerequisites.
3. Navigate to the project directory.
4. Make necessary changes and run the notebooks to start the ARLP system.
5. Follow the on-screen instructions to upload your own image and get the license plate number.


## Technologies Used

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
- ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
- Tesseract OCR
- EasyOCR
- Inception ResNet v2

### Inception ResNet v2

Inception ResNet v2 is a deep convolutional neural network (CNN) architecture that combines the Inception and ResNet modules. Google introduced it in 2016 and is known for its exceptional performance in image recognition tasks. The architecture employs residual connections to address the vanishing gradient problem and utilizes the Inception modules to capture multi-scale features efficiently.

![Inception ResNet v2](assets/boids.jpg)
<br>
*Inception ResNet v2's architecture*

Inception-ResNet-v2 is a convolutional neural network trained on over a million images from the ImageNet database. The network is 164 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The Inception-ResNet-v2 was used for the classification task. Inception ResNet v2 consists of multiple blocks of Inception-ResNet modules, each containing a combination of Inception and ResNet components. These modules allow the network to learn complex hierarchical features from input images, making it suitable for image classification, object detection, and feature extraction tasks.

```python
# Example code to load Inception ResNet v2 model using TensorFlow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2

# Load pre-trained Inception ResNet v2 model
model = InceptionResNetV2(weights='imagenet', include_top=True)
```

The `InceptionResNetV2` class provided by TensorFlow's Keras API allows you to easily load the pre-trained Inception ResNet v2 model trained on the ImageNet dataset. This model can be fine-tuned or used as a feature extractor for various computer vision tasks, including number plate recognition.


## Literature Survey

#### Python Programming Language

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

#### Machine Learning

Machine learning is a branch of artificial intelligence (AI) that focuses on developing algorithms and techniques that enable computers to learn from data and make predictions or decisions without being explicitly programmed. It encompasses various approaches, such as supervised, unsupervised, and reinforcement learning.

#### Neural Networks

Neural networks are computational models inspired by the structure and function of biological neural networks in the human brain. They consist of interconnected nodes, called neurons, organized in layers. Each neuron applies a weighted sum of its inputs and an activation function to produce an output. Neural networks can learn complex patterns and relationships from data, making them suitable for various tasks, including image recognition, natural language processing, and reinforcement learning.

#### TensorFlow

TensorFlow is an open-source machine learning framework developed by Google Brain for building and training deep learning models. It provides a comprehensive ecosystem of tools, libraries, and resources for efficiently developing and deploying machine learning applications. TensorFlow supports high-level APIs for easy model development and low-level APIs for flexibility and customization.

#### Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep neural networks designed for processing structured grid-like data, such as images. They consist of multiple layers, including convolutional layers, pooling layers, and fully connected layers. CNNs are highly effective for tasks such as image classification, object detection, and image segmentation due to their ability to learn hierarchical representations of features from input images automatically.

#### YOLO (You Only Look Once)

YOLO is a state-of-the-art object detection algorithm that operates with high real-time accuracy. Unlike traditional object detection methods that use sliding windows and region proposal techniques, YOLO frames object detection as a regression problem to spatially separated bounding boxes and associated class probabilities directly from full images in a single pass through the network. YOLO achieves high detection accuracy and efficiency, making it popular for various real-time applications.

## Implementation

#### Data Preprocessing

First, we collected the dataset from Kaggle, which consists of images of vehicles along with corresponding XML files containing coordinates for bounding boxes around the license plates. We utilized the `glob` and `os` modules in Python to process and extract the required information from the dataset efficiently. Each image was converted into a numpy array for further processing.

The datasets we used are 
- [Labeled licence plates dataset](https://www.kaggle.com/datasets/achrafkhazri/labeled-licence-plates-dataset),
- [Automatic Number Plate Recognition](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection) and
- [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

There are 567 + 453 + 433 = 1453 images in total.

![Boids Algorithm](assets/boids.jpg)
<br>
*Boids Algorithm*

#### Inception ResNet v2 Approach

For the Inception ResNet v2 approach, we leveraged transfer learning by importing the pre-trained model from TensorFlow's Keras API. We added two additional layers on top of the pre-trained model to fine-tune it for our specific task. The model was then trained using the collected dataset. `TensorBoard` was used to monitor various metrics during training. Once trained, the model could predict the location of the license plate, i.e., the bounding box coordinates. Subsequently, we cropped the license plate region from the image and applied EasyOCR and Tesseract for text recognition.

![Boids Algorithm](assets/boids.jpg)
<br>
*Boids Algorithm*

#### CNN Approach

In the CNN approach, instead of importing a pre-trained model like Inception ResNet v2, we constructed our own CNN architecture. This involved hardcoding the layers of the CNN model to suit our requirements. The model was trained similarly using the collected dataset, and the license plate region was cropped and processed for text recognition. `TensorBoard` was used to monitor various metrics during training.

![Boids Algorithm](assets/boids.jpg)
<br>
*Boids Algorithm*

#### YOLO Approach

Similarly, for the YOLO approach, we developed our YOLO model architecture. YOLO operates differently from traditional CNNs as it directly predicts bounding boxes and class probabilities from full images in a single pass. Using our dataset, we trained the YOLO model and performed text recognition on the detected license plate regions. `TensorBoard` was used to monitor various metrics during training.

By employing these three approaches, we aimed to compare their performances and determine the most suitable method for our Vehicle Number Plate Recognition system tailored for Indian vehicles.

![Boids Algorithm](assets/boids.jpg)
<br>
*Boids Algorithm*

## Results

These are the results we got.
....................................

## References

1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [Deep Learning with TensorFlow 2 and Keras: Regression, ConvNets, GANs, RNNs, NLP, and more with TensorFlow 2 and the Keras API, 2nd Edition](https://www.amazon.com/Deep-Learning-TensorFlow-Keras-Regression/dp/1800208616)
3. [TensorFlow.org Tutorials](https://www.tensorflow.org/tutorials)
4. [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
5. [Inception-ResNet-v2: The 2017 winner of ILSVRC image classification](https://arxiv.org/abs/1602.07261)
6. [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## License:

This repository is licensed under the [BSD-3-Clause License](..................)
