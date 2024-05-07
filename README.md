# Advanced Recognition of License Plates (ARLP)

[![GitHub issues](https://img.shields.io/github/issues/PrudhviNallagatla/Automated-Recognition-of-License-Plates-ARLP?color=red&label=Issues&style=flat)](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/issues)
[![GitHub license](https://img.shields.io/github/license/PrudhviNallagatla/Automated-Recognition-of-License-Plates-ARLP?color=green&label=License&style=flat)](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/LICENSE)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

To use ARLP, click on this link and run this Jupyter Book:   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP_Results.ipynb)


## Introduction

In today's digital era, the demand for intelligent systems capable of understanding and processing visual information is ever-increasing. One critical application domain is the recognition and interpretation of vehicle number plates, which are essential for various purposes ranging from law enforcement to traffic management. As part of IEEE NITK, this project focuses on developing a robust Vehicle Number Plate Recognition (NPR) system tailored specifically for Indian vehicles.

We designed a customized CNN model, taking inspiration from various CNN architectures worldwide. Additionally, we implemented transfer learning by incorporating the Inception ResNet V2 model and trained it concurrently. We then evaluated and compared the outcomes of both approaches. We employed TesseractOCR and EasyOCR to extract text from number plates.

## Objective

The primary objective of this project is to design and deploy a specialized Number Plate Recognition system for Indian vehicles. The goal is to create a **Custom CNN model from scratch** by studying various architecture of CNNs and also create a transfer learning model using **Inception Resnet v2** to enable efficient and accurate identification of number plates, considering the unique characteristics and variations of license plates. This NPR system aims to provide a practical solution for automated toll collection, traffic monitoring, and law enforcement applications.

## User Guide

To get started with ARLP, follow these steps:

You can download the books and run them on your local machine. However, you can run the results book on Google Colab.
1. Click on the "Open in Colab" button.
2. Then, run all the cells in Google Colab.
3. Follow the on-screen instructions to upload your own image and get the license plate number.

## Technologies Used

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
- OpenCV
- Tesseract OCR
- EasyOCR
- Inception ResNet v2

### Inception ResNet v2

Inception ResNet v2 is a deep convolutional neural network (CNN) architecture that combines the Inception and ResNet modules. Google introduced it in 2016 and is known for its exceptional performance in image recognition tasks. The architecture employs residual connections to address the vanishing gradient problem and utilizes the Inception modules to capture multi-scale features efficiently.

![Inception ResNet v2](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/Inceptionresnetv2_img.png)
<br>
*Inception ResNet v2's architecture*

Inception-ResNet-v2 is a convolutional neural network trained on over a million images from the ImageNet database. The network is 164 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The Inception-ResNet-v2 was used for the classification task. Inception ResNet v2 consists of multiple blocks of Inception-ResNet modules, each containing a combination of Inception and ResNet components. These modules allow the network to learn complex hierarchical features from input images, making it suitable for image classification, object detection, and feature extraction tasks.

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

#### Spatial Attention:

Spatial attention mechanisms focus on relevant spatial regions within input data. The `SpatialAttention` layer in the code computes attention maps based on input feature maps' spatial information. It generates attention maps highlighting important regions, allowing the network to prioritize relevant areas for improved performance in tasks like image recognition.

#### Channel Attention:

Channel attention mechanisms emphasize or suppress specific channels within feature maps. The `cbam_block` function computes attention weights for each channel using global pooling and dense layers. These weights modulate feature maps, highlighting informative channels. By recalibrating feature representations adaptively, channel attention mechanisms enhance CNNs' ability to capture fine-grained details.

#### CBAM (Convolutional Block Attention Module):

The Convolutional Block Attention Module (CBAM) integrates spatial and channel attention mechanisms. By combining them, CBAM enables networks to focus on relevant spatial regions and informative channels within feature maps. This adaptive recalibration enhances feature representations, improving performance in image classification and object detection tasks, as demonstrated in the provided code.

#### DPN (Dual Path Networks) :

Dual Path Networks (DPNs) integrate dual paths within building blocks. The `dpn_block` function constructs blocks with dense and residual paths. DPNs capture richer features and promote effective information exchange by concatenating information from both paths. This approach improves pattern learning and performance in image classification and semantic segmentation tasks.

#### Squeeze-and-Excitation (SE) :

The Squeeze-and-Excitation (SE) block models channel-wise dependencies within neural networks. Combining squeeze (global pooling) and excitation (fully connected layers) operations, SE blocks adaptively recalibrate feature maps. This enhances feature discrimination and generalization by emphasizing informative channels while suppressing less relevant ones. Integrated into CNNs, SE blocks enhance representational power and performance in image-related tasks.

## Implementation

### Data Preprocessing

First, we collected the dataset from Kaggle, which consists of images of vehicles along with corresponding XML files containing coordinates for bounding boxes around the license plates. We utilized the `glob` and `os` modules in Python to process and extract the required information from the dataset efficiently. Each image was converted into a numpy array for further processing.

The datasets we used are 
- [Labeled licence plates dataset](https://www.kaggle.com/datasets/achrafkhazri/labeled-licence-plates-dataset),
- [Automatic Number Plate Recognition](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection) and
- [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

There are 567 + 453 + 433 = 1453 images in total.

### Custom CNN Approach

We created a Custom model with around 96 million parameters (96,127,521). We then stored it as a .keras file. The total size of the file is 366 MB. We implemented Inception-ResNet Blocks, Attention Mechanisms such as Spatial Attention, Channel Attention, CBAM Blocks and Dual Path Networks (DPNs). We used the Swish activation function instead of regular activation functions (like relu). The model's Architecture graph is in this repo's ARLP Resources folder - [link](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/ARLP_Custom_CNN_architecture.png)

We've achieved an encouraging accuracy of 64.4897% with our model. Although there's room for improvement in tuning the hyperparameters, this presents an exciting opportunity for future refinement and enhancement of our project. With a positive outlook, we're eager to delve into further optimization to unlock the full potential of our model and elevate its performance even further.

#### Inception-ResNet Blocks

We tried imitating the architecture of Inception-ResNet models and tried improving them. So, these blocks are pivotal components in the architecture and are responsible for feature extraction and refinement. They combine Inception and ResNet architecture elements, incorporating convolutional layers with residual connections to facilitate efficient feature propagation and learning.

- The `inception_resnet_block` function defines the structure of each Inception-ResNet block.
- This function applies convolutional operations, including 1x1, 3x3, and concatenation layers, to process the input feature maps.
- These blocks capture complex patterns and enhance feature representations by combining different convolutional pathways and leveraging residual connections.

#### Attention Mechanisms

The architecture integrates advanced attention mechanisms, including Spatial Attention, Channel Attention, and Convolutional Block Attention Module (CBAM), to dynamically adjust feature representations and focus on relevant information.

- The `SpatialAttention` layer computes attention maps based on the spatial information of input feature maps.
- The `cbam_block` function implements channel attention by computing attention weights for each channel using global pooling and dense layers.
- These attention mechanisms selectively emphasize important spatial regions and informative channels within feature maps, enhancing the network's ability to capture relevant details and improve recognition accuracy.

#### Dual Path Networks (DPNs)

Dual Path Networks (DPNs) are incorporated to promote effective information exchange and feature learning within the architecture. They enhance the network's capability to capture richer feature representations, improving recognition accuracy.

- The `dpn_block` function constructs building blocks consisting of dense and residual paths.
- These paths capture different aspects of the input data and are concatenated to facilitate information exchange.
- DPNs promote effective feature learning and enhance the network's discriminative power by combining features from both paths.

#### Swish Activation Function

Throughout the architecture, the Swish activation function introduces non-linearity and facilitates faster convergence during training.

- Swish activation is applied to convolutional layers and other network parts to introduce non-linearities.
- It combines the desirable properties of ReLU with the smoothness of sigmoid activations, promoting faster convergence and improved learning dynamics.

### Inception ResNet v2 Approach

For the Inception ResNet v2 approach, we leveraged transfer learning by importing the pre-trained model from TensorFlow's Keras API. We added two additional layers on top of the pre-trained model to fine-tune it for our specific task. The model was then trained using the collected dataset. `TensorBoard` was used to monitor various metrics during training. Once trained, the model could predict the location of the license plate, i.e., the bounding box coordinates. Subsequently, we cropped the license plate region from the image and applied EasyOCR and Tesseract for text recognition. This is the architecture: [link](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/ARLP_Inception_architecture.png)

## Results

![Comparision between both the models](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/Model_comparision.png)
<br>
*Comparision between both the models*

![Comparision between Cropped images of both the models](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/Cropped_Image_comparision.png)
<br>
*Comparision between Cropped images of both the models*

- As the accuracy of the Custom CNN can be improved further (Accuracy: 64.4897 %), for now, it was unable to detect the precise location of the license plate. However, the Inception model performed well (Accuracy: 94.2857 %).

![Accuracy Vs Epoch Graph of the Custom Model](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/Accuracy%20vs%20epoch%20graph.png)
<br>
*Accuracy Vs Epoch Graph of the Custom Model*

- The steep line in the graph is due to various factors, but we think it's mainly due to the spillage of Registers in local memory -
  `ptxas warning : Registers are spilled to local memory in function 'loop_add_subtract_fusion_49', 224 bytes spill stores, 224 bytes spill loads`.
- We also have to work more on hyperparameter tuning to increase the overall accuracy.

![Accuracy Vs Epoch Graph of the Inception Model](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/ARLP%20Resources/Accuracy%20vs%20epoch%20graph%20-%20inception.png)
<br>
*Accuracy Vs Epoch Graph of the Inception Model*

- The inception model has shown satisfactory results. We will try to improve it further in the future.
- When we compare Tesseract OCR and Easy OCR, the latter works better at recognizing the characters.

## Future Scope

- Our primary focus will be on enhancing the accuracy of the custom model by optimizing its architecture and hyperparameters to their fullest potential.
- We are keen on implementing YOLO (You Only Look Once), a state-of-the-art object detection algorithm, to advance our model's capabilities in accurately detecting and localizing license plates.
- Concurrently, we will dedicate efforts to improve the accuracy of the Inception model through fine-tuning and additional training iterations.
- Expanding our dataset will be a priority, allowing our models to learn from a more diverse and extensive range of samples, thereby enhancing their generalization and robustness in real-world scenarios.

## References

1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [TensorFlow.org Tutorials](https://www.tensorflow.org/tutorials)

Some Reference papers we used are:

3. **Inception ResNet V2**:
   - Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2017). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. arXiv preprint arXiv:1602.07261.
   - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2818-2826.

5. **Attention Mechanisms**:
   - Vaswani, A., et al. (2017). Attention Is All You Need. Proceedings of the 31st Conference on Neural Information Processing Systems (NeurIPS), 5998-6008.
   - Wang, F., Jiang, M., Qian, C., Yang, S., Li, C., Zhang, H., & Wang, X. (2018). Residual Attention Network for Image Classification. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3156-3164.

6. **Dual Path Networks (DPNs)**:
   - Chen, Y., Kalantidis, Y., Li, J., Yan, S., & Feng, J. (2017). Dual Path Networks. Advances in Neural Information Processing Systems (NeurIPS), 4467-4475.

7. **Swish Activation Function**:
   - Ramachandran, P., Zoph, B., & Le, Q. (2017). Searching for Activation Functions. International Conference on Learning Representations (ICLR).

8. **Squeeze-and-Excitation (SE) Block**:
   - Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7132-7141.

## License:

This repository is licensed under the [BSD-3-Clause License](https://github.com/PrudhviNallagatla/Advanced-Recognition-of-License-Plates-ARLP/blob/main/LICENSE)
