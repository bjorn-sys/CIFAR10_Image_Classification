# CIFAR10_Image_Classification
---
The CIFAR-10 dataset is a collection of 60,000 small color images, each 32x32 pixels in size. It’s used to train and test image classification models.  There are 10 classes of objects: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.  Out of the 60,000 images, 50,000 are for training and 10,000 are for testing.

***📋 Project Overview***
---
* This project implements and compares multiple neural network architectures for classifying images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes.

***Models Implemented:***
---
* Simple MLP (1 hidden layer)

* Deep MLP with ReLU (5 hidden layers)

* Deep MLP with Tanh (5 hidden layers)

* AlexNetMiniLite (CNN with BatchNorm and Dropout)

> **Quick Start**
**Prerequisites**
---

* pip install torch torchvision matplotlib scikit-learn
* 
***Dataset Information***
  
***CIFAR-10 Specifications:***
---

* Total Images: 60,000

* Training Set: 50,000 images

* Test Set: 10,000 images

* Image Size: 32×32 pixels RGB

* Classes: 10 categories including airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

* Class Distribution: Perfectly balanced (5,000 images per class in training)

***Model Architectures***
---
**1. Simple MLP**
---

* Architecture: Linear(3072 → 128) → ReLU → Linear(128 → 10)

* Parameters: ~400,000

* Test Accuracy: 47.4%

**2. Deep MLP with ReLU**
---

* Architecture: 5 hidden layers with ReLU activation

* Parameters: ~460,000

* Test Accuracy: 48.3%

**3. Deep MLP with Tanh**
---

* Architecture: 5 hidden layers with Tanh activation

* Parameters: ~460,000

* Test Accuracy: 40.0%

**4. AlexNetMiniLite (CNN)**
---

* Architecture: 3 convolutional blocks + 2 fully connected layers

* Features: Batch Normalization, Dropout, Data Augmentation

* Test Accuracy: 84.97%

**Performance Comparison**
---

* Model	Test Accuracy	Key Characteristics
* Simple MLP	47.4%	Basic architecture, limited capacity
* Deep MLP (ReLU)	48.3%	More layers, better than Tanh
* Deep MLP (Tanh)	40.0%	Suffers from vanishing gradients
* AlexNetMiniLite	84.97%	Best performer with CNN advantages
* Technical Implementation
* Data Preprocessing
* Normalization: Pixel values scaled from [0,255] to [0,1]

* Data Augmentation: Random cropping, horizontal flipping, color jittering

* Train/Val Split: 45,000 training + 5,000 validation images

**Training Configuration**
---

* Optimizer: Adam with learning rate 0.001

* Loss Function: CrossEntropyLoss

* Batch Size: 64

* Epochs: 50-100 depending on model

* Learning Rate Scheduler: ReduceLROnPlateau

**Key Findings**
---
**MLP Limitations:**

* Destroy spatial relationships by flattening images

* Require massive parameters for basic functionality

* Lack translation invariance

* Struggle with generalization on image data

**CNN Advantages:**
---

* Spatial hierarchy through convolutional layers

* Parameter sharing reduces model complexity

* Translation invariance for better generalization

* Feature hierarchy from low to high-level patterns

**Insights & Recommendations**
---

* Architecture Matters: CNNs significantly outperform MLPs for image classification tasks

* Activation Functions: ReLU generally performs better than Tanh in deep networks

* Regularization: BatchNorm and Dropout are crucial for preventing overfitting

* Data Augmentation: Essential for improving model generalization


username : admin

password : admin123
