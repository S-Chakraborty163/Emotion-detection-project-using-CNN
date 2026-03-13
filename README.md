# 🎭 Facial Emotion Recognition (FER)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Deployment-Gradio-green.svg)](https://gradio.app)

## 📌 Project Overview
This project implements a deep learning-based **Facial Emotion Recognition (FER) system** that classifies human emotions from facial expressions. The model is trained on the **FER2013 dataset** and utilizes both custom CNN architectures and transfer learning (VGG16/ResNet50V2) to achieve high accuracy in emotion classification.

---

## 🎯 Key Features
* **Multiple Architectures**: Implements 4 distinct models ranging from custom baselines to advanced Transfer Learning.
* **Transfer Learning**: Fine-tuned **VGG16** and **ResNet50V2** for superior feature extraction.
* **Advanced Training**: Includes data augmentation, class balancing, and automated callbacks (`EarlyStopping`, `ReduceLROnPlateau`).
* **Deployment**: Real-time emotion detection via a **Gradio** web interface.
* **Evaluation**: Detailed analytics including Confusion Matrices, ROC Curves, and Accuracy/Loss plots.

---

## 📊 Dataset: FER2013
The model is trained on 35,887 grayscale images (48x48 pixels) categorized into 7 emotions:

| Icon | Emotion | Icon | Emotion |
| :--- | :--- | :--- | :--- |
| 😠 | **Angry** | 😲 | **Surprise** |
| 😨 | **Fear** | 😐 | **Neutral** |
| 😟 | **Sad** | 🤢 | **Disgust** |
| 😊 | **Happy** | | |

---

## 🏗️ Model Architectures
| Model | Description |
| :--- | :--- |
| **CNN1** | Custom baseline CNN architecture. |
| **CNN2** | CNN enhanced with **Data Augmentation**. |
| **CNN3** | Transfer Learning using **VGG16** (Fine-tuned top layers). |
| **CNN4** | Transfer Learning using **ResNet50V2**. |

---

## 📈 Performance & Optimization
To ensure model robustness and prevent overfitting, the following techniques were implemented:
* **Data Augmentation**: Rotation, zoom, width/height shifts, and horizontal flips.
* **Regularization**: Integrated Dropout layers and Batch Normalization.
* **Optimization**: Adam optimizer with dynamic learning rate scheduling.
* **Class Weighting**: Applied to handle dataset imbalance across emotion categories.

---

## 🛠️ Technical Stack
* **Deep Learning**: TensorFlow, Keras
* **Computer Vision**: OpenCV, PIL
* **Data Science**: Pandas, NumPy, Scikit-learn
* **Visualization**: Matplotlib, Seaborn
* **Deployment**: Gradio

---

## 🧠 How it Works

### 1. The CNN Architecture (FER)
The models process 48x48 grayscale images through a series of convolutional layers to extract spatial features:

* **Feature Extraction**: Convolutional layers use filters to detect edges, textures, and eventually complex facial landmarks (eyes, mouth).
* **Dimensionality Reduction**: Max-pooling layers reduce the spatial size of the representation to decrease computational load and prevent overfitting.
* **Classification**: Fully connected (Dense) layers with **Softmax activation** output the probability distribution across the 7 emotion classes.


