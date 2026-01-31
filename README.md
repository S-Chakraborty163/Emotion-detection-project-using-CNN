🎭 Facial Emotion Recognition using CNN & Transfer Learning
📌 Project Overview
This project implements a deep learning-based Facial Emotion Recognition (FER) system that classifies human emotions from facial expressions. The model is trained on the FER2013 dataset and utilizes both custom CNN architectures and transfer learning with pre-trained models (VGG16) to achieve high accuracy in emotion classification.

🎯 Key Features
Multiple CNN Architectures: Implements 4 different CNN models including custom CNNs and transfer learning approaches

Transfer Learning: Utilizes VGG16 and ResNet50V2 pre-trained models with fine-tuning

Data Augmentation: Extensive image augmentation techniques to improve model generalization

Comprehensive Evaluation: Includes confusion matrices, classification reports, ROC curves, and accuracy/loss visualizations

Gradio Deployment: User-friendly web interface for real-time emotion detection

Professional Training Pipeline: Implements callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) and class balancing

📊 Dataset
FER2013 Dataset: Contains 35,887 grayscale facial images (48x48 pixels) labeled with 7 emotion categories:

😠 Angry

😨 Fear

😟 Sad

😊 Happy

😲 Surprise

😐 Neutral

🤢 Disgust

🏗️ Model Architectures
CNN1 - Custom baseline CNN

CNN2_With_Augmentation - CNN with data augmentation

CNN3_VGG16 - Transfer learning with VGG16 (top layers fine-tuned)

CNN4_ResNet50 - Transfer learning with ResNet50V2

🚀 Deployment
The model is deployed using Gradio for real-time emotion detection. Users can upload images or use their webcam to detect facial emotions instantly.

📈 Performance
Data Augmentation: Rotation, zoom, width/height shifts, horizontal flips

Optimization: Adam optimizer with learning rate scheduling

Regularization: Dropout layers, batch normalization

Early Stopping: Prevents overfitting and saves best model

Class Weighting: Addresses class imbalance in the dataset

🛠️ Technical Implementation
Framework: TensorFlow/Keras

Pre-trained Models: VGG16, ResNet50V2

Image Processing: OpenCV, PIL

Visualization: Matplotlib, Seaborn

Evaluation: Scikit-learn metrics

Deployment: Gradio
