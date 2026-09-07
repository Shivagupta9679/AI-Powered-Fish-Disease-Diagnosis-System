AI Powered Fish Disease Diagnosis System
📌 Project Overview

Fish diseases are a major challenge in aquaculture, leading to reduced fish health, lower productivity, and economic losses. Early and accurate disease detection is essential for improving fish health management and preventing disease outbreaks.

This project presents a deep learning-based fish disease detection system that classifies fish images as Healthy or Infected using computer vision techniques. The project implements and compares a Custom CNN, VGG16, and DenseNet with transfer learning to identify the most effective model for binary image classification. Data augmentation and early stopping are applied to improve model generalization and reduce overfitting.
🎯Objectives

    Develop an automated fish disease detection system using deep learning.
    Compare the performance of CNN and transfer learning models.
    Improve classification accuracy using data augmentation.
    Evaluate model performance using multiple classification metrics.

🛠️Technologies Used

    Python
    TensorFlow / Keras
    NumPy
    Pandas
    Matplotlib
    Scikit-learn
    OpenCV

🧠Deep Learning Models

The following models were implemented and evaluated:

    Custom Convolutional Neural Network (CNN)
    VGG16 (Transfer Learning)
    DenseNet (Transfer Learning)

Techniques used:

    Data Augmentation
    Early Stopping
    Model Checkpoint
    Class Weight Balancing
    Fine-Tuning

🔬 Methodology

    Collected fish disease images from the Mendeley dataset.
    Preprocessed images using resizing, normalization, and data augmentation.
    Trained CNN, VGG16, and DenseNet models with transfer learning.
    Applied Early Stopping and Class Weights to improve model performance.
    Evaluated the models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    Compared model performance and selected the best-performing model for fish disease detection.

📂Dataset

The dataset used in this study is publicly available.

    Source: Mendeley Data
    Description: A labeled freshwater fish image dataset containing multiple fish disease categories along with healthy fish images.

📁Project Structure

Fish-Disease-Detection/
│
├── dataset/
├── notebooks/
│   └── Fish_Disease_Detection.ipynb
├── models/
├── images/
├── README.md
└── requirements.txt

📈Results

    Custom CNN: 69% test accuracy
    VGG16 (fine-tuned): ~97% validation accuracy (best model)
    Custom DenseNet-style: 86.8% test accuracy

🔮 Future Improvements

    Multi-class fish disease classification
    Explainable AI using Grad-CAM
    Streamlit web application
    Cloud deployment

👨‍💻Author

Shiva Gupta

    M.Tech
    Aquacultural Engineering
    Indian Institute of Technology Kharagpur
