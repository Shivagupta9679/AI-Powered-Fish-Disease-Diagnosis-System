# 🐟AI Powered Fish Disease Diagnosis System

## 📌 Project Overview
Fish diseases are a major challenge in aquaculture, leading to reduced fish health, lower productivity, and economic losses. Early and accurate disease detection is essential for improving fish health management and preventing disease outbreaks.

This project combines deep learning-based image classification with an LLM-powered RAG pipeline to support fish disease detection and provide relevant informational guidance.The system uses a fine-tuned VGG16 model for image classification and FAISS with the OpenAI API to retrieve relevant information and generate contextual responses.

## 🎯Objectives
- Develop an automated fish disease detection system using deep learning.
- Compare the performance of CNN and transfer learning models.
- Improve classification accuracy using data augmentation.
- Evaluate model performance using multiple classification metrics.
- Integrate RAG and an LLM to provide relevant disease-related information.
- Build a Streamlit interface for local application use.

## 🛠️Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- RAG

## 🧠Deep Learning Models
The following models were implemented and evaluated:

- Custom Convolutional Neural Network (CNN)
- VGG16 (Transfer Learning)
- DenseNet (Transfer Learning)

Techniques used:

- Data Augmentation
- Early Stopping
- Model Checkpoint
- Class Weight Balancing
- Fine-Tuning

## 🔬 Methodology

- Collected fish disease images from the Mendeley dataset.
- Preprocessed images using resizing, normalization, and data augmentation.
- Trained **CNN**, **VGG16**, and **DenseNet** models with transfer learning.
- Applied **Early Stopping** and **Class Weights** to improve model performance.
- Evaluated the models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- Compared model performance and selected the best-performing model for fish disease detection.
- Used FAISS for retrieving relevant information and the OpenAI API for generating contextual responses.
- Built a Streamlit interface and ran the application locally through VS Code.

## 📂Dataset
The dataset used in this study is publicly available.

- Source: Mendeley Data
- Description: A labeled freshwater fish image dataset containing multiple fish disease categories along with healthy fish images.

## 📁Project Structure
```
Fish-Disease-Detection/
AI-Powered-Fish-Disease-Diagnosis-System/
│
├── dataset/
├── notebooks/
│   └── Fish_Disease_Detection.ipynb
├── models/
├── images/
├── app.py
├── requirements.txt
└── Readme.md
```

## 📈Results
- Custom CNN: 69% test accuracy
- VGG16 (fine-tuned): ~97% validation accuracy (best model)
- Custom DenseNet-style: 86.8% test accuracy

## 👨‍💻Author

Shiva Gupta

- M.Tech
- Aquacultural Engineering
- Indian Institute of Technology Kharagpur
