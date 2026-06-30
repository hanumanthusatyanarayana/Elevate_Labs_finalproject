# Elevate_Labs_finalproject
# 🌿 Plant Disease Detection System

[![Model Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen)](https://cropcare-ai.streamlit.app/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/AI-TensorFlow%2FKeras-orange)](https://www.tensorflow.org/)

An AI-powered environmental intelligence platform designed to assist farmers and agricultural researchers in identifying plant diseases instantly. This system leverages deep learning to bridge the gap between traditional manual diagnosis and modern automated solutions.

## 🚀 Live Demo
Experience the real-time application here: **[cropcare-ai.streamlit.app](https://cropcare-ai.streamlit.app/)**

---

## ✨ Features
* **🧠 High-Accuracy Analysis:** Utilizes a Convolutional Neural Network (CNN) to achieve over 95% accuracy in classifying leaf images.
* **📸 Multimodal Input:** Supports both direct image uploads and real-time capture using your device's camera.
* **🔍 Extensive Coverage:** Comprises 38 distinct classes of healthy and diseased leaf images, including Apple Scab, Corn Rust, and Tomato Bacterial Spot.
* **⚡ Instant Feedback:** Provides immediate predictions with confidence scores and clear visual feedback.
* **📱 Responsive Dashboard:** Built with a clean, intuitive Streamlit UI for seamless use on both mobile and desktop.

---

## 🏗️ Methodology
The system follows a rigorous machine learning pipeline from data ingestion to cloud deployment:

1. **Data Collection:** A labeled dataset of 38 classes of healthy and diseased plant images.
2. **Preprocessing:** Image data preparation using libraries like NumPy and OpenCV.
3. **Model Development:** Custom-built CNN architecture designed with TensorFlow and Keras.
4. **Evaluation:** Rigorous performance assessment using Accuracy, Precision, Recall, and Loss metrics.
5. **Integration:** Deployment of the trained model into a Streamlit web app for real-time predictions.

---

## 📊 Model Performance Highlights
Based on rigorous evaluation of the CNN model:
* **Overall Accuracy:** 94.8%
* **Precision:** 95.4%
* **Recall:** 94.4%
* **Loss:** 0.149

---

## 🛠️ Technology Stack
* **Programming Language:** Python
* **AI Framework:** TensorFlow / Keras
* **Image Processing:** PIL (Pillow), NumPy
* **Web Framework:** Streamlit
* **Environment:** Google Colab

---

## 📂 Project Structure
```text
Plant-Disease-Detection/
│
├── .devcontainer/          # Development environment configuration
├── Plant_Disease_Detection_System.ipynb  # Research and model training
├── README.md               # You are here
├── app.py                  # Main Streamlit web application
├── model.h5                # Pre-trained CNN model (38 classes)
└── requirements.txt        # Project dependencies

```

---

## 🔮 Future Roadmap

* **Dataset Expansion:** Including more plant species and diverse disease categories for better generalization.
* **Edge Optimization:** Applying techniques like pruning and quantization to increase inference speed on low-power devices.
* **IoT Integration:** Combining with drones or smart farming devices for real-time disease surveillance.
* **Multilingual Support:** Adding regional language support to assist rural farming communities.

---

## 👨‍💻 Author

**Hanumanthu Satyanarayana** *AI Researcher & Software Engineer*

---

*Aligning with UN Sustainable Development Goals: **SDG 2 (Zero Hunger)** and **SDG 9 (Innovation & Infrastructure)***.
