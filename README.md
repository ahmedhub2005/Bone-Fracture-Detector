
# 🦴 Bone Fracture Detector

A web-based application for detecting bone fractures from X-ray images using deep learning. Built with **TensorFlow**, **Keras**, and **Streamlit**, this project aims to provide a fast, accessible diagnostic tool for medical professionals and researchers.

## 🚀 Features

- Upload X-ray images and receive instant fracture predictions
- Deep learning model trained on labeled fracture datasets
- Streamlit-powered UI for seamless interaction
- Real-time visualization of prediction confidence

## 🧠 Model Architecture

- **Base Model**: Pretrained CNN (e.g., ResNet50 or MobileNetV2)
- **Input**: X-ray image (JPG/PNG)
- **Output**: Binary classification — Fractured / Not Fractured
- **Training**: Fine-tuned on curated medical image datasets

## 🛠️ Installation

```bash
git clone https://github.com/ahmedhub2005/Bone-Fracture-Detector.git
cd Bone-Fracture-Detector
pip install -r requirements.txt
streamlit run app.py
