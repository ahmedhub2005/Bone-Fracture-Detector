import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from tensorflow import keras
from keras import layers, models
import requests

weights_path = "bone_fracture.weights.h5"
url = "https://huggingface.co/sonic222/bone-fracture-detector/resolve/main/bone_fracture.weights.h5"


if not os.path.exists(weights_path):
    try:
        with open(weights_path, "wb") as f:
            f.write(requests.get(url).content)
    except Exception as e:
        st.error(f"Failed to download model weights: {e}")
        st.stop()

@st.cache_resource
def load_model():
    try:
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
            layers.MaxPooling2D((2,2)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()

model = load_model()

st.title("Bone Fracture Detector")
st.write("Upload an X-ray image and the model will predict whether a fracture is present.")

uploaded_file = st.file_uploader("Upload an X-ray image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((150,150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.success("No Fracture Detected.")
        else:
            st.error("Fracture Detected!")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")













