import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from PIL import Image
import numpy as np
import os


import gdown


weights_path = "D:\python for data science\bone_fracture.weights.h5"
file_id = "1qxGz0jKHIVWMAEpef3N3Wfj4lx1KvEdM"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(weights_path):
    with st.spinner("Downloading model weights... (this may take a minute)"):
        gdown.download(url, weights_path, quiet=False)


@st.cache_resource
def load_model():
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

model = load_model()


st.title("Bone Fracture Detector")
st.write("Upload an X-ray image and the model will predict whether a fracture is present.")

uploaded_file = st.file_uploader("Upload an X-ray image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
 
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

   
    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)[0][0]

    
    if prediction > 0.5:
        st.success(" No Fracture Detected.")
    else:
        st.error(" Fracture Detected!")






