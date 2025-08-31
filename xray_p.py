import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from tensorflow import keras
from keras.models import load_model as keras_load_model
from keras import layers
import requests
import cv2
import tensorflow as tf

st.set_page_config(page_title="Bone Fracture Detector", layout="centered")

# ==========================
# 1. Model Handling
# ==========================
model_path = "final_bone_fracture.h5"
url = "https://huggingface.co/sonic222/bone-fracture-detector/resolve/main/final_bone_fracture.h5"

if not os.path.exists(model_path):
    try:
        with st.spinner("Downloading model..."):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# ==========================
# 2. Load Model
# ==========================
@st.cache_resource
def load_model(model_path=model_path):
    try:
        model = keras_load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ==========================
# 3. Grad-CAM Function
# ==========================
def make_gradcam_heatmap(img_array, model):
  
    _ = model(img_array)

  
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]

    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap_color, 0.4, 0)
    return overlay

# ==========================
# 4. Prediction Function
# ==========================
def predict_image(image: Image.Image):
    img = image.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    return pred, img_array

# ==========================
# 5. Streamlit Interface
# ==========================
st.title("🦴 Bone Fracture Detector")
st.write("Upload an X-ray image and the model will predict whether a fracture is present.")

with st.sidebar:
    st.header("Options")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)

uploaded_file = st.file_uploader("Upload an X-ray image:", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Predicting..."):
            pred, img_array = predict_image(image)
            confidence = pred if pred > 0.5 else 1 - pred
            confidence_percent = round(confidence*100,2)

            if pred > 0.5:
                st.success(f"No Fracture Detected ✅\nConfidence: {confidence_percent}%")
            else:
                st.error(f"Fracture Detected ⚠️\nConfidence: {confidence_percent}%")

            # Grad-CAM
            if show_gradcam:
                heatmap = make_gradcam_heatmap(img_array, model)
                overlay = display_gradcam(image, heatmap)
                st.image(overlay, caption="Grad-CAM Heatmap Overlay", use_container_width=True)

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")












