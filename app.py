import streamlit as st
import numpy as np
import cv2
import gdown
import os
import h5py
from tensorflow.keras.models import load_model
from PIL import Image

# Google Drive file ID of model.h5
file_id = "1r8JloPkXvxPzkr1dz6Ow7RznSiwZkM7o"
model_path = "model.h5"

# Function to verify if model file is valid
def is_valid_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except:
        return False

# Download model if not present or invalid
if not os.path.exists(model_path) or not is_valid_h5(model_path):
    st.write("Downloading model... Please wait.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the trained model
if is_valid_h5(model_path):
    model = load_model(model_path)
else:
    st.error("Failed to download or load model. Please check the file.")

# Define the prediction function
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3]) / 255.0
    prediction = model.predict(img)
    result = np.argmax(prediction)
    return result, prediction

# Streamlit app UI
st.title("♻️ Waste Classification")
st.write("Upload an image to classify it as **Recyclable** or **Organic waste**.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)
    
    st.write("🔄 **Classifying... Please wait.**")
    result, prediction = predict_fun(image)

    labels = ["♻️ Recyclable", "🌱 Organic Waste"]
    st.success(f"**Prediction:** {labels[result]}")
    st.write("📊 **Prediction probabilities:**", prediction)
