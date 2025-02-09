import streamlit as st
import numpy as np
import cv2
import gdown
import os
from tensorflow.keras.models import load_model
from PIL import Image

# App title and design
st.set_page_config(page_title="Plastic Waste Classifier", page_icon="🌟", layout="centered")
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #FF5733;
            text-align: center;
        }
        .subtext {
            font-size: 18px;
            color: #666;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("About This App")
st.sidebar.info("This app classifies plastic waste as **Recyclable** or **Organic** using a CNN model.")
st.sidebar.markdown("**Created by: Gokul 🚀**")
st.sidebar.markdown("---")

# Google Drive file ID of model.h5
file_id = "1r8JloPkXvxPzkr1dz6Ow7RznSiwZkM7o"
model_path = "model.h5"

# Download model if not present
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.sidebar.write("Downloading model... Please wait.")
    gdown.download(url, model_path, quiet=False)

# Load the trained model
model = load_model(model_path)

# Define the prediction function
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3])
    img = img / 255.0
    prediction = model.predict(img)
    result = np.argmax(prediction)
    return result, prediction

# Streamlit app
st.markdown("<p class='title'>Plastic Waste Classification</p>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload an image to classify it as Recyclable or Organic waste.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("\n")
    st.write("Classifying...")
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    result, prediction = predict_fun(image)
    progress.empty()
    
    if result == 0:
        st.success("♻️ The image is classified as **Recyclable**!")
    elif result == 1:
        st.warning("🍂 The image is classified as **Organic Waste**!")
    
    st.write("**Prediction Probabilities:**", prediction)

# Footer
st.markdown("<p class='footer'>Made with ❤️ by Gokul</p>", unsafe_allow_html=True)
