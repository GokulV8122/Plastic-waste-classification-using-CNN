import streamlit as st
import numpy as np
import cv2
import gdown
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Google Drive file ID of model.h5
file_id = "YOUR_FILE_ID"
model_path = "model.h5"

# Download model if not present
if not os.path.exists(model_path):
    url = f"https://drive.google.com/file/d/1r8JloPkXvxPzkr1dz6Ow7RznSiwZkM7o/view?usp=sharing"
    st.write("Downloading model... Please wait.")
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
st.title("Plastic Waste Classification")
st.write("Upload an image to classify it as Recyclable or Organic waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    result, prediction = predict_fun(image)

    if result == 0:
        st.write("The image shown is Recyclable")
    elif result == 1:
        st.write("The image shown is Organic waste")

    st.write("Prediction probabilities:", prediction)
