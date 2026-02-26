import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import json
# load model
@st.cache_resource
def get_model():
    model = load_model('models/model.h5',custom_objects={'preprocess_input': preprocess_input})
    return model
model = get_model()
#load class names
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    st.error("Class names file is missing. Please ensure 'class_names.json' is in the correct directory.")

st.title("Vehicle Type Classification")

st.header("Upload an image of a vehicle to classify its type.")
st.subheader("Supported types: Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains")

uploaded_file = st.file_uploader("Choose an image...", type = ['jpg','jpeg', 'png'], key = "upload")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded Image', use_container_width = True)
    st.write("Classifying...")

    #image preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    # prediction
    prediction = model.predict(img_array)
    score = prediction[0]

    # results
    prediction_class = class_names[np.argmax(score)]
    st.success(f"This image most likely belongs to **{prediction_class}**")
    st.write(f"Confidence: {100 * np.max(score):.2f}%")
