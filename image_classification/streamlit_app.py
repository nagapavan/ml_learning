import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Set page design
st.set_page_config(page_title='COVID-19 Chest X-Ray Detector', layout='centered')
st.title('🩺 COVID-19 Chest X-Ray Classification')
st.write("Upload a chest X-ray image to detect if it is Covid, Normal, or Viral Pneumonia.")

# Load the model and class names relative to this script's directory
import os

@st.cache_resource
def load_artifacts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_covid_cnn_model.h5')
    pickle_path = os.path.join(script_dir, 'class_names.pkl')
    
    model = tf.keras.models.load_model(model_path)
    with open(pickle_path, 'rb') as f:
        class_names = pickle.load(f)
    return model, class_names

try:
    model, class_names = load_artifacts()
    st.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Image file uploader
uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    # Preprocess image
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize((150, 150))
    image_array = np.array(image_resized)
    image_normalized = image_array.astype('float32') / 255.0
    input_tensor = np.expand_dims(image_normalized, axis=0)

    # Run prediction
    with st.spinner('Analyzing X-ray image...'):
        predictions = model.predict(input_tensor)
        pred_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][pred_class_idx] * 100
        predicted_class = class_names[pred_class_idx]

    # Display results
    st.write("---")
    st.subheader("Prediction Result:")

    if predicted_class == 'Covid':
        st.error(f"🚨 **COVID-19 Detected** (Confidence: {confidence:.2f}%)")
    elif predicted_class == 'Viral Pneumonia':
        st.warning(f"⚠️ **Viral Pneumonia Detected** (Confidence: {confidence:.2f}%)")
    else:
        st.success(f"✅ **Normal (No abnormality detected)** (Confidence: {confidence:.2f}%)")