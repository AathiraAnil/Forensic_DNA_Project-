import os
# Force the stable engine
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# --- LOAD MODELS ---
# Since files are in the same folder as this script, no paths are needed
try:
    rf_model = joblib.load("random_forest_model.pkl")
    dl_model = tf.keras.models.load_model("dl_model.keras")
    st.sidebar.success("✅ Models Loaded")
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- UI ---
st.title("🔬 Forensic DNA Phenotyping")

s1 = st.selectbox("SNP 1", [0, 1, 2])
s2 = st.selectbox("SNP 2", [0, 1, 2])
s3 = st.selectbox("SNP 3", [0, 1, 2])
s4 = st.selectbox("SNP 4", [0, 1, 2])

if st.button("Predict"):
    input_data = np.array([[s1, s2, s3, s4]])
    
    # Prediction Logic
    prediction = rf_model.predict(input_data)[0]
    predicted_color = str(prediction).lower()
    
    if "brown" in predicted_color:
        img = "brown_eye.png"
    elif "green" in predicted_color:
        img = "green_eye.png"
    else:
        img = "blue_eye.png"
        
    st.write(f"### Result: {predicted_color.capitalize()}")
    
    # Show image (since it's in the same main folder)
    if os.path.exists(img):
        st.image(img, width=300)
    else:
        st.warning(f"Image {img} not found in the repository.")
