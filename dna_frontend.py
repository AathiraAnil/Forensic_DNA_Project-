import os
# Force the legacy engine before anything else
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Try-Except block for TensorFlow to catch the "Keras cannot be imported" error
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    st.error("🚨 TensorFlow/Keras installation failed on the server. Try Rebooting the App.")

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    for root, dirs, files in os.walk(BASE_DIR):
        if name in files:
            return os.path.join(root, name)
    return None

st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction System")

# --- LOAD MODELS ---
rf_path = find_file("random_forest_model.pkl")
dl_path = find_file("dl_model.h5")

rf_model = None
dl_model = None

if rf_path:
    try:
        rf_model = joblib.load(rf_path)
    except Exception as e:
        st.warning(f"⚠️ RF Model version mismatch. Error: {str(e)[:50]}...")
        st.info("💡 Tip: Re-save your .pkl file on your laptop using 'protocol=4' for better compatibility.")

if dl_path:
    try:
        # Using the standard Keras loader
        dl_model = tf.keras.models.load_model(dl_path, compile=False)
    except Exception as e:
        st.warning(f"⚠️ DL Model load failed. Error: {str(e)[:50]}...")

# --- UI LOGIC ---
if rf_model or dl_model:
    st.divider()
    st.subheader("Enter SNP Data")
    
    # 4 SNP Inputs
    col1, col2 = st.columns(2)
    with col1:
        s1 = st.selectbox("SNP 1", [0, 1, 2])
        s2 = st.selectbox("SNP 2", [0, 1, 2])
    with col2:
        s3 = st.selectbox("SNP 3", [0, 1, 2])
        s4 = st.selectbox("SNP 4", [0, 1, 2])

    if st.button("Predict Phenotype"):
        input_data = np.array([[s1, s2, s3, s4]])
        
        # Use whichever model loaded successfully
        if rf_model:
            prediction = rf_model.predict(input_data)[0]
            st.success(f"**Random Forest Prediction:** {prediction}")
        
        if dl_model:
            # Simple categorical check (assuming 3 classes: Blue, Brown, Green)
            dl_raw = dl_model.predict(input_data)
            classes = ["Blue", "Brown", "Green"]
            dl_res = classes[np.argmax(dl_raw)]
            st.info(f"**Deep Learning Prediction:** {dl_res}")
else:
    st.error("No models could be loaded. Please check your GitHub file structure.")
