import os
# MANDATORY: This must be the very first line to stop the "Recursion" error
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Special handling for TensorFlow to prevent "Keras not found" errors
try:
    import tensorflow as tf
except ImportError:
    st.error("🚨 TensorFlow is not installed. Check your requirements.txt.")

# --- PAGE SETUP ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction System")
st.sidebar.header("Model Status")

# --- LOAD MODELS ---
# We assume the files are in the main folder based on your screenshot
rf_model = None
dl_model = None

try:
    # 1. Load the Random Forest model
    rf_model = joblib.load("random_forest_model.pkl")
    
    # 2. Load the NEW .keras file from Google Colab
    dl_model = tf.keras.models.load_model("dl_model.keras")
    
    st.sidebar.success("✅ Both Models Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.sidebar.info("Ensure 'dl_model.keras' and 'random_forest_model.pkl' are in the main GitHub folder.")

# --- INPUT UI ---
st.divider()
st.subheader("Enter SNP Genotypes")
st.info("Please select the genotype for each SNP (0, 1, or 2).")

col1, col2 = st.columns(2)
with col1:
    snp1 = st.selectbox("SNP 1 (rs12913832)", [0, 1, 2], help="HERC2 gene variant")
    snp2 = snp1 # Duplicate for demo if you only have 4 inputs total
    snp2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    snp3 = st.selectbox("SNP 3", [0, 1, 2])
    snp4 = st.selectbox("SNP 4", [0, 1, 2])

# --- PREDICTION LOGIC ---
if st.button("Predict Eye Color", type="primary"):
    if rf_model is not None and dl_model is not None:
        # Prepare input data (1 row, 4 features)
        input_data = np.array([[snp1, snp2, snp3, snp4]])
        
        # 1. Random Forest Prediction
        rf_pred = rf_model.predict(input_data)[0]
        
        # 2. Deep Learning Prediction
        dl_probs = dl_model.predict(input_data)
        categories = ["Blue", "Brown", "Green"]
        dl_pred = categories[np.argmax(dl_probs)]
        
        st.divider()
        
        # --- DISPLAY RESULTS ---
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("#### Random Forest")
            st.success(f"**Result: {rf_pred}**")
            
        with res_col2:
            st.write("#### Deep Learning")
            st.info(f"**Result: {dl_pred}**")
            
        # --- IMAGE DISPLAY ---
        # Logic to pick the right image based on RF prediction
        color_key = str(rf_pred).lower()
        if "brown" in color_key:
            img_file = "brown_eye.png"
        elif "green" in color_key:
            img_file = "green_eye.png"
        else:
            img_file = "blue_eye.png"
            
        if os.path.exists(img_file):
            st.image(img_file, width=400, caption=f"Predicted Phenotype: {rf_pred}")
        else:
            st.warning(f"⚠️ Image '{img_file}' not found in the GitHub main folder.")
            
    else:
        st.error("Models are not loaded. Prediction cannot proceed.")

st.divider()
st.caption("B.Tech AIML Mini Project - Forensic DNA Phenotyping")
