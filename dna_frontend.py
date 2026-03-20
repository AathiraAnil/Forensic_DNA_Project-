import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# -------------------------
# Setup Paths (CRITICAL FOR DEPLOYMENT)
# -------------------------
# This finds the absolute path to your script's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load Models Safely
# -------------------------

# 1. Random Forest Path
rf_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
if os.path.isfile(rf_path):
    rf_model = joblib.load(rf_path)
else:
    st.error(f"Random Forest model not found at: {rf_path}")

# 2. Deep Learning Path (FIXED: Removed ../)
dl_path = os.path.join(BASE_DIR, "models", "dl_model.h5")
if os.path.isfile(dl_path):
    dl_model = tf.keras.models.load_model(dl_path)
else:
    st.error(f"Deep Learning model not found at: {dl_path}")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Forensic DNA Phenotyping", page_icon="🔬")
st.title("🔬 Forensic DNA Phenotyping - Iris Color Prediction")
st.write("Upload SNP values to predict **eye color** using ML + Deep Learning")

# SNP inputs (example for model with 4 SNPs as per your loop)
snp_labels = [f"SNP {i+1}" for i in range(4)]

snp_values = []
for snp in snp_labels:
    value = st.selectbox(snp, ["0", "1", "2"], index=0)
    snp_values.append(int(value))

input_data = np.array(snp_values).reshape(1, -1)

# -------------------------
# Predict Button
# -------------------------

if st.button("Predict Eye Color"):
    # Random Forest Prediction
    rf_pred = rf_model.predict(input_data)[0]

    # Normalize output
    rf_pred = str(rf_pred).lower().strip()

    # Map prediction
    if "brown" in rf_pred:
        predicted_color = "Brown"
        img_name = "brown_eye.png"
    elif "green" in rf_pred:
        predicted_color = "Green"
        img_name = "green_eye.png"
    else:
        predicted_color = "Blue"
        img_name = "blue_eye.png"

    # Display result
    st.success(f"🎯 Predicted Eye Color: **{predicted_color}**")

    # 3. Display Image Safely (FIXED: Removed ../)
    img_path = os.path.join(BASE_DIR, "images", img_name)
    
    if os.path.isfile(img_path):
        st.image(img_path, width=250)
    else:
        st.warning(f"Eye image not found at: {img_path}")
