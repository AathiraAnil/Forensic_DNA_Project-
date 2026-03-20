import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD MODELS SAFELY ---
# 1. Random Forest
rf_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
if os.path.isfile(rf_path):
    rf_model = joblib.load(rf_path)
else:
    st.error(f"❌ Random Forest model NOT found at: {rf_path}")
    st.stop() # Stops the app here so it doesn't crash later

# 2. Deep Learning
dl_path = os.path.join(BASE_DIR, "models", "dl_model.h5")
if os.path.isfile(dl_path):
    dl_model = tf.keras.models.load_model(dl_path)
else:
    st.error(f"❌ Deep Learning model NOT found at: {dl_path}")
    st.stop()

# --- STREAMLIT UI ---
st.set_page_config(page_title="DNA Phenotyping", page_icon="🔬")
st.title("🔬 Forensic DNA Phenotyping - Iris Color Prediction")
st.write("Upload SNP values to predict **eye color**.")

snp_labels = [f"SNP {i+1}" for i in range(4)]
snp_values = []
for snp in snp_labels:
    value = st.selectbox(snp, ["0", "1", "2"], index=0)
    snp_values.append(int(value))

input_data = np.array(snp_values).reshape(1, -1)

# --- PREDICTION ---
if st.button("Predict Eye Color"):
    # RF Prediction
    rf_pred = rf_model.predict(input_data)[0]
    rf_pred = str(rf_pred).lower().strip()

    if "brown" in rf_pred:
        predicted_color = "Brown"
        img_name = "brown_eye.png"
    elif "green" in rf_pred:
        predicted_color = "Green"
        img_name = "green_eye.png"
    else:
        predicted_color = "Blue"
        img_name = "blue_eye.png"

    st.success(f"🎯 Predicted Eye Color: **{predicted_color}**")

    # Display Image Safely
    img_path = os.path.join(BASE_DIR, "images", img_name)
    if os.path.isfile(img_path):
        st.image(img_path, width=250)
    else:
        st.warning(f"Note: Image '{img_name}' not found in /images folder.")
