import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SMART LOAD: RANDOM FOREST ---
rf_paths = [
    os.path.join(BASE_DIR, "models", "random_forest_model.pkl"),
    os.path.join(BASE_DIR, "Models", "random_forest_model.pkl"),
    os.path.join(BASE_DIR, "random_forest_model.pkl")
]

rf_model = None
for path in rf_paths:
    if os.path.isfile(path):
        rf_model = joblib.load(path)
        break

if rf_model is None:
    st.error(f"❌ Random Forest Model NOT found. I checked: {rf_paths}")
    st.stop()

# --- SMART LOAD: DEEP LEARNING ---
dl_paths = [
    os.path.join(BASE_DIR, "models", "dl_model.h5"),
    os.path.join(BASE_DIR, "Models", "dl_model.h5"),
    os.path.join(BASE_DIR, "dl_model.h5")
]

dl_model = None
for path in dl_paths:
    if os.path.isfile(path):
        dl_model = tf.keras.models.load_model(path)
        break

if dl_model is None:
    st.error(f"❌ Deep Learning Model NOT found. I checked: {dl_paths}")
    st.stop()

# --- STREAMLIT UI ---
st.set_page_config(page_title="DNA Phenotyping", page_icon="🔬")
st.title("🔬 Forensic DNA Phenotyping - Iris Color Prediction")

snp_labels = [f"SNP {i+1}" for i in range(4)]
snp_values = []
for snp in snp_labels:
    value = st.selectbox(snp, ["0", "1", "2"], index=0)
    snp_values.append(int(value))

input_data = np.array(snp_values).reshape(1, -1)

# --- PREDICTION ---
if st.button("Predict Eye Color"):
    rf_pred = str(rf_model.predict(input_data)[0]).lower().strip()

    if "brown" in rf_pred:
        predicted_color, img_name = "Brown", "brown_eye.png"
    elif "green" in rf_pred:
        predicted_color, img_name = "Green", "green_eye.png"
    else:
        predicted_color, img_name = "Blue", "blue_eye.png"

    st.success(f"🎯 Predicted Eye Color: **{predicted_color}**")

    # Display Image
    img_path = os.path.join(BASE_DIR, "images", img_name)
    if os.path.isfile(img_path):
        st.image(img_path, width=250)
    else:
        st.warning(f"Image '{img_name}' not found in /images folder.")
