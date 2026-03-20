import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# -------------------------
# Load Models
# -------------------------
rf_model = joblib.load("models/random_forest_model.pkl")
dl_model = tf.keras.models.load_model("../models/dl_model.h5")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔬 Forensic DNA Phenotyping - Iris Color Prediction")
st.write("Upload SNP values to predict **eye color** using ML + Deep Learning")

# SNP inputs (example for model with 10 SNPs)
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

    # DEBUG OUTPUT
    st.write("RF RAW MODEL OUTPUT:", rf_pred)

    # Normalize output
    rf_pred = rf_pred.lower()

    # Map prediction
    if rf_pred == "brown":
        predicted_color = "Brown"
    elif rf_pred == "green":
        predicted_color = "Green"
    else:
        predicted_color = "Blue"

    # Display result
    st.success(f"🎯 Predicted Eye Color: **{predicted_color}**")

    # Display Image
    if predicted_color == "Blue":
        st.image("../images/blue_eye.png", width=250)
    elif predicted_color == "Green":
        st.image("../images/green_eye.png", width=250)
    else:
        st.image("../images/brown_eye.png", width=250)
