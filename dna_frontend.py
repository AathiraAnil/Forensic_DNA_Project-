import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# -------------------------
# Load Models
# -------------------------
rf_model = joblib.load("../models/random_forest_model.pkl")
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
    # RF Prediction
    rf_pred = rf_model.predict(input_data)[0]
    

    if rf_pred == "Brown":
        predicted_color = "Brown"
    elif rf_pred == "Green":
        predicted_color = "Green"
    else:
        predicted_color = "Blue"

    st.success(f"🎯 Predicted Eye Color:      **{predicted_color}**")


    labels = ["Blue", "Green", "Brown"]

    st.success(f"🎯 Random Forest Prediction: **{rf_pred}**")
    

   
   