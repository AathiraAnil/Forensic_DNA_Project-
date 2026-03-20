import os
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")

# --- THE MODEL BUILDER ---
def load_manual_model():
    # We build the 'socket' to match your '10-pin plug'
    model = models.Sequential([
        layers.Input(shape=(10,)), 
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    # Load the weights file you uploaded
    model.load_weights("model_weights.weights.h5")
    return model

# Load models
try:
    rf_model = joblib.load("random_forest_model.pkl")
    dl_model = load_manual_model()
    st.sidebar.success("✅ Models Ready")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")

# --- THE UI ---
st.subheader("Enter your 4 SNPs")
st.write("The model expects 10 values, so we will fill the rest with '0' automatically.")

col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2])
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

if st.button("Predict"):
    # We take your 4 inputs + 6 zeros to satisfy the model's '10-pin' requirement
    final_input = np.array([[s1, s2, s3, s4, 0, 0, 0, 0, 0, 0]])
    
    # 1. Random Forest Prediction (Assuming it also needs 10)
    try:
        rf_res = rf_model.predict(final_input)[0]
        st.write(f"### Random Forest Result: **{rf_res}**")
        
        # 2. Deep Learning Prediction
        dl_probs = dl_model.predict(final_input)
        classes = ["Blue", "Brown", "Green"]
        dl_res = classes[np.argmax(dl_probs)]
        st.write(f"### Deep Learning Result: **{dl_res}**")
        
        # 3. Show Image
        img = f"{str(rf_res).lower()}_eye.png"
        if os.path.exists(img):
            st.image(img, width=400)
    except Exception as e:
        st.error(f"Prediction Error: {e}. Your Random Forest model might expect a different number of inputs.")
