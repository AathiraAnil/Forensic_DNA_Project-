import os
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction System")
st.sidebar.header("Model Status")

# --- MANUAL MODEL BUILDER (DL) ---
def load_manual_dl_model():
    # FIXED: The architecture now exactly matches your weights: 10 -> 32 -> 16 -> 3
    model = models.Sequential([
        layers.Input(shape=(10,)), 
        layers.Dense(32, activation='relu'), # Layer 1 (32 nodes)
        layers.Dense(16, activation='relu'), # Layer 2 (16 nodes)
        layers.Dense(3, activation='softmax') # Output Layer (3 classes)
    ])
    # Load the raw math (weights) you downloaded from Colab
    model.load_weights("model_weights.weights.h5")
    return model

# --- LOAD BOTH MODELS ---
rf_model = None
dl_model = None

try:
    # 1. Load Random Forest (Expects 4 features)
    if os.path.exists("random_forest_model.pkl"):
        rf_model = joblib.load("random_forest_model.pkl")
    
    # 2. Load Deep Learning (Expects 10 features via weights file)
    if os.path.exists("model_weights.weights.h5"):
        dl_model = load_manual_dl_model()
    
    if rf_model and dl_model:
        st.sidebar.success("✅ Both Models Ready")
    else:
        if not rf_model: st.sidebar.error("❌ RF Model (.pkl) missing")
        if not dl_model: st.sidebar.error("❌ Weights (.weights.h5) missing")
except Exception as e:
    st.sidebar.error(f"❌ Load Error: {e}")

# --- USER INPUT UI ---
st.divider()
st.subheader("Enter 4 SNP Genotypes")
st.info("Select the genotype (0, 1, or 2) for the 4 target SNPs.")

col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2])
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

# --- PREDICTION LOGIC ---
if st.button("Predict Eye Color", type="primary"):
    if rf_model is not None and dl_model is not None:
        try:
            # --- DATA PREPARATION ---
            # Random Forest expects exactly 4 inputs
            rf_input = np.array([[s1, s2, s3, s4]])
            
            # Deep Learning expects 10 inputs (4 real + 6 dummy zeros)
            dl_input = np.array([[s1, s2, s3, s4, 0, 0, 0, 0, 0, 0]])
            
            # --- EXECUTE PREDICTIONS ---
            # 1. Random Forest Prediction
            rf_pred = rf_model.predict(rf_input)[0]
            
            # 2. Deep Learning Prediction
            dl_probs = dl_model.predict(dl_input)
            categories = ["Blue", "Brown", "Green"]
            dl_pred = categories[np.argmax(dl_probs)]
            
            st.divider()
            res_c1, res_c2 = st.columns(2)
            
            with res_c1:
                st.write("#### Random Forest")
                st.success(f"**Result: {rf_pred}**")
                
            with res_c2:
                st.write("#### Deep Learning")
                st.info(f"**Result: {dl_pred}**")
                
            # --- IMAGE DISPLAY ---
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
                st.warning(f"⚠️ Image '{img_file}' not found in the repository.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Models not loaded. Check the sidebar for missing files.")

st.divider()
