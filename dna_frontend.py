import os
import streamlit as st
import numpy as np
import joblib

# THE FIX: Use the legacy-specific library to stop the "Recursion" loop
try:
    import tensorflow as tf
    import tf_keras as keras  # This is the secret weapon
except ImportError:
    st.error("🚨 Libraries are still installing. Please wait 2-3 minutes.")

st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction System")

# --- LOAD MODELS ---
rf_model = None
dl_model = None

try:
    # 1. Load Random Forest
    rf_model = joblib.load("random_forest_model.pkl")
    
    # 2. Load Deep Learning using the SPECIAL legacy loader
    # This avoids the "Maximum Recursion Depth" error completely
    if os.path.exists("dl_model.keras"):
        dl_model = keras.models.load_model("dl_model.keras", compile=False)
        st.sidebar.success("✅ Models Loaded Successfully")
    else:
        st.sidebar.error("❌ 'dl_model.keras' not found in main folder.")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")

# --- UI LOGIC ---
st.divider()
st.subheader("Enter SNP Genotypes")
col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2])
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

if st.button("Predict Eye Color", type="primary"):
    if rf_model and dl_model:
        input_data = np.array([[s1, s2, s3, s4]])
        
        # RF Prediction
        rf_res = rf_model.predict(input_data)[0]
        
        # DL Prediction
        dl_raw = dl_model.predict(input_data)
        classes = ["Blue", "Brown", "Green"]
        dl_res = classes[np.argmax(dl_raw)]
        
        st.write(f"#### Result: {rf_res}")
        
        # Image Display
        img_name = f"{str(rf_res).lower()}_eye.png"
        if os.path.exists(img_name):
            st.image(img_name, width=400)
        else:
            st.warning(f"Image {img_name} not found.")
    else:
        st.error("Models not loaded.")
