import os
import streamlit as st
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction System")
st.sidebar.header("Model Status")

# --- LOAD RANDOM FOREST MODEL ---
rf_model = None

try:
    # Load Random Forest (Expects 4 features)
    if os.path.exists("random_forest_model.pkl"):
        rf_model = joblib.load("random_forest_model.pkl")
        st.sidebar.success("✅ Random Forest Model Ready")
    else:
        st.sidebar.error("❌ 'random_forest_model.pkl' not found.")
except Exception as e:
    st.sidebar.error(f"❌ Load Error: {e}")

# --- USER INPUT UI ---
st.divider()
st.subheader("Enter SNP Genotypes")
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
    if rf_model is not None:
        try:
            # Random Forest expects exactly 4 inputs
            rf_input = np.array([[s1, s2, s3, s4]])
            
            # Execute Prediction
            rf_pred = rf_model.predict(rf_input)[0]
            
            st.divider()
            st.write("### Prediction Result")
            st.success(f"The predicted eye color is: **{rf_pred}**")
            
            # --- IMAGE DISPLAY ---
            # Standardize filename to lowercase
            color_key = str(rf_pred).lower()
            if "brown" in color_key:
                img_file = "brown_eye.png"
            elif "green" in color_key:
                img_file = "green_eye.png"
            else:
                img_file = "blue_eye.png"
                
            if os.path.exists(img_file):
                st.image(img_file, width=400, caption=f"Phenotype: {rf_pred}")
            else:
                st.warning(f"⚠️ Image '{img_file}' not found in the repository.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not loaded. Check the sidebar for errors.")

st.divider()
