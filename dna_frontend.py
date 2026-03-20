import os
import streamlit as st
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Iris Color Prediction (Random Forest)")
st.sidebar.header("Model Status")

# --- LOAD RANDOM FOREST MODEL ---
rf_model = None

try:
    # Load the Random Forest model (Expects 4 features)
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

# Using two columns for a cleaner layout
col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2], help="Major determinant for Blue/Brown eyes")
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

# --- PREDICTION LOGIC ---
if st.button("Predict Eye Color", type="primary"):
    if rf_model is not None:
        try:
            # 1. Prepare input as a 2D array (1 row, 4 columns)
            rf_input = np.array([[s1, s2, s3, s4]])
            
            # 2. Execute Prediction
            rf_pred = rf_model.predict(rf_input)[0]
            
            # 3. Get Probabilities (To see if the model is actually changing its mind)
            probs = rf_model.predict_proba(rf_input)[0]
            classes = rf_model.classes_
            
            st.divider()
            
            # --- DISPLAY RESULTS ---
            st.write("### Prediction Result")
            st.success(f"The predicted eye color is: **{rf_pred}**")
            
            # Show confidence levels for debugging the 'Always Green' issue
            with st.expander("View Prediction Confidence"):
                for cls, prob in zip(classes, probs):
                    st.write(f"**{cls}:** {prob*100:.2f}%")
            
            # --- IMAGE DISPLAY ---
            # Standardize filename to lowercase to match your files
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
        st.error("Model not loaded. Check the sidebar for errors.")

st.divider()
