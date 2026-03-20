import os
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Multi-Color Probability Analysis")
st.sidebar.header("Model Status")

# --- LOAD MODEL ---
rf_model = None
try:
    if os.path.exists("random_forest_model.pkl"):
        rf_model = joblib.load("random_forest_model.pkl")
        st.sidebar.success("✅ Random Forest Model Ready")
    else:
        st.sidebar.error("❌ 'random_forest_model.pkl' not found.")
except Exception as e:
    st.sidebar.error(f"❌ Load Error: {e}")

# --- USER INPUT UI ---
st.divider()
st.subheader("Input DNA Genotypes")
col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2])
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

# --- PREDICTION LOGIC ---
if st.button("Run Full Analysis", type="primary"):
    if rf_model is not None:
        try:
            # 1. Prepare Input
            rf_input = np.array([[s1, s2, s3, s4]])
            
            # 2. Get Probabilities for ALL classes
            probs = rf_model.predict_proba(rf_input)[0]
            classes = rf_model.classes_
            
            # 3. Create a Dataframe for the Chart
            prob_df = pd.DataFrame({
                'Color': classes,
                'Probability (%)': probs * 100
            }).set_index('Color')

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("📊 Probability Distribution")
            
            # Display the Bar Chart
            st.bar_chart(prob_df)

            # Display the Winning Prediction
            winner = rf_model.predict(rf_input)[0]
            st.success(f"**Highest Probability Phenotype: {winner}**")

            # --- DYNAMIC IMAGE DISPLAY ---
            color_key = str(winner).lower()
            img_file = f"{color_key}_eye.png"
                
            if os.path.exists(img_file):
                st.image(img_file, width=300, caption=f"Visualized Phenotype: {winner}")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
    else:
        st.error("Model not loaded.")

st.divider()
