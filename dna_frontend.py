import os
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic DNA Phenotyping", layout="centered")
st.title("🔬 Forensic DNA Phenotyping")
st.write("### Multi-Color Iris Probability Analysis")
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
st.subheader("Enter Genotype Values")
st.info("Genotypes: 0 = Homozygous Recessive, 1 = Heterozygous, 2 = Homozygous Dominant")

col1, col2 = st.columns(2)
with col1:
    s1 = st.selectbox("SNP 1", [0, 1, 2], help="The primary HERC2/OCA2 marker")
    s2 = st.selectbox("SNP 2", [0, 1, 2])
with col2:
    s3 = st.selectbox("SNP 3", [0, 1, 2])
    s4 = st.selectbox("SNP 4", [0, 1, 2])

# --- PREDICTION LOGIC ---
if st.button("Analyze DNA Profile", type="primary"):
    if rf_model is not None:
        try:
            # 1. Prepare input for the model
            # We use a 2D array [ [s1, s2, s3, s4] ]
            rf_input = np.array([[s1, s2, s3, s4]])
            
            # 2. Get the specific probabilities for EACH color
            probs = rf_model.predict_proba(rf_input)[0]
            classes = rf_model.classes_
            
            # 3. Create a DataFrame for the Chart
            # This shows Blue, Brown, and Green simultaneously
            prob_df = pd.DataFrame({
                'Phenotype': classes,
                'Probability (%)': probs * 100
            }).set_index('Phenotype')

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("📊 Probability Distribution")
            
            # Show the Bar Chart for all 3 colors
            st.bar_chart(prob_df)

            # Highlight the most likely color
            winner = classes[np.argmax(probs)]
            confidence = np.max(probs) * 100
            
            st.success(f"**Predicted Primary Phenotype: {winner} ({confidence:.2f}% Confidence)**")

            # --- IMAGE DISPLAY ---
            color_key = str(winner).lower()
            if "brown" in color_key:
                img_file = "brown_eye.png"
            elif "green" in color_key:
                img_file = "green_eye.png"
            else:
                img_file = "blue_eye.png"
                
            if os.path.exists(img_file):
                st.image(img_file, width=350, caption=f"Visualized Result: {winner}")
            else:
                st.warning(f"⚠️ Image '{img_file}' not found in GitHub.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
    else:
        st.error("Model file missing. Check your GitHub repository.")

st.divider()
