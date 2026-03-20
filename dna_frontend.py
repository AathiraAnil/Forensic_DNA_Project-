import os
# This MUST be the first line
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- HELPER TO FIND FILES ---
def find_model(filename):
    paths = [
        os.path.join(BASE_DIR, "models", filename),
        os.path.join(BASE_DIR, "Models", filename),
        os.path.join(BASE_DIR, filename)
    ]
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

# --- LOAD MODELS ---
rf_model = None
rf_path = find_model("random_forest_model.pkl")
if rf_path:
    try:
        rf_model = joblib.load(rf_path)
    except Exception as e:
        st.error(f"⚠️ Random Forest Load Error: {e}")
        st.info("This is likely a version mismatch. Try re-saving your model on your laptop.")
else:
    st.error("❌ random_forest_model.pkl not found!")

dl_model = None
dl_path = find_model("dl_model.h5")
if dl_path:
    try:
        dl_model = tf.keras.models.load_model(dl_path, compile=False)
    except Exception as e:
        st.error(f"⚠️ Deep Learning Load Error: {e}")
else:
    st.error("❌ dl_model.h5 not found!")

# --- UI ---
st.title("🔬 DNA Phenotyping")

if rf_model and dl_model:
    st.success("✅ All models loaded successfully!")
    # ... rest of your SNP input and prediction logic here ...
else:
    st.warning("The app cannot predict until the models above are fixed.")
