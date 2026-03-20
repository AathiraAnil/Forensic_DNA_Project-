import os
# This MUST be the first line to prevent the recursion loop
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# --- THE CLEAN LOAD ---
try:
    # 1. Load the Random Forest model
    rf_model = joblib.load("random_forest_model.pkl")
    
    # 2. Load the NEW .keras file you made in Colab
    # IMPORTANT: Change 'dl_model.h5' to 'dl_model.keras' here!
    dl_model = tf.keras.models.load_model("dl_model.keras")
    
    st.sidebar.success("✅ Models Loaded Successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Check if you deleted the .h5 and uploaded the .keras file to GitHub.")
