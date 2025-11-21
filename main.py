import streamlit as st
import joblib
import os

st.title("DEBUG MODE")

# Show working directory
st.write("Current Working Directory:", os.getcwd())
st.write("Files in this directory:", os.listdir())

# Test load models
files = ["model_logreg.pkl", "model_gb.pkl", "preprocessor.pkl"]

for f in files:
    st.write(f"Testing load for: {f}")
    try:
        model = joblib.load(f)
        st.success(f"{f} LOADED SUCCESSFULLY")
    except Exception as e:
        st.error(f"{f} FAILED: {e}")
