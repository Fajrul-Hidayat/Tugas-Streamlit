import streamlit as st
import joblib
import os
import sklearn
import pandas as pd

st.set_page_config(page_title="DEBUG MODE", page_icon="🧪", layout="centered")

st.title("🧪 STREAMLIT DEBUG MODE — MODEL LOADER")

# ==========================================================
# SHOW ENVIRONMENT INFORMATION
# ==========================================================
st.subheader("📌 Environment Info")

st.write("**Python Version:**")
st.code(os.popen("python --version").read())

st.write("**Installed scikit-learn version:**")
st.code(sklearn.__version__)

st.subheader("📁 Current Working Directory")
st.code(os.getcwd())

st.subheader("📂 Files in Dir
