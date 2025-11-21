import streamlit as st
import os
import json
import numpy as np
import joblib
import pandas as pd
import sklearn
import sys
import traceback

st.set_page_config(page_title="DEBUG MODE — Model Loader", layout="wide")

st.title("🛠 DEBUG MODE — Manual Preprocessor & Model Loader")

# ==========================================================
# 1. Environment Info
# ==========================================================
st.header("📌 Environment Info")

st.write("**Python Version:**", sys.version)
st.write("**NumPy Version:**", np.__version__)
st.write("**Pandas Version:**", pd.__version__)
st.write("**Scikit-learn Version:**", sklearn.__version__)

# ==========================================================
# 2. List Files
# ==========================================================
st.header("📂 Files in Working Directory")

files = os.listdir(".")
st.json(files)

required_files = [
    "model_logreg.pkl",
    "model_gb.pkl",
    "columns.json",
    "scaler_mean.npy",
    "scaler_std.npy"
]

st.write("### Expected Files:")
st.json(required_files)

missing = [f for f in required_files if f not in files]

if missing:
    st.error("❌ Missing files:")
    st.json(missing)
else:
    st.success("✅ All required files found!")

# ==========================================================
# 3. File Loader Function
# ==========================================================
def try_load_file(path, loader):
    st.subheader(f"Testing load for: **{path}**")
    try:
        obj = loader(path)
        st.success(f"✅ Loaded successfully: {path}")
        return obj
    except Exception as e:
        st.error(f"❌ Error loading {path}: {e}")
        st.code(traceback.format_exc())
        return None

# ==========================================================
# 4. Test loading each file
# ==========================================================
st.header("🔍 TEST LOADING FILES")

# Load JSON
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

columns = try_load_file("columns.json", load_json)

# Load NumPy arrays
mean_vals = try_load_file("scaler_mean.npy", np.load)
std_vals  = try_load_file("scaler_std.npy", np.load)

# Load model files
model_lr = try_load_file("model_logreg.pkl", joblib.load)
model_gb = try_load_file("model_gb.pkl", joblib.load)

# ==========================================================
# 5. Display Loaded Objects
# ==========================================================
st.header("📦 LOADED OBJECTS SUMMARY")

loaded = {
    "columns.json loaded": columns is not None,
    "scaler_mean.npy loaded": mean_vals is not None,
    "scaler_std.npy loaded": std_vals is not None,
    "model_logreg.pkl loaded": model_lr is not None,
    "model_gb.pkl loaded": model_gb is not None,
}

st.json(loaded)

if all(loaded.values()):
    st.success("🎉 ALL FILES SUCCESSFULLY LOADED — READY FOR main.py!")
else:
    st.error("⚠️ Some files FAILED to load. Please fix the above errors.")
