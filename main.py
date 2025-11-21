import streamlit as st
import joblib
import os
import sklearn
import pandas as pd

st.set_page_config(page_title="DEBUG MODE", layout="centered")

st.title("STREAMLIT DEBUG MODE — MODEL LOADER")

# ==========================================================
# SHOW ENVIRONMENT INFORMATION
# ==========================================================
st.subheader("Environment Info")

st.write("Python Version:")
st.code(os.popen("python --version").read())

st.write("Installed scikit-learn version:")
st.code(sklearn.__version__)

st.subheader("Current Working Directory")
st.code(os.getcwd())

st.subheader("Files in Directory")
st.code(os.listdir())

# ==========================================================
# TRY LOADING MODELS
# ==========================================================
def try_load(file):
    st.write(f"Testing load for: {file}")
    if file not in os.listdir():
        st.error(f"File {file} NOT FOUND in directory.")
        return None
    try:
        obj = joblib.load(file)
        st.success(f"{file} successfully loaded!")
        st.write("Type:", type(obj))
        return obj
    except Exception as e:
        st.error(f"Error while loading {file}:")
        st.exception(e)
        return None

st.subheader("Model Loading Tests")

pre = try_load("preprocessor.pkl")
lr = try_load("model_logreg.pkl")
gb = try_load("model_gb.pkl")

# ==========================================================
# OPTIONAL TEST: Transformation / Prediction
# ==========================================================
st.subheader("Testing Transformation & Prediction")

if pre is not None:
    try:
        st.write("Inspecting preprocessor expected input columns...")
        if hasattr(pre, "feature_names_in_"):
            st.code(pre.feature_names_in_)
        else:
            st.warning("Preprocessor has no attribute feature_names_in_")
    except Exception as e:
        st.error("Error inspecting preprocessor:")
        st.exception(e)

if pre is not None and lr is not None:
    try:
        st.write("Creating fake input dataframe for testing...")

        try:
            fake_cols = list(pre.feature_names_in_)
        except:
            fake_cols = ["age","job","marital","education","default",
                         "balance","housing","loan","contact","duration",
                         "day","month","campaign","pdays","previous","poutcome"]

        fake_df = pd.DataFrame([{col: 0 for col in fake_cols}])
        st.code(fake_df.head())

        st.write("Transforming fake input...")
        transformed = pre.transform(fake_df)

        st.success("Transformation SUCCESSFUL!")
        st.write("Transformed shape:", transformed.shape)

        st.write("Trying prediction with Logistic Regression...")
        pred = lr.predict(transformed)
        st.success(f"Prediction: {pred}")

    except Exception as e:
        st.error("Transformation / prediction FAILED:")
        st.exception(e)
else:
    st.info("Not enough components to test transformation/prediction.")
