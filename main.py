import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Bank Marketing Prediction", layout="centered")

st.title("💰 Bank Term Deposit Prediction")
st.write("Aplikasi ini memprediksi apakah nasabah akan mengambil term deposit.")

# ==========================================================
# LOAD MODELS & PREPROCESSOR FILES
# ==========================================================
@st.cache_resource
def load_all_files():
    try:
        model_lr = joblib.load("model_logreg.pkl")
        model_gb = joblib.load("model_gb.pkl")

        with open("columns.json", "r") as f:
            columns = json.load(f)

        scaler_mean = np.load("scaler_mean.npy")
        scaler_std  = np.load("scaler_std.npy")

        return model_lr, model_gb, columns, scaler_mean, scaler_std

    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None


model_lr, model_gb, columns, scaler_mean, scaler_std = load_all_files()

if any(v is None for v in [model_lr, model_gb, columns, scaler_mean, scaler_std]):
    st.error("❌ File model atau preprocessor tidak ditemukan.")
    st.stop()
else:
    st.success("✅ Semua model dan preprocessor berhasil dimuat!")


# ==========================================================
# USER INPUT FORM
# ==========================================================
st.header("📝 Input Data Nasabah")

# daftar feature asli dari dataset
original_features = [
    "age","job","marital","education","default","balance","housing","loan",
    "contact","day","month","duration","campaign","pdays","previous","poutcome"
]

user_input = {}

col1, col2 = st.columns(2)

with col1:
    user_input["age"] = st.number_input("Age", 18, 100, 30)
    user_input["job"] = st.selectbox("Job", ["admin.","blue-collar","entrepreneur","housemaid",
                                             "management","retired","self-employed","services",
                                             "student","technician","unemployed","unknown"])
    user_input["marital"] = st.selectbox("Marital Status", ["married","single","divorced"])
    user_input["education"] = st.selectbox("Education", ["primary","secondary","tertiary","unknown"])
    user_input["default"] = st.selectbox("Default Credit?", ["yes","no"])
    user_input["balance"] = st.number_input("Balance", value=0)

with col2:
    user_input["housing"] = st.selectbox("Housing Loan?", ["yes","no"])
    user_input["loan"] = st.selectbox("Personal Loan?", ["yes","no"])
    user_input["contact"] = st.selectbox("Contact Type", ["cellular","telephone","unknown"])
    user_input["day"] = st.number_input("Last Contact Day", 1, 31, 15)
    user_input["month"] = st.selectbox("Month", 
                                       ["jan","feb","mar","apr","may","jun","jul","aug",
                                        "sep","oct","nov","dec"])
    user_input["duration"] = st.number_input("Call Duration (sec)", value=100)
    user_input["campaign"] = st.number_input("Campaign Calls", value=1)
    user_input["pdays"] = st.number_input("Days Passed After Campaign (-1 = never)", value=-1)
    user_input["previous"] = st.number_input("Previous Contacts", value=0)
    user_input["poutcome"] = st.selectbox("Previous Outcome", ["success","failure","nonexistent"])


# ==========================================================
# PREDICT BUTTON
# ==========================================================
if st.button("🔍 Prediksi"):

    # =======================
    # 1. Buat DataFrame input
    # =======================
    df_user = pd.DataFrame([user_input])

    # =======================
    # 2. One-hot encoding manual
    # =======================
    df_encoded = pd.get_dummies(df_user)

    # pastikan semua kolom sama dengan saat training
    for col in columns:
        if col not in df_encoded:
            df_encoded[col] = 0

    # urutkan kolom
    df_encoded = df_encoded[columns]

    # =======================
    # 3. Scaling manual
    # =======================
    X = (df_encoded - scaler_mean) / scaler_std

    # =======================
    # 4. Predict
    # =======================
    pred_lr = model_lr.predict(X)[0]
    pred_gb = model_gb.predict(X)[0]

    # voting ensemble
    votes = [pred_lr, pred_gb]
    pred_final = 1 if votes.count(1) > votes.count(0) else 0

    # =======================
    # 5. Show Result
    # =======================
    st.subheader("📊 Hasil Prediksi:")

    result_text = "YES (Nasabah berminat)" if pred_final == 1 else "NO (Nasabah tidak berminat)"

    st.write(f"**Final Prediction (Voting):** {result_text}")
    st.write(f"- Logistic Regression: {pred_lr}")
    st.write(f"- Gradient Boosting: {pred_gb}")

    if pred_final == 1:
        st.success("💚 Nasabah kemungkinan besar BERMINAT mengambil term deposit.")
    else:
        st.error("❤️‍🩹 Nasabah kemungkinan TIDAK berminat mengambil term deposit.")
