import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💰",
    layout="centered"
)

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    try:
        model_lr = joblib.load("model_logreg.pkl")
        model_gb = joblib.load("model_gb.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return model_lr, model_gb, preprocessor
    except:
        return None, None, None

model_lr, model_gb, preprocessor = load_models()

st.title("💰 Bank Marketing Prediction App")
st.markdown("Prediksi apakah nasabah akan mengambil **term deposit** berdasarkan data marketing bank.")

if model_lr is None or model_gb is None or preprocessor is None:
    st.error("❌ Model tidak ditemukan. Pastikan file .pkl berada dalam folder yang benar.")
    st.stop()

# Sidebar
st.sidebar.title("📊 Model Selection")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Logistic Regression", "Gradient Boosting", "Voting Ensemble"]
)

# ===============================
# Input Form
# ===============================
st.markdown("### 📝 Masukkan Data Nasabah")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=99, value=30)
    job = st.selectbox("Job", [
        "admin.","blue-collar","entrepreneur","housemaid","management",
        "retired","self-employed","services","student","technician",
        "unemployed","unknown"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Default Credit?", ["yes", "no"])

with col2:
    balance = st.number_input("Balance", value=0)
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    duration = st.number_input("Call Duration (sec)", value=100)

day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
month = st.selectbox("Month", [
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
])
campaign = st.number_input("Campaign Calls", value=1)
pdays = st.number_input("Days Passed After Campaign (-1 = never)", value=-1)
previous = st.number_input("Previous Contacts", value=0)
poutcome = st.selectbox("Previous Outcome", ["success", "failure", "nonexistent"])

# Convert to DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "duration": duration,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# ===============================
# Prediction
# ===============================
st.markdown("### 🔍 Prediksi")

if st.button("Predict"):
    try:
        if model_choice == "Logistic Regression":
            pred = model_lr.predict(input_data)[0]

        elif model_choice == "Gradient Boosting":
            pred = model_gb.predict(input_data)[0]

        else:
            voting_model = VotingClassifier(
                estimators=[
                    ("lr", model_lr),
                    ("gb", model_gb)
                ],
                voting="hard"
            )
            voting_model.fit([[0]], ["no"])  # dummy fit (required by VotingClassifier API)
            pred = voting_model.predict(input_data)[0]

        if pred == "yes":
            st.success("💚 Nasabah BERMINAT mengambil term deposit.")
        else:
            st.error("❤️‍🩹 Nasabah TIDAK tertarik mengambil term deposit.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
