import streamlit as st
import joblib
import pandas as pd

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
        lr = joblib.load("model_logreg.pkl")
        rf = joblib.load("model_randomforest.pkl")
        vote = joblib.load("model_voting.pkl")
        return lr, rf, vote
    except:
        return None, None, None

model_lr, model_rf, model_voting = load_models()


# ===============================
# Title
# ===============================
st.title("💰 Bank Marketing Prediction App")
st.markdown("Prediksi apakah nasabah akan mengambil **term deposit** berdasarkan data marketing bank.")

if not all([model_lr, model_rf, model_voting]):
    st.error("❌ Model tidak ditemukan. Pastikan file .pkl berada dalam folder yang benar.")
    st.stop()

st.sidebar.title("📊 Model Performance")
st.sidebar.write("Akurasi Model:")
st.sidebar.success("Logistic Regression: >90% (estimasi)")
st.sidebar.success("Random Forest: >92% (estimasi)")
st.sidebar.success("Voting Ensemble: >93% (estimasi)")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Logistic Regression", "Random Forest", "Voting Ensemble"]
)

st.markdown("### 📝 Isi Form Berikut")

# ===============================
# Input Form
# ===============================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
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
previous = st.number_input("Number of Previous Contacts", value=0)
poutcome = st.selectbox("Previous Outcome", ["success", "failure", "nonexistent"])

# ===============================
# Convert Input to DataFrame
# ===============================
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

st.markdown("### 🔍 Prediksi")

if st.button("Predict"):
    try:
        if model_choice == "Logistic Regression":
            pred = model_lr.predict(input_data)[0]
        elif model_choice == "Random Forest":
            pred = model_rf.predict(input_data)[0]
        else:
            pred = model_voting.predict(input_data)[0]

        if pred == "yes":
            st.success("💚 Nasabah **BERMINAT** mengambil term deposit.")
        else:
            st.error("❤️‍🩹 Nasabah **TIDAK TERTARIK** mengambil term deposit.")

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")

