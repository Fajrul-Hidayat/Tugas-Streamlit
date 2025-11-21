import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💰",
    layout="centered"
)

# ===============================
# Load Models + Preprocessor
# ===============================
@st.cache_resource
def load_all():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        model_lr = joblib.load("model_logreg.pkl")
        model_gb = joblib.load("model_gb.pkl")
        return preprocessor, model_lr, model_gb
    except Exception as e:
        return None, None, None

preprocessor, model_lr, model_gb = load_all()

st.title("💰 Bank Marketing Prediction App")
st.markdown("Prediksi apakah nasabah akan mengambil **term deposit** berdasarkan data marketing bank.")

# If models are missing
if preprocessor is None or model_lr is None or model_gb is None:
    st.error("❌ Model tidak ditemukan. Pastikan file .pkl berada di folder root GitHub.")
    st.stop()

# ===============================
# Sidebar Model Selection
# ===============================
st.sidebar.title("📊 Pilih Model")
model_choice = st.sidebar.selectbox(
    "Metode Prediksi",
    ["Logistic Regression", "Gradient Boosting", "Voting Ensemble"]
)

# ===============================
# User Input Form
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
input_df = pd.DataFrame([{
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
        # Transform input using preprocessor
        transformed = preprocessor.transform(input_df)

        # Logistic Regression
        if model_choice == "Logistic Regression":
            pred = model_lr.predict(transformed)[0]

        # Gradient Boosting
        elif model_choice == "Gradient Boosting":
            pred = model_gb.predict(transformed)[0]

        # Voting (manual)
        else:
            pred_lr = model_lr.predict(transformed)[0]
            pred_gb = model_gb.predict(transformed)[0]
            votes = [pred_lr, pred_gb]
            pred = max(set(votes), key=votes.count)

        # Output
        if pred == "yes":
            st.success("💚 Nasabah kemungkinan **BERMINAT** mengambil term deposit.")
        else:
            st.error("❤️‍🩹 Nasabah kemungkinan **TIDAK tertarik** mengambil term deposit.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
