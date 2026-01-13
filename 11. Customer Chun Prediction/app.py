import streamlit as st
import pickle
import pandas as pd
import joblib

# =========================
# LOAD MODEL PIPELINE
# =========================
model = joblib.load("best_model_pipeline.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.write("Quick prediction with smart defaults")

st.divider()

# =========================
# USER INPUTS (IMPORTANT ONLY)
# =========================
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 24)

internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)"]
)

monthly = st.slider("Monthly Charges", 20, 120, 70)

# =========================
# DEFAULT GOOD CUSTOMER VALUES
# =========================
input_df = pd.DataFrame([{
    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": tenure,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": internet,
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": contract,
    "Paperless Billing": "Yes",
    "Payment Method": payment,
    "Monthly Charges": monthly,
    "Total Charges": tenure * monthly
}])

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ Customer likely to CHURN (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Customer likely to STAY (Churn Probability: {prob:.2%})")
