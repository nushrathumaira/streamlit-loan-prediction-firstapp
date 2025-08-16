import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model, scaler, and feature order
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🔮 Loan Default Prediction App")

# ---------------------------
# User inputs
# ---------------------------
credit_policy = st.selectbox("Credit Policy (Meets Lending Criteria?)", [1, 0])
purpose = st.selectbox("Purpose of Loan", [
    'all_other', 'credit_card', 'debt_consolidation',
    'educational', 'major_purchase', 'small_business'
])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 10.5)
installment = st.number_input("Installment Amount", value=300.0)
log_annual_inc = st.number_input("Log of Annual Income", value=10.5)
dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 18.0)
fico = st.slider("FICO Credit Score", 300, 850, 720)
days_with_cr_line = st.number_input("Days with Credit Line", value=5000)
revol_bal = st.number_input("Revolving Balance", value=10000)
revol_util = st.slider("Revolving Line Utilization Rate (%)", 0.0, 150.0, 50.0)
inq_last_6mths = st.slider("Inquiries in Last 6 Months", 0, 10, 1)
delinq_2yrs = st.slider("Delinquencies in Last 2 Years", 0, 10, 0)
pub_rec = st.slider("Number of Public Records", 0, 5, 0)

# Purpose one-hot encoding → create dummy columns consistent with training
purpose_cols = [col for col in feature_names if col.startswith("purpose_")]
purpose_data = dict.fromkeys(purpose_cols, 0)

if purpose != "all_other":  # because 'all_other' was dropped in training
    purpose_col = f"purpose_{purpose}"
    if purpose_col in purpose_data:
        purpose_data[purpose_col] = 1

# Build input dictionary
input_data = {
    "credit.policy": credit_policy,
    "int.rate": int_rate,
    "installment": installment,
    "log.annual.inc": log_annual_inc,
    "dti": dti,
    "fico": fico,
    "days.with.cr.line": days_with_cr_line,
    "revol.bal": revol_bal,
    "revol.util": revol_util,
    "inq.last.6mths": inq_last_6mths,
    "delinq.2yrs": delinq_2yrs,
    "pub.rec": pub_rec,
}
input_data.update(purpose_data)

# Convert to DataFrame and reorder columns
X_input = pd.DataFrame([input_data])[feature_names]

# Scale features
X_scaled = scaler.transform(X_input)

# Prediction
if st.button("Predict"):
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.write("Feature vector length:", X_scaled.shape[1])  # Debug

    if prediction == 1:
        st.error(f"⚠️ High Risk of Not Fully Paying (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Likely to Fully Pay (Probability: {prob:.2%})")

    """During training, you save the exact feature order.

In the app, you reconstruct the input as a DataFrame and reorder columns with feature_names.

That guarantees the model sees the exact same feature count & order as it was trained on.
    """