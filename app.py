import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("C:/Users/hsgee/Downloads/Telco-Customer-Churn.csv")

# Cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

# Encoding
le_dict = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Features
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# STREAMLIT UI
# =========================
st.title("📊 Customer Churn Prediction App")

# 🔹 Basic Inputs
gender = st.selectbox("Gender", le_dict["gender"].classes_)
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", le_dict["Partner"].classes_)
dependents = st.selectbox("Dependents", le_dict["Dependents"].classes_)
tenure = st.slider("Tenure", 0, 72, 12)
phone = st.selectbox("Phone Service", le_dict["PhoneService"].classes_)
internet = st.selectbox("Internet Service", le_dict["InternetService"].classes_)
contract = st.selectbox("Contract", le_dict["Contract"].classes_)

# 🔹 Extra Inputs
multiple = st.selectbox("Multiple Lines", le_dict["MultipleLines"].classes_)
online_sec = st.selectbox("Online Security", le_dict["OnlineSecurity"].classes_)
online_backup = st.selectbox("Online Backup", le_dict["OnlineBackup"].classes_)
device = st.selectbox("Device Protection", le_dict["DeviceProtection"].classes_)
tech = st.selectbox("Tech Support", le_dict["TechSupport"].classes_)
tv = st.selectbox("Streaming TV", le_dict["StreamingTV"].classes_)
movies = st.selectbox("Streaming Movies", le_dict["StreamingMovies"].classes_)
paperless = st.selectbox("Paperless Billing", le_dict["PaperlessBilling"].classes_)
payment = st.selectbox("Payment Method", le_dict["PaymentMethod"].classes_)

# 🔹 Charges
monthly = st.number_input("Monthly Charges", 0, 10000, 3000)
total = st.number_input("Total Charges", 0, 100000, 5000)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    input_dict = {
        "gender": le_dict["gender"].transform([gender])[0],
        "SeniorCitizen": senior,
        "Partner": le_dict["Partner"].transform([partner])[0],
        "Dependents": le_dict["Dependents"].transform([dependents])[0],
        "tenure": tenure,
        "PhoneService": le_dict["PhoneService"].transform([phone])[0],
        "MultipleLines": le_dict["MultipleLines"].transform([multiple])[0],
        "InternetService": le_dict["InternetService"].transform([internet])[0],
        "OnlineSecurity": le_dict["OnlineSecurity"].transform([online_sec])[0],
        "OnlineBackup": le_dict["OnlineBackup"].transform([online_backup])[0],
        "DeviceProtection": le_dict["DeviceProtection"].transform([device])[0],
        "TechSupport": le_dict["TechSupport"].transform([tech])[0],
        "StreamingTV": le_dict["StreamingTV"].transform([tv])[0],
        "StreamingMovies": le_dict["StreamingMovies"].transform([movies])[0],
        "Contract": le_dict["Contract"].transform([contract])[0],
        "PaperlessBilling": le_dict["PaperlessBilling"].transform([paperless])[0],
        "PaymentMethod": le_dict["PaymentMethod"].transform([payment])[0],
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X.columns)  # ensures correct order

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will NOT churn")
