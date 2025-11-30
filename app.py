import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill the information below and the model will predict your heart disease risk.")

# Load models
decision_tree = joblib.load("decision_tree_model.joblib")
random_forest = joblib.load("random_forest_model.joblib")
xgboost_model = joblib.load("xgb_model.joblib")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("Decision Tree", "Random Forest", "XGBoost")
)

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
resting_bp = st.number_input("RestingBP", 0, 200, 120)
cholesterol = st.number_input("Cholesterol", 0, 700, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", (0,1))
max_hr = st.number_input("MaxHR", 0, 220, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

sex = st.selectbox("Sex", ("M", "F"))
chest_pain = st.selectbox("Chest Pain Type", ("ASY", "ATA", "NAP", "TA"))
resting_ecg = st.selectbox("Resting ECG", ("Normal", "ST", "LVH"))
exercise_angina = st.selectbox("Exercise Angina", ("Y", "N"))
st_slope = st.selectbox("ST Slope", ("Up", "Flat", "Down"))

# Convert categorical inputs to one-hot encoding
input_dict = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,
    "Sex_F": 1 if sex == "F" else 0,
    "Sex_M": 1 if sex == "M" else 0,
    "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
    "RestingECG_LVH": 1 if resting_ecg == "LVH" else 0,
    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
    "ExerciseAngina_N": 1 if exercise_angina == "N" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
    "ST_Slope_Down": 1 if st_slope == "Down" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0,
}

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict"):
    if model_choice == "Decision Tree":
        model = decision_tree
    elif model_choice == "Random Forest":
        model = random_forest
    else:
        model = xgboost_model

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("💚 Low risk of heart disease")
