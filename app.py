import streamlit as st
import numpy as np
import joblib
import pandas as pd

MODEL_PATH = r"D:\FREELANCE_PROJECTS\diabetes-client-readmit-prediction\models\rf__model.joblib"
model = joblib.load(MODEL_PATH)

ordinal_mappings = {
    'age': {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }
}

st.sidebar.header("Patient Data")

def user_input_features():
    st.sidebar.subheader("Demographics")
    age = st.sidebar.selectbox("Age Range", list(ordinal_mappings['age'].keys()), index=5)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)

    st.sidebar.subheader("Hospital Stay & Visits")
    time_in_hospital = st.sidebar.number_input("Time in Hospital (days)", min_value=0, value=3, step=1)
    num_lab_procedures = st.sidebar.number_input("Number of Lab Procedures", min_value=0, value=41, step=1)
    num_procedures = st.sidebar.number_input("Number of Procedures", min_value=0, value=0, step=1)

    st.sidebar.subheader("Medication & Diagnoses")
    num_medications = st.sidebar.number_input("Number of Medications", min_value=0, value=5, step=1)
    number_diagnoses = st.sidebar.number_input("Number of Diagnoses", min_value=0, value=1, step=1)

    st.sidebar.subheader("Visit History")
    number_outpatient = st.sidebar.number_input("Number of Outpatient Visits", min_value=0, value=0, step=1)
    number_emergency = st.sidebar.number_input("Number of Emergency Visits", min_value=0, value=0, step=1)
    number_inpatient = st.sidebar.number_input("Number of Inpatient Visits", min_value=0, value=0, step=1)

    st.sidebar.subheader("Indicators")
    change = st.sidebar.selectbox("Change in Medications", ["No", "Ch"], index=0)

    return {
        'age': age,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'change': change
    }

input_data = pd.DataFrame([user_input_features()])

input_data['age'] = input_data['age'].map(ordinal_mappings['age'])
input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == "Male" else 0)
input_data['change'] = input_data['change'].apply(lambda x: 1 if x == "Ch" else 0)

st.title("Diabetes Patient Readmission Prediction")

if st.sidebar.button("Predict"):
    if model:
        prediction = model.predict(input_data)
        result = "Likely to be Readmitted" if prediction[0] == 1 else "Not Likely to be Readmitted"

        st.write("### Prediction Result")
        st.success(result)
    else:
        st.error("Model not loaded. Please check the file path and try again.")
