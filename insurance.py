import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import date

# 1. Load Model and Columns (cached)
@st.cache_resource
def load_model():
    try:
        with open('new_insurance_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('expected_col.pkl', 'rb') as f:
            expected_columns = pickle.load(f)
        return model, expected_columns
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model, expected_columns = load_model()

# 2. App Title
st.title("🏥 Insurance Premium Predictor")
st.markdown("Predict customer insurance premiums using a trained LightGBM model.")

# 3. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        health_score = st.slider("Health Score (0-100)", 0, 100, 75)
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])

    with col2:
        credit_score = st.slider("Credit Score (300-850)", 300, 850, 700)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
        insurance_duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=30, value=3)

    exercise_frequency = st.slider("Exercise Frequency (days/week)", 0, 7, 3)
    previous_claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=1)
    policy_start = st.date_input("Policy Start Date", value=date(2023, 1, 1))

    submitted = st.form_submit_button("Predict Premium")

# 4. Prediction Logic
if submitted:
    try:
        # Build input DataFrame
        input_data = pd.DataFrame([{
            "Age": age,
            "Annual Income": annual_income,
            "Health Score": health_score,
            "Credit Score": credit_score,
            "Vehicle Age": vehicle_age,
            "Insurance Duration": insurance_duration,
            "Exercise Frequency": exercise_frequency,
            "Previous Claims": previous_claims,
            "Policy Start Date": policy_start,
            "Marital Status": marital_status,
            "Policy Type": policy_type
        }])

        # Feature Engineering
        input_data['Policy_Age_Days'] = (pd.Timestamp.now() - pd.to_datetime(input_data['Policy Start Date'])).dt.days
        input_data.drop(columns=["Policy Start Date"], inplace=True)

        # One-hot encoding
        input_data = pd.get_dummies(input_data, columns=["Marital Status", "Policy Type"])

        # Add missing columns and ensure correct order
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_columns]

        # Predict
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success(f"### Predicted Premium: ${prediction:,.2f}")
        st.metric("Confidence Range", f"${prediction*0.9:,.0f} - ${prediction*1.1:,.0f}", delta="±10%")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# 5. Sidebar Information
with st.sidebar:
    st.header("📌 About")
    st.write("This app uses a LightGBM model trained on historical insurance data to predict customer premiums.")
    st.subheader("Model Validation Metrics")
    st.metric("Validation RMSE", "$845.46")
    st.metric("Validation R²", "0.0435")

    if st.button("🔄 Refresh Model"):
        st.cache_resource.clear()
        st.rerun()
