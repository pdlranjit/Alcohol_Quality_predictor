import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('my_trained_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine's chemical properties to predict its quality")

# Input fields
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.0, min_value=0.0)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.5, min_value=0.0)
    citric_acid = st.number_input("Citric Acid", value=0.2, min_value=0.0)
    residual_sugar = st.number_input("Residual Sugar", value=2.0, min_value=0.0)
    chlorides = st.number_input("Chlorides", value=0.08, min_value=0.0)
    free_sulfur = st.number_input("Free Sulfur Dioxide", value=15.0, min_value=0.0)

with col2:
    total_sulfur = st.number_input("Total Sulfur Dioxide", value=50.0, min_value=0.0)
    density = st.number_input("Density", value=0.997, min_value=0.0, format="%.4f")
    pH = st.number_input("pH", value=3.3, min_value=0.0)
    sulphates = st.number_input("Sulphates", value=0.6, min_value=0.0)
    alcohol = st.number_input("Alcohol (%)", value=10.0, min_value=0.0)

if st.button("Predict Quality", type="primary"):
    # Create input array in correct order
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol
    ]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.success(f"**Predicted Quality: {prediction} / 10**")
    
    # Show confidence
    st.write("Confidence Scores:")
    for i, prob in enumerate(probability):
        if prob > 0.1:  # only show meaningful ones
            st.write(f"Quality {i+3 if i+3 <=8 else i}: **{prob:.1%}**")