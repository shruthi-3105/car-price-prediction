import streamlit as st
import pandas as pd
from joblib import load

# -------------------------------
# Load trained pipeline
# -------------------------------
pipe = load('LinearRegressionModel.joblib')  # Must be in same folder as app.py

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("Car Price Prediction System")
st.write("Predict the selling price of a car based on input features")

# Input fields
name = st.text_input("Car Name", "Swift")
company = st.text_input("Company Name", "Maruti")
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1, value=2019)
kms_driven = st.number_input("KMs Driven", min_value=0, step=500, value=10000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Car Price"):
    # Create DataFrame with same columns as used in training
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']  # Match training columns
    )

    # Make prediction using the trained pipeline
    prediction = pipe.predict(input_df)[0]

    st.success(f"Estimated Selling Price: {prediction:.2f} Lakhs")


