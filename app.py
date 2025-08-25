import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load trained pipeline + metadata
# -------------------------------
# Debug: show where we are and list files
st.write("Working directory:", os.getcwd())
st.write("Files in script folder:", os.listdir(os.path.dirname(__file__)))

# Safe path for model file
model_path = os.path.join(os.path.dirname(__file__), "LinearRegressionModel.joblib")
saved = joblib.load(model_path)

pipe = saved["model"]
expected_columns = saved["columns"]

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
    # Create DataFrame with same columns as training
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=expected_columns
    )
    prediction = pipe.predict(input_df)[0]
    st.success(f"Estimated Selling Price: Rs.{prediction:.2f} Lakhs")




