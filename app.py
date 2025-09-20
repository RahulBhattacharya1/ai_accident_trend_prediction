import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/accident_predictor.pkl")

st.title("🚂 Railway Accidents Predictor (EU)")

# User inputs
year = st.number_input("Enter Year", min_value=2010, max_value=2030, value=2025, step=1)
accident_type = st.selectbox("Accident Type", ["COLLIS", "DERAIL", "LEVELCROSS", "OTHERS"])
country = st.text_input("Country Code (e.g., DE, FR, IT)", "DE")

# Prepare input
input_df = pd.DataFrame([[year, accident_type, country]], columns=["date", "accident", "geography"])
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Align with training features
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[model_features]

# Prediction
prediction = model.predict(input_encoded)[0]
st.success(f"Predicted number of accidents: {prediction:.2f}")
