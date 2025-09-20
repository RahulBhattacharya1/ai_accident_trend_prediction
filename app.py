import json
import numpy as np
import pandas as pd
import streamlit as st

# --- Load lightweight artifacts ---
with open("models/accident_linear.json", "r") as f:
    ART = json.load(f)

FEATURES = ART["features"]
COEF = np.array(ART["coef"], dtype=float)
INTERCEPT = float(ART["intercept"])

st.title("Railway Accidents Predictor (EU)")

st.markdown(
    "Predicts annual accident counts by **year**, **accident type**, and **country code** "
    "using a linear model trained on your dataset."
)

# --- Simple inputs ---
year = st.number_input("Year", min_value=2004, max_value=2035, value=2025, step=1)
accident_type = st.text_input("Accident type (e.g., COLLIS, DERAIL, LEVELCROSS, OTHER)", "COLLIS").strip().upper()
country = st.text_input("Country code (e.g., DE, FR, IT, ES, PL, RO)", "DE").strip().upper()

# --- Build a single-row frame like training time (date, accident, geography) ---
inp = pd.DataFrame([[year, accident_type, country]], columns=["date", "accident", "geography"])

# One-hot encode the same way as training (drop_first=True)
inp_enc = pd.get_dummies(inp, drop_first=True)

# Align to training features (add missing cols as 0, preserve order)
for col in FEATURES:
    if col not in inp_enc.columns:
        inp_enc[col] = 0
inp_enc = inp_enc[FEATURES]  # exact same column order

x = inp_enc.values.astype(float).reshape(1, -1)
y_pred = float(x.dot(COF)[0] + INTERCEPT)

st.subheader("Prediction")
st.success(f"Estimated number of accidents: {y_pred:.2f}")

st.caption(
    "Note: If a new/rare accident type or country code wasn’t present in training, "
    "the model treats it like the base level because of one-hot 'drop_first' encoding."
)
