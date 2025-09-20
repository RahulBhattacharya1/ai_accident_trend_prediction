import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("models/accident_linear.json")

st.title("Railway Accidents Predictor (EU)")
st.write(
    "Predict annual accident counts by year, accident type, and country code using a "
    "simple linear model trained on your dataset."
)

# ---- Load lightweight artifacts (no pickles/joblib) ----
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at {MODEL_PATH}. "
        "Upload 'accident_linear.json' to the models/ folder in your repo."
    )
    st.stop()

try:
    with open(MODEL_PATH, "r") as f:
        ART = json.load(f)
    FEATURES = ART["features"]
    COEF = np.array(ART["coef"], dtype=float)   # <-- make sure this name is COEF
    INTERCEPT = float(ART["intercept"])
except Exception as e:
    st.error(f"Failed to read model artifacts: {e}")
    st.stop()

# Basic validation
if len(FEATURES) != len(COEF):
    st.error("Artifact mismatch: number of features does not match number of coefficients.")
    st.stop()

# ---- Inputs ----
year = st.number_input("Year", min_value=2004, max_value=2035, value=2025, step=1)
accident_type = st.text_input(
    "Accident type (e.g., COLLIS, DERAIL, LEVELCROSS, OTHER)", "COLLIS"
).strip().upper()
country = st.text_input(
    "Country code (e.g., DE, FR, IT, ES, PL, RO)", "DE"
).strip().upper()

# ---- Build encoded row exactly like training (one-hot with drop_first=True) ----
inp = pd.DataFrame([[year, accident_type, country]], columns=["date", "accident", "geography"])
inp_enc = pd.get_dummies(inp, drop_first=True)

# Align to training features (add missing cols as 0, keep order)
for col in FEATURES:
    if col not in inp_enc.columns:
        inp_enc[col] = 0
inp_enc = inp_enc[FEATURES].astype(float)

# ---- Predict ----
x = inp_enc.to_numpy().reshape(1, -1)
y_pred = float(x.dot(COEF)[0] + INTERCEPT)  # <-- use COEF here

st.subheader("Prediction")
st.success(f"Estimated number of accidents: {y_pred:.2f}")

# Optional: show the encoded feature vector for debugging
with st.expander("Show encoded features (debug)"):
    st.write(pd.DataFrame([COEF], columns=FEATURES, index=["coef"]).T.head(20))
    st.write("Input encoded row:")
    st.write(inp_enc.head(1))
