import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="MLflow Model Predictor", layout="centered")
st.title("MLflow Shooting Models Predictor")

# API base
API_BASE = "http://localhost:8000"

# Model selection with types
MODEL_CHOICES = {
    "GLM (Model 1)": "model1",
    "GBM (Model 2)": "model2",
    "Stacked Ensemble (Model 3)": "model3"
}

model_label = st.selectbox("Select model:", list(MODEL_CHOICES.keys()))
model_choice = MODEL_CHOICES[model_label]
endpoint = f"{API_BASE}/predict_{model_choice}"

st.markdown("---")
st.subheader("Input Features")

# Feature input
feature1 = st.number_input("FG%", value=50.0, step=0.1)
feature2 = st.number_input("3P%", value=35.0, step=0.1)
feature3 = st.number_input("2P%", value=45.0, step=0.1)
feature4 = st.number_input("eFG%", value=52.0, step=0.1)
feature5 = st.number_input("FT%", value=80.0, step=0.1)

# Optional: batch input
st.markdown("You can also enter multiple records in a DataFrame below (optional)")
batch_input = st.text_area(
    "Paste CSV data (columns: FG%,3P%,2P%,eFG%,FT%)",
    height=150
)

if st.button("Predict"):
    try:
        if batch_input.strip():
            df_batch = pd.read_csv(pd.compat.StringIO(batch_input))
            records = df_batch.to_dict(orient="records")
        else:
            records = [{
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "feature4": feature4,
                "feature5": feature5
            }]

        payload = {"records": records}
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

        st.success("Predictions received!")
        st.json(data)

    except Exception as e:
        st.error(f"Error: {e}")
