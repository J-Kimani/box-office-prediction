# Revenue_Pred.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load model
model = joblib.load("model.pkl")

# Load encoders and feature order
with open("freq_maps.pkl", "rb") as f:
    freq_maps = pickle.load(f)

with open("target_maps.pkl", "rb") as f:
    target_maps = pickle.load(f)

with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

st.title("🎬 Movie Revenue Prediction App")
st.markdown("Enter the details of the movie below to predict its box office revenue.")

# Input fields in main layout (no sidebar)
budget = st.number_input("Budget (USD)", min_value=1000, step=1000000, value=100000000)
release_year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2023)
release_month = st.number_input("Release Month", min_value=1, max_value=12, value=7)
vote_average = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1, value=7.5)
vote_count = st.number_input("Vote Count", min_value=0, step=10, value=1200)
runtime = st.number_input("Runtime (minutes)", min_value=1, step=1, value=130)

director = st.text_input("Director", "Christopher Nolan")
lead_actor = st.text_input("Lead Actor", "Cillian Murphy")
primary_company = st.text_input("Primary Production Company", "Universal Pictures")

# Toggle to show/hide debug panel
show_debug = st.checkbox("Show Feature Vector Debug", value=False)

if st.button("Predict Revenue"):
    # Apply transformations
    log_budget = np.log1p(budget)
    director_enc = target_maps["director"].get(director, 0)
    actor_enc = target_maps["lead_actor"].get(lead_actor, 0)
    company_enc = target_maps["primary_company"].get(primary_company, 0)

    # Build input DataFrame
    X_input = pd.DataFrame([{
        "log_budget": log_budget,
        "release_year": release_year,
        "release_month": release_month,
        "vote_average": vote_average,
        "vote_count": vote_count,
        "runtime": runtime,
        "director_target_enc": director_enc,
        "lead_actor_target_enc": actor_enc,
        "primary_company_target_enc": company_enc
    }])

    # Reorder columns to match training
    X_input = X_input[feature_order]

    # 🔍 Show debug panel only if toggle is on
    if show_debug:
        st.subheader("Debug: Feature Vector")
        st.write(X_input)

    # Predict log revenue
    log_revenue_pred = model.predict(X_input.values)[0]

    # Convert back to normal revenue
    revenue_pred = np.expm1(log_revenue_pred)

    st.success(f"💰 Predicted Revenue: **${revenue_pred:,.2f}**")
