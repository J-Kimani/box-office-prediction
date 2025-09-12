
# Revenue_Pred.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Page config
st.set_page_config(page_title="Box Office Predictor", layout="wide")

# 🎨 Add wallpaper background with CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4"); /* Example cinema wallpaper */
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.7);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

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

# Create two columns
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget (USD)", min_value=1000, step=1000000, value=100000000)
    release_year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2023)

    # Month dropdown with names but mapped to numbers
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_name = st.selectbox("Release Month", list(months.keys()), index=6)  # Default July
    release_month = months[month_name]

    vote_average = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1, value=7.5)
    vote_count = st.number_input("Vote Count", min_value=0, step=10, value=1200)
    runtime = st.number_input("Runtime (minutes)", min_value=1, step=1, value=130)

with col2:
    main_genre_list = ['Action', 'Adventure', 'Fantasy', 'Animation', 'Other', 'Drama',
                       'Thriller', 'Comedy', 'Romance', 'Crime', 'Horror']
    main_genre = st.selectbox("Main Genre", main_genre_list, index=0)

    original_language_list = ['English', 'Japanese', 'French', 'Chinese', 'Spanish',
                              'German', 'Hindi', 'Other', 'Italian']
    original_language = st.selectbox("Original Language", original_language_list, index=0)

    primary_country_list = ['United States of America', 'United Kingdom', 'Other', 'New Zealand', 'China',
                            'Canada', 'Germany', 'Japan', 'France', 'Australia', 'Italy', 'Spain',
                            'India', 'Unknown', 'Hong Kong', 'Mexico']
    primary_country = st.selectbox("Primary Country", primary_country_list, index=0)

    director_list = ['Other', 'Sam Raimi', 'Ridley Scott', 'Tim Burton', 'Michael Bay',
                     'Steven Spielberg', 'Robert Zemeckis', 'Martin Scorsese', 'Oliver Stone',
                     'Shawn Levy', 'Ron Howard', 'Richard Donner', 'Chris Columbus',
                     'Joel Schumacher', 'Steven Soderbergh', 'Tony Scott', 'Renny Harlin',
                     'Brian De Palma', 'Paul W.S. Anderson', 'Barry Levinson', 'Bobby Farrelly',
                     'Clint Eastwood', 'Robert Rodriguez', 'Rob Reiner', 'Joel Coen',
                     'Francis Ford Coppola', 'Spike Lee', 'John Carpenter', 'Kevin Smith',
                     'Woody Allen', 'Richard Linklater']
    director = st.selectbox("Director", director_list, index=0)

    lead_actor_list = ['Other', 'Johnny Depp', 'Christian Bale', 'Ben Affleck', 'Mark Wahlberg',
                       'Tom Hanks', 'Brad Pitt', 'Leonardo DiCaprio', 'Harrison Ford', 'Tom Cruise',
                       'Kevin Costner', 'Keanu Reeves', 'Arnold Schwarzenegger', 'Nicolas Cage',
                       'Ben Stiller', 'John Travolta', 'Bruce Willis', 'Jim Carrey',
                       'Dwayne Johnson', 'Matt Damon', 'Will Ferrell', 'George Clooney',
                       'Sandra Bullock', 'Denzel Washington', 'Robert De Niro',
                       'Sylvester Stallone', 'Eddie Murphy', 'Adam Sandler', 'Robin Williams',
                       'Meryl Streep', 'Sean Connery']
    lead_actor = st.selectbox("Lead Actor", lead_actor_list, index=0)

    primary_company_list = ['Other', 'Walt Disney Pictures', 'Columbia Pictures', 'Warner Bros.',
                            'Paramount Pictures', 'New Line Cinema', 'Universal Pictures',
                            'Twentieth Century Fox Film Corporation', 'Village Roadshow Pictures',
                            'DreamWorks SKG', 'Summit Entertainment', 'Regency Enterprises', 'Lionsgate',
                            'Columbia Pictures Corporation', 'TriStar Pictures', 'Touchstone Pictures',
                            'Miramax Films', 'United Artists', 'The Weinstein Company',
                            'Metro-Goldwyn-Mayer (MGM)', 'Unknown', 'Fox Searchlight Pictures']
    primary_company = st.selectbox("Primary Production Company", primary_company_list, index=0)

# Debug toggle
show_debug = st.checkbox("Show Feature Vector Debug", value=False)

if st.button("Predict Revenue"):
    # Apply transformations
    log_budget = np.log1p(budget)
    director_enc = target_maps["director"].get(director, 0)
    actor_enc = target_maps["lead_actor"].get(lead_actor, 0)
    company_enc = target_maps["primary_company"].get(primary_company, 0)
    genre_enc = freq_maps["main_genre"].get(main_genre, 0)
    language_enc = freq_maps["original_language"].get(original_language, 0)
    country_enc = freq_maps["primary_country"].get(primary_country, 0)

    # Build input DataFrame
    X_input = pd.DataFrame([{
        "log_budget": log_budget,
        "release_year": release_year,
        "release_month": release_month,
        "vote_average": vote_average,
        "vote_count": vote_count,
        "runtime": runtime,
        "main_genre_freq_enc": genre_enc,
        "original_language_freq_enc": language_enc,
        "primary_country_freq_enc": country_enc,
        "director_target_enc": director_enc,
        "lead_actor_target_enc": actor_enc,
        "primary_company_target_enc": company_enc
    }])

    # Reorder columns to match training
    X_input = X_input[feature_order]

    # 🔍 Debug panel
    if show_debug:
        st.subheader("Debug: Feature Vector")
        st.write(X_input)

    # Predict
    log_revenue_pred = model.predict(X_input.values)[0]
    revenue_pred = np.expm1(log_revenue_pred)

    st.success(f"💰 Predicted Revenue: **${revenue_pred:,.2f}**")
