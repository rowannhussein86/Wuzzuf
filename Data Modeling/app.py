import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# =========================
# 1-Load dataset & model
# =========================
df_filtered = pd.read_csv('cleaned_jobs.csv')  

with open('department_predictor_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']
title_vectorizer = model_data['title_vectorizer']
categories_vectorizer = model_data['categories_vectorizer']
company_vectorizer = model_data['company_vectorizer']
experience_level_mapping = model_data['experience_level_mapping']
job_type_mapping = model_data['job_type_mapping']

# =========================
# 2-Preprocessing functions
# =========================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s/\-&]', ' ', text)
    text = re.sub(r'\b(\d+\.?\d*)\b', ' NUM ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_experience_year(value):
    if pd.isna(value) or str(value).lower() in ['unknown', 'nan', 'none']:
        return 0
    try:
        val = float(value)
        return 20 if val > 50 else val
    except:
        numbers = re.findall(r'\d+', str(value))
        if numbers:
            return float(numbers[0]) / 12 if 'month' in str(value).lower() else float(numbers[0])
        return 0

# =========================
# 3-Prediction function
# =========================
def predict_department(job_title, categories, company, experience_year, 
                       experience_level, job_type, num_skills):
    # Clean text
    title_cleaned = preprocess_text(job_title)
    categories_cleaned = preprocess_text(categories)
    company_cleaned = preprocess_text(company)

    # Encode categorical
    exp_level_encoded = experience_level_mapping.get(experience_level, 1)
    job_type_encoded = job_type_mapping.get(job_type, 0)

    # Vectorize text
    title_vec = title_vectorizer.transform([title_cleaned])
    categories_vec = categories_vectorizer.transform([categories_cleaned])
    company_vec = company_vectorizer.transform([company_cleaned])

    # Numerical features
    exp_year_cleaned = clean_experience_year(experience_year)
    numeric_features = np.array([[exp_year_cleaned, exp_level_encoded, job_type_encoded, num_skills]])

    # Combine all features
    features_combined = np.hstack([
        title_vec.toarray(),
        categories_vec.toarray(),
        company_vec.toarray(),
        numeric_features
    ])

    # Make prediction
    prediction = model.predict(features_combined)[0]
    probabilities = model.predict_proba(features_combined)[0]

    # Top 3 normalized probabilities
    top_n = 3
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_probs = probabilities[top_indices]
    normalized_probs = top_probs / top_probs.sum()

    result = {
        'department': label_encoder.inverse_transform([prediction])[0],
        'top_predictions': []
    }

    for idx, norm_prob in zip(top_indices, normalized_probs):
        result['top_predictions'].append({
            'department': label_encoder.inverse_transform([idx])[0],
            'probability': float(norm_prob)
        })

    return result

# =========================
# 4-Streamlit interface
# =========================
st.title("Job Department Predictor")
st.write("Predict the most likely department for a job posting")

# Dropdown options
job_titles = sorted(df_filtered['Title'].dropna().unique())
categories = sorted(df_filtered['categories'].dropna().unique())
companies = sorted(df_filtered['company'].dropna().unique())

selected_title = st.selectbox("Job Title", job_titles)
selected_category = st.selectbox("Job Category", categories)
selected_company = st.selectbox("Company Name", companies)
experience_year = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
experience_level = st.selectbox("Experience Level", list(experience_level_mapping.keys()))
job_type = st.selectbox("Job Type", list(job_type_mapping.keys()))
num_skills = st.number_input("Number of Skills Required", min_value=0, max_value=50, value=5, step=1)

if st.button("Predict Department"):
    result = predict_department(selected_title, selected_category, selected_company,
                                experience_year, experience_level, job_type, num_skills)
    
    st.subheader("Prediction Result")
    st.write(f"**Predicted Department :** {result['department']}")
    st.write("**Top 3 Predictions :**")
    for pred in result['top_predictions']:
        st.write(f"- {pred['department']}: {pred['probability']*100:.2f}%")
