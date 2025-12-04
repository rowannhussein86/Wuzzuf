import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ======================
# 1. Load Dataset & Model
# ======================
df = pd.read_csv('cleaned_jobs.csv')
model_data = joblib.load('enhanced_department_predictor.pkl')

# Extract model components
best_model = model_data['model']
label_encoder = model_data['label_encoder']
experience_level_mapping = model_data['experience_level_mapping']
job_type_mapping = model_data['job_type_mapping']
feature_columns = model_data['feature_columns']

# ======================
# 2. Define helper functions
# ======================
def enhanced_preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s/\-&]', ' ', text)
    text = re.sub(r'\b(\d+\.?\d*)\b', ' NUM ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features_from_title(title):
    title_lower = str(title).lower()
    features = {
        'contains_senior': 1 if 'senior' in title_lower else 0,
        'contains_junior': 1 if 'junior' in title_lower else 0,
        'contains_lead': 1 if 'lead' in title_lower else 0,
        'contains_manager': 1 if 'manager' in title_lower else 0,
        'contains_director': 1 if 'director' in title_lower else 0,
        'contains_data': 1 if 'data' in title_lower else 0,
        'contains_analyst': 1 if 'analyst' in title_lower else 0,
        'contains_engineer': 1 if 'engineer' in title_lower else 0,
        'contains_developer': 1 if 'developer' in title_lower else 0,
        'contains_marketing': 1 if 'marketing' in title_lower else 0,
        'contains_sales': 1 if 'sales' in title_lower else 0,
        'contains_finance': 1 if 'finance' in title_lower or 'financial' in title_lower else 0,
        'contains_hr': 1 if 'hr' in title_lower or 'human' in title_lower else 0,
        'title_length': len(str(title)),
        'title_word_count': len(str(title).split())
    }
    return features

def enhanced_clean_experience_year(value):
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

def get_confidence_level(probability):
    if probability > 0.7:
        return "High"
    elif probability > 0.5:
        return "Medium"
    elif probability > 0.3:
        return "Low"
    else:
        return "Very Low"

def enhanced_predict_department(job_title, categories, company, experience_year, experience_level, job_type, num_skills):
    title_cleaned = enhanced_preprocess_text(job_title)
    categories_cleaned = enhanced_preprocess_text(categories)
    company_cleaned = enhanced_preprocess_text(company)
    title_features = extract_features_from_title(job_title)
    input_data = pd.DataFrame({
        'title_text': [title_cleaned],
        'categories_text': [categories_cleaned],
        'company_text': [company_cleaned],
        'experience_year': [enhanced_clean_experience_year(experience_year)],
        'experience_level': [experience_level_mapping.get(experience_level, 1)],
        'job_type': [job_type_mapping.get(job_type, 0)],
        'num_skills': [num_skills]
    })
    for key, value in title_features.items():
        input_data[key] = [value]
    input_data['exp_x_skills'] = input_data['experience_year'] * input_data['num_skills']
    input_data['level_x_skills'] = input_data['experience_level'] * input_data['num_skills']
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]
    prediction = best_model.predict(input_data)[0]
    probabilities = best_model.predict_proba(input_data)[0]
    confidence = probabilities[prediction]
    top_n = 5
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    result = {'department': label_encoder.inverse_transform([prediction])[0],
              'confidence': float(confidence),
              'top_predictions': []}
    for idx in top_indices:
        result['top_predictions'].append({
            'department': label_encoder.inverse_transform([idx])[0],
            'probability': float(probabilities[idx]),
            'confidence_level': get_confidence_level(probabilities[idx])
        })
    return result

# ======================
# 3. Streamlit Interface
# ======================
st.title("Job Department Predictor")
st.write("Predict the most likely department for a job posting")

# Dropdown options
job_titles = sorted(df['Title'].dropna().unique())
categories = sorted(df['categories'].dropna().unique())
companies = sorted(df['company'].dropna().unique())

selected_title = st.selectbox("Select Job Title", job_titles)
selected_category = st.selectbox("Select Job Category", categories)
selected_company = st.selectbox("Select Company Name", companies)
experience_year = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
experience_level = st.selectbox("Experience Level", list(experience_level_mapping.keys()))
job_type = st.selectbox("Job Type", list(job_type_mapping.keys()))
num_skills = st.number_input("Number of Skills Required", min_value=0, max_value=50, value=5, step=1)

if st.button("Predict Department"):
    result = enhanced_predict_department(selected_title, selected_category, selected_company, experience_year, experience_level, job_type, num_skills)
    st.subheader("Prediction Result")
    st.write(f"**Predicted Department:** {result['department']}")
    st.write(f"**Confidence:** {result['confidence']:.2%} ({get_confidence_level(result['confidence'])})")
    st.write("**Top Predictions:**")
    for i, pred in enumerate(result['top_predictions'], 1):
        st.write(f"{i}. {pred['department']}: {pred['probability']:.2%} ({pred['confidence_level']})")