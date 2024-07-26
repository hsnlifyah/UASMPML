import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load and preprocess data
df = pd.read_csv("C:\\Users\\ASUS\\Documents\\Semester 4\\MPML_UAS\\FinalExam\\FinalExam\\onlinefoods.csv")

# Load model and preprocessor
model = joblib.load("C:\\Users\\ASUS\\Documents\\Semester 4\\MPML_UAS\\FinalExam\\FinalExam\\random_forest_model.pkl")
preprocessor = joblib.load("C:\\Users\\ASUS\\Documents\\Semester 4\\MPML_UAS\\FinalExam\\FinalExam\\preprocessor.pkl")

# Input form for user
st.title('Prediksi Output untuk Online Foods')

# Input features
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.selectbox('Occupation', ['Employee', 'House wife', 'Self Employeed', 'Student'])
monthly_income = st.selectbox('Monthly Income', ['Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000', 'No Income'])
educational_qualifications = st.selectbox('Educational Qualifications', ['School', 'Graduate', 'Post Graduate', 'Ph.D', 'Uneducated'])
feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
age = st.number_input('Age', min_value=0)
family_size = st.number_input('Family Size', min_value=0)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')

# Create DataFrame from inputs
user_input = pd.DataFrame({
    'Gender': [gender],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Monthly Income': [monthly_income],
    'Educational Qualifications': [educational_qualifications],
    'Feedback': [feedback],
    'Age': [age],
    'Family size': [family_size],
    'latitude': [latitude],
    'longitude': [longitude]
})

# Button to make prediction
if st.button('Predict'):
    try:
        # Apply preprocessing
        user_input_processed = preprocessor.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_processed)
        prediction_proba = model.predict_proba(user_input_processed)
        
        # Display prediction
        st.write('### Hasil Prediksi')
        st.write(f'Output Prediksi: {prediction[0]}')
        st.write(f'Probabilitas Prediksi: {prediction_proba[0]}')
    except ValueError as e:
        st.error(f"Error during preprocessing: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
