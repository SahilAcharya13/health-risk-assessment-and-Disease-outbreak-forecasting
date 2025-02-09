import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
""", unsafe_allow_html=True)

# Load the provided data
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Data Preprocessing
# Assuming 'diabetes' column is the target variable
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'smoking_history']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit Web Application
st.title("Diabetes Prediction")

st.write("Enter patient details to predict whether the patient is diabetic or not:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=150, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "current", "No Info"])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=6.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)

# Preprocess user input
gender_encoded = label_encoders['gender'].transform([gender])[0]
smoking_history_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

user_input = pd.DataFrame({
    'gender': [gender_encoded],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history_encoded],
    'bmi': [bmi],
    'HbA1c_level': [hba1c_level],  # Ensure consistency in feature names
    'blood_glucose_level': [blood_glucose_level]
})

if st.button("Predict"):
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        st.write("The patient is predicted to be diabetic.")
    else:
        st.write("The patient is predicted not to be diabetic.")
