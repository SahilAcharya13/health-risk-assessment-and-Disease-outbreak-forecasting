import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Load the dataset
data = pd.read_csv('D:/8th_Sem/streamlit/framingham.csv')

# Data Preprocessing
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Streamlit app
st.title("Heart Attack Prediction")
st.write("Enter the required details to get the prediction:")

age = st.number_input("Enter Age:")
male = st.radio("Gender:", options=["Male", "Female"])
if male == "Male":
    male = 1
else:
    male = 0

education = st.radio("Education Level:", options=["Some High School", "High School or GED", "Some College or Vocational School", "College"])
education_mapping = {"Some High School": 1, "High School or GED": 2, "Some College or Vocational School": 3, "College": 4}
education = education_mapping[education]

current_smoker = st.radio("Are you a current smoker?", options=["Yes", "No"])
if current_smoker == "Yes":
    current_smoker = 1
else:
    current_smoker = 0

cigs_per_day = st.number_input("Enter the number of cigarettes smoked per day:", value=0)
bpmeds = st.radio("Are you on blood pressure medications?", options=["Yes", "No"])
if bpmeds == "Yes":
    bpmeds = 1
else:
    bpmeds = 0

prevalent_stroke = st.radio("Do you have a prevalent stroke?", options=["Yes", "No"])
if prevalent_stroke == "Yes":
    prevalent_stroke = 1
else:
    prevalent_stroke = 0

prevalent_hyp = st.radio("Do you have prevalent hypertension?", options=["Yes", "No"])
if prevalent_hyp == "Yes":
    prevalent_hyp = 1
else:
    prevalent_hyp = 0

diabetes = st.radio("Do you have diabetes?", options=["Yes", "No"])
if diabetes == "Yes":
    diabetes = 1
else:
    diabetes = 0

tot_chol = st.number_input("Enter total cholesterol level:")
sys_bp = st.number_input("Enter systolic blood pressure:")
dia_bp = st.number_input("Enter diastolic blood pressure:")
bmi = st.number_input("Enter BMI:")
heart_rate = st.number_input("Enter heart rate:")
glucose = st.number_input("Enter glucose level:")

# Make prediction
input_data = [[male, age, education, current_smoker, cigs_per_day, bpmeds, prevalent_stroke, prevalent_hyp, diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose]]

prediction = model.predict(input_data)[0]
if st.button("Predict"):
    if prediction == 1:
        st.subheader("Based on the provided information, there's a high chance of a heart attack within the next 10 years.")
    else:
        st.subheader("Based on the provided information, there's a low chance of a heart attack within the next 10 years.")


