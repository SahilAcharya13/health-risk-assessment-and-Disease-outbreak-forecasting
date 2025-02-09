import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
def load_data():
    return pd.read_csv("stroke_data.csv")

df = load_data()

# Basic Level Analysis
st.title("Basic Data Analysis")

st.write(df.head(5))
# Average age
st.subheader("1. Average Age:")
average_age = df['age'].mean()
st.write(f"The average age of individuals is {average_age:.2f} years.")

# Count of males and females
st.subheader("2. Count of Males and Females:")
gender_counts = df['gender'].value_counts()
st.write(gender_counts)

# Percentage of individuals with hypertension
st.subheader("3. Percentage of Individuals with Hypertension:")
hypertension_percentage = (df['hypertension'].sum() / len(df)) * 100
st.write(f"{hypertension_percentage:.2f}% of individuals have hypertension.")

# Most common work type
st.subheader("4. Most Common Work Type:")
common_work_type = df['work_type'].mode()[0]
st.write(f"The most common work type is {common_work_type}.")

# Average BMI
st.subheader("5. Average BMI:")
average_bmi = df['bmi'].mean()
st.write(f"The average BMI of individuals is {average_bmi:.2f}.")

# Medium Level Analysis
st.title("Medium Data Analysis")



# Association between hypertension and heart disease
st.subheader("6. Association between Hypertension and Heart Disease:")
hypertension_heart_disease = pd.crosstab(df['hypertension'], df['heart_disease'])
st.write(hypertension_heart_disease)

# Distribution of smoking status among individuals with and without a history of stroke
st.subheader("7. Distribution of Smoking Status among Individuals with and without Stroke:")
smoking_stroke = pd.crosstab(df['stroke'], df['smoking_status'])
st.write(smoking_stroke)

# Difference in average glucose levels between smokers and non-smokers
st.subheader("8. Difference in Average Glucose Levels between Smokers and Non-Smokers:")
smokers_glucose = df[df['smoking_status'] == 'smokes']['avg_glucose_level'].mean()
non_smokers_glucose = df[df['smoking_status'] == 'never smoked']['avg_glucose_level'].mean()
st.write(f"Average glucose level of smokers: {smokers_glucose:.2f}")
st.write(f"Average glucose level of non-smokers: {non_smokers_glucose:.2f}")

# Patterns between work type and likelihood of stroke
st.subheader("9. Patterns between Work Type and Likelihood of Stroke:")
work_stroke = pd.crosstab(df['work_type'], df['stroke'])
st.write(work_stroke)


