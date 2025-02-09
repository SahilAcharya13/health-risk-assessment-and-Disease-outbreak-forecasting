import pandas as pd
import streamlit as st
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
    df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

    return df

df = load_data()

# Define functions for tasks
def find_highest_lowest():
    int_columns = df.select_dtypes(include='int64').columns
    result = {}
    for col in int_columns:
        result[col] = {
            'Highest': df[col].max(),
            'Lowest': df[col].min()
        }
    return result

def count_difficulty_breathing():
    return df[(df['Age'] < 18) & (df['Difficulty Breathing'] == 'Yes')].shape[0]

def count_blood_pressure():
    return df[(df['Age'] < 18) & (df['Blood Pressure'] != 'Normal')].shape[0]

def count_cholesterol():
    return df[(df['Age'] < 18) & (df['Cholesterol Level'] != 'Normal')].shape[0]

def count_outcome_gender(gender):
    positive = df[(df['Gender'] == gender) & (df['Outcome Variable'] == 'Positive')].shape[0]
    negative = df[(df['Gender'] == gender) & (df['Outcome Variable'] == 'Negative')].shape[0]
    return positive, negative

def count_diseases():
    return df['Disease'].nunique()

# Streamlit app
st.title("Disease Analysis")
st.write(df.head(5))
# Task 1: Print highest and lowest values in integer columns
st.header("Highest and Lowest Values in Integer Columns")
highest_lowest = find_highest_lowest()
for col, values in highest_lowest.items():
    st.write(f"{col}: Highest - {values['Highest']}, Lowest - {values['Lowest']}")

# Task 2: Count records with age < 18 and Difficulty in Breathing
st.header("Count of Records with Age < 18 and Difficulty in Breathing")
count_diff_breathing = count_difficulty_breathing()
st.write(f"Records with age < 18 and Difficulty in Breathing: {count_diff_breathing}")

# Task 3: Count records with age < 18 and Blood Pressure
st.header("Count of Records with Age < 18 and Blood Pressure")
count_blood_press = count_blood_pressure()
st.write(f"Records with age < 18 and Blood Pressure: {count_blood_press}")

# Task 4: Count records with age < 18 and Cholesterol
st.header("Count of Records with Age < 18 and Cholesterol")
count_cholesterol_val = count_cholesterol()
st.write(f"Records with age < 18 and Cholesterol: {count_cholesterol_val}")

# Task 5: Count males and females with Outcome Variable as Positive and Negative
st.header("Count of Outcome Variable by Gender")
male_positive, male_negative = count_outcome_gender('Male')
female_positive, female_negative = count_outcome_gender('Female')
st.write(f"Males: Positive - {male_positive}, Negative - {male_negative}")
st.write(f"Females: Positive - {female_positive}, Negative - {female_negative}")

# Task 6: Count the number of unique diseases
st.header("Count of Unique Diseases")
num_diseases = count_diseases()
st.write(f"Number of types of diseases in the dataset: {num_diseases}")


# Get unique diseases for dropdown menu
diseases = df['Disease'].unique()

# Streamlit app
st.title("Disease Analysis")

# Dropdown to select disease
selected_disease = st.selectbox("Select Disease", diseases)

# Filter data based on selected disease
filtered_data = df[df['Disease'] == selected_disease]

# Display filtered data
st.header(f"Data for {selected_disease}")
st.write(filtered_data)