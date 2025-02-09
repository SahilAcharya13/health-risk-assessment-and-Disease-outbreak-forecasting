import pandas as pd
import streamlit as st
import plotly.express as px
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

# Easy Questions Solutions
# 1. What is the overall distribution of diseases in the dataset?
# 2. How many male and female patients are there in the dataset?
# 3. What is the distribution of age among patients?
# 4. How many patients have a positive outcome in the dataset?
# 5. Visualize the age distribution for patients diagnosed with different diseases
# 6. Correlation between blood pressure and cholesterol levels among patients
# 7. Distribution of blood pressure between patients with positive and negative outcomes



# Plotting
st.title("Disease Visualization")

# 1. Overall distribution of diseases
disease_distribution = df['Disease'].value_counts()

# Select top 5 diseases and group the rest as 'Others'
top_5_diseases = disease_distribution.head(5)
others_count = disease_distribution.sum() - top_5_diseases.sum()
others_series = pd.Series({'Others': others_count})
top_5_with_others = pd.concat([top_5_diseases, others_series])

# Plotting
st.subheader("Overall Distribution of Diseases")

fig = px.pie(
    values=top_5_with_others.values,
    names=top_5_with_others.index,
    title="Top 5 Diseases and Others",
)
st.plotly_chart(fig)

# 2. Gender distribution
gender_distribution = df['Gender'].value_counts()
st.subheader("Gender Distribution of Patients")
fig2 = px.bar(x=gender_distribution.index, y=gender_distribution.values, labels={'x':'Gender', 'y':'Count'}, color=gender_distribution.index, title="Gender Distribution")
st.plotly_chart(fig2)


# 3. Distribution of age among patients
age_distribution = df['Age']
st.subheader("Distribution of Age Among Patients")
fig4 = px.histogram(x=age_distribution, nbins=20, title="Age Distribution")
st.plotly_chart(fig4)

# 4. Distribution of outcomes
outcome_distribution = df['Outcome Variable'].value_counts()
st.subheader("Distribution of Outcome Variable")
fig5 = px.pie(names=outcome_distribution.index, values=outcome_distribution.values, title="Outcome Distribution")
st.plotly_chart(fig5)




# 5. Age distribution for patients diagnosed with different diseases
age_disease_distribution = df.groupby('Disease')['Age'].mean()
st.subheader("Age Distribution for Patients Diagnosed with Different Diseases")
fig1 = px.bar(x=age_disease_distribution.index, y=age_disease_distribution.values, labels={'x':'Disease', 'y':'Average Age'}, title="Average Age of Patients Diagnosed with Different Diseases")
st.plotly_chart(fig1)

# 6. Correlation between blood pressure and cholesterol levels
blood_pressure_cholesterol_corr = df.groupby('Blood Pressure')['Cholesterol Level'].value_counts().unstack()
st.subheader("Correlation between Blood Pressure and Cholesterol Levels")
st.table(blood_pressure_cholesterol_corr)

# 7. Distribution of blood pressure between patients with positive and negative outcomes
blood_pressure_outcome_distribution = df.groupby('Outcome Variable')['Blood Pressure'].value_counts().unstack()
st.subheader("Distribution of Blood Pressure between Patients with Positive and Negative Outcomes")
st.table(blood_pressure_outcome_distribution)