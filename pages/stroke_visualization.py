import streamlit as st
import pandas as pd
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
    return pd.read_csv("stroke_data.csv")

df = load_data()

# Medium Level Questions and Graphs
st.title("Data Visualization")

# 1. Correlation between age and average glucose level
st.subheader("Correlation between Age and Average Glucose Level")
fig_correlation = px.scatter(df, x='age', y='avg_glucose_level', trendline='ols', title='Correlation between Age and Average Glucose Level')
st.plotly_chart(fig_correlation)

# 2. Distribution of smoking status among individuals with and without a history of stroke
st.subheader("Distribution of Smoking Status among Individuals with and without a History of Stroke")
smoking_stroke = df.groupby(['stroke', 'smoking_status']).size().reset_index(name='count')
fig_smoking_stroke = px.bar(smoking_stroke, x='smoking_status', y='count', color='stroke', barmode='group', title='Distribution of Smoking Status among Individuals with and without Stroke', labels={'count': 'Count', 'smoking_status': 'Smoking Status', 'stroke': 'Stroke Status'})
st.plotly_chart(fig_smoking_stroke)

# 3. Difference in average glucose levels between smokers and non-smokers
st.subheader("Difference in Average Glucose Levels between Smokers and Non-Smokers")
smokers_glucose = df[df['smoking_status'] == 'smokes']['avg_glucose_level'].mean()
non_smokers_glucose = df[df['smoking_status'] == 'never smoked']['avg_glucose_level'].mean()
fig_diff_glucose = px.bar(x=['Smokers', 'Non-Smokers'], y=[smokers_glucose, non_smokers_glucose], title='Difference in Average Glucose Levels between Smokers and Non-Smokers', labels={'x': 'Smoking Status', 'y': 'Average Glucose Level'})
st.plotly_chart(fig_diff_glucose)

# 4. Association between hypertension and heart disease
st.subheader("Association between Hypertension and Heart Disease")
hypertension_heart_disease = df.groupby(['hypertension', 'heart_disease']).size().reset_index(name='count')
fig_hypertension_heart_disease = px.bar(hypertension_heart_disease, x='hypertension', y='count', color='heart_disease', barmode='group', title='Association between Hypertension and Heart Disease', labels={'count': 'Count', 'hypertension': 'Hypertension', 'heart_disease': 'Heart Disease'})
st.plotly_chart(fig_hypertension_heart_disease)

# 5. Patterns between work type and likelihood of stroke
st.subheader("Patterns between Work Type and Likelihood of Stroke")
work_stroke = df.groupby(['work_type', 'stroke']).size().reset_index(name='count')
fig_work_stroke = px.bar(work_stroke, x='work_type', y='count', color='stroke', barmode='group', title='Patterns between Work Type and Likelihood of Stroke', labels={'count': 'Count', 'work_type': 'Work Type', 'stroke': 'Stroke Status'})
st.plotly_chart(fig_work_stroke)
