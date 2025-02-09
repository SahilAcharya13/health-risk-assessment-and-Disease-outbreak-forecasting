import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
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

# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Create a new page
st.title("Diabetes Analysis")

# Display the dataset
st.write("## Dataset")
st.write(data)

# Filter out non-numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Display bar graph for maximum values of columns
st.write("## Bar Graph of Maximum Values of Columns")
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=numeric_data.columns, y=numeric_data.max())
plt.xlabel("Columns of Dataset")
plt.ylabel("Maximum values of Columns")
plt.xticks(rotation=45)
st.pyplot(plt.gcf(), transparent=True)

# Iterate through each column and create a pie chart for gender and smoking history
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        continue  # Skip numerical columns for this visualization
    st.write(f"## Pie Chart for {column}")
    plt.figure(figsize=(10, 6))
    pie_chart = data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.ylabel('')
    st.pyplot(plt.gcf(), transparent=True)

# Display a pie chart for Diabetes
st.write("## Pie Chart for Diabetes")
diabetes_counts = data['diabetes'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(diabetes_counts, labels=['Not Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(plt.gcf(), transparent=True)

# Display a Histogram for age
st.title("Histogram of Age")
fig = px.histogram(data, x='age', nbins=20, title="Distribution of Age")
fig.update_layout(xaxis_title="Age", yaxis_title="Frequency")
st.plotly_chart(fig)

# Display the box plot for BMI for gender
st.title("Box plot of BMI by gender")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='gender', y='bmi')
plt.title('Box Plot of BMI by Gender')
plt.xlabel('Gender')
plt.ylabel('BMI')
st.pyplot(plt.gcf(), transparent=True)

# Display count plot for hypertension by smoking history
st.title("Count Plot of Hypertension by Smoking History")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='smoking_history', hue='hypertension')
plt.title('Count Plot of Hypertension by Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Count')
plt.legend(title='Hypertension', labels=['No', 'Yes'])
st.pyplot(plt.gcf(), transparent=True)