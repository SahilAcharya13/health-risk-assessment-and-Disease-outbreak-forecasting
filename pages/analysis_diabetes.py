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
st.title("DiabetesS Analysis")

# Display the dataset
st.write("## Dataset")
st.write(data)

# Dropdown menu for dataset analysis, visualization and encoding
data_option = st.selectbox("Choose a Data Operation Tab:", ("Select an Option","Data Analysis", "Data Encoding"))

if data_option == "Data Analysis":
    # Display the dataset columns
    st.write("## Column")
    st.write(data.columns)

    # Display the data shape
    st.write("## Shape of Dataset")
    st.write(data.shape)

    # Display the dataset description
    st.write("## Dataset Description")
    st.write(data.describe())

    # Display Null Values
    st.write("## Null Values")
    st.write(data.isnull().sum())

    # Print maximum and minimum values
    st.write("## Maximum and Minimum Values")
    for col in data.columns:
        st.write(f"Maximum value in {col}: {data[col].max()}")
        st.write(f"Minimum value in {col}: {data[col].min()}")

    # Print total count of samples
    st.write("## Total Count of Samples")
    st.write(f"Total samples: {len(data)}")

elif data_option == "Data Encoding":

    st.title("Label Encoding and Data Splitting")

    # Display the original dataset
    st.subheader("Original Dataset")
    st.write(data)

    # Perform label encoding
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])
    data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])

    # Splitting the dataset into features (X) and target (Y)
    X = data.drop(columns=['diabetes'])
    Y = data['diabetes']

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Display the encoded dataset and the split data
    st.subheader("Encoded Dataset")
    st.write(data)

    st.subheader("Training and Testing Data")
    st.write("X_train:")
    st.write(X_train)
    st.write("X_test:")
    st.write(X_test)
    st.write("Y_train:")
    st.write(Y_train)
    st.write("Y_test:")
    st.write(Y_test)