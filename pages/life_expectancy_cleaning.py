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

# Function to load the dataset
def load_data():
    data = pd.read_csv("Life Expectancy Data.csv")
    return data


# Function to display null values and rows with null values
def display_null_values(df):
    null_values = df.isnull().sum()
    st.write("Total Null Values in the Dataset:", null_values.sum())
    st.write("Null Values per Column:")
    st.write(null_values)
    st.write("Rows with Null Values:")
    st.write(df[df.isnull().any(axis=1)])


# Function to fill null values
def fill_null_values(df):
    # Fill null values with mean or mode depending on the field requirement
    df['Life expectancy '].fillna(df['Life expectancy '].mean(), inplace=True)
    df['Adult Mortality'].fillna(df['Adult Mortality'].mean(), inplace=True)
    df['Alcohol'].fillna(df['Alcohol'].mean(), inplace=True)
    df['Hepatitis B'].fillna(df['Hepatitis B'].mode()[0], inplace=True)
    df[' BMI '].fillna(df[' BMI '].mean(), inplace=True)
    df['Polio'].fillna(df['Polio'].mode()[0], inplace=True)
    df['Total expenditure'].fillna(df['Total expenditure'].mean(), inplace=True)
    df['Diphtheria '].fillna(df['Diphtheria '].mode()[0], inplace=True)
    df['GDP'].fillna(df['GDP'].mean(), inplace=True)
    df['Population'].fillna(df['Population'].mean(), inplace=True)
    df[' thinness  1-19 years'].fillna(df[' thinness  1-19 years'].mean(), inplace=True)
    df[' thinness 5-9 years'].fillna(df[' thinness 5-9 years'].mean(), inplace=True)
    df['Income composition of resources'].fillna(df['Income composition of resources'].mean(), inplace=True)
    df['Schooling'].fillna(df['Schooling'].mean(), inplace=True)


# Main function
def main():
        st.title("Life Expectancy Data Cleaning")


        # Load the data
        df = load_data()

        # Display null values
        st.subheader("Null Values in the Dataset Before Cleaning")
        display_null_values(df)

        # Fill null values
        fill_null_values(df)

        # Display null values again to confirm filling
        st.subheader("Null Values After Cleaning")
        display_null_values(df)


# Run the main function
if __name__ == "__main__":
    main()
