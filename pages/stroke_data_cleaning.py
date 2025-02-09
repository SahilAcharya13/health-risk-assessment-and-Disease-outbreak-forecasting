import streamlit as st
import pandas as pd
import re
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


def load_data():
    return pd.read_csv("stroke_data.csv")

df = load_data()

# Streamlit app
st.title("Data Cleaning with Regular Expressions")

# Display the original dataset
st.subheader("Original Dataset:")
st.write(df)



# Find total null values
total_null = df.isnull().sum().sum()
st.write(f"Total Null Values: {total_null}")

# Print number of null values for each column
st.subheader("Null Values per Column:")
null_per_column = df.isnull().sum()
st.write(null_per_column)

# Remove rows with missing values
if st.checkbox("Remove Rows with Missing Values"):
    df_cleaned = df.dropna()
    st.write("Rows with missing values have been removed.")
    st.write(df_cleaned)

# Fill null values in the 'bmi' column with mean or mode
fill_method = st.radio("Select Fill Method:", ["Mean", "Mode"])

if fill_method == "Mean":
    mean_bmi = df['bmi'].mean()
    df['bmi'] = df['bmi'].fillna(mean_bmi)
    st.write("Null values in 'bmi' column have been filled with the mean value.")
elif fill_method == "Mode":
    mode_bmi = df['bmi'].mode()[0]
    df['bmi'] = df['bmi'].fillna(mode_bmi)
    st.write("Null values in 'bmi' column have been filled with the mode value.")

# Display the cleaned dataset
st.subheader("Cleaned Dataset:")
st.write(df)

# Remove unnecessary characters from specific columns
columns_to_clean = st.multiselect("Select Columns to Clean:", df.columns)
if columns_to_clean:
    for column in columns_to_clean:
        pattern = st.text_input(f"Enter regex pattern to remove from '{column}':")
        if pattern:
            df[column] = df[column].apply(lambda x: re.sub(pattern, '', str(x)))
            st.write(f"Regex pattern '{pattern}' has been removed from column '{column}'.")
    st.write(df)

column_to_remove = st.selectbox("Select Column:", df.columns)
value_to_remove = st.text_input(f"Enter Value to Remove from '{column_to_remove}':")

# Remove rows with specified value
if st.button("Remove Rows"):
    df_cleaned = df[df[column_to_remove] != value_to_remove]
    st.write(f"Rows containing '{value_to_remove}' in column '{column_to_remove}' have been removed.")
    st.write("Remaining Data:")
    st.write(df_cleaned)
