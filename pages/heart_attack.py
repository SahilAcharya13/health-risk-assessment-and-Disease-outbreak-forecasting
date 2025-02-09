import streamlit as st
import pandas as pd

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


def print_rows_columns(data):
    st.write(data)
    st.subheader("Print Rows and Columns")
    st.write("Number of rows:", data.shape[0])
    st.write("Number of columns:", data.shape[1])
def check_null_values(data):
    st.subheader("Check for Null Values")
    null_values = data.isnull().sum()
    st.write(null_values)
    if null_values.sum() == 0:
        st.write("No null values found in the dataset.")
    else:
        st.write("Null values found in the dataset.")
def remove_null_values(data):
    original_count = data.shape[0]
    data.dropna(inplace=True)
    removed_count = original_count - data.shape[0]
    st.write(f"Removed {removed_count} rows with null values.")
    return data

def fill_null_with_mean(data):
    data_filled_mean = data.fillna(data.mean())
    st.write("Null values filled with mean:")
    st.write(data_filled_mean)
    return data_filled_mean

def fill_null_with_mode(data):
    data_filled_mode = data.fillna(data.mode().iloc[0])
    st.write("Null values filled with mode:")
    st.write(data_filled_mode)
    return data_filled_mode

def main():
    st.title("Data Cleaning")

    # Load the dataset
    data = pd.read_csv('framingham.csv')

    print_rows_columns(data)

    check_null_values(data)

    data = remove_null_values(data)

    check_null_values(data)

    data = fill_null_with_mean(data)

    data = fill_null_with_mode(data)

    st.subheader("Final Cleaned Dataset")
    st.write(data)

if __name__ == "__main__":
    main()
