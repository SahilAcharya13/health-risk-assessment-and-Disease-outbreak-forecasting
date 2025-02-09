import streamlit as st
from streamlit_option_menu import option_menu

# Define the options for the sidebar menu
options = [
    "healthcare.py",
    "analysis_diabetes.py",
    "bone_fracture.py",
    "brain_tumor.py",
    "covid19.py",
    "diabetes_data_visualization.py",
    "diabetes.py",
    "disease_analysis.py",
    "disease_data_visualization.py",
    "health_care.py",
    "heart_attack.py",
    "ht_prediction.py",
    "life_expectancy_cleaning.py",
    "life_expectancy_visualization.py"
]

# Define the URLs for each page
page_urls = {
    "healthcare.py": "http://localhost:8501/healthcare",
    "analysis_diabetes.py": "http://localhost:8501/analysis_diabetes",
    "bone_fracture.py": "http://localhost:8501/bone_fracture",
    "brain_tumor.py": "http://localhost:8501/brain_tumor",
    "covid19.py": "http://localhost:8501/covid19",
    "diabetes_data_visualization.py": "http://localhost:8501/diabetes_data_visualization",
    "diabetes.py": "http://localhost:8501/diabetes",
    "disease_analysis.py": "http://yourdomain.com/disease_analysis",
    "disease_data_visualization.py": "http://yourdomain.com/disease_data_visualization",
    "health_care.py": "http://yourdomain.com/health_care",
    "heart_attack.py": "http://yourdomain.com/heart_attack",
    "ht_prediction.py": "http://yourdomain.com/ht_prediction",
    "life_expectancy_cleaning.py": "http://yourdomain.com/life_expectancy_cleaning",
    "life_expectancy_visualization.py": "http://yourdomain.com/life_expectancy_visualization"
}

# Streamlit sidebar navigation
with st.sidebar:
    with st.sidebar.expander("Menu"):
        selected = option_menu(
            menu_title="Select a Page",
            options=options,
            default_index=0,
        )

# Display content based on the selected page
if selected:
    st.title(f"Welcome to the {selected} page")
    st.markdown(f"[Go to {selected}]({page_urls[selected]})")
