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
    data = pd.read_csv("Life Expectancy Data.csv")
    return data


df = load_data()

# Sidebar
st.title("Life Expectancy Data Visualization")
visualization_option = st.selectbox(
    "Select Visualization",
    ["Select ", "Top and Bottom Countries by Life Expectancy",
     "GDP vs. Life Expectancy", "Alcohol Consumption vs. Life Expectancy",
     "BMI Distribution", "Prevalence of Thinness 5-9 Years", "Country-wise Life Expectancy"]
)
# Main content




if visualization_option == "Top and Bottom Countries by Life Expectancy":
    top_countries = df.groupby('Country')['Life expectancy '].mean().nlargest(10).index.tolist()
    bottom_countries = df.groupby('Country')['Life expectancy '].mean().nsmallest(10).index.tolist()

    fig = px.bar(df[df['Country'].isin(top_countries + bottom_countries)],
                 x='Country', y='Life expectancy ',
                 title='Top and Bottom Countries by Life Expectancy',
                 color='Country')
    st.plotly_chart(fig)

elif visualization_option == "GDP vs. Life Expectancy":
    fig = px.scatter(df, x='GDP', y='Life expectancy ',
                     title='GDP vs. Life Expectancy',
                     trendline='ols',
                     labels={'GDP': 'GDP', 'Life expectancy ': 'Life Expectancy'})
    st.plotly_chart(fig)

elif visualization_option == "Alcohol Consumption vs. Life Expectancy":
    fig = px.scatter(df, x='Alcohol', y='Life expectancy ',
                     title='Alcohol Consumption vs. Life Expectancy',
                     trendline='ols',
                     labels={'Alcohol': 'Alcohol Consumption', 'Life expectancy ': 'Life Expectancy'})
    st.plotly_chart(fig)
elif visualization_option == "BMI Distribution":
    fig = px.histogram(df, x=' BMI ', title='BMI Distribution')
    st.plotly_chart(fig)


elif visualization_option == "Prevalence of Thinness 5-9 Years":
    fig = px.bar(df, x='Country', y=' thinness 5-9 years', title='Prevalence of Thinness 5-9 Years')
    st.plotly_chart(fig)

elif visualization_option == "Country-wise Life Expectancy":
    avg_life_expectancy = df.groupby('Country')['Life expectancy '].mean().reset_index()
    fig = px.choropleth(avg_life_expectancy, locations='Country', locationmode='country names',
                        color='Life expectancy ', hover_name='Country',
                        title='Country-wise Life Expectancy')
    st.plotly_chart(fig)