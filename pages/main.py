import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import  OneHotEncoder,LabelEncoder
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

heart_data = pd.read_csv("Datasets/diabetes_prediction_dataset.csv")

st.set_page_config(page_icon=":exclamation:",page_title="PARAMETER BASED HEALTH RISK ASSESSMENT")
st.markdown(
    """
        <style>
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
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
    """,
    unsafe_allow_html=True
)
df = pd.read_csv("Datasets/diabetes_prediction_dataset.csv")
with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Data Analysis (Diabetes)", "Data Visualization (Diabetes)","Data Encoding (Diabetes)", "Diabetes Prediction","DATA CLEANING (Heart)","DATA VISUALIZATION (Heart)",
                                         "Heart Attack Prediction","stroke_analysis","Stroke_Cleaning","Stroke_Visualization","Life_Expectancy","Life_Expectancy_Visualization","Feedback","LOGOUT"],
                         icons=['house', 'activity ', 'bar-chart-fill', 'bag-fill','person-lines-fill','recycle','bar-chart-line','activity','gear-wide','recycle','file-bar-graph-fill','heart-half','graph-up-arrow','person-rolodex','arrow-right-square-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def about():
    st.title("About Health Assessment Project")
    st.write("""
        This project is designed to provide health assessment based on various parameters such as age, gender, hypertension, 
        heart disease, smoking history, BMI, HbA1c level, blood glucose level, and diabetes status.

        ### Features:
        - **Age:** Age of the individual.
        - **Gender:** Gender of the individual.
        - **Hypertension:** Whether the individual has hypertension (0: No, 1: Yes).
        - **Heart Disease:** Whether the individual has heart disease (0: No, 1: Yes).
        - **Smoking History:** Smoking history of the individual (never, former, current).
        - **BMI:** Body Mass Index of the individual.
        - **HbA1c Level:** HbA1c level of the individual.
        - **Blood Glucose Level:** Blood glucose level of the individual.
        - **Diabetes:** Diabetes status of the individual (0: No, 1: Yes).

        ### Data Source:
        The dataset used in this project is sourced from [provide data source if available].

        ### Tools Used:
        - **Streamlit:** For building the web application.
        - **Pandas:** For data manipulation.
        - **Seaborn and Matplotlib:** For data visualization.

        ### Contact:
        If you have any questions or suggestions, feel free to contact us at [provide contact information].
        """)

def Data_analysis():
    st.title("Data Analysis")
    st.write("Data Overview:")
    st.write(df.head())
    st.write("Columns:", df.columns)
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Shape:", df.shape)
    st.write("Info:\n")
    st.write(df.info())

    # Find the highest and lowest HbA1c levels
    highest_hba1c = df["HbA1c_level"].max()
    lowest_hba1c = df["HbA1c_level"].min()



    st.write("Highest HbA1c level:", highest_hba1c)
    st.write("Lowest HbA1c level:", lowest_hba1c)

    # Find the highest and lowest blood glucose levels
    highest_glucose = df["blood_glucose_level"].max()
    lowest_glucose = df["blood_glucose_level"].min()
    st.write("Highest blood glucose level:", highest_glucose)
    st.write("Lowest blood glucose level:", lowest_glucose)

def region():
    # Data Visualization
    st.title("DATA VISUALIZATION")
    # Histogram of age
    st.write("Histogram of Age:")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.histplot(data=df, x='age', bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of Age')
    st.write(fig)

    # Box plot of BMI by gender
    st.write("Box plot of BMI by gender:")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.boxplot(data=df, x='gender', y='bmi')
    plt.xlabel('Gender')
    plt.ylabel('BMI')
    plt.title('Box plot of BMI by Gender')
    st.write(fig)

    # Count plot of hypertension by smoking history
    st.write("Count plot of hypertension by smoking history:")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.countplot(data=df, x='smoking_history', hue='hypertension', order=df['smoking_history'].value_counts().index)
    plt.xlabel('Smoking History')
    plt.ylabel('Count')
    plt.title('Count plot of Hypertension by Smoking History')
    st.write(fig)

    # Pie chart of gender distribution
    st.write("Pie chart of Gender Distribution:")
    gender_distribution = df['gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.title('Gender Distribution')
    st.write(fig)

    # Pie chart of smoking wise count
    st.write("Pie chart of Smoking Wise Count:")
    smoking_distribution = df['smoking_history'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(smoking_distribution, labels=smoking_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.title('Smoking Wise Count')
    st.write(fig)

    # Pie chart of diabetic and non-diabetic
    st.write("Pie chart of Diabetic and Non-Diabetic:")
    diabetes_distribution = df['diabetes'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(diabetes_distribution, labels=diabetes_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.title('Diabetic and Non-Diabetic')
    st.write(fig)


def product():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the dataset
    def load_data():
        return pd.read_csv("Datasets/diabetes_prediction_dataset.csv")

    df = load_data()

    # Data preprocessing
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    ohe = OneHotEncoder()
    encoded_column = ohe.fit_transform(df[["smoking_history"]])
    df["smoking_history"] = encoded_column.toarray()

    # Split features and target variable
    x = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    # Train the Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    # Streamlit UI
    st.title('Diabetes Risk Prediction')
    st.write('Enter the following details to predict diabetes risk:')

    pregnancies = st.number_input("Pregnancy", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, value=0.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=0.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=0.0)
    insulin = st.number_input("Insulin Level", min_value=0.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=0, value=0)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Make predictions
    if st.button('Predict'):
        prediction = rfc.predict(input_data)
        prediction_label = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.write(f"Predicted Diabetes Risk: {prediction_label}")

        # Model evaluation
        y_pred = rfc.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        st.write(f"Accuracy: {accuracy:.2f}%")

def contact():
    st.title("INFO")
    st.write("Please fill out the form below to get in touch with us.")

    # Input fields
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    # Submit button
    if st.button("Submit"):
        if name and email and message:
            st.success("Thank you! Your message has been submitted.")
            # You can add code here to handle the submission, such as sending an email or saving to a database
        else:
            st.error("Please fill out all fields.")

def encoding():
    st.subheader("DATA BEFORE ENCODING")
    st.dataframe(df)
    st.subheader("DATA AFTER ENCODING")
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    ohe = OneHotEncoder()
    encoded_column = ohe.fit_transform(df[["smoking_history"]])
    df["smoking_history"] = encoded_column.toarray()

    # Split features and target variable
    x = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    st.subheader("X TRAIN")
    st.dataframe(x_train)
    st.subheader("X TEST")
    st.dataframe(x_test)

    st.subheader("Y_TRAIN")
    st.dataframe(y_train)
    st.subheader("Y_TEST")
    st.dataframe(y_test)

def data_cleaning():
    st.subheader("HEART DATASET")
    st.dataframe(heart_data)

    st.subheader("ROWS")
    st.write(heart_data.shape[0])
    st.subheader("COLUMNS")
    st.write(heart_data.shape[1])

    st.subheader("NULL VALUES BY COLUMNS")
    st.dataframe(heart_data.isnull().sum())


def predictions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier

    # Load the dataset
    data = pd.read_csv("Datasets/framingham.csv")

    # Separate features (X) and target variable (y)
    X = data.drop('TenYearCHD', axis=1)

    # Initialize and train the XGBoost model
    model = XGBClassifier()
    model.fit(X, data['TenYearCHD'])

    # Define function to take user input and make predictions
    def predict_with_input():
        user_input = {}
        for feature in X.columns:
            user_input[feature] = st.number_input(feature, step=0.01)

        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.write("The model predicts that the individual has a 10-year risk of coronary heart disease (CHD).")
        else:
            st.write(
                "The model predicts that the individual does not have a 10-year risk of coronary heart disease (CHD).")

    # Streamlit UI
    st.title('CHD Risk Prediction')

    st.write("Enter the values for the following features:")
    predict_with_input()

def heart_visualization():
    # Load the data
    data = pd.read_csv("Datasets/framingham.csv")

    # Set the title
    st.title('Exploratory Data Analysis')

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('AGE Histogram')
    plt.hist(data['age'], bins=20, alpha=0.7)
    st.pyplot(fig)

    # Count plot for 'education'
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Education')
    sns.countplot(x='education', data=data)
    st.pyplot(fig)

    # Count plot for 'currentSmoker'
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Current Smoker')
    sns.countplot(x='currentSmoker', data=data)
    st.pyplot(fig)

    # Count plot for 'BPMeds'
    fig, ax = plt.subplots(figsize=(10, 8))
    st.subheader('Count Plot for BPMeds')
    sns.countplot(x='BPMeds', data=data)
    st.pyplot(fig)

    # Count plot for 'prevalentStroke'
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Prevalent Stroke')
    sns.countplot(x='prevalentStroke', data=data)
    st.pyplot(fig)

    # Count plot for 'prevalentHyp'
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Prevalent Hyp')
    sns.countplot(x='prevalentHyp', data=data)
    st.pyplot(fig)

    # Count plot for 'diabetes'
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Diabetes')
    sns.countplot(x='diabetes', data=data)
    st.pyplot(fig)
def stroke_analysis():
    s_df = pd.read_csv("Datasets/stroke_data.csv")

    # Basic Level Analysis
    st.title("Basic Data Analysis")

    st.write(s_df.head(5))
    # Average age
    st.subheader("1. Average Age:")
    average_age = s_df['age'].mean()
    st.write(f"The average age of individuals is {average_age:.2f} years.")

    # Count of males and females
    st.subheader("2. Count of Males and Females:")
    gender_counts = s_df['gender'].value_counts()
    st.write(gender_counts)

    # Percentage of individuals with hypertension
    st.subheader("3. Percentage of Individuals with Hypertension:")
    hypertension_percentage = (s_df['hypertension'].sum() / len(s_df)) * 100
    st.write(f"{hypertension_percentage:.2f}% of individuals have hypertension.")

    # Most common work type
    st.subheader("4. Most Common Work Type:")
    common_work_type = s_df['work_type'].mode()[0]
    st.write(f"The most common work type is {common_work_type}.")

    # Average BMI
    st.subheader("5. Average BMI:")
    average_bmi = s_df['bmi'].mean()
    st.write(f"The average BMI of individuals is {average_bmi:.2f}.")

    # Medium Level Analysis
    st.title("Medium Data Analysis")

    # Association between hypertension and heart disease
    st.subheader("6. Association between Hypertension and Heart Disease:")
    hypertension_heart_disease = pd.crosstab(s_df['hypertension'], s_df['heart_disease'])
    st.write(hypertension_heart_disease)

    # Distribution of smoking status among individuals with and without a history of stroke
    st.subheader("7. Distribution of Smoking Status among Individuals with and without Stroke:")
    smoking_stroke = pd.crosstab(s_df['stroke'], s_df['smoking_status'])
    st.write(smoking_stroke)

    # Difference in average glucose levels between smokers and non-smokers
    st.subheader("8. Difference in Average Glucose Levels between Smokers and Non-Smokers:")
    smokers_glucose = s_df[s_df['smoking_status'] == 'smokes']['avg_glucose_level'].mean()
    non_smokers_glucose = s_df[s_df['smoking_status'] == 'never smoked']['avg_glucose_level'].mean()
    st.write(f"Average glucose level of smokers: {smokers_glucose:.2f}")
    st.write(f"Average glucose level of non-smokers: {non_smokers_glucose:.2f}")

    # Patterns between work type and likelihood of stroke
    st.subheader("9. Patterns between Work Type and Likelihood of Stroke:")
    work_stroke = pd.crosstab(s_df['work_type'], s_df['stroke'])
    st.write(work_stroke)

def Stroke_Visualization():
    sv_df = pd.read_csv("Datasets/stroke_data.csv")

    # Medium Level Questions and Graphs
    st.title("Data Visualization")

    # 1. Correlation between age and average glucose level
    st.subheader("Correlation between Age and Average Glucose Level")
    fig_correlation = px.scatter(sv_df, x='age', y='avg_glucose_level', trendline='ols',
                                 title='Correlation between Age and Average Glucose Level')
    st.plotly_chart(fig_correlation)

    # 2. Distribution of smoking status among individuals with and without a history of stroke
    st.subheader("Distribution of Smoking Status among Individuals with and without a History of Stroke")
    smoking_stroke = sv_df.groupby(['stroke', 'smoking_status']).size().reset_index(name='count')
    fig_smoking_stroke = px.bar(smoking_stroke, x='smoking_status', y='count', color='stroke', barmode='group',
                                title='Distribution of Smoking Status among Individuals with and without Stroke',
                                labels={'count': 'Count', 'smoking_status': 'Smoking Status',
                                        'stroke': 'Stroke Status'})
    st.plotly_chart(fig_smoking_stroke)

    # 3. Difference in average glucose levels between smokers and non-smokers
    st.subheader("Difference in Average Glucose Levels between Smokers and Non-Smokers")
    smokers_glucose = sv_df[sv_df['smoking_status'] == 'smokes']['avg_glucose_level'].mean()
    non_smokers_glucose = sv_df[sv_df['smoking_status'] == 'never smoked']['avg_glucose_level'].mean()
    fig_diff_glucose = px.bar(x=['Smokers', 'Non-Smokers'], y=[smokers_glucose, non_smokers_glucose],
                              title='Difference in Average Glucose Levels between Smokers and Non-Smokers',
                              labels={'x': 'Smoking Status', 'y': 'Average Glucose Level'})
    st.plotly_chart(fig_diff_glucose)

    # 4. Association between hypertension and heart disease
    st.subheader("Association between Hypertension and Heart Disease")
    hypertension_heart_disease = sv_df.groupby(['hypertension', 'heart_disease']).size().reset_index(name='count')
    fig_hypertension_heart_disease = px.bar(hypertension_heart_disease, x='hypertension', y='count',
                                            color='heart_disease', barmode='group',
                                            title='Association between Hypertension and Heart Disease',
                                            labels={'count': 'Count', 'hypertension': 'Hypertension',
                                                    'heart_disease': 'Heart Disease'})
    st.plotly_chart(fig_hypertension_heart_disease)

    # 5. Patterns between work type and likelihood of stroke
    st.subheader("Patterns between Work Type and Likelihood of Stroke")
    work_stroke = sv_df.groupby(['work_type', 'stroke']).size().reset_index(name='count')
    fig_work_stroke = px.bar(work_stroke, x='work_type', y='count', color='stroke', barmode='group',
                             title='Patterns between Work Type and Likelihood of Stroke',
                             labels={'count': 'Count', 'work_type': 'Work Type', 'stroke': 'Stroke Status'})
    st.plotly_chart(fig_work_stroke)

def Stroke_Cleaning():
    sv_df = pd.read_csv("Datasets/stroke_data.csv")

    # Streamlit app
    st.title("Data Cleaning with Regular Expressions")

    # Display the original dataset
    st.subheader("Original Dataset:")
    st.write(sv_df)

    # Find total null values
    total_null = sv_df.isnull().sum().sum()
    st.write(f"Total Null Values: {total_null}")

    # Print number of null values for each column
    st.subheader("Null Values per Column:")
    null_per_column = sv_df.isnull().sum()
    st.write(null_per_column)

    # Remove rows with missing values
    if st.checkbox("Remove Rows with Missing Values"):
        sv_df_cleaned = sv_df.dropna()
        st.write("Rows with missing values have been removed.")
        st.write(sv_df_cleaned)

    # Fill null values in the 'bmi' column with mean or mode
    fill_method = st.radio("Select Fill Method:", ["Mean", "Mode"])

    if fill_method == "Mean":
        mean_bmi = sv_df['bmi'].mean()
        sv_df['bmi'] = sv_df['bmi'].fillna(mean_bmi)
        st.write("Null values in 'bmi' column have been filled with the mean value.")
    elif fill_method == "Mode":
        mode_bmi = sv_df['bmi'].mode()[0]
        sv_df['bmi'] = sv_df['bmi'].fillna(mode_bmi)
        st.write("Null values in 'bmi' column have been filled with the mode value.")

    # Display the cleaned dataset
    st.subheader("Cleaned Dataset:")
    st.write(sv_df)

    # Remove unnecessary characters from specific columns
    columns_to_clean = st.multiselect("Select Columns to Clean:", sv_df.columns)
    if columns_to_clean:
        for column in columns_to_clean:
            pattern = st.text_input(f"Enter regex pattern to remove from '{column}':")
            if pattern:
                sv_df[column] = sv_df[column].apply(lambda x: re.sub(pattern, '', str(x)))
                st.write(f"Regex pattern '{pattern}' has been removed from column '{column}'.")
        st.write(sv_df)

    column_to_remove = st.selectbox("Select Column:", sv_df.columns)
    value_to_remove = st.text_input(f"Enter Value to Remove from '{column_to_remove}':")

    # Remove rows with specified value
    if st.button("Remove Rows"):
        sv_df_cleaned = sv_df[sv_df[column_to_remove] != value_to_remove]
        st.write(f"Rows containing '{value_to_remove}' in column '{column_to_remove}' have been removed.")
        st.write("Remaining Data:")
        st.write(sv_df_cleaned)

def Life_Expectancy():
    def load_data():
        data = pd.read_csv("Datasets/Life Expectancy Data.csv")
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
def Life_Expectancy_Visualization():
    lev_df = pd.read_csv("Datasets/Life Expectancy Data.csv")

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
        top_countries = lev_df.groupby('Country')['Life expectancy '].mean().nlargest(10).index.tolist()
        bottom_countries = lev_df.groupby('Country')['Life expectancy '].mean().nsmallest(10).index.tolist()

        fig = px.bar(lev_df[lev_df['Country'].isin(top_countries + bottom_countries)],
                     x='Country', y='Life expectancy ',
                     title='Top and Bottom Countries by Life Expectancy',
                     color='Country')
        st.plotly_chart(fig)

    elif visualization_option == "GDP vs. Life Expectancy":
        fig = px.scatter(lev_df, x='GDP', y='Life expectancy ',
                         title='GDP vs. Life Expectancy',
                         trendline='ols',
                         labels={'GDP': 'GDP', 'Life expectancy ': 'Life Expectancy'})
        st.plotly_chart(fig)

    elif visualization_option == "Alcohol Consumption vs. Life Expectancy":
        fig = px.scatter(lev_df, x='Alcohol', y='Life expectancy ',
                         title='Alcohol Consumption vs. Life Expectancy',
                         trendline='ols',
                         labels={'Alcohol': 'Alcohol Consumption', 'Life expectancy ': 'Life Expectancy'})
        st.plotly_chart(fig)
    elif visualization_option == "BMI Distribution":
        fig = px.histogram(lev_df, x=' BMI ', title='BMI Distribution')
        st.plotly_chart(fig)


    elif visualization_option == "Prevalence of Thinness 5-9 Years":
        fig = px.bar(lev_df, x='Country', y=' thinness 5-9 years', title='Prevalence of Thinness 5-9 Years')
        st.plotly_chart(fig)

    elif visualization_option == "Country-wise Life Expectancy":
        avg_life_expectancy = lev_df.groupby('Country')['Life expectancy '].mean().reset_index()
        fig = px.choropleth(avg_life_expectancy, locations='Country', locationmode='country names',
                            color='Life expectancy ', hover_name='Country',
                            title='Country-wise Life Expectancy')
        st.plotly_chart(fig)

if choose == "About":
    about()
elif choose == "Data Analysis (Diabetes)":
    Data_analysis()
elif choose == "Data Visualization (Diabetes)":
    region()

elif choose =="Data Encoding (Diabetes)":
    encoding()
elif choose == "Diabetes Prediction":
    # Execute the Signup.py script as a separate process
    product()
elif choose == "DATA CLEANING (Heart)":
    data_cleaning()
elif choose =="DATA VISUALIZATION (Heart)":
    heart_visualization()
elif choose =="Heart Attack Prediction":
    predictions()
elif choose == "stroke_analysis":
    stroke_analysis()
elif choose == "Stroke_Cleaning":
    Stroke_Cleaning()

elif choose =="Stroke_Visualization":
    Stroke_Visualization()
elif choose =="Life_Expectancy":
    Life_Expectancy()

elif choose =="Life_Expectancy_Visualization":
    Life_Expectancy_Visualization()
elif choose == "Feedback":
    contact()
elif choose == "LOGOUT":
    st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/Login">', unsafe_allow_html=True)
    st.header("Redirecting...")