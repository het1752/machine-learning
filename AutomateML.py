import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

st.title("AutomateML- Makes Common Tasks Easier")
st.text("Made By Human")
# Function to analyze datetime column
def analyze_datetime_column(df, column):
    df[column] = pd.to_datetime(df[column])
    df.set_index(column, inplace=True)
    daily_data = df.resample('D').mean()
    monthly_data = df.resample('M').mean()
    yearly_data = df.resample('Y').mean()

    st.write("### Time Series Analysis:")
    st.write("#### Daily Data:")
    st.write(daily_data.head())

    st.write("#### Monthly Data:")
    st.write(monthly_data.head())

    st.write("#### Yearly Data:")
    st.write(yearly_data.head())

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Remove datetime columns if any
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_columns:
        data.drop(columns=datetime_columns, inplace=True)
        st.warning("Datetime columns have been removed.")

    # Display the uploaded data
    st.write("### Uploaded Data:")
    st.write(data)

    # Exploratory Data Analysis (EDA)
    st.write("### Summary Statistics:")
    st.write(data.describe())

    st.write("### Correlation Matrix:")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    # Handling outliers
    st.write("### Outliers Handling:")
    outlier_columns = st.multiselect("Select columns to handle outliers:", data.columns)
    for column in outlier_columns:
        st.write(f"#### {column} Outliers Handling:")
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
        st.write(f"**Lower Bound:** {lower_bound}, **Upper Bound:** {upper_bound}")
        st.write(f"**Number of Outliers:** {len(outliers)}")
        st.write(outliers)

    # More Statistical Info
    st.write("### More Statistical Info:")
    statistical_info = st.selectbox("Select statistical info:", ["Value Counts", "Pie Chart"])
    if statistical_info == "Value Counts":
        st.write("#### Value Counts:")
        for column in data.columns:
            st.write(f"**{column}:**")
            st.write(data[column].value_counts())
    elif statistical_info == "Pie Chart":
        st.write("#### Pie Chart:")
        pie_column = st.selectbox("Select column for Pie Chart:", data.columns)
        pie_chart_data = data[pie_column].value_counts()
        st.write(pie_chart_data)
        fig, ax = plt.subplots()
        ax.pie(pie_chart_data, labels=pie_chart_data.index, autopct='%1.1f%%')
        st.pyplot(fig)

    # Time Series Analysis for datetime columns
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_columns:
        st.write("### Time Series Analysis:")
        for column in datetime_columns:
            analyze_datetime_column(data, column)

    # Identify and Encode Categorical Data
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        st.write("### Categorical Columns:")
        st.write(categorical_columns)
        st.write("#### Encoding Categorical Data:")
        for column in categorical_columns:
            st.write(f"Encoding {column}...")
            label_encoder = LabelEncoder()
            data[column] = label_encoder.fit_transform(data[column])

        st.write("### Encoded Data:")
        st.write(data)
    else:
        st.write("No categorical columns found.")

    # Data Analysis based on Task Type
    st.write("## Data Analysis based on Task Type:")
    task_type = st.selectbox("Select task type:", ["Classification", "Regression"])
    if task_type == "Classification":
        st.write("### Classification Task Analysis:")
        target_column_classification = st.selectbox("Select target column for classification:", data.columns)
        st.write("#### Class Distribution:")
        st.write(data[target_column_classification].value_counts())

        st.write("#### Class Imbalance:")
        imbalance_threshold_classification = st.slider("Select imbalance threshold for classification:",
                                                        min_value=0.1, max_value=0.9, step=0.1)
        class_counts_classification = data[target_column_classification].value_counts(normalize=True)
        imbalanced_classes_classification = class_counts_classification[class_counts_classification < imbalance_threshold_classification]
        st.write(imbalanced_classes_classification)

        # Machine Learning Tasks - Classification
        st.write("## Machine Learning Tasks - Classification:")
        st.write("### Model Selection and Training:")
        X_classification = data.drop(columns=[target_column_classification])
        y_classification = data[target_column_classification]

        # Split data into train and test sets
        X_train_classification, X_test_classification, y_train_classification, y_test_classification = \
            train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler_classification = StandardScaler()
        X_train_scaled_classification = scaler_classification.fit_transform(X_train_classification)
        X_test_scaled_classification = scaler_classification.transform(X_test_classification)

        # Model Selection and Training
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        for name, classifier in classifiers.items():
            classifier.fit(X_train_scaled_classification, y_train_classification)
            y_pred_classification = classifier.predict(X_test_scaled_classification)
            accuracy_classification = accuracy_score(y_test_classification, y_pred_classification)
            st.write(f"{name} Accuracy:", accuracy_classification)

            # Confusion Matrix
            st.write(f"#### Confusion Matrix for {name}:")
            cm_classification = confusion_matrix(y_test_classification, y_pred_classification)
            st.write(cm_classification)


    elif task_type == "Regression":

        st.write("### Regression Task Analysis:")

        target_column_regression = st.selectbox("Select target column for regression:", data.columns)



        st.write("#### Skewness of Target Variable:")

        skewness = data[target_column_regression].skew()

        st.write(skewness)

        # Machine Learning Tasks - Regression

        st.write("## Machine Learning Tasks - Regression:")

        st.write("### Model Selection and Training:")

        X_regression = data.drop(columns=[target_column_regression])

        y_regression = data[target_column_regression]

        # Split data into train and test sets

        X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

    # Feature Scaling

        scaler_regression = StandardScaler()

        X_train_scaled_regression = scaler_regression.fit_transform(X_train_regression)

        X_test_scaled_regression = scaler_regression.transform(X_test_regression)

    # Model Selection and Training

    regressors = {

        'Linear Regression': LinearRegression(),

        'Random Forest': RandomForestRegressor(),

        'Support Vector Machine': SVR(),

        'K-Nearest Neighbors': KNeighborsRegressor()

    }

    for name, regressor in regressors.items():
        regressor.fit(X_train_scaled_regression, y_train_regression)

        y_pred_regression = regressor.predict(X_test_scaled_regression)

        mse = mean_squared_error(y_test_regression, y_pred_regression)

        st.write(f"{name} MSE:", mse)

