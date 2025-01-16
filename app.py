import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define the numeric features (these are the columns from your creditcard.csv)
numeric_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Load the models and scaler
with open('logistic_model1.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('adaboost_model1.pkl', 'rb') as f:
    adaboost_model = pickle.load(f)

with open('xgb_model1.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to handle missing values using median imputation
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Function to handle outliers using IQR method
def handle_outliers(data):
    for column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = np.where(data[column] > upper_bound, upper_bound,
                              np.where(data[column] < lower_bound, lower_bound, data[column]))
    return data

# Function to scale input data
def scale_data(data):
    return scaler.transform(data)

# Streamlit app
st.title("Credit Card Fraud Detection")

# Input features
st.sidebar.header("User Input Features")
input_data = {}
for feature in numeric_features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# Create DataFrame and handle missing values and outliers
input_df = pd.DataFrame(input_data, index=[0])
input_df = handle_missing_values(input_df)
input_df = handle_outliers(input_df)
scaled_input = scale_data(input_df)

# Model selection
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "AdaBoost", "XGBoost"])

if st.sidebar.button("Predict"):
    # Make prediction
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(scaled_input)
        prob = logistic_model.predict_proba(scaled_input)
    elif model_choice == "AdaBoost":
        prediction = adaboost_model.predict(scaled_input)
        prob = adaboost_model.predict_proba(scaled_input)
    else:
        prediction = xgb_model.predict(scaled_input)
        prob = xgb_model.predict_proba(scaled_input)

    # Display results
    st.write("Prediction: ", "Fraud" if prediction[0] == 1 else "Not Fraud")
    st.write("Probability of Fraud: {:.2f}%".format(prob[0][1] * 100))
    
    # Display confidence metrics
    st.write("\nConfidence Metrics:")
    st.write("Probability of Normal Transaction: {:.2f}%".format(prob[0][0] * 100))
    st.write("Probability of Fraudulent Transaction: {:.2f}%".format(prob[0][1] * 100))
