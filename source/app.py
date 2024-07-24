import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load the saved models
try:
    ohe = joblib.load('ohe.pkl')
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly.pkl')
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# Function to categorize models
def categorize_model(model):
    entry_level = ['114', '116', '118', '120', '123', '125', '135', '216', '218', '220', '225', 'X1', 'X2', 'i3', 'Z4']
    middle_level = ['316', '318', '320', '325', '328', '330', '335', '418', '420', '425', '430', '435', '518', '520', '523', '525', '528', '530', '535', 'X3', 'X4', 'i4', 'i5']
    high_end = ['630', '635', '640', '650', '730', '735', '740', '750', '8', 'X5', 'X6', 'X7', 'M135', 'M235', 'M3', 'M4', 'M5', 'M550', 'i7', 'i8']
    
    if any(model.startswith(prefix) for prefix in entry_level):
        return 'entry level'
    elif any(model.startswith(prefix) for prefix in middle_level):
        return 'middle level'
    elif any(model.startswith(prefix) for prefix in high_end):
        return 'high end'
    else:
        return 'middle level'  # Default to middle level if not found

def preprocess_input(data):
    df = pd.DataFrame(data)
    
    # Convert 'registration_date' and 'sold_at' to datetime
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['sold_at'] = pd.to_datetime(df['sold_at'])
    
    # Extract year and month from 'registration_date' and 'sold_at'
    df['registration_year'] = df['registration_date'].dt.year
    df['registration_month'] = df['registration_date'].dt.month
    df['sold_year'] = df['sold_at'].dt.year
    df['sold_month'] = df['sold_at'].dt.month
    
    # Apply the model categorization
    df['model_category'] = df['model_key'].apply(categorize_model)
    
    # Define the features
    categorical_features = ['maker_key', 'model_key', 'fuel', 'paint_color', 'car_type', 'model_category']
    numerical_features = ['mileage', 'engine_power', 'registration_year', 'registration_month', 'sold_year', 'sold_month']
    
    # Transform the data
    X_categorical = ohe.transform(df[categorical_features])
    X_numerical = scaler.transform(df[numerical_features])
    X_poly = poly.transform(X_numerical)
    
    # Concatenate categorical and numerical features
    X = np.hstack((X_categorical, X_poly))
    
    return X

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = preprocess_input(data)
        predictions = model.predict(X)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
