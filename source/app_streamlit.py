import streamlit as st
import requests
import pandas as pd

# Define the Streamlit app
def main():
    st.title("BMW Price Prediction")

    # Create input fields for user input
    maker_key = st.selectbox("Maker Key", ["BMW"])
    model_key = st.text_input("Model Key", "320i")
    mileage = st.number_input("Mileage", value=20000)
    engine_power = st.number_input("Engine Power", value=150)
    registration_date = st.date_input("Registration Date")
    fuel = st.selectbox("Fuel Type", ["petrol", "diesel", "electric", "hybrid"])
    paint_color = st.text_input("Paint Color", "black")
    car_type = st.selectbox("Car Type", ["sedan", "coupe", "suv", "hatchback", "convertible"])
    sold_at = st.date_input("Sold At")
    feature_1 = st.checkbox("Feature 1", value=True)
    feature_2 = st.checkbox("Feature 2", value=True)
    feature_3 = st.checkbox("Feature 3", value=False)
    feature_4 = st.checkbox("Feature 4", value=True)
    feature_5 = st.checkbox("Feature 5", value=True)
    feature_6 = st.checkbox("Feature 6", value=True)
    feature_7 = st.checkbox("Feature 7", value=True)
    feature_8 = st.checkbox("Feature 8", value=False)

    # Create a dictionary to hold the input data
    input_data = {
        "maker_key": maker_key,
        "model_key": model_key,
        "mileage": mileage,
        "engine_power": engine_power,
        "registration_date": registration_date.strftime("%Y-%m-%d"),
        "fuel": fuel,
        "paint_color": paint_color,
        "car_type": car_type,
        "sold_at": sold_at.strftime("%Y-%m-%d"),
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": feature_4,
        "feature_5": feature_5,
        "feature_6": feature_6,
        "feature_7": feature_7,
        "feature_8": feature_8
    }

    # When the user clicks the "Predict" button
    if st.button("Predict"):
        # Send the input data to the Flask API
        response = requests.post("http://localhost:5000/predict", json=[input_data])
        
        # Display the prediction result
        if response.status_code == 200:
            prediction = response.json()[0]
            st.success(f"The predicted price is: {prediction:.2f}")
        else:
            st.error(f"Error: {response.json()['error']}")

if __name__ == "__main__":
    main()
