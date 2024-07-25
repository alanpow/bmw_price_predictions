# BMW Price Prediction

This project predicts the price of BMW cars using machine learning models. It includes a Flask API for making predictions and a Streamlit app for interactive user inputs and visualization. The data being used for the modeling is from the **BMW Pricing Challenge** data set from Kaggle (https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge/data). 

I ran a number of regression models on the avaialable features in this data set, which can be observed in the `BMW_Price_EDA.ipynb` and the `modelling_engineering.ipynb`. Those files show EDA and some statistics and analytics of the various variables, and the modelling notebook show various feature engineering work as well as fitting and comparing of models. The `personal_expenditure_eda.ipynb` notebook contains some EDA of a secondary data set containing consumer expenditure behaviour, which I tried to merge to the BMW pricing data to find seasonal trends for consumer behaviour and spending.

## Project Structure

bmw_price_predictions/  
├── source/  
│ ├── app.py  
│ ├── app_streamlit.py  
│ ├── ohe.pkl  
│ ├── poly.pkl  
│ ├── scaler.pkl  
│ ├── model.pkl  
│ ├── Dockerfile  
│ ├── requirements.txt  
│ ├── BMW_Price_EDA.ipynb  
│ ├── model_training_BAD.ipynb  
│ ├── model_training_GOOD.ipynb  
│ ├── modelling_engineering.ipynb  
│ ├── personal_expenditure_eda  
├── data/  
│ ├── bmw_pricing_challenge.csv  
│ ├── PCEC96.csv  

## Components

### 1. Flask API (`app.py`)

The Flask API is responsible for handling HTTP POST requests for price predictions. It preprocesses the input data, applies the trained model, and returns the predicted price.

- **Endpoints**: 
  - `/predict`: Accepts JSON input and returns predicted price.

- **Key Functions**:
  - `preprocess_input(data)`: Preprocesses the input data.
  - `categorize_model(model)`: Categorizes the model into entry-level, middle-level, or high-end.

#### Example Request

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '[
    {
        "maker_key": "BMW",
        "model_key": "320i",
        "mileage": 20000,
        "engine_power": 150,
        "registration_date": "2017-05-05",
        "fuel": "petrol",
        "paint_color": "black",
        "car_type": "sedan",
        "sold_at": "2018-05-05",
        "feature_1": true,
        "feature_2": true,
        "feature_3": false,
        "feature_4": true,
        "feature_5": true,
        "feature_6": true,
        "feature_7": true,
        "feature_8": false
    }
]'
```

### 2. Streamlit App (app_streamlit.py)

The Streamlit app provides an interactive interface for users to input car details and get the predicted price. It also displays the available model keys for selection.

- Key Components:
  - `load_data()`: Loads the dataset and returns the DataFrame.
  - `main()`: Main function to run the Streamlit app.

#### Running the Streamlit App

```bash
streamlit run source/app_streamlit.py
```

### 3. Data Preprocessing and Model Training
The preprocessing and model training are done in a Jupyter Notebook and the relevant code is exported to a Python script to generate the necessary pickle files (`ohe.pkl`, `poly.pkl`, `scaler.pkl`, and `model.pkl`).

#### Preprocessing Steps
1. Convert 'registration_date' and 'sold_at' to datetime.
2. Extract year and month from 'registration_date' and 'sold_at'.
3. Categorize models into entry-level, middle-level, or high-end.
4. Apply OneHotEncoder for categorical features.
5. Apply StandardScaler for numerical features.
6. Generate polynomial features for interaction terms.
#### Model Training
- The model used is a RandomForestRegressor.
- The model is trained on the preprocessed data and saved as model.pkl.
 
### Docker Setup
#### Dockerfile
The Dockerfile sets up the environment for the Flask API and the Streamlit app.

```dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source directory contents into /code
COPY . /code/

# Command to run the application
CMD ["streamlit", "run", "source/app_streamlit.py"]
```

### Requirements
The `requirements.txt` file lists the dependencies for the project.

```makefile
Flask==2.1.1
Werkzeug==2.0.3
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.5.0
joblib==1.2.0
streamlit
```

### Getting Started
#### Prerequisites
1. Docker
2. Python 3.9
3. Streamlit

#### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/yourusername/bmw_price_predictions.git
cd bmw_price_predictions
```
2. Build the Docker image:
```bash
docker build -t bmw-price-predictions .
```
3. Run the Docker container:
```bash
docker run -d -p 5000:5000 bmw-price-predictions
```
4. Access the Streamlit app:
Open your browser and go to `http://localhost:5000`.

## EDA and Model Evaluation
Extensive exploratory data analysis (EDA) and model evaluation have been performed using Jupyter Notebooks. The notebooks include detailed steps for preprocessing, feature engineering, model training, and evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score, and Explained Variance Score.

## Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
