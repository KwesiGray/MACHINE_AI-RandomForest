# MACHINE_AI-RandomForest
Solar Radiation Prediction using Machine Learning 🌞  This project leverages machine learning techniques to predict solar radiation levels based on weather data.
Solar Radiation Prediction using Machine Learning

This repository contains a Jupyter notebook focused on predicting solar radiation using various machine learning models. The project explores different regression techniques to build a predictive model based on weather-related features.
Table of Contents

    Overview
    Dataset
    Exploratory Data Analysis (EDA)
    Modeling Approach
    Feature Selection
    Model Evaluation
    Results
    Usage
    Dependencies
    Contributing
    License

Overview

This project aims to predict solar radiation levels using machine learning techniques. By leveraging weather data, the notebook demonstrates how different features (like temperature, humidity, wind direction, and pressure) correlate with solar radiation and how various models perform in predicting it.
Dataset

The dataset used in this analysis includes columns like:

    Temperature
    Humidity
    Wind Direction
    Pressure
    Solar Radiation (target)

The data is trimmed to only include relevant columns necessary for model training.
Exploratory Data Analysis (EDA)

Several visualizations are used to understand the relationships between the features and the target variable:

    Scatter plots showing feature relationships with solar radiation.
    Correlation heatmaps to identify which features are most predictive.
    Analysis reveals that temperature has the highest correlation with solar radiation.

Modeling Approach

Three regression models are explored:

    Random Forest Regressor
    Decision Tree Regressor
    Gradient Boosting Regressor

The models are evaluated based on their ability to predict solar radiation accurately.
Recursive Feature Elimination (RFE)

The notebook uses RFE to identify the most important features that contribute to predicting solar radiation, further optimizing model performance.
Model Evaluation

Each model is assessed using metrics like:

    Root Mean Squared Error (RMSE)
    R² Score
    Feature Importance Analysis

Results

The notebook provides insights into model performance before and after applying feature selection. Feature importance scores are visualized to understand which factors have the most influence on solar radiation predictions.
Usage

To run this notebook locally:

    Clone the repository:

git clone https://github.com/yourusername/solar-radiation-prediction.git

Install the required dependencies:

pip install -r requirements.txt

Open and run the Jupyter notebook:

    jupyter notebook Solar_Rad.ipynb

Dependencies

Make sure to have the following Python packages installed:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    scipy

These can be installed using:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

Contributing

Feel free to contribute by submitting a pull request. Please make sure to follow the project's coding guidelines.
License

This project is open-source and available under the MIT License.
