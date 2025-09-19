# AI4SIDS
AI4SIDS Project Repo

AI4SIDS Predictive Model: Flood Risk Analysis and Prediction
This notebook focuses on developing a predictive model to assess and forecast flood risk. It integrates data from various sources, including IoT river gauges, weather sensors, and social media, to build a comprehensive dataset for analysis and modeling.

The primary objectives of this notebook are:

Data Integration and Preparation: Load, merge, and clean data from disparate sources, ensuring it is in a usable format for machine learning.
Feature Engineering: Create relevant time-based features from timestamp data to capture temporal patterns.
Exploratory Data Analysis (EDA): Visualize the spatial distribution of flood events and river levels to gain insights into the data.
Model Development: Train machine learning models (specifically, a Random Forest Classifier and Regressors) to:
Classify whether a flood event is likely to occur.
Predict the location (latitude and longitude) and timing (hour) of flood events when they are predicted to occur.
Model Evaluation: Assess the performance of the trained models using appropriate metrics (e.g., ROC AUC, Classification Report, Mean Absolute Error).
Future Simulation and Prediction: Generate simulated future data and use the trained models to predict flood risk, location, and hour for a future period (e.g., the next 7 days).
Visualization of Predictions: Visualize the simulated future flood predictions on a map to provide a clear spatial and temporal understanding of potential flood events.
This work aims to contribute to the development of an AI-driven system for Small Island Developing States (SIDS) to enhance their capacity for predicting and managing flood risks.
