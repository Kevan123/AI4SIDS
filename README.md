# AI4SIDS
AI4SIDS Project Repo

# AI4SIDS Predictive Model: Flood Risk Analysis and Prediction

This notebook focuses on developing a predictive model to assess and forecast flood risk. It integrates data from various sources, including IoT river gauges, weather sensors, and social media, to build a comprehensive dataset for analysis and modeling.

The primary objectives of this notebook are:

1.  **Data Integration and Preparation**: Load, merge, and clean data from disparate sources, ensuring it is in a usable format for machine learning.
2.  **Feature Engineering**: Create relevant time-based features from timestamp data to capture temporal patterns.
3.  **Exploratory Data Analysis (EDA)**: Visualize the spatial distribution of flood events and river levels to gain insights into the data.
4.  **Model Development**: Train machine learning models (specifically, a Random Forest Classifier and Regressors) to:
    *   Classify whether a flood event is likely to occur.
    *   Predict the location (latitude and longitude) and timing (hour) of flood events when they are predicted to occur.
5.  **Model Evaluation**: Assess the performance of the trained models using appropriate metrics (e.g., ROC AUC, Classification Report, Mean Absolute Error).
6.  **Future Simulation and Prediction**: Generate simulated future data and use the trained models to predict flood risk, location, and hour for a future period (e.g., the next 7 days).
7.  **Visualization of Predictions**: Visualize the simulated future flood predictions on a map to provide a clear spatial and temporal understanding of potential flood events.

This work aims to contribute to the development of an AI-driven system for Small Island Developing States (SIDS) to enhance their capacity for predicting and managing flood risks.

import plotly.express as px

# Assuming 'data' DataFrame is already created and processed as in previous cells
# Ensure 'iot_Latitude', 'iot_Longitude', and 'Flood_Event' columns exist

if not data.empty:
    # Create a scatter plot on a map
    fig = px.scatter_mapbox(data,
                            lat="iot_Latitude",
                            lon="iot_Longitude",
                            color="Flood_Event", # Color points by flood event (0 or 1)
                            size="iot_River Level (m)", # Size of points by river level
                            color_continuous_scale="Viridis",
                            zoom=7,
                            height=600,
                            hover_data=["iot_Timestamp", "iot_Sensor ID", "iot_River Level (m)", "Flood_Event"])

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(title_text="Flood Events and River Levels at Sensor Locations")
    fig.show()
else:
    print("The 'data' DataFrame is empty.")
