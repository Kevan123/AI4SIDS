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

The notebook follows a typical machine learning project workflow, tailored for a predictive flood risk analysis task. Here's the structural overview:

_**Explaining the code:**_
**Problem Definition and Objectives (Markdown Cell - jCVgtb1uPBAS):**
This sets the stage, outlining the goal: predicting flood risk for SIDS using integrated data.
It clearly lists the key stages: Data Integration, Feature Engineering, EDA, Model Development (Classification and Regression), Evaluation, Simulation, and Visualization. This provides a roadmap for the subsequent code.

**Data Acquisition, Integration, and Initial Preparation (Code Cell - KAoXKFlqPV2i):**
Loading: Reads raw data from disparate sources (IoT gauges, weather sensors, social media).
Standardization: Renames key columns to create common identifiers for merging.
Provenance: Prefixes columns to track the origin of each feature, a good practice for integrated datasets.
Merging: Combines the data sources based on the standardized identifier, creating a unified dataset.
Initial Cleaning: Handles missing values using imputation (mean for numerical, most frequent for categorical).
Target Transformation: Converts the categorical flood event label ("Yes/No") into a binary numerical format (1/0), which is required for most classification algorithms.
Temporal Feature Engineering: Extracts time-based features (hour, day of week, month) from timestamps, recognizing the importance of temporal patterns in environmental data.
Filtering: Removes rows with critical missing information (latitude, longitude, target), ensuring data completeness for core modeling steps.

**Data Perturbation (Code Cell - DbVWOuQek6JT):**
Label Noise Injection: This is a specific step, likely for robustness testing or simulating real-world data imperfections. It intentionally flips a small percentage of the flood labels. This modified data (data after this step) is then used for subsequent model training.

**Model Development and Evaluation (Code Cell - ur8WowqUPFXV & pFUOn-N3mJII):**
Feature Selection: Explicitly defines the feature columns to be used for modeling.
Data Splitting: Divides the data into training and testing sets for model training and unbiased evaluation.
Classification Model:
Uses a RandomForestClassifier within a Pipeline. The pipeline includes StandardScaler to standardize features, which is often beneficial for tree-based models and crucial for many other algorithms.
Trains the classifier to predict the binary flood event.
Evaluates the classifier using standard metrics: classification_report (precision, recall, f1-score) and roc_auc_score.
Regression Models (Conditional):
Filters the data to include only instances where a flood event occurred (target = 1). This is a crucial design choice, indicating that location and hour prediction is only relevant given a flood is predicted.
Trains separate RandomForestRegressor models to predict Latitude, Longitude, and Hour for these flood events.
Evaluates regressors using mean_absolute_error (MAE), a common metric for regression tasks.
Reporting: Collects and displays the evaluation metrics in a structured format (evaluation_results dictionary and subsequent display).
Exploratory Data Visualization (Code Cell - -nxACSw1OPV0):
Spatial Visualization: Uses plotly.express to create an interactive scatter plot on a map.
Mapping Features: Maps latitude and longitude to spatial position, flood event status to color, and river level to marker size. This provides a visual EDA of the spatial distribution and characteristics of historical flood events.

<img width="1449" height="600" alt="TT_floods_19 09 2025" src="https://github.com/user-attachments/assets/b98915b5-f1a7-486a-82bf-040b3a0ced46" />

**Simulated Historical Data Generation (Code Cell - NgNGSCG0FM7S):**
This section creates a new simulated dataset, distinct from the initial merged data.
Feature Definition: Selects a subset of features for this simulation.
Imputation: Fills missing values with default assumptions for the simulation.
Rule-Based Simulation: Defines a simulate_flood function with simple rules based on river level, rainfall, and sentiment to generate a "Flood Risk (Simulated)" target variable. This is a synthetic target, likely created for demonstration or testing purposes separate from the model trained on the original data.
Saving: Saves this newly simulated historical dataset to a CSV file.

**Future Data Simulation and Prediction (Code Cells - aDQ8cj6OBq9F, 3KcsqnrVDtoi, HvM5dpyJFvuE):**
Future Data Generation: Creates a synthetic dataset for a future time period (7 days). It simulates feature values (rainfall, river level, sentiment) based on statistical properties (mean/std) derived from recent historical data and incorporates some basic temporal patterns (e.g., higher rain at night).
Model Application:
Loads the previously generated simulated historical data (from step 6) as the training data and the newly generated future data as the testing data for this specific prediction task. This implies the models trained in step 4 on the original data are not directly used here; a new classifier is trained on the simulated data.
Engineers time-based features on both simulated datasets.
Trains a RandomForestClassifier on the simulated historical data.
Uses this newly trained classifier to predict the Flood_Probability on the simulated future data.
Conditional Spatial/Temporal Prediction: Filters the future data to identify instances with a predicted flood probability above a threshold (0.5).
Applying Regressors: The intent is to predict location/hour for the simulated future floods, it would apply the previously trained regressors (either from step 4 or a newly trained set, depending on the specific code execution path) to the filtered future data.
Visualization of Future Predictions: Uses plotly.express to visualize the predicted flood locations and their associated probabilities on a map for the future 7-day period.

<img width="1449" height="800" alt="tt_time_series_19 09 2025" src="https://github.com/user-attachments/assets/e0a34138-7390-40e8-96b9-33da89d3a376" />

**Utility/Exploration (Code Cells - S3lAIJHRGntU, tab2BlaJOdNd):**
Data Inspection: Displays the head of the merged_data DataFrame, likely for a quick check of the data structure and content after the initial merge.
Time Series Visualization: Plots key features (Rainfall, River Level) and the simulated flood risk over time from the simulated historical dataset (generated in step 6). This helps understand the temporal dynamics of the simulated data.

<img width="1449" height="600" alt="newplot" src="https://github.com/user-attachments/assets/4e5d67f1-23bc-442d-94b7-57e3cc4938c2" />

In essence, the notebook demonstrates a complete predictive modeling cycle: from data integration and preparation, through model training and evaluation, to generating and visualizing predictions on simulated future scenarios. It uses both classification (for flood occurrence) and regression (for location/time) models, acknowledging the multi-faceted nature of the prediction problem. The inclusion of data simulation steps is a practical approach for generating test cases or exploring hypothetical future conditions.
