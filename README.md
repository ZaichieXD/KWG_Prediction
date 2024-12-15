# Forecasting Models Repository 🌟
This repository hosts two advanced machine learning models:

Weather Prediction Model 🌤️
Sales Forecasting Model 📈
Both models utilize time-series forecasting techniques, feature engineering, and optimization strategies to achieve high prediction accuracy and generalizability.

## Features 📊
### Weather Prediction Model 🌤️
Time-Series Forecasting: Leverages historical weather data for temperature, humidity, pressure, and wind dynamics.
Dynamic Feature Engineering: Incorporates lagging, differencing, and scaling for accurate temporal analysis.
Custom Scaling: Features like time differences are carefully normalized to balance model inputs.
### Sales Forecasting Model 📈
Demand Prediction: Predicts future sales trends using past sales data and related features like promotions, seasonality, and holidays.
Feature Engineering: Lag and rolling averages are used to capture temporal sales patterns.
Outlier Handling: Preprocessing includes identifying and mitigating data anomalies.
### Approach 🧠
1. Libraries Used:
   * Data Handling: Pandas, NumPy
   * Visualization: Matplotlib
   * Machine Learning: TensorFlow/Keras
   * Preprocessing: Scikit-learn
   * API: FastAPI for serving predictions

2. Model Architectures:
   * Weather Prediction:
      * Built using LSTM layers for sequential data.
      * Dense layers refine predictions with ReLU and linear activation functions.
      * Dropout for regularization.
   * Sales Forecasting:
      * Combines LSTMs with feature-dense layers for time-series and categorical data.
      * Regularization and adaptive learning rates prevent overfitting.
      
3. Training Optimization:

   * Early stopping monitors validation performance.
   * ReduceLROnPlateau dynamically adjusts learning rates.

4. Feature Processing:

   * Weather model emphasizes temporal features like time differences, lags, and weather metrics.
   * Sales model focuses on seasonality, trends, and promotional influences.

How to install:  
  &nbsp; Install Python 3.12.7, Latest version python does not have tensorflow on pip
  &nbsp; Clone the repository  
  &nbsp; ``` git clone https://github.com/ZaichieXD/KWG_Prediction ```

  &nbsp;Change Directory  
  &nbsp;``` cd KWG_Prediction ```

  &nbsp;Install requirements  
  &nbsp;``` pip install -r requirements ``` 

  &nbsp;Run the app on port 8080  
  &nbsp;``` fastapi dev models_api.py --port=8080 ```

  
  
  
