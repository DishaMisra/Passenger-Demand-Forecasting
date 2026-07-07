# Passenger Demand Forecasting using SARIMA and Random Forest

## Overview

This project implements an end-to-end time series forecasting pipeline using both statistical and machine learning approaches. It compares the forecasting performance of **SARIMA** and **Random Forest** models on monthly passenger demand data.

> **Dataset:** Airline Passengers Dataset (used as a proxy for monthly demand forecasting). The pipeline can be adapted to retail sales or other monthly demand datasets.

---

## Features

- Time Series Exploratory Data Analysis (EDA)
- Trend and Seasonality Analysis
- Rolling Mean and Rolling Standard Deviation
- Augmented Dickey-Fuller (ADF) Stationarity Test
- ACF and PACF Analysis
- Monthly Seasonality Visualization
- SARIMA Forecasting
- Random Forest Forecasting using Lag Features
- Model Comparison using RMSE, MAE, and MAPE

---

## Project Structure

```
Passenger-Demand-Forecasting/
│
├── data/
│   ├── load_data.py
│   └── sales_data.csv
│
├── notebooks/
│   ├── 01_eda.py
│   └── 02_forecasting.py
│
├── outputs/
│   ├── 01_raw_series.png
│   ├── 02_rolling_statistics.png
│   ├── 03_seasonal_decomposition.png
│   ├── 04_monthly_seasonality.png
│   ├── 05_sales_distribution.png
│   ├── 06_boxplot.png
│   ├── 07_acf_pacf_raw.png
│   ├── 07_acf_pacf_differenced.png
│   ├── 04_forecast_comparison.png
│   └── model_comparison_metrics.csv
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. Load and preprocess the dataset.
2. Perform exploratory data analysis.
3. Test stationarity using the ADF test.
4. Analyze autocorrelation using ACF and PACF.
5. Train a SARIMA model.
6. Engineer lag-based features and train a Random Forest model.
7. Evaluate both models using RMSE, MAE, and MAPE.
8. Compare forecasting performance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

---

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

---

## How to Run

```bash
pip install -r requirements.txt

python data/load_data.py

python notebooks/01_eda.py

python notebooks/02_forecasting.py
```

---

## Outputs

The project generates:

- Raw Time Series Plot
- Rolling Statistics Plot
- Seasonal Decomposition
- Monthly Seasonality Plot
- Sales Distribution Histogram
- Box Plot
- ACF and PACF Plots
- Forecast Comparison Plot
- Model Performance Metrics (CSV)

---

## Future Improvements

- Hyperparameter tuning
- XGBoost and LightGBM forecasting
- Interactive Streamlit dashboard
- Support for real retail sales datasets
