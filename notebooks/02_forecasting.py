"""
02_forecasting.py
------------------
Fits SARIMA and a Random Forest (lagged features), compares via RMSE/MAE/MAPE.
Run from project root: python notebooks/02_forecasting.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data.load_data import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 12


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_test_split_series(df, test_size=TEST_SIZE):
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


def fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train["sales"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def forecast_sarima(fitted_model, steps):
    forecast = fitted_model.get_forecast(steps=steps)
    return forecast.predicted_mean


def make_lag_features(df, n_lags=12):
    feat_df = df.copy()
    for lag in range(1, n_lags + 1):
        feat_df[f"lag_{lag}"] = feat_df["sales"].shift(lag)
    feat_df["month_of_year"] = feat_df.index.month
    feat_df = feat_df.dropna()
    return feat_df


def fit_random_forest(df, test_size=TEST_SIZE, n_lags=12):
    feat_df = make_lag_features(df, n_lags=n_lags)
    train_feat = feat_df.iloc[:-test_size]
    test_feat = feat_df.iloc[-test_size:]

    feature_cols = [c for c in feat_df.columns if c != "sales"]
    X_train, y_train = train_feat[feature_cols], train_feat["sales"]
    X_test, y_test = test_feat[feature_cols], test_feat["sales"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test, model


def main():
    df = load_dataset()
    train, test = train_test_split_series(df)

    print("=== Fitting SARIMA ===")
    sarima_model = fit_sarima(train)
    sarima_preds = forecast_sarima(sarima_model, steps=TEST_SIZE)
    sarima_preds.index = test.index

    print("\n=== Fitting Random Forest (lagged features) ===")
    rf_preds, rf_actual, rf_model = fit_random_forest(df)

    results = {}
    for name, preds, actual in [
        ("SARIMA", sarima_preds, test["sales"]),
        ("Random Forest", rf_preds, rf_actual),
    ]:
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        mape_val = mape(actual, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mape_val}
        print(f"\n{name} performance on held-out test set:")
        print(f"  RMSE : {rmse:.2f}")
        print(f"  MAE  : {mae:.2f}")
        print(f"  MAPE : {mape_val:.2f}%")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"))
    print(f"\nSaved metrics table to {OUTPUT_DIR}/model_comparison_metrics.csv")

    plt.figure(figsize=(11, 5))
    plt.plot(train.index[-24:], train["sales"].iloc[-24:], label="Train (recent)", color="gray")
    plt.plot(test.index, test["sales"], label="Actual", color="black", linewidth=2)
    plt.plot(test.index, sarima_preds, label="SARIMA Forecast", linestyle="--")
    plt.plot(rf_actual.index, rf_preds, label="Random Forest Forecast", linestyle="--")
    plt.legend()
    plt.title("Forecast Comparison: SARIMA vs Random Forest")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_forecast_comparison.png"), dpi=120)
    plt.close()

    print(f"Saved forecast comparison plot to {OUTPUT_DIR}/04_forecast_comparison.png")
    winner = results_df["RMSE"].idxmin()
    print(f"\n=== Winner (lowest RMSE): {winner} ===")


if __name__ == "__main__":
    main()