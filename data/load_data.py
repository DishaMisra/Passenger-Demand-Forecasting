"""
load_data.py
------------
Loads a monthly retail-sales-style time series for the forecasting project.
Tries to download the classic public "Airline Passengers" dataset. If there's
no internet connection, falls back to generating a synthetic retail sales
series with similar trend + seasonality + noise structure.
"""

import pandas as pd
import numpy as np

REAL_DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)


def load_real_dataset() -> pd.DataFrame:
    df = pd.read_csv(REAL_DATA_URL)
    df.columns = ["month", "passengers"]
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.set_index("month")
    df = df.rename(columns={"passengers": "sales"})
    return df


def generate_synthetic_dataset(n_months: int = 96, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2017-01-01", periods=n_months, freq="MS")

    t = np.arange(n_months)
    trend = 200 + 1.8 * t
    seasonality = 40 * np.sin(2 * np.pi * t / 12) + 15 * np.cos(2 * np.pi * t / 6)
    noise = rng.normal(0, 12, n_months)

    sales = trend + seasonality + noise
    sales = np.round(np.clip(sales, a_min=10, a_max=None), 0)

    df = pd.DataFrame({"month": dates, "sales": sales}).set_index("month")
    return df


def load_dataset(prefer_real: bool = True) -> pd.DataFrame:
    if prefer_real:
        try:
            df = load_real_dataset()
            print("Loaded REAL dataset (Airline Passengers, monthly).")
            return df
        except Exception as e:
            print(f"Could not fetch real dataset ({e}). Falling back to synthetic data.")

    df = generate_synthetic_dataset()
    print("Loaded SYNTHETIC dataset (generated locally, no internet required).")
    return df


if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    print(data.tail())
    data.to_csv("data/sales_data.csv")
    print("\nSaved to data/sales_data.csv")