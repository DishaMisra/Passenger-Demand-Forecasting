"""
01_eda.py
---------
Exploratory Data Analysis for Retail Demand Forecasting

Generates:
1. Raw Time Series
2. Rolling Mean & Standard Deviation
3. Seasonal Decomposition
4. Monthly Seasonality
5. Sales Distribution
6. Box Plot
7. ADF Stationarity Test
8. ACF & PACF Plots

Run:
python notebooks/01_eda.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from data.load_data import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# Raw Time Series
# -------------------------------------------------------

def plot_raw_series(df):

    plt.figure(figsize=(10,5))

    plt.plot(df.index, df["sales"], linewidth=2)

    plt.title("Monthly Sales Over Time")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "01_raw_series.png"),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Rolling Statistics
# -------------------------------------------------------

def plot_rolling_statistics(df):

    rolling_mean = df["sales"].rolling(window=12).mean()
    rolling_std = df["sales"].rolling(window=12).std()

    plt.figure(figsize=(10,5))

    plt.plot(df.index, df["sales"], label="Original", alpha=0.6)

    plt.plot(
        rolling_mean,
        label="12-Month Rolling Mean",
        linewidth=2
    )

    plt.plot(
        rolling_std,
        label="12-Month Rolling Std",
        linewidth=2
    )

    plt.title("Rolling Mean & Standard Deviation")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "02_rolling_statistics.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Seasonal Decomposition
# -------------------------------------------------------

def plot_decomposition(df):

    decomposition = seasonal_decompose(
        df["sales"],
        model="additive",
        period=12
    )

    fig = decomposition.plot()

    fig.set_size_inches(10,8)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_seasonal_decomposition.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Monthly Seasonality
# -------------------------------------------------------

def plot_monthly_seasonality(df):

    monthly = df.copy()

    monthly["Month"] = monthly.index.month_name()

    month_order = [
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ]

    monthly.groupby("Month")["sales"] \
        .mean() \
        .reindex(month_order) \
        .plot(
            kind="bar",
            figsize=(10,5)
        )

    plt.title("Average Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "04_monthly_seasonality.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Distribution
# -------------------------------------------------------

def plot_distribution(df):

    plt.figure(figsize=(8,4))

    plt.hist(df["sales"], bins=15)

    plt.title("Sales Distribution")

    plt.xlabel("Sales")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_sales_distribution.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Box Plot
# -------------------------------------------------------

def plot_boxplot(df):

    plt.figure(figsize=(5,5))

    plt.boxplot(df["sales"])

    plt.title("Sales Box Plot")
    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_boxplot.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# ADF Test
# -------------------------------------------------------

def run_adf_test(series, label=""):

    result = adfuller(series.dropna())

    print(f"\nADF Test {label}")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")

    if result[1] <= 0.05:
        print("Result : Stationary")
    else:
        print("Result : Non-Stationary")


# -------------------------------------------------------
# ACF & PACF
# -------------------------------------------------------

def plot_acf_pacf(series, label):

    fig, axes = plt.subplots(1,2, figsize=(12,4))

    plot_acf(series.dropna(), ax=axes[0], lags=24)
    plot_pacf(series.dropna(), ax=axes[1], lags=24)

    axes[0].set_title("ACF")
    axes[1].set_title("PACF")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"07_acf_pacf_{label}.png"
        ),
        dpi=120
    )

    plt.close()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    df = load_dataset()

    print("\nDataset Summary\n")
    print(df.describe())

    plot_raw_series(df)

    plot_rolling_statistics(df)

    plot_decomposition(df)

    plot_monthly_seasonality(df)

    plot_distribution(df)

    plot_boxplot(df)

    run_adf_test(df["sales"], "(Raw Series)")

    diff_series = df["sales"].diff()

    run_adf_test(diff_series, "(1st Difference)")

    plot_acf_pacf(df["sales"], "raw")

    plot_acf_pacf(diff_series, "differenced")

    print("\nEDA Completed Successfully!")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()