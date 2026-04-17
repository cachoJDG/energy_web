# Energy Consumption Forecast Dashboard

Python web app (Streamlit) to analyze household energy usage, forecast future demand, and generate simple actionable recommendations.

## Features

- Data input from CSV or built-in demo dataset.
- Historical visualizations:
  - hourly time series,
  - daily consumption,
  - average hourly profile,
  - weekday vs weekend comparison,
  - day/hour heatmap.
- Forecasting with `RandomForestRegressor` for:
  - next 24 hours,
  - next 7 days.
- Metrics on a time-based validation split:
  - MAE,
  - RMSE,
  - comparison against a naive baseline (same hour, previous day).
- Automatic recommendations based on usage patterns.

## Accepted CSV format

The app can parse either of these structures:

1. `datetime`, `consumption_kWh`
2. `fecha/date`, `hora/hour`, `consumption_kWh`

It also recognizes common consumption column variants (`consumo_kWh`, `consumo`, `kWh`, `consumption`, `energy_kWh`, etc.).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Where to get CSV data

- Use the built-in **Demo dataset** in the sidebar (fastest way to test).
- Download open datasets from:
  - Kaggle (search for `household energy consumption hourly`)
  - UCI Machine Learning Repository (`Individual household electric power consumption`)
  - Open Power System Data (time series energy datasets)

If your source data is minute-level or has multiple columns, just keep/aggregate to:
- one datetime column (`datetime`) and
- one target column (`consumption_kWh`).
