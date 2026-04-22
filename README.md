# Energy Consumption Forecast Dashboard

Python web app (Streamlit) to analyze household energy usage, forecast future demand, and generate simple actionable recommendations.

## Features

- Data input from CSV/Excel upload or built-in demo dataset.
- Historical visualizations:
  - hourly time series,
  - daily consumption,
  - average hourly profile,
  - weekday vs weekend comparison,
  - day/hour heatmap.
- Forecasting for:
  - next 24 hours,
  - next 7 days.
- Model comparison and automatic selection:
  - `RandomForestRegressor`,
  - naive seasonal baseline (same hour, previous day),
  - app uses the best validation performer for future forecast.
- Metrics on a time-based validation split:
  - MAE,
  - RMSE,
  - comparison against a naive baseline (same hour, previous day).
- Data quality diagnostics panel (coverage, missing hours, range, mean/max).
- Automatic detection of cumulative meter series and conversion to hourly deltas.
- Automatic recommendations based on usage patterns.

## Accepted input format

The app supports CSV and Excel (`.xlsx`, `.xls`) files.

It can parse either of these structures:

1. `datetime`, `consumption_kWh`
2. `fecha/date`, `hora/hour`, `consumption_kWh`
3. Open Power System Data household structure (multi-header with sheets like `60min` or `15min`)

It also recognizes common consumption column variants (`consumo_kWh`, `consumo`, `kWh`, `consumption`, `energy_kWh`, etc.).

For OPSD-style files, the app auto-detects a suitable household `grid_import` series and converts it into the dashboard format (`datetime`, `consumption_kWh`).

If the detected series behaves like a cumulative meter reading, the app automatically converts it to hourly consumption (`delta kWh`) before training/plotting.

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

Alternative (recommended):

```bash
./run_project.sh
```

This script creates/uses `.venv`, installs dependencies, and launches Streamlit.

## Where to get data

- Use the built-in **Demo dataset** in the sidebar (fastest way to test).
- Download open datasets from:
  - Kaggle (search for `household energy consumption hourly`)
  - UCI Machine Learning Repository (`Individual household electric power consumption`)
  - Open Power System Data (time series energy datasets)
- This repository includes `household_data.xlsx` (OPSD-style), which now works directly from the uploader.

If your source data is minute-level or has multiple columns, just keep/aggregate to:
- one datetime column (`datetime`) and
- one target column (`consumption_kWh`).
