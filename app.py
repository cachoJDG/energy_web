from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastArtifacts:
    model: RandomForestRegressor
    test_frame: pd.DataFrame
    future_frame: pd.DataFrame
    metrics: pd.DataFrame
    residual_std: float


def generate_demo_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now().floor("h")
    start = end - pd.Timedelta(days=days)
    idx = pd.date_range(start=start, end=end, freq="h")

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    base = 1.9 + 0.65 * np.sin((hour - 6) / 24 * 2 * np.pi)
    evening_spike = np.where((hour >= 19) & (hour <= 22), 0.95, 0.0)
    weekend = np.where(dow >= 5, 0.3, 0.0)
    trend = np.linspace(0.0, 0.12, len(idx))
    noise = rng.normal(0.0, 0.18, len(idx))

    consumption = np.clip(base + evening_spike + weekend + trend + noise, 0.2, None)
    return pd.DataFrame({"datetime": idx, "consumption_kWh": consumption.round(3)})


def _find_col(columns: list[str], options: tuple[str, ...]) -> Optional[str]:
    for col in columns:
        if col in options:
            return col
    return None


def load_consumption_data(uploaded_file) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file)
    if raw.empty:
        raise ValueError("The CSV file is empty.")

    raw.columns = [str(c).strip() for c in raw.columns]
    lower_map = {c.lower(): c for c in raw.columns}
    low_cols = list(lower_map.keys())

    consumption_low = _find_col(
        low_cols,
        (
            "consumo_kwh",
            "consumo",
            "kwh",
            "consumption",
            "consumption_kwh",
            "energy_kwh",
        ),
    )
    if consumption_low is None:
        raise ValueError(
            "Consumption column not found. Try one of: consumption_kWh, consumo_kWh, consumption, consumo, or kWh."
        )
    consumption_col = lower_map[consumption_low]

    datetime_low = _find_col(low_cols, ("datetime", "fecha_hora", "timestamp", "date_time"))
    fecha_low = _find_col(low_cols, ("fecha", "date"))
    hora_low = _find_col(low_cols, ("hora", "hour"))

    if datetime_low:
        dt_series = pd.to_datetime(raw[lower_map[datetime_low]], errors="coerce")
    elif fecha_low and hora_low:
        dt_series = pd.to_datetime(
            raw[lower_map[fecha_low]].astype(str) + " " + raw[lower_map[hora_low]].astype(str),
            errors="coerce",
        )
    elif fecha_low:
        dt_series = pd.to_datetime(raw[lower_map[fecha_low]], errors="coerce")
    else:
        raise ValueError(
            "Date/time columns not found. Use either datetime or fecha/date + hora/hour."
        )

    df = pd.DataFrame(
        {
            "datetime": dt_series,
            "consumption_kWh": pd.to_numeric(raw[consumption_col], errors="coerce"),
        }
    ).dropna()

    if df.empty:
        raise ValueError("No valid rows remained after parsing the CSV.")

    df = df.sort_values("datetime")
    df = df.set_index("datetime").resample("h").mean().interpolate(limit_direction="both")
    return df.reset_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


def train_and_forecast(df: pd.DataFrame, horizon_hours: int) -> ForecastArtifacts:
    feat_df = build_features(df)
    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    test_size = max(24, min(len(feat_df) // 4, 24 * 14))
    if len(feat_df) <= test_size + 24:
        raise ValueError("You need more history (minimum recommended: 3 days of hourly data).")

    train_df = feat_df.iloc[:-test_size]
    test_df = feat_df.iloc[-test_size:].copy()

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["consumption_kWh"])

    test_df["pred_model"] = model.predict(test_df[feature_cols])

    full_series = feat_df.set_index("datetime")["consumption_kWh"]
    test_df["pred_naive"] = test_df["datetime"].map(full_series.shift(24))

    valid_naive = test_df.dropna(subset=["pred_naive"])
    mae_model, rmse_model = mae_rmse(test_df["consumption_kWh"], test_df["pred_model"])
    mae_naive, rmse_naive = mae_rmse(valid_naive["consumption_kWh"], valid_naive["pred_naive"])

    metrics = pd.DataFrame(
        {
            "model": ["RandomForest", "Naive (same hour, previous day)"],
            "MAE": [mae_model, mae_naive],
            "RMSE": [rmse_model, rmse_naive],
        }
    )

    future_idx = pd.date_range(
        start=df["datetime"].max() + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )
    future_df = pd.DataFrame({"datetime": future_idx})
    future_df = build_features(future_df)
    future_df["pred_model"] = model.predict(future_df[feature_cols])

    residual_std = float((test_df["consumption_kWh"] - test_df["pred_model"]).std())

    return ForecastArtifacts(
        model=model,
        test_frame=test_df,
        future_frame=future_df,
        metrics=metrics,
        residual_std=residual_std,
    )


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    recs: list[str] = []
    dfx = df.copy()
    dfx["hour"] = dfx["datetime"].dt.hour
    dfx["dow"] = dfx["datetime"].dt.dayofweek

    hour_means = dfx.groupby("hour")["consumption_kWh"].mean().sort_values(ascending=False)
    top_hours = sorted(hour_means.head(3).index.tolist())
    recs.append(f"Your peak consumption is usually between {top_hours[0]:02d}:00 and {top_hours[-1]:02d}:00.")

    morning = dfx[dfx["hour"].between(7, 11)]["consumption_kWh"].mean()
    evening = dfx[dfx["hour"].between(19, 22)]["consumption_kWh"].mean()
    if evening > morning * 1.12:
        recs.append(
            "Consider shifting part of high-load usage to the morning to reduce evening peaks."
        )

    weekday = dfx[dfx["dow"] < 5]["consumption_kWh"].mean()
    weekend = dfx[dfx["dow"] >= 5]["consumption_kWh"].mean()
    if weekend > weekday * 1.1:
        recs.append("Your weekend pattern rises; schedule long appliance loads outside peak hours.")
    elif weekend < weekday * 0.9:
        recs.append("Your weekend pattern drops; keep this routine to maintain savings.")
    else:
        recs.append("Your weekend pattern is close to weekdays, showing stable behavior.")

    night_base = dfx[dfx["hour"].between(1, 5)]["consumption_kWh"].mean()
    overall = dfx["consumption_kWh"].mean()
    if night_base > overall * 0.75:
        recs.append("Your night base load is high; check standby devices and always-on equipment.")

    return recs


st.set_page_config(page_title="Energy Consumption Forecast Dashboard", page_icon="⚡", layout="wide")
st.title("⚡ Energy Consumption Forecast Dashboard")
st.caption("Analyze home usage history, forecast demand, and receive actionable recommendations.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Demo dataset", "Upload CSV"], index=0)
    horizon_option = st.selectbox("Forecast horizon", ["Next 24 hours", "Next 7 days"], index=0)
    st.markdown(
        "**Recommended CSV format:** `datetime` + `consumption_kWh` or `fecha/date` + `hora/hour` + `consumption_kWh`."
    )

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is None:
        st.info("Waiting for a file. In the meantime, you can try the demo dataset.")
        st.stop()
    try:
        data = load_consumption_data(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
else:
    data = generate_demo_data()

horizon_hours = 24 if "24" in horizon_option else 24 * 7

try:
    artifacts = train_and_forecast(data, horizon_hours=horizon_hours)
except Exception as exc:
    st.error(str(exc))
    st.stop()

tab_overview, tab_forecast, tab_patterns, tab_recs = st.tabs(
    ["Overview", "Forecast", "Patterns", "Recommendations"]
)

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    total_kwh = data["consumption_kWh"].sum()
    daily_avg = data.set_index("datetime")["consumption_kWh"].resample("D").sum().mean()
    max_row = data.iloc[data["consumption_kWh"].idxmax()]
    avg_hourly = data["consumption_kWh"].mean()

    col1.metric("Total analyzed consumption", f"{total_kwh:,.1f} kWh")
    col2.metric("Daily average", f"{daily_avg:,.2f} kWh/day")
    col3.metric("Maximum peak", f"{max_row['consumption_kWh']:.2f} kWh")
    col4.metric("Hourly average", f"{avg_hourly:.2f} kWh")

    hist_fig = px.line(
        data,
        x="datetime",
        y="consumption_kWh",
        title="Hourly historical consumption",
        labels={"datetime": "Date", "consumption_kWh": "kWh"},
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    daily = data.set_index("datetime")["consumption_kWh"].resample("D").sum().reset_index()
    by_hour = data.assign(hour=data["datetime"].dt.hour).groupby("hour")["consumption_kWh"].mean().reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(daily, x="datetime", y="consumption_kWh", title="Daily consumption", labels={"consumption_kWh": "kWh"}),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.line(by_hour, x="hour", y="consumption_kWh", title="Average hourly profile", labels={"consumption_kWh": "kWh"}),
        use_container_width=True,
    )

with tab_forecast:
    st.subheader("Model performance")
    metrics_display = artifacts.metrics.copy()
    metrics_display["MAE"] = metrics_display["MAE"].map(lambda x: f"{x:.3f}")
    metrics_display["RMSE"] = metrics_display["RMSE"].map(lambda x: f"{x:.3f}")
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    test_df = artifacts.test_frame
    comp_fig = go.Figure()
    comp_fig.add_trace(
        go.Scatter(x=test_df["datetime"], y=test_df["consumption_kWh"], mode="lines", name="Actual")
    )
    comp_fig.add_trace(
        go.Scatter(x=test_df["datetime"], y=test_df["pred_model"], mode="lines", name="RF prediction")
    )
    comp_fig.add_trace(
        go.Scatter(
            x=test_df["datetime"],
            y=test_df["pred_naive"],
            mode="lines",
            name="Naive",
            line={"dash": "dot"},
        )
    )
    comp_fig.update_layout(title="Actual vs predicted on validation window", yaxis_title="kWh")
    st.plotly_chart(comp_fig, use_container_width=True)

    st.subheader(f"Forecast: {horizon_option.lower()}")
    recent = data.tail(24 * 3)
    future = artifacts.future_frame
    sigma = artifacts.residual_std

    fut_fig = go.Figure()
    fut_fig.add_trace(
        go.Scatter(x=recent["datetime"], y=recent["consumption_kWh"], mode="lines", name="Recent data")
    )
    fut_fig.add_trace(
        go.Scatter(x=future["datetime"], y=future["pred_model"], mode="lines", name="Forecast")
    )
    fut_fig.add_trace(
        go.Scatter(
            x=future["datetime"],
            y=future["pred_model"] + 1.96 * sigma,
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    fut_fig.add_trace(
        go.Scatter(
            x=future["datetime"],
            y=future["pred_model"] - 1.96 * sigma,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.15)",
            name="Approx. error band",
        )
    )
    fut_fig.update_layout(yaxis_title="kWh")
    st.plotly_chart(fut_fig, use_container_width=True)

    st.dataframe(
        future[["datetime", "pred_model"]].rename(columns={"pred_model": "forecast_kWh"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_patterns:
    p = data.copy()
    p["hour"] = p["datetime"].dt.hour
    p["day_name"] = p["datetime"].dt.day_name()
    p["day_of_week"] = p["datetime"].dt.dayofweek

    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = (
        p.groupby(["day_name", "hour"])["consumption_kWh"]
        .mean()
        .reset_index()
        .assign(day_name=lambda d: pd.Categorical(d["day_name"], categories=ordered_days, ordered=True))
        .sort_values(["day_name", "hour"])
    )
    heat_pivot = heat.pivot(index="day_name", columns="hour", values="consumption_kWh")

    heatmap = px.imshow(
        heat_pivot,
        labels={"x": "Hour", "y": "Day", "color": "kWh"},
        aspect="auto",
        title="Average consumption heatmap by day/hour",
        color_continuous_scale="YlOrRd",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    p["day_type"] = np.where(p["day_of_week"] >= 5, "Weekend", "Weekday")
    comp = p.groupby(["day_type", "hour"])["consumption_kWh"].mean().reset_index()
    pat_fig = px.line(
        comp,
        x="hour",
        y="consumption_kWh",
        color="day_type",
        title="Weekday vs weekend comparison",
        labels={"consumption_kWh": "kWh", "hour": "Hour"},
    )
    st.plotly_chart(pat_fig, use_container_width=True)

with tab_recs:
    recs = generate_recommendations(data)
    st.subheader("Automatic recommendations")
    for recommendation in recs:
        st.markdown(f"- {recommendation}")

    best_hour = (
        data.assign(hour=data["datetime"].dt.hour)
        .groupby("hour")["consumption_kWh"]
        .mean()
        .sort_values()
        .index[0]
    )
    st.success(
        f"Suggested hour for shiftable loads: **{best_hour:02d}:00** (lowest historical average)."
    )
