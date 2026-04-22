from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
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
    selected_model: str


@dataclass
class LoadedData:
    frame: pd.DataFrame
    source_note: str
    diagnostics: dict[str, float | int | str]


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


def _clean_multiheader_piece(piece) -> str:
    if piece is None or (isinstance(piece, float) and np.isnan(piece)):
        return ""
    text = str(piece).strip()
    return "" if text.startswith("Unnamed:") else text


def _flatten_column_name(col) -> str:
    if isinstance(col, tuple):
        pieces = [_clean_multiheader_piece(part) for part in col]
        pieces = [part for part in pieces if part]
        return "|".join(pieces)
    return _clean_multiheader_piece(col)


def _as_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _detect_and_convert_cumulative(series: pd.Series) -> tuple[pd.Series, bool]:
    s = series.dropna()
    if len(s) < 200:
        return series, False

    diffs = s.diff().dropna()
    if diffs.empty:
        return series, False

    nonnegative_ratio = float((diffs >= 0).mean())
    span = float(s.max() - s.min())
    median_step = float(diffs.abs().median())

    looks_cumulative = nonnegative_ratio > 0.95 and span > 100 and median_step > 0
    if not looks_cumulative:
        return series, False

    converted = series.diff()
    converted = converted.where(converted >= 0)
    return converted, True


def _finalize_timeseries(df: pd.DataFrame, source_note: str) -> LoadedData:
    cleaned = df.dropna(subset=["datetime"]).copy()
    if cleaned.empty or "consumption_kWh" not in cleaned.columns:
        raise ValueError("No valid rows remained after parsing.")

    cleaned["consumption_kWh"], converted_from_cumulative = _detect_and_convert_cumulative(
        cleaned["consumption_kWh"]
    )

    cleaned = cleaned.sort_values("datetime").groupby("datetime", as_index=False)["consumption_kWh"].mean()
    hourly = cleaned.set_index("datetime").resample("h").mean()

    missing_before = int(hourly["consumption_kWh"].isna().sum())
    total_rows = int(len(hourly))

    hour_median = hourly["consumption_kWh"].groupby(hourly.index.hour).transform("median")
    hourly["consumption_kWh"] = hourly["consumption_kWh"].fillna(hour_median)
    hourly["consumption_kWh"] = hourly["consumption_kWh"].interpolate(
        method="time", limit=3, limit_direction="both"
    )
    hourly["consumption_kWh"] = hourly["consumption_kWh"].fillna(hourly["consumption_kWh"].median())

    if hourly["consumption_kWh"].notna().sum() == 0:
        raise ValueError("No numeric consumption values were detected after cleaning.")

    cleaned = hourly.reset_index()
    diagnostics = {
        "rows": int(len(cleaned)),
        "start": str(cleaned["datetime"].min()),
        "end": str(cleaned["datetime"].max()),
        "missing_before_fill": missing_before,
        "coverage_before_fill_pct": round(100 * (1 - (missing_before / max(total_rows, 1))), 2),
        "mean_kwh": round(float(cleaned["consumption_kWh"].mean()), 4),
        "max_kwh": round(float(cleaned["consumption_kWh"].max()), 4),
        "converted_from_cumulative": int(converted_from_cumulative),
    }

    if converted_from_cumulative:
        source_note = f"{source_note} Detected cumulative meter readings and converted to hourly consumption deltas."

    return LoadedData(frame=cleaned, source_note=source_note, diagnostics=diagnostics)


def _read_uploaded_table(uploaded_file, multi_header: bool = False) -> pd.DataFrame:
    file_name = (uploaded_file.name or "").lower()
    header = [0, 1, 2, 3, 4] if multi_header else 0

    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
    file_buffer = BytesIO(file_bytes)

    if file_name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(file_buffer, sheet_name="60min", header=header)
        except ValueError:
            file_buffer.seek(0)
            return pd.read_excel(file_buffer, sheet_name=0, header=header)

    return pd.read_csv(file_buffer, header=header, sep=None, engine="python", low_memory=False)


def _load_standard_format(raw: pd.DataFrame) -> LoadedData:
    if raw.empty:
        raise ValueError("The uploaded file is empty.")

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
            "load",
            "power_kwh",
        ),
    )
    if consumption_low is None:
        raise ValueError(
            "Consumption column not found. Try one of: consumption_kWh, consumo_kWh, consumption, consumo, kWh, energy_kWh."
        )
    consumption_col = lower_map[consumption_low]

    datetime_low = _find_col(
        low_cols,
        (
            "datetime",
            "fecha_hora",
            "timestamp",
            "date_time",
            "utc_timestamp",
            "cet_cest_timestamp",
            "time",
        ),
    )
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
            "consumption_kWh": _as_numeric(raw[consumption_col]),
        }
    )

    return _finalize_timeseries(
        df,
        source_note=f"Loaded standard format using `{consumption_col}` as target consumption series.",
    )


def _load_opsd_multiheader(raw: pd.DataFrame) -> LoadedData:
    if raw.empty:
        raise ValueError("The uploaded OPSD-style file is empty.")

    raw = raw.copy()
    raw.columns = [_flatten_column_name(col) for col in raw.columns]
    raw.columns = [col if col else f"col_{idx}" for idx, col in enumerate(raw.columns)]

    low_cols = [col.lower() for col in raw.columns]
    ts_low = _find_col(low_cols, ("utc_timestamp", "cet_cest_timestamp", "datetime", "timestamp"))
    if ts_low is None:
        raise ValueError("Timestamp column not found in OPSD-style structure.")

    timestamp_col = raw.columns[low_cols.index(ts_low)]
    dt_series = pd.to_datetime(raw[timestamp_col], errors="coerce", utc=True)
    dt_series = dt_series.dt.tz_convert(None)

    cols_for_scores: list[tuple[float, str, pd.Series]] = []
    for col in raw.columns:
        if col == timestamp_col:
            continue
        numeric = _as_numeric(raw[col])
        valid_ratio = float(numeric.notna().mean())
        if valid_ratio < 0.05:
            continue

        lowered = col.lower()
        score = valid_ratio * 10
        if "grid_import" in lowered:
            score += 100
        if "residential" in lowered:
            score += 35
        if "household" in lowered:
            score += 15
        if "kwh" in lowered:
            score += 5
        if "interpolated" in lowered:
            score -= 80
        if any(token in lowered for token in ("grid_export", "pv", "storage", "charge", "decharge")):
            score -= 30
        if any(
            token in lowered
            for token in (
                "dishwasher",
                "washing_machine",
                "freezer",
                "refrigerator",
                "machine_",
                "compressor",
                "area_room",
                "cooling_",
                "ventilation",
                "heat_pump",
                "circulation_pump",
            )
        ):
            score -= 35

        cols_for_scores.append((score, col, numeric))

    if not cols_for_scores:
        raise ValueError("No valid numeric consumption series found in OPSD-style file.")

    cols_for_scores.sort(key=lambda x: x[0], reverse=True)
    _, selected_col, consumption = cols_for_scores[0]

    df = pd.DataFrame({"datetime": dt_series, "consumption_kWh": consumption})
    df = df[df["datetime"].astype(str).str.lower() != "utc_timestamp"]

    return _finalize_timeseries(
        df,
        source_note=f"Loaded OPSD-style data (series: `{selected_col}`).",
    )


def load_consumption_data(uploaded_file) -> LoadedData:
    errors: list[str] = []

    try:
        raw_standard = _read_uploaded_table(uploaded_file, multi_header=False)
        return _load_standard_format(raw_standard)
    except Exception as exc:
        errors.append(f"standard parser: {exc}")

    try:
        raw_opsd = _read_uploaded_table(uploaded_file, multi_header=True)
        return _load_opsd_multiheader(raw_opsd)
    except Exception as exc:
        errors.append(f"OPSD parser: {exc}")

    details = " | ".join(errors)
    raise ValueError(
        "Could not parse the uploaded file. Supported: standard CSV/Excel (datetime + consumption) or OPSD-style household data. "
        f"Details: {details}"
    )


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


def add_lag_features(df: pd.DataFrame, lag_hours: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lag_hours:
        out[f"lag_{lag}"] = out["consumption_kWh"].shift(lag)
    out["roll_mean_24"] = out["consumption_kWh"].shift(1).rolling(24, min_periods=6).mean()
    out["roll_std_24"] = out["consumption_kWh"].shift(1).rolling(24, min_periods=6).std()
    return out


def _recursive_forecast(
    model: RandomForestRegressor,
    history_series: pd.Series,
    future_idx: pd.DatetimeIndex,
    feature_cols: list[str],
    lag_hours: list[int],
) -> pd.DataFrame:
    history = history_series.copy()
    preds: list[float] = []

    for dt in future_idx:
        row = build_features(pd.DataFrame({"datetime": [dt]}))
        for lag in lag_hours:
            row[f"lag_{lag}"] = history.iloc[-lag] if len(history) >= lag else np.nan

        last24 = history.iloc[-24:] if len(history) >= 24 else history
        row["roll_mean_24"] = float(last24.mean())
        row["roll_std_24"] = float(last24.std()) if len(last24) > 1 else 0.0
        row = row.fillna(history.median())

        pred = float(model.predict(row[feature_cols])[0])
        preds.append(pred)
        history.loc[dt] = pred

    return pd.DataFrame({"datetime": future_idx, "pred_model": preds})


def _naive_future_forecast(history_series: pd.Series, future_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if len(history_series) == 0:
        return pd.DataFrame({"datetime": future_idx, "pred_model": np.zeros(len(future_idx))})

    pattern = history_series.tail(24).to_numpy()
    preds = [float(pattern[i % len(pattern)]) for i in range(len(future_idx))]
    return pd.DataFrame({"datetime": future_idx, "pred_model": preds})


def mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


def train_and_forecast(df: pd.DataFrame, horizon_hours: int) -> ForecastArtifacts:
    lag_hours = [1, 24]
    if len(df) >= 24 * 21:
        lag_hours.append(168)

    feat_df = add_lag_features(build_features(df), lag_hours=lag_hours)
    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ] + [f"lag_{lag}" for lag in lag_hours] + ["roll_mean_24", "roll_std_24"]

    feat_df = feat_df.dropna(subset=feature_cols + ["consumption_kWh"]).copy()

    test_size = max(24, min(len(feat_df) // 4, 24 * 14))
    if len(feat_df) <= test_size + 48:
        raise ValueError("You need more history (minimum recommended: 14 days of hourly data).")

    train_df = feat_df.iloc[:-test_size]
    test_df = feat_df.iloc[-test_size:].copy()

    model = RandomForestRegressor(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["consumption_kWh"])

    test_df["pred_model"] = model.predict(test_df[feature_cols])

    full_series = df.set_index("datetime")["consumption_kWh"]
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
    future_df_rf = _recursive_forecast(
        model=model,
        history_series=full_series,
        future_idx=future_idx,
        feature_cols=feature_cols,
        lag_hours=lag_hours,
    )
    future_df_naive = _naive_future_forecast(history_series=full_series, future_idx=future_idx)

    selected_model = "RandomForest"
    test_df["pred_selected"] = test_df["pred_model"]
    future_df = future_df_rf.copy()
    if mae_naive < mae_model:
        selected_model = "Naive seasonal (24h)"
        test_df["pred_selected"] = test_df["pred_naive"].fillna(test_df["pred_model"])
        future_df = future_df_naive.copy()

    residual_std = float((test_df["consumption_kWh"] - test_df["pred_selected"]).std())

    return ForecastArtifacts(
        model=model,
        test_frame=test_df,
        future_frame=future_df,
        metrics=metrics,
        residual_std=residual_std,
        selected_model=selected_model,
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
    source = st.radio("Source", ["Demo dataset", "Upload file"], index=0)
    horizon_option = st.selectbox("Forecast horizon", ["Next 24 hours", "Next 7 days"], index=0)
    st.markdown(
        "**Recommended formats:** standard `datetime + consumption_kWh` or OPSD household structure (`60min`/`15min`)."
    )

if source == "Upload file":
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Waiting for a file. In the meantime, you can try the demo dataset.")
        st.stop()
    try:
        loaded = load_consumption_data(uploaded)
        data = loaded.frame
        source_note = loaded.source_note
        data_diagnostics = loaded.diagnostics
        print(
            "[UPLOAD] "
            f"file={uploaded.name} rows={data_diagnostics['rows']} "
            f"range={data_diagnostics['start']} -> {data_diagnostics['end']} "
            f"coverage_before_fill={data_diagnostics['coverage_before_fill_pct']}% "
            f"cumulative_converted={data_diagnostics['converted_from_cumulative']}"
        )
        print(f"[UPLOAD] {source_note}")
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
else:
    data = generate_demo_data()
    source_note = "Built-in synthetic household dataset."
    data_diagnostics = {
        "rows": int(len(data)),
        "start": str(data["datetime"].min()),
        "end": str(data["datetime"].max()),
        "missing_before_fill": 0,
        "coverage_before_fill_pct": 100.0,
        "mean_kwh": round(float(data["consumption_kWh"].mean()), 4),
        "max_kwh": round(float(data["consumption_kWh"].max()), 4),
        "converted_from_cumulative": 0,
    }

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
    st.caption(source_note)
    with st.expander("Data quality checks", expanded=False):
        st.write(
            {
                "rows": data_diagnostics["rows"],
                "time_range": f"{data_diagnostics['start']} -> {data_diagnostics['end']}",
                "coverage_before_fill_pct": data_diagnostics["coverage_before_fill_pct"],
                "missing_hours_before_fill": data_diagnostics["missing_before_fill"],
                "mean_kWh": data_diagnostics["mean_kwh"],
                "max_kWh": data_diagnostics["max_kwh"],
                "converted_from_cumulative": data_diagnostics["converted_from_cumulative"],
            }
        )
        if data_diagnostics["coverage_before_fill_pct"] < 80:
            st.warning(
                "This upload had many missing points before filling. Forecast quality can degrade with sparse data."
            )

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
    st.plotly_chart(hist_fig, width="stretch")

    daily = data.set_index("datetime")["consumption_kWh"].resample("D").sum().reset_index()
    by_hour = data.assign(hour=data["datetime"].dt.hour).groupby("hour")["consumption_kWh"].mean().reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(daily, x="datetime", y="consumption_kWh", title="Daily consumption", labels={"consumption_kWh": "kWh"}),
        width="stretch",
    )
    c2.plotly_chart(
        px.line(by_hour, x="hour", y="consumption_kWh", title="Average hourly profile", labels={"consumption_kWh": "kWh"}),
        width="stretch",
    )

with tab_forecast:
    st.subheader("Model performance")
    st.caption(f"Active forecast for future horizon: `{artifacts.selected_model}`")
    if artifacts.selected_model != "RandomForest":
        st.info("Naive seasonal forecast was selected because it performed better than RandomForest on validation data.")
    metrics_display = artifacts.metrics.copy()
    metrics_display["MAE"] = metrics_display["MAE"].map(lambda x: f"{x:.3f}")
    metrics_display["RMSE"] = metrics_display["RMSE"].map(lambda x: f"{x:.3f}")
    st.dataframe(metrics_display, width="stretch", hide_index=True)

    test_df = artifacts.test_frame
    display_window_hours = min(len(test_df), 24 * 7)
    test_view = test_df.tail(display_window_hours)

    comp_fig = go.Figure()
    comp_fig.add_trace(
        go.Scatter(x=test_view["datetime"], y=test_view["consumption_kWh"], mode="lines", name="Actual")
    )
    comp_fig.add_trace(
        go.Scatter(
            x=test_view["datetime"],
            y=test_view["pred_selected"],
            mode="lines",
            name=f"Selected ({artifacts.selected_model})",
        )
    )
    if artifacts.selected_model != "Naive seasonal (24h)":
        comp_fig.add_trace(
            go.Scatter(
                x=test_view["datetime"],
                y=test_view["pred_naive"],
                mode="lines",
                name="Naive",
                line={"dash": "dot"},
            )
        )
    comp_fig.update_layout(
        title=f"Actual vs predicted on validation window (last {display_window_hours // 24} days)",
        yaxis_title="kWh",
    )
    st.plotly_chart(comp_fig, width="stretch")

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
    st.plotly_chart(fut_fig, width="stretch")

    st.dataframe(
        future[["datetime", "pred_model"]].rename(columns={"pred_model": "forecast_kWh"}),
        width="stretch",
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
    st.plotly_chart(heatmap, width="stretch")

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
    st.plotly_chart(pat_fig, width="stretch")

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
