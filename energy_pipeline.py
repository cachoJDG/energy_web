from __future__ import annotations

"""Core data and forecasting pipeline for the energy dashboard."""

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastArtifacts:
    """Container for outputs produced by the forecasting pipeline."""

    model: RandomForestRegressor
    test_frame: pd.DataFrame
    future_frame: pd.DataFrame
    metrics: pd.DataFrame
    residual_std: float
    selected_model: str


@dataclass
class LoadedData:
    """Container for cleaned data and metadata after file parsing."""

    frame: pd.DataFrame
    source_note: str
    diagnostics: dict[str, float | int | str]


@dataclass
class ApplianceProfileData:
    """Normalized appliance-level hourly reference curves."""

    frame: pd.DataFrame
    source_note: str
    diagnostics: dict[str, float | int | str]


STANDARD_SHIFTABLE_LOADS: tuple[dict[str, float | int | str], ...] = (
    {
        "name": "Washing machine",
        "duration_hours": 2,
        "energy_kwh": 1.1,
        "notes": "Typical laundry cycle",
    },
    {
        "name": "Dishwasher",
        "duration_hours": 2,
        "energy_kwh": 1.3,
        "notes": "Typical evening-cleaning load",
    },
    {
        "name": "Tumble dryer",
        "duration_hours": 2,
        "energy_kwh": 2.4,
        "notes": "High-load but flexible appliance",
    },
)

STANDARD_EV_PROFILE = {
    "battery_kwh": 60.0,
    "charger_kw": 7.2,
    "target_soc_partial": 0.5,
    "target_soc_full": 0.8,
}

FLEXIBLE_APPLIANCE_KEYWORDS = {
    "lave-linge": "Washing machine",
    "lave-vaisselle": "Dishwasher",
    "seche-linge": "Tumble dryer",
    "sèche-linge": "Tumble dryer",
}


def generate_demo_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hourly household-like consumption for quick testing."""
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


def _hourly_profile(df: pd.DataFrame) -> pd.Series:
    profile = df.assign(hour=df["datetime"].dt.hour).groupby("hour")["consumption_kWh"].mean()
    return profile.reindex(range(24)).interpolate(limit_direction="both")


def _best_window(profile: pd.Series, duration_hours: int, pick: str) -> tuple[int, float]:
    extended = np.r_[profile.to_numpy(), profile.to_numpy()[: max(duration_hours - 1, 0)]]
    window_means = pd.Series(extended).rolling(duration_hours, min_periods=duration_hours).mean().dropna()
    window_means.index = range(len(window_means))
    start_idx = int(window_means.idxmin() if pick == "min" else window_means.idxmax())
    return start_idx, float(window_means.loc[start_idx])


def build_shiftable_load_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    profile = _hourly_profile(df)
    rows: list[dict[str, float | int | str]] = []

    for appliance in STANDARD_SHIFTABLE_LOADS:
        duration = int(appliance["duration_hours"])
        best_start, best_mean = _best_window(profile, duration_hours=duration, pick="min")
        peak_start, peak_mean = _best_window(profile, duration_hours=duration, pick="max")

        rows.append(
            {
                "appliance": appliance["name"],
                "duration_hours": duration,
                "energy_kwh": float(appliance["energy_kwh"]),
                "typical_peak_window": f"{peak_start:02d}:00-{(peak_start + duration) % 24:02d}:00",
                "recommended_window": f"{best_start:02d}:00-{(best_start + duration) % 24:02d}:00",
                "expected_load_drop_kwh": round(max(peak_mean - best_mean, 0.0), 2),
                "notes": appliance["notes"],
            }
        )

    return pd.DataFrame(rows).sort_values(["expected_load_drop_kwh", "energy_kwh"], ascending=[False, False])


def build_uploaded_appliance_recommendations(
    household_df: pd.DataFrame, appliance_profile_df: pd.DataFrame
) -> pd.DataFrame:
    household_profile = _hourly_profile(household_df)
    rows: list[dict[str, float | int | str]] = []

    grouped = appliance_profile_df.groupby(["appliance_key", "appliance"], sort=False)
    for (_, appliance_name), appliance_rows in grouped:
        appliance_key = appliance_rows["appliance_key"].iloc[0]
        if appliance_key not in FLEXIBLE_APPLIANCE_KEYWORDS:
            continue

        profile = (
            appliance_rows.groupby("hour")["consumption_wh_per_hour"]
            .mean()
            .reindex(range(24))
            .fillna(0.0)
        )
        if profile.sum() <= 0:
            continue

        typical_hour = int(profile.idxmax())
        recommended_hour = int(household_profile.idxmin())
        daily_energy_kwh = float(profile.sum() / 1000)
        overlap_now = float(household_profile.loc[typical_hour] + (profile.loc[typical_hour] / 1000))
        overlap_shifted = float(household_profile.loc[recommended_hour] + (profile.loc[typical_hour] / 1000))

        rows.append(
            {
                "appliance": FLEXIBLE_APPLIANCE_KEYWORDS[appliance_key],
                "source_appliance_name": appliance_name,
                "sample_size": int(appliance_rows["sample_size"].dropna().iloc[0]) if appliance_rows["sample_size"].notna().any() else 0,
                "period": appliance_rows["period"].iloc[0],
                "typical_hour": f"{typical_hour:02d}:00",
                "recommended_hour": f"{recommended_hour:02d}:00",
                "daily_energy_kwh": round(daily_energy_kwh, 3),
                "estimated_peak_overlap_drop_kwh": round(max(overlap_now - overlap_shifted, 0.0), 2),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "appliance",
                "source_appliance_name",
                "sample_size",
                "period",
                "typical_hour",
                "recommended_hour",
                "daily_energy_kwh",
                "estimated_peak_overlap_drop_kwh",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["estimated_peak_overlap_drop_kwh", "daily_energy_kwh"], ascending=[False, False]
    )


def build_ev_charging_recommendation(
    df: pd.DataFrame, future_df: Optional[pd.DataFrame] = None
) -> dict[str, float | int | str]:
    profile = _hourly_profile(df)
    analysis_profile = profile.copy()

    if future_df is not None and not future_df.empty and "pred_model" in future_df:
        future_profile = future_df.assign(hour=future_df["datetime"].dt.hour).groupby("hour")["pred_model"].mean()
        analysis_profile = ((profile * 0.5) + (future_profile.reindex(range(24)).fillna(profile) * 0.5)).sort_index()

    peak_start, peak_mean = _best_window(analysis_profile, duration_hours=2, pick="max")

    partial_hours = max(
        1,
        int(
            np.ceil(
                STANDARD_EV_PROFILE["battery_kwh"]
                * STANDARD_EV_PROFILE["target_soc_partial"]
                / STANDARD_EV_PROFILE["charger_kw"]
            )
        ),
    )
    full_hours = max(
        partial_hours,
        int(
            np.ceil(
                STANDARD_EV_PROFILE["battery_kwh"]
                * STANDARD_EV_PROFILE["target_soc_full"]
                / STANDARD_EV_PROFILE["charger_kw"]
            )
        ),
    )

    partial_start, partial_mean = _best_window(analysis_profile, duration_hours=partial_hours, pick="min")
    full_start, full_mean = _best_window(analysis_profile, duration_hours=full_hours, pick="min")

    return {
        "assumption": "60 kWh EV battery with a 7.2 kW home charger",
        "peak_window": f"{peak_start:02d}:00-{(peak_start + 2) % 24:02d}:00",
        "partial_target_percent": int(STANDARD_EV_PROFILE["target_soc_partial"] * 100),
        "partial_charge_hours": partial_hours,
        "partial_recommended_window": f"{partial_start:02d}:00-{(partial_start + partial_hours) % 24:02d}:00",
        "partial_expected_load_drop_kwh": round(max(peak_mean - partial_mean, 0.0), 2),
        "full_target_percent": int(STANDARD_EV_PROFILE["target_soc_full"] * 100),
        "full_charge_hours": full_hours,
        "full_recommended_window": f"{full_start:02d}:00-{(full_start + full_hours) % 24:02d}:00",
        "full_expected_load_drop_kwh": round(max(peak_mean - full_mean, 0.0), 2),
    }


def _find_col(columns: list[str], options: tuple[str, ...]) -> Optional[str]:
    for col in columns:
        if col in options:
            return col
    return None


def _normalize_text(value: str) -> str:
    return (
        str(value)
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ô", "o")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ç", "c")
        .lower()
    )


def _extract_hour_from_slot(slot: str) -> Optional[int]:
    text = str(slot).strip()
    if not text.startswith("["):
        return None
    try:
        return int(text[1:].split(",", 1)[0])
    except (TypeError, ValueError):
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


def load_appliance_profile_data(uploaded_file) -> ApplianceProfileData:
    """Load appliance-specific hourly reference curves."""
    file_name = (uploaded_file.name or "").lower()

    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()

    file_buffer = BytesIO(file_bytes)
    if file_name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file_buffer)
    else:
        try:
            raw = pd.read_csv(file_buffer, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            file_buffer.seek(0)
            raw = pd.read_csv(file_buffer, sep=";", encoding="latin1")

    raw.columns = [str(c).strip() for c in raw.columns]
    lower_map = {c.lower(): c for c in raw.columns}

    appliance_col = lower_map.get("appareil")
    period_col = lower_map.get("periode")
    slot_col = lower_map.get("tranche_horaire")
    wh_col = lower_map.get("consommation_wh/h")
    count_col = lower_map.get("nb appareils considérés") or lower_map.get("nb appareils consideres")
    year_col = lower_map.get("lib_annee")

    required = [appliance_col, period_col, slot_col, wh_col]
    if any(col is None for col in required):
        raise ValueError(
            "Appliance file format not recognized. Required columns: appareil, periode, tranche_horaire, consommation_Wh/h."
        )

    cleaned = pd.DataFrame(
        {
            "appliance": raw[appliance_col].astype(str).str.strip(),
            "period": raw[period_col].astype(str).str.strip(),
            "time_slot": raw[slot_col].astype(str).str.strip(),
            "consumption_wh_per_hour": _as_numeric(raw[wh_col]),
            "sample_size": _as_numeric(raw[count_col]) if count_col else np.nan,
            "reference_year": raw[year_col].astype(str).str.strip() if year_col else "",
        }
    )
    cleaned["hour"] = cleaned["time_slot"].map(_extract_hour_from_slot)
    cleaned["appliance_key"] = cleaned["appliance"].map(_normalize_text)
    cleaned = cleaned.dropna(subset=["hour", "consumption_wh_per_hour"]).copy()
    cleaned["hour"] = cleaned["hour"].astype(int)

    diagnostics = {
        "rows": int(len(cleaned)),
        "appliances": int(cleaned["appliance"].nunique()),
        "periods": int(cleaned["period"].nunique()),
        "reference_years": ", ".join(sorted(cleaned["reference_year"].dropna().unique().tolist())),
    }

    return ApplianceProfileData(
        frame=cleaned,
        source_note="Loaded appliance-specific hourly profiles.",
        diagnostics=diagnostics,
    )


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
        raise ValueError("Date/time columns not found. Use either datetime or fecha/date + hora/hour.")

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
    """Main loader: try standard parser first, then OPSD parser as fallback."""
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


def generate_recommendations(
    df: pd.DataFrame,
    future_df: Optional[pd.DataFrame] = None,
    appliance_profile_df: Optional[pd.DataFrame] = None,
) -> list[str]:
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
        recs.append("Consider shifting part of high-load usage to the morning to reduce evening peaks.")

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

    shiftable = (
        build_uploaded_appliance_recommendations(df, appliance_profile_df)
        if appliance_profile_df is not None and not appliance_profile_df.empty
        else build_shiftable_load_recommendations(df)
    )
    if not shiftable.empty:
        top = shiftable.iloc[0]
        if "typical_peak_window" in top.index:
            recs.append(
                f"{top['appliance']} is the best flexible load to move: from {top['typical_peak_window']} to {top['recommended_window']} to reduce the peak overlap by about {top['expected_load_drop_kwh']:.2f} kWh."
            )
        else:
            recs.append(
                f"The uploaded appliance profile shows {top['source_appliance_name']} is typically used around {top['typical_hour']}; moving it to {top['recommended_hour']} would reduce peak overlap by about {top['estimated_peak_overlap_drop_kwh']:.2f} kWh."
            )

    ev_plan = build_ev_charging_recommendation(df, future_df=future_df)
    recs.append(
        f"For EV charging, avoid the usual peak window {ev_plan['peak_window']}; if the car is not needed until the next day, aim for a {ev_plan['partial_target_percent']}% charge in {ev_plan['partial_recommended_window']} instead of charging immediately."
    )

    return recs
