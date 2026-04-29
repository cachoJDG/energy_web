from __future__ import annotations

"""Streamlit main app for energy analysis and forecasting."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_pipeline import (
    build_ev_charging_recommendation,
    build_shiftable_load_recommendations,
    build_uploaded_appliance_recommendations,
    generate_demo_data,
    generate_recommendations,
    load_appliance_profile_data,
    load_consumption_data,
    train_and_forecast,
)


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
    st.divider()
    st.subheader("Specific appliance data")
    appliance_uploaded = st.file_uploader(
        "Upload appliance-specific CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="appliance_upload",
        help="For files like `elecdom_courbes_horaires_detail_appareils.csv` with columns such as appareil, tranche_horaire and consommation_Wh/h.",
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

appliance_profiles = None
appliance_source_note = "No appliance-specific file loaded. Using standard appliance assumptions."
appliance_diagnostics = None
if appliance_uploaded is not None:
    try:
        appliance_loaded = load_appliance_profile_data(appliance_uploaded)
        appliance_profiles = appliance_loaded.frame
        appliance_source_note = appliance_loaded.source_note
        appliance_diagnostics = appliance_loaded.diagnostics
    except Exception as exc:
        st.sidebar.error(f"Could not read appliance-specific file: {exc}")

try:
    artifacts = train_and_forecast(data, horizon_hours=horizon_hours)
except Exception as exc:
    st.error(str(exc))
    st.stop()

tab_overview, tab_forecast, tab_patterns, tab_recs, tab_appliances = st.tabs(
    ["Overview", "Forecast", "Patterns", "Recommendations", "Appliance Data"]
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
    recs = generate_recommendations(
        data,
        future_df=artifacts.future_frame,
        appliance_profile_df=appliance_profiles,
    )
    shiftable = (
        build_uploaded_appliance_recommendations(data, appliance_profiles)
        if appliance_profiles is not None
        else build_shiftable_load_recommendations(data)
    )
    ev_plan = build_ev_charging_recommendation(data, future_df=artifacts.future_frame)

    st.subheader("Automatic recommendations")
    for recommendation in recs:
        st.markdown(f"- {recommendation}")

    st.subheader("Shiftable household devices")
    if appliance_profiles is not None:
        st.caption("Uploaded appliance-specific curves are combined with your household load profile to suggest better operating windows.")
    else:
        st.caption("Standard appliance profiles are combined with your hourly load profile to suggest better operating windows.")
    st.dataframe(shiftable, width="stretch", hide_index=True)

    st.subheader("EV charging strategy")
    st.caption(ev_plan["assumption"])
    st.markdown(
        f"- Household peak to avoid: **{ev_plan['peak_window']}**\n"
        f"- If the car only needs to be ready tomorrow: charge to **{ev_plan['partial_target_percent']}%** during **{ev_plan['partial_recommended_window']}** "
        f"(about **{ev_plan['partial_charge_hours']} h**, estimated overlap reduction **{ev_plan['partial_expected_load_drop_kwh']:.2f} kWh**).\n"
        f"- For a larger overnight top-up: target **{ev_plan['full_target_percent']}%** during **{ev_plan['full_recommended_window']}** "
        f"(about **{ev_plan['full_charge_hours']} h**, estimated overlap reduction **{ev_plan['full_expected_load_drop_kwh']:.2f} kWh**)."
    )

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

with tab_appliances:
    st.subheader("Specific appliance data")
    st.caption(appliance_source_note)

    if appliance_profiles is None:
        st.info(
            "Upload a file like `elecdom_courbes_horaires_detail_appareils.csv` from the sidebar to inspect appliance-level hourly curves and generate recommendations from them."
        )
    else:
        st.write(appliance_diagnostics)

        appliance_summary = (
            appliance_profiles.groupby("appliance", as_index=False)
            .agg(
                sample_size=("sample_size", "max"),
                period=("period", "first"),
                avg_wh_per_hour=("consumption_wh_per_hour", "mean"),
                total_wh_per_day=("consumption_wh_per_hour", "sum"),
            )
            .sort_values("total_wh_per_day", ascending=False)
        )
        st.dataframe(appliance_summary, width="stretch", hide_index=True)

        selected_appliance = st.selectbox(
            "Appliance profile",
            appliance_profiles["appliance"].drop_duplicates().tolist(),
        )
        appliance_view = appliance_profiles[appliance_profiles["appliance"] == selected_appliance].sort_values("hour")
        appliance_chart = px.line(
            appliance_view,
            x="hour",
            y="consumption_wh_per_hour",
            title=f"Hourly profile for {selected_appliance}",
            labels={"hour": "Hour", "consumption_wh_per_hour": "Wh/h"},
        )
        st.plotly_chart(appliance_chart, width="stretch")

        uploaded_recs = build_uploaded_appliance_recommendations(data, appliance_profiles)
        if not uploaded_recs.empty:
            st.subheader("Recommendations from uploaded appliance data")
            st.dataframe(uploaded_recs, width="stretch", hide_index=True)
