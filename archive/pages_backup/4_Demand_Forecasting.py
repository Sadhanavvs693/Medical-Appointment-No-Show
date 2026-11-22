
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEMAND_PATH = BASE_DIR / "data" / "final_daily_demand.csv"
COMPARISON_PATH = BASE_DIR / "data" / "forecasting_model_comparison.csv"


@st.cache_data
def load_data():
    demand = pd.read_csv(DEMAND_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)

    demand["appointment_date_continuous"] = pd.to_datetime(
        demand["appointment_date_continuous"]
    )

    demand = demand.sort_values("appointment_date_continuous")

    # Ensure a continuous daily time series
    demand = (
        demand.set_index("appointment_date_continuous")
        .asfreq("D")
    )

    demand["appointment_count"] = (
        demand["appointment_count"].fillna(0)
    )

    return demand, comparison


if not DEMAND_PATH.exists() or not COMPARISON_PATH.exists():
    st.error("Required forecasting CSV files are missing from the data folder.")
    st.stop()

demand, comparison = load_data()

st.title("📈 Medical Appointment Demand Forecasting")

st.write(
    "Explore historical appointment demand, compare forecasting "
    "models and view a 30-day demand forecast."
)

st.divider()

# 1. Historical demand
st.subheader("1. Historical Daily Demand")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Historical Appointments",
    f"{demand['appointment_count'].sum():,}"
)

col2.metric(
    "Average Daily Demand",
    f"{demand['appointment_count'].mean():,.1f}"
)

col3.metric(
    "Maximum Daily Demand",
    f"{demand['appointment_count'].max():,}"
)

fig_history = px.line(
    demand.reset_index(),
    x="appointment_date_continuous",
    y="appointment_count",
    title="Historical Daily Appointment Demand",
    labels={
        "appointment_date_continuous": "Appointment Date",
        "appointment_count": "Number of Appointments"
    }
)

st.plotly_chart(fig_history, use_container_width=True)

st.divider()

# 2. Actual future forecast
st.subheader("2. 30-Day Appointment Demand Forecast")

st.write(
    "The SARIMA model is fitted on the historical daily appointment "
    "series and used to estimate demand for the next 30 days."
)

try:
    daily_series = demand["appointment_count"].astype(float)

    sarima_model = SARIMAX(
        daily_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = sarima_model.fit(disp=False, maxiter=200)

    forecast_result = fitted_model.get_forecast(steps=30)

    future_dates = pd.date_range(
        start=demand.index.max() + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    confidence = forecast_result.conf_int()

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_result.predicted_mean.values,
        "Lower Limit": confidence.iloc[:, 0].values,
        "Upper Limit": confidence.iloc[:, 1].values
    })

    # Appointment demand cannot be negative
    forecast_df["Forecast"] = forecast_df["Forecast"].clip(lower=0)
    forecast_df["Lower Limit"] = forecast_df["Lower Limit"].clip(lower=0)
    forecast_df["Upper Limit"] = forecast_df["Upper Limit"].clip(lower=0)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Period",
        "30 Days"
    )

    col2.metric(
        "Average Forecast Demand",
        f"{forecast_df['Forecast'].mean():,.1f}"
    )

    col3.metric(
        "Total Forecast Appointments",
        f"{forecast_df['Forecast'].sum():,.0f}"
    )

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=demand.index,
            y=demand["appointment_count"],
            mode="lines",
            name="Historical Demand"
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper Limit"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper Confidence Limit"
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Lower Limit"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="95% Confidence Interval"
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines",
            name="30-Day Forecast",
            line=dict(dash="dash", width=3)
        )
    )

    fig_forecast.update_layout(
        title="Historical Demand and 30-Day SARIMA Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Appointments",
        hovermode="x unified"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Forecast Values")

    st.dataframe(
        forecast_df.round(2),
        use_container_width=True,
        hide_index=True
    )

except Exception as error:
    st.error(f"Could not generate the SARIMA forecast: {error}")

st.divider()

# 3. Model comparison
st.subheader("3. Forecasting Model Comparison")

st.write(
    "These are the saved evaluation results from the forecasting experiments."
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

if {"Frequency", "Best_Model", "MAE"}.issubset(comparison.columns):

    st.subheader("4. Best Model by Forecasting Frequency")

    fig_mae = px.bar(
        comparison,
        x="Frequency",
        y="MAE",
        color="Best_Model",
        title="Best Model MAE by Forecasting Frequency",
        labels={
            "Frequency": "Forecasting Frequency",
            "MAE": "Mean Absolute Error",
            "Best_Model": "Selected Model"
        }
    )

    st.plotly_chart(fig_mae, use_container_width=True)

if {"Frequency", "Best_Model", "RMSE"}.issubset(comparison.columns):

    fig_rmse = px.bar(
        comparison,
        x="Frequency",
        y="RMSE",
        color="Best_Model",
        title="Best Model RMSE by Forecasting Frequency",
        labels={
            "Frequency": "Forecasting Frequency",
            "RMSE": "Root Mean Squared Error",
            "Best_Model": "Selected Model"
        }
    )

    st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()

# 5. Forecasting observations
st.subheader("5. Forecasting Observations")

st.info(
    """
    **Daily forecasting:** SARIMA achieved the lowest MAE
    among the evaluated daily models.

    **Weekly forecasting:** The Naive model achieved the lowest MAE
    among the evaluated weekly models.

    **Monthly forecasting:** The Naive model achieved the lowest MAE
    among the evaluated monthly models.
    """
)

st.warning(
    """
    **Important limitations**

    The dataset shows a sharp reduction in appointment records
    from April 3, 2021. The reason is unknown.

    The forecast is generated from the full historical series,
    including this sharp reduction. This may strongly influence
    the forecast values.

    The dataset ends on May 12, 2021. Therefore, the 30-day
    forecast represents dates after May 12, 2021, not current
    demand in 2026. Treat it as an experimental forecast, not
    a reliable current healthcare staffing estimate.
    """
)

st.caption(
    "Daily forecast model: SARIMA(1,0,1)(1,0,1,7). "
    "The shaded region represents the model's 95% confidence interval."
)
