
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Demand Analysis",
    page_icon="📅",
    layout="wide"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Medical_appointment_data.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["appointment_date_continuous"] = pd.to_datetime(
        data["appointment_date_continuous"]
    )
    return data

df = load_data()

# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("📅 Medical Appointment Demand Analysis")

st.write(
    "Explore historical appointment demand across different "
    "time periods and medical specialties."
)

st.divider()

# --------------------------------------------------
# Date filter
# --------------------------------------------------

st.subheader("Select Analysis Period")

min_date = df["appointment_date_continuous"].min().date()
max_date = df["appointment_date_continuous"].max().date()

selected_dates = st.date_input(
    "Appointment Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = df[
        df["appointment_date_continuous"].dt.date.between(
            start_date, end_date
        )
    ].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No appointments found for this date range.")
    st.stop()

st.divider()

# --------------------------------------------------
# Key metrics
# --------------------------------------------------

daily_demand = (
    filtered_df.groupby("appointment_date_continuous")
    .size()
)

total_appointments = len(filtered_df)
active_days = len(daily_demand)
average_daily_demand = daily_demand.mean()
maximum_daily_demand = daily_demand.max()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Appointments", f"{total_appointments:,}")
col2.metric("Active Days", f"{active_days:,}")
col3.metric("Average Daily Demand", f"{average_daily_demand:,.1f}")
col4.metric("Maximum Daily Demand", f"{maximum_daily_demand:,}")

st.divider()

# --------------------------------------------------
# Daily demand
# --------------------------------------------------

st.subheader("1. Daily Appointment Demand")

daily_data = (
    filtered_df.groupby("appointment_date_continuous")
    .size()
    .reset_index(name="appointment_count")
)

fig_daily = px.line(
    daily_data,
    x="appointment_date_continuous",
    y="appointment_count",
    title="Daily Appointment Demand",
    labels={
        "appointment_date_continuous": "Appointment Date",
        "appointment_count": "Number of Appointments"
    }
)

st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

# --------------------------------------------------
# Weekly demand
# --------------------------------------------------

st.subheader("2. Weekly Appointment Demand")

weekly_data = (
    filtered_df.set_index("appointment_date_continuous")
    .resample("W-SUN")
    .size()
    .reset_index(name="appointment_count")
)

fig_weekly = px.bar(
    weekly_data,
    x="appointment_date_continuous",
    y="appointment_count",
    title="Weekly Appointment Demand",
    labels={
        "appointment_date_continuous": "Week Ending",
        "appointment_count": "Appointments"
    }
)

st.plotly_chart(fig_weekly, use_container_width=True)

st.divider()

# --------------------------------------------------
# Monthly demand
# --------------------------------------------------

st.subheader("3. Monthly Appointment Demand")

monthly_data = (
    filtered_df.set_index("appointment_date_continuous")
    .resample("MS")
    .size()
    .reset_index(name="appointment_count")
)

fig_monthly = px.bar(
    monthly_data,
    x="appointment_date_continuous",
    y="appointment_count",
    title="Monthly Appointment Demand",
    labels={
        "appointment_date_continuous": "Month",
        "appointment_count": "Appointments"
    }
)

st.plotly_chart(fig_monthly, use_container_width=True)

st.divider()

# --------------------------------------------------
# Specialty-wise demand
# --------------------------------------------------

st.subheader("4. Specialty-wise Appointment Demand")

specialty_data = (
    filtered_df.groupby("specialty")
    .size()
    .reset_index(name="appointment_count")
    .sort_values("appointment_count", ascending=False)
)

fig_specialty = px.bar(
    specialty_data,
    x="specialty",
    y="appointment_count",
    title="Appointment Demand by Specialty",
    labels={
        "specialty": "Medical Specialty",
        "appointment_count": "Number of Appointments"
    }
)

st.plotly_chart(fig_specialty, use_container_width=True)

st.divider()

# --------------------------------------------------
# Weekday demand
# --------------------------------------------------

st.subheader("5. Weekday Appointment Demand")

weekday_data = filtered_df.copy()

weekday_data["weekday"] = (
    weekday_data["appointment_date_continuous"]
    .dt.day_name()
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_summary = (
    weekday_data.groupby("weekday")
    .size()
    .reindex(weekday_order, fill_value=0)
    .reset_index(name="appointment_count")
)

fig_weekday = px.bar(
    weekday_summary,
    x="weekday",
    y="appointment_count",
    title="Total Appointments by Day of Week",
    category_orders={"weekday": weekday_order},
    labels={
        "weekday": "Day of Week",
        "appointment_count": "Appointments"
    }
)

st.plotly_chart(fig_weekday, use_container_width=True)

# --------------------------------------------------
# Data table
# --------------------------------------------------

with st.expander("View Demand Summary Tables"):

    st.subheader("Daily Demand")
    st.dataframe(daily_data, use_container_width=True)

    st.subheader("Specialty Demand")
    st.dataframe(specialty_data, use_container_width=True)

    st.subheader("Monthly Demand")
    st.dataframe(monthly_data, use_container_width=True)

st.caption(
    "Demand represents the number of appointment records "
    "in the dataset for each period."
)
