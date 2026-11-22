
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Resource Planning",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Medical_appointment_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["appointment_date_continuous"] = pd.to_datetime(
        df["appointment_date_continuous"]
    )

    df["specialty"] = (
        df["specialty"]
        .fillna("Unknown")
        .astype(str)
    )

    return df


if not DATA_PATH.exists():
    st.error("Medical appointment dataset not found.")
    st.stop()

df = load_data()

st.title("🏥 Healthcare Resource Planning")

st.write(
    "Explore appointment demand across medical specialties "
    "and weekdays to support healthcare resource planning."
)

st.divider()

# 1. Date filter
st.subheader("1. Select Analysis Period")

min_date = df["appointment_date_continuous"].min().date()
max_date = df["appointment_date_continuous"].max().date()

selected_dates = st.date_input(
    "Appointment date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

filtered_df = df[
    df["appointment_date_continuous"].dt.date.between(
        start_date, end_date
    )
].copy()

if filtered_df.empty:
    st.warning("No appointments found for the selected period.")
    st.stop()

# 2. Key metrics
st.divider()
st.subheader("2. Resource Planning Overview")

daily_demand = (
    filtered_df.groupby("appointment_date_continuous")
    .size()
)

specialty_demand = (
    filtered_df.groupby("specialty")
    .size()
    .sort_values(ascending=False)
)

total_appointments = len(filtered_df)
average_daily_demand = daily_demand.mean()
highest_specialty = specialty_demand.idxmax()
highest_specialty_count = specialty_demand.max()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Appointments",
    f"{total_appointments:,}"
)

col2.metric(
    "Average Daily Demand",
    f"{average_daily_demand:,.1f}"
)

col3.metric(
    "Highest-Demand Specialty",
    highest_specialty,
    help=f"{highest_specialty_count:,} appointments"
)

st.divider()

# 3. Specialty demand
st.subheader("3. Appointment Demand by Specialty")

specialty_chart = (
    filtered_df.groupby("specialty")
    .size()
    .reset_index(name="Appointments")
    .sort_values("Appointments", ascending=True)
)

fig_specialty = px.bar(
    specialty_chart,
    x="Appointments",
    y="specialty",
    orientation="h",
    title="Total Appointments by Medical Specialty",
    labels={
        "specialty": "Medical Specialty",
        "Appointments": "Number of Appointments"
    }
)

st.plotly_chart(fig_specialty, use_container_width=True)

st.divider()


# 4. Weekday demand
st.subheader("4. Appointment Demand by Weekday")

filtered_df["weekday"] = (
    filtered_df["appointment_date_continuous"].dt.day_name()
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_demand = (
    filtered_df.groupby("weekday")
    .agg(
        Total_Appointments=("weekday", "size"),
        Active_Days=("appointment_date_continuous", "nunique")
    )
    .reindex(weekday_order, fill_value=0)
    .reset_index()
)

weekday_demand["Average Daily Demand"] = (
    weekday_demand["Total_Appointments"] /
    weekday_demand["Active_Days"].replace(0, 1)
)

fig_weekday = px.bar(
    weekday_demand,
    x="weekday",
    y="Average Daily Demand",
    title="Average Daily Appointments by Weekday",
    labels={
        "weekday": "Day of Week",
        "Average Daily Demand": "Average Appointments per Day"
    }
)

st.plotly_chart(fig_weekday, use_container_width=True)


st.divider()

# 5. Specialty-level daily demand
st.subheader("5. Daily Demand by Specialty")

selected_specialty = st.selectbox(
    "Select a medical specialty",
    sorted(filtered_df["specialty"].unique())
)

specialty_daily = (
    filtered_df[
        filtered_df["specialty"] == selected_specialty
    ]
    .groupby("appointment_date_continuous")
    .size()
    .reset_index(name="Appointments")
)

fig_daily = px.line(
    specialty_daily,
    x="appointment_date_continuous",
    y="Appointments",
    title=f"Daily Appointment Demand - {selected_specialty}",
    labels={
        "appointment_date_continuous": "Appointment Date",
        "Appointments": "Number of Appointments"
    }
)

st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

# 6. Planning summary
st.subheader("6. Resource Planning Summary")

summary = (
    filtered_df.groupby("specialty")
    .agg(
        Total_Appointments=("specialty", "size"),
        Active_Days=("appointment_date_continuous", "nunique")
    )
    .reset_index()
)

summary["Average_Daily_Demand"] = (
    summary["Total_Appointments"] / summary["Active_Days"]
)

summary = summary.sort_values(
    "Total_Appointments",
    ascending=False
)

st.dataframe(
    summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("7. Planning Considerations")

st.info(
    """
    - Higher-demand specialties may require closer monitoring
      of appointment capacity.
    - Weekday demand patterns can support scheduling decisions.
    - Specialty-level trends can help identify changing service demand.
    - Historical appointment counts can inform preliminary
      resource planning discussions.
    """
)

st.warning(
    """
    **Limitation:** Appointment volume alone cannot determine
    the number of doctors, nurses or other staff required.
    Staff availability, appointment duration, service capacity
    and staffing ratios would also be needed.
    """
)

st.caption(
    "Analysis is based on historical appointment records "
    "and the selected date range."
)
