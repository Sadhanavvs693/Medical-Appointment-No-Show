
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="No-Show Analysis",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Medical_appointment_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# --------------------------------------------------
# Prepare data
# --------------------------------------------------

df["no_show_flag"] = (
    df["no_show"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["yes", "1", "true", "y"])
)

df["attendance_status"] = df["no_show_flag"].map({
    True: "No-show",
    False: "Attended"
})

df["age_group"] = pd.cut(
    df["age"],
    bins=[-1, 12, 18, 30, 45, 60, float("inf")],
    labels=[
        "Under 12",
        "13–18",
        "19–30",
        "31–45",
        "46–60",
        "Over 60"
    ]
)

# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("📊 Medical Appointment No-Show Analysis")

st.write(
    "Explore patient attendance patterns and identify "
    "factors associated with missed medical appointments."
)

st.divider()

# --------------------------------------------------
# Key metrics
# --------------------------------------------------

total_appointments = len(df)
total_no_shows = int(df["no_show_flag"].sum())
total_attended = total_appointments - total_no_shows
no_show_rate = (total_no_shows / total_appointments) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Appointments", f"{total_appointments:,}")
col2.metric("Attended", f"{total_attended:,}")
col3.metric("No-Shows", f"{total_no_shows:,}")
col4.metric("No-Show Rate", f"{no_show_rate:.2f}%")

st.divider()

# --------------------------------------------------
# Attendance distribution
# --------------------------------------------------

st.subheader("1. Attendance Distribution")

attendance_counts = (
    df["attendance_status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Appointments")
)

fig = px.pie(
    attendance_counts,
    names="Status",
    values="Appointments",
    hole=0.45,
    title="Attended vs No-Show Appointments"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# No-show by gender and age
# --------------------------------------------------

st.subheader("2. No-Show Patterns by Patient Demographics")

col1, col2 = st.columns(2)

with col1:
    gender_data = (
        df.groupby("gender", observed=True)["no_show_flag"]
        .mean()
        .mul(100)
        .reset_index(name="No-Show Rate")
    )

    fig_gender = px.bar(
        gender_data,
        x="gender",
        y="No-Show Rate",
        title="No-Show Rate by Gender",
        labels={
            "gender": "Gender",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    age_data = (
        df.groupby("age_group", observed=True)["no_show_flag"]
        .mean()
        .mul(100)
        .reset_index(name="No-Show Rate")
    )

    fig_age = px.bar(
        age_data,
        x="age_group",
        y="No-Show Rate",
        title="No-Show Rate by Age Group",
        labels={
            "age_group": "Age Group",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig_age, use_container_width=True)

st.divider()

# --------------------------------------------------
# Specialty analysis
# --------------------------------------------------

st.subheader("3. Specialty-wise No-Show Analysis")

specialty_data = (
    df.groupby("specialty")
    .agg(
        Total_Appointments=("no_show_flag", "size"),
        No_Show_Rate=("no_show_flag", "mean")
    )
    .reset_index()
)

specialty_data["No_Show_Rate"] *= 100

specialty_data = specialty_data.sort_values(
    "No_Show_Rate",
    ascending=False
)

fig_specialty = px.bar(
    specialty_data,
    x="specialty",
    y="No_Show_Rate",
    hover_data=["Total_Appointments"],
    title="No-Show Rate by Medical Specialty",
    labels={
        "specialty": "Specialty",
        "No_Show_Rate": "No-Show Rate (%)"
    }
)

st.plotly_chart(fig_specialty, use_container_width=True)

st.divider()

# --------------------------------------------------
# SMS analysis
# --------------------------------------------------

st.subheader("4. SMS Reminder Analysis")

sms_data = (
    df.groupby("SMS_received")["no_show_flag"]
    .mean()
    .mul(100)
    .reset_index(name="No-Show Rate")
)

sms_data["SMS Status"] = sms_data["SMS_received"].map({
    0: "SMS Not Received",
    1: "SMS Received"
})

fig_sms = px.bar(
    sms_data,
    x="SMS Status",
    y="No-Show Rate",
    title="No-Show Rate by SMS Reminder Status",
    labels={"No-Show Rate": "No-Show Rate (%)"}
)

st.plotly_chart(fig_sms, use_container_width=True)

st.divider()

# --------------------------------------------------
# Appointment shift analysis
# --------------------------------------------------

st.subheader("5. Appointment Shift Analysis")

shift_data = (
    df.groupby("appointment_shift")["no_show_flag"]
    .mean()
    .mul(100)
    .reset_index(name="No-Show Rate")
)

fig_shift = px.bar(
    shift_data,
    x="appointment_shift",
    y="No-Show Rate",
    title="No-Show Rate by Appointment Shift",
    labels={
        "appointment_shift": "Appointment Shift",
        "No-Show Rate": "No-Show Rate (%)"
    }
)

st.plotly_chart(fig_shift, use_container_width=True)

st.caption(
    "Note: These charts show associations in historical data. "
    "They do not establish that any individual factor causes a no-show."
)
