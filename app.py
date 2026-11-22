
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Medical_appointment_data.csv"
DEMAND_PATH = BASE_DIR / "data" / "final_daily_demand.csv"
COMPARISON_PATH = BASE_DIR / "data" / "forecasting_model_comparison.csv"

MODEL_PATH = BASE_DIR / "models" / "medical_no_show_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "medical_no_show_preprocessor.pkl"


# =========================================================
# 2. CUSTOM SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    """
    <h2 style="color:#1769AA; margin-bottom:0;">
        🏥 Healthcare Analytics
    </h2>
    <p style="color:gray; margin-top:4px;">
        No-Show & Demand Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.markdown("### Dashboard Navigation")

pages = [
    "Home",
    "No Show Prediction",
    "No Show Analysis",
    "Demand Analysis",
    "Demand Forecasting",
    "Resource Planning"
]

page_icons = {
    "Home": "🏠",
    "No Show Prediction": "🔍",
    "No Show Analysis": "📊",
    "Demand Analysis": "📅",
    "Demand Forecasting": "📈",
    "Resource Planning": "🏥"
}

selected_page = st.sidebar.radio(
    "Navigate",
    pages,
    format_func=lambda page: f"{page_icons[page]}  {page}",
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("Healthcare Operations & Resource Management")


# =========================================================
# 3. SHARED DATA LOADING
# =========================================================

@st.cache_data
def load_main_data():
    df = pd.read_csv(DATA_PATH)

    df["appointment_date_continuous"] = pd.to_datetime(
        df["appointment_date_continuous"]
    )

    return df


@st.cache_data
def load_forecasting_data():
    demand = pd.read_csv(DEMAND_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)

    demand["appointment_date_continuous"] = pd.to_datetime(
        demand["appointment_date_continuous"]
    )

    demand = demand.sort_values("appointment_date_continuous")

    demand = (
        demand.set_index("appointment_date_continuous")
        .asfreq("D")
    )

    demand["appointment_count"] = (
        demand["appointment_count"].fillna(0)
    )

    return demand, comparison


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def check_main_data():
    if not DATA_PATH.exists():
        st.error(
            "Medical appointment dataset not found. "
            "Please check the data folder."
        )
        return None

    return load_main_data()


# =========================================================
# 4. HOME PAGE
# =========================================================

def show_home():

    st.title("🏥 Healthcare Analytics Dashboard")

    st.write(
        """
        An interactive dashboard for understanding medical appointment
        no-shows, historical demand patterns, demand forecasting and
        healthcare resource planning.
        """
    )

    st.divider()

    if not DATA_PATH.exists():
        st.error("Medical appointment dataset not found.")
        return

    df = check_main_data()

    if df is None:
        return

    total_appointments = len(df)
    total_features = len(df.columns)

    no_show_flag = (
        df["no_show"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "1", "true", "y"])
    )

    no_show_rate = no_show_flag.mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Appointments",
        f"{total_appointments:,}"
    )

    col2.metric(
        "Dataset Features",
        f"{total_features}"
    )

    col3.metric(
        "No-Show Rate",
        f"{no_show_rate:.2f}%"
    )

    st.caption(
        "Dataset coverage: "
        f"{df['appointment_date_continuous'].min():%d %B %Y}"
        " – "
        f"{df['appointment_date_continuous'].max():%d %B %Y}"
    )

    st.divider()

    st.subheader("Dashboard Modules")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🔍 No-Show Prediction")
            st.write(
                "Estimate the likelihood of a patient missing "
                "a medical appointment using the trained ML model."
            )

        with st.container(border=True):
            st.subheader("📅 Demand Analysis")
            st.write(
                "Explore daily, weekly and monthly appointment "
                "demand patterns."
            )

        with st.container(border=True):
            st.subheader("🏥 Resource Planning")
            st.write(
                "Explore specialty-level and weekday appointment "
                "demand to support resource planning."
            )

    with col2:
        with st.container(border=True):
            st.subheader("📊 No-Show Analysis")
            st.write(
                "Understand no-show patterns across age groups, "
                "gender, specialties, SMS reminders and shifts."
            )

        with st.container(border=True):
            st.subheader("📈 Demand Forecasting")
            st.write(
                "Compare forecasting models and explore the "
                "experimental 30-day SARIMA forecast."
            )

    st.divider()

    with st.expander("Project Information"):

        st.markdown("**Project Objective**")

        st.write(
            """
            Analyze medical appointment records to understand and
            predict patient no-shows, examine appointment demand
            patterns and explore how historical insights can support
            healthcare operations.
            """
        )

        st.markdown("**Dataset Information**")

        st.write(
            """
            - Total records: 109,593
            - Features: 26
            - Domain: Healthcare Operations & Resource Management
            - Dataset period: January 2020 – May 2021
            """
        )

        st.markdown("**Machine Learning Approach**")

        st.write(
            """
            Data preprocessing, exploratory data analysis,
            classification and model evaluation.
            """
        )

        st.markdown("**Forecasting Approach**")

        st.write(
            """
            Historical demand analysis, time-series model comparison
            and SARIMA-based experimental forecasting.
            """
        )


# =========================================================
# 5. NO-SHOW PREDICTION
# =========================================================

def show_no_show_prediction():

    st.title("🏥 Medical Appointment No-Show Prediction")

    st.write(
        "Enter appointment and patient details to estimate "
        "the likelihood of a no-show."
    )

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.error("Model or preprocessing file not found.")
        return

    try:
        model, preprocessor = load_model_files()
    except Exception as error:
        st.error(f"Could not load model files: {error}")
        return

    st.divider()

    specialty_options = [
        "physiotherapy",
        "psychotherapy",
        "speech therapy",
        "occupational therapy",
        "pedagogo",
        "enf",
        "assist",
        "sem especialidade",
        "Unknown"
    ]

    with st.form("prediction_form"):

        st.subheader("1. Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            specialty = st.selectbox(
                "Medical Specialty",
                specialty_options
            )

            gender = st.selectbox(
                "Gender",
                ["F", "M", "I"]
            )

            age = st.slider(
                "Age",
                min_value=2,
                max_value=100,
                value=35
            )

        with col2:
            disability = st.selectbox(
                "Disability",
                ["intellectual", "motor", "Unknown"]
            )

            patient_needs_companion = st.selectbox(
                "Needs Companion",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            diabetes = st.selectbox(
                "Diabetes",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        st.divider()
        st.subheader("2. Appointment Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            appointment_shift = st.selectbox(
                "Appointment Shift",
                ["morning", "afternoon"]
            )

            appointment_time = st.slider(
                "Appointment Time",
                min_value=7,
                max_value=18,
                value=10
            )

        with col2:
            appointment_year = st.selectbox(
                "Appointment Year",
                [2020, 2021]
            )

            appointment_month = st.selectbox(
                "Appointment Month",
                list(range(1, 13))
            )

        with col3:
            appointment_day = st.slider(
                "Appointment Day",
                min_value=1,
                max_value=31,
                value=15
            )

            appointment_dayofweek = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

        st.divider()
        st.subheader("3. Health and Reminder Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            alcoholism = st.selectbox(
                "Alcoholism",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            handcap = st.selectbox(
                "Handicap",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col2:
            scholarship = st.selectbox(
                "Scholarship",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            sms_received = st.selectbox(
                "SMS Received",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            rainy_day_before = st.selectbox(
                "Rainy Day Before",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            storm_day_before = st.selectbox(
                "Storm Day Before",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        st.divider()
        st.subheader("4. Weather Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            rain_intensity = st.selectbox(
                "Rain Intensity",
                ["no_rain", "weak", "moderate", "heavy"]
            )

            average_rain_day = st.number_input(
                "Average Rain per Day",
                min_value=0.0,
                max_value=5.0,
                value=0.1
            )

        with col2:
            heat_intensity = st.selectbox(
                "Heat Intensity",
                ["mild", "cold", "warm", "heavy_cold", "heavy_warm"]
            )

            average_temp_day = st.number_input(
                "Average Temperature",
                min_value=8.0,
                max_value=35.0,
                value=20.0
            )

        with col3:
            max_temp_day = st.number_input(
                "Maximum Temperature",
                min_value=8.0,
                max_value=40.0,
                value=24.0
            )

            max_rain_day = st.number_input(
                "Maximum Rain",
                min_value=0.0,
                max_value=50.0,
                value=0.2
            )

        submitted = st.form_submit_button(
            "Predict Appointment Status",
            use_container_width=True
        )

    if submitted:

        if appointment_time <= 11:
            time_of_day = "Morning"
        elif appointment_time <= 14:
            time_of_day = "Afternoon"
        else:
            time_of_day = "Evening"

        if age <= 12:
            age_group = "Under 12"
        elif age <= 18:
            age_group = "13-18"
        elif age <= 30:
            age_group = "19-30"
        elif age <= 45:
            age_group = "31-45"
        elif age <= 60:
            age_group = "46-60"
        else:
            age_group = "Over 60"

        under_12_years_old = 1 if age <= 12 else 0
        over_60_years_old = 1 if age > 60 else 0

        is_weekend = (
            1 if appointment_dayofweek >= 5 else 0
        )

        input_data = {
            "specialty": specialty,
            "gender": gender,
            "disability": disability,
            "appointment_shift": appointment_shift,
            "rain_intensity": rain_intensity,
            "heat_intensity": heat_intensity,
            "time_of_day": time_of_day,
            "age_group": age_group,
            "appointment_time": appointment_time,
            "age": age,
            "under_12_years_old": under_12_years_old,
            "over_60_years_old": over_60_years_old,
            "patient_needs_companion": patient_needs_companion,
            "average_temp_day": average_temp_day,
            "average_rain_day": average_rain_day,
            "max_temp_day": max_temp_day,
            "max_rain_day": max_rain_day,
            "rainy_day_before": rainy_day_before,
            "storm_day_before": storm_day_before,
            "Hipertension": hypertension,
            "Diabetes": diabetes,
            "Alcoholism": alcoholism,
            "Handcap": handcap,
            "Scholarship": scholarship,
            "SMS_received": sms_received,
            "age_missing": 0,
            "appointment_year": appointment_year,
            "appointment_month": appointment_month,
            "appointment_dayofweek": appointment_dayofweek,
            "is_weekend": is_weekend,
            "appointment_day": appointment_day
        }

        try:
            input_df = pd.DataFrame([input_data])

            transformed_data = preprocessor.transform(input_df)

            prediction = model.predict(transformed_data)[0]
            probabilities = model.predict_proba(transformed_data)[0]

            classes = list(model.classes_)

            no_show_probability = probabilities[
                classes.index(1)
            ] if 1 in classes else 0

            attended_probability = probabilities[
                classes.index(0)
            ] if 0 in classes else 0

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Predicted Status: No-show")
            else:
                st.success("✅ Predicted Status: Attended")

            col1, col2 = st.columns(2)

            col1.metric(
                "No-Show Probability",
                f"{no_show_probability * 100:.2f}%"
            )

            col2.metric(
                "Attendance Probability",
                f"{attended_probability * 100:.2f}%"
            )

            st.write("No-show probability")
            st.progress(float(no_show_probability))

            st.caption(
                "These probabilities are model estimates, "
                "not guarantees of actual patient behaviour."
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


# =========================================================
# 6. NO-SHOW ANALYSIS
# =========================================================

def show_no_show_analysis():

    st.title("📊 Medical Appointment No-Show Analysis")

    st.write(
        "Explore appointment attendance patterns and factors "
        "associated with patient no-shows."
    )

    df = check_main_data()

    if df is None:
        return

    df = df.copy()

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
            "Under 12", "13–18", "19–30",
            "31–45", "46–60", "Over 60"
        ]
    )

    total = len(df)
    no_shows = int(df["no_show_flag"].sum())
    attended = total - no_shows
    no_show_rate = no_shows / total * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Appointments", f"{total:,}")
    col2.metric("Attended", f"{attended:,}")
    col3.metric("No-Shows", f"{no_shows:,}")
    col4.metric("No-Show Rate", f"{no_show_rate:.2f}%")

    st.divider()

    st.subheader("1. Attendance Distribution")

    attendance_counts = (
        df["attendance_status"]
        .value_counts()
        .reset_index()
    )

    attendance_counts.columns = [
        "Attendance Status", "Appointments"
    ]

    fig = px.pie(
        attendance_counts,
        names="Attendance Status",
        values="Appointments",
        hole=0.45,
        title="Appointment Attendance Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("2. No-Show Rate by Gender")

    gender_data = (
        df.groupby("gender")
        .agg(
            Total=("no_show_flag", "size"),
            No_Shows=("no_show_flag", "sum")
        )
        .reset_index()
    )

    gender_data["No-Show Rate"] = (
        gender_data["No_Shows"] /
        gender_data["Total"] * 100
    )

    fig = px.bar(
        gender_data,
        x="gender",
        y="No-Show Rate",
        title="No-Show Rate by Gender",
        labels={
            "gender": "Gender",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. No-Show Rate by Age Group")

    age_data = (
        df.groupby("age_group", observed=True)
        .agg(
            Total=("no_show_flag", "size"),
            No_Shows=("no_show_flag", "sum")
        )
        .reset_index()
    )

    age_data["No-Show Rate"] = (
        age_data["No_Shows"] /
        age_data["Total"] * 100
    )

    fig = px.bar(
        age_data,
        x="age_group",
        y="No-Show Rate",
        title="No-Show Rate by Age Group",
        labels={
            "age_group": "Age Group",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("4. No-Show Rate by Specialty")

    specialty_data = (
        df.groupby("specialty")
        .agg(
            Total=("no_show_flag", "size"),
            No_Shows=("no_show_flag", "sum")
        )
        .reset_index()
    )

    specialty_data["No-Show Rate"] = (
        specialty_data["No_Shows"] /
        specialty_data["Total"] * 100
    )

    specialty_data = specialty_data.sort_values(
        "No-Show Rate",
        ascending=False
    )

    fig = px.bar(
        specialty_data,
        x="specialty",
        y="No-Show Rate",
        hover_data=["Total"],
        title="No-Show Rate by Medical Specialty",
        labels={
            "specialty": "Medical Specialty",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("5. No-Show Rate by SMS Reminder")

    sms_data = (
        df.groupby("SMS_received")
        .agg(
            Total=("no_show_flag", "size"),
            No_Shows=("no_show_flag", "sum")
        )
        .reset_index()
    )

    sms_data["SMS Status"] = sms_data["SMS_received"].map({
        0: "SMS Not Received",
        1: "SMS Received"
    })

    sms_data["No-Show Rate"] = (
        sms_data["No_Shows"] /
        sms_data["Total"] * 100
    )

    fig = px.bar(
        sms_data,
        x="SMS Status",
        y="No-Show Rate",
        title="No-Show Rate by SMS Reminder",
        labels={
            "SMS Status": "SMS Reminder Status",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("6. No-Show Rate by Appointment Shift")

    shift_data = (
        df.groupby("appointment_shift")
        .agg(
            Total=("no_show_flag", "size"),
            No_Shows=("no_show_flag", "sum")
        )
        .reset_index()
    )

    shift_data["No-Show Rate"] = (
        shift_data["No_Shows"] /
        shift_data["Total"] * 100
    )

    fig = px.bar(
        shift_data,
        x="appointment_shift",
        y="No-Show Rate",
        title="No-Show Rate by Appointment Shift",
        labels={
            "appointment_shift": "Appointment Shift",
            "No-Show Rate": "No-Show Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts show associations in the historical dataset; "
        "they do not establish cause and effect."
    )


# =========================================================
# 7. DEMAND ANALYSIS
# =========================================================

def show_demand_analysis():

    st.title("📅 Medical Appointment Demand Analysis")

    st.write(
        "Explore daily, weekly and monthly appointment demand "
        "across the selected period."
    )

    df = check_main_data()

    if df is None:
        return

    df = df.copy()

    min_date = df["appointment_date_continuous"].min().date()
    max_date = df["appointment_date_continuous"].max().date()

    st.subheader("1. Select Analysis Period")

    selected_dates = st.date_input(
        "Appointment date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Please select both a start date and an end date.")
        return

    start_date, end_date = selected_dates

    filtered_df = df[
        df["appointment_date_continuous"].dt.date.between(
            start_date, end_date
        )
    ].copy()

    if filtered_df.empty:
        st.warning("No appointments found for the selected period.")
        return

    st.divider()

    daily_demand = (
        filtered_df.groupby("appointment_date_continuous")
        .size()
    )

    total_appointments = len(filtered_df)
    active_days = daily_demand.size
    average_daily_demand = daily_demand.mean()
    maximum_daily_demand = daily_demand.max()

    st.subheader("2. Demand Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Appointments", f"{total_appointments:,}")
    col2.metric("Active Days", f"{active_days:,}")
    col3.metric("Average Daily Demand", f"{average_daily_demand:,.1f}")
    col4.metric("Maximum Daily Demand", f"{maximum_daily_demand:,}")

    st.divider()

    st.subheader("3. Daily Appointment Demand")

    daily_chart = (
        daily_demand
        .reset_index(name="Appointments")
    )

    fig = px.line(
        daily_chart,
        x="appointment_date_continuous",
        y="Appointments",
        title="Daily Appointment Demand",
        labels={
            "appointment_date_continuous": "Appointment Date",
            "Appointments": "Number of Appointments"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("4. Weekly Appointment Demand")

    weekly_demand = (
        filtered_df.set_index("appointment_date_continuous")
        .resample("W-SUN")
        .size()
        .reset_index(name="Appointments")
    )

    fig = px.bar(
        weekly_demand,
        x="appointment_date_continuous",
        y="Appointments",
        title="Weekly Appointment Demand",
        labels={
            "appointment_date_continuous": "Week",
            "Appointments": "Number of Appointments"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("5. Monthly Appointment Demand")

    monthly_demand = (
        filtered_df.set_index("appointment_date_continuous")
        .resample("MS")
        .size()
        .reset_index(name="Appointments")
    )

    fig = px.bar(
        monthly_demand,
        x="appointment_date_continuous",
        y="Appointments",
        title="Monthly Appointment Demand",
        labels={
            "appointment_date_continuous": "Month",
            "Appointments": "Number of Appointments"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("6. Appointment Demand by Specialty")

    specialty_demand = (
        filtered_df.groupby("specialty")
        .size()
        .reset_index(name="Appointments")
        .sort_values("Appointments", ascending=False)
    )

    fig = px.bar(
        specialty_demand,
        x="specialty",
        y="Appointments",
        title="Appointment Demand by Specialty",
        labels={
            "specialty": "Medical Specialty",
            "Appointments": "Number of Appointments"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("7. Appointment Demand by Weekday")

    filtered_df["weekday"] = (
        filtered_df["appointment_date_continuous"].dt.day_name()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_demand = (
        filtered_df.groupby("weekday")
        .size()
        .reindex(weekday_order, fill_value=0)
        .reset_index(name="Appointments")
    )

    fig = px.bar(
        weekday_demand,
        x="weekday",
        y="Appointments",
        title="Total Appointments by Weekday",
        labels={
            "weekday": "Day of Week",
            "Appointments": "Number of Appointments"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("View Daily Demand Data"):
        st.dataframe(
            daily_chart,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Specialty Demand Data"):
        st.dataframe(
            specialty_demand,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Monthly Demand Data"):
        st.dataframe(
            monthly_demand,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Demand represents the number of appointment records "
        "for each selected period."
    )


# =========================================================
# 8. DEMAND FORECASTING
# =========================================================

def show_demand_forecasting():

    st.title("📈 Medical Appointment Demand Forecasting")

    st.write(
        "Explore historical appointment demand, compare forecasting "
        "models and view a 30-day demand forecast."
    )

    if not DEMAND_PATH.exists() or not COMPARISON_PATH.exists():
        st.error(
            "Required forecasting CSV files are missing from the data folder."
        )
        return

    demand, comparison = load_forecasting_data()

    st.divider()

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

        fitted_model = sarima_model.fit(
            disp=False,
            maxiter=200
        )

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

        forecast_df["Forecast"] = forecast_df["Forecast"].clip(lower=0)
        forecast_df["Lower Limit"] = forecast_df["Lower Limit"].clip(lower=0)
        forecast_df["Upper Limit"] = forecast_df["Upper Limit"].clip(lower=0)

        col1, col2, col3 = st.columns(3)

        col1.metric("Forecast Period", "30 Days")

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

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )

        st.subheader("Forecast Values")

        st.dataframe(
            forecast_df.round(2),
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.error(
            f"Could not generate the SARIMA forecast: {error}"
        )

    st.divider()

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


# =========================================================
# 9. RESOURCE PLANNING
# =========================================================

def show_resource_planning():

    st.title("🏥 Healthcare Resource Planning")

    st.write(
        "Explore appointment demand across medical specialties "
        "and weekdays to support healthcare resource planning."
    )

    df = check_main_data()

    if df is None:
        return

    df = df.copy()

    df["specialty"] = (
        df["specialty"]
        .fillna("Unknown")
        .astype(str)
    )

    st.divider()

    st.subheader("1. Select Analysis Period")

    min_date = df["appointment_date_continuous"].min().date()
    max_date = df["appointment_date_continuous"].max().date()

    selected_dates = st.date_input(
        "Appointment date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Please select both a start date and an end date.")
        return

    start_date, end_date = selected_dates

    filtered_df = df[
        df["appointment_date_continuous"].dt.date.between(
            start_date, end_date
        )
    ].copy()

    if filtered_df.empty:
        st.warning("No appointments found for the selected period.")
        return

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

    st.plotly_chart(
        fig_specialty,
        use_container_width=True
    )

    st.divider()

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

    st.plotly_chart(
        fig_weekday,
        use_container_width=True
    )

    st.divider()

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

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )

    st.divider()

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


# =========================================================
# 10. DISPLAY SELECTED PAGE
# =========================================================

if selected_page == "Home":
    show_home()

elif selected_page == "No Show Prediction":
    show_no_show_prediction()

elif selected_page == "No Show Analysis":
    show_no_show_analysis()

elif selected_page == "Demand Analysis":
    show_demand_analysis()

elif selected_page == "Demand Forecasting":
    show_demand_forecasting()

elif selected_page == "Resource Planning":
    show_resource_planning()
