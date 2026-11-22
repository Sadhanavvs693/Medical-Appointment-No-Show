
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Load model and preprocessing files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "medical_no_show_model.pkl")
    preprocessor = joblib.load(BASE_DIR / "medical_no_show_preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_files()

# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("🏥 Medical Appointment No-Show Prediction")

st.write(
    "Enter the appointment and patient details to estimate "
    "whether the patient is likely to attend or miss the appointment."
)

st.divider()

# --------------------------------------------------
# Patient and appointment details
# --------------------------------------------------

st.subheader("1. Patient and Appointment Details")

col1, col2, col3 = st.columns(3)

with col1:
    specialty = st.selectbox(
        "Specialty",
        [
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
    )

    gender = st.selectbox("Gender", ["F", "M", "I"])

    age = st.slider("Age", 2, 100, 35)

with col2:
    disability = st.selectbox(
        "Disability",
        ["intellectual", "motor", "Unknown"]
    )

    appointment_shift = st.selectbox(
        "Appointment Shift",
        ["morning", "afternoon"]
    )

    appointment_time = st.slider(
        "Appointment Time",
        7, 18, 10
    )

with col3:
    appointment_year = st.selectbox(
        "Appointment Year", [2020, 2021]
    )

    appointment_month = st.selectbox(
        "Appointment Month", list(range(1, 13))
    )

    appointment_day = st.slider(
        "Appointment Day", 1, 31, 15
    )

    appointment_dayofweek = st.selectbox(
        "Day of Week",
        list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ][x]
    )

is_weekend = 1 if appointment_dayofweek >= 5 else 0

# --------------------------------------------------
# Patient health and support details
# --------------------------------------------------

st.divider()
st.subheader("2. Patient Health and Support Details")

col1, col2, col3 = st.columns(3)

with col1:
    patient_needs_companion = st.selectbox(
        "Patient Needs Companion",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    hypertension = st.selectbox(
        "Hypertension", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    diabetes = st.selectbox(
        "Diabetes", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    alcoholism = st.selectbox(
        "Alcoholism", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    handcap = st.selectbox(
        "Handicap", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    scholarship = st.selectbox(
        "Scholarship", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

sms_received = st.selectbox(
    "SMS Received",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# --------------------------------------------------
# Weather details
# --------------------------------------------------

st.divider()
st.subheader("3. Weather Details")

col1, col2, col3 = st.columns(3)

with col1:
    rain_intensity = st.selectbox(
        "Rain Intensity",
        ["no_rain", "weak", "moderate", "heavy"]
    )

    average_rain_day = st.number_input(
        "Average Rainfall",
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
        "Maximum Rainfall",
        min_value=0.0,
        max_value=50.0,
        value=0.2
    )

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

# --------------------------------------------------
# Derived features
# --------------------------------------------------

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
age_missing = 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict No-Show", type="primary", use_container_width=True):

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

        "age_missing": age_missing,
        "appointment_year": appointment_year,
        "appointment_month": appointment_month,
        "appointment_dayofweek": appointment_dayofweek,
        "is_weekend": is_weekend,
        "appointment_day": appointment_day
    }

    input_df = pd.DataFrame([input_data])

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)[0]
    probabilities = model.predict_proba(input_processed)[0]

    classes = list(model.classes_)
    attended_probability = float(probabilities[classes.index(0)])
    no_show_probability = float(probabilities[classes.index(1)])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Predicted: No-show")
    else:
        st.success("✅ Predicted: Attended")

    col1, col2 = st.columns(2)

    col1.metric(
        "No-Show Probability",
        f"{no_show_probability * 100:.2f}%"
    )

    col2.metric(
        "Attendance Probability",
        f"{attended_probability * 100:.2f}%"
    )

    st.progress(no_show_probability)

    st.caption(
        "Probabilities are model estimates, not guarantees "
        "of actual patient attendance."
    )
