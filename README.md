# Medical Appointment No-Show Prediction & Business Analytics

## Project Overview

This project analyzes medical appointment data to understand and predict patient no-shows and translate the findings into practical healthcare business use cases.

The project combines:

* Exploratory Data Analysis (EDA)
* Data cleaning and feature engineering
* Statistical analysis and hypothesis testing
* Machine Learning classification
* Time-series forecasting
* Business analytics
* Streamlit deployment

The goal is to help healthcare organizations understand appointment attendance patterns, identify higher-risk appointments, and support better planning of staffing, resources, and appointment capacity.

---

## Dataset

The dataset contains **109,593 original records and 26 columns**.

After removing 36 exact duplicate records, the analysis contains **109,557 appointments**.

The dataset includes information related to:

* Patient demographics
* Medical conditions
* Appointment timing
* Appointment specialty
* Location
* Weather conditions
* SMS reminders
* Appointment attendance

> The raw dataset is not publicly redistributed because it was provided for project/learning purposes.

---

## Project Workflow

```text
Raw Medical Appointment Data
            ↓
Data Cleaning
            ↓
Exploratory Data Analysis
            ↓
Feature Engineering
            ↓
Statistical Analysis
            ↓
Machine Learning
            ↓
No-Show Risk Prediction
            ↓
Time-Series Forecasting
            ↓
7 Healthcare Business Cases
            ↓
Streamlit Application
```

---

# Machine Learning — No-Show Prediction

### Target Variable

The target variable is `no_show`.

It was converted into a numerical binary target:

```text
no → 0  (Attended)
yes → 1 (No-show)
```

### Class Distribution

| Outcome  | Appointments | Percentage |
| -------- | -----------: | ---------: |
| Attended |       74,726 |     68.21% |
| No-show  |       34,831 |     31.79% |

### Feature Engineering

Important engineered features include:

* `age_missing`
* `appointment_year`
* `appointment_month`
* `appointment_day`
* `appointment_dayofweek`
* `is_weekend`
* `time_of_day`
* `age_group`

Categorical variables were encoded using `OneHotEncoder`, while numerical variables were standardized using `StandardScaler`.

---

## Models Evaluated

### Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 63.41% |
| Precision | 44.30% |
| Recall    | 58.73% |
| F1 Score  | 50.51% |
| ROC-AUC   | 0.6622 |

### Tuned Random Forest

The final Random Forest model used:

* 150 trees
* Maximum depth of 12
* Minimum samples split of 10
* Minimum samples leaf of 5
* Balanced class weights

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 70.70% |
| Precision | 53.60% |
| Recall    | 58.37% |
| F1 Score  | 55.88% |
| ROC-AUC   | 0.7388 |

The tuned Random Forest showed a substantially smaller train-test accuracy gap than the original unrestricted Random Forest, reducing overfitting.

---

# Business Cases

The project extends the prediction model into **7 healthcare business use cases**.

## 1. Risk-Based Patient Engagement

Predicted no-show probabilities were converted into business-defined risk groups.

| Risk Group | Predicted Risk | Observed No-Show Rate | Suggested Action                   |
| ---------- | -------------- | --------------------: | ---------------------------------- |
| Low        | <30%           |                 4.27% | Standard reminder                  |
| Medium     | 30–60%         |                25.56% | SMS reminder + confirmation        |
| High       | >60%           |                63.48% | Priority SMS + rescheduling option |

These thresholds are business-defined and should be validated using operational outcomes before real-world deployment.

---

## 2. Intelligent Staffing Optimization

Historical appointment volume was analyzed by:

* Day of week
* Specialty
* Shift
* Month

A Holt-Winters time-series model was used to forecast daily appointment volume.

The log-transformed forecasting model achieved:

* **MAE: 4.91**
* **RMSE: 5.94**

Specialty and shift analysis showed that workload varies considerably across services.

The analysis can support capacity planning while recognizing that actual staffing also depends on appointment duration, specialist availability, patient complexity, and operational constraints.

---

## 3. Revenue Protection

The dataset contained:

* **34,831 no-show appointments**
* **31.79% overall no-show rate**

The three specialties with the largest number of no-shows were:

* Psychotherapy: 9,263
* Physiotherapy: 7,218
* Speech therapy: 6,234

Together they represented **22,715 no-shows**, or **65.21% of all missed appointments**.

Scenario analysis estimated potential recovered appointment capacity under different reductions in no-shows:

| No-show Reduction | Potentially Recovered Appointments |
| ----------------: | ---------------------------------: |
|               10% |                              3,483 |
|               20% |                              6,966 |
|               30% |                             10,449 |

These are scenario estimates rather than guaranteed financial savings.

---

## 4. Geographic Resource Planning

The `place` field contains a large number of unique location values.

Rather than assuming every unique value represents a city, the analysis treated `place` as a geographic location field.

Among locations with at least 100 appointments, demand was concentrated in a small number of locations.

The three highest-volume locations accounted for:

**32,053 appointments — 29.26% of all appointments.**

This analysis can support geographic resource allocation and identification of locations requiring additional operational attention.

---

## 5. Seasonal & Weather Adaptations

The project examined:

* Monthly no-show rates
* Rain intensity
* Heat intensity
* Previous-day rain
* Previous-day storms

Overall no-show rate:

**31.79%**

No-show rates were relatively stable across most months, while larger differences were observed across heat-intensity categories.

Chi-square tests found statistically significant associations between no-show status and:

* Rainy day before appointment
* Storm day before appointment
* Heat intensity

These findings indicate **associations, not causation**.

---

## 6. Health-Based Segmentation

Health-related variables were analyzed to understand differences in appointment attendance patterns.

The analysis examined:

* Hypertension
* Diabetes
* Alcoholism
* Disability
* Age groups
* Companion requirements
* SMS reminders

Chi-square testing found statistically significant associations between no-show status and:

* Hypertension
* Diabetes
* Alcoholism

`Handcap` did not show a statistically significant association in this analysis.

Health-related findings should be used responsibly and should not be used for clinical decisions or discriminatory treatment.

---

## 7. Specialty-Level Demand Planning

Historical appointment demand was analyzed across specialties.

The largest specialty groups were:

| Specialty            | Appointments | Demand Share |
| -------------------- | -----------: | -----------: |
| Psychotherapy        |       28,642 |       26.14% |
| Speech therapy       |       22,321 |       20.37% |
| Physiotherapy        |       21,001 |       19.17% |
| Occupational therapy |       11,318 |       10.33% |

Psychotherapy, speech therapy, and physiotherapy together represented:

**65.68% of total historical appointment demand.**

Demand was also compared across morning and afternoon shifts and across 17 months of historical data.

---

# Streamlit Application

A Streamlit application was created to demonstrate individual appointment no-show prediction.

The application loads the saved:

* Random Forest model
* Preprocessor
* Feature configuration

and generates a predicted no-show probability and risk category.

### Run the application

```bash
streamlit run app.py
```

---

# Repository Structure

```text
Medical-Appointment-No-Show/
│
├── data/
│   └── Medical_appointment_data.csv
│
├── notebooks/
│   ├── medical_appointment_analysis.ipynb
│   ├── medical_appointment_business_cases.ipynb
│   │
│   └── business_cases/
│       ├── case_1_risk_based_patient_engagement.ipynb
│       ├── case_2_intelligent_staffing_optimization.ipynb
│       ├── case_3_revenue_protection.ipynb
│       ├── case_4_geographic_resource_planning.ipynb
│       ├── case_5_seasonal_weather_adaptations.ipynb
│       ├── case_6_health_based_segmentation.ipynb
│       └── case_7_specialty_level_demand_planning.ipynb
│
├── app.py
├── medical_no_show_model.pkl
├── medical_no_show_preprocessor.pkl
├── medical_no_show_features.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* Scikit-learn
* Statsmodels
* Joblib
* Streamlit
* Jupyter Notebook

---

# Key Skills Demonstrated

### Data Analysis

* Data cleaning
* Missing-value handling
* Duplicate removal
* Exploratory Data Analysis
* Univariate and bivariate analysis
* GroupBy analysis
* Feature engineering

### Statistics

* Chi-square hypothesis testing
* P-values
* Association analysis
* Statistical interpretation

### Machine Learning

* Train-test split
* Stratified sampling
* One-hot encoding
* Feature scaling
* Logistic Regression
* Random Forest
* Hyperparameter tuning
* Class imbalance handling
* Confusion matrix
* Precision
* Recall
* F1 score
* ROC-AUC
* Overfitting analysis

### Time-Series Analysis

* Chronological train-test split
* Holt-Winters Exponential Smoothing
* Log transformation
* MAE
* RMSE
* Daily demand forecasting

### Deployment

* Streamlit
* Saved ML model
* Saved preprocessing pipeline
* Prediction interface

---

# Important Limitations

* The dataset represents a historical period from **2020 to 2021**.
* Historical patterns may not represent current healthcare demand.
* The model predicts statistical risk and does not determine why an individual patient may miss an appointment.
* Weather relationships represent associations rather than causal effects.
* Health-related variables require careful privacy, fairness, and governance considerations.
* Revenue loss could not be calculated because appointment prices and actual revenue data were not provided.
* Geographic analysis is limited by the quality and structure of the `place` field.
* Staffing recommendations should consider operational factors beyond historical appointment volume.

---

# Conclusion

This project demonstrates an end-to-end Data Science workflow, starting with raw healthcare appointment data and progressing through data preparation, statistical analysis, machine learning, forecasting, business analytics, and deployment.

The seven business cases demonstrate how analytical results can be translated into practical healthcare planning scenarios involving patient engagement, staffing, appointment capacity, geographic resources, weather adaptation, health-based segmentation, and specialty demand planning.
