# Medical Appointment No-Show Prediction & Demand Forecasting

## Project Overview

This project analyzes historical medical appointment data to understand and predict patient no-shows and translate analytical findings into practical healthcare business use cases.

The project combines:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Statistical analysis and hypothesis testing
- Machine Learning classification
- Time-series demand analysis and forecasting
- Healthcare business analytics
- Interactive Streamlit dashboard development

The objective is to help healthcare organizations identify appointments at higher risk of no-show, understand historical appointment demand, and support better planning of staffing, resources, and appointment capacity.

## Dataset

The original dataset contains **109,593 records and 26 columns**.

After removing 36 exact duplicate records, the cleaned analysis contains **109,557 appointments**.

The dataset includes information related to:

- Patient demographics
- Medical conditions
- Appointment dates
- Medical specialties
- Locations
- Weather conditions
- SMS reminders
- Appointment attendance

The historical appointment dates range from **January 2020 to May 2021**.

> The raw dataset is not publicly redistributed because it was provided for project and learning purposes.

## Project Workflow

```text
Raw Medical Appointment Data
            |
            v
      Data Cleaning
            |
            v
 Exploratory Data Analysis
            |
            v
    Feature Engineering
            |
            v
   Statistical Analysis
            |
            v
   Machine Learning
            |
            v
  No-Show Risk Prediction
            |
            v
    Demand Analysis
            |
            v
  Time-Series Forecasting
            |
            v
  Healthcare Business Cases
            |
            v
   Streamlit Dashboard
```

---

# 1. Machine Learning — No-Show Prediction

## Target Variable

The target variable is `no_show`.

It was converted into a numerical binary target:

```text
no  → 0 (Attended)
yes → 1 (No-show)
```

## Class Distribution

| Outcome | Appointments | Percentage |
|---|---:|---:|
| Attended | 74,726 | 68.21% |
| No-show | 34,831 | 31.79% |

## Feature Engineering

Important engineered features include:

- `age_missing`
- `appointment_year`
- `appointment_month`
- `appointment_day`
- `appointment_dayofweek`
- `is_weekend`
- `time_of_day`
- `age_group`

Categorical variables were encoded using `OneHotEncoder`, while numerical variables were standardized using `StandardScaler`.

## Models Evaluated

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 63.41% |
| Precision | 44.30% |
| Recall | 58.73% |
| F1 Score | 50.51% |
| ROC-AUC | 0.6622 |

### Tuned Random Forest

The final Random Forest model used:

- 150 trees
- Maximum depth of 12
- Minimum samples split of 10
- Minimum samples leaf of 5
- Balanced class weights

| Metric | Score |
|---|---:|
| Accuracy | 70.70% |
| Precision | 53.60% |
| Recall | 58.37% |
| F1 Score | 55.88% |
| ROC-AUC | 0.7388 |

The tuned Random Forest performed better than Logistic Regression across the reported classification metrics.

The tuned model also showed a substantially smaller train-test accuracy gap than the original unrestricted Random Forest, reducing overfitting.

---

# 2. Demand Analysis

Historical appointment demand was analyzed using the `appointment_date_continuous` field.

The analysis examined:

- Daily appointment volume
- Weekly appointment patterns
- Monthly appointment trends
- Seven-day rolling averages
- Demand distribution
- High-volume dates and outliers
- Changes in historical appointment volume

## Key Findings

| Metric | Value |
|---|---:|
| Historical dates | 498 days |
| Total appointments | 109,593 |
| Average daily appointments | 220.07 |
| Median daily appointments | 126.5 |
| Maximum daily appointments | 1,512 |
| Minimum daily appointments | 1 |

Three unusually high-volume dates were identified:

- October 8, 2020: 974 appointments
- March 29, 2021: 985 appointments
- April 2, 2021: 1,512 appointments

A substantial decline in recorded appointment volume was observed after April 2, 2021.

This pattern was retained for analysis rather than automatically removing the affected dates. The reason for the decline cannot be established from the available dataset.

---

# 3. Demand Forecasting

Time-series forecasting was performed to explore historical appointment demand and compare forecasting approaches.

The analysis included:

- Daily demand forecasting
- Weekly demand forecasting
- Monthly demand forecasting
- Chronological train-test splitting
- Naive forecasting
- Moving Average
- Simple Exponential Smoothing
- Holt's Linear Trend
- Holt-Winters Exponential Smoothing
- ARIMA
- SARIMA
- Model evaluation and comparison

## Forecasting Data

The daily demand series contains 498 observations.

A chronological split was used for daily forecasting:

- Training observations: 398
- Testing observations: 100

Weekly and monthly forecasting were also evaluated using chronological splits.

## Daily Forecast Model Comparison

The daily models were compared using MAE, RMSE, MAPE, and R².

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive | 200.34 | 337.85 | 494.90 | -0.41 |
| Moving Average | 282.04 | 316.85 | 7183.31 | -0.24 |
| SES | 240.85 | 286.84 | 5076.15 | -0.02 |
| Holt | 261.18 | 300.39 | 6031.66 | -0.12 |
| Holt-Winters | 253.33 | 313.39 | 7555.56 | -0.22 |
| ARIMA | 234.95 | 287.45 | 4608.68 | -0.02 |
| SARIMA | 172.25 | 312.65 | 82.95 | -0.21 |

SARIMA achieved the lowest daily MAE and MAPE among the evaluated models. SES achieved the lowest RMSE and the best R².

However, all daily models produced negative R² values, indicating limited forecasting performance relative to a test-set mean baseline.

## Weekly and Monthly Forecasting

Weekly and monthly demand were also evaluated using Naive, SES, Holt, and Holt-Winters forecasting methods.

The Naive model performed best among the evaluated weekly and monthly models based on MAE.

Monthly evaluation was based on a particularly small test sample, so the results should be interpreted cautiously.

## Forecasting Limitations

- Historical appointment volume contains a substantial decline after April 2, 2021.
- The dataset covers only January 2020 to May 2021.
- Daily and weekly forecasting metrics indicate limited predictive performance.
- Monthly evaluation contains only four test observations.
- MAPE can become extremely large when actual demand values are very small.
- Forecasts should be treated as exploratory historical estimates, not reliable estimates of present-day healthcare demand.

The forecasting analysis demonstrates the application of time-series methods and model evaluation rather than a validated operational forecasting system.

---

# 4. Healthcare Business Cases

The project extends the analytical findings into seven healthcare business use cases.

The individual business case notebooks are retained in the `archive/business cases/` directory.

## 4.1 Risk-Based Patient Engagement

Predicted no-show probabilities were converted into business-defined risk groups.

| Risk Group | Predicted Risk | Observed No-Show Rate | Suggested Action |
|---|---|---:|---|
| Low | <30% | 4.27% | Standard reminder |
| Medium | 30–60% | 25.56% | SMS reminder and confirmation |
| High | >60% | 63.48% | Priority SMS and rescheduling option |

These thresholds are business-defined and should be validated using operational outcomes before real-world deployment.

## 4.2 Intelligent Staffing Optimization

Historical appointment volume was analyzed by:

- Day of week
- Specialty
- Shift
- Month

The analysis explores how historical demand patterns can support staffing and capacity planning.

Actual staffing requirements also depend on appointment duration, specialist availability, patient complexity, and operational constraints.

## 4.3 Revenue Protection

The dataset contained:

- 34,831 no-show appointments
- 31.79% overall no-show rate

The three specialties with the largest number of no-shows were:

- Psychotherapy: 9,263
- Physiotherapy: 7,218
- Speech therapy: 6,234

Together they represented **22,715 no-shows, or 65.21% of all missed appointments**.

Scenario analysis estimated potentially recovered appointment capacity under different reductions in no-shows.

| No-Show Reduction | Potentially Recovered Appointments |
|---:|---:|
| 10% | 3,483 |
| 20% | 6,966 |
| 30% | 10,449 |

These are scenario estimates rather than guaranteed financial savings.

## 4.4 Geographic Resource Planning

The `place` field contains multiple unique location values.

Rather than assuming every unique value represents a city, the analysis treated `place` as a geographic location field.

The three highest-volume locations among the analyzed locations accounted for:

**32,053 appointments — 29.26% of all appointments.**

This analysis can support geographic resource allocation and identification of locations requiring additional operational attention.

## 4.5 Seasonal and Weather Adaptations

The project examined:

- Monthly no-show rates
- Rain intensity
- Heat intensity
- Previous-day rain
- Previous-day storms

The overall no-show rate was 31.79%.

Chi-square tests found statistically significant associations between no-show status and:

- Rainy day before appointment
- Storm day before appointment
- Heat intensity

These findings indicate associations, not causation.

## 4.6 Health-Based Segmentation

Health-related variables were analyzed to understand differences in appointment attendance patterns.

The analysis examined:

- Hypertension
- Diabetes
- Alcoholism
- Disability
- Age groups
- Companion requirements
- SMS reminders

Chi-square testing found statistically significant associations between no-show status and hypertension, diabetes, and alcoholism.

`Handcap` did not show a statistically significant association in this analysis.

Health-related findings should be used responsibly and should not be used for clinical decisions or discriminatory treatment.

## 4.7 Specialty-Level Demand Planning

Historical appointment demand was analyzed across specialties.

| Specialty | Appointments | Demand Share |
|---|---:|---:|
| Psychotherapy | 28,642 | 26.14% |
| Speech therapy | 22,321 | 20.37% |
| Physiotherapy | 21,001 | 19.17% |
| Occupational therapy | 11,318 | 10.33% |

Psychotherapy, speech therapy, and physiotherapy together represented:

**65.68% of total historical appointment demand.**

Demand was also compared across morning and afternoon shifts and across 17 months of historical data.

---

# 5. Streamlit Application

An interactive Streamlit dashboard was developed to bring the project's analytical findings together in one application.

The dashboard contains six sections:

| Section | Purpose |
|---|---|
| Home | Project overview and key information |
| No Show Prediction | Predict no-show probability and risk category for an appointment |
| No Show Analysis | Explore historical attendance patterns |
| Demand Analysis | Examine daily, weekly, and monthly appointment demand |
| Demand Forecasting | Explore historical forecasts and model comparisons |
| Resource Planning | Support healthcare resource and capacity planning through analytical insights |

The prediction interface loads the saved Random Forest model, preprocessing pipeline, and feature configuration.

## Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

# 6. Repository Structure

```text
Medical-Appointment-No-Show/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   ├── medical_no_show_model.pkl
│   ├── medical_no_show_preprocessor.pkl
│   └── medical_no_show_features.pkl
│
├── data/
│   ├── final_daily_demand.csv
│   └── forecasting_model_comparison.csv
│
├── notebooks/
│   ├── medical_appointment_analysis.ipynb
│   └── Medical_Appointment_Demand_Forecasting.ipynb
│
├── archive/
│   ├── app_old.py
│   ├── pages_backup/
│   └── business cases/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The original raw dataset is not included in the public repository.

---

# 7. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Statsmodels
- Joblib
- Streamlit
- Jupyter Notebook

---

# 8. Key Skills Demonstrated

### Data Analysis

- Data cleaning
- Missing-value handling
- Duplicate removal
- Exploratory Data Analysis
- Univariate and bivariate analysis
- GroupBy analysis
- Feature engineering

### Statistics

- Chi-square hypothesis testing
- P-values
- Association analysis
- Statistical interpretation

### Machine Learning

- Train-test split
- Stratified sampling
- One-hot encoding
- Feature scaling
- Logistic Regression
- Random Forest
- Hyperparameter tuning
- Class imbalance handling
- Confusion matrix
- Precision
- Recall
- F1 score
- ROC-AUC
- Overfitting analysis

### Time-Series Analysis

- Chronological train-test split
- Naive forecasting
- Moving Average
- Simple Exponential Smoothing
- Holt's Linear Trend
- Holt-Winters Exponential Smoothing
- ARIMA
- SARIMA
- Log transformation
- MAE
- RMSE
- MAPE
- R²

### Deployment

- Streamlit dashboard development
- Saved ML model
- Saved preprocessing pipeline
- Prediction interface
- Multi-section dashboard

---

# 9. Important Limitations

- The dataset represents a historical period from 2020 to 2021.
- Historical patterns may not represent current healthcare demand.
- The model predicts statistical risk and does not determine why an individual patient may miss an appointment.
- Weather relationships represent associations rather than causal effects.
- Health-related variables require careful privacy, fairness, and governance considerations.
- Revenue loss could not be calculated because appointment prices and actual revenue data were not provided.
- Geographic analysis is limited by the quality and structure of the `place` field.
- Staffing recommendations should consider operational factors beyond historical appointment volume.
- Demand forecasting performance is limited by the historical data pattern and small evaluation samples.

---

# 10. Conclusion

This project demonstrates an end-to-end Data Science workflow, starting with historical healthcare appointment data and progressing through data preparation, statistical analysis, machine learning, demand forecasting, business analytics, and deployment.

The six-section Streamlit dashboard brings together appointment-level no-show prediction, attendance analysis, demand analysis, forecasting, and resource planning.

The seven business cases demonstrate how analytical findings can be translated into practical healthcare planning scenarios involving patient engagement, staffing, appointment capacity, geographic resources, weather adaptation, health-based segmentation, and specialty demand planning.

The project highlights how data science can support evidence-based decision-making in healthcare operations while recognizing the limitations of historical data and predictive models.