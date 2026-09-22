# Medical Appointment No-Show Prediction

## 📌 Project Overview

This project predicts whether a patient is likely to **attend or miss a medical appointment** using machine learning.

The project includes data cleaning, exploratory data analysis (EDA), statistical analysis, feature engineering, machine learning model development, evaluation, and a Streamlit prediction application.

---

## 🎯 Project Objective

The main objective is to identify patterns associated with appointment no-shows and build a machine learning model that can estimate the probability of a patient missing an appointment.

The prediction can help healthcare organizations identify appointments that may require additional attention or follow-up.

---

## 📊 Dataset

The dataset contains **109,593 appointment records** with information related to:

* Patient demographics
* Medical conditions
* Appointment timing
* Appointment specialty
* Weather conditions
* SMS reminders
* Appointment date
* No-show status

After removing 36 exact duplicate records, the cleaned dataset contained **109,557 records**.

### Target Variable

`no_show`

* `no` → Patient attended
* `yes` → Patient did not attend

The target distribution in the cleaned dataset was:

* Attended: 68.21%
* No-show: 31.79%

---

## 🔍 Data Preparation

The following preprocessing steps were performed:

* Removed exact duplicate records
* Handled missing categorical values using `"Unknown"`
* Handled missing age values using median imputation
* Created an `age_missing` indicator
* Filled missing weather values using median imputation
* Converted appointment dates into datetime format
* Created date-based features
* Created time-of-day categories
* Created age groups
* Encoded the target variable numerically
* Removed the high-cardinality `place` column from modeling
* Removed the original appointment date from modeling after extracting useful date features

---

## ⚙️ Feature Engineering

Additional features were created from the existing data:

* Appointment year
* Appointment month
* Appointment day
* Day of week
* Weekend indicator
* Time of day
* Age group
* Under-12 indicator
* Over-60 indicator
* Age-missing indicator

---

## 📈 Exploratory Data Analysis

EDA was performed to understand relationships between patient, appointment, and weather-related variables and the no-show outcome.

Key observations included:

* No-show rates varied across age groups and specialties.
* Heat intensity showed noticeable differences in no-show rates.
* Appointment shift and time of day showed associations with no-shows.
* SMS receipt had very similar no-show rates for both groups in this dataset.
* Several medical and demographic variables showed statistically significant associations with the target.

These observations describe associations in the dataset and should not be interpreted as proof of causation.

---

## 📊 Statistical Analysis

Chi-square tests were used for categorical variables and Welch's t-test was used for selected numerical variables.

Examples of variables showing statistically significant associations with no-show status included:

* Gender
* Patient companion requirement
* Hypertension
* Diabetes
* Alcoholism
* Appointment shift
* Rain intensity
* Heat intensity
* Time of day
* Age group

Some variables, such as SMS received, Handicap, Scholarship, and appointment day, did not show statistically significant differences in the tests performed.

---

## 🤖 Machine Learning

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   63.41% |    44.30% | 58.73% |   50.51% |  66.22% |
| Random Forest       |   71.06% |    54.69% | 52.31% |   53.47% |  74.81% |
| Tuned Random Forest |   70.70% |    53.60% | 58.37% |   55.88% |  73.88% |

The **Tuned Random Forest** was selected as the final model because tuning substantially reduced overfitting while maintaining a useful balance between precision and recall.

---

## 🌲 Final Model

The final model is a **Random Forest Classifier** with:

* 150 trees
* Maximum depth of 12
* Minimum samples split of 10
* Minimum samples leaf of 5
* Balanced class weights

### Final Test Performance

* Accuracy: **70.70%**
* Precision: **53.60%**
* Recall: **58.37%**
* F1 Score: **55.88%**
* ROC-AUC: **73.88%**

The model uses a classification threshold of **0.50** for the main application.

---

## 🧪 Model Prediction

The trained model and preprocessing pipeline were saved using `joblib`.

Files:

* `medical_no_show_model.pkl`
* `medical_no_show_preprocessor.pkl`
* `medical_no_show_features.pkl`

The application accepts appointment information and returns:

* Predicted outcome
* No-show probability
* Attended probability

---

## 🖥️ Streamlit Application

A Streamlit application was developed to provide an interactive prediction interface.

The application allows users to enter appointment and patient information and receive a predicted appointment outcome.

### Run the application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in the browser at the local Streamlit address.

---

## 📁 Project Structure

```text
Medical-Appointment-No-Show/
│
├── notebooks/
│   └── medical_appointment_no_show.ipynb
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

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

---

## 📌 Limitations

* The model identifies statistical patterns and does not establish causation.
* Predictions are estimates and are not guaranteed outcomes.
* The dataset contains missing values and required preprocessing.
* Some features, such as `place`, had very high cardinality and were excluded from modeling.
* Model performance may change when applied to a different healthcare population or dataset.
* The application is intended as a demonstration of a machine learning workflow and should not be used as a standalone clinical decision-making system.

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter optimization
* Probability threshold optimization based on the operational objective
* Additional model comparison
* Model explainability using SHAP
* Deployment to a cloud platform
* Monitoring model performance after deployment

---

## 👩‍💻 Project Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Statistical Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Tuning
   ↓
Model Saving
   ↓
Streamlit Deployment
```
