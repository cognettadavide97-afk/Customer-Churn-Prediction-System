# Customer Churn Prediction --- Machine Learning Project

## Project Goal

Build an end‑to‑end **Customer Churn Prediction system** that
demonstrates skills in:

-   Data Engineering
-   Machine Learning
-   Backend API development
-   Frontend data visualization

The system predicts whether a customer is likely to **churn** using the
**Telco Customer Churn dataset** and exposes predictions through an API
and a dashboard.

The project is designed to be **completed in 5 days by a team of 4
people**.

------------------------------------------------------------------------

# Project Architecture

    customer-churn-ml/

    data/
       raw/
          telco_churn.csv
       processed/
          train_raw.csv
          test_raw.csv

    ml/
       preprocessing.py
       train_model.py
       evaluate.py
       predict.py

    analysis/
       eda.py
       plots.py

    backend/
       api.py

    frontend/
       dashboard.py

    utils/
       data_loader.py

    models/
       churn_pipeline_v1.joblib

    outputs/
       metrics.csv
       classification_report.txt
       plots/
          churn_distribution.png
          tenure_churn.png
          monthly_charges_churn.png
          correlation_matrix.png
          feature_importance.png

    requirements.txt
    README.md

------------------------------------------------------------------------

# Pipeline

The project follows a strict machine learning workflow.

    DATASET
       │
       ▼
    EDA + VISUALIZATION
    (analysis/)
       │
       ▼
    DATA CLEANING
    (ml/preprocessing.py)
       │
       ▼
    FEATURE ENGINEERING
    (ml/preprocessing.py)
       │
       ▼
    MODEL TRAINING
    (ml/train_model.py)
       │
       ▼
    MODEL EVALUATION
    (ml/evaluate.py)
       │
       ▼
    PREDICTIONS CSV
    (ml/predict.py)
       │
       ▼
    BACKEND API
    (backend/api.py)
       │
       ▼
    FRONTEND DASHBOARD
    (frontend/dashboard.py)

------------------------------------------------------------------------

# Team Roles

## Giovanni --- Data Engineer

### Files

utils/data_loader.py\
ml/preprocessing.py

### Responsibilities

-   Load the raw dataset
-   Clean missing values
-   Convert column formats
-   Remove duplicates
-   Split dataset into train/test

### Output

data/processed/train_raw.csv\
data/processed/test_raw.csv

------------------------------------------------------------------------

## Davide --- Machine Learning Engineer

### Files

ml/train_model.py\
ml/evaluate.py\
ml/predict.py

### Responsibilities

-   Build ML pipeline
-   Train XGBoost model tuned with Optuna (F2-oriented)
-   Evaluate model performance
-   Save trained model
-   Implement prediction functions

### Output

models/churn_pipeline_v1.joblib\
outputs/metrics.csv\
outputs/classification_report.txt

------------------------------------------------------------------------

## Gabriele --- Backend + Frontend Developer

### Backend

backend/api.py

Endpoints GET /\
POST /predict\
POST /preprocess\
POST /train\
POST /evaluate\
POST /generate-plots

### Frontend

frontend/dashboard.py

Features - Single prediction via API - Visualization support for churn probability - ML pipeline trigger from backend endpoints

------------------------------------------------------------------------

## Elisabetta --- Data Visualization & Analysis

### Files

analysis/eda.py\
analysis/plots.py

### Responsibilities

-   Exploratory Data Analysis
-   Create data visualizations
-   Provide insights about churn patterns

### Charts

outputs/plots/

------------------------------------------------------------------------


# Project Guides (canonical)

- `docs/ARCHITETTURA_GUIDA.md` (architettura tecnica aggiornata)
- `docs/RUNBOOK.md` (comandi operativi, test minimi, quality gate, e2e ridotto)

------------------------------------------------------------------------

# Setup

Create environment

python3.12 -m venv venv\
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# Run Pipeline

Preprocessing

python ml/preprocessing.py

Training

python ml/train_model.py

Evaluation

python ml/evaluate.py

Generate plots

python -m analysis.plots


------------------------------------------------------------------------

# Canonical Documentation

Use these two files as the single source of truth:

- docs/ARCHITETTURA_GUIDA.md
- docs/RUNBOOK.md

------------------------------------------------------------------------

# Minimal checks


Minimal checks (optional toggle true/false in code)

python - <<'PY'
import ml.train_model as tm
import ml.evaluate as ev
import ml.predict as pr

# training: verifica contratto preprocessing (schema+shape)
tm.run_minimal_tests()
# evaluate: verifica metriche chiave presenti nei report
ev.run_minimal_tests()
# predict: verifica output minimo + controllo soglia invalida
pr.run_minimal_tests()
print("Minimal checks passed")


# quality gate regressione modello (soglie recall/f1/auc)
import ml.evaluate as ev
print(ev.run_quality_regression_test())
PY


# reduced end-to-end integration test
python scripts/run_reduced_e2e_test.py

------------------------------------------------------------------------

# Run Application

Start backend

uvicorn backend.api:app --reload

Start dashboard

streamlit run frontend/dashboard.py

------------------------------------------------------------------------

# Expected Outputs

models/ churn_pipeline_v1.joblib

outputs/ metrics.csv\
classification_report.txt\
plots/

------------------------------------------------------------------------

# Technologies

Python 3.12\
Pandas\
Scikit-learn\
XGBoost\
Optuna\
FastAPI\
Streamlit\
Matplotlib\
Seaborn

------------------------------------------------------------------------

# Project Objective

This repository demonstrates a **complete machine learning system**
including:

-   Data preparation
-   Model training
-   Model deployment
-   Data visualization

The goal is to create a project suitable for **portfolio presentation to
companies**.
