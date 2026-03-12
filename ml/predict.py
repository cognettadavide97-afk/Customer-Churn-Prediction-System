"""Scopo del modulo
----------------
Questo modulo viene importato dal backend FastAPI (backend/api.py) e serve a:
- prendere un singolo record (dict) inviato da API/UI
- trasformarlo in un DataFrame compatibile con la Pipeline
- restituire probabilita di churn + predizione binaria

Modelli supportati
------------------
- Base: models/churn_pipeline_v1.joblib
- Ensemble (se presente): models/churn_ensemble_v1.joblib con stack_classifier

Se l'artifact ensemble è disponibile, viene usato di default.
In caso contrario si usa automaticamente il modello base XGBoost.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE = ROOT / "models" / "churn_pipeline_v1.joblib"
MODEL_ENSEMBLE = ROOT / "models" / "churn_ensemble_v1.joblib"

# Carico artifacts una sola volta
pipeline = joblib.load(MODEL_BASE)
ensemble_artifacts = joblib.load(MODEL_ENSEMBLE) if MODEL_ENSEMBLE.exists() else None

if ensemble_artifacts is not None and "stack_classifier" in ensemble_artifacts:
    active_model = "ensemble"
    stack_classifier = ensemble_artifacts["stack_classifier"]
    print(f"[predict] Loaded ensemble model: {MODEL_ENSEMBLE}")
else:
    active_model = "xgb"
    stack_classifier = None
    print(f"[predict] Loaded base model: {MODEL_BASE}")


def _align_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Allinea il record alle colonne attese dal preprocessor della pipeline base."""

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers"):
        print("[predict] WARNING: preprocessor non trovato; skip schema alignment")
        return df

    expected_cols: list[str] = []
    for name, _, cols in preprocessor.transformers:
        if cols is None or cols in ("drop", "passthrough"):
            continue
        expected_cols.extend(list(cols))

    # Normalizzazione nomi (spazi/case insensitive)
    expected_lookup = {c.replace(" ", "").lower(): c for c in expected_cols}
    rename_norm = {c: expected_lookup.get(c.replace(" ", "").lower(), c) for c in df.columns}
    rename_norm = {k: v for k, v in rename_norm.items() if k != v}
    if rename_norm:
        print(f"[predict] normalized_cols={rename_norm}")
        df = df.rename(columns=rename_norm)

    missing_cols = [c for c in expected_cols if c not in df.columns]
    for col in missing_cols:
        df[col] = np.nan

    # Feature derivate usate nel training
    if "AvgMonthlySpend" in expected_cols and "AvgMonthlySpend" not in df.columns:
        if "Total Charges" in df.columns and "Tenure Months" in df.columns:
            total_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
            tenure = pd.to_numeric(df["Tenure Months"], errors="coerce").replace(0, np.nan)
            df["AvgMonthlySpend"] = total_charges / tenure

    if "NumServices" in expected_cols and "NumServices" not in df.columns:
        service_cols = [
            "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
            "Tech Support", "Streaming TV", "Streaming Movies", "Phone Service",
        ]
        available_services = [c for c in service_cols if c in df.columns]
        if available_services:
            service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
            service_numeric = (
                df[available_services]
                .replace(service_map)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
            df["NumServices"] = service_numeric.sum(axis=1)

    # Conversione numeriche
    num_cols: list[str] = []
    for name, _, cols in preprocessor.transformers:
        if name == "num":
            num_cols = list(cols)
            break
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordine finale colonne
    return df[expected_cols]


def predict_record(record: dict, threshold: float | None = None) -> dict:
    """Esegue una predizione su un singolo record."""

    df = pd.DataFrame([record]).copy()
    print(f"[predict] input_cols={list(df.columns)}")

    rename_map = {
        "tenure": "Tenure Months",
        "SeniorCitizen": "Senior Citizen",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
    }
    rename_cols = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if rename_cols:
        print(f"[predict] renamed_cols={rename_cols}")
    df = df.rename(columns=rename_cols)

    df = _align_input_schema(df)
    print(f"[predict] aligned_shape={df.shape}")

    # Probabilità churn
    if active_model == "ensemble" and stack_classifier is not None:
        proba = float(stack_classifier.predict_proba(df)[0][1])
        default_pred = int(stack_classifier.predict(df)[0])
        used_model = "stacking_xgb_cat"
    else:
        proba = float(pipeline.predict_proba(df)[0][1])
        default_pred = int(pipeline.predict(df)[0])
        used_model = "xgb_pipeline"

    # Classe binaria
    if threshold is None:
        pred = default_pred
    else:
        thr = float(threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= thr)

    return {
        "model_used": used_model,
        "churn_probability": proba,
        "prediction": pred,
    }


if __name__ == "__main__":
    sample_record = {
        "Tenure Months": 12,
        "Monthly Charges": 70.0,
        "Total Charges": 840.0,
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Payment Method": "Electronic check",
        "Paperless Billing": "Yes",
        "Phone Service": "Yes",
    }
    print("[predict] Running sample prediction...")
    result = predict_record(sample_record, threshold=0.6)
    print(f"[predict] result={result}")