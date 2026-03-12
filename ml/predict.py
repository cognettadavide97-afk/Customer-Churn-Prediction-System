"""Prediction helper.

Scopo del modulo
----------------
Questo modulo viene importato dal backend FastAPI (backend/api.py) e serve a:
- prendere un singolo record (dict) inviato da API/UI
- trasformarlo in un DataFrame compatibile con la Pipeline
- restituire probabilita di churn + predizione binaria

Problema che risolve
--------------------
In produzione spesso arrivano record parziali (non tutte le feature).
La Pipeline e stata addestrata con lo schema derivato da train_raw.csv.
Qui:
- ricaviamo le colonne attese dal preprocessor salvato nella pipeline
- allineiamo i nomi alle colonne attese (normalizzazione semplice)
- aggiungiamo le colonne mancanti come NA (gli imputers le gestiranno)
- calcoliamo le feature derivate usate nel training (AvgMonthlySpend, NumServices)

Soglia di decisione
-------------------
Il modello produce una probabilita proba.
- Se threshold e None: usiamo pipeline.predict (default del modello)
- Se threshold e un float (0-1): decidiamo churn se proba >= threshold

Nota operativa
--------------
I nomi delle colonne in input dovrebbero riflettere quelli del CSV originale
(es. "Tenure Months", "Monthly Charges", "Total Charges").
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"

# Carico una sola volta la pipeline (utile per performance nel backend)
pipeline = joblib.load(MODEL)
print(f"[predict] Loaded model: {MODEL}")


def predict_record(record: dict, threshold: float | None = None) -> dict:
    """Esegue una predizione su un singolo record.

    Args:
        record: dizionario con (alcune) feature del cliente.
        threshold: soglia opzionale per convertire la probabilit� in classe.

    Returns:
        dict con probabilit� e predizione binaria.
    """

    # 1) Dict -> DataFrame (una sola riga)
    df = pd.DataFrame([record]).copy()
    print(f"[predict] input_cols={list(df.columns)}")

    # 2) Normalizzazione nomi (alcuni input UI/API usano nomi diversi dal training)
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
    # 3) Allineamento schema: stesse colonne viste in training
    # Recupero l'elenco delle feature attese dal ColumnTransformer salvato in pipeline.
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers"):
        expected_cols: list[str] = []
        for name, _, cols in preprocessor.transformers:
            if cols is None or cols in ("drop", "passthrough"):
                continue
            expected_cols.extend(list(cols))

        expected_lookup = {c.replace(" ", "").lower(): c for c in expected_cols}
        rename_norm = {c: expected_lookup.get(c.replace(" ", "").lower(), c) for c in df.columns}
        rename_norm = {k: v for k, v in rename_norm.items() if k != v}
        if rename_norm:
            print(f"[predict] normalized_cols={rename_norm}")
            df = df.rename(columns=rename_norm)
        missing_cols = [c for c in expected_cols if c not in df.columns]
        print(f"[predict] expected_cols={len(expected_cols)}")
        print(f"[predict] missing_cols={len(missing_cols)} sample={missing_cols[:10]}")

        # Aggiungo colonne mancanti come NA: gli imputers nel preprocessing le gestiranno.
        for col in missing_cols:
            df[col] = np.nan

        # Feature derivate: se non arrivano dal client ma possiamo calcolarle, le calcoliamo.
        if "AvgMonthlySpend" in expected_cols and "AvgMonthlySpend" not in df.columns:
            if "Total Charges" in df.columns and "Tenure Months" in df.columns:
                total_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
                tenure = pd.to_numeric(df["Tenure Months"], errors="coerce").replace(0, np.nan)
                df["AvgMonthlySpend"] = total_charges / tenure
                print("[predict] computed feature: AvgMonthlySpend")

        if "NumServices" in expected_cols and "NumServices" not in df.columns:
            service_cols = [
                "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
                "Tech Support", "Streaming TV", "Streaming Movies", "Phone Service"
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
                print("[predict] computed feature: NumServices")

        # Conversione numeriche: spesso UI/API inviano numeri come stringhe.
        num_cols: list[str] = []
        for name, _, cols in preprocessor.transformers:
            if name == "num":
                num_cols = list(cols)
                break
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Riordino: alcune pipeline si aspettano un ordine consistente.
        df = df[expected_cols]
        print(f"[predict] aligned_shape={df.shape}")
    else:
        # Caso raro: pipeline senza preprocessor (non atteso in questo progetto)
        print("[predict] WARNING: preprocessor not found; skipping schema alignment")

    # 4) Probabilità churn (classe positiva = 1)
    proba = float(pipeline.predict_proba(df)[0][1])
    print(f"[predict] proba={proba:.4f}")

    # 5) Classe binaria
    if threshold is None:
        # Usa la logica di default del modello (tipicamente soglia 0.5)
        pred = int(pipeline.predict(df)[0])
        print("[predict] threshold=None -> using pipeline.predict")
    else:
        thr = float(threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= thr)
        print(f"[predict] threshold={thr:.3f} -> pred=(proba>=threshold)")

    print(f"[predict] output pred={pred}")

    return {
        "churn_probability": proba,
        "prediction": pred,
    }

if __name__ == "__main__":
    # Esempio rapido: esegui `python ml/predict.py` per vedere i print di verifica.
    # Puoi modificare questo record con valori reali dal dataset.
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