"""Scopo del modulo
----------------
Questo modulo viene importato dal backend FastAPI (backend/api.py) e serve a:
- prendere un singolo record (dict) inviato da API/UI
- trasformarlo in un DataFrame compatibile con la Pipeline
- restituire probabilita di churn + predizione binaria

Modello supportato
------------------
- Base: models/churn_pipeline_v1.joblib (XGBoost ottimizzato)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE = ROOT / "models" / "churn_pipeline_v1.joblib"
RUN_MINIMAL_TESTS_DEFAULT = False

# Mappa didattica:
# - _align_input_schema: rende robusto l'ingresso dati reali (API/UI)
# - predict_record: applica soglia decisionale e restituisce output standard
# - run_minimal_tests: presidia il contratto minimo del modulo

# Carico artifact una sola volta
pipeline = joblib.load(MODEL_BASE)
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


def _save_prediction_probability_plot(probability: float, output_path: Path) -> None:
    """Salva un grafico semplice della probabilità churn per documentazione/report.

    Utile per esplicitare rapidamente il risultato di una predizione in contesti
    business o documentali (es. allegato a report interni).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [1.0 - probability, probability]
    labels = ["Stay", "Churn"]
    colors = ["#2a9d8f", "#e76f51"]

    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel("Probabilità")
    plt.title("Predizione churn (singolo cliente)")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def predict_record(
    record: dict,
    threshold: float | None = None,
    debug: bool = False,
    save_plot_path: str | None = None,
) -> dict:
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
    if debug:
        missing_count = int(df.isna().sum().sum())
        print(f"[predict][debug] missing_values_after_alignment={missing_count}")
        print(f"[predict][debug] aligned_columns={list(df.columns)}")

    # Probabilità churn con modello XGBoost base.
    proba = float(pipeline.predict_proba(df)[0][1])
    default_pred = int(pipeline.predict(df)[0])
    used_model = "xgb_pipeline"

    # Classe binaria
    # La soglia permette di regolare il trade-off business:
    # soglia più bassa -> più churn intercettati ma più falsi positivi.
    if threshold is None:
        pred = default_pred
    else:
        thr = float(threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= thr)

    result = {
        "model_used": used_model,
        "churn_probability": proba,
        "prediction": pred,
    }

    if save_plot_path:
        plot_path = Path(save_plot_path)
        _save_prediction_probability_plot(probability=proba, output_path=plot_path)
        result["prediction_plot_path"] = str(plot_path)

    return result


def run_minimal_tests() -> None:
    """Test minimi obbligatori del modulo predict (attivabili via flag)."""

    # Test 1: inferenza base con record realistico (contratto output minimo).
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

    res = predict_record(sample_record, threshold=0.5, debug=False)
    assert {"model_used", "churn_probability", "prediction"}.issubset(res.keys())

    # Test 2: validazione input threshold (deve alzare ValueError se fuori range).
    try:
        predict_record(sample_record, threshold=1.2, debug=False)
        raise AssertionError("Soglia invalida non intercettata")
    except ValueError:
        pass


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
    if RUN_MINIMAL_TESTS_DEFAULT:
        print("[predict] Esecuzione test minimi attiva")
        run_minimal_tests()
        print("[predict] Test minimi completati")

    result = predict_record(sample_record, threshold=0.6)
    print(f"[predict] result={result}")
