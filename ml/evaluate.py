"""Evaluation script.

Scopo del modulo
----------------
Verifica le performance del modello salvato da ml/train_model.py sul test set
prodotto da ml/preprocessing.py.

Input
-----
- models/churn_pipeline_v1.joblib
- models/churn_ensemble_v1.joblib (opzionale)
- data/processed/test_raw.csv (colonna target "Churn Value")

Output
------
- outputs/metrics.csv: metriche riassuntive (base + ensemble se disponibile)
- outputs/classification_report.txt: report dettagliato per classe
- outputs/confusion_matrix_xgb.png e outputs/confusion_matrix_ensemble.png
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Setup path progetto
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carico artifacts
pipeline = joblib.load(MODELS_DIR / "churn_pipeline_v1.joblib")
ensemble_path = MODELS_DIR / "churn_ensemble_v1.joblib"
ensemble_artifacts = joblib.load(ensemble_path) if ensemble_path.exists() else None
if ensemble_artifacts is None:
    print("[evaluate] Ensemble artifact non trovato: valuto solo XGBoost base.")
else:
    print(f"[evaluate] Loaded ensemble artifact: {ensemble_path}")

# Carico test set
df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
print(f"Test shape: {df_test.shape}")

target_col = "Churn Value"
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]


def _compute_metrics(y_true, preds, proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


# ===== Modello base XGBoost =====
preds_xgb = pipeline.predict(X_test)
proba_xgb = pipeline.predict_proba(X_test)[:, 1]
metrics_xgb = _compute_metrics(y_test, preds_xgb, proba_xgb)
metrics_xgb["model"] = "xgb_pipeline"

print("\n=== Metrics: XGBoost pipeline ===")
for k, v in metrics_xgb.items():
    if k != "model":
        print(f"{k}: {v:.4f}")

# ===== Ensemble stacking (se presente) =====
metrics_rows = [metrics_xgb]
report_text = "=== XGBoost pipeline ===\n"
report_text += classification_report(y_test, preds_xgb, digits=4, zero_division=0)

ConfusionMatrixDisplay.from_predictions(y_test, preds_xgb)
plt.title("Matrice di Confusione - XGBoost pipeline")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_xgb.png")
plt.close()

if ensemble_artifacts is not None:
    stack_classifier = ensemble_artifacts["stack_classifier"]
    proba_ens = stack_classifier.predict_proba(X_test)[:, 1]
    preds_ens = (proba_ens >= 0.5).astype(int)

    metrics_ens = _compute_metrics(y_test, preds_ens, proba_ens)
    metrics_ens["model"] = "stacking_xgb_cat"
    metrics_rows.append(metrics_ens)

    print("\n=== Metrics: Stacking ensemble (XGBoost + CatBoost) ===")
    for k, v in metrics_ens.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    report_text += "\n\n=== Stacking ensemble (XGBoost + CatBoost) ===\n"
    report_text += classification_report(y_test, preds_ens, digits=4, zero_division=0)

    ConfusionMatrixDisplay.from_predictions(y_test, preds_ens)
    plt.title("Matrice di Confusione - Stacking ensemble")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix_ensemble.png")
    plt.close()

# Salvataggi output
pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "metrics.csv", index=False)
(OUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

print(f"\nSaved metrics to: {OUT_DIR / 'metrics.csv'}")
print(f"Saved report to: {OUT_DIR / 'classification_report.txt'}")