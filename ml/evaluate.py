"""Evaluation script.

Scopo del modulo
----------------
Verifica le performance del modello salvato da ml/train_model.py sul test set
prodotto da ml/preprocessing.py.

Input
-----
- models/churn_pipeline_v1.joblib
- data/processed/test_raw.csv (colonna target "Churn Value")

Output
------
- outputs/metrics.csv: metriche riassuntive
- outputs/classification_report.txt: report dettagliato per classe
- plot matrice di confusione

Metriche
--------
- accuracy: accuratezza globale
- precision: tra i predetti churn, quanti sono churn reali
- recall: tra i churn reali, quanti ne intercetto
- f1: armonica di precision e recall
- roc_auc: qualita del ranking probabilistico (indipendente dalla soglia)
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

# Carico la pipeline completa (preprocessing + modello)
pipeline = joblib.load(MODELS_DIR / "churn_pipeline_v1.joblib")

# Carico il test set già processato (stesse colonne/feature del train)
df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
print(f"Test shape: {df_test.shape}")

# Split X/y
target_col = "Churn Value"
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

# Predizioni
# - preds: classi (0/1) usando la logica del modello (tipicamente soglia 0.5)
# - proba: probabilità della classe positiva (churn)
preds = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

# Calcolo metriche
metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds, zero_division=0),
    "recall": recall_score(y_test, preds, zero_division=0),
    "f1": f1_score(y_test, preds, zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba),
}

# Print rapidi per verifica manuale
print(f"Recall: {metrics['recall']:.2f}")
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"Precision: {metrics['precision']:.2f}")
print(f"F1-score: {metrics['f1']:.2f}")
print(f"ROC-AUC: {metrics['roc_auc']:.2f}")

# Salvo metriche su CSV (comodo per dashboard o tracking)
pd.DataFrame([metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)
print(f"Saved metrics to: {OUT_DIR / 'metrics.csv'}")

# Report più dettagliato per classe
report = classification_report(y_test, preds, digits=4, zero_division=0)
(OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

# Matrice di confusione: mostra FP/FN in modo immediato
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("Matrice di Confusione - Errori del modello")
#plt.show()
plt.savefig(MODELS_DIR / "confusion_matrix.png")
plt.close()