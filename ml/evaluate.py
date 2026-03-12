"""Model evaluation utilities.

Obiettivo:
- Caricare modello/i e test set
- Calcolare metriche coerenti
- Salvare report e matrici di confusione
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

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
TARGET_COL = "Churn Value"
RUN_MINIMAL_TESTS_DEFAULT = False
RUN_QUALITY_GATE_DEFAULT = False

# Mappa didattica del modulo:
# - evaluate_model: produce metriche + confusion matrix + report testuale
# - run_minimal_tests: verifica contratto metriche senza costi elevati
# - run_quality_regression_test: soglia di sbarramento coerente con obiettivo churn


def _compute_metrics(y_true, preds, proba) -> dict:
    """Calcola metriche principali di classificazione binaria."""

    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def evaluate_model(
    model_path: Path | str,
    test_data_path: Path | str,
    out_dir: Path | str,
    plot_out_dir: Path | str | None = None,
    target_col: str = TARGET_COL,
) -> dict:
    """Esegue valutazione del modello XGBoost base e salva gli output."""

    model_path = Path(model_path)
    test_data_path = Path(test_data_path)
    out_dir = Path(out_dir)
    plot_out_dir = Path(plot_out_dir) if plot_out_dir else out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carico artifact e test set.
    pipeline = joblib.load(model_path)

    df_test = pd.read_csv(test_data_path)
    if target_col not in df_test.columns:
        raise ValueError(f"Target '{target_col}' non trovato nel test set.")

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # 2) Valutazione modello base.
    preds_xgb = pipeline.predict(X_test)
    proba_xgb = pipeline.predict_proba(X_test)[:, 1]
    metrics_xgb = _compute_metrics(y_test, preds_xgb, proba_xgb)
    metrics_xgb["model"] = "xgb_pipeline"

    report_text = "=== XGBoost pipeline ===\n"
    report_text += classification_report(y_test, preds_xgb, digits=4, zero_division=0)
    metrics_rows = [metrics_xgb]

    # Confusion matrix = lettura immediata di FP/FN: utile per stimare
    # impatto operativo (campagne inutili vs churn non intercettati).
    ConfusionMatrixDisplay.from_predictions(y_test, preds_xgb)
    plt.title("Matrice di Confusione - XGBoost pipeline")
    plt.tight_layout()
    plt.savefig(plot_out_dir / "confusion_matrix_xgb.png")
    plt.close()

    # 3) Persistenza output di valutazione.
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    return {row["model"]: {k: v for k, v in row.items() if k != "model"} for row in metrics_rows}


def run_minimal_tests() -> None:
    """Test minimi obbligatori del modulo evaluate (attivabili via flag)."""

    # Test 1: caso base con input semplici e coerenti.
    # Verifica che il calcolo metriche non rompa il contratto atteso.
    y_true = pd.Series([0, 1, 1, 0])
    preds = pd.Series([0, 1, 0, 0])
    proba = pd.Series([0.1, 0.8, 0.4, 0.2])

    metrics = _compute_metrics(y_true, preds, proba)
    # Test 2: presenza delle metriche chiave usate nei report business/tecnici.
    required = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert required.issubset(metrics.keys()), "Metriche obbligatorie mancanti"


def run_quality_regression_test(
    model_path: Path | str = MODELS_DIR / "churn_pipeline_v1.joblib",
    test_data_path: Path | str = PROC_DIR / "test_raw.csv",
    target_col: str = TARGET_COL,
    min_recall: float = 0.78,
    min_f1: float = 0.62,
    min_roc_auc: float = 0.84,
) -> dict:
    """Esegue un quality gate di regressione sul modello.

    Soglie coerenti con l'obiettivo progetto (intercettazione churn):
    - recall prioritaria (>= 0.78),
    - F1 minima per equilibrio operativo,
    - AUC minima per ranking robusto.
    """

    # Questa funzione è il gate usato in CI: se le metriche scendono
    # sotto soglia, il merge deve essere bloccato.
    metrics = evaluate_model(
        model_path=model_path,
        test_data_path=test_data_path,
        out_dir=OUT_DIR,
        plot_out_dir=OUT_DIR,
        target_col=target_col,
    )["xgb_pipeline"]

    failures: list[str] = []
    if metrics["recall"] < min_recall:
        failures.append(f"recall {metrics['recall']:.4f} < soglia {min_recall:.4f}")
    if metrics["f1"] < min_f1:
        failures.append(f"f1 {metrics['f1']:.4f} < soglia {min_f1:.4f}")
    if metrics["roc_auc"] < min_roc_auc:
        failures.append(f"roc_auc {metrics['roc_auc']:.4f} < soglia {min_roc_auc:.4f}")

    if failures:
        raise AssertionError("Quality regression test fallito: " + " | ".join(failures))

    return metrics


def main(run_minimal_tests_flag: bool = RUN_MINIMAL_TESTS_DEFAULT) -> None:
    """Entrypoint CLI per valutare il modello di default del progetto."""

    if run_minimal_tests_flag:
        print("[evaluate] Esecuzione test minimi attiva")
        run_minimal_tests()
        print("[evaluate] Test minimi completati")

    if RUN_QUALITY_GATE_DEFAULT:
        print("[evaluate] Esecuzione quality regression test attiva")
        gate_metrics = run_quality_regression_test()
        print(f"[evaluate] Quality regression test superato: {gate_metrics}")

    metrics = evaluate_model(
        model_path=MODELS_DIR / "churn_pipeline_v1.joblib",
        test_data_path=PROC_DIR / "test_raw.csv",
        out_dir=OUT_DIR,
        plot_out_dir=OUT_DIR,
        target_col=TARGET_COL,
    )
    print("Valutazione completata:", metrics)


if __name__ == "__main__":
    main()
