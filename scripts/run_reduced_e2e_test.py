"""E2E integration test su dataset ridotto.

Scopo:
- eseguire un training veloce su un sottoinsieme dei dati processati
- valutare il modello risultante
- validare presenza output minimi e metriche chiave
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml.train_model as tm
import ml.evaluate as ev


def _build_reduced_dataset(df: pd.DataFrame, target_col: str, n_per_class: int) -> pd.DataFrame:
    """Crea un dataset ridotto bilanciato per classe."""

    chunks = []
    for cls in sorted(df[target_col].dropna().unique()):
        cls_df = df[df[target_col] == cls].head(n_per_class)
        chunks.append(cls_df)
    out = pd.concat(chunks, ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_df = pd.read_csv(root / "data" / "processed" / "train_raw.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test_raw.csv")

    reduced_train = _build_reduced_dataset(train_df, target_col=tm.TARGET_COL, n_per_class=120)
    reduced_test = _build_reduced_dataset(test_df, target_col=tm.TARGET_COL, n_per_class=60)

    with tempfile.TemporaryDirectory(prefix="churn_e2e_") as tmp:
        tmp_dir = Path(tmp)
        train_path = tmp_dir / "train_small.csv"
        test_path = tmp_dir / "test_small.csv"
        model_dir = tmp_dir / "models"
        out_dir = tmp_dir / "outputs"

        reduced_train.to_csv(train_path, index=False)
        reduced_test.to_csv(test_path, index=False)

        model_path = tm.run_training_pipeline(
            train_data_path=train_path,
            models_dir=model_dir,
            target_col=tm.TARGET_COL,
            n_trials=2,
            random_state=42,
        )

        metrics = ev.evaluate_model(
            model_path=model_path,
            test_data_path=test_path,
            out_dir=out_dir,
            plot_out_dir=out_dir,
            target_col=ev.TARGET_COL,
        )

        model_file = Path(model_path)
        assert model_file.exists(), "Artifact modello non creato"
        assert (out_dir / "metrics.csv").exists(), "metrics.csv non creato"
        assert (out_dir / "classification_report.txt").exists(), "classification_report.txt non creato"
        assert (out_dir / "confusion_matrix_xgb.png").exists(), "confusion_matrix_xgb.png non creata"

        key_metrics = metrics["xgb_pipeline"]
        for key in ["recall", "f1", "roc_auc"]:
            assert key in key_metrics, f"metrica {key} mancante"

        print("[integration] reduced e2e passed")
        print("[integration] metrics:", key_metrics)


if __name__ == "__main__":
    main()
