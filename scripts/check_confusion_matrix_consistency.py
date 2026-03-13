"""Check di coerenza confusion matrix su artifact modello corrente.

Uso:
python scripts/check_confusion_matrix_consistency.py
python scripts/check_confusion_matrix_consistency.py --expected 771 261 72 297
"""

from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"
DEFAULT_TEST = ROOT / "data" / "processed" / "test_raw.csv"
TARGET_COL = "Churn Value"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--expected",
        nargs=4,
        type=int,
        metavar=("TN", "FP", "FN", "TP"),
        help="Valori attesi confusion matrix per confronto rapido",
    )
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.test)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    tn, fp, fn, tp = confusion_matrix(y, pipeline.predict(X)).ravel()
    print(f"[cm] current TN={tn} FP={fp} FN={fn} TP={tp}")

    if args.expected:
        etn, efp, efn, etp = args.expected
        matches = (tn, fp, fn, tp) == (etn, efp, efn, etp)
        print(f"[cm] expected TN={etn} FP={efp} FN={efn} TP={etp}")
        print(f"[cm] match={matches}")


if __name__ == "__main__":
    main()
