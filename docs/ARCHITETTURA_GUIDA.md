# Guida Architettura (fonte principale)

## Obiettivo
Fornire una vista unica dell'architettura tecnica del progetto churn, utile per sviluppo, manutenzione e allineamento business.

## Componenti
- `ml/preprocessing.py`: pulizia dati, feature engineering, split train/test.
- `ml/train_model.py`: training XGBoost (Optuna + early stopping) e diagnostica.
- `ml/evaluate.py`: valutazione modello, report metriche e confusion matrix.
- `ml/predict.py`: inferenza singolo record con allineamento schema input.
- `analysis/plots.py`: orchestrazione grafici analitici (`generate_all_plots`).
- `backend/api.py`: endpoint per predizione, preprocessing, training, evaluation, plotting.

## Flusso dati
1. `data/raw/Telco_customer_churn.csv`
2. `ml/preprocessing.py` -> `data/processed/train_raw.csv`, `test_raw.csv`
3. `ml/train_model.py` -> `models/churn_pipeline_v1.joblib` + grafici diagnostici
4. `ml/evaluate.py` -> `outputs/metrics.csv`, `classification_report.txt`, `confusion_matrix_xgb.png`
5. `ml/predict.py` / `backend/api.py` -> scoring operativo

## Scelte tecniche chiave
- Modello unico in produzione: **XGBoost** (riduce complessità e rischio operativo).
- Tuning con Optuna orientato recall (`F2`) per obiettivo churn interception.
- Quality gate su metriche business in `ml/evaluate.py` (`run_quality_regression_test`).
- Test minimi integrati nei moduli ML (`run_minimal_tests`) per regressioni contrattuali.

## Nota documentale
Questa guida e `docs/RUNBOOK.md` sono i riferimenti principali. I file `ANALISI_*.md` restano come storico/approfondimento.
