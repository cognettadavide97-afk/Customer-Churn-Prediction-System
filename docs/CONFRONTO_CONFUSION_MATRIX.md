# Confronto confusion matrix dopo rimozione ensemble

## Check eseguito
Comandi usati:

```bash
python ml/train_model.py
python ml/evaluate.py
python scripts/check_confusion_matrix_consistency.py
```

Output canonici aggiornati:
- `outputs/metrics.csv`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix_xgb.png`
- `models/overfitting_report.csv`

## Matrice di confusione corrente (XGBoost)
Valori correnti:
- TN = 761
- FP = 271
- FN = 66
- TP = 303

## Analisi overfitting
Il report `models/overfitting_report.csv` mostra gap train-test contenuti:
- gap F1 = 0.0191
- gap AUC = 0.0209

Questo indica una generalizzazione più stabile rispetto a configurazioni più aggressive (profondità elevata e minore regolarizzazione).

## Perché i numeri possono cambiare anche con solo XGBoost
La differenza non dipende dall'ensemble: può verificarsi anche in modalità XGBoost-only quando cambia l'artifact modello.

Cause principali:
1. retraining con iperparametri diversi,
2. artifact `churn_pipeline_v1.joblib` proveniente da run diversi,
3. differenze di ambiente (es. versione sklearn).

## Correzioni applicate per ridurre overfitting
In `ml/train_model.py` il tuning Optuna è stato aggiornato con:
- sampler seedato (`TPESampler(seed=random_state)`),
- search space più regolarizzato (`max_depth`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`),
- obiettivo che premia F2/AUC e penalizza il gap train-test in cross-validation.
