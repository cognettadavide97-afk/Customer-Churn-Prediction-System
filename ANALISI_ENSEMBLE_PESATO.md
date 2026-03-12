# Analisi ensemble pesato (XGBoost dominante) vs XGBoost-only

## Richiesta implementata
È stato introdotto un **ensemble pesato** con priorità XGBoost:
- `xgb_weight = 0.90`
- `secondary_weight = 0.10`

Obiettivo: mantenere l'ensemble ma minimizzare il contributo del secondo modello.

## Modifiche tecniche
- In `run_training_pipeline(...)` è stato aggiunto `xgb_ensemble_weight` e il salvataggio del blocco `weighted_blend` nell'artifact ensemble.
- In `evaluate.py` l'ensemble viene valutato prioritariamente come blend pesato XGBoost-dominant.
- In `predict.py` la predizione online usa il blend pesato quando presente.

## Confronto run (nuovo)
Metriche complete in `outputs/weighted_ensemble_comparison.csv`.

### Test set
- **weighted_ensemble_xgb_base**:
  - Accuracy: 0.7559
  - Precision: 0.5242
  - Recall: 0.7940
  - F1: 0.6315
  - ROC-AUC: 0.8566
  - Confusion: TN=766, FP=266, FN=76, TP=293

- **weighted_ensemble_blend_90_10**:
  - Accuracy: 0.7566
  - Precision: 0.5251
  - Recall: 0.7940
  - F1: 0.6321
  - ROC-AUC: 0.8576
  - Confusion: TN=767, FP=265, FN=76, TP=293

- **weighted_xgb_only_base**:
  - Accuracy: 0.7630
  - Precision: 0.5332
  - Recall: 0.8049
  - F1: 0.6415
  - ROC-AUC: 0.8555
  - Confusion: TN=772, FP=260, FN=72, TP=297

## Lettura dei risultati
1. Il blend pesato 90/10 migliora leggermente il solo XGB della stessa run ensemble, ma non cambia recall/FN.
2. Il modello XGBoost-only resta migliore per obiettivo churn-interception:
   - recall più alta (0.8049)
   - FN più bassi (72 vs 76)
   - F1 migliore.
3. L'ensemble pesato è utile solo per piccoli guadagni incrementali, ma in questo dataset non supera XGB-only sulla metrica chiave di business.

## Raccomandazione
Per un piano aziendale orientato a intercettare più churn possibili, usare **XGBoost-only** come modello primario.
L'ensemble pesato 90/10 può restare come variante di controllo, ma non come modello di produzione principale.
