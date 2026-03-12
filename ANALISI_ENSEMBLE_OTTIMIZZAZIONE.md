# Analisi ensemble e ottimizzazione orientata al churn interception

## Obiettivo business
Massimizzare il numero di clienti churn intercettati (recall alta) per supportare un piano aziendale di retention.

## Modifiche applicate al modello
1. **Tuning Optuna orientato a recall**: da scoring F1 a **F2-score** (recall pesata maggiormente).
2. **Stacking più robusto**:
   - meta-modello LogisticRegression con `class_weight="balanced"`;
   - modello secondario dell'ensemble con fallback CatBoost/RandomForest bilanciato.
3. **Parametri XGBoost più stabili su classe minoritaria**: aggiunta di `max_delta_step=1`.

## Setup sperimentale
Sono state eseguite 2 run ottimizzate (entrambe con Optuna):

- **Run A (ensemble ON)**: `enable_ensemble=True`
- **Run B (XGBoost-only)**: `enable_ensemble=False`

Confronto numerico completo salvato in: `outputs/ensemble_optimization_comparison.csv`.

---

## Risultati test set

- **opt_ensemble_xgb_base**
  - Accuracy: **0.7587**
  - Precision: **0.5279**
  - Recall: **0.7940**
  - F1: **0.6342**
  - ROC-AUC: **0.8557**

- **opt_ensemble_stacking**
  - Accuracy: **0.7702**
  - Precision: **0.5438**
  - Recall: **0.7913**
  - F1: **0.6446**
  - ROC-AUC: **0.8589**

- **opt_xgb_only_base**
  - Accuracy: **0.7595**
  - Precision: **0.5288**
  - Recall: **0.7967**
  - F1: **0.6357**
  - ROC-AUC: **0.8563**

### Lettura business
- Se il KPI primario è **intercettare più churn possibili**, il miglior valore di recall è nel **XGBoost-only** (0.7967), di poco sopra gli altri.
- Lo **stacking** ottimizzato ha migliore equilibrio generale (AUC/F1/accuracy migliori), ma recall leggermente inferiore.

---

## Matrici di confusione (test)

- **opt_ensemble_xgb_base**: TN=770, FP=262, FN=76, TP=293
- **opt_ensemble_stacking**: TN=787, FP=245, FN=77, TP=292
- **opt_xgb_only_base**: TN=770, FP=262, FN=75, TP=294

### Interpretazione
- **XGBoost-only** riduce di 1 i falsi negativi rispetto allo stacking (75 vs 77): è piccolo ma coerente con l'obiettivo recall-first.
- **Stacking** riduce i falsi positivi (245 vs 262), utile se il costo di contattare clienti non-churn è alto.

---

## Overfitting (train vs test)

Gap osservati:

- **opt_ensemble_xgb_base**
  - F1 gap: **0.0173**
  - Recall gap: **0.0408**
  - AUC gap: **0.0150**

- **opt_ensemble_stacking**
  - F1 gap: **0.0267**
  - Recall gap: **0.0536**
  - AUC gap: **0.0281**

- **opt_xgb_only_base**
  - F1 gap: **0.0182**
  - Recall gap: **0.0448**
  - AUC gap: **0.0187**

### Valutazione
- Non emergono segnali di overfitting grave, ma lo stacking è il più sensibile (gap più alti).
- XGBoost-only e XGB base in ensemble risultano più stabili.

---

## Raccomandazione per piano aziendale

### Modello consigliato (priorità intercettazione churn)
**XGBoost-only ottimizzato (Run B)** per massimizzare recall nel contesto attuale.

### Modalità operativa suggerita
1. Usare ranking probabilistico churn e non solo classe binaria.
2. Definire 3 fasce operative:
   - **Alta priorità**: top rischio (intervento retention immediato)
   - **Media priorità**: contatto con offerte mirate
   - **Bassa priorità**: monitoraggio
3. Rivalutare soglia decisionale su base mensile (budget/capacità contatto).

### Quando usare lo stacking
Se l'azienda vuole ridurre il numero di contatti non necessari (FP), lo stacking è una valida alternativa con trade-off recall leggermente peggiore.
