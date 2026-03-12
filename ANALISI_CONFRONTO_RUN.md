# Confronto sperimentale: run con ensemble vs run XGBoost-only

## Setup eseguito

Sono state effettuate due run di training su `train_raw.csv` con tuning Optuna:

1. **Run con ensemble abilitato** (`enable_ensemble=True`):
   - modello base XGBoost ottimizzato
   - stacking classifier (XGBoost + modello secondario)
2. **Run senza ensemble** (`enable_ensemble=False`):
   - solo XGBoost ottimizzato con Optuna

Per entrambe le run è stata poi eseguita la valutazione su `test_raw.csv` e un confronto train/test per verificare overfitting.

I risultati numerici consolidati sono in: `outputs/run_comparison_metrics_detailed.csv`.

---

## Risultati principali (test set)

Dal test set emergono questi valori:

- **Ensemble run – XGBoost base**: Accuracy **0.7616**, F1 **0.6409**, Recall **0.8076**, Precision **0.5312**, ROC-AUC **0.8593**.
- **Ensemble run – Stacking**: Accuracy **0.8016**, F1 **0.6029**, Recall **0.5718**, Precision **0.6375**, ROC-AUC **0.8592**.
- **XGBoost-only run**: Accuracy **0.7602**, F1 **0.6403**, Recall **0.8103**, Precision **0.5292**, ROC-AUC **0.8592**.

### Lettura veloce

- Lo **stacking** migliora l'**accuracy** e la **precision**, ma riduce molto la **recall** e peggiora l'**F1** rispetto a XGBoost base.
- Il modello **XGBoost-only** ha prestazioni quasi sovrapponibili al base della run ensemble, con leggero vantaggio di recall.

---

## Analisi matrici di confusione (test)

Dalla matrice di confusione (`TN, FP, FN, TP`):

- **Ensemble run – XGBoost base**: `769, 263, 71, 298`
- **Ensemble run – Stacking**: `912, 120, 158, 211`
- **XGBoost-only run**: `766, 266, 70, 299`

### Interpretazione

- Lo **stacking** riduce molto i **falsi positivi** (120 vs ~265), ma aumenta molto i **falsi negativi** (158 vs ~70).
- Quindi lo stacking è più “conservativo”: segnala meno churn, con maggiore precisione ma peggior copertura dei churn reali.
- Se l'obiettivo business è **intercettare più churn possibili**, XGBoost base è più adatto (FN più bassi, recall più alta).

---

## Overfitting: confronto train vs test

Gap train-test osservati:

- **Ensemble run – XGBoost base**:
  - F1 gap: **0.0385**
  - AUC gap: **0.0320**
  - Accuracy gap: **0.0214**
- **Ensemble run – Stacking**:
  - F1 gap: **0.1076**
  - AUC gap: **0.0596**
  - Accuracy gap: **0.0520**
- **XGBoost-only run**:
  - F1 gap: **0.0252**
  - AUC gap: **0.0232**
  - Accuracy gap: **0.0113**

### Valutazione overfitting

- Lo **stacking** mostra il livello di overfitting più evidente (gap più alti su tutte le metriche).
- Il modello **XGBoost-only** è quello più stabile (gap più contenuti).
- Il base della run ensemble è intermedio.

---

## Motivazioni probabili

1. **Complessità modello**: lo stacking aggiunge un metaclassificatore e aumenta la capacità del modello, con maggiore rischio di adattamento al train.
2. **Obiettivo implicito della soglia 0.5**: lo stacking sposta il trade-off verso precisione, penalizzando recall.
3. **Dimensione/rumore del dataset**: con dati non enormi, modelli più complessi possono catturare pattern specifici del train.

---

## Conclusione operativa

- Se il KPI principale è **recall/F1 sul churn**, conviene usare **XGBoost-only ottimizzato con Optuna**.
- Se il KPI principale è ridurre i **falsi allarmi** (FP), lo **stacking** può essere considerato, ma con tuning soglia e forte monitoraggio overfitting.
- In stato attuale, la scelta più robusta è **XGBoost-only** per equilibrio prestazioni/stabilità.
