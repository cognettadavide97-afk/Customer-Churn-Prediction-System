# Analisi aggiornata del progetto Customer Churn Prediction

## 1) Stato attuale del progetto

Il progetto è strutturato come pipeline ML end-to-end con separazione chiara tra:

- preparazione dati (`ml/preprocessing.py`),
- training (`ml/train_model.py`),
- valutazione (`ml/evaluate.py`),
- inferenza (`ml/predict.py`),
- analisi/visualizzazione (`analysis/plots.py`),
- orchestrazione API (`backend/api.py`).

La direzione tecnica è coerente: **modello singolo XGBoost ottimizzato con Optuna**, senza ensemble in runtime.

---

## 2) Punti di forza principali

### 2.1 Architettura governabile

La scelta XGBoost-only semplifica deploy e manutenzione:

- un artifact principale (`models/churn_pipeline_v1.joblib`),
- minore rischio di mismatch training/inferenza,
- rollback più rapido.

### 2.2 Orchestrazione import-safe

I moduli principali espongono funzioni e `main()` dedicati, riducendo side-effects all'import e facilitando integrazione backend.

### 2.3 Focus business-oriented sulle metriche

Il tuning usa F2 (priorità recall), mentre la valutazione mantiene set metriche comprensibile a stakeholder business (accuracy, precision, recall, F1, AUC + confusion matrix).

### 2.4 Maggiore chiarezza test minima

Sono presenti **test minimi obbligatori** nei moduli ML (`run_minimal_tests`) con toggle attivabile/disattivabile:

- training: verifica contratto preprocessing;
- evaluate: verifica disponibilità metriche chiave;
- predict: verifica contratto output e validazione soglia.

Questa base riduce regressioni banali e rende più affidabile l'evoluzione del codice.

---

## 3) Criticità ancora presenti

1. **Documentazione ridondante**: più file analitici con overlap informativo.
2. **Output sperimentali versionati nel repo**: ottimo per portfolio, ma rumoroso in sviluppo continuo.
3. **Frontend non completamente uniforme**: convivenza di componenti con maturità diversa.
4. **Testing ancora baseline**: buoni test minimi, ma manca suite completa (unit + integration + API smoke automatizzati in CI).

---

## 4) Valutazione tecnica sintetica

- **Robustezza pipeline ML:** 8/10
- **Manutenibilità codice:** 8/10
- **Prontezza produzione (small-scale):** 7.5/10
- **Qualità documentazione strategica:** 8/10
- **Maturità QA/testing:** 6.5/10

Valutazione complessiva: **progetto solido da portfolio avanzato, con base production-light concreta**.

---

## 5) Commenti sui test (stato attuale)

### Cosa coprono bene

- **Contratti fondamentali**: schema preprocessing, output metriche, output predict.
- **Errori frequenti**: soglie invalide e regressioni di firma/chiavi output.
- **Esecuzione rapida**: adatti a pre-check locale e CI veloce.

### Cosa manca ancora

- test di integrazione su pipeline completa con fixture stabile;
- test API endpoint (`/predict`, `/train`, `/evaluate`) in modalità automatica;
- test di regressione su qualità modello (soglie minime F1/recall).

### Raccomandazione immediata

Mantenere i test minimi sempre attivi in CI e aggiungere progressivamente 2 layer:

1. **Integration test ML** (train+evaluate su mini dataset),
2. **Smoke test API** con payload realistici.

---

## 6) Raccomandazioni prioritarie (prossimi step)

### Priorità alta

1. Consolidare la documentazione in 1-2 file guida (architettura + runbook).
2. Automatizzare in CI i test minimi esistenti.
3. Aggiungere un test integrazione end-to-end su dataset ridotto.

### Priorità media

4. Definire policy artifact/output (cosa versionare, cosa rigenerare).
5. Standardizzare report periodico soglia business (trade-off recall/FP).

### Priorità bassa

6. Rifinire UI/UX con percorso frontend unico.

---

## 7) Conclusione

La traiettoria del progetto è corretta: semplificazione architetturale, migliore governabilità e maggiore allineamento con obiettivi aziendali.

La scelta XGBoost-only, unita a tuning Optuna, diagnostica esplicita e test minimi obbligatori, costituisce una base credibile per un piano retention efficace e incrementale.
