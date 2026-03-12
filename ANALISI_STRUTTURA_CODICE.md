# Analisi e valutazione della struttura del codice

## 1) Executive summary

La repository ha una **struttura macro coerente per un progetto ML end-to-end** (data, ml, backend, frontend, analysis, models, outputs) e una buona separazione per domini funzionali.

Punti positivi principali:
- pipeline ML completa con preprocessing, training, valutazione e inferenza;
- presenza di artifact e output già versionati utili a demo/portfolio;
- presenza di backend API e frontend per esposizione del modello.

Rischi principali:
- alcune incongruenze tra documentazione e implementazione reale;
- moduli con logica eseguita all'import (effetti collaterali e avvio costoso);
- API backend che invoca funzioni non presenti nei moduli chiamati;
- coesistenza di due frontend (Streamlit e web statico) con contratto API non allineato in modo uniforme.

Valutazione sintetica:
- **Architettura generale:** 7/10
- **Manutenibilità:** 5.5/10
- **Prontezza produzione:** 4.5/10
- **Valore didattico/portfolio:** 8/10

---

## 2) Punti di forza strutturali

### 2.1 Separazione per layer
La struttura a cartelle (`ml/`, `analysis/`, `backend/`, `frontend/`, `utils/`) è chiara e leggibile. Questo aiuta onboarding e ownership per area.

### 2.2 Pipeline ML solida come concetto
Nel training viene usata una `Pipeline` sklearn con `ColumnTransformer` (imputazione + scaling + one-hot), e ciò riduce mismatch tra train e inferenza.

### 2.3 Inferenza resiliente ai nomi in input
`ml/predict.py` allinea schema e rinomina campi in ingresso, con gestione colonne mancanti e feature derivate (`AvgMonthlySpend`, `NumServices`). Questo è un buon compromesso pragmatico per integrare UI/API eterogenee.

### 2.4 Attenzione a diagnostica del modello
Nel training sono inclusi tuning (Optuna), handling sbilanciamento classi, early stopping, learning curve e diagnostica overfitting. Per progetto portfolio è un plus concreto.

---

## 3) Criticità architetturali (priorità alta)

### 3.1 Contratto backend non allineato al codice reale
In `backend/api.py` gli endpoint `/train`, `/evaluate`, `/generate-plots` chiamano funzioni che non risultano implementate con quei nomi (`run_training_pipeline`, `evaluate_model`, `generate_all_plots`).

**Impatto:** endpoint potenzialmente non funzionanti a runtime.

### 3.2 Esecuzione pesante all'import dei moduli ML
`ml/train_model.py` e `ml/evaluate.py` eseguono logica operativa a livello modulo (carico dati/artifact, training/evaluation) senza essere racchiusa integralmente in funzioni orchestrabili.

**Impatto:** importare i moduli dal backend può avere effetti collaterali e costi computazionali inattesi.

### 3.3 Disallineamenti documentazione vs implementazione
`README.md` descrive endpoint e feature non coerenti con il codice corrente (ad es. endpoint batch/results/download citati ma non presenti in `backend/api.py`; riferimento a RandomForest mentre il training usa XGBoost/CatBoost).

**Impatto:** riduce affidabilità percepita e velocità di onboarding.

### 3.4 Doppia strategia frontend non governata
Sono presenti sia `frontend/dashboard.py` (Streamlit) sia frontend statico (`index.html`, `script.js`, `style.css`). Non emerge una convenzione esplicita su quale sia il canale ufficiale.

**Impatto:** duplicazione effort, rischio divergenza UX/API.

---

## 4) Criticità di qualità/manutenibilità (priorità media)

1. **Naming inconsistente colonne target** (`Churn Value` vs `ChurnValue`) in più punti.
2. **Mix lingua IT/EN** in file, commenti, nomi variabili e output utente.
3. **File artifact nel repo** (modelli, immagini, output): ottimo per demo, ma da governare con politica chiara (`models/` e `outputs/` potrebbero crescere rapidamente).
4. **Hardcoded path/parametri** in più script: utile per prototipo, meno per esecuzioni ripetibili multi-ambiente.

---

## 5) Raccomandazioni operative (roadmap)

### Sprint 1 (stabilizzazione tecnica)
- Introdurre funzioni orchestratrici esplicite:
  - `ml.train_model.run_training_pipeline(...)`
  - `ml.evaluate.evaluate_model(...)`
  - `analysis.plots.generate_all_plots(...)`
- Spostare tutta la logica operativa dentro funzioni e mantenere nel modulo solo definizioni + `if __name__ == "__main__"`.
- Allineare naming target/feature in un solo standard.

### Sprint 2 (contratti e documentazione)
- Definire un unico contratto API (OpenAPI già disponibile via FastAPI).
- Aggiornare README con:
  - stack reale (XGBoost/CatBoost);
  - endpoint effettivi;
  - flusso frontend ufficiale.
- Aggiungere `docs/architecture.md` con diagramma componenti e flussi I/O.

### Sprint 3 (industrializzazione leggera)
- Introdurre test minimi:
  - unit test su `clean_raw` e mapping inferenza;
  - smoke test endpoint `/predict`.
- Aggiungere configurazione centralizzata (`settings.py` o `.env`).
- Definire policy artifact/versioning (es. tenere solo ultimo modello + metriche).

---

## 6) Valutazione finale

Il progetto è **molto buono come base didattica e portfolio end-to-end**, con una struttura riconoscibile e un workflow ML completo. Per essere robusto in contesto team/prodotto serve però una fase di consolidamento: ridurre effetti collaterali all'import, riallineare contratti API-funzioni, e rendere documentazione e naming pienamente consistenti.

Con 1–2 sprint di hardening, la codebase può passare da “ottimo prototipo dimostrativo” a “progetto tecnicamente affidabile per evoluzione continuativa”.
