# Strategia del modello Churn: scelte progettuali in `ml/` e impatto sul piano aziendale

## 1) Obiettivo del modello (prodotto + business)

L'obiettivo del progetto non è solo classificare il churn, ma **guidare azioni di retention economicamente sostenibili**.
Il modello deve quindi:

1. intercettare una quota utile di clienti a rischio;
2. trasformare le probabilità in priorità operative;
3. mantenere stabile il processo decisionale nel tempo.

Per questo le scelte nella cartella `ml/` sono state orientate a **robustezza, ripetibilità e leggibilità operativa**, non alla sola complessità algoritmica.

---

## 2) Perché la progettazione in `ml/` è stata fatta così

## 2.1 `ml/preprocessing.py`: coerenza del dato prima del modello

- centralizza cleaning, feature engineering e split train/test;
- garantisce coerenza tra feature usate in training e in predizione;
- riduce errori di schema (una delle principali cause di drift operativo).

**Impatto business:** un dato più coerente riduce decisioni errate in campagne retention.

## 2.2 `ml/train_model.py`: pipeline unica XGBoost + Optuna

- usa una `Pipeline` sklearn (preprocess + model) per evitare mismatch training/inferenza;
- ottimizza iperparametri con Optuna;
- mantiene diagnostica incorporata (`overfitting_train_vs_test.png`, `learning_curve_f1.png`, `feature_importance.png`).

**Scelta chiave:** mantenere un solo modello principale (XGBoost), più semplice da governare rispetto all'ensemble.

**Impatto business:** meno complessità = più velocità di aggiornamento modello e deploy più affidabile.

## 2.3 `ml/evaluate.py`: output decision-ready

- produce metriche, classification report e confusion matrix in modo standardizzato;
- rende confrontabili versioni modello/soglia nel tempo.

**Impatto business:** dashboard KPI più chiara per decidere budget/campagne retention.

## 2.4 `ml/predict.py`: inferenza allineata e stabile

- allinea naming/input schema in arrivo da API/UI;
- usa sempre lo stesso artifact del training.
- include modalità `debug` e salvataggio grafico della probabilità per supportare spiegazione verso stakeholder non tecnici.

**Impatto business:** riduce errori a runtime e mantiene affidabilità dei punteggi churn usati dal team commerciale.

---

## 3) Perché i risultati possono essere validi anche con recall non massima e FP alti

Nel churn il costo degli errori è asimmetrico:

- **FN (falso negativo):** churn non intercettato → perdita ricavi e CLV;
- **FP (falso positivo):** cliente contattato inutilmente → costo operativo/campagna.

Un modello è ottimale quando il **trade-off economico netto** è favorevole, non quando una singola metrica è "perfetta".

Quindi:
- FP relativamente alti possono essere accettabili se il valore dei churn evitati compensa;
- recall non massima può essere comunque adeguata se combinata con ranking probabilistico e soglie operative corrette.

---

## 4) Perché F1 resta fondamentale per il piano aziendale

La F1 bilancia precision e recall, evitando estremi:

- solo recall alta → troppi FP e costi non sostenibili;
- solo precision alta → si perdono churn importanti.

Usare F1 come metrica di riferimento permette di scegliere un modello che resta utile sia al Data Team sia al Team Retention.

> Nota operativa: in fase di tuning la pipeline può privilegiare recall (es. F2), ma F1 resta essenziale per il governo complessivo del piano e per i confronti tra release.

---

## 5) Perché XGBoost (single model) è preferibile all'ensemble in questo caso

1. **Governance più semplice**
   - un artifact principale;
   - debugging e rollback più rapidi.

2. **Stabilità tecnica**
   - meno dipendenze/rami logici;
   - minore rischio di regressioni in API/inferenza.

3. **Rapporto beneficio/complessità migliore**
   - i confronti storici nel progetto mostrano guadagni ensemble non sempre netti sui KPI prioritari;
   - la complessità extra non è sempre giustificata.

4. **Adozione aziendale più veloce**
   - spiegabilità più lineare;
   - ciclo retrain-deploy più corto.

---

## 6) Grafici consigliati per valorizzare le scelte fatte

Per comunicare correttamente la validità del modello e la bontà della progettazione `ml/`, i grafici più efficaci sono:

1. **`outputs/confusion_matrix_xgb.png`**
   - mostra il trade-off reale tra FN e FP;
   - utile per definire soglia e capacità del team retention.

2. **`models/overfitting_train_vs_test.png`**
   - dimostra stabilità train/test;
   - supporta la fiducia nel modello per uso operativo.

3. **`models/learning_curve_f1.png`**
   - evidenzia andamento prestazionale al crescere dei dati;
   - aiuta a decidere se investire in più dati o in più tuning.

4. **`models/feature_importance.png`**
   - rende il modello spiegabile a stakeholder business;
   - aiuta a trasformare insight in azioni (es. piano su Contract/Charges/Tenure).

5. **`outputs/plots/Churn_Rate_by_Contract_Type.png`** + **`outputs/plots/churn_rate_grouped_by_tenure_group.png`**
   - collegano predizione e strategia commerciale;
   - ideali per definire campagne per segmento.

6. **Grafico predittivo singolo da `ml/predict.py`** (opzione `save_plot_path`)
   - mostra in modo immediato probabilità Stay vs Churn su singolo cliente;
   - utile per documenti di allineamento commerciale e discussioni operative caso per caso.

---

## 7) Traduzione in piano aziendale efficace

1. Scoring periodico clienti (settimanale o quindicinale).
2. Segmentazione in 3 fasce rischio (alta/media/bassa).
3. Regole di azione per fascia (incentivo, contatto, monitoraggio).
4. Monitoraggio KPI:
   - churn evitato,
   - costo campagna,
   - ROI retention,
   - stabilità metriche modello (F1, recall, AUC).
5. Revisione mensile soglia decisionale in base alla capacità commerciale.

Questo approccio rende la progettazione tecnica della cartella `ml/` direttamente utile alla strategia aziendale.

---

## 8) Conclusione

Le scelte fatte in `ml/` (pipeline unica, XGBoost ottimizzato, valutazione standardizzata, inferenza robusta) favoriscono un modello **più governabile, più stabile e più utile al business**.

In questo progetto, la combinazione tra semplicità architetturale e metriche bilanciate è il fattore che abilita davvero un buon piano retention, più di una complessità algoritmica non sempre premiata sul campo.
