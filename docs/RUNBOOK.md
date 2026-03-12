# Runbook Operativo

## 1) Setup
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Esecuzione pipeline ML
```bash
python ml/preprocessing.py
python ml/train_model.py
python ml/evaluate.py
```

## 3) Plot analitici
```bash
python -m analysis.plots
```

## 4) Test minimi modulo ML
```bash
python - <<'PY'
import ml.train_model as tm
import ml.evaluate as ev
import ml.predict as pr

tm.run_minimal_tests()
ev.run_minimal_tests()
pr.run_minimal_tests()
print('Minimal checks passed')
PY
```

## 5) Quality gate regressione qualità modello
```bash
python - <<'PY'
import ml.evaluate as ev
print(ev.run_quality_regression_test())
PY
```

## 6) Test integrazione ridotto end-to-end
```bash
python scripts/run_reduced_e2e_test.py
```

## 7) API
```bash
uvicorn backend.api:app --reload
```
Endpoint principali: `/predict`, `/preprocess`, `/train`, `/evaluate`, `/generate-plots`.
