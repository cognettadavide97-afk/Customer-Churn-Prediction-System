"""Training script (XGBoost + Optuna).

Scopo del modulo
----------------
Questo file produce un artifact unico (una sklearn.Pipeline) che include:
- preprocessing (imputazione, scaling, one-hot encoding)
- modello (XGBoost)

Perche una Pipeline
-------------------
Salvare una Pipeline (preprocessor + modello) evita mismatch tra:
- training: feature engineering + trasformazioni
- inferenza: input (API/UI) che deve subire le stesse trasformazioni

Input attesi
------------
- data/processed/train_raw.csv generato da ml/preprocessing.py
- Colonna target: "Churn Value" (0/1) + feature numeriche e categoriche

Output prodotti
--------------
- models/churn_pipeline_v1.joblib: pipeline XGBoost addestrata, usata da evaluate.py e predict.py
- models/churn_ensemble_v1.joblib: artifact con XGBoost + CatBoost + stacking classifier (LogisticRegression finale)

Concetti chiave
---------------
- Class imbalance: scale_pos_weight = neg/pos bilancia la classe positiva in XGBoost
- Tuning con Optuna: ottimizza F1 su cross-validation stratificata

Nota su F1 e soglia
-------------------
scoring="f1" usa predizioni binarie con soglia 0.5.
La soglia in produzione puo essere gestita in ml/predict.py (parametro threshold).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def tune_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Ottimizza gli iperparametri di XGBoost con Optuna."""

    import optuna

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = XGBClassifier(**{**base_params, **trial_params})
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X, y, scoring="f1", cv=cv, n_jobs=-1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def _build_catboost_model(scale_pos_weight: float, random_state: int = 42) -> CatBoostClassifier:
    """Costruisce un CatBoostClassifier con parametri stabili/clonabili."""

    class_weights = (1.0, float(scale_pos_weight)) if scale_pos_weight > 0 else (1.0, 1.0)
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=random_state,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        class_weights=class_weights,
        verbose=False,
    )


def train_catboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    random_state: int = 42,
) -> Pipeline:
    """Allena un secondo modello predittivo con CatBoost dentro una Pipeline sklearn."""

    cat_model = _build_catboost_model(scale_pos_weight=scale_pos_weight, random_state=random_state)

    cat_pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", cat_model),
        ]
    )
    cat_pipeline.fit(X, y)
    return cat_pipeline


def train_sklearn_stacking_classifier_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    xgb_params: dict,
    random_state: int = 42,
) -> StackingClassifier:
    """Allena un ensemble con sklearn StackingClassifier usando XGBoost e CatBoost."""

    xgb_estimator = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", XGBClassifier(**xgb_params)),
        ]
    )

    cat_estimator = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", _build_catboost_model(scale_pos_weight=scale_pos_weight, random_state=random_state)),
        ]
    )

    stack_clf = StackingClassifier(
        estimators=[("xgb", xgb_estimator), ("cat", cat_estimator)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=False,
        stack_method="predict_proba",
    )

    stack_clf.fit(X, y)
    return stack_clf


def train_xgb_catboost_linear_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    xgb_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    model_params: dict,
    random_state: int = 42,
) -> dict:
    """Allena l'ensemble con StackingClassifier e restituisce i componenti fitted."""

    xgb_fitted = clone(xgb_pipeline)
    xgb_fitted.fit(X, y)

    cat_fitted = train_catboost_pipeline(
        X=X,
        y=y,
        preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    stack_classifier = train_sklearn_stacking_classifier_ensemble(
        X=X,
        y=y,
        preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight,
        xgb_params=model_params,
        random_state=random_state,
    )

    return {
        "xgb_pipeline": xgb_fitted,
        "cat_pipeline": cat_fitted,
        "stack_classifier": stack_classifier,
    }


def predict_with_linear_ensemble(
    stack_classifier: StackingClassifier,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice probabilità e classi usando StackingClassifier (XGBoost + CatBoost)."""

    ensemble_prob = stack_classifier.predict_proba(X)[:, 1]
    ensemble_pred = (ensemble_prob >= threshold).astype(int)
    return ensemble_prob, ensemble_pred


# ------------------------------
# 1) Setup path di progetto
# ------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# 2) Caricamento dati di training
# ------------------------------
df_train = pd.read_csv(PROC_DIR / "train_raw.csv")
print(f"Train shape: {df_train.shape}")

target_col = "Churn Value"
X_raw = df_train.drop(columns=[target_col])
y_raw = df_train[target_col]


# ------------------------------
# 3) Class imbalance handling
# ------------------------------
pos = int((y_raw == 1).sum())
neg = int((y_raw == 0).sum())
scale_pos_weight = (neg / pos) if pos else 1.0

print(f"scale_pos_weight: {scale_pos_weight:.3f}")
print(f"Churn rate (mean target): {y_raw.mean():.3f}")


# ------------------------------
# 4) Definizione preprocessing
# ------------------------------
num_features = X_raw.select_dtypes(include=["number"]).columns.tolist()
cat_features = X_raw.select_dtypes(include=["object"]).columns.tolist()

print(f"Features -> numeric: {len(num_features)} | categorical: {len(cat_features)}")

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features),
    ]
)


# ------------------------------
# 5) Tuning Optuna
# ------------------------------
best_params = tune_xgb_with_optuna(
    X_raw,
    y_raw,
    preprocessor,
    scale_pos_weight=scale_pos_weight,
)
print(f"Optuna best params: {best_params}")


# ------------------------------
# 6) Training finale con best params
# ------------------------------
model_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
}
model_params.update(best_params)

# Modello standard per pipeline/diagnostica/salvataggio
model = XGBClassifier(**model_params)

# Pipeline finale standard (senza early stopping)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# Modello separato solo per training con early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

print("Training model (early stopping)...")

preprocessor_es = clone(preprocessor)
X_tr_p = preprocessor_es.fit_transform(X_tr)
X_val_p = preprocessor_es.transform(X_val)

model_es = XGBClassifier(**{**model_params, "early_stopping_rounds": 30})
model_es.fit(
    X_tr_p,
    y_tr,
    eval_set=[(X_val_p, y_val)],
    verbose=False,
)

best_iteration = getattr(model_es, "best_iteration", None)
if best_iteration is not None:
    model_params["n_estimators"] = int(best_iteration) + 1
    print(f"Early stopping -> n_estimators finale: {model_params['n_estimators']}")

print("Training completed.")

# Aggiorno pipeline finale con il numero di alberi stimato dall'early stopping
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(**model_params)),
    ]
)

# Fit finale della pipeline completa su tutto il dataset
# così la pipeline salvata è davvero fitted e riutilizzabile
pipeline.fit(X_raw, y_raw)

# ------------------------------
# 6-bis) Training ensemble XGBoost + CatBoost + LogisticRegression
# ------------------------------
ensemble_artifacts = train_xgb_catboost_linear_ensemble(
    X=X_raw,
    y=y_raw,
    xgb_pipeline=pipeline,
    preprocessor=preprocessor,
    scale_pos_weight=scale_pos_weight,
    model_params=model_params,
    random_state=42,
)

ensemble_prob_train, ensemble_pred_train = predict_with_linear_ensemble(
    stack_classifier=ensemble_artifacts["stack_classifier"],
    X=X_raw,
)
ensemble_f1_train = f1_score(y_raw, ensemble_pred_train, zero_division=0)
ensemble_auc_train = roc_auc_score(y_raw, ensemble_prob_train)
print(f"Ensemble train F1: {ensemble_f1_train:.4f} | AUC: {ensemble_auc_train:.4f}")


# ------------------------------
# Overfitting diagnostics
# ------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

diag_pipeline = clone(pipeline)
diag_pipeline.fit(X_tr, y_tr)

pred_tr = diag_pipeline.predict(X_tr)
proba_tr = diag_pipeline.predict_proba(X_tr)[:, 1]
pred_te = diag_pipeline.predict(X_te)
proba_te = diag_pipeline.predict_proba(X_te)[:, 1]

m_tr = _metrics(y_tr, pred_tr, proba_tr)
m_te = _metrics(y_te, pred_te, proba_te)

labels = list(m_tr.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 4))
plt.bar(x - width / 2, [m_tr[k] for k in labels], width, label="train")
plt.bar(x + width / 2, [m_te[k] for k in labels], width, label="test")
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.title("Diagnostica overfitting: metriche train vs test")
plt.legend()
plt.tight_layout()
plt.savefig(MODELS_DIR / "overfitting_train_vs_test.png")
plt.close()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    clone(pipeline),
    X_raw,
    y_raw,
    cv=3,
    scoring="f1",
    train_sizes=np.linspace(0.2, 1.0, 6),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 4))
plt.plot(train_sizes, train_mean, "o-", label="train F1")
plt.plot(train_sizes, test_mean, "o-", label="cv F1")
plt.xlabel("Numero esempi di training")
plt.ylabel("F1-score")
plt.title("Learning curve (F1)")
plt.legend()
plt.tight_layout()
plt.savefig(MODELS_DIR / "learning_curve_f1.png")
plt.close()


# ------------------------------
# 7) Salvataggio artifact
# ------------------------------
model_path = MODELS_DIR / "churn_pipeline_v1.joblib"
ensemble_path = MODELS_DIR / "churn_ensemble_v1.joblib"
joblib.dump(pipeline, model_path)
joblib.dump(ensemble_artifacts, ensemble_path)
print("Pipeline addestrata e salvata correttamente.")
print(f"Model saved to: {model_path}")
print(f"Ensemble saved to: {ensemble_path}")


# ------------------------------
# 8) Quick interpretability check
# ------------------------------
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False).head(10)

feature_importance_df.plot(kind="barh", x="feature", y="importance")
plt.title("Top 10 Feature per importanza nel Churn")
plt.tight_layout()
plt.savefig(MODELS_DIR / "feature_importance.png")
plt.close()