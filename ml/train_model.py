"""Training pipeline for churn prediction.

Obiettivo:
- Caricare i dati processati
- Costruire preprocessing + modello
- Eseguire tuning e training finale
- Salvare artifact e grafici diagnostici
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TARGET_COL = "Churn Value"
RUN_MINIMAL_TESTS_DEFAULT = False

# Mappa didattica del modulo:
# 1) build_preprocessor -> definisce il contratto dati tra train/inferenza
# 2) tune_xgb_with_optuna -> cerca iperparametri con focus recall (F2)
# 3) run_training_pipeline -> orchestration completa e salvataggio artifact
# 4) save_diagnostics -> grafici per spiegare performance e rischio overfitting

# =========================
# Funzioni di supporto
# =========================
def tune_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Ottimizza XGBoost con Optuna usando F2 in cross-validation.

    F2 pesa maggiormente la recall rispetto alla precisione:
    utile quando l'obiettivo è intercettare più churn possibili.
    """

    import optuna

    # Parametri fissi che garantiscono coerenza tra trial e ripetibilità.
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    def objective(trial: optuna.Trial) -> float:
        # Iperparametri candidati per trovare il miglior compromesso bias/variance.
        # Il range è volutamente conservativo per evitare modelli troppo instabili
        # in produzione (focus: generalizzazione + robustezza operativa).
        # Search space pensato per ridurre overfitting senza deprimere recall:
        # - max_depth più contenuto evita alberi troppo specifici del train
        # - min_child_weight/gamma aumentano la soglia minima di split
        # - reg_alpha/reg_lambda aggiungono penalizzazione L1/L2
        # - subsample/colsample_bytree introducono bagging/feature sampling
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 650),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        }

        # Ogni trial allena l'intera pipeline (preprocess + modello), così
        # la metrica riflette il comportamento reale end-to-end.
        model = XGBClassifier(**{**base_params, **trial_params})
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        f2_scorer = make_scorer(fbeta_score, beta=2)
        # return_train_score=True è intenzionale: serve per stimare gap train-test
        # direttamente nella funzione obiettivo e scoraggiare trial overfittati.
        cv_res = cross_validate(
            pipeline,
            X,
            y,
            scoring={"f2": f2_scorer, "roc_auc": "roc_auc"},
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )

        mean_f2 = float(np.mean(cv_res["test_f2"]))
        mean_auc = float(np.mean(cv_res["test_roc_auc"]))
        overfit_gap = max(0.0, float(np.mean(cv_res["train_f2"]) - mean_f2))

        # Obiettivo bilanciato: massimizza capacità di intercettazione churn
        # penalizzando gap train/test (generalizzazione).
        # Peso business-oriented:
        # - F2 (75%): priorità intercettare churner (recall)
        # - AUC (25%): stabilità del ranking probabilistico
        # - penalità gap (10%): evita soluzioni con train score gonfiato
        return (0.75 * mean_f2) + (0.25 * mean_auc) - (0.10 * overfit_gap)

    # Pruner per ridurre costo computazionale interrompendo trial deboli.
    # Sampler seedato: rende la ricerca Optuna riproducibile tra run nello stesso ambiente.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Crea il preprocessing numerico + categorico in standard sklearn."""

    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

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

    # ColumnTransformer: contratto unico tra training ed inferenza.
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ]
    )


def _metrics(y_true, y_pred, y_prob) -> dict:
    """Calcola metriche standard per confronto train/test."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_diagnostics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, models_dir: Path) -> None:
    """Genera grafici diagnostici (overfitting, learning curve, feature importance)."""

    # 1) Overfitting check: confronto train/test su split holdout.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
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
    plt.savefig(models_dir / "overfitting_train_vs_test.png")
    plt.close()

    # Report numerico complementare al grafico:
    # utile in PR/CI per verificare rapidamente quanto è ampio il gap.
    pd.DataFrame(
        {
            "metric": labels,
            "train": [m_tr[k] for k in labels],
            "test": [m_te[k] for k in labels],
            "gap_train_minus_test": [m_tr[k] - m_te[k] for k in labels],
        }
    ).to_csv(models_dir / "overfitting_report.csv", index=False)

    # 2) Learning curve: andamento F1 al crescere dei dati.
    train_sizes, train_scores, test_scores = learning_curve(
        clone(pipeline),
        X,
        y,
        cv=3,
        scoring="f1",
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="train F1")
    plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="cv F1")
    plt.xlabel("Numero esempi di training")
    plt.ylabel("F1-score")
    plt.title("Learning curve (F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(models_dir / "learning_curve_f1.png")
    plt.close()

    # 3) Top feature importances dal modello XGBoost.
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    top_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .head(10)
    )

    top_df.plot(kind="barh", x="feature", y="importance")
    plt.title("Top 10 Feature per importanza nel Churn")
    plt.tight_layout()
    plt.savefig(models_dir / "feature_importance.png")
    plt.close()


# =========================
# Orchestrazione principale
# =========================
def run_training_pipeline(
    train_data_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    target_col: str = TARGET_COL,
    n_trials: int = 24,
    random_state: int = 42,
) -> str:
    """Esegue il training completo e salva gli artifact principali.

    Returns:
        Percorso del modello base salvato.
    """

    train_data_path = Path(train_data_path) if train_data_path else PROC_DIR / "train_raw.csv"
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Caricamento dataset e split X/y.
    df_train = pd.read_csv(train_data_path)
    if target_col not in df_train.columns:
        raise ValueError(f"Target '{target_col}' non trovato nel dataset.")

    X_raw = df_train.drop(columns=[target_col])
    y_raw = df_train[target_col]

    # 2) Bilanciamento classe positiva per XGBoost.
    # Questo passaggio è cruciale in churn prediction perché la classe churn
    # è tipicamente minoritaria e va resa più "visibile" al modello.
    pos = int((y_raw == 1).sum())
    neg = int((y_raw == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    # 3) Preprocessing e tuning.
    preprocessor = build_preprocessor(X_raw)
    best_params = tune_xgb_with_optuna(
        X_raw,
        y_raw,
        preprocessor,
        scale_pos_weight=scale_pos_weight,
        n_trials=n_trials,
        random_state=random_state,
    )

    # Template parametri finali del modello:
    # unisce baseline robusta + iperparametri trovati da Optuna.
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        # max_delta_step stabilizza update con classe sbilanciata
        # (evita salti troppo aggressivi nei log-odds della classe positiva).
        "max_delta_step": 1,
    }
    model_params.update(best_params)

    # 4) Early stopping su split interno per stimare n_estimators robusto.
    # Obiettivo: ridurre overfitting mantenendo capacità predittiva utile al business.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=random_state
    )
    preprocessor_es = clone(preprocessor)
    X_tr_p = preprocessor_es.fit_transform(X_tr)
    X_val_p = preprocessor_es.transform(X_val)

    model_es = XGBClassifier(**{**model_params, "early_stopping_rounds": 30})
    model_es.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)

    best_iteration = getattr(model_es, "best_iteration", None)
    if best_iteration is not None:
        # best_iteration è 0-indexed: +1 per ottenere il numero reale di alberi.
        model_params["n_estimators"] = int(best_iteration) + 1

    # 5) Fit finale del modello base su tutto il training set.
    # Dopo tuning + stima early-stopping, il fit finale usa tutti i dati
    # disponibili per massimizzare il segnale informativo.
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(**model_params)),
        ]
    )
    pipeline.fit(X_raw, y_raw)

    # 6) Salvataggio artifact (solo modello XGBoost ottimizzato).
    # Scelta progettuale: un solo artifact favorisce deploy e governance.
    model_path = models_dir / "churn_pipeline_v1.joblib"
    joblib.dump(pipeline, model_path)

    # 7) Diagnostica visiva per monitoraggio qualità modello.
    # Nota didattica: questi grafici sono leggibili anche da stakeholder non tecnici
    # e aiutano a motivare le scelte del piano retention (priorità recall).
    # I grafici supportano l'argomentazione tecnica verso stakeholder business.
    save_diagnostics(pipeline, X_raw, y_raw, models_dir=models_dir)

    return str(model_path)


def run_minimal_tests() -> None:
    """Test minimi obbligatori del modulo training (attivabili via flag).

    Questi test verificano i contratti fondamentali senza eseguire training completo.
    """

    # Test 1: schema misto numerico/categorico con missing.
    # Serve a verificare che il preprocessing gestisca correttamente i casi minimi reali.
    sample = pd.DataFrame(
        {
            "num_col": [1.0, 2.0, np.nan],
            "cat_col": ["A", "B", None],
            "Churn Value": [0, 1, 0],
        }
    )
    X = sample.drop(columns=["Churn Value"])
    preprocessor = build_preprocessor(X)

    assert isinstance(preprocessor, ColumnTransformer), "build_preprocessor deve restituire ColumnTransformer"

    # Test 2: il preprocessing non deve alterare il numero di record.
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X), "Numero righe preprocessate non coerente"


def main(run_minimal_tests_flag: bool = RUN_MINIMAL_TESTS_DEFAULT) -> None:
    """Entrypoint CLI del training."""

    if run_minimal_tests_flag:
        print("[train_model] Esecuzione test minimi attiva")
        run_minimal_tests()
        print("[train_model] Test minimi completati")

    model_path = run_training_pipeline()
    print(f"Training completato. Modello salvato in: {model_path}")


if __name__ == "__main__":
    main()
