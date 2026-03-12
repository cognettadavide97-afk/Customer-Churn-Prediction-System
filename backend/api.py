from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import analysis.plots
import ml.evaluate
import ml.predict as predictor
import ml.preprocessing
import ml.train_model

app = FastAPI(title="Telco Churn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

OUTPUTS_FRONTEND_DIR = ROOT / "outputs_front-end"
OUTPUTS_FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs-frontend", StaticFiles(directory=OUTPUTS_FRONTEND_DIR), name="outputs-frontend")


class CustomerData(BaseModel):
    Gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., description="Mesi di permanenza")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    record = data.model_dump()
    return predictor.predict_record(record)


@app.post("/preprocess")
def trigger_preprocessing():
    try:
        ml.preprocessing.main()
        return {"status": "success", "message": "Preprocessing completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train")
def trigger_training(n_trials: int = 50):
    try:
        model_path = ml.train_model.run_training_pipeline(
            train_data_path=ml.train_model.PROC_DIR / "train_raw.csv",
            models_dir=ml.train_model.MODELS_DIR,
            target_col="Churn Value",
            n_trials=n_trials,
        )
        return {"status": "success", "model_path": model_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/evaluate")
def trigger_evaluation():
    try:
        metrics = ml.evaluate.evaluate_model(
            model_path=ml.evaluate.MODELS_DIR / "churn_pipeline_v1.joblib",
            test_data_path=ml.evaluate.PROC_DIR / "test_raw.csv",
            out_dir=ml.evaluate.OUT_DIR,
            plot_out_dir=ml.evaluate.OUT_DIR,
            target_col="Churn Value",
        )
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/generate-plots")
def trigger_plots():
    try:
        analysis.plots.generate_all_plots(root_path=ROOT)
        return {"status": "success", "message": "Plots generated in outputs/plots"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

