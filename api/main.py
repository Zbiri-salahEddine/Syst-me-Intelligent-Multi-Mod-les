"""
API REST - Optimisation du ROI Marketing
=========================================
Service d'inference FastAPI exposant le modele XGBoost en production.

Endpoints :
  GET  /health    -> statut du service et du modele
  POST /predict   -> prediction de ventes pour un scenario budgetaire
  GET  /model-info -> informations sur le modele en production

Lancement :
    cd "Projet Data Science"
    uvicorn api.main:app --reload --port 8000

Documentation Swagger auto :
    http://localhost:8000/docs
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import INFLUENCER_CATEGORIES  # noqa: E402

# =============================================================================
# CHARGEMENT DU MODELE (au demarrage, une seule fois)
# =============================================================================

MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"

_pipeline = None
_model_metadata: dict = {}


def _load_pipeline():
    global _pipeline, _model_metadata
    model_path = MODELS_PATH / "final_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Modele introuvable : {model_path}. "
            "Lancez d'abord notebooks/03_modeling.py et 04_evaluation.py."
        )
    _pipeline = joblib.load(model_path)

    # Lecture des metriques depuis le CSV d'evaluation
    eval_path = REPORTS_PATH / "evaluation_results.csv"
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        xgb_row = df[df["Model"] == "XGBoost"].iloc[0]
        _model_metadata = {
            "name": "XGBoost Regressor",
            "task": "regression",
            "target": "Sales (millions $)",
            "rmse_test": float(xgb_row["RMSE_test"]),
            "r2_test": float(xgb_row["R2_test"]),
            "mape_test_pct": float(xgb_row["MAPE_test_%"]),
            "overfitting_verdict": xgb_row["Verdict"],
        }


_load_pipeline()


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="ROI Marketing - API d'inference",
    description=(
        "Predit le volume de ventes (en millions $) a partir d'un mix media "
        "marketing (TV, Radio, Social Media, Influencer). "
        "Modele en production : XGBoost (R² = 0.9985, MAPE = 1.84%)."
    ),
    version="1.0.0",
    contact={"name": "Salah Eddine Zbiri", "email": "claudezbiri@gmail.com"},
)


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class PredictRequest(BaseModel):
    """Donnees d'entree pour la prediction de ventes."""

    TV: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Budget TV en millions $",
        examples=[230.1],
    )
    Radio: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Budget Radio en millions $",
        examples=[37.8],
    )
    Social_Media: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Budget Social Media en millions $",
        examples=[69.2],
        alias="Social Media",
    )
    Influencer: str = Field(
        ...,
        description=f"Type d'influencer : {INFLUENCER_CATEGORIES}",
        examples=["Macro"],
    )

    @field_validator("Influencer")
    @classmethod
    def validate_influencer(cls, v: str) -> str:
        if v not in INFLUENCER_CATEGORIES:
            raise ValueError(
                f"Influencer '{v}' invalide. Valeurs acceptees : {INFLUENCER_CATEGORIES}"
            )
        return v

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """Reponse de la prediction."""

    predicted_sales_M: float = Field(description="Ventes predites en millions $")
    roi_estimated: float = Field(description="ROI estime (Sales / Budget total)")
    total_budget_M: float = Field(description="Budget total en millions $")
    model_used: str = Field(description="Nom du modele utilise")
    input_received: dict = Field(description="Donnees d'entree recues (verification)")


class HealthResponse(BaseModel):
    """Reponse du endpoint de sante."""

    status: str
    model_loaded: bool
    model_name: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modele en production."""

    name: str
    task: str
    target: str
    rmse_test: float
    r2_test: float
    mape_test_pct: float
    overfitting_verdict: str
    features: list[str]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Etat du service",
    tags=["Monitoring"],
)
def health_check():
    """Verifie que le service est actif et que le modele est correctement charge."""
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        model_name=_model_metadata.get("name", "unknown"),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Informations sur le modele",
    tags=["Monitoring"],
)
def model_info():
    """Retourne les performances et les caracteristiques du modele en production."""
    if not _model_metadata:
        raise HTTPException(status_code=503, detail="Metadonnees du modele non disponibles.")

    preprocessor = _pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out().tolist()

    return ModelInfoResponse(
        name=_model_metadata["name"],
        task=_model_metadata["task"],
        target=_model_metadata["target"],
        rmse_test=_model_metadata["rmse_test"],
        r2_test=_model_metadata["r2_test"],
        mape_test_pct=_model_metadata["mape_test_pct"],
        overfitting_verdict=_model_metadata["overfitting_verdict"],
        features=feature_names,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prediction de ventes",
    tags=["Inference"],
)
def predict(request: PredictRequest):
    """
    Predit le volume de ventes (en millions $) pour un scenario budgetaire.

    - **TV** : budget television en millions $
    - **Radio** : budget radio en millions $
    - **Social Media** : budget social media en millions $ (alias accepte)
    - **Influencer** : type d'influencer parmi Nano / Micro / Macro / Mega

    Retourne la prediction de ventes, le ROI estime et le budget total.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modele non charge. Verifiez les logs.")

    # Construction du DataFrame d'entree (meme format que pendant l'entrainement)
    input_df = pd.DataFrame([{
        "TV": request.TV,
        "Radio": request.Radio,
        "Social Media": request.Social_Media,
        "Influencer": request.Influencer,
    }])

    # Prediction via le pipeline complet (preprocessing inclus)
    try:
        predicted_sales = float(_pipeline.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prediction : {str(e)}",
        )

    total_budget = request.TV + request.Radio + request.Social_Media
    roi = predicted_sales / total_budget if total_budget > 0 else 0.0

    return PredictResponse(
        predicted_sales_M=round(predicted_sales, 4),
        roi_estimated=round(roi, 4),
        total_budget_M=round(total_budget, 4),
        model_used=_model_metadata.get("name", "XGBoost"),
        input_received={
            "TV": request.TV,
            "Radio": request.Radio,
            "Social Media": request.Social_Media,
            "Influencer": request.Influencer,
        },
    )
