"""
src/evaluation.py
-----------------
Helpers pour l'evaluation des modeles : metriques + visualisations.
Reutilisable par le notebook 04_evaluation.py et par le dashboard.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calcule les principales metriques de regression.

    - MAE  : erreur absolue moyenne (en unite de Sales -> millions)
    - RMSE : racine de l'erreur quadratique moyenne (penalise les grosses erreurs)
    - R^2  : variance expliquee (1 = parfait, 0 = predit la moyenne)
    - MAPE : erreur en pourcentage (interpretable pour le metier)
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def detect_overfitting(train_metrics: dict, test_metrics: dict) -> Dict[str, float]:
    """
    Compare les metriques train vs test pour detecter l'overfitting.

    - Gap R^2 > 0.05 => potentiel overfitting
    - Gap RMSE relatif > 20% => idem
    """
    gap_r2 = train_metrics["R2"] - test_metrics["R2"]
    gap_rmse_pct = (test_metrics["RMSE"] - train_metrics["RMSE"]) / train_metrics["RMSE"] * 100

    if gap_r2 > 0.05 or gap_rmse_pct > 20:
        verdict = "OVERFITTING"
    elif gap_r2 < -0.02 or gap_rmse_pct < -10:
        verdict = "UNDERFITTING"
    else:
        verdict = "OK"

    return {
        "gap_R2": round(gap_r2, 4),
        "gap_RMSE_%": round(gap_rmse_pct, 2),
        "verdict": verdict,
    }


def compute_residuals(y_true, y_pred) -> pd.DataFrame:
    """Renvoie un DataFrame avec residus + stats utiles."""
    residuals = y_true - y_pred
    return pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
        "pct_error": np.abs(residuals) / np.abs(y_true) * 100,
    })
