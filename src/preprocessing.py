"""
src/preprocessing.py
--------------------
Module de preprocessing reutilisable par :
- L'entrainement des modeles (notebooks/03_modeling.py)
- Le dashboard Streamlit (dashboard/app.py)
- L'API FastAPI (optionnelle)

Principe cle : un seul pipeline sklearn pour TOUT le projet, fit UNIQUEMENT
sur le train set pour eviter toute fuite de donnees (data leakage).
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# CONSTANTES METIER
# =============================================================================

TARGET = "Sales"

NUMERIC_FEATURES = ["TV", "Radio", "Social Media"]
CATEGORICAL_FEATURES = ["Influencer"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categories Influencer dans l'ordre business (de la plus petite a la plus grande portee)
INFLUENCER_CATEGORIES = ["Nano", "Micro", "Macro", "Mega"]


# =============================================================================
# CHARGEMENT DES DONNEES
# =============================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Charge le CSV brut sans transformation."""
    df = pd.read_csv(path)
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes ou la cible Sales est manquante.
    Justification : on ne peut pas apprendre une prediction sans label.
    Ne PAS imputer une cible (ce serait une grave faute methodologique).
    """
    n_before = len(df)
    df_clean = df.dropna(subset=[TARGET]).reset_index(drop=True)
    n_dropped = n_before - len(df_clean)
    if n_dropped > 0:
        print(f"[clean_target] {n_dropped} lignes supprimees (NaN sur {TARGET})")
    return df_clean


# =============================================================================
# CONSTRUCTION DU PIPELINE
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer qui prend en charge :
    - Numeriques : imputation mediane + standardisation
    - Categoriel : imputation par valeur constante + OneHot

    Le ColumnTransformer est FITTABLE : on l'entrainera UNIQUEMENT sur le train.
    """

    # Pipeline numerique : imputation puis scaling
    # - SimpleImputer(median) : robuste aux outliers (mieux que mean)
    # - StandardScaler : necessaire pour la Regression Lineaire et le MLP,
    #   sans impact sur les arbres (RF, XGBoost) mais inoffensif.
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Pipeline categoriel : imputation par categorie 'missing' puis OneHot
    # - handle_unknown='ignore' : permet de gerer des valeurs inedites en production
    #   sans crash (le dashboard pourrait envoyer une categorie inconnue).
    # - sparse_output=False : on prefere des arrays denses pour SHAP & MLP
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            categories=[INFLUENCER_CATEGORIES],
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # ignore les colonnes non listees
        verbose_feature_names_out=False,  # noms de features lisibles
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Recupere les noms des colonnes apres transformation (pour SHAP & importance)."""
    return preprocessor.get_feature_names_out().tolist()


# =============================================================================
# PREPARATION TRAIN/TEST
# =============================================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separe les features (X) de la cible (y)."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split train/test stratifie. Pas de stratification car regression
    (la stratification est pour classification deséquilibrée).
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"[split] Train : {X_train.shape[0]} lignes  |  Test : {X_test.shape[0]} lignes")
    return X_train, X_test, y_train, y_test


# =============================================================================
# SAUVEGARDE / CHARGEMENT DU PIPELINE
# =============================================================================

def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    """Serialise le preprocessor fit pour reutilisation (dashboard, API)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[save] Preprocessor sauvegarde : {path}")


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Charge un preprocessor deja entraine."""
    return joblib.load(path)
