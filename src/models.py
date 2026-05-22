"""
src/models.py
-------------
Definitions des 4 modeles + grilles d'hyperparametres pour GridSearchCV.

Strategie :
- Chaque modele est encapsule dans un Pipeline complet (preprocessor + estimator)
- Cela garantit zero data leakage MEME pendant la cross-validation
  (le preprocessor est refit a chaque fold uniquement sur les donnees train de ce fold)
- Les grilles sont VOLONTAIREMENT raisonnables (pas de sur-tuning)

Justifications des modeles :
- LinearRegression : baseline indispensable, interpretable, rapide
- RandomForest : capture les non-linearites, robuste, peu sensible aux hyperparams
- XGBoost : etat de l'art sur donnees tabulaires, gradient boosting
- MLPRegressor : Deep Learning (multi-couches + backprop), exigence du sujet
"""

from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor


# =============================================================================
# DEFINITION DES MODELES
# =============================================================================

def get_models_config() -> Dict[str, Dict]:
    """
    Renvoie un dictionnaire {nom_modele: {estimator, param_grid}}.

    param_grid : prefixe 'model__' obligatoire car les hyperparametres
                 sont passes au sein du Pipeline (etape nommee 'model').
    """

    config = {

        # 1. BASELINE : Regression Lineaire
        # ---------------------------------
        # Aucun hyperparametre a tuner (modele entierement determine).
        # Sert de reference pour mesurer le gain apporte par les autres.
        "Linear Regression": {
            "estimator": LinearRegression(),
            "param_grid": {},  # pas de tuning
        },

        # 2. RANDOM FOREST
        # ----------------
        # Capture les non-linearites et interactions sans feature engineering.
        # Robuste aux outliers, peu sensible au scaling.
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
            # 2 * 3 * 2 = 12 combinaisons * 5 folds = 60 fits
        },

        # 3. XGBOOST (Gradient Boosting)
        # ------------------------------
        # Etat de l'art sur donnees tabulaires, generalement le plus performant.
        # Tres bon compromis biais/variance via boosting + regularisation.
        "XGBoost": {
            "estimator": XGBRegressor(
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                objective="reg:squarederror",
            ),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
            },
            # 2 * 3 * 2 = 12 combinaisons * 5 folds = 60 fits
        },

        # 4. MLP (Deep Learning - sklearn)
        # --------------------------------
        # Multi-Layer Perceptron : reseau de neurones avec backpropagation.
        # Architecture: input -> hidden layer(s) -> output.
        # IMPORTANT : necessite des features scaled (assure par notre pipeline).
        "MLP (Deep Learning)": {
            "estimator": MLPRegressor(
                random_state=42,
                max_iter=500,
                early_stopping=True,        # arret anticipe si plateau (anti-overfit)
                validation_fraction=0.1,    # 10% du train pour early stopping
                n_iter_no_change=20,
            ),
            "param_grid": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
                "model__alpha": [0.0001, 0.001],  # L2 regularization
                "model__learning_rate_init": [0.001, 0.01],
            },
            # 4 * 2 * 2 = 16 combinaisons * 5 folds = 80 fits
        },
    }

    return config


# =============================================================================
# CONSTRUCTION D'UN PIPELINE COMPLET (preprocessing + modele)
# =============================================================================

def build_full_pipeline(estimator) -> Pipeline:
    """
    Construit le pipeline complet : preprocessing + modele.

    Pourquoi : permet la cross-validation sans data leakage car le preprocessor
    sera refit a chaque fold uniquement sur les donnees train de ce fold.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
