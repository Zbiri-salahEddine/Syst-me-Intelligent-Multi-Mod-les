"""
03 - Modelisation : entrainement des 4 modeles avec GridSearchCV
=================================================================

Pour chaque modele :
1. Encapsulation dans un Pipeline complet (preprocessor + estimator)
2. GridSearchCV 5-fold sur le train uniquement
3. Selection des meilleurs hyperparametres
4. Re-fit sur tout le train avec ces hyperparams
5. Sauvegarde du pipeline (preprocessor + best estimator)

Sortie :
- models/<nom_modele>.joblib  (4 fichiers)
- models/training_results.csv (CV scores + temps + best params)
"""

# %%
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

# Permet d'importer src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_full_pipeline, get_models_config  # noqa: E402

# Chemins
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# %% 1. Chargement des donnees train/test
print("=" * 70)
print("CHARGEMENT DES DONNEES")
print("=" * 70)
X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")


# %% 2. Configuration de la cross-validation
print("\n" + "=" * 70)
print("CONFIGURATION CROSS-VALIDATION")
print("=" * 70)

# KFold 5 (pas Stratified : on est en regression)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrique : RMSE negatif (sklearn maximise, donc on neg-RMSE)
# r2 sera aussi suivi
scoring = {
    "neg_rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
refit_metric = "neg_rmse"  # on choisit le meilleur modele sur le RMSE

print(f"CV : KFold 5-folds (shuffle, random_state=42)")
print(f"Metrique principale : {refit_metric} (RMSE)")
print(f"Metriques suivies   : {list(scoring.keys())}")


# %% 3. Entrainement de chaque modele
print("\n" + "=" * 70)
print("ENTRAINEMENT DES 4 MODELES")
print("=" * 70)

models_config = get_models_config()
training_results = []

for model_name, cfg in models_config.items():
    print(f"\n{'-' * 70}")
    print(f">>> {model_name}")
    print(f"{'-' * 70}")

    # Construction du pipeline complet
    pipeline = build_full_pipeline(cfg["estimator"])

    # GridSearchCV (ou simple fit si pas de param_grid)
    t0 = time.time()

    if cfg["param_grid"]:
        n_combos = np.prod([len(v) for v in cfg["param_grid"].values()])
        print(f"GridSearchCV : {n_combos} combinaisons * 5 folds = {n_combos * 5} fits")

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_rmse = -grid.best_score_  # car neg_rmse
        cv_r2 = grid.cv_results_["mean_test_r2"][grid.best_index_]
    else:
        # Pas de tuning, on cross-validate juste pour avoir le score
        from sklearn.model_selection import cross_validate
        print("Pas de GridSearch (baseline) - Cross-validation seule")
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        cv_rmse = -cv_results["test_neg_rmse"].mean()
        cv_r2 = cv_results["test_r2"].mean()

        # Fit final sur tout le train
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = {}

    elapsed = time.time() - t0

    # Affichage des resultats
    print(f"\nResultats {model_name} :")
    print(f"  - CV RMSE       : {cv_rmse:.3f}")
    print(f"  - CV R^2        : {cv_r2:.4f}")
    print(f"  - Temps         : {elapsed:.1f}s")
    if best_params:
        print(f"  - Best params   : {best_params}")

    # Sauvegarde du modele
    model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = MODELS_PATH / f"{model_filename}.joblib"
    joblib.dump(best_model, save_path)
    print(f"  - Modele sauvegarde : {save_path.name}")

    # Stockage pour le tableau recapitulatif
    training_results.append({
        "Model": model_name,
        "CV_RMSE": round(cv_rmse, 4),
        "CV_R2": round(cv_r2, 4),
        "Time_sec": round(elapsed, 2),
        "Best_Params": str(best_params) if best_params else "N/A",
    })


# %% 4. Tableau recapitulatif des resultats CV
print("\n" + "=" * 70)
print("TABLEAU RECAPITULATIF (Cross-validation 5-fold sur le TRAIN)")
print("=" * 70)

results_df = pd.DataFrame(training_results)
results_df = results_df.sort_values("CV_RMSE", ascending=True).reset_index(drop=True)
print(results_df.to_string(index=False))

# Sauvegarde des resultats
results_df.to_csv(MODELS_PATH / "training_results.csv", index=False)
print(f"\nResultats sauvegardes : {MODELS_PATH / 'training_results.csv'}")


# %% 5. Conclusion methodologique
print("\n" + "=" * 70)
print("ANALYSE METHODOLOGIQUE")
print("=" * 70)
best_model_name = results_df.iloc[0]["Model"]
worst_model_name = results_df.iloc[-1]["Model"]
best_rmse = results_df.iloc[0]["CV_RMSE"]
baseline_rmse = results_df[results_df["Model"] == "Linear Regression"]["CV_RMSE"].values[0]
gain_pct = (baseline_rmse - best_rmse) / baseline_rmse * 100

print(f"""
+ Meilleur modele (CV)  : {best_model_name}  (RMSE = {best_rmse:.3f})
+ Modele le moins bon   : {worst_model_name}
+ Gain vs baseline lineaire : {gain_pct:+.2f}%

INTERPRETATION :
{'  -> Les modeles complexes apportent un gain reel.' if gain_pct > 5 else
 '  -> Faible gain des modeles complexes : la relation est essentiellement lineaire.' if gain_pct > 0 else
 '  -> La baseline lineaire est imbattable : signe que la relation est tres lineaire.'}

Cette analyse va guider le choix du modele final pour la production.
Le RMSE seul ne suffit pas : on va aussi considerer interpretabilite,
temps d'entrainement et facilite de deploiement (Etape 4).
""")

print("Etape 3 terminee. Prochaine etape : evaluation detaillee (notebooks/04_evaluation.py)")
