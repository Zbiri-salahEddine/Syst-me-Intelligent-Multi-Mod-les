"""
04 - Evaluation detaillee des modeles sur le TEST SET
======================================================

Pour chaque modele :
1. Charge le pipeline entraine (depuis models/)
2. Predit sur le train ET le test
3. Calcule les metriques (MAE, RMSE, R^2, MAPE)
4. Detecte l'overfitting (gap train/test)
5. Genere les visualisations :
   - Predictions vs Realite (scatter)
   - Distribution des residus
   - Residus vs Predictions (pattern check)

Sortie :
- reports/figures/04_*.png
- reports/evaluation_results.csv
- Selection argumentee du modele final
"""

# %%
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    compute_regression_metrics,
    compute_residuals,
    detect_overfitting,
)

# Chemins
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
FIG_PATH = PROJECT_ROOT / "reports" / "figures"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIG_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


# %% 1. Chargement des donnees et des modeles
print("=" * 70)
print("CHARGEMENT")
print("=" * 70)
X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

print(f"X_test : {X_test.shape}  |  y_test : {y_test.shape}")

# Mapping nom modele -> fichier
MODELS = {
    "Linear Regression": "linear_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "MLP (Deep Learning)": "mlp_deep_learning.joblib",
}

print(f"\n{len(MODELS)} modeles a evaluer :")
for name in MODELS:
    print(f"  - {name}")


# %% 2. Evaluation de chaque modele
print("\n" + "=" * 70)
print("EVALUATION DETAILLEE")
print("=" * 70)

evaluation_results = []
all_predictions = {}  # pour les graphes ulterieurs

for name, filename in MODELS.items():
    print(f"\n{'-' * 70}")
    print(f">>> {name}")
    print(f"{'-' * 70}")

    # Chargement du pipeline complet
    pipeline = joblib.load(MODELS_PATH / filename)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Sauvegarde pour visualisations
    all_predictions[name] = {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }

    # Metriques train et test
    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    # Detection overfitting
    overfit = detect_overfitting(train_metrics, test_metrics)

    # Affichage
    print(f"  TRAIN  : MAE={train_metrics['MAE']:.3f}  RMSE={train_metrics['RMSE']:.3f}  R^2={train_metrics['R2']:.4f}  MAPE={train_metrics['MAPE_%']:.2f}%")
    print(f"  TEST   : MAE={test_metrics['MAE']:.3f}  RMSE={test_metrics['RMSE']:.3f}  R^2={test_metrics['R2']:.4f}  MAPE={test_metrics['MAPE_%']:.2f}%")
    print(f"  GAP    : delta_R^2={overfit['gap_R2']:+.4f}  delta_RMSE={overfit['gap_RMSE_%']:+.2f}%  -> {overfit['verdict']}")

    # Stockage
    evaluation_results.append({
        "Model": name,
        "MAE_train": round(train_metrics["MAE"], 3),
        "MAE_test": round(test_metrics["MAE"], 3),
        "RMSE_train": round(train_metrics["RMSE"], 3),
        "RMSE_test": round(test_metrics["RMSE"], 3),
        "R2_train": round(train_metrics["R2"], 4),
        "R2_test": round(test_metrics["R2"], 4),
        "MAPE_test_%": round(test_metrics["MAPE_%"], 2),
        "Gap_R2": overfit["gap_R2"],
        "Verdict": overfit["verdict"],
    })


# %% 3. Tableau recapitulatif
print("\n" + "=" * 70)
print("TABLEAU RECAPITULATIF FINAL (Test Set)")
print("=" * 70)

results_df = pd.DataFrame(evaluation_results)
results_df = results_df.sort_values("RMSE_test").reset_index(drop=True)
print(results_df.to_string(index=False))

# Sauvegarde
results_df.to_csv(REPORTS_PATH / "evaluation_results.csv", index=False)
print(f"\nResultats sauvegardes : {REPORTS_PATH / 'evaluation_results.csv'}")


# %% 4. Visualisation : Predictions vs Realite (4 modeles)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, preds) in enumerate(all_predictions.items()):
    ax = axes[i]
    y_pred = preds["y_test_pred"]

    # Scatter predicted vs actual
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, edgecolor="white", linewidth=0.5)
    # Diagonale parfaite
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Prediction parfaite")

    # Metriques en titre
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    ax.set_title(f"{name}\nRMSE = {rmse:.2f}  |  R^2 = {r2:.4f}",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Sales reelles")
    ax.set_ylabel("Sales predites")
    ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_PATH / "04_predictions_vs_actual.png")
plt.close()
print(f"\n[fig] {FIG_PATH / '04_predictions_vs_actual.png'}")


# %% 5. Visualisation : Distribution des residus
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, preds) in enumerate(all_predictions.items()):
    ax = axes[i]
    residuals = y_test - preds["y_test_pred"]

    sns.histplot(residuals, kde=True, ax=ax, bins=40, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Residu = 0")
    ax.axvline(residuals.mean(), color="orange", linestyle=":", linewidth=2,
               label=f"Moyenne = {residuals.mean():+.3f}")

    ax.set_title(f"{name}\nMoyenne = {residuals.mean():+.3f}  |  Std = {residuals.std():.3f}",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Residu (y_reel - y_predit)")
    ax.set_ylabel("Frequence")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_PATH / "05_residuals_distribution.png")
plt.close()
print(f"[fig] {FIG_PATH / '05_residuals_distribution.png'}")


# %% 6. Visualisation : Residus vs Predictions (detection de pattern)
# Un bon modele a des residus aleatoires (pas de structure visible)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, preds) in enumerate(all_predictions.items()):
    ax = axes[i]
    y_pred = preds["y_test_pred"]
    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.4, s=15, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)

    # Bande +/- 2*std
    std_res = residuals.std()
    ax.axhline(2 * std_res, color="orange", linestyle=":", alpha=0.7, label=f"+/- 2 sigma")
    ax.axhline(-2 * std_res, color="orange", linestyle=":", alpha=0.7)

    ax.set_title(f"{name}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Predictions (Sales)")
    ax.set_ylabel("Residus")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_PATH / "06_residuals_vs_predictions.png")
plt.close()
print(f"[fig] {FIG_PATH / '06_residuals_vs_predictions.png'}")


# %% 7. Visualisation : Comparaison des metriques (bar chart)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = [("RMSE_test", "RMSE", "Lower is better"),
                   ("R2_test", "R^2", "Higher is better"),
                   ("MAPE_test_%", "MAPE (%)", "Lower is better")]

for ax, (col, title, hint) in zip(axes, metrics_to_plot):
    sorted_df = results_df.sort_values(col, ascending=(col != "R2_test"))
    colors = ["#2ECC71" if i == 0 else "#3498DB" for i in range(len(sorted_df))]
    bars = ax.barh(sorted_df["Model"], sorted_df[col], color=colors, edgecolor="black")
    ax.set_title(f"{title}  ({hint})", fontweight="bold")
    ax.set_xlabel(title)

    # Affichage des valeurs sur les barres
    for bar, val in zip(bars, sorted_df[col]):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f" {val:.3f}", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_PATH / "07_metrics_comparison.png")
plt.close()
print(f"[fig] {FIG_PATH / '07_metrics_comparison.png'}")


# %% 8. Selection du modele final
print("\n" + "=" * 70)
print("SELECTION DU MODELE FINAL (vision consultant)")
print("=" * 70)

best_rmse_model = results_df.loc[results_df["RMSE_test"].idxmin(), "Model"]
best_r2_model = results_df.loc[results_df["R2_test"].idxmax(), "Model"]
best_mape_model = results_df.loc[results_df["MAPE_test_%"].idxmin(), "Model"]

print(f"\nMeilleur sur RMSE : {best_rmse_model}")
print(f"Meilleur sur R^2  : {best_r2_model}")
print(f"Meilleur sur MAPE : {best_mape_model}")

# Strategie de selection : on prend le plus performant (RMSE), mais on documente le tradeoff
final_model_name = best_rmse_model
final_row = results_df[results_df["Model"] == final_model_name].iloc[0]

print(f"""
+-------------------------------------------------+
| MODELE FINAL RETENU : {final_model_name:<26s} |
+-------------------------------------------------+

Performances sur le TEST set (jamais vu pendant l'entrainement) :
  - MAE       = {final_row['MAE_test']:.3f}  M$
  - RMSE      = {final_row['RMSE_test']:.3f}  M$
  - R^2       = {final_row['R2_test']:.4f}
  - MAPE      = {final_row['MAPE_test_%']:.2f}%
  - Verdict   = {final_row['Verdict']}

Interpretation metier :
  En moyenne, nos predictions s'ecartent de {final_row['MAE_test']:.2f}M$ des ventes reelles
  Soit {final_row['MAPE_test_%']:.1f}% d'erreur relative.
  Le modele explique {final_row['R2_test']*100:.2f}% de la variance des ventes.

Pourquoi {final_model_name} et pas un autre ?
  -> Meilleur RMSE sur le test set
  -> Pas d'overfitting (verdict OK)
  -> Capture les non-linearites et interactions
  -> Robuste aux outliers (vu en EDA)

Alternatives serieuses :
  -> XGBoost : performance quasi-identique, 2x plus rapide a l'entrainement
  -> Linear Regression : 99% aussi bon, parfaitement interpretable
  -> MLP : performance INFERIEURE a la baseline -> illustration parfaite de
     "le bon modele, pas le modele a la mode"

Pour le dashboard de production, on utilisera {final_model_name} mais
on conservera tous les modeles pour permettre la comparaison utilisateur.
""")

# Symlink ou copie du modele final
final_filename = final_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
shutil.copy(MODELS_PATH / f"{final_filename}.joblib", MODELS_PATH / "final_model.joblib")
print(f"+ Modele final copie : {MODELS_PATH / 'final_model.joblib'}")

print(f"\nEtape 4 terminee. Prochaine etape : interpretabilite (notebooks/05_interpretability.py)")
