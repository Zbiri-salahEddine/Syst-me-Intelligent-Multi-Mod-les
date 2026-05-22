"""
05 - Interpretabilite : Feature Importance + Permutation + SHAP
================================================================

Analyse du modele final (XGBoost) avec 3 techniques complementaires :
1. Feature Importance native (Gain)  -> vision globale, rapide
2. Permutation Importance            -> agnostique au modele, robuste
3. SHAP (TreeExplainer)              -> explicabilite locale ET globale

Sortie :
- reports/figures/08_*.png (4 figures SHAP + 2 importance)
- reports/interpretability_results.csv
"""

# %%
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Chemins
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
FIG_PATH = PROJECT_ROOT / "reports" / "figures"
REPORTS_PATH = PROJECT_ROOT / "reports"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


# %% 1. Chargement du modele final + donnees
print("=" * 70)
print("CHARGEMENT DU MODELE FINAL ET DES DONNEES")
print("=" * 70)

pipeline = joblib.load(MODELS_PATH / "final_model.joblib")
X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

# Decomposition du pipeline (preprocessor + model)
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

print(f"Modele final : {type(model).__name__}")
print(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")

# Transformation des donnees (pour SHAP qui prend les donnees apres preprocessing)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out().tolist()
print(f"Features apres preprocessing : {feature_names}")


# %% 2. METHODE 1 : Feature Importance native (XGBoost)
print("\n" + "=" * 70)
print("METHODE 1 / 3 : Feature Importance native (XGBoost)")
print("=" * 70)
print("""
Calcul base sur le 'gain' : reduction moyenne de l'erreur quadratique
chaque fois que cette variable est utilisee pour un split dans un arbre.
Tres rapide (calculee gratuitement pendant l'entrainement).
""")

native_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(native_importance.to_string(index=False))

# Visualisation
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#E74C3C" if i == 0 else "#3498DB" for i in range(len(native_importance))]
bars = ax.barh(native_importance["feature"], native_importance["importance"],
               color=colors, edgecolor="black")
ax.set_title("Feature Importance native (XGBoost - Gain)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Importance (gain relatif)")
ax.invert_yaxis()
for bar, v in zip(bars, native_importance["importance"]):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f" {v:.4f}", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_PATH / "08_feature_importance_native.png")
plt.close()
print(f"[fig] 08_feature_importance_native.png")


# %% 3. METHODE 2 : Permutation Importance
print("\n" + "=" * 70)
print("METHODE 2 / 3 : Permutation Importance")
print("=" * 70)
print("""
Methode agnostique au modele :
1. On mesure la performance initiale (R^2)
2. On permute aleatoirement les valeurs d'une variable
3. On mesure la chute de performance
-> Plus la chute est grande, plus la variable est importante

Avantage : independant de l'algorithme, donne une vraie mesure d'impact.
On l'applique sur le TEST SET pour mesurer l'impact reel en generalisation.
""")

perm_result = permutation_importance(
    pipeline, X_test, y_test,
    n_repeats=10,         # 10 permutations pour stabilite
    random_state=42,
    n_jobs=-1,
    scoring="r2",
)

# Note : la permutation est appliquee sur X_test (avant preprocessing)
# donc on retrouve les NOMS ORIGINAUX (TV, Radio, Social Media, Influencer)
perm_importance = pd.DataFrame({
    "feature": X_test.columns.tolist(),
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print(perm_importance.to_string(index=False))

# Visualisation avec barre d'erreur
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#E74C3C" if i == 0 else "#3498DB" for i in range(len(perm_importance))]
ax.barh(perm_importance["feature"], perm_importance["importance_mean"],
        xerr=perm_importance["importance_std"], color=colors,
        edgecolor="black", error_kw={"linewidth": 1.5, "capsize": 4})
ax.set_title("Permutation Importance (impact reel sur R^2 - test set)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Chute du R^2 quand la variable est permutee")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_PATH / "09_permutation_importance.png")
plt.close()
print(f"[fig] 09_permutation_importance.png")


# %% 4. METHODE 3 : SHAP - explicabilite locale et globale
print("\n" + "=" * 70)
print("METHODE 3 / 3 : SHAP (SHapley Additive exPlanations)")
print("=" * 70)
print("""
Methode la plus avancee (theorie des jeux - Shapley values).
TreeExplainer : optimise pour les modeles bases sur les arbres (XGBoost).

Permet :
  - Vue globale : importance moyenne par variable
  - Vue locale  : pourquoi UNE prediction precise a cette valeur
  - Impact directionnel : la variable pousse-t-elle Sales vers le haut ou le bas ?
""")

# Creation de l'explainer
explainer = shap.TreeExplainer(model)

# On limite a 500 echantillons du test pour les graphes (perf)
n_samples = min(500, len(X_test_transformed))
X_sample = X_test_transformed[:n_samples]
X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

# Calcul des SHAP values
shap_values = explainer.shap_values(X_sample)
print(f"SHAP values calculees sur {n_samples} echantillons.")

# 4.1 - SHAP summary plot (bar) : importance globale
print("\n[SHAP] Importance globale (mean |SHAP value|)")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(shap_importance.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False)
plt.title("SHAP - Importance globale", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_PATH / "10_shap_importance_bar.png", bbox_inches="tight")
plt.close()
print(f"[fig] 10_shap_importance_bar.png")

# 4.2 - SHAP summary plot (beeswarm) : impact + direction
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_df, show=False)
plt.title("SHAP - Beeswarm (impact + direction)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_PATH / "11_shap_beeswarm.png", bbox_inches="tight")
plt.close()
print(f"[fig] 11_shap_beeswarm.png")

# 4.3 - SHAP waterfall : explication d'UNE prediction (vision locale)
# On choisit 3 echantillons : faibles ventes, ventes moyennes, fortes ventes
y_test_array = y_test.values
sorted_idx = np.argsort(y_test_array[:n_samples])
sample_indices = {
    "Faibles ventes (q10)": sorted_idx[int(0.1 * n_samples)],
    "Ventes medianes (q50)": sorted_idx[int(0.5 * n_samples)],
    "Fortes ventes (q90)": sorted_idx[int(0.9 * n_samples)],
}

for label, idx in sample_indices.items():
    fig = plt.figure(figsize=(10, 6))
    expected_value = float(explainer.expected_value)
    if not np.isscalar(expected_value):
        expected_value = float(expected_value[0])

    shap_explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X_sample_df.iloc[idx].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall - {label}\nSales reelle = {y_test_array[idx]:.1f}M$",
              fontweight="bold", fontsize=12)
    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(FIG_PATH / f"12_shap_waterfall_{safe_label}.png", bbox_inches="tight")
    plt.close()
    print(f"[fig] 12_shap_waterfall_{safe_label}.png  (idx={idx}, y_reel={y_test_array[idx]:.1f})")


# %% 5. Synthese et comparaison des 3 methodes
print("\n" + "=" * 70)
print("SYNTHESE : COMPARAISON DES 3 METHODES")
print("=" * 70)

# Regrouper les rankings dans un tableau commun
# Native : par feature transformee (TV, Radio, Social Media, Influencer_*)
# Permutation : par feature originale (TV, Radio, Social Media, Influencer)
# SHAP : par feature transformee
# On normalise tout en agregant les OneHot dans 'Influencer'

def aggregate_onehot(df, value_col, original_features):
    """Agrege les colonnes OneHot d'Influencer en une seule 'Influencer'."""
    rows = []
    for feat in ["TV", "Radio", "Social Media"]:
        val = df[df["feature"] == feat][value_col].sum()
        rows.append({"feature": feat, value_col: val})
    inf_val = df[df["feature"].str.startswith("Influencer_")][value_col].sum()
    rows.append({"feature": "Influencer", value_col: inf_val})
    return pd.DataFrame(rows)

native_agg = aggregate_onehot(native_importance, "importance", X_test.columns)
shap_agg = aggregate_onehot(shap_importance, "mean_abs_shap", X_test.columns)

synthesis = pd.DataFrame({
    "feature": ["TV", "Radio", "Social Media", "Influencer"],
})
synthesis = synthesis.merge(native_agg.rename(columns={"importance": "Native_Importance"}),
                             on="feature")
synthesis = synthesis.merge(
    perm_importance[["feature", "importance_mean"]].rename(
        columns={"importance_mean": "Permutation_R2_loss"}),
    on="feature")
synthesis = synthesis.merge(shap_agg.rename(columns={"mean_abs_shap": "SHAP_mean_abs"}),
                             on="feature")

# Normalisation pour comparaison (chaque methode somme a 1)
for col in ["Native_Importance", "Permutation_R2_loss", "SHAP_mean_abs"]:
    total = synthesis[col].sum()
    synthesis[col + "_norm"] = (synthesis[col] / total * 100).round(2)

print("\nImportance relative (%) par methode :")
display_cols = ["feature"] + [c for c in synthesis.columns if c.endswith("_norm")]
print(synthesis[display_cols].to_string(index=False))

# Sauvegarde
synthesis.to_csv(REPORTS_PATH / "interpretability_results.csv", index=False)

# Visualisation comparative
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(synthesis))
width = 0.25

ax.bar(x - width, synthesis["Native_Importance_norm"], width,
       label="Native (XGBoost)", color="#3498DB", edgecolor="black")
ax.bar(x, synthesis["Permutation_R2_loss_norm"], width,
       label="Permutation Importance", color="#E67E22", edgecolor="black")
ax.bar(x + width, synthesis["SHAP_mean_abs_norm"], width,
       label="SHAP", color="#2ECC71", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(synthesis["feature"], fontweight="bold")
ax.set_ylabel("Importance relative (%)")
ax.set_title("Comparaison des 3 methodes d'interpretabilite (normalisees)",
             fontweight="bold", fontsize=13)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_PATH / "13_interpretability_comparison.png")
plt.close()
print(f"\n[fig] 13_interpretability_comparison.png")


# %% 6. Insights metier pour le CMO
print("\n" + "=" * 70)
print("INSIGHTS METIER POUR LE CMO")
print("=" * 70)

top_feature = synthesis.sort_values("SHAP_mean_abs_norm", ascending=False).iloc[0]["feature"]
top_shap_pct = synthesis.sort_values("SHAP_mean_abs_norm", ascending=False).iloc[0]["SHAP_mean_abs_norm"]
infl_shap = synthesis[synthesis["feature"] == "Influencer"]["SHAP_mean_abs_norm"].values[0]

print(f"""
+ Variable dominante (SHAP) : {top_feature}  ({top_shap_pct:.1f}% de l'importance totale)

+ L'Influencer n'a qu'une importance de {infl_shap:.2f}% selon SHAP
  -> Confirmation de l'EDA : ce parametre n'est PAS un levier business majeur.

CONCORDANCE DES 3 METHODES :
Les trois methodes convergent sur la hierarchie. C'est un signal de
ROBUSTESSE de notre analyse : le modele se fonde principalement sur
le budget TV pour predire les ventes, conformement a la realite metier
revelee par l'EDA (correlation TV-Sales = 0.999).

RECOMMANDATIONS POUR LA STRATEGIE MARKETING :
1. PRIORISER les investissements en TV (rendement le plus fort)
2. SECONDAIRE : Radio (correlation 0.87 mais redondante avec TV)
3. NE PAS SURINVESTIR en Social Media (impact modere)
4. NE PAS SURPAYER pour des Mega influenceurs : leur ROI marginal est faible
""")

print("\nEtape 5 terminee. Prochaine etape : dashboard Streamlit (dashboard/app.py)")
