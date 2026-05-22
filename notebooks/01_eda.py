"""
01 - Analyse Exploratoire des Données (EDA)
Projet : Optimisation du ROI Marketing
Auteur : Salah Eddine Zbiri
Objectif : Comprendre la structure du dataset, identifier les patterns metier
           et detecter les problemes (NaN, outliers, multicolinearite) avant
           toute modelisation.

Ce fichier est structure en cellules (# %%) executables dans VSCode
(extension Python interactive) ou convertible en notebook avec jupytext.
"""

# %% [markdown]
# # EDA - Marketing & Sales Dataset
#
# **Posture metier** : nous sommes en mission pour un CMO qui veut comprendre
# quels canaux (TV, Radio, Social Media, Influencer) genrent reellement les
# ventes, afin d'optimiser l'allocation budgetaire future.

# %% Imports et configuration
import matplotlib
# IMPORTANT : backend non-interactif force pour eviter les fenetres bloquantes
# en mode script. Si tu veux les figures interactives, lance les cellules
# une par une dans Jupyter/VSCode avec '# %%' (et commente la ligne ci-dessous).
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

def _show():
    """Ferme la figure (mode script) ou l'affiche (mode interactif)."""
    if matplotlib.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "marketing_and_sales.csv"
FIG_PATH = PROJECT_ROOT / "reports" / "figures"
FIG_PATH.mkdir(parents=True, exist_ok=True)

print(f"Dataset path : {DATA_PATH}")
print(f"Figures path : {FIG_PATH}")


# %% 1. Chargement et apercu general
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("APERCU GENERAL")
print("=" * 70)
print(f"Shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nPremieres lignes :")
print(df.head())

# %% 2. Types et memoire
print("=" * 70)
print("TYPES DE DONNEES & MEMOIRE")
print("=" * 70)
df.info()

# %% 3. Statistiques descriptives
print("=" * 70)
print("STATISTIQUES DESCRIPTIVES (variables numeriques)")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("STATISTIQUES (variables categorielles)")
print("=" * 70)
print(df.describe(include="object"))

# %% 4. Detection des valeurs manquantes
print("=" * 70)
print("VALEURS MANQUANTES")
print("=" * 70)
nan_count = df.isnull().sum()
nan_pct = (df.isnull().mean() * 100).round(2)
nan_report = pd.DataFrame({"NaN_count": nan_count, "NaN_%": nan_pct})
print(nan_report)

print(f"\nNombre total de lignes avec au moins 1 NaN : {df.isnull().any(axis=1).sum()}")
print(f"Soit {df.isnull().any(axis=1).mean()*100:.2f}% du dataset")

# Visualisation des NaN : barplot suffit pour ce dataset (~200 lignes)
fig, ax = plt.subplots(figsize=(10, 5))
nan_count.plot(kind="bar", ax=ax, color="#C73E1D", edgecolor="black")
ax.set_title("Nombre de valeurs manquantes par colonne", fontweight="bold")
ax.set_xlabel("Colonne")
ax.set_ylabel("Nombre de NaN")
for i, v in enumerate(nan_count.values):
    ax.text(i, v + 0.2, str(int(v)), ha="center", fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_PATH / "01_missing_values.png")
_show()

# %% 5. Detection des doublons
print("=" * 70)
print("DOUBLONS")
print("=" * 70)
duplicates = df.duplicated().sum()
print(f"Nombre de lignes dupliquees : {duplicates}")

# %% 6. Distribution de la variable cible : Sales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme + KDE
sns.histplot(df["Sales"], kde=True, bins=40, ax=axes[0], color="#2E86AB")
axes[0].set_title("Distribution de Sales (variable cible)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Sales (en millions)")
axes[0].axvline(df["Sales"].mean(), color="red", linestyle="--", label=f"Moyenne = {df['Sales'].mean():.1f}")
axes[0].axvline(df["Sales"].median(), color="green", linestyle="--", label=f"Mediane = {df['Sales'].median():.1f}")
axes[0].legend()

# Boxplot
sns.boxplot(y=df["Sales"], ax=axes[1], color="#2E86AB")
axes[1].set_title("Boxplot de Sales (detection outliers)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Sales (en millions)")

plt.tight_layout()
plt.savefig(FIG_PATH / "02_sales_distribution.png")
_show()

print(f"\nSkewness Sales : {df['Sales'].skew():.3f}  (0 = symetrique)")
print(f"Kurtosis Sales : {df['Sales'].kurtosis():.3f}  (0 = normale)")

# %% 7. Distribution des features numeriques
numeric_cols = ["TV", "Radio", "Social Media"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors = ["#A23B72", "#F18F01", "#C73E1D"]

for i, col in enumerate(numeric_cols):
    # Ligne 1 : histogrammes
    sns.histplot(df[col].dropna(), kde=True, bins=40, ax=axes[0, i], color=colors[i])
    axes[0, i].set_title(f"Distribution : {col}", fontweight="bold")
    axes[0, i].set_xlabel(f"{col} (millions)")

    # Ligne 2 : boxplots
    sns.boxplot(y=df[col].dropna(), ax=axes[1, i], color=colors[i])
    axes[1, i].set_title(f"Boxplot : {col}", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_PATH / "03_features_distributions.png")
_show()

# Skewness des features
print("\nSkewness des features :")
for col in numeric_cols:
    print(f"  {col:15s} : {df[col].skew():.3f}")

# %% 8. Distribution de la variable categorielle Influencer
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comptage
counts = df["Influencer"].value_counts()
sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette="viridis")
axes[0].set_title("Repartition des types d'Influencer", fontweight="bold")
axes[0].set_xlabel("Type d'Influencer")
axes[0].set_ylabel("Nombre de campagnes")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

# Pourcentage
pct = (counts / counts.sum() * 100).round(1)
axes[1].pie(pct, labels=pct.index, autopct="%1.1f%%", colors=sns.color_palette("viridis", 4), startangle=90)
axes[1].set_title("Repartition en %", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_PATH / "04_influencer_distribution.png")
_show()

print(f"\nClasses equilibrees : {pct.min():.1f}% min vs {pct.max():.1f}% max")

# %% 9. Relations bivariees : chaque budget vs Sales
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(numeric_cols):
    sns.scatterplot(
        data=df, x=col, y="Sales",
        hue="Influencer", palette="viridis",
        alpha=0.6, ax=axes[i], s=30
    )
    # Ligne de regression globale
    sns.regplot(
        data=df, x=col, y="Sales",
        scatter=False, color="red", line_kws={"linewidth": 2, "linestyle": "--"},
        ax=axes[i]
    )
    corr_val = df[[col, "Sales"]].corr().iloc[0, 1]
    axes[i].set_title(f"{col} vs Sales  (correlation = {corr_val:.3f})", fontweight="bold")
    axes[i].set_xlabel(f"{col} (millions)")
    axes[i].set_ylabel("Sales (millions)")

plt.tight_layout()
plt.savefig(FIG_PATH / "05_budgets_vs_sales.png")
_show()

# %% 10. Impact de l'Influencer sur les Sales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
sns.boxplot(data=df, x="Influencer", y="Sales", ax=axes[0], palette="viridis",
            order=["Nano", "Micro", "Macro", "Mega"])
axes[0].set_title("Distribution Sales par type d'Influencer", fontweight="bold")
axes[0].set_xlabel("Type d'Influencer")
axes[0].set_ylabel("Sales (millions)")

# Barplot moyennes
sns.barplot(data=df, x="Influencer", y="Sales", ax=axes[1], palette="viridis",
            order=["Nano", "Micro", "Macro", "Mega"], errorbar="ci")
axes[1].set_title("Sales moyenne par type d'Influencer (IC 95%)", fontweight="bold")
axes[1].set_xlabel("Type d'Influencer")
axes[1].set_ylabel("Sales moyenne (millions)")

plt.tight_layout()
plt.savefig(FIG_PATH / "06_influencer_vs_sales.png")
_show()

# Statistiques par groupe
print("\nStatistiques Sales par Influencer :")
print(df.groupby("Influencer")["Sales"].agg(["mean", "median", "std", "count"]).round(2))

# %% 11. Matrice de correlation
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[numeric_cols + ["Sales"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
    ax=ax, mask=mask, annot_kws={"fontweight": "bold"}
)
ax.set_title("Matrice de correlation (Pearson)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_PATH / "07_correlation_matrix.png")
_show()

print("\nCorrelations avec Sales (triees) :")
sales_corr = corr_matrix["Sales"].drop("Sales").sort_values(ascending=False)
print(sales_corr)

# %% 12. Pairplot multivarie (echantillonnage pour la perf)
sample = df.dropna().sample(n=min(1000, len(df.dropna())), random_state=42)
g = sns.pairplot(
    sample,
    hue="Influencer",
    palette="viridis",
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.5, "s": 20}
)
g.fig.suptitle(f"Pairplot - Vue multivariee (echantillon n={len(sample)})", y=1.01, fontweight="bold", fontsize=14)
g.savefig(FIG_PATH / "08_pairplot.png")
_show()

# %% 13. Detection des outliers (methode IQR)
print("=" * 70)
print("DETECTION D'OUTLIERS (methode IQR)")
print("=" * 70)

def detect_outliers_iqr(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper

for col in numeric_cols + ["Sales"]:
    n_out, low, high = detect_outliers_iqr(df[col].dropna())
    pct = n_out / df[col].dropna().shape[0] * 100
    print(f"  {col:15s} : {n_out:4d} outliers ({pct:5.2f}%)  | bornes [{low:.2f} ; {high:.2f}]")

# %% 14. Insights metier synthetiques
print("\n" + "=" * 70)
print("INSIGHTS METIER POUR LE CMO")
print("=" * 70)

best_feature = sales_corr.abs().idxmax()
print(f"\n1. Le canal le plus correle aux Sales : **{best_feature}** ({sales_corr[best_feature]:.3f})")

print(f"\n2. Hierarchie d'impact (correlation absolue) :")
for feat, val in sales_corr.abs().sort_values(ascending=False).items():
    print(f"   - {feat:15s} : {val:.3f}")

best_inf = df.groupby("Influencer")["Sales"].mean().idxmax()
worst_inf = df.groupby("Influencer")["Sales"].mean().idxmin()
diff_pct = (df.groupby("Influencer")["Sales"].mean().max() /
            df.groupby("Influencer")["Sales"].mean().min() - 1) * 100
print(f"\n3. Influencer le plus performant en moyenne : **{best_inf}**")
print(f"   Influencer le moins performant : **{worst_inf}**")
print(f"   Ecart relatif : {diff_pct:.1f}%")

print(f"\n4. Qualite des donnees :")
print(f"   - Volume : {df.shape[0]:,} campagnes (suffisant pour ML/DL)")
print(f"   - NaN a traiter : {df.isnull().any(axis=1).sum()} lignes ({df.isnull().any(axis=1).mean()*100:.1f}%)")
print(f"   - Doublons : {duplicates}")

# Verification de la multicolinearite
high_corr_pairs = []
for i, c1 in enumerate(numeric_cols):
    for c2 in numeric_cols[i+1:]:
        c = abs(corr_matrix.loc[c1, c2])
        if c > 0.7:
            high_corr_pairs.append((c1, c2, c))

if high_corr_pairs:
    print(f"\n5. ALERTE multicolinearite (|r| > 0.7) :")
    for c1, c2, c in high_corr_pairs:
        print(f"   {c1} - {c2} : {c:.3f}")
else:
    print(f"\n5. Pas de multicolinearite detectee entre features (|r| < 0.7) : OK")

# %% 15. Conclusions pour le preprocessing
print("\n" + "=" * 70)
print("DECISIONS PREPROCESSING (Etape suivante)")
print("=" * 70)
print("""
A. NaN sur TV  -> Imputation par la mediane (robuste aux outliers)
   Justification : moins de 5% de NaN, distribution legerement skewed.

B. Encoding 'Influencer' -> OneHotEncoder (4 categories, pas d'ordre intrinseque)
   Note : on pourrait tester un OrdinalEncoder (Nano < Micro < Macro < Mega)
   si l'ordre de portee est business-sense.

C. Scaling -> StandardScaler pour les modeles sensibles a l'echelle
   (Regression Lineaire, MLP). Pas necessaire pour les arbres (RF, XGBoost).

D. Train/Test split -> 80/20 random (regression, pas de stratification).
   random_state=42 pour reproductibilite.

E. Pipeline sklearn -> ColumnTransformer + Pipeline pour eviter le data leakage.
   Le scaler et l'imputer seront FIT uniquement sur le train.
""")

# %% Fin
print("EDA terminee. Voir les figures dans :", FIG_PATH)
