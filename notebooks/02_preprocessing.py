"""
02 - Preprocessing : execution du pipeline
==========================================
Ce script orchestre le preprocessing :
1. Charge les donnees brutes
2. Nettoie la cible
3. Split train/test
4. Construit et FIT le pipeline UNIQUEMENT sur le train (anti data leakage)
5. Transforme train et test
6. Sauvegarde les datasets transformes + le preprocessor

Sortie :
- data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
- models/preprocessor.joblib
"""

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permet d'importer src/ depuis notebooks/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    clean_target,
    get_feature_names,
    load_raw_data,
    prepare_train_test,
    save_preprocessor,
)

# Chemins
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "marketing_and_sales.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# %% 1. Chargement
print("=" * 70)
print("ETAPE 1 / 5  -  Chargement des donnees brutes")
print("=" * 70)
df_raw = load_raw_data(DATA_RAW)
print(f"Donnees chargees : {df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes")


# %% 2. Nettoyage de la cible
print("\n" + "=" * 70)
print("ETAPE 2 / 5  -  Nettoyage de la cible (Sales)")
print("=" * 70)
df = clean_target(df_raw)
print(f"Donnees apres nettoyage : {df.shape[0]:,} lignes")


# %% 3. Split train/test
print("\n" + "=" * 70)
print("ETAPE 3 / 5  -  Split train/test (80/20, random_state=42)")
print("=" * 70)
X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=0.2, random_state=42)

# Verification de la distribution Sales (eviter un test pas representatif)
print(f"\nDistribution Sales :")
print(f"  Train : moyenne = {y_train.mean():.2f}  | std = {y_train.std():.2f}")
print(f"  Test  : moyenne = {y_test.mean():.2f}  | std = {y_test.std():.2f}")
diff_pct = abs(y_train.mean() - y_test.mean()) / y_train.mean() * 100
print(f"  Ecart moyenne train/test : {diff_pct:.2f}%  ({'OK' if diff_pct < 5 else 'ATTENTION'})")


# %% 4. Construction et FIT du preprocessor (UNIQUEMENT sur le train)
print("\n" + "=" * 70)
print("ETAPE 4 / 5  -  Fit du pipeline sur le TRAIN uniquement")
print("=" * 70)
print("Note : aucune transformation ne voit le test set a ce stade.")
print("       C'est la garantie anti data leakage.\n")

preprocessor = build_preprocessor()
preprocessor.fit(X_train)

# Aperçu de la structure du pipeline
print("Structure du pipeline :")
print(preprocessor)


# %% 5. Transformation train et test
print("\n" + "=" * 70)
print("ETAPE 5 / 5  -  Transformation train + test")
print("=" * 70)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = get_feature_names(preprocessor)
print(f"Features apres transformation : {feature_names}")
print(f"Shape train transforme : {X_train_transformed.shape}")
print(f"Shape test transforme  : {X_test_transformed.shape}")


# %% 6. Verification de l'absence de NaN apres transformation
print("\n" + "=" * 70)
print("CONTROLE QUALITE  -  Verification post-transformation")
print("=" * 70)
nan_train = np.isnan(X_train_transformed).sum()
nan_test = np.isnan(X_test_transformed).sum()
print(f"NaN restants train : {nan_train}  (doit etre 0)")
print(f"NaN restants test  : {nan_test}  (doit etre 0)")

# Verification scaling : numeric features doivent avoir mean ~0, std ~1 sur TRAIN
print(f"\nVerification du scaling sur le train (features numeriques) :")
for i, name in enumerate(feature_names[:3]):
    mean = X_train_transformed[:, i].mean()
    std = X_train_transformed[:, i].std()
    print(f"  {name:15s} : mean = {mean:+.4f}  | std = {std:+.4f}  (cible: 0 / 1)")


# %% 7. Sauvegarde
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)

# Sauvegarde des donnees brutes (avant transformation) pour permettre
# au modeling de refaire des CV avec le pipeline complet
X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
print(f"X_train, X_test, y_train, y_test sauvegardes dans : {DATA_PROCESSED}")

# Sauvegarde des donnees transformees (utile pour debug / dashboard)
np.save(DATA_PROCESSED / "X_train_transformed.npy", X_train_transformed)
np.save(DATA_PROCESSED / "X_test_transformed.npy", X_test_transformed)
print(f"Versions transformees (.npy) sauvegardees")

# Sauvegarde du preprocessor (CRUCIAL pour reutilisation en prod)
save_preprocessor(preprocessor, MODELS_PATH / "preprocessor.joblib")


# %% 8. Conclusions methodologiques
print("\n" + "=" * 70)
print("CONCLUSIONS METHODOLOGIQUES")
print("=" * 70)
print("""
+ Pipeline sklearn : preprocessing reutilisable et reproductible
+ Fit uniquement sur le train  -> aucun data leakage
+ Imputation mediane sur les numeriques -> robuste aux outliers
+ OneHotEncoder pour 'Influencer' avec ordre business explicite
+ Standardisation -> requise pour Regression Lineaire et MLP
+ handle_unknown='ignore' sur OneHot -> robuste a des categories
  inconnues envoyees par le dashboard ou l'API

POSTURE PROFESSIONNELLE :
Le meme pipeline qui a vu le train sera applique a tout nouveau scenario
en production (dashboard, API). Aucune divergence entre training et serving.
""")

print("Etape 2 terminee. Prochaine etape : modelisation (notebooks/03_modeling.py)")
