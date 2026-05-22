# Optimisation du Retour sur Investissement Marketing

**Projet Data Science M1 — EFREI | Sujet 3**
Auteur : Salah Eddine Zbiri

Systeme intelligent multi-modeles de prediction des ventes a partir des investissements marketing multicanal (TV, Radio, Social Media, Influencer). Le projet couvre l'ensemble du pipeline Data Science : EDA, preprocessing, modelisation, evaluation, interpretabilite, dashboard interactif et API REST.

---

## Resultats du modele en production

| Modele | RMSE test | R² test | MAPE test | Verdict |
|---|---|---|---|---|
| **XGBoost** *(production)* | **3.538** | **0.9985** | **1.84%** | OK |
| Random Forest | 3.701 | 0.9984 | 1.90% | OK |
| Linear Regression | 5.879 | 0.9960 | 1.87% | OK |
| MLP (Deep Learning) | 5.983 | 0.9958 | 2.26% | OK |

> XGBoost retenu : meilleur RMSE, zero overfitting, capture les non-linearites.

---

## Structure du projet

```
Projet Data Science/
├── api/
│   └── main.py              # API REST FastAPI (/predict, /health, /model-info)
├── dashboard/
│   └── app.py               # Dashboard Streamlit (5 pages interactives)
├── data/
│   ├── raw/
│   │   └── marketing_and_sales.csv
│   └── processed/           # Donnees train/test + versions transformees (.npy)
├── models/
│   ├── final_model.joblib   # XGBoost pipeline (preprocessor + modele)
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── mlp_deep_learning.joblib
│   ├── preprocessor.joblib
│   ├── training_results.csv # Scores CV des 4 modeles
│   └── training_results.csv
├── notebooks/
│   ├── 01_eda.py            # Analyse exploratoire (EDA)
│   ├── 02_preprocessing.py  # Pipeline preprocessing + split train/test
│   ├── 03_modeling.py       # Entrainement GridSearchCV des 4 modeles
│   ├── 04_evaluation.py     # Evaluation detaillee + selection modele final
│   └── 05_interpretability.py # Feature Importance + Permutation + SHAP
├── reports/
│   ├── evaluation_results.csv
│   ├── interpretability_results.csv
│   └── figures/             # 13 figures (EDA, evaluation, SHAP)
├── src/
│   ├── preprocessing.py     # Module preprocessing reutilisable
│   ├── models.py            # Definitions des 4 modeles + grilles hyperparams
│   └── evaluation.py        # Metriques + detection overfitting
└── requirements.txt
```

---

## Installation

```bash
# Creer un environnement virtuel (recommande)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Installer les dependances
pip install -r requirements.txt
```

---

## Execution du pipeline complet

Les scripts sont executes dans l'ordre depuis la racine du projet :

```bash
# 1. Analyse exploratoire
python notebooks/01_eda.py

# 2. Preprocessing + split train/test
python notebooks/02_preprocessing.py

# 3. Entrainement des 4 modeles (GridSearchCV)
python notebooks/03_modeling.py

# 4. Evaluation detaillee sur le test set
python notebooks/04_evaluation.py

# 5. Interpretabilite (Feature Importance + Permutation + SHAP)
python notebooks/05_interpretability.py
```

> Les notebooks sont des fichiers `.py` avec cellules `# %%`, executables dans VSCode (extension Python Interactive) ou Jupyter via jupytext.

---

## Lancement du dashboard

```bash
streamlit run dashboard/app.py
```

Ouvre automatiquement `http://localhost:8501`

**Pages disponibles :**
- **Accueil & KPI** — indicateurs strategiques + insights CMO
- **Exploration des donnees** — filtres interactifs, scatter plots, matrice de correlation
- **Simulation budgetaire** — prediction de ventes en temps reel + ROI estime
- **Comparaison des modeles** — tableau comparatif, bar charts, predictions vs realite
- **Explication SHAP** — importance globale des variables + waterfall local par scenario

---

## Lancement de l'API REST

```bash
uvicorn api.main:app --reload --port 8000
```

**Documentation interactive (Swagger) :** `http://localhost:8000/docs`

### Endpoints

#### `GET /health`
Verifie l'etat du service.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "XGBoost Regressor"
}
```

#### `POST /predict`
Predit les ventes pour un scenario budgetaire.

**Corps de la requete :**
```json
{
  "TV": 230.1,
  "Radio": 37.8,
  "Social Media": 69.2,
  "Influencer": "Macro"
}
```

**Reponse :**
```json
{
  "predicted_sales_M": 22.15,
  "roi_estimated": 0.066,
  "total_budget_M": 337.1,
  "model_used": "XGBoost Regressor",
  "input_received": {
    "TV": 230.1,
    "Radio": 37.8,
    "Social Media": 69.2,
    "Influencer": "Macro"
  }
}
```

**Valeurs valides pour `Influencer`** : `"Nano"`, `"Micro"`, `"Macro"`, `"Mega"`

#### `GET /model-info`
Retourne les performances et les caracteristiques du modele en production.

### Test rapide avec curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"TV\": 230.1, \"Radio\": 37.8, \"Social Media\": 69.2, \"Influencer\": \"Macro\"}"
```

---

## Dataset

Source : [Kaggle — Dummy Advertising and Sales Data](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

| Variable | Type | Description |
|---|---|---|
| TV | float | Budget television (millions $) |
| Radio | float | Budget radio (millions $) |
| Social Media | float | Budget social media (millions $) |
| Influencer | string | Type : Nano / Micro / Macro / Mega |
| **Sales** | float | **Variable cible** — ventes (millions $) |

~4 500 campagnes, relations realistes entre budgets et ventes.

---

## Methodologie

### Anti data leakage
Le preprocessor (imputation + scaling + encoding) est **fit uniquement sur le train set** et applique en transform sur le test set. Les pipelines sklearn garantissent ce principe meme pendant la cross-validation (le preprocessing est refit a chaque fold).

### Modeles et justifications
- **Linear Regression** — baseline interpretable, reference pour mesurer le gain des modeles complexes
- **Random Forest** — capture les non-linearites, robuste aux outliers, peu sensible au scaling
- **XGBoost** — etat de l'art sur donnees tabulaires, meilleur compromis performance/stabilite
- **MLP (Deep Learning)** — reseau de neurones multicouches avec early stopping et regularisation L2

### Interpretabilite (3 methodes)
1. **Feature Importance native (XGBoost)** — importance basee sur le gain de chaque split
2. **Permutation Importance** — methode agnostique au modele, mesuree sur le test set
3. **SHAP (TreeExplainer)** — explicabilite locale (waterfall) et globale (beeswarm)

> Les 3 methodes convergent : **TV est le levier dominant** (>90% de l'importance), l'Influencer a un impact negligeable (<0.13%).

---

## Competences RNCP40875 — Bloc 2 couvertes

- C3.1 — Preparation et transformation des donnees (pipeline sklearn, anti data leakage)
- C3.2 — Communication infographique (dashboard Streamlit 5 pages, graphiques interactifs)
- C3.3 — Analyse exploratoire (EDA complete avec insights metier)
- C4.1 — Strategie d'integration de l'IA (API REST + dashboard orienté CMO)
- C4.2 — Developpement de modeles predictifs (4 modeles ML/DL, GridSearchCV)
- C4.3 — Evaluation et comparaison des modeles (metriques adaptees, overfitting detection)
