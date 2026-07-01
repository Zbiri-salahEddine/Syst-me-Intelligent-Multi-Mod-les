"""
06 - Eco-responsabilite : empreinte energetique et carbone de l'entrainement (C4.3)
====================================================================================

Lit models/training_results.csv (temps d'entrainement reels) et estime :
- Energie consommee (Wh)
- CO2 emis (mix France ADEME + moyenne mondiale AIE)
- Equivalent km en voiture thermique

Sortie :
- reports/ecoresponsibility.csv
- reports/figures/ecoresponsibility_co2.png (si matplotlib dispo)

Usage : python notebooks/06_ecoresponsibility.py
"""

# %%
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ecoresponsibility import (  # noqa: E402
    CARBON_INTENSITY_FRANCE,
    CARBON_INTENSITY_WORLD,
    CPU_TDP_WATTS,
    build_report,
)

MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"


# %% 1. Chargement des temps d'entrainement
training_results_path = MODELS_PATH / "training_results.csv"
if not training_results_path.exists():
    raise SystemExit(
        "training_results.csv introuvable. Lancez d'abord notebooks/03_modeling.py."
    )

training_results = pd.read_csv(training_results_path)
print("=" * 70)
print("ECO-RESPONSABILITE - EMPREINTE DE L'ENTRAINEMENT")
print("=" * 70)
print(f"Hypotheses : CPU {CPU_TDP_WATTS:.0f} W | "
      f"France {CARBON_INTENSITY_FRANCE:.0f} gCO2/kWh | "
      f"Monde {CARBON_INTENSITY_WORLD:.0f} gCO2/kWh\n")


# %% 2. Construction du rapport
report = build_report(training_results)
print(report.to_string(index=False))

REPORTS_PATH.mkdir(parents=True, exist_ok=True)
report_csv = REPORTS_PATH / "ecoresponsibility.csv"
report.to_csv(report_csv, index=False)
print(f"\nRapport ecrit : {report_csv}")


# %% 3. Lecture metier
total = report[report["Model"] == "TOTAL"].iloc[0]
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(f"""
Entrainement complet des 4 modeles (GridSearchCV) :
  - Energie totale   : {total['Energy_Wh']:.2f} Wh
  - CO2 (mix France) : {total['CO2_France_g']:.2f} gCO2eq
  - CO2 (mix Monde)  : {total['CO2_World_g']:.2f} gCO2eq
  - Equivalent       : ~{total['Car_km_eq']:.3f} km en voiture thermique

Insight : sur donnees tabulaires (~4500 lignes), l'empreinte d'entrainement est
negligeable. Le choix du modele final (XGBoost) ne degrade pas l'eco-responsabilite :
l'inference est quasi instantanee. Le levier carbone principal serait un re-entrainement
frequent ou un passage au deep learning lourd, a arbitrer selon le gain de performance.
""")


# %% 4. Figure (optionnelle)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = report[report["Model"] != "TOTAL"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_data["Model"], plot_data["CO2_France_g"], color="#2d8f6f")
    ax.set_xlabel("CO2 estime (gCO2eq, mix France)")
    ax.set_title("Empreinte carbone de l'entrainement par modele")
    ax.invert_yaxis()
    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_PATH / "ecoresponsibility_co2.png"
    fig.savefig(figure_path, dpi=120)
    print(f"Figure ecrite : {figure_path}")
except ImportError:
    print("matplotlib non disponible : figure ignoree (CSV produit quand meme).")
