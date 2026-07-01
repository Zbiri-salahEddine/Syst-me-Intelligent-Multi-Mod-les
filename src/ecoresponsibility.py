"""
src/ecoresponsibility.py
------------------------
Estimation de l'empreinte energetique et carbone de l'entrainement des modeles (C4.3).

Approche : transparente et reproductible, sans dependance lourde.
- On part du temps d'entrainement reel (colonne Time_sec de training_results.csv).
- Energie (kWh) = puissance_moyenne_W x temps_h x PUE / 1000
- CO2 (gCO2eq)  = energie_kWh x intensite_carbone_gCO2_par_kWh

Hypotheses documentees (modifiables) :
- CPU_TDP_WATTS : puissance moyenne du CPU sous charge (laptop/desktop classique).
- PUE : Power Usage Effectiveness (1.0 pour une machine locale, ~1.5 en datacenter).
- Intensite carbone : mix electrique. France ~ 56 gCO2/kWh (ADEME), Monde ~ 475 gCO2/kWh (AIE).

Si la librairie `codecarbon` est installee, `track_emissions_live` permet une mesure
materielle reelle en complement de l'estimation (cross-check).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# --- Hypotheses (transparentes, ajustables) ---------------------------------
CPU_TDP_WATTS = 45.0          # puissance moyenne CPU sous charge (W)
PUE = 1.0                     # machine locale (pas de datacenter)
CARBON_INTENSITY_FRANCE = 56.0   # gCO2eq / kWh (mix France, ADEME)
CARBON_INTENSITY_WORLD = 475.0   # gCO2eq / kWh (moyenne mondiale, AIE)
CAR_G_CO2_PER_KM = 120.0      # voiture thermique moyenne (gCO2/km) pour l'equivalent


@dataclass(frozen=True)
class EmissionEstimate:
    time_seconds: float
    energy_wh: float
    co2_france_g: float
    co2_world_g: float
    car_km_equivalent: float


def estimate_emissions(
    time_seconds: float,
    cpu_watts: float = CPU_TDP_WATTS,
    pue: float = PUE,
) -> EmissionEstimate:
    """Estime energie et CO2 a partir d'un temps d'entrainement (secondes)."""
    energy_wh = cpu_watts * (time_seconds / 3600.0) * pue
    energy_kwh = energy_wh / 1000.0
    co2_france = energy_kwh * CARBON_INTENSITY_FRANCE
    co2_world = energy_kwh * CARBON_INTENSITY_WORLD
    return EmissionEstimate(
        time_seconds=time_seconds,
        energy_wh=round(energy_wh, 4),
        co2_france_g=round(co2_france, 4),
        co2_world_g=round(co2_world, 4),
        car_km_equivalent=round(co2_world / CAR_G_CO2_PER_KM, 5),
    )


def build_report(training_results: pd.DataFrame) -> pd.DataFrame:
    """Construit le tableau eco-responsabilite a partir de training_results.csv.

    Attend une colonne `Model` et `Time_sec`.
    """
    if "Time_sec" not in training_results.columns:
        raise KeyError("La colonne 'Time_sec' est requise (issue de notebooks/03_modeling.py).")

    rows = []
    for record in training_results.to_dict(orient="records"):
        estimate = estimate_emissions(float(record["Time_sec"]))
        rows.append({
            "Model": record.get("Model", "?"),
            "Time_sec": round(float(record["Time_sec"]), 2),
            "Energy_Wh": estimate.energy_wh,
            "CO2_France_g": estimate.co2_france_g,
            "CO2_World_g": estimate.co2_world_g,
            "Car_km_eq": estimate.car_km_equivalent,
        })

    report = pd.DataFrame(rows).sort_values("CO2_France_g").reset_index(drop=True)
    total = {
        "Model": "TOTAL",
        "Time_sec": round(report["Time_sec"].sum(), 2),
        "Energy_Wh": round(report["Energy_Wh"].sum(), 4),
        "CO2_France_g": round(report["CO2_France_g"].sum(), 4),
        "CO2_World_g": round(report["CO2_World_g"].sum(), 4),
        "Car_km_eq": round(report["Car_km_eq"].sum(), 5),
    }
    return pd.concat([report, pd.DataFrame([total])], ignore_index=True)


def track_emissions_live(func, *args, **kwargs):
    """Mesure materielle reelle via codecarbon, si installe (sinon estimation seule).

    Renvoie (resultat_func, emissions_kgCO2 | None).
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        return func(*args, **kwargs), None

    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    tracker.start()
    try:
        result = func(*args, **kwargs)
    finally:
        emissions_kg = tracker.stop()
    return result, emissions_kg
