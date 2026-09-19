"""Feature engineering shared by training, the API and the browser export.

Keeping this in one module is what makes the live demo trustworthy: the
JavaScript in docs/index.html reproduces exactly these transformations, so
a prediction made in a browser is identical to one made by the Python model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "failure_next_90d"
ID_COL = "waterpoint_id"

NUMERIC = [
    "age_years", "depth_m", "static_water_level_m", "gw_trend_m_per_year",
    "rainfall_mm_12mo", "population_served", "duty_cycle", "water_tds_ppm",
    "iron_ppm", "chloride_ppm", "distance_to_workshop_km", "road_access_score",
    "days_since_service", "has_tariff", "caretaker_trained",
    "failures_past_24mo", "mean_repair_delay_days", "uptime_pct_12mo",
    "handle_effort_index",
]

CATEGORICAL = ["source_type", "power_source", "management"]

# Engineered features. Each one encodes a piece of domain knowledge that a
# raw column cannot express on its own.
ENGINEERED = [
    "lift_head_ratio",        # how much of the borehole is dry column
    "stress_index",           # duty cycle x lift - the core mechanical load
    "neglect_score",          # service gap scaled by breakdown history
    "corrosion_load",         # mineral aggression x age
    "isolation_penalty",      # how hard this point is to reach and fix
    "failures_per_year",      # normalised breakdown frequency
    "drawdown_5yr",           # projected extra lift in 5 years
    "pre_monsoon",            # seasonal low-water flag
    "downtime_days_12mo",     # uptime converted to lost days
]

FEATURES = NUMERIC + ENGINEERED  # categoricals are one-hot appended at fit time


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    depth = d["depth_m"].clip(lower=1.0)
    d["lift_head_ratio"] = (d["static_water_level_m"] / depth).clip(0, 1.5)
    d["stress_index"] = d["duty_cycle"] * np.sqrt(d["static_water_level_m"].clip(lower=0.5))
    d["neglect_score"] = (d["days_since_service"] / 365.0) * (1.0 + d["failures_past_24mo"])
    d["corrosion_load"] = (d["water_tds_ppm"] / 1000.0) * (1.0 + d["iron_ppm"].fillna(0.6)) \
        * np.log1p(d["age_years"])
    d["isolation_penalty"] = d["distance_to_workshop_km"] / \
        d["road_access_score"].fillna(6.0).clip(lower=1.0)
    d["failures_per_year"] = d["failures_past_24mo"] / 2.0
    d["drawdown_5yr"] = d["static_water_level_m"] + 5.0 * d["gw_trend_m_per_year"]
    d["pre_monsoon"] = d["season_month"].isin([4, 5, 6]).astype(int)
    d["downtime_days_12mo"] = (100.0 - d["uptime_pct_12mo"].fillna(95.0)) * 3.65
    return d


def build_matrix(df: pd.DataFrame, columns: list[str] | None = None):
    """Return (X, column_names). Missing values are median/zero filled."""
    d = add_engineered(df)
    num = d[NUMERIC + ENGINEERED].astype(float)
    cat = pd.get_dummies(d[CATEGORICAL].astype(str), prefix=CATEGORICAL)
    X = pd.concat([num, cat], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.astype(float), list(X.columns)
