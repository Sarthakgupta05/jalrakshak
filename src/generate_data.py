"""
JalRakshak - dataset builder.

Generates a causally-grounded synthetic census of rural water points
(handpumps, borewells, piped stands, spring boxes).

Why synthetic? The real analogue is the Taarifa / DrivenData "Pump it Up"
water-point registry plus state IMIS asset data. Those are open but not
redistributable inside a hackathon repo, so we encode the SAME documented
failure physics into a generator. Every driver below is taken from field
literature on rural water supply, not invented at random:

  * mechanical wear rises super-linearly with pump age and duty cycle
  * falling water tables force pumps to lift further -> cylinder/rod stress
  * high iron + TDS causes scaling and valve seat corrosion
  * long distance to the nearest mechanic workshop means small faults
    escalate into full breakdowns before anyone arrives
  * community-managed points with an active tariff outperform free ones
  * failures cluster: a point that broke twice already will break again

The resulting task is non-trivial: signal is spread across interacting
features, the positive class is a minority (~18%), and there is irreducible
noise, so a model cannot exceed ~0.90 AUC. That is deliberate.

Usage:  python src/generate_data.py --n 12000 --out data/waterpoints.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RNG_SEED = 42

DISTRICTS = [
    "Barmer", "Jaisalmer", "Bikaner", "Jodhpur", "Nagaur", "Churu",
    "Jhunjhunu", "Pali", "Ajmer", "Bhilwara", "Udaipur", "Banswara",
]
# districts in the arid west have deeper, faster-falling water tables
ARIDITY = {
    "Barmer": 0.95, "Jaisalmer": 1.00, "Bikaner": 0.90, "Jodhpur": 0.80,
    "Nagaur": 0.75, "Churu": 0.78, "Jhunjhunu": 0.60, "Pali": 0.62,
    "Ajmer": 0.50, "Bhilwara": 0.45, "Udaipur": 0.30, "Banswara": 0.25,
}
SOURCE_TYPES = ["handpump_india_mk2", "handpump_mk3", "submersible_borewell",
                "piped_standpost", "spring_box"]
SOURCE_P = [0.34, 0.16, 0.24, 0.18, 0.08]
MANAGEMENT = ["village_water_committee", "gram_panchayat", "private_operator",
              "unmanaged"]
MANAGEMENT_P = [0.34, 0.36, 0.12, 0.18]
POWER = ["handpump", "grid_electric", "solar_pv", "gravity"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build(n: int, seed: int = RNG_SEED, target_rate: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    district = rng.choice(DISTRICTS, size=n)
    aridity = np.array([ARIDITY[d] for d in district])

    source_type = rng.choice(SOURCE_TYPES, size=n, p=SOURCE_P)
    management = rng.choice(MANAGEMENT, size=n, p=MANAGEMENT_P)

    power = np.where(
        np.isin(source_type, ["handpump_india_mk2", "handpump_mk3"]), "handpump",
        np.where(source_type == "spring_box", "gravity",
                 rng.choice(["grid_electric", "solar_pv"], size=n, p=[0.62, 0.38])),
    )

    age_years = np.clip(rng.gamma(shape=2.4, scale=3.4, size=n), 0.2, 34).round(1)

    # deeper boreholes in arid districts
    depth_m = np.clip(rng.normal(38 + 58 * aridity, 16), 6, 210).round(1)
    depth_m = np.where(source_type == "spring_box", rng.uniform(2, 9, n).round(1), depth_m)

    # static water level sits below ground and is pushed down by aridity
    static_water_level_m = np.clip(
        depth_m * rng.uniform(0.25, 0.72, n) * (0.7 + 0.5 * aridity), 1, 190
    ).round(1)

    # metres the table drops per year - the slow killer
    gw_trend_m_per_year = np.clip(rng.normal(0.28 + 0.95 * aridity, 0.34), -0.5, 3.4).round(2)

    rainfall_mm_12mo = np.clip(rng.normal(680 - 470 * aridity, 130), 60, 1300).round(0)

    households_served = np.clip(rng.lognormal(mean=3.35, sigma=0.62, size=n), 4, 700).astype(int)
    population_served = (households_served * rng.normal(5.4, 0.7, n)).clip(12, 4200).astype(int)

    # duty cycle: litres lifted per day per design capacity
    design_capacity_lpd = np.where(power == "handpump", 9000,
                                   np.where(power == "gravity", 7000, 26000))
    daily_yield_l = np.clip(population_served * rng.normal(41, 9, n), 200, None)
    duty_cycle = np.clip(daily_yield_l / design_capacity_lpd, 0.02, 3.2).round(3)

    water_tds_ppm = np.clip(rng.normal(620 + 1250 * aridity, 380), 90, 4200).round(0)
    iron_ppm = np.clip(rng.gamma(1.9, 0.42, n) * (0.6 + aridity), 0.0, 9.5).round(2)
    chloride_ppm = np.clip(water_tds_ppm * rng.uniform(0.18, 0.44, n), 20, 2000).round(0)

    distance_to_workshop_km = np.clip(rng.gamma(2.2, 9.0, n), 0.5, 145).round(1)
    road_access_score = np.clip(rng.normal(6.6 - 0.035 * distance_to_workshop_km, 1.6), 1, 10).round(1)

    days_since_service = np.clip(rng.gamma(2.0, 95, n), 3, 1500).astype(int)
    # managed points get serviced more often
    mgmt_service_mult = np.select(
        [management == "village_water_committee", management == "gram_panchayat",
         management == "private_operator", management == "unmanaged"],
        [0.72, 0.92, 0.66, 1.75],
    )
    days_since_service = np.clip((days_since_service * mgmt_service_mult), 3, 2200).astype(int)

    has_tariff = (rng.random(n) < np.select(
        [management == "village_water_committee", management == "gram_panchayat",
         management == "private_operator", management == "unmanaged"],
        [0.74, 0.41, 0.93, 0.05])).astype(int)

    caretaker_trained = (rng.random(n) < (0.25 + 0.42 * has_tariff)).astype(int)

    # historical breakdowns - driven by age, duty cycle and neglect (clustering)
    hist_lam = (0.16 * age_years * (0.55 + duty_cycle)
                + 0.0016 * days_since_service
                + 0.30 * (management == "unmanaged"))
    failures_past_24mo = rng.poisson(np.clip(hist_lam, 0, 9)).clip(0, 12)

    mean_repair_delay_days = np.clip(
        rng.normal(4 + 0.42 * distance_to_workshop_km - 1.6 * road_access_score
                   - 3.0 * has_tariff, 6), 0.5, 120).round(1)

    uptime_pct_12mo = np.clip(
        rng.normal(96 - 3.4 * failures_past_24mo - 0.09 * mean_repair_delay_days, 4),
        22, 100).round(1)

    # a cheap vibration / stroke-effort proxy a field worker can log on a phone
    handle_effort_index = np.clip(
        rng.normal(3.0 + 0.055 * static_water_level_m + 1.5 * duty_cycle
                   + 0.11 * age_years + 0.9 * iron_ppm / 3.0, 1.2), 0.5, 10).round(2)

    season_month = rng.integers(1, 13, n)
    # pre-monsoon (Apr-Jun) = lowest tables, highest stress
    pre_monsoon = np.isin(season_month, [4, 5, 6]).astype(int)

    # ---------- the true (latent) failure process ----------
    z = (
        0.088 * age_years
        + 0.0021 * age_years ** 2
        + 0.95 * duty_cycle
        + 0.0105 * static_water_level_m
        + 0.62 * gw_trend_m_per_year
        + 0.00022 * water_tds_ppm
        + 0.20 * iron_ppm
        + 0.0138 * distance_to_workshop_km
        + 0.00092 * days_since_service
        + 0.34 * failures_past_24mo
        + 0.017 * mean_repair_delay_days
        + 0.26 * handle_effort_index
        - 0.58 * has_tariff
        - 0.44 * caretaker_trained
        - 0.081 * road_access_score
        - 0.020 * uptime_pct_12mo
        + 0.46 * pre_monsoon
        + 0.42 * (management == "unmanaged")
        - 0.24 * (management == "private_operator")
        + 0.33 * (source_type == "handpump_india_mk2") * (age_years > 12)
        + 0.30 * (power == "grid_electric")
        - 0.18 * (power == "gravity")
        # interaction: a deep water table matters far more on a hand pump
        + 0.011 * static_water_level_m * (power == "handpump")
        # interaction: corrosive water plus long neglect
        + 0.00018 * water_tds_ppm * (days_since_service > 365)
        + rng.normal(0, 0.62, n)  # irreducible noise
    )
    # Solve for the intercept that reproduces the field-reported base rate of
    # non-functional rural water points (~18% over a 90-day window).
    lo, hi = -40.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    p = sigmoid(z + intercept)
    failure_next_90d = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "waterpoint_id": [f"WP-{i:06d}" for i in range(1, n + 1)],
        "district": district,
        "source_type": source_type,
        "power_source": power,
        "management": management,
        "age_years": age_years,
        "depth_m": depth_m,
        "static_water_level_m": static_water_level_m,
        "gw_trend_m_per_year": gw_trend_m_per_year,
        "rainfall_mm_12mo": rainfall_mm_12mo,
        "households_served": households_served,
        "population_served": population_served,
        "duty_cycle": duty_cycle,
        "water_tds_ppm": water_tds_ppm,
        "iron_ppm": iron_ppm,
        "chloride_ppm": chloride_ppm,
        "distance_to_workshop_km": distance_to_workshop_km,
        "road_access_score": road_access_score,
        "days_since_service": days_since_service,
        "has_tariff": has_tariff,
        "caretaker_trained": caretaker_trained,
        "failures_past_24mo": failures_past_24mo,
        "mean_repair_delay_days": mean_repair_delay_days,
        "uptime_pct_12mo": uptime_pct_12mo,
        "handle_effort_index": handle_effort_index,
        "season_month": season_month,
        "failure_next_90d": failure_next_90d,
    })

    # realistic messiness: sensors and surveys miss values
    for col, frac in [("iron_ppm", 0.06), ("uptime_pct_12mo", 0.04),
                      ("road_access_score", 0.03), ("handle_effort_index", 0.05)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", default="data/waterpoints.csv")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--rate", type=float, default=0.18)
    a = ap.parse_args()

    df = build(a.n, a.seed, a.rate)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  rows={len(df)}  "
          f"failure_rate={df.failure_next_90d.mean():.3f}")


if __name__ == "__main__":
    main()
