"""
Build docs/index.html - the self-contained live demo.

The model, the demo fleet and the evaluation metrics are inlined into a
single HTML file, so the page works three ways with no configuration:
GitHub Pages, any static host, or double-clicked straight off disk.

Usage:  python src/build_demo.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

TEMPLATE = "src/demo_template.html"
OUT = "docs/index.html"
FLEET_N = 240

RAW_COLS = [
    "waterpoint_id", "district", "source_type", "power_source", "management",
    "age_years", "depth_m", "static_water_level_m", "gw_trend_m_per_year",
    "rainfall_mm_12mo", "population_served", "duty_cycle", "water_tds_ppm",
    "iron_ppm", "chloride_ppm", "distance_to_workshop_km", "road_access_score",
    "days_since_service", "has_tariff", "caretaker_trained",
    "failures_past_24mo", "mean_repair_delay_days", "uptime_pct_12mo",
    "handle_effort_index", "season_month", "failure_next_90d",
]


def main() -> None:
    os.makedirs("docs", exist_ok=True)

    model = open("docs/model.json").read()
    metrics = json.load(open("models/metrics.json"))
    slim = {"dataset": metrics["dataset"], "test": metrics["test"],
            "selected_model": metrics["selected_model"],
            "comparison": metrics["comparison"]}

    df = pd.read_csv("data/waterpoints.csv").sample(FLEET_N, random_state=11)
    df = df[RAW_COLS].fillna({"iron_ppm": 0.6, "uptime_pct_12mo": 95.0,
                              "road_access_score": 6.0, "handle_effort_index": 4.0})
    fleet = df.round(3).to_json(orient="records")

    html = open(TEMPLATE).read()
    html = (html.replace("__MODEL_JSON__", model)
                .replace("__FLEET_JSON__", fleet)
                .replace("__METRICS_JSON__", json.dumps(slim)))
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  {os.path.getsize(OUT)/1024:.0f} KB  (fully self-contained)")


if __name__ == "__main__":
    main()
