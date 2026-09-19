"""Command-line scoring.  Usage: python src/predict.py data/waterpoints.csv --top 20"""
from __future__ import annotations
import argparse, joblib, pandas as pd
from triage import score_frame

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()

bundle = joblib.load("models/model.joblib")
scored = score_frame(pd.read_csv(a.csv), bundle)
cols = ["waterpoint_id", "district", "risk", "population_served", "crew_days", "priority"]
res = scored.sort_values("priority", ascending=False)[cols]
if a.out:
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
print(res.head(a.top).round(3).to_string(index=False))
