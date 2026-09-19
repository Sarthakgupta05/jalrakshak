"""
JalRakshak REST API (Flask).

Endpoints
    GET  /health              service + model status
    GET  /metrics             held-out evaluation metrics
    POST /predict             {..water point fields..} -> risk + band
    POST /triage              {"budget": 40, "waterpoints": [...]} -> repair plan
    GET  /                    serves the offline demo page

Run:  python app/api.py     (http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import build_matrix          # noqa: E402
from triage import knapsack, score_frame   # noqa: E402

app = Flask(__name__, static_folder=None)
BUNDLE = joblib.load(os.path.join(ROOT, "models", "model.joblib"))
METRICS = json.load(open(os.path.join(ROOT, "models", "metrics.json")))

REQUIRED = ["source_type", "power_source", "management", "age_years", "depth_m",
            "static_water_level_m", "gw_trend_m_per_year", "population_served",
            "duty_cycle", "water_tds_ppm", "distance_to_workshop_km",
            "days_since_service", "failures_past_24mo", "mean_repair_delay_days",
            "season_month"]

DEFAULTS = {"rainfall_mm_12mo": 520, "iron_ppm": 0.9, "chloride_ppm": 400,
            "road_access_score": 6.0, "has_tariff": 0, "caretaker_trained": 0,
            "uptime_pct_12mo": 92.0, "handle_effort_index": 4.5}


def band(p: float) -> str:
    if p >= 0.45:
        return "high"
    return "elevated" if p >= BUNDLE["threshold"] else "low"


@app.get("/health")
def health():
    return jsonify(status="ok", model=METRICS["selected_model"],
                   features=len(BUNDLE["columns"]),
                   threshold=BUNDLE["threshold"])


@app.get("/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/predict")
def predict():
    payload = request.get_json(force=True) or {}
    rows = payload if isinstance(payload, list) else [payload]
    missing = [k for k in REQUIRED if k not in rows[0]]
    if missing:
        return jsonify(error="missing required fields", missing=missing), 400
    df = pd.DataFrame([{**DEFAULTS, **r} for r in rows])
    X, _ = build_matrix(df, BUNDLE["columns"])
    p = BUNDLE["model"].predict_proba(X.values)[:, 1]
    out = [{"risk_90d": round(float(v), 4), "band": band(float(v)),
            "action": "schedule preventive visit" if v >= BUNDLE["threshold"]
                      else "routine cycle"} for v in p]
    return jsonify(out if isinstance(payload, list) else out[0])


@app.post("/triage")
def triage_endpoint():
    payload = request.get_json(force=True) or {}
    budget = float(payload.get("budget", 40))
    rows = payload.get("waterpoints") or []
    if not rows:
        return jsonify(error="provide a non-empty 'waterpoints' list"), 400
    df = pd.DataFrame([{**DEFAULTS, **r} for r in rows])
    scored = score_frame(df, BUNDLE)
    picks = knapsack(scored, budget)
    sel = scored.loc[picks].sort_values("priority", ascending=False)
    return jsonify({
        "budget_crew_days": budget,
        "crew_days_used": round(float(sel["crew_days"].sum()), 2),
        "points_selected": int(len(sel)),
        "expected_person_days_protected": round(float(sel["benefit"].sum())),
        "plan": [{"waterpoint_id": r.get("waterpoint_id", f"row-{i}"),
                  "risk_90d": round(float(r["risk"]), 4),
                  "crew_days": round(float(r["crew_days"]), 2),
                  "priority": round(float(r["priority"]), 1)}
                 for i, r in sel.reset_index(drop=True).iterrows()],
    })


@app.get("/")
def demo():
    return send_from_directory(os.path.join(ROOT, "docs"), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
