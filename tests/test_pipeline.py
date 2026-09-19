"""End-to-end tests for the JalRakshak pipeline."""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import ENGINEERED, TARGET, build_matrix   # noqa: E402
from generate_data import build                          # noqa: E402
from triage import knapsack, policy_picks, score_frame, realised_person_days_saved  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(os.path.join(ROOT, "models", "model.joblib"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(os.path.join(ROOT, "data", "waterpoints.csv"))


def test_generator_is_deterministic():
    a, b = build(500, seed=1), build(500, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_generator_hits_target_prevalence():
    d = build(4000, seed=3, target_rate=0.18)
    assert 0.15 < d[TARGET].mean() < 0.21


def test_engineered_features_present_and_finite(df):
    X, cols = build_matrix(df.head(300))
    for f in ENGINEERED:
        assert f in cols
    assert np.isfinite(X.values).all(), "feature matrix contains NaN or inf"


def test_build_matrix_column_order_is_stable(bundle, df):
    X, cols = build_matrix(df.head(50), bundle["columns"])
    assert cols == bundle["columns"]


def test_single_row_scores_identically_to_batch(bundle, df):
    sub = df.head(40)
    Xb, _ = build_matrix(sub, bundle["columns"])
    pb = bundle["model"].predict_proba(Xb.values)[:, 1]
    for i in range(0, 40, 7):
        Xs, _ = build_matrix(sub.iloc[[i]], bundle["columns"])
        ps = bundle["model"].predict_proba(Xs.values)[0, 1]
        assert abs(ps - pb[i]) < 1e-9


def test_model_beats_random_by_a_wide_margin(bundle, df):
    from sklearn.metrics import roc_auc_score
    X, _ = build_matrix(df, bundle["columns"])
    p = bundle["model"].predict_proba(X.values)[:, 1]
    assert roc_auc_score(df[TARGET], p) > 0.85


def test_risk_responds_to_known_causal_drivers(bundle, df):
    """Neglect must raise predicted risk. If this fails the model is nonsense."""
    row = df.head(1).copy()
    row["days_since_service"] = 30
    row["failures_past_24mo"] = 0
    X1, _ = build_matrix(row, bundle["columns"])
    low = bundle["model"].predict_proba(X1.values)[0, 1]
    row["days_since_service"] = 1400
    row["failures_past_24mo"] = 7
    X2, _ = build_matrix(row, bundle["columns"])
    high = bundle["model"].predict_proba(X2.values)[0, 1]
    assert high > low


def test_triage_respects_the_crew_day_budget(bundle, df):
    scored = score_frame(df.sample(400, random_state=5).reset_index(drop=True), bundle)
    picks = knapsack(scored, 25.0)
    assert scored.loc[picks, "crew_days"].sum() <= 25.0 + 1e-9
    assert len(picks) > 0


def test_triage_outperforms_every_heuristic(bundle, df):
    scored = score_frame(df.sample(900, random_state=6).reset_index(drop=True), bundle)
    ours = realised_person_days_saved(scored, policy_picks(scored, 60, "jalrakshak"))
    for pol in ["random", "oldest_first", "most_complaints", "largest_village"]:
        alt = realised_person_days_saved(scored, policy_picks(scored, 60, pol))
        assert ours > alt, f"triage lost to {pol}"


def test_metrics_file_is_complete():
    m = json.load(open(os.path.join(ROOT, "models", "metrics.json")))
    for k in ["roc_auc", "pr_auc", "brier", "recall", "lift_at_top20pct"]:
        assert k in m["test"]
    assert m["test"]["roc_auc"] > 0.85


def test_exported_browser_model_matches_sklearn(bundle, df):
    """The JSON shipped to the browser must reproduce sklearn's raw scores."""
    M = json.load(open(os.path.join(ROOT, "docs", "model.json")))
    X, _ = build_matrix(df.head(200), bundle["columns"])
    xs = X.values

    def leaf(t, x):
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
        return t["v"][n]

    manual = np.array([M["init"] + M["learning_rate"] * sum(leaf(t, x) for t in M["trees"])
                       for x in xs])
    assert np.max(np.abs(manual - bundle["model"].decision_function(xs))) < 1e-6


def test_demo_page_is_self_contained():
    html = open(os.path.join(ROOT, "docs", "index.html")).read()
    assert "__MODEL_JSON__" not in html, "template placeholder was not filled"
    assert 'id="modeldata"' in html and 'id="fleetdata"' in html
    assert "src=" not in html.split("<script")[1][:200], "no external script may be loaded"


def test_api_predict_and_triage():
    sys.path.insert(0, os.path.join(ROOT, "app"))
    from api import app
    client = app.test_client()
    assert client.get("/health").status_code == 200

    payload = {"source_type": "handpump_india_mk2", "power_source": "handpump",
               "management": "unmanaged", "age_years": 18, "depth_m": 90,
               "static_water_level_m": 62, "gw_trend_m_per_year": 1.6,
               "population_served": 900, "duty_cycle": 1.5, "water_tds_ppm": 2400,
               "distance_to_workshop_km": 70, "days_since_service": 900,
               "failures_past_24mo": 5, "mean_repair_delay_days": 30,
               "season_month": 5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.get_json()["risk_90d"] <= 1.0

    bad = client.post("/predict", json={"age_years": 5})
    assert bad.status_code == 400

    t = client.post("/triage", json={"budget": 6, "waterpoints": [payload] * 12})
    assert t.status_code == 200
    assert t.get_json()["crew_days_used"] <= 6.0 + 1e-6
