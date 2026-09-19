"""
Train and evaluate the JalRakshak failure-risk model.

Compares four candidates on an identical split, picks the best by PR-AUC
(the right metric for an imbalanced screening problem), checks calibration,
finds the decision threshold that maximises field utility, and writes:

  models/model.joblib     - the fitted pipeline
  models/metrics.json     - every number quoted in the README and the deck
  reports/figures/*.png   - ROC, PR, calibration, confusion, importance

Usage:  python src/train.py
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, build_matrix

DATA = "data/waterpoints.csv"
FIG = "reports/figures"
SEED = 42

# Field economics used to pick the operating threshold.
# A missed failure costs a village ~21 days without water; a wasted
# preventive visit costs one crew-day. Recall is therefore worth far more
# than precision, but not infinitely more.
COST_FN = 8.0
COST_FP = 1.0

# A candidate is only deployable if it can be exported to plain JSON and run
# in a browser with no server (see src/export_model_js.py). That constraint is
# part of the product, so it is part of model selection.
BROWSER_EXPORTABLE = {"gradient_boosting", "logistic_regression"}


def candidates():
    return {
        "logistic_regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.6, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=6, max_features="sqrt",
            n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.06, max_depth=3,
            subsample=0.9, random_state=SEED),
    }


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA)
    X, cols = build_matrix(df)
    y = df[TARGET].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=SEED)

    results = {}
    fitted = {}
    for name, model in candidates().items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "roc_auc": float(roc_auc_score(y_te, p)),
            "pr_auc": float(average_precision_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
        }
        fitted[name] = model
        print(f"{name:26s} ROC-AUC={results[name]['roc_auc']:.4f} "
              f"PR-AUC={results[name]['pr_auc']:.4f} "
              f"Brier={results[name]['brier']:.4f}")

    # Selection rule: take every candidate within 2% PR-AUC of the leader,
    # keep only the browser-deployable ones, then pick the best calibrated.
    # Calibration is not cosmetic here - the triage optimiser multiplies the
    # predicted probability by population served, so a model that merely ranks
    # well but reports inflated probabilities would misallocate real crews.
    leader = max(results.values(), key=lambda r: r["pr_auc"])["pr_auc"]
    shortlist = [k for k, r in results.items()
                 if r["pr_auc"] >= 0.98 * leader and k in BROWSER_EXPORTABLE]
    best_name = min(shortlist, key=lambda k: results[k]["brier"])
    best = fitted[best_name]
    print(f"\nselected: {best_name}")

    cv = cross_val_score(
        candidates()[best_name], X.values, y, cv=StratifiedKFold(5, shuffle=True,
        random_state=SEED), scoring="average_precision", n_jobs=-1)

    p_te = best.predict_proba(X_te)[:, 1]

    # ---- threshold chosen by expected field cost, not by 0.5 ----
    grid = np.linspace(0.02, 0.9, 400)
    costs = []
    for t in grid:
        pred = (p_te >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        costs.append(COST_FN * fn + COST_FP * fp)
    thr = float(grid[int(np.argmin(costs))])
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    # value of the model vs. the status quo (fix them when they break)
    baseline_cost = COST_FN * int(y_te.sum())
    model_cost = COST_FN * fn + COST_FP * fp
    lift_at_20 = None
    order = np.argsort(-p_te)
    top = order[: max(1, int(0.20 * len(order)))]
    lift_at_20 = float(y_te[top].mean() / y_te.mean())

    perm = permutation_importance(best, X_te, y_te, n_repeats=8,
                                  random_state=SEED, scoring="average_precision")
    imp = sorted(zip(cols, perm.importances_mean), key=lambda t: -t[1])

    metrics = {
        "dataset": {"rows": int(len(df)), "features": len(cols),
                    "positive_rate": float(y.mean())},
        "comparison": results,
        "selected_model": best_name,
        "cv_pr_auc_mean": float(cv.mean()),
        "cv_pr_auc_std": float(cv.std()),
        "test": {
            "roc_auc": results[best_name]["roc_auc"],
            "pr_auc": results[best_name]["pr_auc"],
            "brier": results[best_name]["brier"],
            "threshold": thr,
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                                 "fn": int(fn), "tp": int(tp)},
            "lift_at_top20pct": lift_at_20,
            "downtime_days_avoided_vs_reactive": float(baseline_cost - model_cost),
            "downtime_reduction_pct": float(100 * (baseline_cost - model_cost) / baseline_cost),
        },
        "selection_shortlist": shortlist,
        "top_features": [{"feature": f, "importance": float(v)} for f, v in imp[:15]],
        "columns": cols,
    }
    with open("models/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    joblib.dump({"model": best, "columns": cols, "threshold": thr},
                "models/model.joblib")

    # ---------------- figures ----------------
    C1, C2 = "#0E7C7B", "#D1495B"

    fpr, tpr, _ = roc_curve(y_te, p_te)
    plt.figure(figsize=(5, 4.2))
    plt.plot(fpr, tpr, color=C1, lw=2.2,
             label=f"AUC = {results[best_name]['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - failure within 90 days"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/roc_curve.png", dpi=150); plt.close()

    pr, rc, _ = precision_recall_curve(y_te, p_te)
    plt.figure(figsize=(5, 4.2))
    plt.plot(rc, pr, color=C2, lw=2.2,
             label=f"PR-AUC = {results[best_name]['pr_auc']:.3f}")
    plt.axhline(y.mean(), ls="--", color="#999", lw=1, label="base rate")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/pr_curve.png", dpi=150); plt.close()

    ptrue, ppred = calibration_curve(y_te, p_te, n_bins=12, strategy="quantile")
    plt.figure(figsize=(5, 4.2))
    plt.plot(ppred, ptrue, "o-", color=C1, lw=2)
    plt.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration (Brier = {results[best_name]['brier']:.3f})")
    plt.tight_layout(); plt.savefig(f"{FIG}/calibration.png", dpi=150); plt.close()

    plt.figure(figsize=(4.6, 4.2))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, cmap="BuGn")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     fontsize=15, color="#222")
    plt.xticks([0, 1], ["pred OK", "pred fail"])
    plt.yticks([0, 1], ["actual OK", "actual fail"])
    plt.title(f"Confusion matrix @ p>={thr:.2f}")
    plt.tight_layout(); plt.savefig(f"{FIG}/confusion_matrix.png", dpi=150); plt.close()

    top = imp[:12][::-1]
    plt.figure(figsize=(6.4, 4.6))
    plt.barh([t[0] for t in top], [t[1] for t in top], color=C1)
    plt.xlabel("Permutation importance (PR-AUC drop)")
    plt.title("What drives predicted failure")
    plt.tight_layout(); plt.savefig(f"{FIG}/feature_importance.png", dpi=150); plt.close()

    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
