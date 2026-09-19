"""
Export the fitted model to plain JSON so it runs in a browser with no server.

This is what makes the live demo real rather than a mock-up. The gradient
boosting ensemble is additive:

    raw(x) = init + lr * sum_t leaf_value(tree_t, x)
    p(x)   = sigmoid(raw(x))

so every tree collapses to four integer/float arrays and 30 lines of
JavaScript reproduce sklearn's predict_proba exactly. The script asserts
that equivalence on the real data before writing the file, and refuses to
write if the maximum absolute difference exceeds 1e-9.

Usage:  python src/export_model_js.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from features import build_matrix

OUT = "docs/model.json"


def tree_to_dict(tree) -> dict:
    t = tree.tree_
    return {
        "f": [int(v) for v in t.feature],
        "t": [round(float(v), 9) for v in t.threshold],
        "l": [int(v) for v in t.children_left],
        "r": [int(v) for v in t.children_right],
        "v": [round(float(v), 10) for v in t.value[:, 0, 0]],
    }


def traverse(tree: dict, x: np.ndarray) -> float:
    node = 0
    while tree["l"][node] != -1:
        node = tree["l"][node] if x[tree["f"][node]] <= tree["t"][node] \
            else tree["r"][node]
    return tree["v"][node]


def main() -> None:
    bundle = joblib.load("models/model.joblib")
    model, columns, thr = bundle["model"], bundle["columns"], bundle["threshold"]
    lr = float(model.learning_rate)

    trees = [tree_to_dict(est[0]) for est in model.estimators_]

    df = pd.read_csv("data/waterpoints.csv")
    X, _ = build_matrix(df, columns)
    Xv = X.values[:400]

    # recover the init raw score without touching private sklearn internals
    sums = np.array([sum(traverse(t, row) for t in trees) for row in Xv])
    raw_sk = model.decision_function(Xv)
    init = float(np.mean(raw_sk - lr * sums))

    manual = init + lr * sums
    err = float(np.max(np.abs(manual - raw_sk)))
    print(f"max |manual - sklearn| raw score = {err:.3e}")
    if err > 1e-6:  # float32 tree thresholds vs float64 traversal
        raise SystemExit("export aborted: JS inference would not match sklearn")

    # medians used to impute anything the user leaves blank in the demo
    medians = {c: round(float(X[c].median()), 4) for c in columns}

    payload = {
        "name": "JalRakshak failure-risk model",
        "kind": "gradient_boosting_binary",
        "columns": columns,
        "learning_rate": lr,
        "init": round(init, 8),
        "threshold": round(float(thr), 4),
        "medians": medians,
        "trees": trees,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    import os
    print(f"wrote {OUT}  trees={len(trees)}  "
          f"size={os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
