#!/usr/bin/env bash
# Reproduce every artifact in this repo from scratch.
set -e
python src/generate_data.py --n 12000 --out data/waterpoints.csv
python src/train.py
python src/triage.py --budget 120
python src/export_model_js.py
python src/build_demo.py
pytest -q tests/test_pipeline.py
node tests/test_demo_parity.js
echo "Done. Open docs/index.html in a browser."
