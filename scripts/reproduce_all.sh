#!/usr/bin/env bash
# Reproduce every table and figure of the paper from scratch.
# Runtime: ~15-20 min on a single commodity CPU core.
set -e
cd "$(dirname "$0")/../src"

echo "[1/8] (optional) hyperparameter search -- skipped by default;"
echo "      configs/tuned_configs.json ships with the selected configuration."
# python romacs_tuning.py

echo "[2/8] Main missingness study (Tables main/why-trees/safety, confusion figs)"
python romacs_experiment.py

echo "[3/8] TOPSIS comparator"
python topsis_baseline.py

echo "[4/8] Staleness sweep + conformal abstention"
python romacs_v2_experiment.py

echo "[5/8] Structured MAR robustness check"
python romacs_mar_check.py

echo "[6/8] Feature importances"
python feature_importance.py

echo "[7/8] Inference latency"
python latency_benchmark.py

echo "[8/8] Figures"
python make_figures.py
python make_figures_v2.py
python fig1_redesign.py

echo "Done. See ../results and ../figures."
