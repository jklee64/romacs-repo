"""
latency_benchmark.py — inference-latency measurement for the RoMaCS selectors.

Measures, on a commodity CPU (no GPU), the wall-clock inference latency of each trained
selector:
  - single-sample latency (ms/decision): time to classify one observation, averaged
    over many repetitions -- the operationally relevant metric for a per-event VTS
    decision; and
  - batch-amortized latency (ms/sample): full test set predicted at once, divided by
    the number of samples -- relevant for bulk/offline processing.
Reported as mean +/- std over seeds. Uses the same tuned models as the main experiment.
"""
import time
import numpy as np
import pandas as pd

from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness,
    add_missingness_indicators, train_test_split_by_trajectory,
)
from romacs_experiment import make_models, indicator_feature_cols, ALL_LABELS

FEATS = indicator_feature_cols()
SEEDS = [0, 1, 2]           # a few seeds for mean +/- std (training is the cost, not timing)
SINGLE_REPS = 300           # repetitions for single-sample timing
WARMUP = 25


def time_single(clf, x_row):
    """Mean ms to predict one sample (warm), over SINGLE_REPS reps."""
    for _ in range(WARMUP):
        clf.predict(x_row)
    ts = []
    for _ in range(SINGLE_REPS):
        t0 = time.perf_counter()
        clf.predict(x_row)
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return float(np.mean(ts))


def time_batch(clf, X):
    """ms per sample when the whole test set is predicted at once (best of 3 passes)."""
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        clf.predict(X)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best / len(X)


def main():
    rows = []
    for seed in SEEDS:
        df = generate_dataset(GenConfig(n_trajectories=150, seed=seed))
        tr, te = train_test_split_by_trajectory(df, test_size=0.30, seed=seed)
        tr_m = inject_mcar_missingness(tr, 0.25, seed + 1000)
        tr_feat, fills = add_missingness_indicators(tr_m)
        te_m = inject_mcar_missingness(te, 0.25, seed + 7)
        te_feat, _ = add_missingness_indicators(te_m, impute_values=fills)
        Xtr, ytr = tr_feat[FEATS].to_numpy(float), tr_feat["label"].to_numpy(int)
        Xte = te_feat[FEATS].to_numpy(float)
        x1 = Xte[:1]  # one sample

        for name, clf in make_models(seed).items():
            clf.fit(Xtr, ytr)
            rows.append({
                "seed": seed, "model": name,
                "single_ms": time_single(clf, x1),
                "batch_ms_per_sample": time_batch(clf, Xte),
            })
        print(f"  seed {seed} timed.")

    d = pd.DataFrame(rows)
    agg = d.groupby("model").agg(
        single_ms_mean=("single_ms", "mean"), single_ms_std=("single_ms", "std"),
        batch_us_mean=("batch_ms_per_sample", lambda s: s.mean() * 1e3),  # -> microseconds
    ).reindex(["DecisionTree", "Bagging", "AdaBoost", "RUSBoost", "XGBoost", "LogReg", "MLP"])
    agg.to_csv("../results/latency.csv")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nInference latency (commodity CPU, mean over seeds):")
    print(agg.to_string())
    print("\nColumns: single_ms = ms per single-sample decision; "
          "batch_us = microseconds/sample amortized over the test set.")


if __name__ == "__main__":
    main()
