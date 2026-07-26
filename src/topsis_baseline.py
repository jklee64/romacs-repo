"""
topsis_baseline.py — classical MADM (TOPSIS) comparator for RoMaCS.

Rationale
---------
In heterogeneous-network selection, the classical non-learning approach is
multi-attribute decision making (MADM), most commonly TOPSIS: candidate networks
are scored by their closeness to an ideal solution over weighted QoS attributes.
Our manuscript cites this line of work but did not use it as a comparator. This
module adds it, so that the learned selectors are benchmarked against BOTH
    (i) the label-generating utility policy (PolicyLOCF), and
    (ii) a standard TOPSIS ranker (TopsisLOCF),
under the identical missingness protocol.

Design (kept deliberately parallel to the policy baseline so the comparison is fair)
-----------------------------------------------------------------------------------
* Feasibility gate: identical to the oracle policy -- a channel must be available and
  meet the priority-dependent PER / throughput / latency requirements. This mirrors
  how MADM is used in practice (admission filter, then ranking).
* Ranking: TOPSIS over six criteria
      RSSI (benefit), SINR (benefit), PER (cost),
      throughput (benefit), latency (cost), monetary cost (cost)
  with priority-dependent weights derived from the same MSG_REQUIREMENTS weights,
  so neither comparator gets a hand-tuned advantage.
* Missing QoS: filled by last-observation-carried-forward along the trajectory,
  exactly as for PolicyLOCF; channels still missing at cold start are non-selectable.
* If no channel is feasible -> NO_CHANNEL.

RSSI and SINR are shifted into a positive range before vector normalization, since
the standard TOPSIS normalization assumes non-negative entries.
"""

import numpy as np
import pandas as pd

from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness, locf_impute,
    add_missingness_indicators, train_test_split_by_trajectory,
    CHANNELS, CHANNEL_NAMES, QOS_METRICS, MSG_REQUIREMENTS,
    THROUGHPUT_PER_KB, NO_CHANNEL_LABEL,
)
from romacs_experiment import compute_metrics, ALL_LABELS

# criterion order and whether each is a benefit (+1) or a cost (-1)
CRITERIA = ["rssi_dbm", "sinr_db", "per", "throughput_kbps", "latency_ms", "cost_norm"]
BENEFIT = np.array([+1, +1, -1, +1, -1, -1], dtype=float)

# shifts making RSSI/SINR non-negative for vector normalization
RSSI_SHIFT, SINR_SHIFT = 145.0, 25.0


def _weights(priority: int) -> np.ndarray:
    """Priority-dependent criterion weights, derived from the oracle's utility weights
    so that TOPSIS and the policy express the same operational preferences."""
    r = MSG_REQUIREMENTS[priority]
    w_rel, w_thr = r["w_reliability"], r["w_throughput"]
    w_lat, w_cost = r["w_latency"], r["w_cost"]
    # RSSI and SINR are signal-quality proxies; give them a share of the
    # reliability weight rather than inventing a new preference.
    w = np.array([0.5 * w_rel, 0.5 * w_rel, w_rel, w_thr, w_lat, w_cost], dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(CRITERIA), 1.0 / len(CRITERIA))


def topsis_select(row: dict) -> int:
    """Select a channel for one (LOCF-imputed) observation using feasibility + TOPSIS."""
    priority = int(row["msg_priority"])
    req = MSG_REQUIREMENTS[priority]
    req_tput = req["base_throughput_kbps"] + THROUGHPUT_PER_KB * float(row["msg_size_kb"])

    feasible, matrix = [], []
    for idx, name in enumerate(CHANNEL_NAMES):
        vals = {m: row[f"{name}__{m}"] for m in QOS_METRICS}
        # cold-start missing after LOCF -> not selectable (same rule as PolicyLOCF)
        if any(pd.isna(v) for v in vals.values()):
            continue
        if int(row[f"{name}__available"]) != 1:
            continue
        spec = CHANNELS[name]
        if vals["per"] > req["max_per"]:
            continue
        if vals["throughput_kbps"] < req_tput:
            continue
        if spec.latency_ms > req["max_latency_ms"]:
            continue
        feasible.append(idx)
        matrix.append([
            vals["rssi_dbm"] + RSSI_SHIFT,
            vals["sinr_db"] + SINR_SHIFT,
            vals["per"],
            vals["throughput_kbps"],
            spec.latency_ms,
            spec.cost_norm,
        ])

    if not feasible:
        return NO_CHANNEL_LABEL
    if len(feasible) == 1:
        return feasible[0]

    X = np.asarray(matrix, dtype=float)
    X = np.clip(X, 0.0, None)

    # (1) vector normalization
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom

    # (2) weighting
    V = R * _weights(priority)

    # (3) ideal / anti-ideal per criterion
    ideal = np.where(BENEFIT > 0, V.max(axis=0), V.min(axis=0))
    anti = np.where(BENEFIT > 0, V.min(axis=0), V.max(axis=0))

    # (4) separation measures and closeness coefficient
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    total = s_plus + s_minus
    closeness = np.divide(s_minus, total, out=np.full_like(total, 0.5), where=total > 0)

    return feasible[int(np.argmax(closeness))]


def topsis_predict(df_locf: pd.DataFrame) -> np.ndarray:
    return np.array([topsis_select(r) for r in df_locf.to_dict("records")], dtype=int)


SEEDS = list(range(8))
SWEEP = [0.0, 0.10, 0.25, 0.50, 0.75]


def main():
    records = []
    for seed in SEEDS:
        df = generate_dataset(GenConfig(n_trajectories=600, seed=seed))
        _, te = train_test_split_by_trajectory(df, test_size=0.30, seed=seed)
        for p in SWEEP:
            te_m = inject_mcar_missingness(te, p, seed + int(1e6 * p) + 7)
            te_locf = locf_impute(te_m)
            yte = te_locf["label"].to_numpy(int)
            prio = te_locf["msg_priority"].to_numpy(int)
            yp = topsis_predict(te_locf)
            rec = {"seed": seed, "system": "TopsisLOCF", "p": p}
            rec.update(compute_metrics(yte, yp, prio))
            records.append(rec)
        print(f"  seed {seed} done.")

    d = pd.DataFrame(records)
    d.to_csv("../results/topsis_long.csv", index=False)
    agg = d.groupby("p")[["macro_f1", "macro_f1_emergency", "f1_NO_CHANNEL", "f1_LTE_5G"]].agg(["mean", "std"])
    print("\nTOPSIS (MADM) baseline across missingness sweep:")
    print(agg.round(3).to_string())
    agg.to_csv("../results/topsis_summary.csv")


if __name__ == "__main__":
    main()
