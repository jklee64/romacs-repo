"""
feature_importance.py — RoMaCS interpretability analysis (paper contribution 4).

Aggregates XGBoost gain-based feature importances over the 8 seeds under the
missingness-aware training protocol, groups them by feature category, and produces:
  - a horizontal bar chart of the top-N features (colored by category), and
  - a category-level importance summary.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier
import json as _json, os as _os
def _tuned_xgb():
    p=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'results','tuned_configs.json')
    if _os.path.exists(p):
        import json
        return json.load(open(p)).get('XGBoost',{})
    return {}
_XGB=_tuned_xgb()

import romacs_datagen as dg
from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness,
    add_missingness_indicators, train_test_split_by_trajectory,
)
from romacs_experiment import (
    indicator_feature_cols, SCENARIO_COLS, QOS_COLS, AVAIL_COLS, IND_COLS, ALL_LABELS,
)

plt.rcParams.update({"font.size": 9.5, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif"})

SEEDS = list(range(8))
FEATS = indicator_feature_cols()  # 52 features, fixed order


def category_of(feat: str) -> str:
    if feat in SCENARIO_COLS:
        return "Scenario"
    if feat in AVAIL_COLS:
        return "Availability"
    if feat.endswith("__isnan"):
        return "Missingness indicator"
    return "QoS"


def pretty(feat: str) -> str:
    # human-readable labels
    m = {"distance_to_shore_km": "distance to shore", "sea_state": "sea state",
         "traffic_density": "traffic density", "weather_severity": "weather severity",
         "msg_priority": "msg priority", "msg_size_kb": "msg size", "hour_of_day": "hour"}
    if feat in m:
        return m[feat]
    f = feat.replace("__isnan", " (missing?)").replace("__available", " avail.")
    f = f.replace("__rssi_dbm", " RSSI").replace("__sinr_db", " SINR")
    f = f.replace("__per", " PER").replace("__throughput_kbps", " tput")
    f = f.replace("VHF_DSC", "VHF").replace("AIS_VDES", "AIS").replace("SATELLITE", "SAT")
    f = f.replace("LTE_5G", "LTE").replace("_", " ")
    return f


def main():
    imp = np.zeros((len(SEEDS), len(FEATS)))
    for k, seed in enumerate(SEEDS):
        df = generate_dataset(GenConfig(n_trajectories=150, seed=seed))
        tr, _ = train_test_split_by_trajectory(df, test_size=0.30, seed=seed)
        tr_missing = inject_mcar_missingness(tr, p=0.25, seed=seed + 1000)
        tr_feat, _ = add_missingness_indicators(tr_missing)
        X = tr_feat[FEATS].to_numpy(float)
        y = tr_feat["label"].to_numpy(int)
        clf = XGBClassifier(**_XGB, tree_method="hist", objective="multi:softprob",
                            num_class=len(ALL_LABELS), random_state=seed,
                            importance_type="gain", verbosity=0)
        clf.fit(X, y)
        fi = clf.feature_importances_
        imp[k] = fi / fi.sum()  # normalize to sum 1 per seed
    mean = imp.mean(0)
    std = imp.std(0)

    table = pd.DataFrame({
        "feature": FEATS, "label": [pretty(f) for f in FEATS],
        "category": [category_of(f) for f in FEATS],
        "importance": mean, "std": std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    table.to_csv("../results/feature_importance.csv", index=False)

    # category-level summary
    cat = table.groupby("category")["importance"].sum().sort_values(ascending=False)
    print("Category-level importance share:")
    for c, v in cat.items():
        print(f"  {c:24s}: {100*v:5.1f}%")
    print("\nTop 15 features:")
    print(table.head(15)[["label", "category", "importance", "std"]].to_string(index=False))

    # ---- figure: top-N horizontal bar chart, colored by category ----
    topN = 18
    sub = table.head(topN).iloc[::-1]  # reverse for barh (largest on top)
    colors = {"Scenario": "#1f77b4", "QoS": "#2ca02c",
              "Availability": "#ff7f0e", "Missingness indicator": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.barh(range(len(sub)), sub["importance"].values,
            xerr=sub["std"].values, capsize=2,
            color=[colors[c] for c in sub["category"]], edgecolor="#333", linewidth=0.4)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["label"].values, fontsize=8.5)
    ax.set_xlabel("Normalized gain importance (mean $\\pm$ std over 8 seeds)")
    ax.set_title("Top feature importances (XGBoost, missingness-aware)")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [Patch(facecolor=colors[c], edgecolor="#333", label=c) for c in colors]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("../figures/fig5_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved fig5_feature_importance.png")


if __name__ == "__main__":
    main()
