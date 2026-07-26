"""
romacs_mar_check.py — structured-missingness (MAR) robustness check.

Compares the primary learned selector (XGBoost) and the policy baseline under two
missingness mechanisms at matched overall rates:
  - MCAR: each QoS cell missing independently (main results), and
  - MAR : RSSI rarely missing, PER/throughput often missing, unavailable channels
          more often missing (structured, realistic).
Both mechanisms are applied for missingness-aware training and for testing.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import json as _json, os as _os
def _tuned_xgb():
    p=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'results','tuned_configs.json')
    if _os.path.exists(p):
        import json
        return json.load(open(p)).get('XGBoost',{})
    return {}
_XGB=_tuned_xgb()
from romacs_datagen import (
    GenConfig, generate_dataset, inject_mcar_missingness, inject_mar_missingness,
    locf_impute, add_missingness_indicators, train_test_split_by_trajectory,
)
from romacs_experiment import (
    indicator_feature_cols, policy_predict, compute_metrics, ALL_LABELS,
)

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif"})

SEEDS = list(range(8))
SWEEP = [0.0, 0.10, 0.25, 0.50, 0.75]
P_TR = 0.25
FEATS = indicator_feature_cols()
INJECT = {"MCAR": inject_mcar_missingness, "MAR": inject_mar_missingness}


def run():
    records = []
    for seed in SEEDS:
        df = generate_dataset(GenConfig(n_trajectories=600, seed=seed))
        tr, te = train_test_split_by_trajectory(df, test_size=0.30, seed=seed)
        for mech, inject in INJECT.items():
            # missingness-aware training under this mechanism
            tr_m = inject(tr, P_TR, seed + 1000)
            tr_feat, fills = add_missingness_indicators(tr_m)
            Xtr = tr_feat[FEATS].to_numpy(float); ytr = tr_feat["label"].to_numpy(int)
            clf = XGBClassifier(**_XGB, tree_method="hist", objective="multi:softprob",
                                num_class=len(ALL_LABELS), random_state=seed, verbosity=0)
            clf.fit(Xtr, ytr)
            for p in SWEEP:
                te_m = inject(te, p, seed + int(1e6 * p) + 7)
                te_feat, _ = add_missingness_indicators(te_m, impute_values=fills)
                Xte = te_feat[FEATS].to_numpy(float)
                yte = te_feat["label"].to_numpy(int)
                prio = te_feat["msg_priority"].to_numpy(int)
                # XGBoost
                mx = compute_metrics(yte, clf.predict(Xte), prio)
                records.append({"seed": seed, "mech": mech, "selector": "XGBoost",
                                "p": p, "macro_f1": mx["macro_f1"],
                                "emerg": mx["macro_f1_emergency"]})
                # Policy (LOCF)
                yp = policy_predict(locf_impute(te_m))
                mp = compute_metrics(yte, yp, prio)
                records.append({"seed": seed, "mech": mech, "selector": "PolicyLOCF",
                                "p": p, "macro_f1": mp["macro_f1"],
                                "emerg": mp["macro_f1_emergency"]})
        print(f"  seed {seed} done.")
    return pd.DataFrame(records)


def main():
    df = run()
    df.to_csv("../results/mar_check_long.csv", index=False)
    piv = df.groupby(["selector", "mech", "p"])["macro_f1"].mean().unstack("p")
    print("\nMean macro-F1 by selector x mechanism:")
    print(piv.round(3).to_string())
    piv.to_csv("../results/mar_vs_mcar_summary.csv")

    # figure
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        ("XGBoost", "MCAR"): dict(color="#9467bd", ls="-", marker="o"),
        ("XGBoost", "MAR"):  dict(color="#9467bd", ls="--", marker="^"),
        ("PolicyLOCF", "MCAR"): dict(color="black", ls="-", marker="s"),
        ("PolicyLOCF", "MAR"):  dict(color="black", ls="--", marker="D"),
    }
    for (sel, mech), st in styles.items():
        g = df[(df.selector == sel) & (df.mech == mech)].groupby("p")["macro_f1"]
        m, s = g.mean().reindex(SWEEP), g.std().reindex(SWEEP)
        ax.plot(SWEEP, m.values, label=f"{sel} ({mech})", linewidth=2, **st)
        ax.fill_between(SWEEP, (m - s).values, (m + s).values, alpha=0.10,
                        color=st["color"])
    ax.set_xlabel("QoS missingness rate  p")
    ax.set_ylabel("macro-F1")
    ax.set_title("MCAR vs. MAR (structured) missingness (mean $\\pm$ std, 8 seeds)")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("../figures/fig6_mar_vs_mcar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved fig6_mar_vs_mcar.png")


if __name__ == "__main__":
    main()
