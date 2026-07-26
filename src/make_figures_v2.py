"""
make_figures_v2.py — IoTJ revision figures.

Fig. 1  fig1_dataflow.png          data-flow with degradation injection (missingness OR
                                   staleness) and the conformal abstention layer
Fig. 2  fig2_qos_vs_distance.png   unchanged physical model (regenerated)
Fig. 3  fig_degradation.png        missingness sweep incl. TopsisLOCF
Fig. 7  fig7_staleness.png         staleness sweep (AgeAware/NoAgeFeat/StaleBlind/Policy)
Fig. 8  fig8_abstention.png        abstention rate & committed emergency error vs p

Outputs to ../figures/.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, ".")
from make_figures import make_fig2, _box, _arrow

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
OUT = "../figures"


# --------------------------------------------------------------------------- #
# FIG. 1 v2 — data-flow with degradation injection + conformal abstention
# --------------------------------------------------------------------------- #
def make_fig1_v2(path):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C_GEN = "#dbe9f6"; C_ORACLE = "#fde9d0"; C_MISS = "#f6dbdb"
    C_LEARN = "#dff0db"; C_POLICY = "#efe0f4"; C_EVAL = "#eeeeee"; C_CONF = "#fff3c4"

    # Top row: generation pipeline
    _box(ax, (0.02, 0.78), 0.20, 0.17,
         "Trajectory\ngeneration\n(vessel motion,\nenv. random walk)", C_GEN)
    _box(ax, (0.28, 0.78), 0.20, 0.17,
         "Physical QoS\nmodel (complete)\n$d\\!\\to\\!$PL$\\to$RSSI\n$\\to$SINR$\\to$PER,$T$", C_GEN)
    _box(ax, (0.54, 0.78), 0.20, 0.17,
         "Oracle labeling\n(utility policy +\nNO_CHANNEL)", C_ORACLE)
    _box(ax, (0.80, 0.78), 0.18, 0.17, "Labeled\nbenchmark\n$(\\mathbf{x}, y)$", C_GEN)
    _arrow(ax, (0.22, 0.865), (0.28, 0.865))
    _arrow(ax, (0.48, 0.865), (0.54, 0.865))
    _arrow(ax, (0.74, 0.865), (0.80, 0.865))

    # Degradation injection (missingness OR staleness)
    _box(ax, (0.33, 0.545), 0.44, 0.14,
         "Degradation injection (QoS only)\nMCAR/MAR missingness (rate $p$)  or\n"
         "staleness via probing (rate $r$, AoI $\\boldsymbol{\\tau}$)", C_MISS)
    _arrow(ax, (0.89, 0.78), (0.89, 0.615))
    _arrow(ax, (0.89, 0.615), (0.77, 0.615))

    # Two branches
    _box(ax, (0.03, 0.30), 0.44, 0.17,
         "Learner path\nimpute+indicate or AoI representation\n"
         "$\\to$ tree-ensemble core (XGBoost et al.)\n$\\to$ class probabilities $\\hat{\\pi}(\\mathbf{x})$",
         C_LEARN)
    _box(ax, (0.55, 0.30), 0.42, 0.17,
         "Non-learning comparators\nLOCF along trajectory $\\to$ utility policy\n"
         "or TOPSIS ranking\n(cold-start channels excluded)", C_POLICY)
    _arrow(ax, (0.43, 0.545), (0.25, 0.47), text="train/test", rad=-0.15)
    _arrow(ax, (0.62, 0.545), (0.76, 0.47), rad=0.15)

    # Conformal abstention layer
    _box(ax, (0.03, 0.115), 0.44, 0.125,
         "Mondrian conformal abstention ($\\alpha$)\n"
         "$|\\Gamma(\\mathbf{x})|=1$: commit  \u2022  else: defer\nwith short-list $\\Gamma(\\mathbf{x})$",
         C_CONF)
    _arrow(ax, (0.25, 0.30), (0.25, 0.24))

    # Evaluation
    _box(ax, (0.20, 0.005), 0.60, 0.075,
         "Evaluation:  macro-F1 · AUDC · emergency & per-class F1 · coverage · "
         "abstention rate · Wilcoxon", C_EVAL, fontsize=7.4)
    _arrow(ax, (0.25, 0.115), (0.38, 0.08))
    _arrow(ax, (0.76, 0.30), (0.62, 0.08), rad=0.1)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG. 3 v2 — missingness sweep incl. TOPSIS
# --------------------------------------------------------------------------- #
def make_fig3_v2(path):
    r = pd.read_csv("../results/results_long.csv")
    r = r[r.variant.isin(["aware", "baseline"])]
    t = pd.read_csv("../results/topsis_long.csv")
    df = pd.concat([r[["seed", "system", "p", "macro_f1"]],
                    t[["seed", "system", "p", "macro_f1"]]], ignore_index=True)
    systems = ["DecisionTree", "Bagging", "AdaBoost", "RUSBoost", "XGBoost",
               "LogReg", "MLP", "PolicyLOCF", "TopsisLOCF"]
    ps = sorted(df.p.unique())
    plt.figure(figsize=(7.5, 5.5))
    for s in systems:
        g = df[df.system == s].groupby("p")["macro_f1"]
        mean, std = g.mean().reindex(ps), g.std().reindex(ps)
        if s == "PolicyLOCF":
            style = dict(marker="s", linewidth=2, linestyle="--", color="black")
        elif s == "TopsisLOCF":
            style = dict(marker="D", linewidth=2, linestyle="--", color="#555555")
        elif s in ("LogReg", "MLP"):
            style = dict(marker="^", linewidth=2, linestyle=":")
        else:
            style = dict(marker="o", linewidth=2)
        plt.plot(ps, mean.values, label=s, **style)
        plt.fill_between(ps, (mean - std).values, (mean + std).values, alpha=0.10)
    plt.xlabel("QoS missingness rate  $p$")
    plt.ylabel("macro-F1")
    plt.title("Robustness under increasing QoS missingness (mean $\\pm$ std, 8 seeds)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# FIG. 7 — staleness sweep
# --------------------------------------------------------------------------- #
def make_fig7(path):
    df = pd.read_csv("../results/results_v2/staleness_raw.csv")
    order = ["AgeAware", "NoAgeFeat", "StaleBlind", "PolicyStale"]
    styles = {
        "AgeAware":   dict(marker="o", linewidth=2.2, color="#2ca02c"),
        "NoAgeFeat":  dict(marker="v", linewidth=1.8, color="#1f77b4", linestyle="-."),
        "StaleBlind": dict(marker="^", linewidth=2, color="#d62728", linestyle=":"),
        "PolicyStale": dict(marker="s", linewidth=2, color="black", linestyle="--"),
    }
    labels = {"AgeAware": "AgeAware (proposed)", "NoAgeFeat": "NoAgeFeat (Abl. C1)",
              "StaleBlind": "StaleBlind (Abl. C2)", "PolicyStale": "PolicyStale"}
    ages = df.groupby("r")["mean_age"].mean().sort_index(ascending=False)
    rs = ages.index.tolist()  # descending probe rate = increasing age
    x = ages.values
    plt.figure(figsize=(7.0, 5.0))
    for s in order:
        g = df[df.system == s].groupby("r")["macroF1"]
        mean = g.mean().reindex(rs).values
        std = g.std().reindex(rs).values
        plt.plot(x, mean, label=labels[s], **styles[s])
        plt.fill_between(x, mean - std, mean + std, alpha=0.12,
                         color=styles[s]["color"])
    plt.xlabel("Mean age of information $\\bar{\\tau}$ (decision epochs)")
    plt.ylabel("macro-F1")
    plt.title("Robustness under measurement staleness (mean $\\pm$ std, 8 seeds)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# FIG. 8 — abstention: deferral rate & committed emergency error vs p
# --------------------------------------------------------------------------- #
def make_fig8(path):
    df = pd.read_csv("../results/results_v2/abstention_raw.csv")
    df = df[df.regime == "MCAR"]
    ps = sorted(df.level.unique())
    g = df.groupby("level")
    fig, ax1 = plt.subplots(figsize=(7.0, 4.6))
    # left axis: emergency error before/after
    eb_m, eb_s = g["errEM_base"].mean().reindex(ps), g["errEM_base"].std().reindex(ps)
    es_m, es_s = g["errEM_sel"].mean().reindex(ps), g["errEM_sel"].std().reindex(ps)
    ax1.plot(ps, eb_m, marker="o", color="#d62728", linewidth=2,
             label="Emergency error (all decisions)")
    ax1.fill_between(ps, eb_m - eb_s, eb_m + eb_s, alpha=0.12, color="#d62728")
    ax1.plot(ps, es_m, marker="o", color="#2ca02c", linewidth=2,
             label="Emergency error (committed only)")
    ax1.fill_between(ps, es_m - es_s, es_m + es_s, alpha=0.12, color="#2ca02c")
    ax1.set_xlabel("QoS missingness rate  $p$")
    ax1.set_ylabel("Emergency-scenario error rate")
    ax1.grid(True, alpha=0.3)
    # right axis: abstention rate
    ax2 = ax1.twinx()
    ab_m, ab_s = g["abst"].mean().reindex(ps), g["abst"].std().reindex(ps)
    ax2.bar(ps, ab_m.values, width=0.045, alpha=0.25, color="#1f77b4",
            label="Abstention rate", zorder=0)
    ax2.set_ylabel("Abstention (deferral) rate")
    ax2.set_ylim(0, 0.35)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax1.set_title("Conformal abstention ($\\alpha=0.05$, mean $\\pm$ std, 8 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    make_fig1_v2(f"{OUT}/fig1_dataflow.png")
    make_fig2(f"{OUT}/fig2_qos_vs_distance.png")
    make_fig3_v2(f"{OUT}/fig_degradation.png")
    make_fig7(f"{OUT}/fig7_staleness.png")
    make_fig8(f"{OUT}/fig8_abstention.png")
    print("saved fig1, fig2, fig3, fig7, fig8")
