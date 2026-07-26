"""
make_figures.py — generate Fig. 1 (data-flow schematic) and Fig. 2 (QoS vs distance).
Fig. 2 curves are computed directly from the romacs_datagen physical model.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import romacs_datagen as dg
from romacs_datagen import CHANNELS, CHANNEL_NAMES, channel_qos

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})

# ----------------------------------------------------------------------------- #
# FIG. 2 — QoS vs distance (mean +/- std over shadowing), per channel
# ----------------------------------------------------------------------------- #
def make_fig2(path):
    distances = np.linspace(1, 80, 60)
    sea_state, weather, traffic = 2.0, 0.2, 0.3
    n_mc = 300  # Monte-Carlo draws per distance to average out shadowing
    metrics = ["rssi_dbm", "sinr_db", "per", "throughput_kbps"]
    titles = ["(a) RSSI", "(b) SINR", "(c) PER", "(d) Throughput"]
    ylabels = ["RSSI (dBm)", "SINR (dB)", "PER", "Throughput (kbps)"]

    # colors/markers chosen to remain distinguishable in grayscale
    styles = {
        "VHF_DSC":   dict(color="#1f77b4", ls="-"),
        "dPMR":      dict(color="#ff7f0e", ls="--"),
        "AIS_VDES":  dict(color="#2ca02c", ls="-."),
        "LTE_5G":    dict(color="#d62728", ls=":"),
        "SATELLITE": dict(color="#9467bd", ls="-"),
    }
    labels = {"VHF_DSC": "VHF-DSC", "dPMR": "dPMR", "AIS_VDES": "AIS/VDES",
              "LTE_5G": "LTE/5G", "SATELLITE": "Satellite"}

    data = {m: {c: (np.zeros_like(distances), np.zeros_like(distances)) for c in CHANNEL_NAMES}
            for m in metrics}
    for c in CHANNEL_NAMES:
        spec = CHANNELS[c]
        for j, d in enumerate(distances):
            samples = {m: [] for m in metrics}
            rng = np.random.default_rng(1000 + j)
            for _ in range(n_mc):
                q = channel_qos(spec, d, sea_state, weather, traffic, rng)
                for m in metrics:
                    samples[m].append(q[m])
            for m in metrics:
                arr = np.array(samples[m])
                data[m][c][0][j] = arr.mean()
                data[m][c][1][j] = arr.std()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, m, title, ylab in zip(axes.ravel(), metrics, titles, ylabels):
        for c in CHANNEL_NAMES:
            mean, std = data[m][c]
            ax.plot(distances, mean, label=labels[c], linewidth=1.8, **styles[c])
            ax.fill_between(distances, mean - std, mean + std, alpha=0.10,
                            color=styles[c]["color"])
        ax.set_title(title)
        ax.set_xlabel("Distance to shore (km)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        if m == "throughput_kbps":
            ax.set_yscale("symlog")
    axes.ravel()[0].legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# FIG. 1 — data-flow schematic
# ----------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, ec="#333333", fontsize=8.5, bold=False):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, weight="bold" if bold else "normal",
            linespacing=1.3)
    return (x, y, w, h)


def _arrow(ax, p_from, p_to, text=None, color="#333333", rad=0.0):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.3, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)
    if text:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx, my + 0.018, text, ha="center", va="bottom", fontsize=7.2,
                color=color, style="italic")


def make_fig1(path):
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C_GEN = "#dbe9f6"; C_ORACLE = "#fde9d0"; C_MISS = "#f6dbdb"
    C_LEARN = "#dff0db"; C_POLICY = "#efe0f4"; C_EVAL = "#eeeeee"

    # Top row: generation pipeline
    b1 = _box(ax, (0.02, 0.72), 0.20, 0.20,
              "Trajectory\ngeneration\n(vessel motion,\nenv. random walk)", C_GEN)
    b2 = _box(ax, (0.28, 0.72), 0.20, 0.20,
              "Physical QoS\nmodel (complete)\n$d\\!\\to\\!$PL$\\to$RSSI\n$\\to$SINR$\\to$PER,$T$", C_GEN)
    b3 = _box(ax, (0.54, 0.72), 0.20, 0.20,
              "Oracle labeling\n(utility policy +\nNO_CHANNEL)", C_ORACLE)
    _arrow(ax, (0.22, 0.82), (0.28, 0.82))
    _arrow(ax, (0.48, 0.82), (0.54, 0.82))

    # Complete (x, y) -> feature/label store
    b4 = _box(ax, (0.80, 0.72), 0.18, 0.20,
              "Labeled\nbenchmark\n$(\\mathbf{x}, y)$", C_GEN)
    _arrow(ax, (0.74, 0.82), (0.80, 0.82))

    # Down to missingness
    b5 = _box(ax, (0.40, 0.44), 0.30, 0.16,
              "MCAR missingness injection\n(QoS only, rate $p$)", C_MISS)
    _arrow(ax, (0.89, 0.72), (0.89, 0.52), rad=0.0)
    _arrow(ax, (0.89, 0.52), (0.70, 0.52))

    # Two branches: learner and policy
    b6 = _box(ax, (0.05, 0.14), 0.40, 0.20,
              "Learner path\nimpute + missingness indicators (52-D)\n"
              "$\\to$ tree ensembles (DT, Bagging,\nAdaBoost, RUSBoost, XGBoost)", C_LEARN)
    b7 = _box(ax, (0.55, 0.14), 0.40, 0.20,
              "Policy baseline\nLOCF imputation along trajectory\n"
              "$\\to$ rule-based selection\n(cold-start channels excluded)", C_POLICY)
    _arrow(ax, (0.48, 0.44), (0.25, 0.34), text="train/test", rad=-0.15)
    _arrow(ax, (0.62, 0.44), (0.75, 0.34), rad=0.15)

    # Evaluation
    b8 = _box(ax, (0.25, 0.005), 0.50, 0.085,
              "Evaluation:  macro-F1 · balanced acc. · per-class & emergency F1 · "
              "robustness (AUDC) · Wilcoxon", C_EVAL, fontsize=7.6)
    _arrow(ax, (0.25, 0.14), (0.42, 0.09), rad=0.0)
    _arrow(ax, (0.75, 0.14), (0.58, 0.09), rad=0.0)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig2("../figures/fig2_qos_vs_distance.png")
    make_fig1("../figures/fig1_dataflow.png")
    print("saved fig1 and fig2")
