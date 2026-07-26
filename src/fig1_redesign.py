"""Fig. 1 redesign — academic dataflow schematic with guaranteed text fit.

Design rules:
  - grid layout, generous padding, orthogonal arrows
  - every label is measured with the actual renderer and the font size is
    reduced until the text fits inside its box with >= 6% padding
  - no floating italic labels overlapping boxes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIT_LOG = []
plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "mathtext.fontset": "dejavuserif",
})

FIGW, FIGH = 7.4, 5.6

def box_fit(ax, fig, xy, w, h, text, fc, fontsize=9.0, bold_first=False, ec="#333333"):
    """Rounded box + centered text; shrink font until text bbox fits box."""
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.008,rounding_size=0.015",
                           linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(patch)
    fs = fontsize
    while fs > 5.0:
        t = ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fs, zorder=3, linespacing=1.35)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        bb_ax = bb.transformed(ax.transData.inverted())
        tw, th = bb_ax.width, bb_ax.height
        if tw <= 0.94 * w and th <= 0.90 * h:
            FIT_LOG.append((text.split("\n")[0][:30], round(fs,2), round(tw/w,2), round(th/h,2)))
            break
        t.remove()
        fs -= 0.25
    else:
        FIT_LOG.append((text.split("\n")[0][:30], "OVERFLOW", round(tw/w,2), round(th/h,2)))
    return (x, y, w, h)

def arrow(ax, p_from, p_to, rad=0.0, color="#333333", lw=1.4):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=12,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)

def elbow(ax, pts, color="#333333", lw=1.4):
    """Orthogonal polyline with arrowhead on the last segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, linewidth=lw, zorder=1, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)

def make_fig1(path):
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C_GEN = "#dbe9f6"; C_ORACLE = "#fde9d0"; C_DEG = "#f6dbdb"
    C_LEARN = "#dff0db"; C_POLICY = "#efe0f4"; C_EVAL = "#eeeeee"; C_CONF = "#fff3c4"

    # ---------------- Row 1: benchmark generation (y: 0.80-0.97) ----------
    y1, h1 = 0.80, 0.17
    b_gen = box_fit(ax, fig, (0.015, y1), 0.215, h1,
        "Trajectory\ngeneration\n(vessel motion,\nenv. random walk)", C_GEN, fontsize=8.0)
    b_qos = box_fit(ax, fig, (0.265, y1), 0.215, h1,
        "Physical QoS\nmodel (complete)\n$d\\to$PL$\\to$RSSI\n$\\to$SINR$\\to$PER, $T$", C_GEN, fontsize=8.0)
    b_orc = box_fit(ax, fig, (0.515, y1), 0.215, h1,
        "Oracle labeling\n(utility policy\n$+$ NO_CHANNEL)", C_ORACLE, fontsize=8.0)
    b_ben = box_fit(ax, fig, (0.765, y1), 0.215, h1,
        "Labeled\nbenchmark\n$(\\mathbf{x},\\,y)$", C_GEN, fontsize=8.0)
    ym = y1 + h1/2
    arrow(ax, (0.230, ym), (0.265, ym))
    arrow(ax, (0.480, ym), (0.515, ym))
    arrow(ax, (0.730, ym), (0.765, ym))

    # ---------------- Row 2: degradation injection (y: 0.565-0.70) --------
    y2, h2 = 0.565, 0.135
    b_deg = box_fit(ax, fig, (0.25, y2), 0.50, h2,
        "Degradation injection (QoS only, train/test)\n"
        "missingness: MCAR/MAR at rate $p$\n"
        "staleness: probing at rate $r$ $\\Rightarrow$ AoI $\\boldsymbol{\\tau}$", C_DEG, fontsize=8.0)
    # benchmark -> down -> left -> into degradation box (orthogonal)
    elbow(ax, [(0.8725, y1), (0.8725, y2 + h2/2), (0.75, y2 + h2/2)])

    # ---------------- Row 3: two branches (y: 0.30-0.475) -----------------
    y3, h3 = 0.30, 0.175
    b_lea = box_fit(ax, fig, (0.015, y3), 0.475, h3,
        "Learner path\nimpute$+$indicate (52-D) or AoI repr.\n"
        "$\\to$ tree-ensemble core (XGBoost default)\n"
        "$\\to$ class probabilities $\\hat{\\pi}(\\mathbf{x})$", C_LEARN, fontsize=8.0)
    b_pol = box_fit(ax, fig, (0.535, y3), 0.45, h3,
        "Non-learning comparators\nLOCF along trajectory\n"
        "$\\to$ utility policy or TOPSIS ranking\n(cold-start channels excluded)", C_POLICY, fontsize=8.0)
    # degradation -> both branches (orthogonal)
    elbow(ax, [(0.375, y2), (0.375, y2 - 0.045), (0.2525, y2 - 0.045), (0.2525, y3 + h3)])
    elbow(ax, [(0.625, y2), (0.625, y2 - 0.045), (0.760, y2 - 0.045), (0.760, y3 + h3)])

    # ---------------- Row 4: conformal abstention (y: 0.115-0.245) --------
    y4, h4 = 0.125, 0.14
    b_cnf = box_fit(ax, fig, (0.015, y4), 0.475, h4,
        "Mondrian conformal abstention (level $\\alpha$)\n"
        "$|\\Gamma(\\mathbf{x})|{=}1$: commit to channel\n"
        "else: defer to operator with short-list $\\Gamma(\\mathbf{x})$", C_CONF, fontsize=8.0)
    arrow(ax, (0.2525, y3), (0.2525, y4 + h4))

    # ---------------- Row 5: evaluation (y: 0.0-0.06) ---------------------
    y5, h5 = 0.0, 0.09
    b_eval = box_fit(ax, fig, (0.10, y5), 0.80, h5,
        "Evaluation:  macro-F1 $\\cdot$ AUDC $\\cdot$ emergency/per-class F1\n"
        "coverage $\\cdot$ abstention rate $\\cdot$ Wilcoxon signed-rank", C_EVAL,
        fontsize=7.5)
    arrow(ax, (0.2525, y4), (0.2525, y5 + h5))
    elbow(ax, [(0.760, y3), (0.760, y5 + h5)])

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_fig1("../figures/fig1_dataflow.png")
    print("saved")
    for row in FIT_LOG:
        print(row)
