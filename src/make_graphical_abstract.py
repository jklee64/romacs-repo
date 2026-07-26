"""
make_graphical_abstract.py — IEEE-spec graphical abstract for the RoMaCS manuscript.

IEEE specification: 660 x 295 px, landscape, < 45 KB, named 'gagraphic'.

The layout is deliberately overlap-proof: each curve is labelled at its own right-hand
endpoint (distinct y values 0.90 / 0.55 / 0.30), so no annotation can collide with
another curve or with the axes. Values are the measured 8-seed means of Table 3.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

P   = [0.0, 0.10, 0.25, 0.50, 0.75]
XGB = [0.978, 0.973, 0.965, 0.944, 0.902]
POL = [1.000, 0.917, 0.823, 0.708, 0.548]
TOP = [0.566, 0.507, 0.439, 0.369, 0.304]

CH  = ["VHF-DSC", "dPMR", "AIS/VDES", "LTE/5G", "Satellite"]
OBS = ["-58 dBm", "?", "-63 dBm", "?", "-96 dBm"]

W, H, DPI = 660, 295, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor("white")

def fit_text(fig, txt, limit_px, min_size=5.5):
    """Shrink a Text artist until its rendered width fits limit_px."""
    fig.canvas.draw()
    while txt.get_window_extent().width > limit_px and txt.get_fontsize() > min_size:
        txt.set_fontsize(txt.get_fontsize() - 0.2)
        fig.canvas.draw()
    return txt.get_fontsize(), txt.get_window_extent().width


# ------------------------------------------------------------ LEFT: the problem
axL = fig.add_axes([0.012, 0.20, 0.325, 0.74])
axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
_title = axL.text(0.5, 1.00, "Incomplete QoS readings",
                  ha="center", va="top", fontsize=9.6, fontweight="bold",
                  color="#1a1a1a")
_fs, _wpx = fit_text(fig, _title, axL.get_window_extent().width * 0.98)
print(f"left title: fontsize={_fs:.1f}  text={_wpx:.0f}px  "
      f"limit={axL.get_window_extent().width*0.98:.0f}px")

y0, dy = 0.74, 0.165
for i, (c, v) in enumerate(zip(CH, OBS)):
    y = y0 - i * dy
    miss = (v == "?")
    axL.add_patch(FancyBboxPatch(
        (0.02, y - 0.060), 0.96, 0.120,
        boxstyle="round,pad=0.005,rounding_size=0.02", linewidth=1.0,
        edgecolor="#c0392b" if miss else "#8a9aa5",
        facecolor="#fdecea" if miss else "#eef2f5"))
    axL.text(0.08, y, c, ha="left", va="center", fontsize=8.4, color="#1a1a1a")
    axL.text(0.94, y, v, ha="right", va="center",
             fontsize=11.0 if miss else 8.4,
             fontweight="bold" if miss else "normal",
             color="#c0392b" if miss else "#2c3e50")

# ------------------------------------------------------------- RIGHT: the result
axR = fig.add_axes([0.395, 0.30, 0.375, 0.62])
axR.plot(P, XGB, color="#6a3d9a", lw=2.6, marker="o", ms=4.2, zorder=3, clip_on=False)
axR.plot(P, POL, color="#111111", lw=2.2, ls="--", marker="s", ms=3.8, zorder=2, clip_on=False)
axR.plot(P, TOP, color="#5d6d7e", lw=2.2, ls="-.", marker="D", ms=3.6, zorder=2, clip_on=False)

axR.set_xlim(0, 0.75); axR.set_ylim(0.24, 1.04)
axR.set_xticks([0, 0.25, 0.50, 0.75])
axR.set_xticklabels(["0", "25", "50", "75"], fontsize=8.0)
axR.set_yticks([0.4, 0.6, 0.8, 1.0])
axR.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=8.0)
axR.set_xlabel("QoS values missing (%)", fontsize=8.6, labelpad=1.0)
axR.set_ylabel("macro-F1", fontsize=8.6, labelpad=1.0)
axR.grid(True, alpha=0.22, lw=0.6)
for s in axR.spines.values():
    s.set_linewidth(0.8)

# End labels sit in offset space to the right of each curve's final point.
# The three endpoints are 0.902 / 0.548 / 0.304, so collision is impossible.
for yval, name, delta, colour in (
        (XGB[-1], "learned", "-8%",  "#6a3d9a"),
        (POL[-1], "policy",  "-45%", "#111111"),
        (TOP[-1], "TOPSIS",  "-46%", "#5d6d7e")):
    axR.annotate(f"{name}\n{delta}", xy=(0.75, yval),
                 xytext=(6, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=8.6, color=colour,
                 fontweight="bold", linespacing=1.15, annotation_clip=False)

# ------------------------------------------------------- BOTTOM: the takeaway
axB = fig.add_axes([0.012, 0.015, 0.976, 0.135])
axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")
box = FancyBboxPatch((0.002, 0.06), 0.996, 0.88,
                     boxstyle="round,pad=0.004,rounding_size=0.06",
                     linewidth=1.1, edgecolor="#6a3d9a", facecolor="#f3eefa")
axB.add_patch(box)

TAKEAWAY = ("At 75% missing QoS: learned selector keeps 92%, "
            "rule-based and MADM keep half")

# Auto-fit: shrink the font until the rendered text fits inside the box with a
# margin. Measured rather than estimated, so the result cannot overflow.
txt = axB.text(0.5, 0.5, TAKEAWAY, ha="center", va="center",
               fontsize=8.6, color="#4a2d6a", fontweight="bold")
box_px = axB.get_window_extent().width * 0.90          # 10% side margin
_fs, _wpx = fit_text(fig, txt, box_px)
print(f"takeaway: fontsize={_fs:.1f}  text={_wpx:.0f}px  limit={box_px:.0f}px")

out = "../results/gagraphic.png"
fig.savefig(out, dpi=DPI, facecolor="white")
plt.close(fig)

print("raw:", os.path.getsize(out), "bytes")
print("size:", Image.open(out).size)
if os.path.getsize(out) > 45000:
    Image.open(out).convert("RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=256).save(out, optimize=True)
    print("optimised ->", os.path.getsize(out), "bytes")

Image.open(out).convert("RGB").resize((1980, 885), Image.LANCZOS)\
     .save("../results/gagraphic_inspect_3x.png")
print("inspection copy written")
