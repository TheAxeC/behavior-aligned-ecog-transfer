"""
Method-overview schematic (Figure 1). Pipeline, left to right:
heterogeneous (non-corresponding) electrode grids -> PRIVATE per-subject behavior-aligned
spatial filter phi_i -> SHARED finger-motor component space -> SHARED TCN decoder g ->
continuous 5-finger trajectories.  Message: only phi_i is subject-specific; g is shared,
so f_i = g . phi_i and the alignment crosses the grid gap.

    python3 -m experiments.plot_method   # writes results/interp/method_overview.png
"""
import os
import numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = os.path.expanduser("~/data/stanford_fingerflex/{c}/{c}_fingerflex.mat")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "interp")
FCOL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
FING = ["thumb", "index", "middle", "ring", "little"]
plt.rcParams.update({"font.size": 12})


def grid_xy(c):
    L = np.asarray(sio.loadmat(DATA.format(c=c))['locs'], float)
    return L[:, 1], L[:, 2]


def flex_snip(c, n=320):
    y = np.asarray(sio.loadmat(DATA.format(c=c))['flex'], float)[::40]
    s = 4000 // 40
    return y[s:s + n]


def box(ax, x, y, w, h, text, fc, fs=12, ec="#333", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, weight=weight)


def arrow(ax, x0, y0, x1, y1, lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22,
                                 lw=lw, color="#3b3b3b", zorder=1,
                                 shrinkA=0, shrinkB=0))


def run():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(15, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    cy = 0.56   # vertical center of the flow
    rows = [0.78, 0.56, 0.34]
    hy = 0.95   # header row
    for x, t in [(0.10, "Non-corresponding grids"), (0.325, "Private alignment"),
                 (0.545, "Shared component space"), (0.745, "Shared decoder"),
                 (0.925, "Output")]:
        ax.text(x, hy, t, ha="center", fontsize=13.5, weight="bold", color="#1a1a1a")

    # ---- Stage 1: heterogeneous grids ----
    subs = list(zip(["zt", "wc", "mv"], rows, ["donor", "donor", "target"]))
    for c, yc, role in subs:
        gx, gy = grid_xy(c)
        a = fig.add_axes([0.025, yc - 0.085, 0.15, 0.17]); a.axis("off")
        col = "#3b6fb0" if role == "donor" else "#c0392b"
        a.scatter(gx, gy, s=13, c=col, edgecolors="k", linewidths=0.4)
        a.set_title(f"subject {c} ({role})", fontsize=11, pad=2)
        a.set_aspect("equal")
        arrow(ax, 0.18, yc, 0.245, yc)

    # ---- Stage 2: private behavior-aligned filter (one box per subject) ----
    for yc in rows:
        box(ax, 0.245, yc - 0.065, 0.16, 0.13,
            r"$\varphi_i$" + "\nbehavior-aligned\nspatial filter", "#fbe2cf", fs=11.5)
        arrow(ax, 0.405, yc, 0.475, cy)

    # ---- Stage 3: shared finger-motor component space ----
    box(ax, 0.475, 0.305, 0.165, 0.51, "", "#e6f0fa")
    for i, (col, nm) in enumerate(zip(FCOL, FING)):
        yy = 0.355 + i * 0.092
        ax.add_patch(plt.Circle((0.512, yy), 0.013, color=col, ec="k", lw=0.5, zorder=3))
        ax.text(0.54, yy, f"{nm} axes", ha="left", va="center", fontsize=11,
                color="#1a1a1a", zorder=3)
    arrow(ax, 0.64, cy, 0.705, cy)

    # ---- Stage 4: shared TCN decoder ----
    box(ax, 0.705, cy - 0.115, 0.155, 0.23,
        r"shared TCN" + "\n" + r"decoder $g$" + "\n(dilated causal\nconvs, donors)",
        "#dcefe2", fs=11.5)
    arrow(ax, 0.86, cy, 0.915, cy)

    # ---- Stage 5: predicted finger trajectories ----
    a = fig.add_axes([0.905, cy - 0.21, 0.085, 0.42]); a.axis("off")
    Y = flex_snip("zt"); Y = (Y - Y.min(0)) / (np.ptp(Y, axis=0) + 1e-9)
    for i in range(5):
        a.plot(Y[:, i] * 0.85 + i, color=FCOL[i], lw=1.4)
    a.text(0.5, -0.06, "5-finger\ntrajectories", transform=a.transAxes, ha="center",
           va="top", fontsize=11)

    # ---- bottom banner (two lines, clear of the flow) ----
    ax.add_patch(FancyBboxPatch((0.04, 0.025), 0.92, 0.115,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="#f4f4f4", ec="#aaa", lw=1.2))
    ax.text(0.5, 0.097,
            r"$f_i = g \circ \varphi_i$ — only the alignment $\varphi_i$ is subject-specific; "
            r"the decoder $g$ is shared.", ha="center", va="center", fontsize=12.5)
    ax.text(0.5, 0.057,
            "Behavior defines the component axes, so they mean the same thing across grids "
            "that share no geometry.", ha="center", va="center", fontsize=12.5)

    p = os.path.join(OUT, "method_overview.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    run()
