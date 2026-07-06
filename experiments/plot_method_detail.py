"""
Method deep-dive (Figure 2). Three panels showing the actual mechanism:
  (A) the behavior-aligned spatial filter phi: per-channel correlation between high-gamma
      envelope and one finger's flexion -> a focal cortical spatial pattern (one subject).
  (B) WHY the shared space transfers: the SAME behavior-defined axis (index finger) tracks
      the finger's flexion in TWO different subjects with non-corresponding grids -> the axis
      means the same thing in both.
  (C) the shared TCN decoder: dilated causal convolutions (dilations 1..64) and the receptive
      field that grows to ~1 s without downsampling.

    python3 -m experiments.plot_method_detail   # writes results/interp/method_detail.png
"""
import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import core.run_xsubject_hd as H
from core.bttn_finger import set_preset

DATA = os.path.expanduser("~/data/stanford_fingerflex/{c}/{c}_fingerflex.mat")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "interp")
HG = slice(12, 24)          # upper-half spectral bands ~ high-gamma motor range
FINGER = 1                  # index finger
FCOL = "#1f77b4"
plt.rcParams.update({"font.size": 11})


def locs(c):
    return np.asarray(sio.loadmat(DATA.format(c=c))['locs'], float)


def feats(c):
    """full spectro-spatial features (T, ch, band) and downsampled flexion (T,5)."""
    x, y = H.load(c)
    F = H.spec(H.norm(x))                         # (T, ch, band)
    Y = H.ds(y)
    n = min(len(F), len(Y))
    return F[:n], Y[:n]


def corr_w(M, yf):
    """per-column corr(M[:,j], yf) - the behavior-aligned weights."""
    Mz = (M - M.mean(0)) / (M.std(0) + 1e-9)
    yz = (yf - yf.mean()) / (yf.std() + 1e-9)
    return (Mz * yz[:, None]).mean(0)


def run():
    set_preset('full')
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = GridSpec(2, 2, height_ratios=[1.05, 0.85], hspace=0.42, wspace=0.24,
                  left=0.06, right=0.97, top=0.9, bottom=0.08)

    # ===== Panel A: behavior-aligned spatial filter (subject zt, index finger) =====
    axA = fig.add_subplot(gs[0, 0])
    Ff, Y = feats("zt"); tr = int(0.8 * len(Ff))
    hg = Ff[:, :, HG].mean(2)                                  # per-channel high-gamma power
    w = corr_w(hg[:tr], Y[:tr, FINGER]); L = locs("zt")[:Ff.shape[1]]
    sc = axA.scatter(L[:, 1], L[:, 2], c=w, s=130, cmap="RdBu_r",
                     vmin=-np.abs(w).max(), vmax=np.abs(w).max(), edgecolors="k", linewidths=0.5)
    axA.set_aspect("equal"); axA.axis("off")
    axA.set_title("(A) Behavior-aligned filter $\\varphi$ — subject zt, index finger\n"
                  "channel correlation(high-$\\gamma$, flexion) on the grid", fontsize=11.5)
    cb = fig.colorbar(sc, ax=axA, fraction=0.045, pad=0.02); cb.set_label("corr.", fontsize=10)
    axA.text(0.5, -0.06, "focal sensorimotor weighting, not the whole grid",
             transform=axA.transAxes, ha="center", va="top", fontsize=10, style="italic", color="#555")

    # ===== Panel B: same axis tracks the finger in two different subjects =====
    axB = fig.add_subplot(gs[0, 1])
    WL = 700                                                   # 7 s display window
    for off, (c, lbl) in zip([1.4, -1.4], [("zt", "subject zt"), ("mv", "subject mv")]):
        Ff, Y = feats(c); tr = int(0.8 * len(Ff))
        M = Ff.reshape(len(Ff), -1)                            # (T, ch*band) spectro-spatial
        w = corr_w(M[:tr], Y[:tr, FINGER])
        comp = gaussian_filter1d(M @ w, 8)                    # rank-1 aligned component, smoothed
        yf = gaussian_filter1d(Y[:, FINGER].astype(float), 8)
        r_all = float(np.corrcoef(comp[tr:], yf[tr:])[0, 1])  # honest held-out correlation
        # show a REPRESENTATIVE window (window-r closest to overall r, with real movement)
        cand = [(i, np.corrcoef(comp[i:i + WL], yf[i:i + WL])[0, 1], yf[i:i + WL].std())
                for i in range(tr, len(yf) - WL, 50)]
        cand = [x for x in cand if x[2] > np.median([z[2] for z in cand])]
        st = min(cand, key=lambda x: abs(x[1] - r_all))[0]; win = slice(st, st + WL)
        cz = (comp[win] - comp[win].mean()) / (comp[win].std() + 1e-9)
        yz = (yf[win] - yf[win].mean()) / (yf[win].std() + 1e-9)
        t = np.arange(WL) / 100.0
        axB.fill_between(t, yz * 0.9 + off, off, color="#bbb", alpha=0.6,
                         label="finger flexion" if off > 0 else None, lw=0)
        axB.plot(t, cz * 0.9 + off, color=FCOL, lw=1.8,
                 label="aligned component" if off > 0 else None)
        axB.text(t[-1] + 0.12, off, f"{lbl}\n$r={r_all:.2f}$", va="center", fontsize=10.5, weight="bold")
    axB.set_xlabel("time (s)"); axB.set_yticks([]); axB.set_xlim(0, t[-1] + 1.6)
    axB.legend(loc="upper left", fontsize=9.5, framealpha=0.9, ncol=2)
    axB.set_title("(B) The shared axis is subject-invariant\n"
                  "the index-finger component tracks flexion in both grids", fontsize=11.5)
    axB.text(0.5, -0.18, "a single behavior-aligned component (held-out $r$); the decoder stacks many",
             transform=axB.transAxes, ha="center", va="top", fontsize=10, style="italic", color="#555")
    for s in ("top", "right", "left"): axB.spines[s].set_visible(False)

    # ===== Panel C: TCN dilated causal convolution schematic =====
    axC = fig.add_subplot(gs[1, :])
    dil = [1, 2, 4, 8, 16, 32, 64]; nT = 33
    ys = list(range(len(dil) + 1))
    for lv in ys:
        axC.scatter(range(nT), [lv] * nT, s=16, color="#999", zorder=1)
    axC.text(-2.2, 0, "input", va="center", ha="right", fontsize=9.5, color="#333")
    out = nT - 1
    node = out
    for lv, d in enumerate(dil):
        for src in (node, max(0, node - d)):
            axC.plot([src, node], [lv, lv + 1], color=FCOL, lw=1.3, alpha=0.9, zorder=2)
            axC.scatter([src], [lv], s=34, color=FCOL, zorder=3, edgecolors="k", linewidths=0.3)
        axC.text(nT + 0.6, lv + 0.5, f"dilation {d}", va="center", fontsize=9.5, color="#333")
    axC.scatter([out], [len(dil)], s=60, color="#d62728", zorder=4, edgecolors="k")
    axC.text(out, len(dil) + 0.35, "output $t$", ha="center", fontsize=10, color="#d62728")
    axC.set_xlim(-1, nT + 6); axC.set_ylim(-0.5, len(dil) + 0.9); axC.axis("off")
    axC.text(0.5, -0.12, "causal, length-preserving; receptive field grows to ~1 s without downsampling",
             transform=axC.transAxes, ha="center", va="top", fontsize=10, style="italic", color="#555")
    axC.set_title("(C) Shared TCN decoder $g$ — stacked dilated causal convolutions", fontsize=11.5, loc="left")

    p = os.path.join(OUT, "method_detail.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    run()
