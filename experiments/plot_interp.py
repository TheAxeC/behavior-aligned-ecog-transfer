"""
Interpretability figure for the ward paper: the PRIVATE per-subject finger-aligned spatial
filters (the behavioral alignment) plotted on each subject's cortex. Shows WHICH electrodes
map onto each finger's shared component - i.e. the decoder's glass-box spatial map, the
differentiator vs a black-box foundation model. We expect sensorimotor-cortex localization
and finger somatotopy, consistent across subjects (why the shared component space transfers).

    python3 -m experiments.plot_interp        # writes results/interp/finger_spatial_maps.png
"""
import os
import numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.run_xsubject_hd import load, spatial_filters
import core.run_xsubject_hd as H
from core.bttn_finger import set_preset

SUBJECTS = ['zt', 'mv', 'jc', 'ht']            # representative (2 strong + a hard one)
FINGERS = ['thumb', 'index', 'middle', 'ring', 'little']
DATA = os.path.expanduser("~/data/stanford_fingerflex/{c}/{c}_fingerflex.mat")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "interp")


def locs_of(c):
    return np.asarray(sio.loadmat(DATA.format(c=c))['locs'], float)   # (ch, 3) MNI


def brain_of(c):
    b = sio.loadmat(DATA.format(c=c))['brain'][0, 0]
    return np.asarray(b['vert'], float)                              # (V, 3) cortical vertices


def run():
    set_preset('full')
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(len(SUBJECTS), 5, figsize=(13, 2.7 * len(SUBJECTS)))
    for i, c in enumerate(SUBJECTS):
        x, y = load(c); te = int(0.8 * len(x))
        W = spatial_filters(x[:te], y[:te], r=1)        # (ch, 5) one corr pattern per finger
        L = locs_of(c)[:W.shape[0]]; V = brain_of(c)
        for f in range(5):
            ax = axes[i, f]
            # lateral (sagittal) view: posterior-anterior (y) vs inferior-superior (z)
            ax.scatter(V[:, 1], V[:, 2], s=0.5, c="#e8e8e8", linewidths=0, rasterized=True)
            w = W[:, f]; vmax = np.abs(w).max() + 1e-9
            sc = ax.scatter(L[:, 1], L[:, 2], c=w, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            s=42, edgecolors="k", linewidths=0.4)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if i == 0: ax.set_title(FINGERS[f], fontsize=11)
            if f == 0: ax.set_ylabel(c, fontsize=12, rotation=0, ha="right", va="center")
    fig.suptitle("Private finger-aligned spatial filters on cortex (lateral view) — the behavioral alignment",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "finger_spatial_maps.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    run()
