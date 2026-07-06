"""Calibration-curve figure (Figure 4): cross-subject accuracy vs minutes of target calibration.

Reads the committed results/ward_calib_curve.csv (produced by run_calib_curve.py on the cluster), so it
runs locally with no GPU or raw data. Plots the per-subject curves (light) and the across-subject mean
(bold), marks the CORTEG accuracy level (0.554) and where the mean crosses it (about 4.8 minutes).

    python3 -m experiments.plot_calib        # writes results/interp/calibration_curve.png
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "interp")
CORTEG = 0.554


def load_curve():
    """results/ward_calib_curve.csv -> {target: (minutes[], xsub[])} and the fraction grid."""
    rows = []
    with open(os.path.join(RESULTS, "ward_calib_curve.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append((r["target"], float(r["frac"]), float(r["minutes"]), float(r["xsub"])))
    fracs = sorted({f for _, f, _, _ in rows})
    per_subj = {}
    for t, f, m, x in rows:
        per_subj.setdefault(t, {})[f] = (m, x)
    return fracs, per_subj


def main():
    os.makedirs(OUT, exist_ok=True)
    fracs, per_subj = load_curve()
    # across-subject mean minutes + accuracy at each calibration fraction
    mean_min = [np.mean([per_subj[t][f][0] for t in per_subj]) for f in fracs]
    mean_acc = [np.mean([per_subj[t][f][1] for t in per_subj]) for f in fracs]
    # where the mean curve crosses the CORTEG level (linear interpolation between fraction points)
    cross = None
    for i in range(1, len(fracs)):
        a0, a1 = mean_acc[i - 1], mean_acc[i]
        if (a0 - CORTEG) * (a1 - CORTEG) <= 0 and a1 != a0:
            w = (CORTEG - a0) / (a1 - a0)
            cross = mean_min[i - 1] + w * (mean_min[i] - mean_min[i - 1])
            break

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for t in per_subj:
        mm = [per_subj[t][f][0] for f in fracs]
        xx = [per_subj[t][f][1] for f in fracs]
        ax.plot(mm, xx, color="0.75", lw=1, marker="o", ms=2.5, zorder=1)
    ax.plot(mean_min, mean_acc, color="#d1495b", lw=2.4, marker="o", ms=5, label="cross-subject mean", zorder=3)
    ax.axhline(CORTEG, color="#2e4057", ls="--", lw=1.3, label=f"CORTEG level ({CORTEG})")
    if cross is not None:
        ax.axvline(cross, color="#2e4057", ls=":", lw=1)
        ax.annotate(f"crosses at ~{cross:.1f} min", xy=(cross, CORTEG),
                    xytext=(cross + 0.6, CORTEG - 0.09), fontsize=9, color="#2e4057")
    ax.set_xlabel("target calibration (minutes)")
    ax.set_ylabel("cross-subject mean r")
    ax.set_title("Calibration curve: accuracy vs target calibration time", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "calibration_curve.png"), dpi=150)
    plt.close(fig)
    print(f"# wrote {OUT}/calibration_curve.png  (mean crosses CORTEG {CORTEG} at ~{cross:.1f} min)")


if __name__ == "__main__":
    main()
