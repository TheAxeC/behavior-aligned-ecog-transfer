"""
Calibration-amount curve for the ward paper: cross-subject accuracy vs MINUTES of target
calibration data. Best config (full features, R=4, width=128). For each held-out target, the
shared decoder is donor-pretrained then fine-tuned on increasing slices of the target's own
training data; we report mean-over-fingers r vs calibration minutes. The practical hook +
the head-to-head with CORTEG's 10-30 min calibration.

Run: python3 -m experiments.run_calib_curve   (from ~/projects/ward/code)
"""
import os, time, csv
import numpy as np

from core.run_xsubject_hd import load, spatial_filters, SUBJECTS, PRESETS
from core.seq_decoder import aligned_seq, split, calib_curve_ens, mcorr
from core.bttn_finger import set_preset
import core.run_xsubject_hd as H

PRESET, R, WIDTH, N_ENS = 'full', 4, 128, 3
FRACS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run():
    set_preset(PRESET); dt = 1.0 / H.DS          # seconds per frame
    print(f"# ward calib-curve: {PRESET} R={R} w={WIDTH} fracs={FRACS} (DS={H.DS}Hz)", flush=True)
    al = {}
    for c in SUBJECTS:
        x, y = load(c); te = int(0.8 * len(x)); Wf = spatial_filters(x[:te], y[:te], R); al[c] = aligned_seq(c, Wf)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "ward_calib_curve.csv"); rows = []
    for tgt in SUBJECTS:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt]); t0 = time.time()
        donors = [c for c in SUBJECTS if c != tgt]
        Xd = [al[d][0] for d in donors]; Yd = [al[d][1] for d in donors]
        preds = calib_curve_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, FRACS, n_ens=N_ENS, width=WIDTH)
        for f in FRACS:
            mins = max(SEG_MIN(Xf, f, dt), 0)
            r = mcorr(Yt.T, preds[f])
            rows.append(dict(target=tgt, frac=f, minutes=round(mins, 2), xsub=round(r, 4)))
            print(f"  {tgt} f={f:.2f} ({mins:.2f} min): xsub={r:.4f}", flush=True)
        print(f"  [{tgt} done {time.time()-t0:.0f}s]", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # mean curve over subjects
    print("# MEAN curve (minutes ~ xsub):", flush=True)
    for f in FRACS:
        rs = [r['xsub'] for r in rows if r['frac'] == f]; ms = [r['minutes'] for r in rows if r['frac'] == f]
        print(f"#   f={f}: ~{np.mean(ms):.2f} min -> xsub {np.mean(rs):.4f}", flush=True)
    print(f"# -> {out}", flush=True)


def SEG_MIN(Xf, f, dt):
    from core.seq_decoder import SEG
    k = max(SEG, int(f * Xf.shape[1])); return k * dt / 60.0


if __name__ == "__main__":
    run()
