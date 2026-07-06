"""
Same-protocol BASELINES for the ward paper - isolate the contribution of the behavioral alignment.
Same features/decoder(TCN)/LOSO/eval; only the CROSS-SUBJECT correspondence scheme changes:
  ours        : behavioral finger-aligned spatial filter -> shared 5R component space (the method)
  first43     : naive first-43-channel truncation, NO alignment (the thesis scheme; raw channel
                index has no cross-subject meaning)
  first43_ea  : first-43 + Euclidean Alignment (He & Wu 2020) - generic statistical alignment
Reports cross-subject zero-shot + minimal-calibration(fine-tuned) for each. The discriminating
regime is ZERO-SHOT: does a donor decoder transfer WITHOUT our behavior-defined components?
(External SOTA - FingerFlex / CORTEG - cited from their papers; not re-run.)

Run: python3 -m experiments.run_baselines   (from ~/projects/ward/code)
"""
import os, time, csv
import numpy as np

from core.run_xsubject_hd import load, spatial_filters, SUBJECTS
from core.seq_decoder import (aligned_seq, raw_first43_seq, split,
                              train_predict_ens, xsub_finetune_ens, mcorr)
from core.bttn_finger import set_preset

PRESET, R, WIDTH, N_ENS = 'full', 4, 128, 3
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def build(cond):
    feat = {}
    for c in SUBJECTS:
        if cond == 'ours':
            x, y = load(c); te = int(0.8 * len(x)); Wf = spatial_filters(x[:te], y[:te], R)
            feat[c] = aligned_seq(c, Wf)
        elif cond == 'first43':
            feat[c] = raw_first43_seq(c, align=None)
        elif cond == 'first43_ea':
            feat[c] = raw_first43_seq(c, align='ea')
    return feat


def run():
    set_preset(PRESET)
    print(f"# ward baselines [{PRESET} R={R} w={WIDTH}] - behavioral alignment vs no-align vs EA", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "ward_baselines.csv"); rows = []
    for cond in ['ours', 'first43', 'first43_ea']:
        al = build(cond); t0 = time.time(); zs, ft = [], []
        for tgt in SUBJECTS:
            (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt])
            donors = [c for c in SUBJECTS if c != tgt]
            Xd = [al[d][0] for d in donors]; Yd = [al[d][1] for d in donors]
            pz, _ = train_predict_ens(Xd, Yd, Xv, Yv, Xt, n_ens=N_ENS, width=WIDTH)
            z = mcorr(Yt.T, pz)
            pf = xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, n_ens=N_ENS, width=WIDTH)
            f = mcorr(Yt.T, pf)
            zs.append(z); ft.append(f)
            rows.append(dict(cond=cond, target=tgt, xsub_zs=round(z, 4), xsub_ft=round(f, 4)))
            print(f"  [{cond}] {tgt}: zs={z:.4f} ft={f:.4f}", flush=True)
        print(f"# [{cond}] MEAN zero-shot={np.mean(zs):.4f}  fine-tuned={np.mean(ft):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"# -> {out}", flush=True)


if __name__ == "__main__":
    run()
