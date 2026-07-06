"""
HPC LOSO driver for the TCN seq2seq decoder on the shared finger-aligned component space.
Per config (SLURM_ARRAY_TASK_ID), leave-one-subject-out over the 7 subjects:
  within : TCN trained on the target's own aligned sequence  (within-subject)
  XSUB   : TCN trained on the 6 DONORS' aligned sequences, eval on target  (cross-subject; bar ~0.55)
3-seed ensemble. Writes results/ward_tcn_loso_<task>.csv.

Run: python3 -m experiments.run_loso_tcn <task_id>   (from ~/projects/ward/code)
"""
import sys, os, time, csv
import numpy as np

from core.run_xsubject_hd import load, spatial_filters, SUBJECTS
from core.seq_decoder import aligned_seq, split, train_predict_ens, xsub_finetune_ens, mcorr
from core.bttn_finger import set_preset
import core.seq_decoder as S

GRID = [
    dict(preset='hd',   R=4, width=128, n_ens=3),   # 0  prior best (FT 0.552) + extended-RF TCN
    dict(preset='hd',   R=4, width=192, n_ens=3),   # 1  wider
    dict(preset='full', R=4, width=128, n_ens=3),   # 2  + low-freq (LMP) features
    dict(preset='full', R=4, width=192, n_ens=3),   # 3  + low-freq + wider
]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run(task):
    cfg = GRID[task % len(GRID)]
    set_preset(cfg['preset'])
    print(f"# ward-tcn-loso task={task} cfg={cfg} dev={S.DEV}", flush=True)
    al = {}
    for c in SUBJECTS:
        x, y = load(c); te = int(0.8*len(x)); Wf = spatial_filters(x[:te], y[:te], cfg['R'])
        al[c] = aligned_seq(c, Wf)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"ward_tcn_loso_{task}.csv"); rows = []
    for tgt in SUBJECTS:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt]); t0 = time.time()
        donors = [c for c in SUBJECTS if c != tgt]
        Xd = [al[d][0] for d in donors]; Yd = [al[d][1] for d in donors]
        win, _ = train_predict_ens([Xf], [Yf], Xv, Yv, Xt, n_ens=cfg['n_ens'], width=cfg['width'])
        win = mcorr(Yt.T, win)
        xz, _ = train_predict_ens(Xd, Yd, Xv, Yv, Xt, n_ens=cfg['n_ens'], width=cfg['width'])
        xz = mcorr(Yt.T, xz)                                   # zero-shot (no target labels)
        pft = xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, n_ens=cfg['n_ens'], width=cfg['width'])
        xft = mcorr(Yt.T, pft)                                 # minimal-calibration (donor pretrain + target FT)
        print(f"  {tgt}: within={win:.4f}  XSUB_zs={xz:.4f}  XSUB_ft={xft:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(task=task, **cfg, target=tgt, within=round(win, 4),
                         xsub_zs=round(xz, 4), xsub_ft=round(xft, 4)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"# MEAN within={np.mean([r['within'] for r in rows]):.4f}  "
          f"XSUB_zs={np.mean([r['xsub_zs'] for r in rows]):.4f}  "
          f"XSUB_ft={np.mean([r['xsub_ft'] for r in rows]):.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    run(task)
