"""
Robustness analyses and controls, headline config = full / R=4 / width=128.

Modes (argv[1]):
  seedsweep <seed>  : FULL-9-subject LOSO (incl. cc, wm) for ONE seed (n_ens=1) -> per-(subject) within/
                      zero-shot/fine-tuned. Run as a SLURM array over seeds 0..11. Aggregating the CSVs gives
                      the full-9 mean AND the 7-subset mean, and the seed-level CI on the mean.
  permute           : supervised-but-NOT-shared-axis control. Same behavior-supervised filter, but the target's
                      20 component axes are randomly permuted so the shared correspondence is destroyed while the
                      supervised front-end/denoising is identical. Isolates "shared behavioral axis" from "any
                      supervised front-end". 7 subjects, n_ens=3, reports real vs permuted-target.
  donorbars <k>     : donor-count k with resampled donor subsets (error bars) for the behavioral method.

Run (from ~/projects/ward/code):  python3 -m experiments.run_robustness <mode> [arg]
"""
import sys, os, time, csv
import numpy as np

from core.run_xsubject_hd import load, spatial_filters
from core.seq_decoder import aligned_seq, split, train_predict_ens, xsub_finetune_ens, mcorr
from core.bttn_finger import set_preset

PRESET, R, WIDTH = 'full', 4, 128
SUBJECTS9 = ['bp', 'cc', 'zt', 'jp', 'ht', 'mv', 'wc', 'wm', 'jc']   # full library
SUBJECTS7 = ['bp', 'zt', 'jp', 'ht', 'mv', 'wc', 'jc']              # the manuscript subset
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def align_all(subjects):
    al = {}
    for c in subjects:
        x, y = load(c); te = int(0.8 * len(x))
        al[c] = aligned_seq(c, spatial_filters(x[:te], y[:te], R))
    return al


def seedsweep(seed):
    """full-9 LOSO, single seed, n_ens=1 via seeds=[seed]."""
    set_preset(PRESET)
    print(f"# seedsweep seed={seed} preset={PRESET} R={R} w={WIDTH} subjects=9", flush=True)
    al = align_all(SUBJECTS9)
    rows = []
    for tgt in SUBJECTS9:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt]); t0 = time.time()
        don = [c for c in SUBJECTS9 if c != tgt]
        Xd = [al[d][0] for d in don]; Yd = [al[d][1] for d in don]
        win = mcorr(Yt.T, train_predict_ens([Xf], [Yf], Xv, Yv, Xt, width=WIDTH, seeds=[seed])[0])
        zs = mcorr(Yt.T, train_predict_ens(Xd, Yd, Xv, Yv, Xt, width=WIDTH, seeds=[seed])[0])
        ft = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, width=WIDTH, seeds=[seed]))
        rows.append(dict(seed=seed, target=tgt, within=round(win, 4), xsub_zs=round(zs, 4), xsub_ft=round(ft, 4)))
        print(f"  s{seed} {tgt}: within={win:.4f} zs={zs:.4f} ft={ft:.4f} ({time.time()-t0:.0f}s)", flush=True)
    _write(f"ward_seed{seed:02d}.csv", rows)


def full9ens():
    """full-9 LOSO with the SAME 3-seed ensemble as the headline -> directly comparable full-library mean."""
    set_preset(PRESET)
    print(f"# full9ens preset={PRESET} R={R} w={WIDTH} n_ens=3 subjects=9", flush=True)
    al = align_all(SUBJECTS9)
    rows = []
    for tgt in SUBJECTS9:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt]); t0 = time.time()
        don = [c for c in SUBJECTS9 if c != tgt]
        Xd = [al[d][0] for d in don]; Yd = [al[d][1] for d in don]
        win = mcorr(Yt.T, train_predict_ens([Xf], [Yf], Xv, Yv, Xt, n_ens=3, width=WIDTH)[0])
        zs = mcorr(Yt.T, train_predict_ens(Xd, Yd, Xv, Yv, Xt, n_ens=3, width=WIDTH)[0])
        ft = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, n_ens=3, width=WIDTH))
        rows.append(dict(target=tgt, within=round(win, 4), xsub_zs=round(zs, 4), xsub_ft=round(ft, 4)))
        print(f"  {tgt}: within={win:.4f} zs={zs:.4f} ft={ft:.4f} ({time.time()-t0:.0f}s)", flush=True)
    import numpy as _np
    print(f"# FULL-9 (3-seed) within={_np.mean([r['within'] for r in rows]):.4f} "
          f"zs={_np.mean([r['xsub_zs'] for r in rows]):.4f} ft={_np.mean([r['xsub_ft'] for r in rows]):.4f}", flush=True)
    _write("ward_full9ens.csv", rows)


def permute():
    """supervised-but-not-shared control: permute the TARGET's 20 component axes (donors keep shared order)."""
    set_preset(PRESET)
    print(f"# permute-control preset={PRESET} R={R} w={WIDTH}", flush=True)
    rng = np.random.RandomState(0)
    # real alignment for everyone
    base = {}
    for c in SUBJECTS7:
        x, y = load(c); te = int(0.8 * len(x)); base[c] = (x, y, te, spatial_filters(x[:te], y[:te], R))
    al = {c: aligned_seq(c, base[c][3]) for c in SUBJECTS7}
    rows = []
    for tgt in SUBJECTS7:
        don = [c for c in SUBJECTS7 if c != tgt]
        Xd = [al[d][0] for d in don]; Yd = [al[d][1] for d in don]
        # real target
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt]); t0 = time.time()
        zs_r = mcorr(Yt.T, train_predict_ens(Xd, Yd, Xv, Yv, Xt, width=WIDTH)[0])
        ft_r = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, width=WIDTH))
        # permuted target: same supervised filter, shuffled component correspondence
        x, y, te, Wf = base[tgt]; perm = rng.permutation(Wf.shape[1])
        (Xfp, Yfp), (Xvp, Yvp), (Xtp, Ytp) = split(*aligned_seq(tgt, Wf[:, perm]))
        zs_p = mcorr(Ytp.T, train_predict_ens(Xd, Yd, Xvp, Yvp, Xtp, width=WIDTH)[0])
        ft_p = mcorr(Ytp.T, xsub_finetune_ens(Xd, Yd, Xfp, Yfp, Xvp, Yvp, Xtp, width=WIDTH))
        rows.append(dict(target=tgt, zs_real=round(zs_r, 4), ft_real=round(ft_r, 4),
                         zs_perm=round(zs_p, 4), ft_perm=round(ft_p, 4)))
        print(f"  {tgt}: REAL zs={zs_r:.4f} ft={ft_r:.4f} | PERM zs={zs_p:.4f} ft={ft_p:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    m = lambda k: np.mean([r[k] for r in rows])
    print(f"# MEAN real zs={m('zs_real'):.4f} ft={m('ft_real'):.4f} | perm zs={m('zs_perm'):.4f} "
          f"ft={m('ft_perm'):.4f}", flush=True)
    _write("ward_permute.csv", rows)


def donorbars(k):
    """donor-count k with resampled donor subsets (behavioral method), fine-tuned; error bars."""
    set_preset(PRESET)
    print(f"# donorbars k={k} preset={PRESET} R={R} w={WIDTH}", flush=True)
    al = align_all(SUBJECTS7)
    rng = np.random.RandomState(k)
    NSUB = 5                                  # subsets per target
    rows = []
    for tgt in SUBJECTS7:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt])
        pool = [c for c in SUBJECTS7 if c != tgt]
        seen = set()
        for _ in range(NSUB):
            sub = tuple(sorted(rng.choice(pool, size=k, replace=False)))
            if sub in seen and k < len(pool):
                continue
            seen.add(sub)
            Xd = [al[d][0] for d in sub]; Yd = [al[d][1] for d in sub]
            ft = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, width=WIDTH, seeds=[0]))
            rows.append(dict(k=k, target=tgt, donors="+".join(sub), xsub_ft=round(ft, 4)))
            print(f"  k={k} {tgt} <-{sub}: ft={ft:.4f}", flush=True)
    v = [r['xsub_ft'] for r in rows]
    print(f"# k={k} fine-tuned mean={np.mean(v):.4f} sd={np.std(v, ddof=1):.4f} (n={len(v)})", flush=True)
    _write(f"ward_donorbars_k{k}.csv", rows)


def _write(name, rows):
    os.makedirs(RESULTS, exist_ok=True); out = os.path.join(RESULTS, name)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"# -> {out}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROBUST_MODE", "seedsweep")
    if mode == "seedsweep":
        seedsweep(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    elif mode == "full9ens":
        full9ens()
    elif mode == "permute":
        permute()
    elif mode == "donorbars":
        donorbars(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)))
    else:
        sys.exit(f"unknown mode {mode}")
