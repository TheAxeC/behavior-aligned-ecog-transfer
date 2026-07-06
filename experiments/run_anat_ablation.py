"""
Two robustness analyses for the cross-subject decoder:
  - a STRONGER, non-strawman generic-alignment baseline: ANATOMICAL-COORDINATE alignment -
       project each subject's spectral features onto a common cortical anchor grid (k-means on the
       pooled MNI electrode coordinates `locs`, Gaussian interpolation), then the SAME shared TCN.
       This is a principled grid-agnostic alignment (uses real electrode positions), unlike first-43.
  - DONOR-COUNT ablation for the behavioral method: cross-subject fine-tuned accuracy with
       k = 1,2,4,6 donors - quantifies the population-prior claim.
Same TCN, same LOSO, same eval. full preset, width 128, 3-seed ensemble.

Run: python3 -m experiments.run_anat_ablation   (from ~/projects/ward/code)
"""
import os, time, csv
import numpy as np, scipy.io as sio
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from core.run_xsubject_hd import load, norm, ds, spec, spatial_filters, SUBJECTS, DATA
from core.seq_decoder import aligned_seq, split, train_predict_ens, xsub_finetune_ens, mcorr, SEG
from core.bttn_finger import set_preset
import core.run_xsubject_hd as H

A = 20          # anatomical anchors (match 5R=20 component count for a fair comparison)
SIGMA = 12.0
WIDTH, N_ENS = 128, 3
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def locs(c):
    return np.asarray(sio.loadmat(DATA.format(c=c))['locs'], float)


def anat_seq(c, anchors, sigma=SIGMA):
    """anatomical-alignment baseline: spectral features projected onto the common anchor grid."""
    x, y = load(c); L = locs(c)[:x.shape[1]]
    Wd = np.exp(-(cdist(anchors, L) ** 2) / (2 * sigma ** 2)); Wd /= (Wd.sum(1, keepdims=True) + 1e-12)
    F = spec(norm(x))                                   # (T, ch, bands)
    F = np.einsum('ac,tcf->taf', Wd, F).astype(np.float32)   # (T, A, bands)
    Y = ds(y).astype(np.float32); n = min(len(F), len(Y)); F, Y = F[:n], Y[:n]
    tr = int(0.8 * n); m = F[:tr].mean(0, keepdims=True); s = F[:tr].std(0, keepdims=True) + 1e-6
    return ((F - m) / s).reshape(n, -1).T.astype(np.float32), Y.T.astype(np.float32)


def run():
    set_preset('full')
    rows = []
    # ---- anatomical-alignment baseline ----
    print("# anatomical-coordinate alignment baseline (A=%d anchors) + shared TCN" % A, flush=True)
    anchors = KMeans(n_clusters=A, n_init=10, random_state=0).fit(
        np.concatenate([locs(c) for c in SUBJECTS])).cluster_centers_
    anat = {c: anat_seq(c, anchors) for c in SUBJECTS}
    zs, ft = [], []
    for tgt in SUBJECTS:
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*anat[tgt])
        don = [c for c in SUBJECTS if c != tgt]
        Xd = [anat[d][0] for d in don]; Yd = [anat[d][1] for d in don]
        z = mcorr(Yt.T, train_predict_ens(Xd, Yd, Xv, Yv, Xt, n_ens=N_ENS, width=WIDTH)[0])
        f = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, n_ens=N_ENS, width=WIDTH))
        zs.append(z); ft.append(f); rows.append(dict(exp='anat_align', k='', target=tgt, xsub_zs=round(z,4), xsub_ft=round(f,4)))
        print(f"  [anat] {tgt}: zs={z:.4f} ft={f:.4f}", flush=True)
    print(f"# [anat-align] MEAN zero-shot={np.mean(zs):.4f} fine-tuned={np.mean(ft):.4f}", flush=True)

    # ---- donor-count ablation (behavioral alignment) ----
    print("# donor-count ablation (behavioral alignment), fine-tuned cross-subject", flush=True)
    al = {}
    for c in SUBJECTS:
        x, y = load(c); te = int(0.8*len(x)); al[c] = aligned_seq(c, spatial_filters(x[:te], y[:te], 4))
    for k in (1, 2, 4, 6):
        per = []
        for tgt in SUBJECTS:
            (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(*al[tgt])
            don = [c for c in SUBJECTS if c != tgt][:k]
            Xd = [al[d][0] for d in don]; Yd = [al[d][1] for d in don]
            f = mcorr(Yt.T, xsub_finetune_ens(Xd, Yd, Xf, Yf, Xv, Yv, Xt, n_ens=N_ENS, width=WIDTH))
            per.append(f); rows.append(dict(exp='donor_count', k=k, target=tgt, xsub_zs='', xsub_ft=round(f,4)))
        print(f"# [donors={k}] fine-tuned MEAN={np.mean(per):.4f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "ward_anat_ablation.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=['exp','k','target','xsub_zs','xsub_ft']); w.writeheader(); w.writerows(rows)
    print(f"# -> {out}", flush=True)


if __name__ == "__main__":
    run()
