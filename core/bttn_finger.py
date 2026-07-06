"""
BTTN adapted to ECoG FINGER-FLEXION REGRESSION (ward project's OWN vendored copy of the
Block-Term Tensor Network architecture adapted from the afnet (bttn-incident-af) project - that code
is read-only, so this is a separate, adapted copy; NOT an import across projects).

Changes vs the afnet BTTN:
  - output = 5 continuous finger flexions (Linear -> 5), not a scalar log-risk;
  - loss = MSE (with a correlation-friendly target normalisation), not Cox partial likelihood;
  - input = a windowed spectro-spatial tensor (bands x window-time x finger-components): the
    block-term factors A_k (band signature) and C_k (component signature) stay interpretable, the
    nonlinear head adds the capacity ridge lacked, and the predictive path runs THROUGH the block
    subspaces so the maps stay faithful.
Operates in the SHARED finger-aligned component space (behavioral alignment handles mismatched
grids), so the same network transfers across subjects (donor-trained for cross-subject).

Smoke test (local MPS): python3 -m core.bttn_finger zt        # within-subject, one subject
"""
import sys, time
import numpy as np
import torch, torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, pearsonr

from core.run_xsubject_hd import load, norm, ds, spatial_filters, PRESETS, FS, SUBJECTS
import core.run_xsubject_hd as H

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
WIN = 12          # window length (frames); at 25Hz=0.48s, at 100Hz=0.12s
STRIDE = 1
MAX_EPOCH, PATIENCE, BATCH = 80, 10, 1024


def set_preset(preset):
    H.DS, nb, flo, fhi, _ = PRESETS[preset]; H.STEP = FS // H.DS
    e = np.logspace(np.log10(flo), np.log10(fhi), nb + 1); H.BANDS = [[e[i], e[i+1]] for i in range(nb)]
    return nb


def aligned_TCB(c, Wf, train_frac=0.8):
    """aligned spectro-spatial sequence: (T, comps, bands) + target (T,5).
    Feature z-score uses ONLY the first train_frac of frames (no test leakage)."""
    x, y = load(c)
    F = H.spec(norm(x))                      # (T, ch, bands)
    F = np.einsum('ck,tcf->tkf', Wf, F, optimize=True).astype(np.float32)   # (T, comps, bands)
    Y = ds(y).astype(np.float32)
    n = min(len(F), len(Y)); F, Y = F[:n], Y[:n]
    tr = int(train_frac * n)
    m = F[:tr].mean(0, keepdims=True); s = F[:tr].std(0, keepdims=True) + 1e-6
    return ((F - m) / s).astype(np.float32), Y


def raw_TCB(c, train_frac=0.8):
    """FULL-CHANNEL spectro tensor (T, channels, bands) - no alignment; for the within-subject
    ceiling (does the 10-component bottleneck cap accuracy?). z-score on train frames only."""
    x, y = load(c)
    F = H.spec(norm(x)).astype(np.float32)            # (T, ch, bands)
    Y = ds(y).astype(np.float32)
    n = min(len(F), len(Y)); F, Y = F[:n], Y[:n]
    tr = int(train_frac * n)
    m = F[:tr].mean(0, keepdims=True); s = F[:tr].std(0, keepdims=True) + 1e-6
    return ((F - m) / s).astype(np.float32), Y


def windows(F, Y, w=WIN, stride=STRIDE):
    """(T,comps,bands) -> X (N, bands, w, comps), Yw (N,5); causal: window ending at t predicts Y[t]."""
    T = len(F); idx = np.arange(w - 1, T, stride)
    X = np.stack([F[i - w + 1:i + 1] for i in idx])          # (N, w, comps, bands)
    X = np.transpose(X, (0, 3, 1, 2))                        # (N, bands, w, comps)
    return np.ascontiguousarray(X), Y[idx], idx


class BlockTermLayer(nn.Module):
    """K blocks, each a learnable rank-(rf,rl) Tucker subspace over (band, component) modes
    (A_k band signature, C_k component signature) + multi-scale temporal convs over the window.
    Glass-box: A_k = per-block band profile, C_k = per-block finger-component profile."""
    def __init__(self, F, W, L, K=12, rf=3, rl=2, ksizes=(3, 5, 7), ct=6):
        super().__init__()
        self.K, self.rf, self.rl, self.ct = K, rf, rl, ct
        ch = K * rf * rl
        self.A = nn.Parameter(torch.randn(K, F, rf) * 0.1)        # band factors
        self.C = nn.Parameter(torch.randn(K, L, rl) * 0.1)        # component factors
        self.tconvs = nn.ModuleList([nn.Conv1d(ch, ch * ct, ks, padding=ks // 2, groups=ch) for ks in ksizes])
        self.ch_out = ch * ct * len(ksizes)
        self.bn = nn.BatchNorm1d(self.ch_out)
        self.attn = nn.Conv1d(self.ch_out, 1, 1)

    def _project(self, x):                                       # x (B, F, W, L) -> (B, ch, W)
        z = torch.einsum("bfwl,kfp->bkpwl", x, self.A)
        z = torch.einsum("bkpwl,kls->bkpws", z, self.C)
        B_, K, rf, W, rl = z.shape
        return z.permute(0, 1, 2, 4, 3).reshape(B_, K * rf * rl, W)

    def forward(self, x):
        z0 = self._project(x)
        z = torch.relu(self.bn(torch.cat([c(z0) for c in self.tconvs], dim=1)))   # (B, ch_out, W)
        a = torch.softmax(self.attn(z), dim=-1)
        return torch.cat([(z * a).sum(-1), z.amax(-1)], dim=1)   # (B, 2*ch_out)

    def out_dim(self): return 2 * self.ch_out

    def maps(self):
        A, C = self.A.detach(), self.C.detach()
        r = min(self.rf, self.rl)
        return torch.einsum("kfp,klp->kfl", A[..., :r], C[..., :r]).cpu().numpy()   # (K, band, comp)


class BTTNReg(nn.Module):
    def __init__(self, F, W, L, K=12, hidden=128, p=0.3, n_out=5):
        super().__init__()
        self.block = BlockTermLayer(F, W, L, K)
        d = self.block.out_dim()
        self.head = nn.Sequential(
            nn.BatchNorm1d(d), nn.Dropout(p),
            nn.Linear(d, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(p),
            nn.Linear(hidden, n_out))

    def forward(self, x): return self.head(self.block(x))


def mcorr(yt, yp): return float(np.mean([pearsonr(yt[:, f], yp[:, f])[0] for f in range(5)]))


def _predict_batched(model, X, Ys, Ym):
    """predict in batches, keeping X on CPU (GPU-memory-safe for big donor pools)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.tensor(X[i:i + BATCH], device=DEV)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) * Ys + Ym


def train_predict(Xtr, Ytr, Xva, Yva, Xte, K=12, seed=0, epochs=MAX_EPOCH):
    """train BTTN (MSE), early-stop on val mean-correlation, return continuous test predictions.
    Data stays on CPU; batches are moved to GPU per step (safe for ~9 GB HD donor pools)."""
    torch.manual_seed(seed)
    Ys = Ytr.std(0) + 1e-6; Ym = Ytr.mean(0)
    Ytn = ((Ytr - Ym) / Ys).astype(np.float32)
    F, W, L = Xtr.shape[1:]
    model = BTTNReg(F, W, L, K=K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()
    n = len(Xtr); best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.tensor(Xtr[b], device=DEV); yb = torch.tensor(Ytn[b], device=DEV)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        sched.step(); model.eval()
        pv = _predict_batched(model, Xva, Ys, Ym)
        c = mcorr(Yva, gaussian_filter1d(pv, 6, axis=0))
        if c > best + 1e-4: best, best_state, bad = c, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    pe = _predict_batched(model, Xte, Ys, Ym)
    return gaussian_filter1d(pe, 6, axis=0), best


def train_predict_ens(Xtr, Ytr, Xva, Yva, Xte, K=12, n_ens=3, epochs=MAX_EPOCH):
    """n_ens-seed ensemble: average the (already smoothed) test predictions across seeds."""
    preds, vals = [], []
    for sd in range(n_ens):
        p, v = train_predict(Xtr, Ytr, Xva, Yva, Xte, K=K, seed=sd, epochs=epochs)
        preds.append(p); vals.append(v)
    return np.mean(preds, axis=0), float(np.mean(vals))


def run_within(subj=('zt',), preset='base', K=12):
    nb = set_preset(preset)
    print(f"# BTTN finger-regression WITHIN-subject [{preset}, {nb} bands @ {H.DS}Hz, win={WIN}] dev={DEV}")
    for c in subj:
        x, y = load(c); te = int(0.8 * len(x)); Wf = spatial_filters(x[:te], y[:te], 2)
        F, Y = aligned_TCB(c, Wf)
        X, Yw, _ = windows(F, Y)
        n = len(X); va = int(.65 * n); te2 = int(.8 * n)
        t0 = time.time()
        pe, bv = train_predict(X[:va], Yw[:va], X[va:te2], Yw[va:te2], X[te2:], K=K)
        print(f"  {c}: BTTN within={mcorr(Yw[te2:], pe):.4f}  (val={bv:.3f}, {time.time()-t0:.0f}s)  "
              f"[ridge ref zt 0.471 mv 0.480]")


def run_xsub(targets=('zt', 'mv'), preset='base', K=12):
    """CROSS-SUBJECT: BTTN trained on the 6 DONORS' windows in the shared finger-component space,
    target's private spatial filter fit on its calib slice, early-stop on target val, eval on target
    test. Compares to the ridge XSUB refs."""
    nb = set_preset(preset)
    ref = {'zt': 0.426, 'mv': 0.337, 'jc': 0.379, 'bp': 0.210, 'ht': 0.230, 'jp': 0.142, 'wc': 0.258}
    print(f"# BTTN CROSS-SUBJECT (donor-trained) [{preset}, {nb} bands @ {H.DS}Hz, win={WIN}] dev={DEV}")
    # per-subject private filter + aligned (T,comps,bands)
    al = {}
    for c in H.SUBJECTS:
        x, y = load(c); te = int(0.8 * len(x)); Wf = spatial_filters(x[:te], y[:te], 2)
        al[c] = aligned_TCB(c, Wf)
    for tgt in targets:
        donors = [c for c in H.SUBJECTS if c != tgt]
        Xd = np.concatenate([windows(*al[d])[0] for d in donors])
        Yd = np.concatenate([windows(*al[d])[1] for d in donors])
        Xt, Yt, _ = windows(*al[tgt]); n = len(Xt); va = int(.65*n); te2 = int(.8*n)
        t0 = time.time()
        pe, bv = train_predict(Xd, Yd, Xt[va:te2], Yt[va:te2], Xt[te2:], K=K)
        print(f"  {tgt}: BTTN XSUB={mcorr(Yt[te2:], pe):.4f}  (val={bv:.3f}, {time.time()-t0:.0f}s)  "
              f"[ridge XSUB ref {ref.get(tgt,'?')}]")
        del Xd, Yd


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "xsub" if "xsub" in args else "within"
    subj = tuple(a for a in args if a in H.SUBJECTS) or ('zt', 'mv')
    preset = "hd" if "hd" in args else "base"
    (run_xsub if mode == "xsub" else run_within)(subj, preset=preset)
