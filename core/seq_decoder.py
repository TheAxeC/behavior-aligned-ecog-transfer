"""
Seq2seq decoder on the SHARED finger-aligned component space, to break the windowed-BTTN
ceiling (~0.39). The diagnostic (job 518930) showed the limiter is the DECODER, not the
representation (within_aligned >= within_full; more components don't help). So we keep the
behavioral alignment (novel, interpretable, transfer-enabling) and put a FingerFlex-class
SEQ2SEQ decoder on top: a Temporal Convolutional Network (stacked dilated residual convs,
length-preserving) mapping the aligned (5R*bands, time) sequence -> (5, time) continuous
finger trajectory. Trained on donors for cross-subject transfer.

    python3 -m core.seq_decoder zt mv          # within-subject smoke (local MPS)
    python3 -m core.seq_decoder xsub zt mv     # cross-subject smoke
"""
import sys, time
import numpy as np
import torch, torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from core.run_xsubject_hd import load, spatial_filters, PRESETS, FS, SUBJECTS
from core.bttn_finger import aligned_TCB, set_preset, DEV
import core.run_xsubject_hd as H

SEG = 512          # training-segment length (frames); TCN sees full context within a segment
SMOOTH = 6
MAX_EPOCH, PATIENCE, BATCH = 120, 14, 32


def aligned_seq(c, Wf):
    """(C_in=5R*bands, T) aligned feature sequence + (5, T) target."""
    F, Y = aligned_TCB(c, Wf)                 # (T, 5R, bands), (T, 5)
    T = F.shape[0]
    X = F.reshape(T, -1).T.astype(np.float32)  # (5R*bands, T)
    return X, Y.T.astype(np.float32)           # (C_in, T), (5, T)


# ---- BASELINE feature paths (same protocol, NO behavioral alignment) -----------------------
def _ea_whiten(x):
    """Euclidean Alignment (He & Wu 2020): whiten each subject's channel covariance to identity,
    so a donor-trained decoder's channel weights apply to the target. x: (T, ch) -> (T, ch)."""
    from core.run_xsubject_hd import norm
    xs = norm(x)
    R = (xs.T @ xs) / len(xs)                                   # (ch, ch) mean covariance
    ev, V = np.linalg.eigh(R + 1e-6 * np.eye(R.shape[0]))
    Rinv = V @ np.diag(ev ** -0.5) @ V.T                        # R^{-1/2}
    return (xs @ Rinv).astype(np.float32)


def raw_first43_seq(c, align=None, CH=43):
    """BASELINE: spectral tensor on the FIRST CH channels (the thesis's naive cross-subject scheme;
    raw channel index has no cross-subject correspondence). align='ea' applies Euclidean Alignment.
    Returns (CH*bands, T), (5, T)."""
    from core.run_xsubject_hd import norm, ds, spec
    x, y = load(c); x = x[:, :CH]
    xa = _ea_whiten(x) if align == 'ea' else norm(x)
    F = spec(xa)                                               # (T, CH, bands)  (spec re-z-scores)
    Y = ds(y).astype(np.float32)
    n = min(len(F), len(Y)); F, Y = F[:n], Y[:n]
    tr = int(0.8 * n); m = F[:tr].mean(0, keepdims=True); s = F[:tr].std(0, keepdims=True) + 1e-6
    X = ((F - m) / s).reshape(n, -1).T.astype(np.float32)
    return X, Y.T.astype(np.float32)


class TCNBlock(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        pad = (k - 1) * d // 2
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=pad, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(c, c, k, padding=pad, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(p))

    def forward(self, x): return torch.relu(x + self.net(x))


class TCN(nn.Module):
    """length-preserving dilated-conv seq2seq: (B, C_in, T) -> (B, 5, T)."""
    def __init__(self, c_in, width=128, k=7, dilations=(1, 2, 4, 8, 16, 32, 64), p=0.2, n_out=5):
        super().__init__()
        self.inp = nn.Conv1d(c_in, width, 1)
        self.blocks = nn.ModuleList([TCNBlock(width, k, d, p) for d in dilations])
        self.out = nn.Conv1d(width, n_out, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.out(h)


class UNet1D(nn.Module):
    """1D conv autoencoder with skip connections (FingerFlex-class): (B, C_in, T) -> (B, 5, T).
    Multi-scale encoder/decoder captures both fast (high-gamma bursts) and slow (movement) structure."""
    def __init__(self, c_in, width=64, depth=4, k=9, p=0.2, n_out=5):
        super().__init__()
        self.depth = depth
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.GELU(),
                                 nn.Dropout(p), nn.Conv1d(co, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.GELU())
        self.inp = blk(c_in, width)
        self.down = nn.ModuleList([blk(width * 2 ** i, width * 2 ** (i + 1)) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose1d(width * 2 ** (i + 1), width * 2 ** i, 2, stride=2) for i in reversed(range(depth))])
        self.upblk = nn.ModuleList([blk(width * 2 ** (i + 1), width * 2 ** i) for i in reversed(range(depth))])
        self.out = nn.Conv1d(width, n_out, 1)

    def forward(self, x):
        T0 = x.shape[-1]; m = 2 ** self.depth
        if T0 % m: x = nn.functional.pad(x, (0, m - T0 % m))
        h = self.inp(x); skips = []
        for d in self.down:
            skips.append(h); h = d(nn.functional.max_pool1d(h, 2))
        for u, ub, s in zip(self.up, self.upblk, reversed(skips)):
            h = u(h)
            if h.shape[-1] != s.shape[-1]: h = nn.functional.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = ub(torch.cat([h, s], 1))
        return self.out(h)[..., :T0]


def build_model(c_in, model_type='tcn', width=128):
    return UNet1D(c_in, width=width) if model_type == 'unet' else TCN(c_in, width=width)


def mcorr(yt, yp):  # yt,yp: (T,5)
    return float(np.mean([pearsonr(yt[:, f], yp[:, f])[0] for f in range(5)]))


def corr_loss(pred, targ):
    """1 - mean Pearson over fingers (differentiable), per (B,5,T)."""
    p = pred - pred.mean(-1, keepdim=True); t = targ - targ.mean(-1, keepdim=True)
    num = (p * t).sum(-1)
    den = torch.sqrt((p ** 2).sum(-1) * (t ** 2).sum(-1) + 1e-8)
    return 1 - (num / den).mean()


def _segments(X, Y, seg=SEG, stride=None):
    """list of (C_in, seg) / (5, seg) training segments from a continuous (C_in,T)/(5,T) pair."""
    stride = stride or seg // 2; T = X.shape[1]
    idx = list(range(0, max(1, T - seg + 1), stride))
    xs = np.stack([X[:, i:i + seg] for i in idx]); ys = np.stack([Y[:, i:i + seg] for i in idx])
    return xs.astype(np.float32), ys.astype(np.float32)


def train_predict(Xtr, Ytr, Xva, Yva, Xte, seed=0, width=128, epochs=MAX_EPOCH,
                  init_state=None, lr=2e-3, model_type='tcn'):
    """train seq2seq decoder on segments (MSE + corr loss), early-stop on val continuous corr, predict
    full test seq. Xtr/Ytr: list of (C_in,T)/(5,T) per donor (or single target). init_state warm-starts.
    model_type: 'tcn' or 'unet'. Returns (smoothed_test_pred, best_val_corr, best_state)."""
    torch.manual_seed(seed)
    Ym = np.concatenate([y for y in Ytr], axis=1).mean(1, keepdims=True)
    Ys = np.concatenate([y for y in Ytr], axis=1).std(1, keepdims=True) + 1e-6
    segs = [_segments(x, (y - Ym) / Ys) for x, y in zip(Xtr, Ytr)]
    Xs = np.concatenate([s[0] for s in segs]); Ysg = np.concatenate([s[1] for s in segs])
    model = build_model(Xtr[0].shape[0], model_type=model_type, width=width).to(DEV)
    if init_state is not None: model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    n = len(Xs); best, best_state, bad = -1, None, 0
    Xv = torch.tensor(Xva[None], device=DEV)            # full val seq (1, C_in, T)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.tensor(Xs[b], device=DEV); yb = torch.tensor(Ysg[b], device=DEV)
            opt.zero_grad(); pr = model(xb)
            (mse(pr, yb) + 0.5 * corr_loss(pr, yb)).backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            pv = (model(Xv)[0].cpu().numpy() * Ys + Ym).T
        c = mcorr(Yva.T, gaussian_filter1d(pv, SMOOTH, axis=0))
        if c > best + 1e-4: best, best_state, bad = c, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pe = (model(torch.tensor(Xte[None], device=DEV))[0].cpu().numpy() * Ys + Ym).T
    return gaussian_filter1d(pe, SMOOTH, axis=0), best, best_state


def train_predict_ens(Xtr, Ytr, Xva, Yva, Xte, n_ens=3, width=128, epochs=MAX_EPOCH, model_type='tcn', seeds=None):
    """n_ens-seed ensemble: average the (smoothed) continuous test predictions across seeds.
    Pass an explicit `seeds` list to control which seeds (default range(n_ens))."""
    sds = range(n_ens) if seeds is None else seeds
    preds, vals = [], []
    for sd in sds:
        p, v, _ = train_predict(Xtr, Ytr, Xva, Yva, Xte, seed=sd, width=width, epochs=epochs, model_type=model_type)
        preds.append(p); vals.append(v)
    return np.mean(preds, axis=0), float(np.mean(vals))


def xsub_finetune_ens(Xdon, Ydon, Xtt, Ytt, Xva, Yva, Xte, n_ens=3, width=128,
                      ft_epochs=40, ft_lr=5e-4, model_type='tcn', seeds=None):
    """donor-pretrain -> light target fine-tune (CORTEG-comparable minimal-calibration protocol).
    Per seed: train TCN on the 6 donors (early-stop on target val), then warm-start fine-tune on the
    TARGET's own train slice (fewer epochs, lower lr). Average test predictions over seeds.
    Pass an explicit `seeds` list to control which seeds (default range(n_ens))."""
    sds = range(n_ens) if seeds is None else seeds
    preds = []
    for sd in sds:
        _, _, st = train_predict(Xdon, Ydon, Xva, Yva, Xte, seed=sd, width=width, model_type=model_type)
        p, _, _ = train_predict([Xtt], [Ytt], Xva, Yva, Xte, seed=sd, width=width,
                                init_state=st, epochs=ft_epochs, lr=ft_lr, model_type=model_type)
        preds.append(p)
    return np.mean(preds, axis=0)


def calib_curve_ens(Xdon, Ydon, Xtt, Ytt, Xva, Yva, Xte, fracs, n_ens=3, width=128,
                    ft_epochs=40, ft_lr=5e-4, model_type='tcn'):
    """Calibration-amount curve: pretrain the donor model ONCE per seed, then fine-tune on the FIRST
    `f` fraction of the target's train slice for each f in fracs (cheap - donor pretrain is reused).
    Returns {f: mean-over-seeds smoothed test prediction}."""
    Ttt = Xtt.shape[1]
    preds = {f: [] for f in fracs}
    for sd in range(n_ens):
        _, _, st = train_predict(Xdon, Ydon, Xva, Yva, Xte, seed=sd, width=width, model_type=model_type)
        for f in fracs:
            k = max(SEG, int(f * Ttt))
            p, _, _ = train_predict([Xtt[:, :k]], [Ytt[:, :k]], Xva, Yva, Xte, seed=sd, width=width,
                                    init_state=st, epochs=ft_epochs, lr=ft_lr, model_type=model_type)
            preds[f].append(p)
    return {f: np.mean(preds[f], axis=0) for f in fracs}


def split(X, Y):
    T = X.shape[1]; va = int(.65*T); te = int(.8*T)
    return (X[:, :va], Y[:, :va]), (X[:, va:te], Y[:, va:te]), (X[:, te:], Y[:, te:])


def run_within(subj, preset='hd', R=2, model_type='tcn'):
    set_preset(preset)
    print(f"# {model_type.upper()} seq2seq WITHIN-subject [{preset}, R={R}] dev={DEV}")
    for c in subj:
        x, y = load(c); te = int(0.8*len(x)); Wf = spatial_filters(x[:te], y[:te], R)
        X, Y = aligned_seq(c, Wf)
        (Xf, Yf), (Xv, Yv), (Xt, Yt) = split(X, Y)
        t0 = time.time(); pe, bv, _ = train_predict([Xf], [Yf], Xv, Yv, Xt, model_type=model_type)
        print(f"  {c}: TCN within={mcorr(Yt.T, pe):.4f}  (val={bv:.3f}, {time.time()-t0:.0f}s)")


def run_xsub(subj, preset='hd', R=2):
    set_preset(preset)
    print(f"# TCN seq2seq CROSS-SUBJECT [{preset}, R={R}] dev={DEV}")
    al = {}
    for c in SUBJECTS:
        x, y = load(c); te = int(0.8*len(x)); Wf = spatial_filters(x[:te], y[:te], R); al[c] = aligned_seq(c, Wf)
    for tgt in subj:
        (_, _), (Xv, Yv), (Xt, Yt) = split(*al[tgt])
        donors = [c for c in SUBJECTS if c != tgt]
        Xtr = [al[d][0] for d in donors]; Ytr = [al[d][1] for d in donors]
        t0 = time.time(); pe, bv, _ = train_predict(Xtr, Ytr, Xv, Yv, Xt)
        print(f"  {tgt}: TCN XSUB={mcorr(Yt.T, pe):.4f}  (val={bv:.3f}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    subj = tuple(a for a in args if a in SUBJECTS) or ('zt', 'mv')
    (run_xsub if "xsub" in args else run_within)(subj)
