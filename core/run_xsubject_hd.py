"""
Cross-subject feasibility with FINGERFLEX-GRADE features (the scale-up of run_xsubject.py).
Same shared/private architecture + Ridge decoder + continuous eval, but features go from the
crippled 25 Hz / 8-band rep to 100 Hz / ~20 high-gamma bands / longer lags (memory-light
filterbank+Hilbert equivalent of FingerFlex's 40-Morlet spectrogram). float32 throughout.

Goal: lift within-subject toward the ~0.67 bar so cross-subject (XSUB) follows toward CORTEG's ~0.55.

    python3 -m core.run_xsubject_hd            # HD preset (100Hz, 20 bands), R=2
    python3 -m core.run_xsubject_hd base       # the old 25Hz/8-band rep (sanity vs recorded 0.39/0.29)
    python3 -m core.run_xsubject_hd hd 3        # HD, R=3
"""
import sys, time
import numpy as np, scipy.io as sio, os
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import zscore, pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge

FS = 1000
PRESETS = {
    # name: (downsample_Hz, n_bands, f_lo, f_hi, lag_frames)
    "hd":   (100, 20, 40, 300, [0, 5, 10, 15, 20, 25]),     # 0..0.25s @100Hz, high-gamma only
    "full": (100, 24,  2, 300, [0, 5, 10, 15, 20, 25]),     # 2-300Hz: adds LMP/low-freq motor potential
    "base": (25,   8, 40, 300, [0, 2, 4, 6]),               # the validated 25Hz/8-band rep
}
HG = [70, 150]; SMOOTH = 6; R = 2
ALPHAS = [10., 100., 1e3, 1e4]
SUBJECTS = ['bp', 'zt', 'jp', 'ht', 'mv', 'wc', 'jc']
DATA = os.path.expanduser("~/data/stanford_fingerflex/{c}/{c}_fingerflex.mat")

# set by run()
DS = STEP = None; BANDS = None; LAGS = None

def load(c):
    d = sio.loadmat(DATA.format(c=c)); x = np.asarray(d['data'], np.float32); y = np.asarray(d['flex'], np.float32)
    n = min(len(x), len(y)); return x[:n], y[:n]
def norm(x):
    x = zscore(x, axis=0); return (x - np.median(x, axis=1, keepdims=True)).astype(np.float32)
def ds(a): return a[::STEP]

def spatial_filters(x_tr, y_tr, r=R):
    """sign-consistent finger-aligned spatial filters (n_ch x 5r), supervised on the calib slice."""
    xs = norm(x_tr); sos = butter(4, HG, btype='bandpass', fs=FS, output='sos')
    hg = ds(np.abs(hilbert(sosfiltfilt(sos, xs, axis=0), axis=0))); Y = ds(zscore(y_tr, axis=0))
    n = min(len(hg), len(Y)); hg, Y = zscore(hg[:n], axis=0), Y[:n]; cols = []
    for f in range(5):
        w = hg.T @ Y[:, f] / len(Y)
        cols.append(w / (np.linalg.norm(w) + 1e-9))
        if r >= 2:
            C = (hg * np.abs(w)).T @ (hg * np.abs(w)) / len(Y); U = np.linalg.svd(C)[0][:, :r-1]
            for k in range(r-1):
                u = U[:, k]; u = u * np.sign(u @ w + 1e-12); cols.append(u / (np.linalg.norm(u) + 1e-9))
    return np.stack(cols, 1).astype(np.float32)

def spec(x):
    out = np.zeros((len(x)//STEP + (1 if len(x) % STEP else 0), x.shape[1], len(BANDS)), np.float32)
    for i, b in enumerate(BANDS):
        sos = butter(4, b, btype='bandpass', fs=FS, output='sos')
        out[:, :, i] = ds(np.abs(hilbert(sosfiltfilt(sos, x, axis=0), axis=0))).astype(np.float32)
    return out
def lagflat(F):
    mx = max(LAGS); T = F.shape[0]
    X = np.stack([F[mx-l:T-l] for l in LAGS], axis=-1)
    return X.reshape(X.shape[0], -1)
def target(y): return ds(y)[max(LAGS):]

def raw_feats(c):
    x, y = load(c); F = spec(norm(x)); X = lagflat(F); Y = target(y)[:len(X)]
    return zscore(X, axis=0).astype(np.float32), Y
def aligned_feats(c, W):
    x, y = load(c); F = spec(norm(x)); F = np.einsum('ck,tcf->tkf', W, F, optimize=True)
    X = lagflat(F); Y = target(y)[:len(X)]; return zscore(X, axis=0).astype(np.float32), Y

def mcorr(yt, yp): return float(np.mean([pearsonr(yt[:, f], yp[:, f])[0] for f in range(5)]))
def ridge_cv(Xtr, Ytr, Xva, Yva):
    best, a0 = -1, ALPHAS[0]
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, Ytr); v = mcorr(Yva, gaussian_filter1d(r.predict(Xva), SMOOTH, axis=0))
        if v > best: best, a0 = v, a
    return a0
def evalr(m, X, Y): return mcorr(Y, gaussian_filter1d(m.predict(X), SMOOTH, axis=0))

def run(preset="hd", r=R):
    global DS, STEP, BANDS, LAGS
    DS, nb, flo, fhi, LAGS = PRESETS[preset]; STEP = FS // DS
    edges = np.logspace(np.log10(flo), np.log10(fhi), nb + 1); BANDS = [[edges[i], edges[i+1]] for i in range(nb)]
    print(f"# CROSS-SUBJECT HD feasibility [{preset}]: Ridge; {nb} bands {flo}-{fhi}Hz @ {DS}Hz; lags={LAGS}; "
          f"R={r}->{5*r} comps; CONTINUOUS eval. bar: xsub ~0.554 (CORTEG), within ~0.67")
    W = {}
    for c in SUBJECTS:
        x, y = load(c); te = int(0.8*len(x)); W[c] = spatial_filters(x[:te], y[:te], r)
    # hold ONLY the small aligned features for all subjects (needed for donor pooling);
    # compute the big full-channel features per-target inside the loop and free them.
    aliF = {c: aligned_feats(c, W[c]) for c in SUBJECTS}
    DO_FULL = os.environ.get("DO_FULL", "0") == "1"   # within_full is slow (7680 feats) + only a reference
    res = {"within_full": {}, "within_aligned": {}, "XSUB": {}, "XSUB_ft": {}}; t0 = time.time()
    for tgt in SUBJECTS:
        Xa, Ya = aliF[tgt]; n = len(Xa); va = int(.65*n); te = int(.8*n)
        if DO_FULL:
            X, Y = raw_feats(tgt)
            a = ridge_cv(X[:va], Y[:va], X[va:te], Y[va:te])
            res["within_full"][tgt] = evalr(Ridge(alpha=a).fit(X[:te], Y[:te]), X[te:], Y[te:])
            del X, Y
        else:
            res["within_full"][tgt] = float('nan')
        a2 = ridge_cv(Xa[:va], Ya[:va], Xa[va:te], Ya[va:te])
        res["within_aligned"][tgt] = evalr(Ridge(alpha=a2).fit(Xa[:te], Ya[:te]), Xa[te:], Ya[te:])
        donors = [c for c in SUBJECTS if c != tgt]
        Xd = np.concatenate([aliF[c][0] for c in donors]); Yd = np.concatenate([aliF[c][1] for c in donors])
        ad = ridge_cv(Xd, Yd, Xa[va:te], Ya[va:te])
        res["XSUB"][tgt] = evalr(Ridge(alpha=ad).fit(Xd, Yd), Xa[te:], Ya[te:])
        Xft = np.concatenate([Xd, Xa[:te]]); Yft = np.concatenate([Yd, Ya[:te]])
        res["XSUB_ft"][tgt] = evalr(Ridge(alpha=ad).fit(Xft, Yft), Xa[te:], Ya[te:])
        print(f"  {tgt}: within_full={res['within_full'][tgt]:.3f}  within_aligned={res['within_aligned'][tgt]:.3f}  "
              f"XSUB={res['XSUB'][tgt]:.3f}  XSUB+ft={res['XSUB_ft'][tgt]:.3f}")
    print("\n# MEANS: " + "  ".join(f"{k}={np.mean(list(v.values())):.4f}" for k, v in res.items()))
    print(f"# recorded base run: within_full 0.392 / within_aligned 0.390 / XSUB 0.293")
    print(f"# total {time.time()-t0:.0f}s")
    return res

if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "hd"
    r = int(sys.argv[2]) if len(sys.argv) > 2 else R
    run(preset, r)
