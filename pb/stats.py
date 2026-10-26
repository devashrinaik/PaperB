"""Pre-registered statistics: asymmetry from cached errors, AUROC, bootstrap, motion matching, K2 timing."""
import numpy as np
from scipy.stats import rankdata

N_BOOT, SEED = 1000, 0


# ---- asymmetry ---------------------------------------------------------------
def defined(e_f, e_b):
    return np.isfinite(e_f) & np.isfinite(e_b)


def asym(feat, err="l2", normalise=True, med=None):
    """Per-tubelet a(j) on the defined range. Returns (a, mask)."""
    e_f, e_b = feat[f"e_f_{err}"], feat[f"e_b_{err}"]
    m = defined(e_f, e_b)
    a = e_b - e_f
    if normalise:
        a = a / (med if med is not None else np.median(e_f[m]))
    return a, m


def clip_stats(feat, err="l2"):
    """A (mean, primary) and max, normalised and raw; YoCausal indicator."""
    a_n, m = asym(feat, err, True)
    a_r, _ = asym(feat, err, False)
    return dict(A=float(a_n[m].mean()), A_max=float(a_n[m].max()),
                A_raw=float(a_r[m].mean()), A_raw_max=float(a_r[m].max()),
                yocausal=float(feat[f"e_b_{err}"][m].mean() > feat[f"e_f_{err}"][m].mean()),
                n_defined=int(m.sum()))


# ---- AUROC / bootstrap -----------------------------------------------------------
def auroc(score, y):
    """Mann-Whitney AUROC, positives = y==1, ties counted half."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ci(draws, level=0.95):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def bootstrap(fn, n_units, n=N_BOOT, seed=SEED):
    """fn(indices) -> scalar or dict of scalars, over resampled unit indices."""
    rng = np.random.default_rng(seed)
    return [fn(rng.integers(0, n_units, n_units)) for _ in range(n)]


# ---- K1 motion matching ------------------------------------------------------------
def motion_match(F_pos, F_neg, tol=0.10, seed=SEED):
    """Greedy 1:1 matching without replacement: each positive (random order) gets the
    unused negative with nearest F, if within ±tol of the positive's F.
    Returns list of (pos_index, neg_index)."""
    rng = np.random.default_rng(seed)
    F_neg = np.asarray(F_neg, float)
    used = np.zeros(len(F_neg), bool)
    pairs = []
    for i in rng.permutation(len(F_pos)):
        f = F_pos[i]
        d = np.abs(F_neg - f)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and np.isfinite(d[j]) and d[j] <= tol * abs(f):
            used[j] = True
            pairs.append((int(i), j))
    return pairs


def ols_residual(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


# ---- K2 timing ---------------------------------------------------------------------
def merged_series(feats, err="l2", normalise=True):
    """Merge stride-3 streams (offsets 0..2) into one native-frame series.
    Each stream's a(j) is placed at the native frame round(tubelet centre).
    Normalisation uses the median e_f pooled over all streams' defined ranges."""
    if normalise:
        pooled = np.concatenate([f[f"e_f_{err}"][defined(f[f"e_f_{err}"], f[f"e_b_{err}"])] for f in feats])
        med = np.median(pooled)
    t, v = [], []
    for f in feats:
        a, m = asym(f, err, normalise, med if normalise else None)
        t.append(np.floor(f["tubelet_centre"][m] + 0.5).astype(int))
        v.append(a[m])
    t, v = np.concatenate(t), np.concatenate(v)
    o = np.argsort(t, kind="stable")
    return t[o], v[o]


def predict_peak(t, v):
    return int(t[np.argmax(v)])


def predict_first_exceed(t, v, q=95):
    thr = np.percentile(v, q)
    return int(t[np.argmax(v > thr)]) if (v > thr).any() else int(t[np.argmax(v)])


def chance_hit(lo, hi, t_star, tol=8):
    n = hi - lo + 1
    k = max(0, min(hi, t_star + tol) - max(lo, t_star - tol) + 1)
    return k / n
