"""K1: class-level detection, as pre-registered in prereg/K1.md.

  python scripts/k1.py --dataset ssv2 [--camera-covariate] [--out results/K1_ssv2]

Index columns: clip_id, one_way (or fell), optional cls, symmetric.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pb import cache, stats  # noqa: E402

PRIMARY = "vjepa2_vitg"
REPLICATIONS = ["vjepa2_vitl", "vjepa2_1_vitg"]
PRIMARY_CFG = "s3o0_w4x1"
VARIANT_CFGS = ["s3o0_w8x2", "s1o0_w4x1"]
LTX = ("ltx2b", "native", "mse")


def load_clip(dataset, model, tag, cid, err):
    if not cache.exists(dataset, model, tag, cid):
        return None
    f = cache.load(dataset, model, tag, cid)
    if int(f.get("too_short", 0)):
        return None
    return stats.clip_stats(f, err)


def evaluate(A, F, y, pairs_idx):
    """AUROC(A), AUROC(F) (orientation fixed on the full sample), Δ, with paired bootstrap."""
    auc_F_raw = stats.auroc(F, y)
    flip = auc_F_raw < 0.5
    Fo = -F if flip else F
    pos = np.array([p for p, _ in pairs_idx])
    neg = np.array([n for _, n in pairs_idx])

    def on(ix):
        sel = np.concatenate([pos[ix], neg[ix]])
        a, f = stats.auroc(A[sel], y[sel]), stats.auroc(Fo[sel], y[sel])
        return a, f, a - f

    a, f, d = on(np.arange(len(pos)))
    draws = np.array(stats.bootstrap(on, len(pos)))
    return dict(auroc_A=a, auroc_A_ci=stats.ci(draws[:, 0]), auroc_F=f, auroc_F_ci=stats.ci(draws[:, 1]),
                auroc_F_flipped=bool(flip), delta=d, delta_ci=stats.ci(draws[:, 2]), _draws=draws)


def passes(r):
    lo, hi = r["delta_ci"]
    return bool(r["delta"] >= 0.10 and (lo > 0 or hi < 0) and r["auroc_A"] >= 0.70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--camera-covariate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"/mnt/ssd1/devashri/CVPR/results/K1_{a.dataset}"
    os.makedirs(out, exist_ok=True)

    idx = pd.read_csv(f"/mnt/ssd1/devashri/CVPR/data/{a.dataset}/index.csv", dtype={"clip_id": str})
    ycol = "one_way" if "one_way" in idx.columns else "fell"
    idx["y"] = idx[ycol].astype(str).str.lower().isin(["1", "true"])
    if "cls" not in idx.columns:
        idx["cls"] = np.where(idx.y, "one_way", "reversible")
    if "symmetric" not in idx.columns:
        idx["symmetric"] = False
    fpath = os.path.join(cache.ROOT, a.dataset, "_filters.csv")
    filt = pd.read_csv(fpath, dtype={"clip_id": str}).drop_duplicates("clip_id") if os.path.exists(fpath) else pd.DataFrame(columns=["clip_id", "reason"])
    report = dict(dataset=a.dataset, n_index=len(idx),
                  filtered=filt.reason.str.split("@").str[0].value_counts().to_dict())
    idx = idx[~idx.clip_id.isin(filt.clip_id)]

    # flow + length; eligibility is model-independent (same tubelet count for every model)
    rows = []
    for r in idx.itertuples():
        if not cache.exists(a.dataset, "raft", "native", r.clip_id):
            continue
        fl = cache.load(a.dataset, "raft", "native", r.clip_id)
        s = load_clip(a.dataset, PRIMARY, PRIMARY_CFG, r.clip_id, "l2")
        if s is None:
            continue
        rows.append(dict(clip_id=r.clip_id, y=r.y, cls=r.cls, symmetric=bool(r.symmetric),
                         F=float(fl["f"].mean()), cam=float(fl["cam"].mean()), length=int(fl["n_native"])))
    clips = pd.DataFrame(rows)
    too_short = len(idx) - len(clips)
    report["n_eligible"] = len(clips)
    report["n_not_eligible_length_or_missing"] = int(too_short)
    report["too_short_by_class"] = (idx[~idx.clip_id.isin(clips.clip_id)].cls.value_counts().to_dict())

    pos_ix = np.where(clips.y)[0]
    neg_ix = np.where(~clips.y)[0]
    pairs = stats.motion_match(clips.F.values[pos_ix], clips.F.values[neg_ix])
    pairs = [(int(pos_ix[p]), int(neg_ix[n])) for p, n in pairs]
    report["n_pairs"] = len(pairs)
    report["n_unmatched_one_way"] = int(len(pos_ix) - len(pairs))
    report["inconclusive"] = len(pairs) < 50
    matched = sorted({i for p in pairs for i in p})
    m = clips.iloc[matched].reset_index(drop=True)
    remap = {old: new for new, old in enumerate(matched)}
    pairs_m = [(remap[p], remap[n]) for p, n in pairs]
    m.to_csv(os.path.join(out, "matched_clips.csv"), index=False)
    y, F = m.y.values, m.F.values

    models = [(PRIMARY, c, e) for c in [PRIMARY_CFG] + VARIANT_CFGS for e in ("l2", "l1")]
    models += [(r_, PRIMARY_CFG, "l2") for r_ in REPLICATIONS]
    models += [LTX]
    res, A_by = {}, {}
    for model, tag, err in models:
        st = [load_clip(a.dataset, model, tag, c, err) for c in m.clip_id]
        ok = np.array([s is not None for s in st])
        if not ok.any():
            res[f"{model}/{tag}/{err}"] = dict(status="no features")
            continue
        key = f"{model}/{tag}/{err}"
        entry = dict(n_clips=int(ok.sum()))
        keep_pairs = [(p, n) for p, n in pairs_m if ok[p] and ok[n]]
        for stat in ("A", "A_max", "A_raw", "A_raw_max"):
            A = np.array([s[stat] if s else np.nan for s in st])
            r = evaluate(A, F, y, keep_pairs)
            if stat == "A":
                A_by[key] = (A, r.pop("_draws"))
            else:
                r.pop("_draws")
            entry[stat] = r
        entry["pass"] = passes(entry["A"]) and not report["inconclusive"]
        yo = np.array([s["yocausal"] if s else np.nan for s in st])
        entry["yocausal_frac"] = {g: float(np.nanmean(yo[(y == v) & ok])) for g, v in (("one_way", True), ("reversible", False))}
        A = A_by[key][0]
        entry["per_class_auroc"] = {c: stats.auroc(np.concatenate([A[(m.cls == c).values & ok], A[~y & ok]]),
                                                   np.concatenate([np.ones(((m.cls == c).values & ok).sum()), np.zeros((~y & ok).sum())]))
                                    for c in sorted(m.cls[y].unique())}
        sym = m.symmetric.values
        entry["auroc_A_vs_symmetric_only"] = stats.auroc(A[(y | sym) & ok], y[(y | sym) & ok])
        entry["auroc_A_excluding_symmetric"] = stats.auroc(A[~sym & ok], y[~sym & ok])
        if a.camera_covariate:
            resid = np.full(len(A), np.nan)
            resid[ok] = stats.ols_residual(A[ok], np.column_stack([m.cam.values[ok], m.length.values[ok]]))
            entry["A_cam_len_regressed"] = evaluate(resid, F, y, keep_pairs)
            entry["A_cam_len_regressed"].pop("_draws")
        entry["auroc_length"] = stats.auroc(m.length.values, y)
        res[key] = entry

    # paired comparison vs LTX-Video on the same pairs
    ltx_key = "/".join(LTX)
    if ltx_key in A_by:
        A_ltx = A_by[ltx_key][0]
        for model in [PRIMARY] + REPLICATIONS:
            k = f"{model}/{PRIMARY_CFG}/l2"
            if k not in A_by:
                continue
            A_j = A_by[k][0]
            pp = [(p, n) for p, n in pairs_m if np.isfinite(A_j[[p, n]]).all() and np.isfinite(A_ltx[[p, n]]).all()]
            pos = np.array([p for p, _ in pp]); neg = np.array([n for _, n in pp])

            def d(ix):
                sel = np.concatenate([pos[ix], neg[ix]])
                return stats.auroc(A_j[sel], y[sel]) - stats.auroc(A_ltx[sel], y[sel])
            res[k]["auroc_minus_ltx"] = dict(value=d(np.arange(len(pos))), ci=stats.ci(stats.bootstrap(d, len(pos))), n_pairs=len(pp))

    report["results"] = res
    report["decision"] = ("INCONCLUSIVE (<50 pairs)" if report["inconclusive"] else
                          ("PASS" if res.get(f"{PRIMARY}/{PRIMARY_CFG}/l2", {}).get("pass") else "FAIL"))
    with open(os.path.join(out, "k1.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print_summary(report)


def fmt(v, c):
    return f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def print_summary(rep):
    print(f"K1 {rep['dataset']}: eligible {rep['n_eligible']}, pairs {rep['n_pairs']}, filtered {rep['filtered']}")
    for k, e in rep["results"].items():
        if "A" not in e:
            print(f"  {k}: {e.get('status')}")
            continue
        r = e["A"]
        print(f"  {k}: AUROC(A) {fmt(r['auroc_A'], r['auroc_A_ci'])}  AUROC(F) {fmt(r['auroc_F'], r['auroc_F_ci'])}  "
              f"Δ {fmt(r['delta'], r['delta_ci'])}  pass={e['pass']}")
    print("DECISION (primary model):", rep["decision"])


if __name__ == "__main__":
    main()
