"""K2: localisation of the point of no return, as pre-registered in prereg/K2.md.

  python scripts/k2.py --dataset robolab [--out results/K2_robolab]

Uses clips with fell==True and t_star >= 0 (native, 0-based).
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
TOL = 8


def load_streams(dataset, model, tags, cid):
    fs = []
    for t in tags:
        if not cache.exists(dataset, model, t, cid):
            return None
        f = cache.load(dataset, model, t, cid)
        if int(f.get("too_short", 0)):
            return None
        fs.append(f)
    return fs


def ltx_series(dataset, cid, normalise=True):
    if not cache.exists(dataset, "ltx2b", "native", cid):
        return None
    f = cache.load(dataset, "ltx2b", "native", cid)
    a, m = stats.asym(f, "mse", normalise)
    return np.floor(f["latent_centre"][m] + 0.5).astype(int), a[m]


def summarise(rows):
    """rows: list of dict(err=t̂−t*, chance=..., length=...). Bootstrap hit rate over clips."""
    e = np.array([r["err"] for r in rows])
    hit = np.abs(e) <= TOL
    ch = np.array([r["chance"] for r in rows])
    L = np.array([r["length"] for r in rows])
    hit_draws = stats.bootstrap(lambda ix: hit[ix].mean(), len(hit))
    lead = e < 0
    lag = e > 0
    out = dict(n=len(e), hit_rate=float(hit.mean()), hit_rate_ci=stats.ci(hit_draws),
               chance=float(ch.mean()), chance_ci=stats.ci(stats.bootstrap(lambda ix: ch[ix].mean(), len(ch))),
               lead_frac=float(lead.mean()), lead_ci=stats.ci(stats.bootstrap(lambda ix: lead[ix].mean(), len(e))),
               lag_frac=float(lag.mean()), exact_frac=float((e == 0).mean()),
               err_quantiles={q: float(np.percentile(e, q)) for q in (5, 25, 50, 75, 95)},
               err_hist=np.histogram(e, bins=np.arange(-60, 61, 4))[0].tolist())
    for lo, hi, name in ((0, 60, "<60"), (60, 120, "60-120"), (120, 10**9, ">120")):
        sel = (L >= lo) & (L < hi)
        out[f"bin_{name}"] = dict(n=int(sel.sum()), hit_rate=float(hit[sel].mean()) if sel.any() else None,
                                  chance=float(ch[sel].mean()) if sel.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"/mnt/ssd1/devashri/CVPR/results/K2_{a.dataset}"
    os.makedirs(out, exist_ok=True)
    idx = pd.read_csv(f"/mnt/ssd1/devashri/CVPR/data/{a.dataset}/index.csv", dtype={"clip_id": str})
    idx = idx[idx.fell.astype(str).str.lower().isin(["1", "true"]) & (idx.t_star >= 0)]
    fpath = os.path.join(cache.ROOT, a.dataset, "_filters.csv")
    filt = pd.read_csv(fpath, dtype={"clip_id": str}).drop_duplicates("clip_id") if os.path.exists(fpath) else pd.DataFrame(columns=["clip_id", "reason"])
    rep = dict(dataset=a.dataset, n_certified=len(idx), n_filtered=int(idx.clip_id.isin(filt.clip_id).sum()))
    idx = idx[~idx.clip_id.isin(filt.clip_id)]

    merged = [f"s3o{o}_w4x1" for o in range(3)]
    variants = {
        "primary": (merged, "l2", True, "peak"),
        "secondary_first_exceed95": (merged, "l2", True, "first"),
        "raw": (merged, "l2", False, "peak"),
        "offset0_only": (["s3o0_w4x1"], "l2", True, "peak"),
        "stride1": (["s1o0_w4x1"], "l2", True, "peak"),
        "w8x2": ([f"s3o{o}_w8x2" for o in range(3)], "l2", True, "peak"),
    }
    res = {}
    for model in [PRIMARY] + REPLICATIONS + ["ltx2b"]:
        for vname, (tags, err, norm, rule) in variants.items():
            if model == "ltx2b" and vname not in ("primary", "raw", "secondary_first_exceed95"):
                continue
            rows, flow_rows, excl = [], [], {"t_star_outside_defined": 0, "no_features": 0}
            for r in idx.itertuples():
                if model == "ltx2b":
                    s = ltx_series(a.dataset, r.clip_id, norm)
                    if s is None:
                        excl["no_features"] += 1
                        continue
                    t, v = s
                else:
                    fs = load_streams(a.dataset, model, tags, r.clip_id)
                    if fs is None:
                        excl["no_features"] += 1
                        continue
                    t, v = stats.merged_series(fs, err, norm)
                lo, hi = int(t.min()), int(t.max())
                if not (lo <= r.t_star <= hi):
                    excl["t_star_outside_defined"] += 1
                    continue
                th = stats.predict_peak(t, v) if rule == "peak" else stats.predict_first_exceed(t, v)
                fl = cache.load(a.dataset, "raft", "native", r.clip_id)["f"]
                tf = lo + int(np.argmax(fl[lo:min(hi, len(fl) - 1) + 1]))
                ch = stats.chance_hit(lo, hi, int(r.t_star), TOL)
                L = int(fl.shape[0] + 1)
                rows.append(dict(err=th - r.t_star, chance=ch, length=L))
                flow_rows.append(dict(err=tf - r.t_star, chance=ch, length=L))
            if not rows:
                res[f"{model}/{vname}"] = dict(status="no clips", excluded=excl)
                continue
            e = summarise(rows)
            fb = summarise(flow_rows)
            e["flow_baseline"] = {k: fb[k] for k in ("hit_rate", "hit_rate_ci", "err_quantiles")}
            e["excluded"] = excl
            e["pass"] = bool(e["hit_rate"] > 0.60 and fb["hit_rate"] < 0.25)
            res[f"{model}/{vname}"] = e
    rep["results"] = res
    p = res.get(f"{PRIMARY}/primary", {})
    rep["decision"] = "PASS" if p.get("pass") else ("FAIL" if "hit_rate" in p else "NO DATA")
    with open(os.path.join(out, "k2.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"K2 {a.dataset}: certified {rep['n_certified']}, filtered {rep['n_filtered']}")
    for k, e in res.items():
        if "hit_rate" not in e:
            print(f"  {k}: {e.get('status')} {e.get('excluded')}")
            continue
        h, c, fbh = e["hit_rate"], e["hit_rate_ci"], e["flow_baseline"]
        print(f"  {k}: n={e['n']} hit {h:.3f} [{c[0]:.3f},{c[1]:.3f}]  flow {fbh['hit_rate']:.3f} "
              f"[{fbh['hit_rate_ci'][0]:.3f},{fbh['hit_rate_ci'][1]:.3f}]  chance {e['chance']:.3f}  "
              f"lead {e['lead_frac']:.2f}  median err {e['err_quantiles'][50]:+.1f}  pass={e['pass']}")
    print("DECISION (primary model):", rep["decision"])


if __name__ == "__main__":
    main()
