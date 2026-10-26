"""Pre-K1 sanity check: predictor error vs copy-last-tubelet and context-mean baselines on real clips.

  python scripts/sanity_predictor.py --dataset ssv2 --model vjepa2_vitg --n 50 --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pb import stats, video  # noqa: E402
from pb.vjepa import VJEPA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="vjepa2_vitg")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    idx = pd.read_csv(f"/mnt/ssd1/devashri/CVPR/data/{a.dataset}/index.csv", dtype={"clip_id": str})
    idx = idx.sample(frac=1, random_state=0)
    m = VJEPA(a.model, device=a.gpu)
    c, P = 4, m.P
    per_clip = []
    for r in idx.itertuples():
        if len(per_clip) >= a.n:
            break
        fr, _ = video.read_video(r.path)
        if video.check_shortcuts(fr):
            continue
        s, _ = video.sample(fr, 3, 0)
        x = video.to_tensor(s, m.img)
        if x.shape[1] // 2 < c + 1:
            continue
        e = []
        for j in range(c, x.shape[1] // 2):
            w = x[:, 2 * (j - c):2 * (j + 1)].unsqueeze(0).to(m.device, torch.float16)
            ctx = torch.arange(0, c * P, device=m.device)[None]
            tg = torch.arange(c * P, (c + 1) * P, device=m.device)[None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = m._encode_target(w)
                p = m._predict(m._encode_context(w, ctx), [ctx], [tg])
            ht = h[:, c * P:].float()
            e.append([(p.float() - ht).norm(dim=-1).mean().item(),
                      (h[:, (c - 1) * P:c * P].float() - ht).norm(dim=-1).mean().item(),
                      (h[:, :c * P].float().view(1, c, P, -1).mean(1) - ht).norm(dim=-1).mean().item()])
        per_clip.append(np.mean(e, 0))
    E = np.array(per_clip)
    out = {}
    for i, k in enumerate(("predictor", "copy_last", "context_mean")):
        out[k] = dict(mean=float(E[:, i].mean()), ci=stats.ci(stats.bootstrap(lambda ix: E[ix, i].mean(), len(E))))
    for i, k in ((1, "copy_last"), (2, "context_mean")):
        d = E[:, i] - E[:, 0]
        out[f"{k}_minus_predictor"] = dict(mean=float(d.mean()), ci=stats.ci(stats.bootstrap(lambda ix: d[ix].mean(), len(d))))
    out["n_clips"] = len(E)
    os.makedirs("/mnt/ssd1/devashri/CVPR/results", exist_ok=True)
    with open(f"/mnt/ssd1/devashri/CVPR/results/sanity_{a.dataset}_{a.model}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
