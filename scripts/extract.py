"""Cache per-clip features for one model on one GPU.

  python scripts/extract.py --dataset robolab --model vjepa2_vitg --gpu 0 \
      --configs s3o0w4x1,s3o1w4x1,s3o2w4x1 [--shard 0/2]

Index: data/<dataset>/index.csv with at least clip_id, path.
Models: vjepa2_vitg | vjepa2_vitl | vjepa2_1_vitg | raft | ltx
"""
import argparse
import logging
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pb import cache, video  # noqa: E402

log = logging.getLogger("extract")


def parse_configs(s):
    out = []
    for c in s.split(","):
        m = re.fullmatch(r"s(\d+)o(\d+)w(\d+)x(\d+)", c.strip())
        if not m:
            raise ValueError(f"bad config {c!r}; expected e.g. s3o0w4x1")
        out.append(tuple(map(int, m.groups())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--configs", default="s3o0w4x1")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img", type=int, default=None, help="override resolution (2.1 fallback to 256)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    idx = pd.read_csv(f"/mnt/ssd1/devashri/CVPR/data/{a.dataset}/index.csv", dtype={"clip_id": str})
    k, n = map(int, a.shard.split("/"))
    idx = idx.iloc[k::n]
    if a.limit:
        idx = idx.head(a.limit)
    done_filters = set()
    fpath = os.path.join(cache.ROOT, a.dataset, "_filters.csv")
    if os.path.exists(fpath):
        done_filters = set(pd.read_csv(fpath, dtype={"clip_id": str}).clip_id)

    if a.model == "raft":
        from pb.flow import Flow
        m, mname = Flow(device=a.gpu), "raft"
    elif a.model == "ltx":
        from pb.ltx import LTX
        m, mname = LTX(device=a.gpu), "ltx2b"
    else:
        from pb.vjepa import VJEPA
        m = VJEPA(a.model, device=a.gpu, img=a.img)
        mname = a.model if a.img is None else f"{a.model}_{a.img}px"
    cfgs = parse_configs(a.configs) if a.model not in ("raft", "ltx") else [None]

    t0, n_done = time.time(), 0
    for i, r in enumerate(idx.itertuples()):
        cid = r.clip_id
        if cid in done_filters:
            continue
        tags = ["native" if c is None else cache.config_tag(c[0], c[1], c[2], c[3]) for c in cfgs]
        todo = [(c, t) for c, t in zip(cfgs, tags) if not cache.exists(a.dataset, mname, t, cid)]
        if not todo:
            continue
        frames, vfps = video.read_video(r.path)
        fps = float(getattr(r, "fps", None) or vfps or 15.0)
        reason = video.check_shortcuts(frames)
        if reason:
            cache.log_filter(a.dataset, cid, reason)
            done_filters.add(cid)
            continue
        for c, tag in todo:
            if a.model == "raft":
                f, cam = m.clip_flow(frames)
                cache.save(a.dataset, mname, tag, cid, n_native=len(frames), f=f, cam=cam)
                continue
            if a.model == "ltx":
                cache.save(a.dataset, mname, tag, cid, n_native=len(frames), **m.clip_loss(frames, fps=fps))
                continue
            stride, offset, cw, ow = c
            s_frames, nat = video.sample(frames, stride, offset)
            if len(s_frames) // 2 < 2 * cw + 1:
                cache.save(a.dataset, mname, tag, cid, n_native=len(frames), too_short=1)
                continue
            r_ = m.clip_errors(s_frames, c=cw, out=ow, batch=a.batch)
            centres = (nat[0::2] + nat[1::2]) / 2.0
            cache.save(a.dataset, mname, tag, cid, n_native=len(frames), too_short=0,
                       tubelet_centre=centres, stride=stride, offset=offset,
                       **{kk: v for kk, v in r_.items()})
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            log.info("%d/%d clips, %.1fs/clip", i + 1, len(idx), el / n_done)
            cache.log_disk()
    cache.log_disk()
    log.info("done: %d clips in %.0fs", n_done, time.time() - t0)


if __name__ == "__main__":
    main()
