"""Feature store: one .npz per (dataset, model, config, clip). Never overwritten.

Layout (documented in FEATURES.md):
  features/<dataset>/<model>/<config>/<clip_id>.npz
  features/<dataset>/_filters.csv   clip_id, reason (shortcut / length removals)
  features/disk_usage.log
"""
import os
import shutil
import time

import numpy as np

ROOT = "/mnt/ssd1/devashri/CVPR/features"


def config_tag(stride=None, offset=None, c=None, out=None, extra=""):
    parts = []
    if stride is not None:
        parts.append(f"s{stride}o{offset}")
    if c is not None:
        parts.append(f"w{c}x{out}")
    if extra:
        parts.append(extra)
    return "_".join(parts) or "default"


def path(dataset, model, tag, clip_id):
    return os.path.join(ROOT, dataset, model, tag, f"{clip_id}.npz")


def exists(*a):
    return os.path.exists(path(*a))


def save(dataset, model, tag, clip_id, **arrays):
    p = path(dataset, model, tag, clip_id)
    if os.path.exists(p):
        raise FileExistsError(f"refusing to overwrite cached features {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, p)


def load(dataset, model, tag, clip_id):
    with np.load(path(dataset, model, tag, clip_id), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def log_filter(dataset, clip_id, reason):
    p = os.path.join(ROOT, dataset, "_filters.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("clip_id,reason\n")
        f.write(f"{clip_id},{reason}\n")


def log_disk():
    os.makedirs(ROOT, exist_ok=True)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(ROOT) for f in fs)
    free = shutil.disk_usage(ROOT).free
    with open(os.path.join(ROOT, "disk_usage.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} features={size / 2**30:.2f}GiB disk_free={free / 2**30:.0f}GiB\n")
