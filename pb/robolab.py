"""RoboLab (host 84) loader: a tarball of episode videos plus a CSV of certificate frames.

The exact column names are not fixed yet, so each field accepts a list of aliases
(first match wins). Required: an episode id, and either a fall flag or a
certificate-frame column (empty / negative / NaN = no fall). Optional: video
path (else found by episode id inside the extracted tarball), task, fps,
frame-index base (default 0).
"""
import os
import tarfile

import numpy as np
import pandas as pd

ALIASES = {
    "id": ["episode_id", "episode", "clip_id", "id", "ep_id"],
    "cert": ["certificate_frame", "cert_frame", "t_star", "tstar", "certificate", "frame", "event_frame"],
    "fell": ["fell", "fall", "left_support", "one_way", "label", "event"],
    "video": ["video", "video_path", "path", "file", "filename"],
    "task": ["task", "task_name", "env"],
    "fps": ["fps", "hz", "rate"],
}
VIDEO_EXT = (".mp4", ".avi", ".webm", ".mkv", ".mov")
TRUE_STR = {"1", "true", "yes", "fall", "fell", "one_way", "oneway", "y"}


def _col(df, field, required=False):
    for a in ALIASES[field]:
        for c in df.columns:
            if c.strip().lower() == a:
                return c
    if required:
        raise KeyError(f"CSV has no column for '{field}' (tried {ALIASES[field]}); columns: {list(df.columns)}")
    return None


def extract(tar_path, out_dir):
    """Extract once; refuses members that escape out_dir."""
    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir, filter="data")
    open(marker, "w").close()
    return out_dir


def _index_videos(root):
    """episode stem -> path, for video files and for leaf dirs of frames."""
    idx = {}
    for d, subdirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXT):
                idx.setdefault(os.path.splitext(f)[0], os.path.join(d, f))
        if not subdirs and any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
            idx.setdefault(os.path.basename(d), d)
    return idx


def load_index(csv_path, root, default_fps=15.0):
    """Return a DataFrame: clip_id, path, task, fell (bool), t_star (0-based native frame or -1), fps."""
    df = pd.read_csv(csv_path)
    c_id = _col(df, "id", required=True)
    c_cert, c_fell = _col(df, "cert"), _col(df, "fell")
    if c_cert is None and c_fell is None:
        raise KeyError("CSV needs a certificate-frame column or a fall-flag column")
    c_vid, c_task, c_fps = _col(df, "video"), _col(df, "task"), _col(df, "fps")
    base = int(df["frame_index_base"].iloc[0]) if "frame_index_base" in df.columns else 0
    vids = _index_videos(root)

    rows, missing = [], []
    for _, r in df.iterrows():
        cid = str(r[c_id])
        if c_vid is not None and isinstance(r[c_vid], str):
            p = r[c_vid] if os.path.isabs(r[c_vid]) else os.path.join(root, r[c_vid])
        else:
            p = vids.get(cid)
        if p is None or not os.path.exists(p):
            missing.append(cid)
            continue
        cert = pd.to_numeric(r[c_cert], errors="coerce") if c_cert else np.nan
        has_cert = bool(np.isfinite(cert) and cert >= 0)
        if c_fell is not None:
            fell = str(r[c_fell]).strip().lower() in TRUE_STR
        else:
            fell = has_cert
        if fell and not has_cert:
            raise ValueError(f"episode {cid}: marked as fall but has no certificate frame")
        rows.append(dict(
            clip_id=cid, path=p,
            task=str(r[c_task]) if c_task else "",
            fell=fell,
            t_star=int(cert) - base if (fell and has_cert) else -1,
            fps=float(r[c_fps]) if c_fps and pd.notna(r[c_fps]) else default_fps,
        ))
    out = pd.DataFrame(rows)
    out.attrs["missing_videos"] = missing
    return out
