"""Synthetic RoboLab-format tarball for smoke tests only (never reported as results).

A gripper pushes a cube along a table. In fall episodes the cube is pushed past
the edge and falls; the certificate frame is the first frame the cube has no
support. In non-fall episodes the push stops short. 15 Hz, static camera.
"""
import argparse
import os
import tarfile

import av
import numpy as np
import pandas as pd


def render(T, fall, rng, H=240, W=320):
    table_y, edge_x = 150, int(rng.integers(200, 240))
    cube = 24
    x0 = float(rng.integers(60, 100))
    push_start, speed = int(rng.integers(15, 30)), rng.uniform(2.0, 3.5)
    stop_x = (edge_x + cube) if fall else (edge_x - cube - rng.integers(10, 40))
    bg = (rng.normal(0, 6, (H, W, 3)) + 200).clip(0, 255)
    frames, cert = [], -1
    cx, cy, vy = x0, table_y - cube, 0.0
    for t in range(T):
        f = bg.copy()
        f[table_y:table_y + 12, :edge_x] = (120, 80, 40)  # table top
        f[table_y + 12:, 30:42] = (100, 70, 35)           # leg
        if t >= push_start and cx < stop_x:
            cx = min(stop_x, cx + speed)
        supported = cx < edge_x
        if not supported:
            if cert < 0:
                cert = t
            vy += 1.5
            cy = min(H - cube, cy + vy)
        gx = int(min(cx, stop_x)) - 40 if t >= push_start else int(x0) - 60
        f[int(table_y - cube - 6):table_y - 4, max(0, gx):max(0, gx + 36)] = (90, 90, 95)  # gripper
        f[int(cy):int(cy) + cube, int(cx):int(cx) + cube] = (40, 60, 200)                 # cube
        frames.append(f.clip(0, 255).astype(np.uint8))
    return np.stack(frames), cert


def write_mp4(path, frames, fps=15):
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.height, s.width, s.pix_fmt = frames.shape[1], frames.shape[2], "yuv420p"
        s.options = {"crf": "18"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--T", type=int, default=120)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    vdir = os.path.join(a.out, "episodes")
    os.makedirs(vdir, exist_ok=True)
    rows = []
    for i in range(a.n):
        fall = i % 2 == 0
        frames, cert = render(a.T, fall, rng)
        eid = f"ep{i:04d}"
        write_mp4(os.path.join(vdir, f"{eid}.mp4"), frames)
        rows.append(dict(episode_id=eid, task=f"push_cube_{i // 4}", fell=int(fall),
                         certificate_frame=cert if fall else "", fps=15))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "certificates.csv"), index=False)
    with tarfile.open(os.path.join(a.out, "robolab_synthetic.tar.gz"), "w:gz") as tf:
        tf.add(vdir, arcname="episodes")
    print("wrote", a.n, "episodes to", a.out)


if __name__ == "__main__":
    main()
