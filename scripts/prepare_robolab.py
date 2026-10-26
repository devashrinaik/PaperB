"""Extract a RoboLab tarball and write data/<dataset>/index.csv.

  python scripts/prepare_robolab.py --tar <tarball> --csv <certificates.csv> [--dataset robolab]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pb import robolab  # noqa: E402

DATA = "/mnt/ssd1/devashri/CVPR/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="robolab")
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    out = os.path.join(DATA, a.dataset)
    root = robolab.extract(a.tar, os.path.join(out, "raw"))
    df = robolab.load_index(a.csv, root, default_fps=a.fps)
    df.to_csv(os.path.join(out, "index.csv"), index=False)
    miss = df.attrs["missing_videos"]
    print(f"{len(df)} episodes indexed ({df.fell.sum()} fall, {(~df.fell).sum()} non-fall); "
          f"{len(miss)} CSV rows without a video: {miss[:10]}")
    if df.fps.nunique() > 1 or df.fps.iloc[0] != 15:
        print("WARNING: fps differs from the 15 Hz the brief assumes:", df.fps.value_counts().to_dict())


if __name__ == "__main__":
    main()
