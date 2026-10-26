"""Something-Something v2: apply the pre-registered class mapping and write the index.

  python scripts/prepare_ssv2.py --labels <dir with labels.json, validation.json> --videos <dir of <id>.webm>

Writes data/ssv2/class_mapping.csv (every template with its bucket and reason; commit
before K1) and data/ssv2/index.csv (validation clips in the one-way / reversible buckets).
The rule is prereg/K1.md; this script is its only implementation.
"""
import argparse
import json
import os
import re

import pandas as pd

OUT = "/mnt/ssd1/devashri/CVPR/data/ssv2"
ONE_WAY = ("tear", "pour", "break", "spill", "crush")
REVERSIBLE = ("moving", "pushing", "pulling", "rotating", "lifting", "putting")
SYMMETRIC = ("rotating", "turning", "spinning")


def bucket(template):
    t = template.lower().replace("[", "").replace("]", "")
    words = set(re.findall(r"[a-z]+", t))
    if t.startswith("pretending"):
        return "excluded", "pretending"
    if "almost" in words:
        return "excluded", "almost"
    if "but" in words:
        return "excluded", "but"
    if any(w.startswith("drop") for w in words):
        return "excluded", "drop onto/into/next to a surface"
    hit = [k for k in ONE_WAY if any(w.startswith(k) for w in words)]
    if hit:
        return "one_way", f"keyword:{hit[0]}"
    if "off" in words and any(w.startswith(("roll", "fall")) for w in words):
        return "one_way", "leaves surface (roll/fall + off)"
    rv = [k for k in REVERSIBLE if k in words]
    if "lifting" in rv or "putting" in rv:
        if not ("lifting" in words or ("putting" in words and "down" in words)):
            rv = [k for k in rv if k not in ("lifting", "putting")]
    if rv:
        return "reversible", f"keyword:{rv[0]}"
    return "unused", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--videos", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    templates = list(json.load(open(os.path.join(a.labels, "labels.json"))).keys())
    rows = []
    for tpl in sorted(templates):
        b, why = bucket(tpl)
        sym = b == "reversible" and any(s in tpl.lower() for s in SYMMETRIC)
        rows.append(dict(template=tpl, bucket=b, reason=why, symmetric=sym))
    cm = pd.DataFrame(rows)
    cm.to_csv(os.path.join(OUT, "class_mapping.csv"), index=False)
    print(cm.bucket.value_counts().to_string())

    val = json.load(open(os.path.join(a.labels, "validation.json")))
    tmap = cm.set_index("template")
    idx, missing = [], 0
    for v in val:
        tpl = v["template"]
        if tpl not in tmap.index or tmap.loc[tpl, "bucket"] not in ("one_way", "reversible"):
            continue
        p = os.path.join(a.videos, f"{v['id']}.webm")
        if not os.path.exists(p):
            missing += 1
            continue
        idx.append(dict(clip_id=v["id"], path=p, cls=tpl, one_way=tmap.loc[tpl, "bucket"] == "one_way",
                        symmetric=bool(tmap.loc[tpl, "symmetric"]), fps=12.0))
    df = pd.DataFrame(idx)
    df.to_csv(os.path.join(OUT, "index.csv"), index=False)
    print(f"index: {len(df)} clips ({df.one_way.sum()} one-way, {(~df.one_way).sum()} reversible); {missing} videos missing")


if __name__ == "__main__":
    main()
