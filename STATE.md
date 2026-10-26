# STATE

Last updated: 2026-10-25 evening (day 0). After any context compaction, re-read STATE.md, RESULTS.md and CORRECTIONS.md before continuing.

## Environment
- Conda env `paperB`: python 3.12, torch 2.5.1+cu121, diffusers 0.34.0, transformers 4.53.2. Rebuild with `env/build_env.sh`.
- vjepa2 code: `third_party/vjepa2` at commit 204698b, exposed to the env through a `.pth` path file.
- GPUs: 2× RTX 2080 Ti, 11 GB each. fp16 throughout.

## Weights
| Model | File | Status |
|---|---|---|
| V-JEPA 2 ViT-g (PRIMARY) | vjepa2_ckpts/vitg.pt | ok; smoke-tested, 5.8 GB peak |
| V-JEPA 2 ViT-L | vjepa2_ckpts/vitl.pt | ok; smoke-tested, 1.7 GB peak |
| V-JEPA 2.1 ViT-g | vjepa2_ckpts/vjepa2_1_vitg_384.pt | ok at 384 px (5.8 GB peak); hierarchical 4-level target |
| V-JEPA 2-AC ViT-g | vjepa2_ckpts/vjepa2-ac-vitg.pt | present; not in kill tests |
| LTX-Video 2B | weights/LTX-Video | ok; null-caption embedding cached at weights/ltx_null_caption.pt; smoke-tested |
| RAFT-large | torchvision | ok |
| DINO-WM | — | dropped (no real-video checkpoint) |

## Code (all smoke-tested on synthetic clips; synthetic numbers are never results)
- `pb/`: video I/O and filters, V-JEPA errors, LTX loss, RAFT, cache, stats, RoboLab loader.
- `scripts/prepare_robolab.py --tar X --csv Y`: extracts the tarball and writes data/robolab/index.csv. CSV columns are matched by alias, see pb/robolab.py.
- `scripts/prepare_ssv2.py --labels D --videos V`: writes class_mapping.csv, which must be committed before K1, and index.csv.
- `scripts/run_dataset.sh <dataset>`: phase 1 ViT-g + RAFT, phase 2 ViT-L + 2.1, phase 3 LTX.
- `scripts/sanity_predictor.py`: predictor vs copy baselines on 50 real clips, run before K1.
- `scripts/k1.py`, `scripts/k2.py`: pre-registered analyses, written to results/K{1,2}_<dataset>/.

## Data access
| Source | Status |
|---|---|
| RoboLab (host 84) | coming as a tarball plus certificate CSV; loader ready |
| Something-Something v2 | waiting on registration (user) |
| Ego4D PNR | waiting on licence (user); not waited on |
| APT / HowToChange / ChangeIt | not checked |

## Feature cache
Only `features/synthetic_robolab/` (smoke). No real-data features yet. Layout: FEATURES.md.

## Running
Nothing.

## Order once data arrives
1. K1 on whichever of SSv2 / RoboLab arrives first.
   - SSv2: run prepare_ssv2, then commit class_mapping.csv.
   - Then run_dataset.sh, then sanity_predictor, then k1.py.
2. K2 on RoboLab: k2.py after phase 1.
3. LTX-Video on the same clips last (phase 3); then rerun k1/k2 to add the LTX rows.

## Throughput (from smoke tests)
- RoboLab, ~400 clips at ~8 s: ViT-g ~70 s per clip for all 7 configs, most of it stride 1. That is ~4 h on 2 GPUs, and ~1.5 h for the primary configs only.
- SSv2 clips are shorter, so cost is lower per clip; total depends on the clip count after mapping.
