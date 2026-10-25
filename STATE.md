# STATE

Last updated: 2026-10-25 (day 0)

## Environment
- Conda env `paperB` (python 3.12, torch 2.5.1+cu121). Build script `env/build_env.sh`, log `env/build_env.log`.
- vjepa2 code: fresh clone at `third_party/vjepa2`, pinned to commit 204698b (includes V-JEPA 2.1).
- GPUs: 2x RTX 2080 Ti, 11 GB each. No bf16 on Turing, so inference runs in fp16.

## Weights (in /mnt/ssd1/devashri/vjepa2_ckpts, sha256 beside each file)
| Model | File | Status |
|---|---|---|
| V-JEPA 2 ViT-L | vitl.pt (5.1 GB) | done; encoder+predictor+target_encoder verified |
| V-JEPA 2 ViT-g | vitg.pt (16.5 GB) | done; encoder+predictor+target_encoder verified |
| V-JEPA 2-AC ViT-g | vjepa2-ac-vitg.pt | present (downloaded Jul 2026) |
| V-JEPA 2.1 ViT-L (distilled) | vjepa2_1_vitl_dist_vitG_384.pt | done; predictor present (target = ema_encoder); unused fallback |
| V-JEPA 2.1 ViT-g | vjepa2_1_vitg_384.pt | done; predictor + target_encoder verified. This is model 3 |
| DINO-WM | none | dropped (2026-10-25, user decision): public checkpoints are PointMaze/PushT/Wall only. Third model = V-JEPA 2.1 |
| LTX-Video 2B (diffusers) | weights/LTX-Video | downloading; non-JEPA family + YoCausal baseline |
| RAFT (flow control) | torchvision raft_large | via torchvision |

## Data access
| Source | Status |
|---|---|
| RoboLab clips (host 84) | not requested: no host-84 agent reachable from this session on 2026-10-25 |
| Something-Something v2 | user must register at Qualcomm |
| Ego4D (Hands & Objects, PNR) | user must apply for licence |
| APT / HowToChange / ChangeIt | not checked yet |

## Feature cache
Nothing cached. Planned layout: features/<dataset>/<model>/<clip_id>.npz (e_f, e_b per tubelet/latent, meta), flow in features/<dataset>/raft/.

## Running
- env build, weight downloads (background)

## Done
- prereg/common.md, prereg/K1.md, prereg/K2.md written and committed before any measurement (primary A = mean normalised a(t), decided 2026-10-25)

## Pending
- feature extraction code (V-JEPA fwd/rev errors, LTX denoising loss, RAFT flow); smoke test on a dummy clip
- class mapping table (needs SSv2 labels file)
