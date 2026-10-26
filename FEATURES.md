# Feature store

Root: `/mnt/ssd1/devashri/CVPR/features/` (not in git). Each file is written once per (dataset, model, config, clip). `pb/cache.py` refuses to overwrite an existing file, so delete one by hand to re-extract it.

```
features/
  disk_usage.log                        size of the store and free disk, appended per job
  <dataset>/
    _filters.csv                        clip_id, reason (black_frame@t, cut_or_fade@t)
    raft/native/<clip_id>.npz           f[T-1], cam[T-1], n_native
    <vjepa model>/<config>/<clip_id>.npz
    ltx2b/native/<clip_id>.npz
```

## V-JEPA files
- Models: `vjepa2_vitg` (primary), `vjepa2_vitl`, `vjepa2_1_vitg`. A resolution override appends `_<px>px` to the model name.
- Config: `s<stride>o<offset>_w<context>x<target>`, in tubelets. For example `s3o0_w4x1` is the primary setting, and `s3o1_w4x1` and `s3o2_w4x1` are the K2 merge streams.

| key | meaning |
|---|---|
| `e_f_l2`, `e_b_l2`, `e_f_l1`, `e_b_l1` | per-tubelet [N] forward/reversed errors aligned to the physical tubelet; NaN where undefined |
| `tubelet_centre` | [N] native-frame centre of each tubelet |
| `n_tubelets`, `stride`, `offset`, `c`, `out`, `img`, `batch_used`, `n_native` | provenance |
| `too_short` | 1 if the clip has fewer than 2c+1 tubelets (no errors stored) |

## LTX files

| key | meaning |
|---|---|
| `e_f_mse`, `e_b_mse` | [K+1] per-latent denoising MSE averaged over σ and seeds; latent 0 NaN |
| `e_*_mse_by_sigma` | [5, K+1] the same, per σ |
| `latent_centre` | [K+1] native-frame centre (8k−3.5) |
| `sigmas`, `n_frames_used`, `fps`, `batch_used` | provenance |

## Measured cost

From synthetic 120-frame clips on one RTX 2080 Ti:

| model | resolution | time | peak memory |
|---|---|---|---|
| V-JEPA 2 ViT-L | 256 | ~1 s per stride-3 config; ~5 s for stride 1 | 1.7 GB |
| V-JEPA 2 ViT-g | 256 | ~5–6 s per stride-3 config; ~40 s for stride 1 | 5.8 GB |
| V-JEPA 2.1 ViT-g | 384 | ~2.5× ViT-g | 5.8 GB |
| RAFT-large | — | ~6 frame pairs/s | — |
| LTX-Video 2B | 384 | ~11 s per clip (40 transformer passes) | — |
