# Implementation notes (fixed before any data is scored)

These pin down details the pre-registration leaves to implementation. Written 2026-10-25, after smoke tests on synthetic clips only.

## V-JEPA targets and context (`pb/vjepa.py`)
- **V-JEPA 2 (ViT-L, ViT-g):**
  - Context comes from `encoder(x_window, masks=context_token_indices)`.
  - The target is `layer_norm(target_encoder(x_window))`, taken at the target tubelet's tokens.
  - The predictor is `predictor(z, [ctx_idx], [tgt_idx])`.
  - Token order is (tubelet, row, col); a tubelet has (img/16)² tokens.
- **V-JEPA 2.1 ViT-g:**
  - Both encoders are called with `training=True`, so they return the 4-level hierarchical features (layers 9, 19, 29 and 39), concatenated: 4 × 1408 = 5632 dims.
  - The predictor's input embedding expects these 5632 dims, as verified.
  - The target is each 1408-dim level layer-normed separately, then concatenated. This is `forward_target` in `app/vjepa_2_1/train.py` with `levels_predictor > 1`.
  - The prediction is the predictor's `x_pred`, which has the same 5632 dims (checked in code).
  - Errors are L2 over the full 5632-dim vector, averaged over tokens.
- Resolution: 2.1 runs at 384 px. Peak memory is 5.8 GB at batch 2 on an RTX 2080 Ti, so the 256 px fallback is not used.
- Sanity check on synthetic clips: the predictor error is below the error of copying the last context tubelet (ViT-L 23.9 vs 29.9; ViT-g 29.9 vs 37.4). The same check is repeated on 50 real clips before K1, and reported.

## Flow (`pb/flow.py`)
- RAFT-large from torchvision, default weights, 12 iterations. Input is a 256 px centre crop scaled to [−1, 1].
- f(t) = mean per-pixel flow magnitude between native frames t and t+1. Camera motion = magnitude of the per-frame median flow vector.
- K2's flow baseline is the argmax of f(t) over the same native-frame range where the merged a(t) is defined.

## K1 details (`scripts/k1.py`)
- The orientation of AUROC(F) (raw or flipped) is fixed once on the full matched sample. It is then held fixed in every bootstrap draw, which is a paired bootstrap over pairs.
- Clips are eligible if they pass the filters, have flow, and are long enough for the primary window. Eligibility does not depend on the model, because every model has the same tubelet count.
- The same pairs are used for every model and variant. A model that is missing a clip's features drops that pair, and the pair count is reported per model.

## K2 details (`scripts/k2.py`)
- Merged series: the three stride-3 offset streams. Each tubelet value is placed at native frame floor(centre + 0.5), and the series is normalised by the median e_f pooled over the three streams.
- Chance = |[lo, hi] ∩ [t* − 8, t* + 8]| / (hi − lo + 1) for each clip, where [lo, hi] is the native-frame range of the merged series. Clips whose t* falls outside [lo, hi] are excluded and counted.
- t* is 0-based. If the host-84 CSV has a `frame_index_base` column, its value is subtracted.

## LTX-Video (`pb/ltx.py`)
- diffusers 0.34.0, chosen because newer releases need torch ≥ 2.6.
- Null caption: T5 embedding of "" (max length 128), computed once on CPU and cached at `weights/ltx_null_caption.pt`. The T5 encoder is never placed on the GPU.
- VAE latent: `latent_dist.mode()`, normalised with the VAE's latents mean and std. Timestep = 1000σ.
- fp16 produced finite losses on the smoke clips. A non-finite loss raises an error instead of being cached.
