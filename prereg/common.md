# Shared measurement protocol (applies to K1 and K2)

Fixed 2026-10-25, before any feature extraction or measurement.

## Models (frozen, fp16 on RTX 2080 Ti, batch 1–2)
1. **V-JEPA 2 ViT-g (`vitg.pt`) at 256 px: the primary model.** K1 and K2 pass or fail on this model alone.
2. V-JEPA 2 ViT-L (`vitl.pt`) at 256 px: replication.
3. V-JEPA 2.1 ViT-g (`vjepa2_1_vitg_384.pt`) at 384 px: replication. Its checkpoint is verified to contain a predictor. If 384 px does not fit in memory, it falls back to 256 px, and the fallback is recorded in STATE.md and RESULTS.md.

The two replications are reported beside the primary model, with their own pass/fail under the same rule. They do not decide K1 or K2.

DINO-WM is not used: its public checkpoints cover only PointMaze, PushT and Wall, with none for real video. We report this as a limitation. All three models are from the V-JEPA family, so agreement across them counts as weaker evidence than agreement across independent model families. V-JEPA 2-AC is used only on RoboLab clips that come with actions, and it is not part of either kill test.

Every test is run and reported separately for each model. A result that holds on only one model is reported as such.

### Independent family: video diffusion (YoCausal-style baseline)
4. LTX-Video 2B (Lightricks/LTX-Video, diffusers, non-distilled), loaded frozen in fp16. Cosmos-Predict2-2B is gated on HF and so is only a fallback.

This model has two roles:
- It is the YoCausal baseline, computed on the same clips.
- It is the non-JEPA check.

Its protocol:
- **Input.** Centre crop at 384×384, native fps (LTX-Video was trained on native-rate video). The clip is trimmed to the largest 8k+1 frames that fit. The forward and the reversed clip are each VAE-encoded separately; reversal happens in pixel space before encoding.
- **Conditioning.** The text embedding of a fixed null caption (the empty string), computed once and cached to disk. The T5 encoder is then unloaded, and scoring loads only the VAE and the transformer. No captions are used.
- **Denoising loss.** Rectified flow: z_σ = (1−σ)z + σε, with target v = ε − z.
  - σ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
  - 4 noise seeds per σ (seeds 0–3)
  - forward and reversed passes use identical ε seeds
  - The loss is the MSE averaged over latent channels and space, giving one value per latent frame, then averaged over σ and seeds.
- **Asymmetry.** a_diff(k) = L_rev(k) − L_fwd(k), where reversed latent k is mapped to the forward latent covering the same physical frames. Because the causal VAE keeps frame 0 as its own latent, reversed latent k covers native frames 8K−8k..8K−8k+7, while forward latent K+1−k covers 8K−8k+1..8K−8k+8. They are therefore aligned to within 1 native frame. Latent 0, the single first frame, has no counterpart and is excluded.
  - Normalisation, clip statistics (mean primary) and CIs are the same as for the JEPA models.
- **Temporal resolution** is 8 frames per latent; each value is placed at its latent's centre frame (8k−3.5). K2 hits for this model are therefore at latent resolution, a stated limitation. The diffusion RoPE frame-rate scale is set to the clip's native fps.

How it enters the decision:
- The K1/K2 pass/fail decision is taken on the primary model, V-JEPA 2 ViT-g.
- LTX-Video gets the same metrics and the same pass rule, reported as the non-JEPA row.
- A JEPA result that does not hold on LTX-Video is reported as family-specific.
- For each V-JEPA model, the difference AUROC(V-JEPA) − AUROC(LTX) is reported with a paired bootstrap CI.

## Clip preprocessing
- **Primary sampling: stride 3 at the native frame rate**, giving 4 fps for SSv2 (12 fps native) and 5 fps for RoboLab (15 Hz native). This matches V-JEPA 2/2.1 pretraining, which uses 4 fps in every train config. At stride 1 the predictor would see frames about 3× denser in time than in training, in both directions, which would make the asymmetry uninterpretable.
- Variant, always reported: stride 1.
- Frames get a centre crop and are resized to the model's resolution (see Models).
- The temporal unit is the model's tubelet: 2 sampled frames. At stride 3, tubelet j of offset o spans native frames o+6j..o+6j+5. Each tubelet value is placed at its centre native frame.
- Filters are applied on native frames before any measurement, and every removal is logged with its reason:
  - black frame: mean intensity below 10/255 on any frame
  - cut or fade: colour-histogram correlation between consecutive frames below 0.5
  - too short for a window (below)
- SSv2 clips dropped for length are counted and reported for each class.

## Errors
- Windows, measured in tubelets of sampled frames:
  - **Primary 8-in / 2-out:** c = 4 context tubelets (8 sampled frames) and 1 target tubelet (2 frames).
  - **Variant 16-in / 4-out:** 8 context tubelets and 2 target tubelets. Run on clips long enough for it (N ≥ 17 tubelets) and reported alongside, so window length is not a hidden choice.
- Following V-JEPA training:
  - Context tokens come from the online `encoder`, which sees only the context tubelets of the window.
  - The predictor predicts the masked target tubelet(s).
  - Targets are the layer-normed outputs of `target_encoder` on the full window.
  - For V-JEPA 2.1, the target is the hierarchical, layer-normed target that the predictor was trained on. The exact tensor is fixed by the smoke test and recorded in `prereg/impl_notes.md` before any data is scored.
- Forward error for tubelet j: the window's context is the c tubelets before j, and j is the first target tubelet. e_f(j) = mean over tubelet j's spatial tokens of the L2 distance between predicted and target tokens. L1, the training loss, is reported as a variant.
- Reversed error for tubelet j: the same on the clip reversed in time, with context the c tubelets after j. e_b(j) and e_f(j) therefore refer to the same physical tubelet.
- a(j) = e_b(j) − e_f(j) is defined on tubelets c ≤ j ≤ N−1−c, where both errors exist (errors are always taken on the target tubelet adjacent to the context). The minimum clip length is N = 2c + 1 tubelets: 9 tubelets (52 native frames at stride 3) for 8/2, and 17 tubelets (100 native frames) for 16/4. The sampled length is trimmed to an even number so tubelets tile it and reversal maps tubelet i to N−1−i exactly.
- Normalised errors divide by the clip's median e_f over the defined range. Raw errors are reported alongside.

## Clip statistics
- **Primary: A = mean over defined tubelets of the normalised a(j).**
- Secondary: max of the normalised a(j), and both statistics on raw errors.
- We chose mean over max because max grows with clip length, and clip length is a confound these tests control for.

## Flow control
- RAFT-large (torchvision) on consecutive frames at 256 px.
- F = mean over frames of the mean per-pixel flow magnitude.
- Per-frame flow f(t) = the mean per-pixel flow magnitude between frames t and t+1, averaged onto tubelets.
- Camera motion (SSv2) = mean over frames of the magnitude of the median flow vector.

## Engineering rules (do not affect statistics)
- fp16 throughout, batch 1–2.
- Each job checks free GPU memory before it starts, and on OOM retries with a smaller batch instead of dying.
- Features are cached once per (model, clip, stride, offset, window) and never re-extracted. Disk use is logged.

## Statistics
- Bootstrap: 1,000 draws, resampling clips (pairs in K1), with percentile 95 % CIs. Seed 0.
- Every reported number carries its CI.
