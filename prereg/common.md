# Shared measurement protocol (applies to K1 and K2)

Fixed 2026-10-25, before any feature extraction or measurement.

## Models (frozen, fp16 on RTX 2080 Ti)
1. V-JEPA 2 ViT-L (`vitl.pt`): target encoder + mask predictor
2. V-JEPA 2 ViT-g (`vitg.pt`): target encoder + mask predictor
3. V-JEPA 2.1 (`vjepa2_1_vitg_384.pt` if it ships a predictor; else `vjepa2_1_vitl_dist_vitG_384.pt`, verified to include a predictor). Its target encoder is the checkpoint's `ema_encoder`.

DINO-WM is not used: its public checkpoints cover only PointMaze, PushT and Wall, with none for real video. We report this as a limitation. All three models are from the V-JEPA family, so agreement across them counts as weaker evidence than agreement across independent model families. V-JEPA 2-AC is used only on RoboLab clips that come with actions, and it is not part of either kill test.

Every test is run and reported separately for each model. A result that holds on only one model is reported as such.

### Independent family: video diffusion (YoCausal-style baseline)
4. LTX-Video 2B (Lightricks/LTX-Video, diffusers, non-distilled), loaded frozen in fp16. Cosmos-Predict2-2B is gated on HF and so is only a fallback.

This model has two roles:
- It is the YoCausal baseline, computed on the same clips.
- It is the non-JEPA check.

Its protocol:
- **Input.** Centre crop at 384×384, native fps. The clip is trimmed to the largest 8k+1 frames that fit. The forward and the reversed clip are each VAE-encoded separately; reversal happens in pixel space before encoding.
- **Conditioning.** The text embedding of an empty prompt, computed once and cached. No captions are used.
- **Denoising loss.** Rectified flow: z_σ = (1−σ)z + σε, with target v = ε − z.
  - σ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
  - 4 noise seeds per σ (seeds 0–3)
  - forward and reversed passes use identical ε seeds
  - The loss is the MSE averaged over latent channels and space, giving one value per latent frame, then averaged over σ and seeds.
- **Asymmetry.** a_diff(k) = L_rev(k) − L_fwd(k), where reversed latent k is mapped to the forward latent covering the same physical frames. With the causal VAE this alignment is exact only to within 8 frames.
  - Normalisation, clip statistics (mean primary) and CIs are the same as for the JEPA models.
- **Temporal resolution** is 8 frames per latent, so K2 hits for this model are at latent resolution. This is a stated limitation.

How it enters the decision:
- The K1/K2 pass/fail decision is taken on the three V-JEPA models.
- LTX-Video gets the same metrics and the same pass rule, reported as the non-JEPA row.
- A JEPA result that does not hold on LTX-Video is reported as family-specific.
- For each V-JEPA model, the difference AUROC(V-JEPA) − AUROC(LTX) is reported with a paired bootstrap CI.

## Clip preprocessing
- Native frame rate (SSv2 12 fps, RoboLab 15 Hz, Ego4D as released) and stride 1. Frames are resized to the model's resolution (256 for V-JEPA 2-L/g, 384 for 2.1), with a centre crop.
- The temporal unit is the model's tubelet (2 frames). Tubelet j covers frames 2j and 2j+1. Frame-level quantities are placed at the tubelet's first frame.
- Filters are applied before any measurement, and every removal is logged with its reason:
  - black frame: mean intensity below 10/255 on any frame
  - cut or fade: colour-histogram correlation between consecutive frames below 0.5
  - too short: fewer than 18 frames (9 tubelets)

## Errors
- Window: c = 4 context tubelets (8 frames) followed by 1 target tubelet.
- Forward error for tubelet j (j ≥ 4): the predictor sees target-encoder tokens of tubelets j−4..j−1 and predicts the masked tokens of tubelet j. e_f(j) is the mean over spatial tokens of the L2 distance between the predicted tokens and the target-encoder tokens of tubelet j.
- Reversed error for tubelet j (j ≤ N−5): the same computation on the time-reversed clip. The context is tubelets j+4..j+1 and the target is physical tubelet j, so e_b(j) and e_f(j) refer to the same physical tubelet.
- a(t) is defined on tubelets 4 ≤ j ≤ N−5, where both errors exist.
- Normalised errors divide by the clip's median e_f over that defined range. Raw errors are reported alongside.

## Clip statistics
- **Primary: A = mean over defined tubelets of the normalised a(j).**
- Secondary: max of the normalised a(j), and both statistics on raw errors.
- We chose mean over max because max grows with clip length, and clip length is a confound these tests control for.

## Flow control
- RAFT-large (torchvision) on consecutive frames at 256 px.
- F = mean over frames of the mean per-pixel flow magnitude.
- Per-frame flow f(t) = the mean per-pixel flow magnitude between frames t and t+1, averaged onto tubelets.
- Camera motion (SSv2) = mean over frames of the magnitude of the median flow vector.

## Statistics
- Bootstrap: 1,000 draws, resampling clips (pairs in K1), with percentile 95 % CIs. Seed 0.
- Every reported number carries its CI.
