# Corrections log

Every correction to a reported number, a script, or a pre-registration note (pre-registered thresholds never change), with date and reason.

None yet.

- 2026-10-25, before any measurement: prereg/common.md model 3 resolved to V-JEPA 2.1 ViT-g (`vjepa2_1_vitg_384.pt`) after its checkpoint was verified to contain a predictor. This follows the selection rule already in the committed text, and the wording was made specific. No threshold changed.

- 2026-10-25, before any data (user instruction): primary model set to V-JEPA 2 ViT-g. K1 and K2 now pass or fail on it alone; ViT-L and 2.1 are replications. This replaces "at least one of three", which was too permissive.
- 2026-10-25, before any data (user instruction): SSv2 mapping. Classes where an object leaves a surface ("roll off", "falling off") are one-way and take priority over reversible keywords. The only exclusions are "pretending", "almost", "…but" and drops onto a surface. The earlier "fail"/"over"/generic-"fall" exclusions are removed.
- 2026-10-25, before any data (user instruction): 16-in / 4-out window variant added beside the primary 8-in / 2-out.
- 2026-10-25, before any data (user decision): primary sampling is stride 3 (≈4–5 fps, matching V-JEPA pretraining at 4 fps); stride 1 is a variant. K2 merges stride-3 offsets 0, 1 and 2 to recover native-frame timing.
- 2026-10-25, before any data: context tokens now come from the online `encoder`, not the target encoder, as in V-JEPA training (the earlier wording was wrong).
- 2026-10-25, before any data: defined range corrected to c ≤ j ≤ N−1−c, with minimum N = 2c+1. LTX alignment corrected from "within 8 frames" to "within 1 native frame; latent 0 excluded", as derived from the causal VAE.
- 2026-10-25, env: diffusers pinned to 0.34.0 (from 0.42.0) and transformers/peft/accelerate/hub pinned, because newer releases need torch ≥ 2.6.
