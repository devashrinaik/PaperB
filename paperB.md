# Paper B agent brief — "Which Way Can't Time Run" (Server B)

## What this project is
A CVPR 2027 feasibility study (deadline 16 Nov 2026). Question: do frozen video world models know which physical transitions cannot be undone — and can they find the moment it happens inside a clip, without labels?

The signal is temporal asymmetry: a frozen world model's prediction error on a clip played forward vs the same clip played backward. A reversible event (pushing a cup) should be about as predictable in either direction; a one-way event (tearing paper, an object falling off a table) should be harder to predict in reverse. We compute this per clip and per frame in the model's representation space, never in pixels.

This is a probe study on frozen models. Nothing is trained except small linear/MLP probes. There is no simulator on this server.

## What is already known (literature as of 24 Oct 2026) — do not redo, position against
- YoCausal (arXiv 2605.30346): clip-level forward-vs-reversed denoising loss on 13 video diffusion models, 1,232 real clips; humans 79 %, best model 59 %. Clip-level only, generative models only, no localisation, no event taxonomy. This is the baseline we must beat, not our contribution.
- Han et al. (arXiv 2605.07568): a supervised probe on frozen V-JEPA-H reads the arrow of time at 88.8 %. So V-JEPA features encode time direction; the open question is per-event irreversibility and its moment, label-free.
- Latent reachability for safety in DINO-WM is Bajcsy's group (arXiv 2502.00935). We do not build a reachability term.
- Wei et al. 2018 found arrow-of-time shortcuts: camera motion and black frames. We control for both.
- What is clear: label-free localisation of the point of no return inside a clip; simulator-certified events as frame-exact ground truth for a video-model probe.

## Ground truth (no hand labelling)
1. RoboLab clips from host 84 (our own): ~200 episodes where the manipulated object left its support after robot contact (certificate frame index known) and ~200 matched episodes from the same tasks where it did not. Exterior camera, 15 Hz. Request them from the host-84 agent on day 0; do not proceed to K2 without them.
2. Something-Something v2: clip-level class labels only. One-way classes: tearing, pouring, breaking, letting something roll off a surface, spilling, crushing. Reversible: moving, pushing, pulling, rotating, lifting and putting down. Register at Qualcomm's site on day 0.
3. Ego4D Hands & Objects: PNR ("point of no return") frame annotations on ~88k clips. Apply for the licence on day 0; it can take a week or more. Use it the day it arrives.
4. Optional: APT (arXiv 2606.18586, sim-certified transition times on Physion++/CLEVRER), HowToChange, ChangeIt — check availability, do not block on them.

## Models (frozen)
- V-JEPA 2 ViT-L and ViT-g with the predictor (the asymmetry needs the predictor's error, not just encoder features).
- V-JEPA 2-AC (action-conditioned) — use the same way; actions only where available (RoboLab clips).
- V-JEPA 2.1 (arXiv 2603.14482) if weights are public.
- DINO-WM.
- Control: optical-flow magnitude per frame (RAFT or torchvision).

## The measurement
For a clip x_1..x_T and a model with encoder E and predictor P over a context of c frames:
- forward error e_f(t) = || P(E(x_{t-c..t-1})) − E(x_t) ||
- reversed error e_b(t) = the same on the reversed sequence, aligned to the same physical frame
- per-frame asymmetry a(t) = e_b(t) − e_f(t); clip asymmetry A = mean or max over t (report both; pre-register which is primary).
Normalise errors within clip by the median e_f so that motion magnitude alone does not drive A. Report raw and normalised.

## POC, two kill tests, pre-registered before running
Write `prereg/K1.md` and `prereg/K2.md` with the exact pass numbers below before any measurement.

K1 — class-level detection (day 1, 27 Oct)
- Data: SSv2 if access is in; otherwise RoboLab fall vs non-fall clips.
- For each clip: A (asymmetry) and F (mean flow magnitude). Motion-match: pair one-way and reversible clips by F within ±10 %.
- Metric: AUROC of A for one-way vs reversible, and AUROC of F on the same pairs. Bootstrap over clips, 1,000 draws.
- Pass: AUROC(A) − AUROC(F) ≥ 0.10 with the 95 % CI excluding 0, and AUROC(A) ≥ 0.70. Fail: stop the project and report.
- Also report YoCausal's statistic (fraction of clips where reversed error > forward) for comparison.

K2 — localisation of the point of no return (day 2, 28 Oct)
- Data: the RoboLab certified clips (certificate frame t*).
- Per-frame a(t); predicted moment t̂ = argmax a(t) (also report the first t where a(t) exceeds its clip-wise 95th percentile).
- Baselines: argmax of flow magnitude; a random frame (chance = window / clip length).
- Pass: |t̂ − t*| ≤ 8 frames (0.53 s at 15 Hz) in > 60 % of clips, against < 25 % for the flow baseline and the chance rate stated per clip length. Report the full error distribution in frames and the lead/lag sign (does the peak come before or after t*).
- Then the same test on Ego4D PNR frames when the licence arrives.

Decision on 28 Oct: both pass → week 2 (anticipation: does a(t) rise before t*; locus: encoder vs predictor vs rollout by probes; the undo test: error on reversed clips spikes at t*; second ground-truth set). Either fails → park, write a two-page negative note.

## Controls, always
- Static vs handheld camera: RoboLab is static; for SSv2 add a camera-motion covariate (global flow) and report results with it regressed out.
- No black frames, fades or cuts in any clip (filter and log the count removed).
- Symmetric-motion classes (rotating, Newton's-cradle-like) reported separately; YoCausal's failure mode.
- Clip length matched across classes; report length as a covariate.
- Three models minimum (V-JEPA 2-L, V-JEPA 2-g, DINO-WM); a result that holds on one model only is reported as such.

## Rules
- Pre-register every test with its pass number before running it; never move a threshold after seeing data.
- Every number with a bootstrap CI over clips; nothing without one.
- No hand annotation of clips. Class mapping from dataset labels is a table in the repo, committed before K1.
- Both GPUs busy for feature extraction; cache features to disk once per (model, clip); no re-extraction.
- Keep `RESULTS.md` (every result, beside its CI and its pre-registration), `STATE.md` (what is cached, running, pending), a corrections log, and commit at each milestone. After any context compaction, re-read all three before continuing.
- Report each morning: what is cached, what ran, numbers beside CIs, what failed, what is blocked on data access.
- Do not invent datasets, numbers or papers. If a model's weights or a dataset are not accessible, say so and use the next option on the list.

## Deliverables by 28 Oct
`prereg/K1.md`, `prereg/K2.md`, `RESULTS.md` with both tests, a one-page `DECISION.md` saying pass/fail per test with the numbers, and the cached feature store layout documented.
