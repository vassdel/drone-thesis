============================================================
THESIS PLAN
Adapting ContextVAE for UAV Trajectory Prediction
Author: Vassilis
Plan window: April 26, 2026 -> June 15, 2026 (~7 weeks)
Submission target: June 14, 2026 (one day buffer)
============================================================


1. PROJECT SUMMARY
------------------

Goal: replace the LSTM-based trajectory predictor in Odysseas
Ntousis' dual-stage UAV detection-and-tracking architecture
(2024_OdysseasNtousis_DualStageUAV_Board_and_Server) with a
ContextVAE-style timewise variational autoencoder
(2023_PeiXu_ContextVAE), adapted to the aerial UAV domain.

Primary scientific contribution: re-target the ContextVAE map
encoder from rasterized HD semantic maps to aerial orthomap
patches, and validate on aerial-drone trajectory data.

Reference repositories:
- ContextVAE (Pei Xu): https://github.com/xupei0610/ContextVAE
- Ntousis UAV Guidance (Odysseas Ntousis):
  https://github.com/ontousis/uav_guidance/tree/main


2. STATED ASSUMPTIONS
---------------------

Confirmed:
- June 15, 2026 is the SUBMISSION date (defense, if any, is
  separate).
- No live UAV flight test required for submission.
- Hardware compute: NVIDIA A40 (46 GB VRAM, CUDA 12.4) is
  available for training.
- Disk budget: ~200 GB total. WOMD (the originally-considered
  dataset) is too large to host alongside checkpoints and
  preprocessed tensors. inD + uniD (~10 GB combined) fit
  comfortably.
- Full source-code access to Ntousis codebase (server +
  Raspberry Pi side). No physical hardware access yet --
  request submitted on Day 1 of Week 1.

To clarify with supervisor in Week 1:
- Sign-off on the dataset switch (WOMD -> inD/uniD) and the
  resulting refocus toward the aerial-orthomap M-ATTN encoder.
- Confirm whether Greek or English is the submission language.
- Confirm approximate hardware-access lead time, even if access
  itself is uncertain.


3. LOCKED DECISIONS (commit by end of Week 1)
---------------------------------------------

Dataset:
  PRIMARY        inD + uniD (levelXdata, RWTH Aachen)
                 - aerial drone perspective, hovering camera
                 - ~10 GB total on disk
                 - metric SI coordinates already provided
                   (xCenter, yCenter, xVelocity, yVelocity,
                    xAcceleration, yAcceleration, heading)
                 - object class, width, length included
                 - per-recording georeferenced background image
                   (aerial orthomap with orthoPxToMeter scale)
                 - Lanelet2 (.osm) + ASAM OpenDRIVE (.xodr)
                   maps included
  CROSS-DOMAIN   VisDroneVDT-2018 (already used by Ntousis;
  TEST SET       moving-camera UAV with track-ID annotations,
                 pixel coordinates) -- used in Week 4 to
                 evaluate domain transfer.

Coordinate frame:
  Ego-normalized at the last observation frame: rotate target's
  trajectory so the heading at frame T points along +x.
  ContextVAE convention. Heading is provided directly in
  inD/uniD tracks.csv.

Map strategy:
  PRIMARY    Aerial-orthomap M-ATTN encoder.
             Replace ContextVAE's `map_encode` (a CNN over
             rasterized lanes/road-edges) with a CNN ingesting
             heading-rotated, agent-centered crops of
             XX_background.png at ~60x60 m extent (the patch
             size in pixels comes out to ~600x600 px at typical
             inD scale, depending on orthoPxToMeter).
  FALLBACK   No-map S-ATTN-only ablation (use_map=False).
             Kept as the baseline run regardless. Paper Table S3
             documents this configuration as ~95% of the full
             model's performance, so it is defensible if the
             orthomap encoder fails to train.

Observation / prediction horizons:
  inD/uniD record at native ~25 Hz. Downsample to 5 Hz to
  match ContextVAE's nuScenes setup.
    - Observation length L1: 10 frames at 5 Hz = 2 s
    - Prediction horizon  H : ~25 frames at 5 Hz = 5 s
  Reconsider after first training run if metrics are weak.

Number of neighbors Nn:
  Use radius-based selection (ob_radius = 30 m, matching the
  paper) rather than Ntousis' fixed 4-direction neighbor scheme.
  Pad missing neighbor slots to a fixed Nn = 12 with the value
  1e9 so the distance mask filters them out -- this is critical
  and non-obvious. Zero-padding would erroneously pass the mask.

Deployment validation:
  Offline replay on recorded UAV video (Ntousis archives if
  available; VisDroneVDT-2018 otherwise). Live flight is NOT a
  requirement for submission.

Min-max scaling:
  DO NOT carry forward Ntousis' min-max input scaling.
  ContextVAE operates in raw metric units; min-max scaling
  conflicts with the encoder's learned embeddings.

Loss function:
  Standard ContextVAE ELBO (timewise reconstruction +
  per-timestep KL). Do not modify the loss in Phase 1.


4. CRITICAL TECHNICAL CONSTRAINTS (carried over from prior study)
-----------------------------------------------------------------

These are non-obvious requirements that must be satisfied in any
ContextVAE-style training pipeline. Violating any one of them
silently degrades training:

a) The `neighbor` tensor must span the full L1 + L2 window at
   training time (not just the observation window). The
   posterior RNN consumes future neighbor states. At inference
   time it can be trimmed to L1.

b) Padding for absent neighbors uses 1e9, never 0. The distance
   mask in enc() filters by `dist <= ob_radius`. Zero-padded
   neighbors would incorrectly pass the threshold.

c) Map patches must be heading-rotated, not merely cropped, to
   produce an agent-centric view. The rotation reuses the
   target's heading at the first observation frame.

d) Numerical stability of the VAE encoder requires output of
   log-sigma-squared (not sigma) and a closed-form Gaussian KL
   computation. Do not modify the variance parameterization.

e) Environment isolation: ContextVAE references depend on
   PyTorch + numpy. Keep training in a dedicated conda env to
   avoid Python 3.11+ / numpy version conflicts that previously
   broke the WOMD pipeline.


5. WEEKLY SCHEDULE
------------------

WEEK 1 -- April 26 -> May 3
Theme: lock decisions, set up preprocessing, smoke test.

  PROGRESS LOG (as of Tue Apr 28):
    - Repo reorganized from flat layout into a package structure:
      contextvae/ (main.py, context_vae.py, data.py, utils.py),
      configs/, preprocessing/, docs/, tests/. Commit c47b8f5.
    - levelXdata preprocessing notebook drafted at
      preprocessing/levelx_preprocessing.ipynb. Unified loader
      across inD + uniD + rounD (one schema, switch on
      locationId / map path). Discovers recordings under
      <DATA_ROOT>/<dataset>/data/, tallies class labels, emits
      ContextVAE on-disk format (train|val/*.txt, *.info,
      map/*.pkl) at 5 Hz with VEHICLE/TARGET vs VRU grouping,
      heading deg->rad, local-frame xCenter/yCenter verbatim,
      orthomap normalized to [-1,1] with a 3x3 homography
      (local (x,y) -> image (row,col), image-y flipped).
    - GPU/CUDA sanity cell verifies A40 + PyTorch CUDA build.
    - Still pending in Week 1: smoke-test 1-epoch training on a
      single recording; draft Ch 1 + Ch 2.

  Day 1 (Sun Apr 26)
    - Send hardware-access request in writing. Ask explicitly
      for ETA.
    - Send supervisor a short note confirming the dataset
      switch (WOMD -> inD/uniD) and asking for sign-off on
      this thesis plan.

  Day 2-4 (Mon-Wed Apr 27-29)
    - Set up the conda environment (Python 3.10 to remain
      compatible with previous WOMD setup; ContextVAE itself
      is flexible).
    - Implement inD/uniD -> ContextVAE tensor preprocessing.
      Output shapes:
          x        : (L+1, N, 6)
          neighbor : (L+1, N, Nn, 6) with 1e9 padding
          map      : (N, C, H, W) heading-rotated orthomap
                     patches
      Validate that loaded vx, vy match
          (x[t+1] - x[t]) * frameRate
      to within rounding. Validate that headings are in
      degrees and convert to radians where needed.

  Day 4-5 (Wed-Thu Apr 29-30)
    - Smoke test: train for 1 epoch on a single recording.
      Pass criterion = loss decreases monotonically and no
      shape errors.

  Day 5-7 (Fri-Sun May 1-3)
    - Draft Chapter 1 (Introduction) and Chapter 2
      (Background: UAV autonomy, trajectory prediction, VAE
      fundamentals, ContextVAE specifics).

  Exit criterion: smoke test passes; intro and background
  chapters drafted.


WEEK 2 -- May 4 -> May 10
Theme: train both model variants.

  Day 1-2 (Mon-Tue)
    - Train no-map S-ATTN-only baseline on inD + uniD combined.
      Expected wall-clock on A40: 4-8 hours per full run.
      Validate on a held-out split.

  Day 2-4 (Tue-Thu)
    - Implement the aerial-orthomap M-ATTN encoder. Replace
      `map_encode` in context_vae.py:
        - Input: heading-rotated agent-centered crop of
          XX_background.png at known metric scale.
        - Backbone: ResNet-18 (paper Table S1 shows this is
          competitive with much larger backbones).
        - Output dim: same as the existing map_encode output to
          keep M-ATTN compatible.
      Sanity-check: rotated crops should look upright when
      visualized for sample agents.

  Day 4-6 (Thu-Sat)
    - Train the orthomap variant. Compare validation
      minADE_k / minFDE_k for k=1, k=5 against the no-map
      baseline.

  Day 6-7 (Sat-Sun)
    - Draft Chapter 3 (Methodology) including a section on the
      orthomap encoder with figures.

  Exit criterion: two trained checkpoints; metrics in hand;
  Methodology chapter drafted.

  Risk gate: if the orthomap variant does NOT beat no-map by a
  meaningful margin, accept the negative result, do not chase
  it into Week 3, and frame the no-map model as primary.


WEEK 3 -- May 11 -> May 17
Theme: integrate ContextVAE into Ntousis' server-side pipeline,
       offline replay only.

  Day 1-2 (Mon-Tue)
    - Replace tr_pred_lstm in
      server_code_only/main_program/scene_recog_socket_yolov8.py
      with a ContextVAE inference call. Keep:
        * socket protocol
        * timestamped-prediction return path
        * target buffer / rolling-window logic
      Replace:
        * the 4-direction detect_up/detect_down/detect_left/
          detect_right neighbor selection -- use radius-based
          neighbor selection on DeepSORT track centroids
          instead.
        * scaling helpers (scale_data, unscale_data) -- not
          needed in metric-frame inference.

  Day 2-4 (Tue-Thu)
    - Implement homography-based ego-motion compensation.
      Pipeline:
        a) Detect ORB or AKAZE keypoints on each frame's
           background regions (excluding detected vehicles).
        b) Match across consecutive frames; estimate
           homography with RANSAC.
        c) Warp bbox centers into a stabilized reference frame.
      Validate on a clip with stationary objects: their
      compensated positions should remain ~constant across
      frames.

  Day 4-5 (Thu-Fri)
    - End-to-end replay: a recorded UAV video runs through the
      integrated pipeline producing K=5 trajectory samples per
      tracked agent, visualized as overlays on the video.

  Day 5-7 (Fri-Sun)
    - Draft Chapter 4 (System Integration).

  Exit criterion: a recorded UAV video replays through the
  integrated pipeline; predictions visualized.

  Hardware contingency: if hardware access has been confirmed
  by this point, schedule a dry run for Week 5 or 6. Do NOT
  delay Week 3 deliverables waiting for hardware.


WEEK 4 -- May 18 -> May 24
Theme: ablations, results, writing.

  Experiments to run (most are inference-only on existing
  checkpoints; budget for one or two retrains):
    1. ContextVAE no-map vs orthomap M-ATTN on inD/uniD val
       split. Headline result.
    2. ContextVAE vs Ntousis' LSTM, both trained and evaluated
       on matched inD/uniD inputs. Direct comparison vs prior
       work.
    3. Cross-domain: train on inD/uniD, evaluate on
       VisDroneVDT-2018 (after pixel-to-meter scaling estimation
       per scene). Domain-transfer test -- the most thesis-
       interesting result.
    4. Neighbor radius sweep: 10 m / 30 m / 60 m. Tunes for
       UAV-scale interaction range.
    5. Observation horizon sweep: 1 s / 2 s / 4 s. Justifies
       window choice.
    6. K=1 vs K=5 minADE / minFDE. Standard for stochastic
       predictors.

  Day 1-4 (Mon-Thu)
    - Run experiments. Generate plots: trajectory overlays,
      attention heatmaps, error CDFs, ablation bar charts.

  Day 5-7 (Fri-Sun)
    - Write Chapter 5 (Experiments and Results).
    - Draft Chapter 6 (Discussion).

  Exit criterion: all experiments complete; results section
  with plots; all chapters at least drafted.


WEEK 5 -- May 25 -> May 31
Theme: triage. Decide on Day 1 from these three paths.

  PATH A (everything is on schedule, hardware available):
    Hardware-in-the-loop dry run. One short flight or even a
    bench replay with the actual Raspberry Pi -> server
    pipeline closed end-to-end. Records latency and a couple
    of qualitative trajectory predictions. Significantly
    elevates the thesis.

  PATH B (on schedule, no hardware):
    Add a Lanelet2-rasterized semantic map ablation. Compares
    the raw aerial orthomap against an explicit semantic
    rasterization (lanes, road edges) derived from the .osm
    files. Answers: "does road-graph structure add value
    beyond what's visible in the aerial image?"

  PATH C (behind schedule):
    No new experiments. Use the week to consolidate writing,
    redo noisy plots, fill gaps, write the limitations
    section.

  Begin Chapter 7 (Conclusions and Future Work) regardless of
  path.

  Exit criterion: chosen path executed; conclusions chapter
  drafted.


WEEK 6 -- June 1 -> June 7
Theme: complete draft.

  Day 1-3 (Mon-Wed)
    - Finish all remaining chapters. Aim for "submittable but
      rough" by end of Day 3.

  Day 4 (Thu Jun 4)
    - Generate all final figures at publication quality
      (300 dpi vector where possible).

  Day 5 (Fri Jun 5)
    - Send full draft to supervisor. THIS IS A HARD DEADLINE.
      Earlier than the original June 7 plan, because the
      front-loaded schedule built up buffer.

  Day 6-7 (Sat-Sun Jun 6-7)
    - Do not touch the thesis. Let it cool. Use this time for
      defense-prep materials if relevant, or rest.

  Exit criterion: complete draft in supervisor's hands.


WEEK 7 -- June 8 -> June 14
Theme: polish and submit.

  Day 1-3 (Mon-Wed Jun 8-10)
    - Address supervisor feedback. Be ruthless about scope of
      changes -- no fundamental restructuring at this stage.

  Day 4 (Thu Jun 11)
    - Final pass: figures, captions, references, formatting
      against institution template.

  Day 5 (Fri Jun 12)
    - Read the entire thesis cover-to-cover. Catches more
      issues than expected.

  Day 6 (Sat Jun 13)
    - Buffer day for any last issues.

  Day 7 (Sun Jun 14)
    - SUBMIT. One day before deadline. Protects against the
      predictable last-day filesystem / PDF / LaTeX disaster.


6. RISK REGISTER
----------------

R1  Hardware access lead time unknown.
    Probability: medium-high.
    Impact: medium. (No live flight is required for submission,
    but ground-truth validation is weakened without it.)
    Mitigation: request submitted Day 1 of Week 1; full plan
    works without hardware; opportunistic dry run in Week 5
    Path A only if access materializes.

R2  Train/deploy domain gap (hovering inD/uniD -> moving UAV).
    Probability: high.
    Impact: medium. This is now framed as a thesis contribution
    rather than a defect: the cross-domain VisDroneVDT-2018
    experiment in Week 4 explicitly measures it.
    Mitigation: homography compensation in Week 3; honest
    discussion of limits in the limitations section.

R3  Orthomap M-ATTN encoder fails to outperform no-map.
    Probability: medium.
    Impact: low. The negative result is still publishable as
    an ablation; the no-map model remains a defensible primary.
    Mitigation: Week 2 risk gate documented above; do not chase
    it into Week 3.

R4  Training fails to converge on first attempt.
    Probability: low-medium.
    Impact: medium.
    Mitigation: most likely culprits already known from prior
    code-level study (neighbor radius scaling, 1e9 padding,
    posterior RNN window mismatch). Two retries budgeted in
    Week 2.

R5  Supervisor feedback in Week 7 demands rework.
    Probability: medium.
    Impact: high if it lands.
    Mitigation: send draft June 5 (Friday Week 6) instead of
    June 7. Buys two extra days for revision.

R6  Pipeline integration reveals unexpected architectural
    mismatch between ContextVAE and Ntousis' server.
    Probability: low.
    Impact: medium.
    Mitigation: code-level study already done in prior weeks;
    interfaces are clean; socket protocol is dataset-agnostic.

R7  Personal time loss (illness, etc.).
    Probability: nonzero.
    Impact: variable.
    Mitigation: each week ends with writing rather than coding,
    creating implicit buffer; most weeks have a Sunday with
    soft expectations.


7. IMMEDIATE ACTIONS (this week, by Friday May 1)
-------------------------------------------------

[ ] Send hardware-access request in writing.
[ ] Confirm with supervisor: dataset switch (WOMD -> inD/uniD)
    and refocus on aerial-orthomap M-ATTN encoder.
[ ] Confirm with supervisor: submission language; whether June
    15 is firm.
[x] Stand up conda env on the A40 machine; verify PyTorch +
    CUDA work. (Tue Apr 28, sanity cell in
    preprocessing/levelx_preprocessing.ipynb)
[x] Build inD/uniD preprocessing pipeline producing valid
    ContextVAE tensors from one recording. (Tue Apr 28,
    extended to inD + uniD + rounD via a unified loader.)
[ ] Smoke-test training on that one recording.
[ ] Open the LaTeX project; create empty chapter files.
    (Activation energy of starting writing is the largest
    hidden cost in any thesis.)


8. CHAPTER STRUCTURE (working outline)
--------------------------------------

Ch 1  Introduction
        - Motivation: real-time trajectory prediction for UAV
          autonomy.
        - Problem statement: predict aerial-tracked-vehicle
          trajectories under deployment constraints (limited
          on-board compute, no HD maps).
        - Contributions: (1) ContextVAE adapted to aerial UAV
          domain, (2) aerial-orthomap M-ATTN encoder, (3)
          integration with Ntousis dual-stage architecture,
          (4) cross-domain evaluation hovering -> moving UAV.

Ch 2  Background
        - UAV autonomy and dual-stage processing.
        - Object detection and tracking (YOLO family,
          DeepSORT).
        - Trajectory prediction landscape (LSTM-family,
          social pooling, Trajectron++, ContextVAE).
        - Variational autoencoders, timewise VAEs, ELBO.
        - Aerial drone datasets.

Ch 3  Methodology
        - System architecture overview.
        - ContextVAE backbone summary.
        - Aerial-orthomap M-ATTN encoder (the contribution).
        - Training data pipeline (inD + uniD).
        - Loss function and training procedure.

Ch 4  System Integration
        - Bridging ContextVAE into Ntousis' server-side code.
        - Ego-motion compensation.
        - Neighbor restructuring.
        - Inference rolling-window deployment loop.

Ch 5  Experiments and Results
        - Datasets and metrics.
        - Headline result: orthomap M-ATTN vs no-map S-ATTN.
        - Comparison vs Ntousis' LSTM.
        - Cross-domain: hovering -> moving UAV.
        - Ablations: neighbor radius, observation horizon, K.

Ch 6  Discussion
        - Why orthomaps help (or don't).
        - Where domain transfer breaks down.
        - Latency and deployment-readiness analysis.

Ch 7  Conclusions and Future Work
        - Summary of contributions.
        - Open questions: lane semantics from orthomap,
          continuous online adaptation, hardware-in-the-loop
          live evaluation.

Limitations (as a separate short section, not buried in
Discussion):
        - inD/uniD is German urban data; transfer to other
          regions is untested.
        - Hovering-only training data; moving-camera
          performance bounded by homography quality.
        - No live flight evaluation in this thesis.


9. OPEN QUESTIONS FOR CLAUDE CODE / SUPERVISOR
----------------------------------------------

These are decisions deferred until concrete data or signals
appear:

Q1  Final patch size for the aerial orthomap encoder. 60x60 m
    is the starting guess; tune based on inD/uniD vehicle
    speeds.
Q2  Whether to pretrain the orthomap CNN on a self-supervised
    objective (e.g., SimCLR on aerial patches) before joint
    training. Adds time; revisit only if the joint training
    is unstable.
Q3  Whether to include heading and bbox dimensions as extra
    state channels (extending 6 -> 8 dims). Probably no; keep
    architecture surface minimal.
Q4  Exactly which Ntousis recording or VisDroneVDT-2018 clip
    to use for the qualitative replay video. Pick during
    Week 3.
Q5  Final loss-weighting between reconstruction and KL terms.
    Defaults are usually fine; tune only if posterior collapse
    or overconfident predictions appear.


END OF PLAN
============================================================

