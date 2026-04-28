================================================================================
ContextVAE — Implementation Instructions
Data Preprocessing and Training Pipeline
================================================================================

Audience: an engineer who needs to write (a) a preprocessing script that
converts a new dataset into a format ContextVAE can train on, and (b) invoke
training against the resulting data. No prior knowledge of ContextVAE internals
is assumed. All references are to the xupei0610/ContextVAE codebase and the
2023 IEEE RA-L paper by Xu, Hayet, and Karamouzas.

Read section 1 first. Then implement in the order of sections 3 -> 4 -> 5 -> 6.


================================================================================
1. WHAT YOU ARE BUILDING
================================================================================

You are writing ONE thing: a preprocessing script that converts your raw
dataset into a specific on-disk text format. Once that format is produced,
the existing codebase handles everything else -- loading, batching, derivative
computation, rotation, padding, map cropping, training, evaluation. You do
NOT modify data.py, context_vae.py, or main.py.

The model consumes, per training batch, the following tensors (built by
data.py from your files):

    x         shape (L_ob, N, 6)              observation states of targets
    y         shape (L_pred, N, 2)            future ground-truth positions
    neighbor  shape (L_ob + L_pred, N, Nn, 6) neighbor states, 1e9-padded
    map       shape (1, N, C, H, W)           semantic map patch (optional)
    seq_len   shape (N,)                      valid observation length

Where:
    L_ob   = number of observation frames (config: ob_horizon)
    L_pred = number of prediction frames  (config: pred_horizon)
    N      = batch size
    Nn     = number of neighbors in the batch (variable, padded)

The 6 state channels are (x, y, vx, vy, ax, ay). Units: meters, meters/frame,
meters/frame^2. Coordinates are world-frame meters at input; the dataloader
rotates them into the agent's local frame internally.

You do NOT need to produce velocity or acceleration. The dataloader computes
them from positions via finite differences. Supply positions only.


================================================================================
2. HOW THE FULL PIPELINE FITS TOGETHER
================================================================================

    raw dataset
        |
        v
    your preprocessing script (this is what you write)
        |
        v
    per-scene .txt files + .info files + optional map/*.pkl files
        |
        v
    data.py :: Dataloader      (loads text, computes v/a, builds tensors)
        |
        v
    main.py training loop
        |
        v
    context_vae.py :: ContextVAE.learn()
                      (maximize ELBO: reconstruction + KL)

Your only implementation work is the second box. If your preprocessing script
emits files that match the contract in section 3, nothing else needs to
change.


================================================================================
3. THE ON-DISK FORMAT YOUR SCRIPT MUST PRODUCE
================================================================================

For each independent recording ("scene") in your dataset, produce three
kinds of files:

    <target_folder>/
        train/
            <scene_id>.txt        # trajectory rows
            <scene_id>.info       # scene metadata (1 line)
            ...
        val/
            <scene_id>.txt
            <scene_id>.info
            ...
        map/                      # OPTIONAL — only if using the map branch
            <map_name>.pkl
            ...

Train/val split: decide per scene. ContextVAE does not mix the two at
load time.


3.1 TRAJECTORY FILES: <scene>.txt
---------------------------------

One whitespace-separated row per (frame, agent) observation:

    fid  aid  x  y  [heading]  [group]

Fields:

    fid     REQUIRED  integer  frame id; must be consistent across agents in
                               the same scene. Missing frames are OK.
    aid     REQUIRED  integer  agent id; unique within the scene.
    x, y    REQUIRED  float    world-frame coordinates in METERS.
    heading OPTIONAL  float    yaw in RADIANS. If present, the dataloader
                               disables augmentation (flip/rotate/scale) and
                               uses this heading to rotate the trajectory
                               into the agent's local frame. Provide it
                               whenever you can — the model trains better.
    group   OPTIONAL  string   slash-separated tags, e.g. "VEHICLE" or
                               "VEHICLE/CHALLENGE". Must contain at least
                               one of the config's inclusive_groups for the
                               agent to be selectable as a prediction target.

Example rows (heading provided):

    0   1   123.45   678.90   0.7854   VEHICLE/TARGET
    0   2   125.10   679.22   0.8100   VEHICLE
    0   3   120.00   680.00   0.0000   PEDESTRIAN
    1   1   124.20   679.15   0.7854   VEHICLE/TARGET
    1   2   125.85   679.48   0.8100   VEHICLE
    1   3   120.01   680.04   0.0000   PEDESTRIAN
    ...

Example rows (heading omitted; augmentation will be active):

    0   1   123.45   678.90   VEHICLE/TARGET
    0   2   125.10   679.22   VEHICLE


3.2 INFO FILES: <scene>.info
----------------------------

A single line of text (or multiple lines for multi-scenario files; see
process_waymo.py for that style):

    <first_frame_id>  <map_name>

Where first_frame_id is the lowest fid that appears in the corresponding
.txt, and map_name is the basename (without ".pkl") of the map file
associated with the scene. If you are running mapless, still write this
file — just put any placeholder string for map_name and do not pass
--map_dir at training time.


3.3 MAP FILES: map/<map_name>.pkl   (OPTIONAL)
----------------------------------------------

Each pickle file contains a tuple (semantic_map, H):

    semantic_map   numpy.ndarray, shape (C, H_map, W_map), dtype float32,
                   values clipped to [-1, 1]. In the reference scripts C=3
                   (channel semantics vary per dataset; see section 4.5).

    H              numpy.ndarray, shape (3, 3). Homogeneous transform that
                   maps world (x, y, 1)^T to pixel (row, col, 1)^T. Use the
                   convention below exactly (row-major, origin at top-left):

                   H = [[  0,           -MAP_SCALE,   (y_c + 0.5*h)*MAP_SCALE],
                        [  MAP_SCALE,    0,          -(x_c - 0.5*w)*MAP_SCALE],
                        [  0,            0,           1                      ]]

                   where MAP_SCALE is pixels per meter (default 1), (x_c,
                   y_c) is the map center in world coords, and (h, w) is
                   the map size in meters.


================================================================================
4. IMPLEMENTATION STEPS FOR THE PREPROCESSING SCRIPT
================================================================================

Implement these steps in order. They are dataset-agnostic; the
dataset-specific work is only in steps 4.2 and 4.5.

4.1 PARSE COMMAND-LINE ARGUMENTS
--------------------------------

Minimum:

    data_root        path to the raw dataset
    target_folder    path to write the output
    --frameskip      integer, default 1. If the raw data is sampled faster
                     than you want the model to see, apply frameskip HERE
                     (keep every nth frame). Do NOT also apply it in the
                     dataloader; pick one place.
    --workers        number of parallel worker processes, optional

If you plan to use maps:

    --map_scale      pixels per meter, default 1
    --map_folder     where the raw map data lives (dataset-specific)


4.2 ITERATE OVER SCENES IN THE RAW DATASET
------------------------------------------

For each scene:

    a. Iterate over all agents observed in any frame of the scene.
    b. For each (frame, agent), extract:
         - frame index (integer, consistent Δ between consecutive frames)
         - agent id (integer, unique within the scene)
         - world-frame x, y in meters
         - heading in radians (if available)
         - agent type string (e.g. "VEHICLE", "PEDESTRIAN", "CYCLE")
         - a validity flag (if the dataset provides one; many do — e.g.
           Waymo Open Motion has state.valid per agent per frame)
    c. Drop frames where the validity flag is False. Do not write invalid
       rows into the .txt — they will corrupt finite-difference velocity
       and acceleration computation downstream.

Decide which agents are prediction TARGETS vs neighbors. Targets have a
specific tag in their group string (e.g. "TARGET", or "CHALLENGE" in
nuScenes). Neighbors just have their agent-type string (e.g. "VEHICLE").
The model only predicts trajectories for targets; everyone else is context.


4.3 CHECK DATA CONSISTENCY
--------------------------

Before writing, assert:

    - frame_ids in a scene are on a uniform grid (constant Δt). If not,
      resample upstream. The dataloader will raise
      "ValueError: Inconsistent frame interval" if the gcd of frame deltas
      does not equal the minimum delta.
    - every agent appears in at least 2 consecutive frames (single-frame
      agents are stripped by the dataloader anyway; dropping them
      upstream saves loading time).
    - coordinates are in METERS, not pixels or normalized units.
    - coordinates are in a GLOBAL (world) frame, not per-ego-centered.
      Centering is the dataloader's job; doing it yourself will break
      the neighbor radius filter and the map homography.


4.4 APPLY FRAMESKIP (IF ANY) AND WRITE THE .txt / .info FILES
-------------------------------------------------------------

If --frameskip > 1, keep every nth frame per agent. After skipping:

    - Sort rows by (frame_id, agent_id) for readability (not required, but
      makes debugging much easier).
    - Write the .txt with the exact field order from section 3.1.
    - Format hint: the reference scripts use fixed-width formatting to
      keep files diffable, e.g.:

          "{:<5d} {:<6d} {:10.4f} {:10.4f} {:7.4f} {}"

      This is cosmetic; any whitespace-separated float parser will read it.
    - Write the .info file with "<first_frame_id> <map_name>".


4.5 (OPTIONAL) GENERATE THE SEMANTIC MAP
----------------------------------------

Only if you intend to train the map-aware variant.

    a. Determine the geographic bounds of all agent trajectories in the
       scene; pad by ~300 m on each side.
    b. Rasterize the desired map layers into an image of
       (canvas_height, canvas_width) pixels at MAP_SCALE px/m.
    c. Stack the layers into a (C, H, W) array. Reference scripts use C=3:
         - nuScenes: [drivable_area+crosswalk, road_divider, lane_divider]
         - Lyft:     [lane+crosswalk,          road_divider, lane_divider]
         - Waymo:    [lanes,                   road_edges,   road_lines  ]
    d. Normalize to [-1, 1]: semantic_map = (semantic_map*2 - 1).clip(-1, 1)
    e. Build the homography H as shown in section 3.3.
    f. pickle.dump((semantic_map, H), open("map/<map_name>.pkl", "wb")).

For NEW domains (e.g. UAV orthoimagery), you can substitute any 3-channel
representation normalized to [-1, 1]. Only keep C=3 unless you also adjust
the map_model input convolution.


4.6 PARALLELIZE
---------------

Scene processing is embarrassingly parallel. Use
concurrent.futures.ProcessPoolExecutor. Each worker handles one scene
end-to-end and writes its own files. Do not share mutable state between
workers.


================================================================================
5. WHAT THE DATALOADER DOES WITH YOUR FILES (DO NOT REIMPLEMENT)
================================================================================

When training starts, data.py performs the following steps automatically.
You should understand them because they affect what "good" preprocessing
output looks like, but you do NOT write any of this code yourself.

5.1 EXTEND
    Reads your .txt. Verifies uniform Δt. Inserts empty gap-frames where
    needed. Computes per-agent velocity (vx, vy) and acceleration
    (ax, ay) via finite differences in the WORLD frame. Each agent record
    becomes (x, y, vx, vy, ax, ay, heading, group). The first six fields
    are the 6-channel state fed to the model.

5.2 SLIDE
    Slides a window of (ob_horizon + pred_horizon) frames over the scene
    with step = 1 frame. For each window, candidates whose full future
    horizon is observable become training targets. Remaining agents
    become neighbors.

5.3 RADIUS FILTER
    For each target, keeps only neighbors that come within ob_radius
    meters at some point during the window. Neighbors never within the
    radius are dropped entirely.

5.4 PADDING
    Missing neighbor states (agent not present at that frame) are filled
    with 1e9 on all 6 channels. The value 1e9 is required — zero would
    pass the radius filter and poison attention.

5.5 NEIGHBOR TEMPORAL SPAN
    The neighbor tensor covers the FULL window (L_ob + L_pred), not just
    the observation horizon. The posterior RNN during training consumes
    neighbor states at future frames. This is why you cannot strip out
    future neighbor observations during preprocessing.

5.6 LOCAL-FRAME ROTATION
    If heading is present, the dataloader rotates (x, y) of target,
    future, and neighbors so the target faces +x at t=1. Velocity and
    acceleration columns are NOT rotated — they remain world-frame.
    (This is an intentional design choice in the codebase; do not
    change it.)

5.7 MAP CROPPING
    When use_map is on, the dataloader projects the target's first-frame
    world position to map pixels via H, crops a 2*EXT window around it,
    rotates the patch by -heading with grid_sample, and slices to 224x224
    so the target sits at row 122, col 51 of the patch.


================================================================================
6. CONFIGURING AND RUNNING TRAINING
================================================================================

6.1 CREATE A CONFIG FILE
------------------------

Copy config/nuscenes_train.py to config/<yours>_train.py and edit. The
fields you need to set:

    OB_HORIZON      observation length in FRAMES (after frameskip)
    PRED_HORIZON    prediction length in FRAMES (after frameskip)
    MIN_OB_HORIZON  optional; allow shorter history, padded to OB_HORIZON
    OB_RADIUS       neighbor distance cutoff in METERS (stock: 30)
    MAP_SIZE        map crop size in pixels (stock: 224)

    lr              learning rate (stock: 3e-4, Adam, no scheduler)
    epochs          total epochs
    test_since      first epoch on which to run evaluation
    preload_data    True if the full dataset fits in RAM
    pred_samples    k in minADE_k / minFDE_k during evaluation (stock: 5)

    train_dataloader / test_dataloader dicts:
        inclusive_groups   list of group tags that mark prediction targets
                           e.g. ["VEHICLE", "EGO"] or ["CHALLENGE"]
        batch_size         training batch size
        batches_per_epoch  optional cap on batches per epoch
        traj_max_overlap   evaluation-only; caps window overlap to reduce
                           eval time

    model dict:
        horizon            must equal PRED_HORIZON
        ob_radius          must equal OB_RADIUS
        hidden_dim         RNN hidden size (stock: 512)
        map_model          one of: "resnet18", "resnet152", "mobile2",
                                   "efficientnet-b0", etc.
                           Omit / ignore if running mapless.

Reference horizon values from the three shipped configs:

    nuScenes (2 FPS): ob_horizon=5, pred_horizon=12 -> 2.5 s / 6 s
    Lyft     (5 FPS): ob_horizon=6, pred_horizon=15 -> 1.2 s / 3 s
    Waymo    (5 FPS): ob_horizon=6, pred_horizon=15 -> 1.2 s / 3 s


6.2 LAUNCH TRAINING
-------------------

Single GPU (mapless, recommended first pass):

    python main.py \
        --train <data>/train \
        --test  <data>/val \
        --config config/<yours>_train.py \
        --ckpt   runs/<yours>_nomap

Single GPU with maps:

    python main.py \
        --train <data>/train \
        --test  <data>/val \
        --map_dir <data>/map \
        --config  config/<yours>_train.py \
        --ckpt    runs/<yours>_mapped

Multi-GPU (sharded DDP; one process per shard index):

    python main.py --train <data>/train/0 --test <data>/val/0 \
        --config config/<yours>_train.py --ckpt runs/<yours> \
        --rank 0 --workers 4
    python main.py --train <data>/train/1 --test <data>/val/1 \
        --config config/<yours>_train.py --ckpt runs/<yours> \
        --rank 1 --workers 4 --master_addr <ip_of_rank0>
    ...

Evaluation:

    python main.py \
        --test <data>/val [--map_dir <data>/map] \
        --config config/<yours>_eval.py \
        --ckpt   runs/<yours>/ckpt-best


6.3 WHAT THE TRAINING LOOP DOES
-------------------------------

main.py:
    - Builds Dataloader(train_data) and Dataloader(test_data).
    - Instantiates ContextVAE(**config.model) and Adam(lr=config.lr).
    - Loops for config.epochs epochs:
        - For each batch, calls model(*batch) -> (err, kl)
          and model.loss(err, kl) -> {"loss": rec + kl, "rec": ..., "kl": ...}.
        - loss["loss"].backward(); optimizer.step(); optimizer.zero_grad().
        - From epoch config.test_since onward, evaluates minADE_k / minFDE_k
          on the test set and saves ckpt-best if minADE improves.
    - Always saves ckpt-last every epoch.

You do not need to touch this loop for a new dataset.


================================================================================
7. SANITY CHECKS BEFORE YOU COMMIT TO A FULL TRAINING RUN
================================================================================

Do these in order. Each one catches a class of bugs that would otherwise
waste hours of GPU time.

7.1 SMOKE TEST THE FORMAT

    Hand-write one tiny .txt with 3 agents and ~30 frames (section 8
    has an example). Point a minimal config at it. Confirm the
    dataloader loads it without error.

7.2 CHECK TENSOR SHAPES

    Grab one batch manually:

        loader = Dataloader(...); batch = next(iter(loader))
        x, y, neighbor, *rest = batch
        assert x.shape == (OB_HORIZON, N, 6)
        assert y.shape == (PRED_HORIZON, N, 2)
        assert neighbor.shape == (OB_HORIZON + PRED_HORIZON, N, Nn, 6)

7.3 CHECK NEIGHBOR PADDING

    Confirm that absent-neighbor slots equal 1e9, not 0. A fast check:
    neighbor.max() should be >= 1e9 whenever Nn > number of actually
    present neighbors.

7.4 VISUALIZE A FEW TRAJECTORIES

    Plot x[:, i, :2] in the local frame. At t=0 the target should be at
    the origin and oriented along +x (if you supplied heading). Plot
    the neighbors' first-frame positions — they should be within
    ob_radius of the origin.

7.5 OVERFIT A TINY SUBSET

    Take 100 scenes. Train for 50 epochs. If ELBO (rec + kl) does not
    drop monotonically, your data pipeline is broken, not the model.
    Fix it here before scaling up.

7.6 BASELINE COMPARISON

    Once training runs end-to-end, compare against Constant Velocity:
    if ContextVAE does not clearly beat it on minADE_1 after enough
    epochs, the model is not learning from your data. Likely causes:
    wrong ob_radius for the scale of your scenes, wrong frameskip,
    targets filtered out by inclusive_groups, or wrong units
    (pixels instead of meters).


================================================================================
8. MINIMAL EXAMPLE FILES
================================================================================

scene_0001.txt:

    0   1   0.00   0.00   0.0000   VEHICLE/TARGET
    0   2   5.00   0.50   0.1000   VEHICLE
    0   3  -3.00   2.00   0.0000   PEDESTRIAN
    1   1   1.00   0.00   0.0000   VEHICLE/TARGET
    1   2   5.25   0.55   0.1000   VEHICLE
    1   3  -3.02   2.10   0.0000   PEDESTRIAN
    2   1   2.00   0.01   0.0100   VEHICLE/TARGET
    2   2   5.50   0.60   0.1000   VEHICLE
    2   3  -3.04   2.20   0.0000   PEDESTRIAN
    ...

scene_0001.info:

    0   dummy_map

Corresponding config snippet (mapless):

    OB_HORIZON   = 5
    PRED_HORIZON = 10
    OB_RADIUS    = 30

    train_dataloader = dict(
        ob_horizon=OB_HORIZON,
        pred_horizon=PRED_HORIZON,
        ob_radius=OB_RADIUS,
        inclusive_groups=["TARGET"],
        batch_size=32,
    )


================================================================================
9. COMMON PITFALLS — SYMPTOMS AND FIXES
================================================================================

Pitfall                                  Symptom                                       Fix
---------------------------------------  --------------------------------------------  ----------------------------------------------
Zero-padding neighbor slots              Training loss spikes or attention degenerates Use 1e9 for absent neighbor states
Neighbor tensor only spans L_ob          Shape error or NaNs from posterior RNN        Span full (L_ob + L_pred)
Frames with validity=False not filtered  Huge spikes in finite-difference v and a      Drop invalid rows in preprocessing
Per-ego-centered coordinates             Neighbor radius filter drops everyone         Emit GLOBAL world coords
Inconsistent Δt across agents            "ValueError: Inconsistent frame interval"     Resample to a fixed rate upstream
frameskip applied twice                  v and a on unexpected scale; training fragile Apply in preprocessing OR loader, not both
Missing heading when you wanted it       Augmentation silently on; noisy predictions   Supply heading OR accept augmentation
Map H using column-major convention      Crops show black stripes or off-center agent  Use the exact H from section 3.3
Coordinates in pixels, not meters        ob_radius filters everyone; training fails    Convert to meters before writing .txt


================================================================================
10. CHECKLIST BEFORE FIRST TRAINING RUN
================================================================================

[ ] Each .txt has rows in the exact order (fid aid x y [heading] [group])
[ ] Coordinates are in meters, world frame, not pre-centered
[ ] Frame ids are on a uniform grid (constant Δt)
[ ] Single-frame agents have been removed
[ ] Agents of interest have a group tag that matches inclusive_groups
[ ] Each scene has a matching .info file
[ ] If using maps: each map pickle has (semantic_map (C,H,W) in [-1,1], H (3,3))
[ ] Config OB_HORIZON and PRED_HORIZON match the frame rate you wrote at
[ ] config.model.horizon == PRED_HORIZON
[ ] Smoke test passed on a hand-written toy scene
[ ] Tensor shapes verified from a real batch
[ ] Neighbor padding verified as 1e9
[ ] Overfit a small subset to confirm the model learns

If every box is checked, launch the full training run.
