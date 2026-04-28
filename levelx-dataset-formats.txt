# ==============================================================================
# levelXdata Dataset Formats — Reference
# ==============================================================================
# Source: levelXdata format specification PDFs (RWTH Aachen / ika), version 1.1
#         (uniDFormat1_1.pdf, inDFormat_1_1.pdf, rounDFormat.pdf, highDFormat.pdf)
# Provider: ika, RWTH Aachen University — https://levelxdata.com
# Last format spec date observed: 2024-03-08
# ==============================================================================
# Purpose:
#   Reference for ContextVAE adaptation to UAV trajectory prediction. Used to
#   build the inD/uniD preprocessing pipeline (per Thesis_Plan.docx, Week 1).
#   This file documents column names, units, semantics, coordinate frames, and
#   key inter-dataset differences for downstream loaders.
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. SUMMARY OF DATASETS
# ------------------------------------------------------------------------------
# Three datasets share an identical urban/roundabout format (Format v1.1):
#   - uniD : Urban intersections (primarily university campus context)
#   - inD  : Inner-city / urban intersections
#   - rounD: Roundabouts
#
# One dataset uses a separate, older format:
#   - highD: Highway recordings, image-coordinate frame, lane-aware extras
#
# Per-recording file set (all four datasets):
#   XX_recordingMeta.csv   recording-level metadata
#   XX_tracksMeta.csv      per-track summary (one row per agent)
#   XX_tracks.csv          per-frame trajectory rows (the bulk data)
#   XX_background.png      georeferenced top-down image of the road section
#                          (highD uses XX_highway.jpg instead)
#
# Per-location map data (uniD/inD/rounD only — NOT highD):
#   Lanelet2 maps  (.osm)
#   ASAM OpenDRIVE (.xodr, v1.4)
#   3D scene       (.osgb, .fbx)
#
# Convention (uniD/inD/rounD): pedestrians + bicyclists + motorcycles are
# grouped as Vulnerable Road Users (VRUs). VRU bounding-box width and length
# are reported as 0.


# ==============================================================================
# 1. URBAN/ROUNDABOUT FORMAT — uniD, inD, rounD  (identical schema, v1.1)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1  XX_recordingMeta.csv  (one row per recording)
# ------------------------------------------------------------------------------
# Column          Type   Unit     Description
# recordingId     int    -        Unique id of the recording.
# locationId      int    -        Id of the recording location (joins to maps).
# frameRate       float  Hz       Video frame rate (typically ~25 Hz for inD/uniD).
# speedLimit      float  m/s      Speed limit; same for every lane in a recording.
# weekday         str    -        Day of the week of the recording.
# startTime       int    hh       Hour at which recording started.
# duration        float  s        Recording duration.
# numTracks       int    -        Total number of tracked objects.
# numVehicles     int    -        Number of tracked vehicles.
# numVrus         int    -        Number of tracked VRUs.
# latLocation     float  deg      Approximate latitude (NOT UTM; for rough geo only).
# lonLocation     float  deg      Approximate longitude (NOT UTM; for rough geo only).
# xUtmOrigin      float  m        UTM x of the local frame origin. Add to xCenter
#                                 to recover global UTM x.
# yUtmOrigin      float  m        UTM y of the local frame origin. Add to yCenter
#                                 to recover global UTM y.
# orthoPxToMeter  float  m/px     Scale of XX_background.png (ortho px -> meters).
#                                 Use for image-pixel <-> metric conversion.
# exportVersion   str    -        Format version. May be missing in older exports.

# ------------------------------------------------------------------------------
# 1.2  XX_tracksMeta.csv  (one row per track)
# ------------------------------------------------------------------------------
# Column        Type  Unit  Description
# recordingId   int   -     Recording id (same for all rows of a recording).
# trackId       int   -     Track id; assigned in ascending order per recording.
# initialFrame  int   -     First frame of the track.
# finalFrame    int   -     Last frame of the track.
# numFrames     int   -     Total lifetime in frames (= finalFrame - initialFrame + 1).
# width         float m     Object width. 0 for VRUs.
# length        float m     Object length. 0 for VRUs.
# class         str   -     Object class label (e.g., car, truck, bus, pedestrian,
#                           bicycle, motorcycle). Use to filter VRU vs vehicle.

# ------------------------------------------------------------------------------
# 1.3  XX_tracks.csv  (one row per (track, frame) pair — the trajectory data)
# ------------------------------------------------------------------------------
# Column          Type  Unit    Description
# recordingId     int   -       Recording id.
# trackId         int   -       Track id (joins to tracksMeta).
# frame           int   -       Frame index for this row.
# trackLifetime   int   -       Age of the track at this frame (= frame - initialFrame).
# xCenter         float m       x of object centroid in LOCAL frame.
#                               Add xUtmOrigin to obtain global UTM x.
# yCenter         float m       y of object centroid in LOCAL frame.
#                               Add yUtmOrigin to obtain global UTM y.
# heading         float deg     Heading in the local frame (UTM convention; see §1.4).
# width           float m       Bounding-box width. 0 for VRUs.
# length          float m       Bounding-box length. 0 for VRUs.
# xVelocity       float m/s     Velocity along x-axis of local frame.
# yVelocity       float m/s     Velocity along y-axis of local frame.
# xAcceleration   float m/s^2   Acceleration along x-axis of local frame.
# yAcceleration   float m/s^2   Acceleration along y-axis of local frame.
# lonVelocity     float m/s     Longitudinal velocity (along heading).
# latVelocity     float m/s     Lateral velocity (perpendicular to heading).
# lonAcceleration float m/s^2   Longitudinal acceleration.
# latAcceleration float m/s^2   Lateral acceleration.
#
# NOTE: Spec PDF labels the `length` column description as "height" of the
#       object — this is a typo in the spec; the column represents the object's
#       physical length in meters (along its heading), as for `tracksMeta.length`.

# ------------------------------------------------------------------------------
# 1.4  Coordinate System (uniD / inD / rounD)
# ------------------------------------------------------------------------------
# - Global frame: UTM (because data is geo-referenced).
# - Local frame:  UTM-translated; origin (0, 0) sits near the recording site.
#                 SAME local origin across all recordings of one locationId.
# - Axes (local): +x grows to the right, +y grows UPWARDS (right-handed).
# - Heading:      UTM-convention angle, in degrees, in the local frame.
# - Units:        SI throughout (meters, seconds, radians/degrees as labelled).
# - To recover global UTM: x_utm = xCenter + xUtmOrigin
#                          y_utm = yCenter + yUtmOrigin
# - Image alignment: XX_background.png is georeferenced to the local frame via
#                    orthoPxToMeter. Use this scale (and the image extent) to
#                    crop agent-centered, heading-rotated patches for a map
#                    encoder (e.g., ContextVAE M-ATTN aerial-orthomap input).

# ------------------------------------------------------------------------------
# 1.5  Maps (uniD / inD / rounD — NOT highD)
# ------------------------------------------------------------------------------
# ASAM OpenDRIVE v1.4 (.xodr): road network, lane connections, lane width/type
#   (including sidewalks and bicycle lanes), markings, speed limits, traffic
#   islands, parking areas, road stencil markings, key traffic signs (Stop/
#   Yield/Turn), roundabouts, guardrails, crosswalks (incl. zebra), restricted-
#   area markings. Compatible with esmini, CARLA, and similar simulators.
# Lanelet2 (.osm): road network with predecessors/successors, road shape and
#   width, lane number/type (incl. sidewalks and bicycle lanes), lane markings,
#   traffic islands, parking areas, building/park/vegetation surroundings,
#   generalized curb-stone heights, intersection areas, virtual connection
#   lanes on intersections, roundabouts, regulatory elements (traffic lights,
#   right-of-way, stop/wait lines), crosswalks, restricted-area markings.
#   Lanelets are tagged with a `speed_limit` (with unit km/h or mph).
# 3D scene (.osgb, .fbx): for visualization / simulation.


# ==============================================================================
# 2. HIGHWAY FORMAT — highD  (separate, older schema)
# ==============================================================================
# 60 recordings; six locations.
# Image file: XX_highway.jpg (NOT XX_background.png).
# No Lanelet2 / OpenDRIVE / 3D scene maps shipped with highD.

# ------------------------------------------------------------------------------
# 2.1  XX_recordingMeta.csv (highD)  (one row per recording)
# ------------------------------------------------------------------------------
# Column              Type   Unit   Description
# id                  int    -      Unique recording id (note: NOT `recordingId`).
# frameRate           float  Hz     Video frame rate (typically 25 Hz).
# locationId          int    -      Recording location id (six total in highD).
# speedLimit          float  m/s    Speed limit; same for all lanes.
# month               str    -      Month of recording.
# weekDay             str    -      Day of week.
# startTime           str    hh:mm  Start time of recording.
# duration            float  s      Recording duration.
# totalDrivenDistance float  m      Sum of distance covered by all tracked vehicles.
# totalDrivenTime     float  s      Sum of driven time across all vehicles.
# numVehicles         int    -      Number of vehicles tracked (cars + trucks).
# numCars             int    -      Number of cars tracked.
# numTrucks           int    -      Number of trucks tracked.
# upperLaneMarkings   str    m      y-positions of upper lane markings,
#                                   semicolon-separated (e.g., "12.34;15.67;...").
# lowerLaneMarkings   str    m      y-positions of lower lane markings,
#                                   semicolon-separated.

# ------------------------------------------------------------------------------
# 2.2  XX_tracksMeta.csv (highD)  (one row per track)
# ------------------------------------------------------------------------------
# Column            Type  Unit  Description
# id                int   -     Track id (note: NOT `trackId`).
# width             float m     Post-processed bbox width = vehicle LENGTH (sic).
# height            float m     Post-processed bbox height = vehicle WIDTH (sic).
#                               (Highway image-frame convention; see §2.4.)
# initialFrame      int   -     First frame of the track.
# finalFrame        int   -     Last frame of the track.
# numFrames         int   -     Track lifetime in frames.
# class             str   -     "Car" or "Truck".
# drivingDirection  int   -     1 = left direction (upper lanes),
#                               2 = right direction (lower lanes).
# traveledDistance  float m     Total distance covered.
# minXVelocity      float m/s   Minimum velocity along driving direction.
# maxXVelocity      float m/s   Maximum velocity along driving direction.
# meanXVelocity     float m/s   Mean velocity along driving direction.
# minDHW            float m     Minimum Distance Headway. -1 if no preceding car.
# minTHW            float s     Minimum Time Headway. -1 if no preceding car.
# minTTC            float s     Minimum Time-to-Collision. -1 if invalid/none.
# numLaneChanges    int   -     Lane changes detected (by laneId change).

# ------------------------------------------------------------------------------
# 2.3  XX_tracks.csv (highD)  (one row per (track, frame))
# ------------------------------------------------------------------------------
# Column              Type  Unit    Description
# frame               int   -       Frame index.
# id                  int   -       Track id.
# x                   float m       UPPER-LEFT corner of bbox (NOT centroid).
# y                   float m       UPPER-LEFT corner of bbox (NOT centroid).
# width               float m       Bbox width along x (driving direction).
# height              float m       Bbox height along y (lateral).
# xVelocity           float m/s     Longitudinal velocity (image frame).
# yVelocity           float m/s     Lateral velocity (image frame).
# xAcceleration       float m/s^2   Longitudinal acceleration.
# yAcceleration       float m/s^2   Lateral acceleration.
# frontSightDistance  float m       Distance from vehicle CENTER to end of recorded
#                                   highway section in driving direction.
# backSightDistance   float m       Same, but opposite of driving direction.
# dhw                 float m       Distance Headway. 0 if no preceding vehicle.
# thw                 float s       Time Headway. 0 if no preceding vehicle.
# ttc                 float s       Time-to-Collision. 0 if no preceding/invalid.
# precedingXVelocity  float m/s     Longitudinal velocity of preceding vehicle.
#                                   0 if none.
# precedingId         int   -       Id of preceding vehicle in the same lane. 0 if none.
# followingId         int   -       Id of following vehicle in the same lane. 0 if none.
# leftPrecedingId     int   -       Id of preceding vehicle in adjacent left lane.
# leftAlongsideId     int   -       Id of alongside vehicle in adjacent left lane
#                                   (must overlap longitudinally, else preceding/
#                                   following). 0 if none.
# leftFollowingId     int   -       Id of following vehicle in adjacent left lane.
# rightPrecedingId    int   -       Id of preceding vehicle in adjacent right lane.
# rightAlongsideId    int   -       Id of alongside vehicle in adjacent right lane.
#                                   (Spec PDF spells this `rightAlsongsideId` —
#                                   a typo in the spec; verify exact column header
#                                   in the actual CSV before loading.)
# rightFollowingId    int   -       Id of following vehicle in adjacent right lane.
# laneId              int   -       Lane id, 1-indexed; ids derive from lane-marking
#                                   y-positions in recordingMeta. The first and
#                                   last ids may not correspond to drivable lanes.

# ------------------------------------------------------------------------------
# 2.4  Coordinate System (highD)
# ------------------------------------------------------------------------------
# - Global frame: video IMAGE coordinate system (top-left origin).
# - Axes:         +x grows to the right (= direction of travel for lower lanes);
#                 +y grows DOWNWARDS (image convention).
# - Units:        SI (meters, seconds), even though the frame is image-derived
#                 — pixels were converted to meters via calibration.
# - Stabilization: video was stabilized so lane markings run horizontally; lanes
#                  are therefore identifiable purely by their y-coordinate.
# - Implication:  vehicles in the upper lanes move to the LEFT and have a
#                 NEGATIVE xVelocity; lower-lane vehicles have positive xVelocity.
# - WARNING:      `width` and `height` semantics in tracksMeta are physically
#                 swapped relative to intuition (width = vehicle LENGTH along
#                 driving direction, height = vehicle WIDTH lateral). This is a
#                 known quirk of the highD spec.


# ==============================================================================
# 3. KEY DIFFERENCES — uniD/inD/rounD  vs  highD
# ==============================================================================
# Aspect                 | uniD / inD / rounD               | highD
# -----------------------+----------------------------------+-----------------------
# Scene type             | Urban intersections / roundabts. | Highway.
# Camera                 | Hovering drone, top-down.        | Hovering drone, top-down.
# Coordinate frame       | Local UTM-translated, +y UP.     | Image frame, +y DOWN.
# Origin                 | Per-locationId, near road site.  | Image top-left.
# Position semantics     | xCenter, yCenter (centroid).     | x, y (UPPER-LEFT bbox).
# Heading available      | Yes (`heading`, deg, UTM conv.). | No explicit heading;
#                        |                                  | infer from xVelocity sign.
# Track id column name   | `trackId`.                       | `id`.
# Recording id column    | `recordingId`.                   | `id` (in recordingMeta).
# VRUs included          | Yes (pedestrians, bikes, motos). | No (cars + trucks only).
# Object width/length    | `width`, `length` (centroid bbox)| `width`, `height` (swapped
#                        |                                  | semantics — see §2.4).
# Per-frame neighbor IDs | Not provided; compute from        | Provided directly
#                        | positions + radius/lanelet.       | (preceding/following/
#                        |                                  | left/right neighbors).
# Lane info              | Lanelet2 / OpenDRIVE maps         | `laneId` per row, plus
#                        | (.osm / .xodr).                   | lane-marking y-positions
#                        |                                  | in recordingMeta.
# Map files              | Lanelet2, OpenDRIVE, 3D scene.    | None shipped.
# Background image       | XX_background.png                 | XX_highway.jpg (no
#                        | (georeferenced; orthoPxToMeter).  | UTM georeferencing).
# UTM origin fields      | `xUtmOrigin`, `yUtmOrigin`.       | None.
# Headway metrics        | Not in tracks (compute manually). | DHW, THW, TTC per frame
#                        |                                  | + min* aggregates in
#                        |                                  | tracksMeta.


# ==============================================================================
# 4. ContextVAE / Pipeline Notes  (project-specific)
# ==============================================================================
# Per Thesis_Plan.docx:
#
# - Primary datasets: inD + uniD (~10 GB total). Aerial drone perspective with
#   hovering camera, metric SI coordinates already provided. Per-recording
#   georeferenced background image with `orthoPxToMeter` enables agent-centered
#   orthomap patch extraction for the M-ATTN encoder.
#
# - Coordinate frame for ContextVAE: ego-normalized at last observation frame
#   t = T. Rotate target trajectory so heading at frame T aligns with +x.
#   `heading` (in degrees, UTM convention) is provided directly in the urban
#   tracks.csv — convert to radians for rotation matrices.
#
# - Frame-rate handling: inD/uniD are recorded at ~25 Hz; downsample to 5 Hz
#   (every 5th frame). Observation L1 = 10 frames (2 s); prediction H = 25
#   frames (5 s). Verify exact `frameRate` per recording from recordingMeta.
#
# - 6-channel agent state (ContextVAE convention):
#       [xCenter, yCenter, xVelocity, yVelocity, xAcceleration, yAcceleration]
#   All six are present per row in the urban `tracks.csv`. Do NOT use
#   lon/lat velocities/accelerations unless explicitly desired — those are
#   already ego-frame-rotated and would conflict with re-rotation at preprocessing.
#
# - Neighbor handling at training time:
#     * `neighbor` tensor MUST cover the full `L1 + L2` (observation + prediction)
#       window — the posterior RNN consumes future neighbor states.
#     * Pad absent neighbors with `1e9` (NOT zeros) so the radius-distance mask
#       filters them correctly. Zero-padding is a known footgun.
#     * Neighbor selection: radius-based with `ob_radius = 30 m` per the plan.
#       Compute pairwise distances using `xCenter`, `yCenter` (local frame is
#       fine — no need to convert to UTM for relative distances).
#
# - Map patch (M-ATTN, aerial orthomap variant):
#     * Crop XX_background.png centered on the agent's (xCenter, yCenter) at
#       frame T, rotated so the agent's `heading` points along +x (or +y,
#       matching ContextVAE's convention — verify against the reference repo).
#     * Patch extent: ~60 m x 60 m (starting guess per plan).
#     * Pixel size at extraction: `60 / orthoPxToMeter` pixels per side, then
#       resize to the CNN input resolution.
#     * Use `xUtmOrigin`, `yUtmOrigin` only if combining recordings from the
#       same `locationId` and you need a shared image — within a single
#       recording, the local-frame xCenter/yCenter map directly to image pixels
#       via `orthoPxToMeter` (after accounting for the image's local-frame
#       extent and the +y-up vs image-y-down flip).
#
# - Class filtering: tracksMeta.class identifies vehicles vs VRUs. For an
#   initial vehicle-only model, keep only vehicle classes (car, truck, bus,
#   van — exact label set varies by location/recording; inspect the unique
#   `class` values in tracksMeta before hard-coding).
#
# - Fallback / baseline: no-map S-ATTN-only ablation (use_map=False). Per the
#   ContextVAE supplementary (Table S3), this configuration achieves ~95% of
#   full-model performance and is a sensible UAV-deployment baseline.
#
# - rounD as roundabout-only ablation: schema is identical to inD/uniD, so it
#   plugs into the same loader without code changes — only the data split and
#   evaluation logging need a new tag.
#
# - highD is intentionally OUT of scope for the M-ATTN aerial-orthomap variant
#   (no maps shipped, image-frame coordinates, no VRUs, swapped width/height
#   semantics). Use only if a highway-specific baseline is needed; the loader
#   would need a separate code path.


# ==============================================================================
# 5. Quick Loader Sketch  (Python / pandas, urban format)
# ==============================================================================
# import pandas as pd
#
# rec_meta    = pd.read_csv(f"{rec_id:02d}_recordingMeta.csv")
# tracks_meta = pd.read_csv(f"{rec_id:02d}_tracksMeta.csv")
# tracks      = pd.read_csv(f"{rec_id:02d}_tracks.csv")
#
# # Filter to vehicles only (class label set varies — inspect unique values).
# vehicle_classes = {"car", "truck", "bus", "van"}
# vehicle_ids = tracks_meta.loc[
#     tracks_meta["class"].str.lower().isin(vehicle_classes), "trackId"
# ]
# tracks_v = tracks[tracks["trackId"].isin(vehicle_ids)]
#
# # Downsample 25 Hz -> 5 Hz (every 5th frame).
# tracks_v = tracks_v[tracks_v["frame"] % 5 == 0].copy()
#
# # 6-D state vector for ContextVAE.
# state_cols = ["xCenter", "yCenter",
#               "xVelocity", "yVelocity",
#               "xAcceleration", "yAcceleration"]
# # tracks_v.groupby("trackId")[state_cols] -> per-agent trajectory tensors.
# # (Pad with 1e9 for absent timesteps; align frames across agents per scene.)


# ==============================================================================
# 6. Sources
# ==============================================================================
# - uniD format spec     : /mnt/project/uniDFormat1_1.pdf  (v1.1, 2024-03-08)
# - inD format spec      : /mnt/project/inDFormat_1_1.pdf  (v1.1, 2024-03-08)
# - rounD format spec    : /mnt/project/rounDFormat.pdf    (2024-03-08)
# - highD format spec    : /mnt/project/highDFormat.pdf
# - levelXdata website   : https://levelxdata.com
# - inD paper            : Bock et al., "The inD Dataset", IV 2020,
#                          arXiv:1911.07602
# - rounD paper          : Krajewski et al., "The rounD Dataset", ITSC 2020
# - highD paper          : Krajewski et al., "The highD Dataset", ITSC 2018,
#                          DOI:10.1109/ITSC.2018.8569552
# - uniD                 : levelXdata (no separate paper as of spec date).
# - ContextVAE paper     : /mnt/project/2023_PeiXu_ContextVAE.pdf
#                          (Xu, Hayet, Karamouzas, IEEE RA-L / ICRA 2024)
# - ContextVAE repo      : https://github.com/xupei0610/ContextVAE
# ==============================================================================