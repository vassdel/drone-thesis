import os, sys
import numpy as np
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Waymo Open Motion -> ContextVAE on-disk format. Per TFRecord shard writes
# <shard>.txt trajectory rows and <shard>.info (per-scenario fid offset + id),
# plus one rasterized semantic map per scenario map/.../<scenario>.pkl =
# (semantic_map [3,H,W] in [-1,1], H 3x3). Scenarios are sharded into --split
# groups for distributed training. Native 10 Hz data is downsampled by
# --frameskip (2 -> 5 Hz).
#
# Environment:
#   conda create -n waymo && conda activate waymo
#   conda install python=3.9
#   conda install -c conda-forge openexr-python gsutil
#   pip install waymo-open-dataset-tf-2-6-0 opencv-python
#
# Data download (~444 GB):
#   mkdir dataset
#   gsutil -m cp -r \
#     "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training" \
#     "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/validation" \
#     dataset
#
# Run:
#   python process_waymo.py dataset 5FPS --frameskip 2 --split 8

import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

# Agent-quality thresholds, adapted from l5kit's select_agents.py: break a track
# on a large yaw change, bbox-area ratio, distance from the ego vehicle, or a
# per-step displacement jump.
TH_YAW_DEGREE = 30
TH_EXTENT_RATIO = 1.1
TH_DISTANCE_AV = 50
TH_DISPLACEMENT_DISTANCE = 4.5  # meters per 0.1 s step
def angular_distance(ang0, ang1):
    return (ang0-ang1 + np.pi) % (2*np.pi) - np.pi
def in_angular_distance(yaw0, yaw1, threshold):
    return abs(angular_distance(yaw1, yaw0))*180/np.pi < threshold
def in_extent_ratio(extent0, extent1, threshold):
    area0 = extent0[0]*extent0[1]
    area1 = extent1[0]*extent1[1]
    if area0 < 0.0 or area1 < 0.01: return False
    ratio = area0/area1 if area0 > area1 else area1/area0
    return ratio < threshold
def in_displacement_distance(pos0, pos1, threshold):
    dx = pos1[0]-pos0[0]
    dy = pos1[1]-pos0[1]
    return dx*dx + dy*dy < threshold*threshold


# Sub-pixel rasterization settings, adapted from l5kit's semantic_rasterizer.py.
# cv2 draws at fixed-point precision given by `shift`.
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]

# Map world polylines through H to fixed-point cv2 pixel coords, split back into
# the original per-polyline groups.
def world2subpixel(polylines, H):
    indices = np.cumsum([len(_) for _ in polylines])
    points = np.concatenate(polylines).reshape(-1, 2, 1)
    points = np.concatenate((points, np.ones((points.shape[0], 1, 1))), 1)
    pixels = (H @ points).squeeze(-1)[:,:2]
    # cv2 expects (column, row); H produces (row, column), so swap.
    pixels = pixels[:,::-1]
    subpixels = pixels * CV2_SHIFT_VALUE
    subpixels = subpixels.astype(np.int)
    return np.split(subpixels, indices[:-1])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("target_folder", type=str)
parser.add_argument("--frameskip", type=int, default=2)
parser.add_argument("--map_scale", type=int, default=1)
parser.add_argument("--workers", type=int, default=None)
parser.add_argument("--split", type=int, default=8)
settings = parser.parse_args()

FRAMESKIP = settings.frameskip
MAP_SCALE = settings.map_scale
WORKERS = min(30, multiprocessing.cpu_count()) if settings.workers is None else settings.workers
MARGIN = 280  # meters of padding around the agent extent for the map raster

# Decode one scenario proto into trajectory rows plus a rasterized semantic map.
# Returns (name, semantic_map, H, x, y, heading, info) or None if empty. When
# track_suggested_only, only Waymo's "tracks_to_predict" agents are targets.
def parse(scenario_data, track_suggested_only=True):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(scenario_data.numpy())
    name = scenario.scenario_id

    tracks = dict()
    count = 0
    agents = {0:[]}

    # Self-driving-car positions per frame, used for the distance-to-ego filter.
    ego = [[s.center_x, s.center_y] for s in scenario.tracks[scenario.sdc_track_index].states]

    # Track indices Waymo flags as prediction targets.
    toi = [i.track_index for i in scenario.tracks_to_predict]

    for i, track in enumerate(scenario.tracks):
        if scenario.sdc_track_index == i:
            agent_type = "EGO"
        elif track.object_type == 1:
            agent_type = "VEHICLE"
        elif track.object_type == 2:
            agent_type = "PEDESTRIAN"
        elif track.object_type == 3:
            agent_type = "CYCLIST"
        else:
            continue
        interested = i in toi if track_suggested_only else True
        
        for fid, s in enumerate(track.states):
            if not s.valid:
                continue
            need_record = fid % FRAMESKIP == 0
            tracked = track.id in tracks and tracks[track.id] is not None
            # Wrap heading to (-pi, pi].
            heading = s.heading % (2*np.pi)
            if heading > np.pi: heading -= 2*np.pi

            extent = s.length, s.width, s.height
            pos = s.center_x, s.center_y
            lost = False
            if agent_type == "EGO":
                agent_id = 0
            else:
                # Even suggested-to-predict agents are filtered: some are poorly
                # tracked in the source data.
                if np.linalg.norm([s.center_x-ego[fid][0], s.center_y-ego[fid][1]]) > TH_DISTANCE_AV:
                    # Too far from ego (Lyft-style distance filter): mark lost.
                    lost = True
                if tracked:
                    # Targets may legitimately skip a frame at the downsampled
                    # rate, so accept a FRAMESKIP-sized gap; others must be
                    # consecutive.
                    interval = fid - tracks[track.id][1][0]
                    if track_suggested_only and interested:
                        tracked = interval == 1 or interval == FRAMESKIP
                    else:
                        tracked = interval == 1
                # Break the track on a frame gap or a large displacement/yaw/
                # extent jump: start a fresh agent id.
                if not tracked or \
                (not in_displacement_distance(tracks[track.id][1][3], pos, TH_DISPLACEMENT_DISTANCE)) or \
                (not in_angular_distance(tracks[track.id][1][1], heading, TH_YAW_DEGREE)) or \
                (not in_extent_ratio(tracks[track.id][1][2], extent, TH_EXTENT_RATIO)):
                    agent_id = None
                else:
                    agent_id = tracks[track.id][0]
            if need_record:
                if agent_id == 0:
                    a_type = "EGO"
                else:
                    if agent_id is None:
                        count += 1
                        agent_id = count
                    if agent_id not in agents: agents[agent_id] = []
                    # Tag non-target vehicles "_NOINTREST" and lost agents
                    # "_IGNORE"; the loader keeps them as context but not targets.
                    if agent_type == "VEHICLE" and not interested:
                        a_type = agent_type + "_NOINTREST"
                    elif lost:
                        a_type = agent_type + "_IGNORE"
                    else:
                        a_type = agent_type
                agents[agent_id].append((s.center_x, s.center_y, heading, a_type, extent, track.id, fid))
            if lost:
                tracks[track.id] = None
            else:
                tracks[track.id] = agent_id, (fid, heading, extent, pos)

    info = []
    x, y, heading = [], [], []
    for agent_id, record in agents.items():
        if len(record) < 2: continue
        for x_, y_, yaw, agent_type, agent_extent, track_id, fid in record:
            x.append(x_)
            y.append(y_)
            heading.append(yaw)
            info.append((fid, agent_id, agent_type, agent_extent, track_id))
    
    if not info: return None

    # Size the raster to the agent extent plus MARGIN on each side.
    x_min = np.min(x)-MARGIN
    x_max = np.max(x)+MARGIN
    y_min = np.min(y)-MARGIN
    y_max = np.max(y)+MARGIN

    x_min = int(np.floor(x_min))
    x_max = int(np.ceil(x_max))
    y_min = int(np.floor(y_min))
    y_max = int(np.ceil(y_max))

    x_center = (x_min+x_max)*0.5
    y_center = (y_min+y_max)*0.5
    canvas_size = int(np.ceil((y_max-y_min)*MAP_SCALE)), int(np.ceil((x_max-x_min)*MAP_SCALE))
    height = 1.*canvas_size[0]/MAP_SCALE
    width = 1.*canvas_size[1]/MAP_SCALE

    # Homography: world (x, y) -> image (row, col), origin at the upper-left corner.
    H = np.array([
        [0., -MAP_SCALE, (y_center+0.5*height)*MAP_SCALE],
        [MAP_SCALE, 0., -(x_center-0.5*width)*MAP_SCALE],
        [0., 0., 1.]
    ])

    lanes = []
    road_lines = []
    road_edges = []
    crosswalks = []
    for m in scenario.map_features:
        feature_type = m.WhichOneof("feature_data")
        feature = getattr(m, feature_type)
        if feature_type == "lane":
            lanes.append([[p.x, p.y] for p in feature.polyline])
        elif feature_type == "road_line":
            road_lines.append([[p.x, p.y] for p in feature.polyline])
        elif feature_type == "road_edge":
            road_edges.append([[p.x, p.y] for p in feature.polyline])
        elif feature_type == "crosswalk":
            crosswalks.append([[p.x, p.y] for p in feature.polygon])
    
    map_features = (lanes, road_lines, road_edges, crosswalks)

    
    im_lanes = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_road_edges = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_road_lines = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    im_crosswalks = np.zeros(shape=(canvas_size[0], canvas_size[1]), dtype=np.uint8)
    if lanes:
        lanes = world2subpixel(lanes, H)
        cv2.polylines(im_lanes, lanes, False, 255, **CV2_SUB_VALUES)
    if road_edges:
        road_edges = world2subpixel(road_edges, H)
        cv2.polylines(im_road_edges, road_edges, False, 255, **CV2_SUB_VALUES)
    if road_lines:
        road_lines = world2subpixel(road_lines, H)
        cv2.polylines(im_road_lines, road_lines, False, 255, **CV2_SUB_VALUES)
    if crosswalks:
        crosswalks = world2subpixel(crosswalks, H)
        cv2.fillPoly(im_crosswalks, crosswalks, 255, **CV2_SUB_VALUES)

    # 3 channels: lanes+crosswalks, road edges, road lines (lanes OR'd into each)
    # scaled to [-1, 1].
    im_road_edges = np.maximum(im_road_edges, im_lanes).astype(np.float32)/255
    im_road_lines = np.maximum(im_road_lines, im_lanes).astype(np.float32)/255
    im_lanes = np.maximum(im_crosswalks, im_lanes).astype(np.float32)/255
    semantic_map = np.stack((im_lanes, im_road_edges, im_road_lines))
    semantic_map = (semantic_map*2 - 1).clip(-1, 1)  # [0, 1] -> [-1, 1]

    return name, semantic_map, H, x, y, heading, info


# Parse one TFRecord shard, split its scenarios across --split groups, and write
# per-group .txt / .info files and per-scenario map .pkl files. Agent ids and
# frame ids are offset per scenario so concatenated scenarios stay distinct.
def process(ith, dataset_file, path, map_path):
    dataset = tf.data.TFRecordDataset(dataset_file, compression_type='')
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    dataset = dataset.with_options(options)
    dataset_name = os.path.basename(dataset_file)

    items = []
    dims = []
    for data in dataset:
        res = parse(data) 
        if res is None: continue
        items.append(res)

        m = res[1]
        dims.append([m.shape[1], m.shape[2]])

    # Partition this shard's scenarios into --split near-equal groups; the
    # remainder is spread starting at offset `ith` to balance across shards.
    n_scenarios = len(items)
    if settings.split > 1:
        slices = [n_scenarios // settings.split] * settings.split
        rem = n_scenarios % settings.split
        for i in range(rem):
            slices[(ith+i)%len(slices)] += 1
        slices = np.cumsum(slices)
        fraction = [items[0 if i == 0 else slices[i-1]:e] for i, e in enumerate(slices)]
        subfolder = [str(p) for p in range(settings.split)]
        for p in subfolder:
            os.makedirs(os.path.join(map_path, p, dataset_name+"_"+p), exist_ok=True)
    else:
        subfolder = [""]
        fraction = [items]
        os.makedirs(os.path.join(map_path, dataset_name), exist_ok=True)
    

    for items, p in zip(fraction, subfolder):
        filepath = os.path.join(path, p, dataset_name)
        map_filepath = os.path.join(map_path, str(p), dataset_name)
        if p:
            filepath += "_" + p 
            map_filepath += "_" + p 

        content = []
        mapinfo = []
        # Offsets keep agent ids / frame ids unique when scenarios are concatenated.
        last_id, last_fid = -1, -FRAMESKIP
        for scenario_name, m, h, x, y, heading, info in items:
            agent_offset = last_id + 1
            fid_offset = last_fid + FRAMESKIP
            for x_, y_, heading_, (fid_, agent_id_, agent_type, agent_extent, track_id) in zip(x, y, heading, info):
                fid = fid_ + fid_offset
                agent_id = agent_id_ + agent_offset
                content.append("{:<5d} {:<6d} {:10.4f} {:10.4f} {:7.4f} {} {:.4f}/{:.4f}/{:.4f} {}".format(
                    fid, agent_id, x_, y_, heading_, agent_type, agent_extent[0], agent_extent[1], agent_extent[2], track_id
                ))
                last_id = max(last_id, agent_id)
                last_fid = max(last_fid, fid)

            mapinfo.append("{:<5d} {}".format(fid_offset, scenario_name))
            with open(os.path.join(map_filepath, "{}.pkl".format(scenario_name)), "wb") as f:
                pickle.dump((m, h), f)
        if not content: continue

        with open(filepath + ".txt", "w") as f:
            f.write("\n".join(content))
        with open(filepath + ".info", "w") as f:
            f.write("\n".join(mapinfo))

# Recursively collect every file path under `path` into `datasets`.
def load(path, datasets):
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            load(os.path.join(path, f), datasets)
    else:
        datasets.append(path)

if __name__ == "__main__":
    assert(os.path.exists(settings.data))
    datasets = []
    target_path = []
    map_path = []

    load(settings.data, datasets)
    for f in datasets:
        relpath = os.path.relpath(os.path.dirname(f), settings.data)
        target_path.append(os.path.join(settings.target_folder, relpath))
        map_path.append(os.path.join(settings.target_folder, "map", relpath))
    if settings.split > 1:
        for i in range(settings.split):
            for folder in set(target_path):
                os.makedirs(os.path.join(folder, str(i)), exist_ok=True)
            for folder in set(map_path):
                os.makedirs(os.path.join(folder, str(i)), exist_ok=True)
    else:
        for folder in set(target_path):
            os.makedirs(folder, exist_ok=True)
        for folder in set(map_path):
            os.makedirs(folder, exist_ok=True)

    print("Processing data...")
    args = []
    for i, (ds, path, mpath) in enumerate(zip(datasets, target_path, map_path)):
        args.append((i, ds, path, mpath) )

    done = 0
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=WORKERS) as p:
        futures = [p.submit(process, *arg) for arg in args]
        for fut in as_completed(futures):
            done += 1
            sys.stderr.write("\r  {}/{} {:.2%}".format(done, len(datasets), done/len(datasets)))
    