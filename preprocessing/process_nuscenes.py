import os, sys
import pickle
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from pyquaternion import Quaternion

# nuScenes -> ContextVAE on-disk format. For each prediction-challenge case
# writes <scene>/<token>.txt (rows: fid aid x y heading group) and .info
# (first_frame map_name); one rasterized semantic map per location is written to
# map/<location>.pkl as (semantic_map [3,H,W] in [-1,1], H 3x3).
#
# Environment:
#   conda create -n nuscenes && conda activate nuscenes
#   conda install python=3.9
#   pip install nuscenes-devkit==1.1.9 pyquaternion==0.9.9 pandas==1.3.4

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=str)
parser.add_argument("target_folder", type=str)
parser.add_argument("--map_scale", type=int, default=1)
parser.add_argument("--workers", type=int, default=None)
settings = parser.parse_args()

VERSION = "v1.0-trainval"
MAP_SCALE = settings.map_scale  # raster pixels per meter
DATA_ROOT = settings.data_root
WORKERS = min(20, multiprocessing.cpu_count()//2) if settings.workers is None else settings.workers
nuscenes = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)

# Rasterize a location's HD map over the agents' bounding box. Returns
# (semantic_map [3,H,W] in [-1,1], H) with H mapping world (x, y) -> image (row, col).
def get_semantic_map(map_name, x_min, x_max, y_min, y_max, map_scale):
    scene_map = NuScenesMap(dataroot=DATA_ROOT, map_name=map_name)

    # Pad the extent by 300 m so agent-centered map crops stay in bounds at eval.
    x_min = x_min - 300
    x_max = x_max + 300
    y_min = y_min - 300
    y_max = y_max + 300

    x_min = int(np.floor(x_min))
    x_max = int(np.ceil(x_max))
    y_min = int(np.floor(y_min))
    y_max = int(np.ceil(y_max))

    x_center = (x_min+x_max)*0.5
    y_center = (y_min+y_max)*0.5
    canvas_size = int(np.ceil((y_max-y_min)*map_scale)), int(np.ceil((x_max-x_min)*map_scale))
    height = 1.*canvas_size[0]/map_scale
    width = 1.*canvas_size[1]/map_scale
    patch_box = x_center, y_center, height, width
    mask = scene_map.get_map_mask(patch_box, 0., ["lane", "road_segment", "drivable_area", "road_divider", "lane_divider", "ped_crossing"], canvas_size)
    # 3 channels: drivable area + crossings, road divider, lane divider.
    semantic_map = np.stack((np.max(mask[:3], axis=0)*0.75+mask[5]*0.25, mask[3], mask[4]))
    semantic_map = (semantic_map*2-1).clip(-1, 1)  # [0, 1] -> [-1, 1]

    # Homography: world (x, y) -> image (row, col); origin at the upper-left corner.
    H = np.array([
        [0., -map_scale, (y_center+0.5*height)*map_scale],
        [map_scale, 0., -(x_center-0.5*width)*map_scale],
        [0., 0., 1.]
    ])
    # nuScenes counts rows bottom-to-top; flip rows to match image convention.
    semantic_map = semantic_map[:,::-1,:]

    return semantic_map, H

# Build per-case .txt / .info files for one scene, tracking the agents' world
# bounding box so the caller can rasterize a tight map. Each case spans up to
# 6 history + 12 future frames (2 s + 6 s at nuScenes' 2 Hz annotation rate).
def generate(path, map_name, tokens):

    # Track the world-space bounding box of all agents in this scene.
    x_max = -9999999
    y_max = -9999999
    x_min = 9999999
    y_min = 9999999

    all_samples = [sample_token for _, sample_token in tokens]
    all_tokens = [inst_token+"_"+sample_token for inst_token, sample_token in tokens]
    processed_sample = set()
    processed_token = set()
    for sample_token in all_samples:
        if sample_token in processed_sample: continue

        present = nuscenes.get("sample", sample_token)
        samples, sample = [], present

        # Walk backward for history, stopping once 6 frames pass with no case token.
        hist = 0
        while sample["prev"] and hist < 6:
            sample = nuscenes.get("sample", sample["prev"])
            samples.append(sample)
            if sample["token"] in all_samples:
                hist = 0
            else:
                hist += 1
            
        samples = list(reversed(samples))
        samples.append(present)
        sample = present
        # Walk forward for future, stopping once 12 frames pass with no case token.
        fut = 0
        while sample["next"] and fut < 12:
            sample = nuscenes.get("sample", sample["next"])
            samples.append(sample)
            if sample["token"] in all_samples:
                fut = 0
            else:
                fut += 1

        interested = dict()
        records = dict()
        data = dict()
        for fid, sample in enumerate(samples):
            sample_token = sample["token"]
            processed_sample.add(sample_token)
            for ann_token in sample['anns']:
                ann = nuscenes.get('sample_annotation', ann_token)
                category = ann['category_name']
                # Skip annotations with no attributes unless they are vehicles
                # (drops static/background objects).
                if not len(ann["attribute_tokens"]) and not "vehicle" in category:
                    continue
                if "cycle" in category:
                    agent_type = "CYCLE"
                elif "vehicle" in category:
                    agent_type = "VEHICLE"
                elif "human" in category:
                    agent_type = "PEDESTRIAN"
                else:
                    continue
                inst = ann["instance_token"]
                x = ann["translation"][0]
                y = ann["translation"][1]
                heading = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
                if inst not in records: records[inst] = []
                records[inst].append((fid, x, y, heading, agent_type))
                
                data[len(data)] = {
                    'frame_id': fid,
                    'node_id': inst
                }
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                token = inst + "_" + sample_token
                if token in all_tokens:
                    if token not in processed_token:
                        processed_token.add(token)
                        if inst in interested:
                            interested[inst].append((fid, token))
                        else:
                            interested[inst] = [(fid, token)]
            
            camera_data = nuscenes.get("sample_data", sample["data"]["CAM_FRONT"])
            ann = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
            inst = "EGO"
            agent_type = "EGO"
            x = ann["translation"][0]
            y = ann["translation"][1]
            heading = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
            if inst not in records: records[inst] = []
            records[inst].append((fid, x, y, heading, agent_type))
        
            data[len(data)] = {
                'frame_id': fid,
                'node_id': inst
            }
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            x_min = min(x_min, x)
            y_min = min(y_min, y)

        data = pd.DataFrame.from_dict(data, "index")
        data.sort_values('frame_id', inplace=True)
        # Keep agents seen in more than one frame (need at least a velocity step).
        agents = [inst for inst in pd.unique(data["node_id"]) if len(records[inst]) > 1]

        # One case per target agent: a 4-history + 12-future window around its
        # prediction frame. The target's group is tagged "/CHALLENGE".
        for target_inst, val in interested.items():
            for target_fid, token in val:
                content = []
                first_frame = 100000
                for aid, inst in enumerate(agents, 1):
                    cat = records[inst][0][-1]
                    for fid, px, py, yaw, _ in records[inst]:
                        if fid < target_fid-4: continue
                        if fid > target_fid+12: break
                        first_frame = min(first_frame, fid)
                        group = cat
                        if inst == target_inst:
                            group = group + "/CHALLENGE"
                        content.append("{} {} {} {} {} {}".format(fid, aid, px, py, yaw, group))

                with open(os.path.join(path, "{}.txt".format(token)), "w") as f:
                    f.write("\n".join(content))
                with open(os.path.join(path, "{}.info".format(token)), "w") as f:
                    f.write("{} {}".format(first_frame, map_name))
    
    return map_name, (x_min, x_max, y_min, y_max)


if __name__ == "__main__":
    splits = create_splits_scenes()
    datasets = ["train", "val"]
    path = {subset: os.path.join(settings.target_folder, subset) for subset in datasets}
    map_path = os.path.join(settings.target_folder, "map")
    os.makedirs(map_path, exist_ok=True)
    for p in path: os.makedirs(p, exist_ok=True)

    print("Reading data...")
    args = []
    for subset in datasets:
        case_tokens = get_prediction_challenge_split(subset, dataroot=DATA_ROOT)
        task = {}
        for token in case_tokens:
            instance_token, sample_token = token.split("_")
            scene_token = nuscenes.get("sample", sample_token)["scene_token"]
            if scene_token not in task:
                scene = nuscenes.get("scene", scene_token)
                map_name = nuscenes.get('log', scene["log_token"])["location"]
                p = os.path.join(path[subset], scene["name"])
                os.makedirs(p, exist_ok=True)
                task[scene_token] = [p, map_name, []]
            task[scene_token][-1].append((instance_token, sample_token))
        for arg in task.values():
            args.append(arg)

    M = dict()
    done = 0
    sys.stderr.write("\rPreparing... {}/{} {:.2%}".format(done, len(args), done/len(args)))
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork"), max_workers=WORKERS) as p:
        futures = [p.submit(generate, *_) for _ in args]
        for fut in as_completed(futures):
            done += 1
            sys.stderr.write("\rProcessing...{}/{} {:.2%}".format(done, len(args), done/len(args)))
        for fut in futures:
            m, d = fut.result()
            if m in M:
                corner = M[m]
                M[m] = (min(d[0], corner[0]), max(d[1], corner[1]), min(d[2], corner[2]), max(d[3], corner[3]))
            else:
                M[m] = d
    print()
    # Rasterize one map per location, sized to cover every agent seen there.
    print("Generating map...")
    for map_name, (x_min, x_max, y_min, y_max) in M.items():
        scene_map = NuScenesMap(dataroot=DATA_ROOT, map_name=map_name)
        semantic_map, H = get_semantic_map(map_name, x_min, x_max, y_min, y_max, MAP_SCALE)
        os.makedirs(map_path, exist_ok=True)
        with open(os.path.join(map_path, "{}.pkl".format(map_name)), "wb") as f:
            pickle.dump((semantic_map, H), f)

