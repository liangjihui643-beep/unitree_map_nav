"""Compare legacy fixed-elevation mapping with candidate-based contract mapping.

Both maps consume the same contract observations and the same external poses.
Simulator geometry is used only to score the resulting maps. The truth score
uses a body-height horizontal probe so that under-roof floor is not counted as
a wall, while a wall inside the robot's height band is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import cv2
import numpy as np
from scipy.spatial import cKDTree

from go2_rescue.contract_map import ContractMap, LegacyMap
from go2_rescue.public_task import PublicTask


def truth_grid(model, origin, resolution, shape):
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    floor = np.zeros(shape, bool)
    occupied = np.zeros(shape, bool)
    body = np.zeros(shape, bool)
    height = np.full(shape, np.nan)
    group = np.array([1, 1, 1, 1, 1, 0], np.uint8)
    gid = np.array([-1], np.int32)
    directions = [np.array([np.cos(a), np.sin(a), 0.]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    for i, j in np.ndindex(shape):
        xy = origin + (np.array([i, j]) + .5) * resolution
        z = mujoco.mj_ray(model, state, np.array([xy[0], xy[1], 3.]),
                          np.array([0., 0., -1.]), group, True, -1, gid)
        if z < 0:
            continue
        height[i, j] = 3. - z
        g = int(gid[0])
        body_name = model.body(int(model.geom_bodyid[g])).name or ""
        name = model.geom(g).name or ""
        floor[i, j] = bool((body_name.startswith("base_ground") or body_name.startswith("ramp")
                            or name.startswith("ascent") or name.startswith("platform")
                            or name.startswith("descent")) and height[i, j] < 1.)
        occupied[i, j] = not floor[i, j]
        if floor[i, j]:
            for dz in (.15, .40):
                for direction in directions:
                    hit = mujoco.mj_ray(model, state, np.array([xy[0], xy[1], height[i, j] + dz]),
                                        direction, group, True, -1, gid)
                    if hit >= 0 and hit < .6:
                        body[i, j] = True
                        break
                if body[i, j]:
                    break
    return floor, occupied, body, height


def metrics(occupied, free, truth, origin, resolution, region):
    floor, truth_occupied, truth_body, height = truth
    result = {"occupied_cells": int((occupied & region).sum())}
    result["ramp_false_occupied"] = int((occupied & floor & (height > .25) & region).sum())
    result["flat_false_occupied"] = int((occupied & floor & (height <= .25) & region).sum())
    body_cells = origin + (np.argwhere(truth_body & region) + .5) * resolution
    occupied_cells = origin + (np.argwhere(occupied & region) + .5) * resolution
    if len(body_cells) and len(occupied_cells):
        result["body_obstacle_confirmed"] = int(
            (cKDTree(occupied_cells).query(body_cells)[0] <= .20 + 1e-8).sum())
    result["body_obstacle_reference"] = int((truth_body & region).sum())
    result["false_free"] = int((free & truth_body & region).sum())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    task = PublicTask.load(args.task)
    config = task.config()
    data = np.load(args.capture / "frames.npz")
    lidar, poses, quats = data["lidar"], data["pose"], data["quaternion"]
    legacy = LegacyMap(config)
    contract = ContractMap(config)
    timeline = []
    for index, (obs, pose, quat) in enumerate(zip(lidar, poses, quats)):
        legacy.update(obs, pose, quat)
        contract.update(obs, pose, quat, frame_index=index)
        if index % 10 == 0:
            timeline.append({"frame": index,
                             "ground_cells": int(np.isfinite(contract.ground).sum()),
                             "wall_cells": int(contract.wall.sum()),
                             "free_cells": int(contract.published_free().sum()),
                             "occupied_cells": int(contract.published_occupied().sum()),
                             "pending_cells": int((contract.pending_cells > 0).sum())})
    contract.finalize()
    model = mujoco.MjModel.from_binary_path(str(task.scene_path))
    origin, resolution = legacy.origin, legacy.resolution
    truth = truth_grid(model, origin, resolution, legacy.shape)
    lo = np.floor((poses.min(axis=0)[:2] - origin - 2.) / resolution).astype(int)
    hi = np.ceil((poses.max(axis=0)[:2] - origin + 2.) / resolution).astype(int)
    region = np.zeros(legacy.shape, bool)
    region[max(0, lo[0]):hi[0], max(0, lo[1]):hi[1]] = True
    legacy_free = legacy.free
    contract_free = contract.published_free()
    contract_occupied = contract.published_occupied()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "contract_maps.npz",
                        origin=origin, resolution=resolution,
                        legacy_occupied=legacy.occupied, legacy_free=legacy_free,
                        legacy_overhead=legacy.overhead,
                        contract_occupied=contract_occupied, contract_free=contract_free,
                        contract_ground=contract.ground, contract_near_body_seed=contract.near_body_seed,
                        contract_wall=contract.wall, contract_wall_votes=contract.wall_votes,
                        contract_overhead=contract.overhead,
                        contract_undetermined=contract.undetermined,
                        contract_pending_cells=contract.pending_cells,
                        contract_seen=contract.seen, truth_floor=truth[0],
                        truth_occupied=truth[1], truth_body=truth[2], truth_height=truth[3])
    observed = (contract.seen | contract.pending_cells > 0) & region
    pending = (contract.pending_cells > 0) & region
    pending_mask = ((contract.occupied > 0) | (contract.overhead > 0)
                    | (contract.pending_cells > 0))
    pending_xy = origin + (np.argwhere(pending_mask) + .5) * resolution
    truth_occ_xy = origin + (np.argwhere(truth[1]) + .5) * resolution
    pending_recall = (int((cKDTree(pending_xy).query(truth_occ_xy)[0] <= .20 + 1e-8).sum())
                      if len(pending_xy) and len(truth_occ_xy) else 0)
    summary = {
        "schema": "go2-contract-map-compare/1",
        "frames": int(len(lidar)),
        "region_cells": int(region.sum()),
        "legacy": metrics(legacy.occupied > 0, legacy_free, truth, origin, resolution, region),
        "contract": metrics(contract_occupied, contract_free, truth, origin, resolution, region),
        "contract_pending_structure": metrics((contract.occupied > 0) | contract.wall
                                              | (contract.overhead > 0)
                                              | (contract.pending_cells > 0),
                                              contract_free, truth, origin, resolution, region),
        "free_occupied_conflicts": contract.free_occupied_conflicts(),
        "contract_pending_cells": int((contract.pending_cells > 0).sum()),
        "contract_pending_ratio_observed": float(pending.sum() / max(1, observed.sum())),
        "pending_structure_diagnostic": {
            "topdown_recall": pending_recall,
            "reference": int(len(truth_occ_xy)),
            "note": "candidate coverage diagnostic; not a confirmed-occupancy score",
        },
        "legacy_occupied_cells": int((legacy.occupied > 0).sum()),
        "contract_occupied_cells": int(contract_occupied.sum()),
        "legacy_free_cells": int(legacy_free.sum()),
        "contract_free_cells": int(contract_free.sum()),
        "contract_ground_cells": int(np.isfinite(contract.ground).sum()),
        "contract_near_body_seed_cells": int(contract.near_body_seed.sum()),
        "contract_wall_cells": int(contract.wall.sum()),
        "contract_wall_segments": len(contract.wall_segments),
        "contract_wall_votes": int((contract.wall_votes > 0).sum()),
        "contract_traversable_cells": int((contract.published_free()).sum()),
        "contract_undetermined_cells": int(((contract.undetermined > 0)
                                            | (contract.pending_cells > 0)).sum()),
        "contract_seen_cells": int(contract.seen.sum()),
        "legacy_seen_cells": int(legacy.seen.sum()),
        "online_timeline_samples": len(timeline),
        "online_first_wall_frame": next((item["frame"] for item in timeline
                                         if item["wall_cells"] > 0), None),
        "online_first_ground_frame": next((item["frame"] for item in timeline
                                           if item["ground_cells"] > 0), None),
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "online_timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    active = np.argwhere(region)
    lo_i, lo_j = active.min(axis=0)
    hi_i, hi_j = active.max(axis=0) + 1
    def panel(is_legacy=False):
        image = np.full((hi_i - lo_i, hi_j - lo_j, 3), 120, np.uint8)
        if is_legacy:
            image[legacy_free[lo_i:hi_i, lo_j:hi_j]] = (170, 220, 170)
            image[(legacy.occupied > 0)[lo_i:hi_i, lo_j:hi_j]] = (25, 25, 25)
        else:
            image[np.isfinite(contract.ground)[lo_i:hi_i, lo_j:hi_j]] = (170, 220, 170)
            image[contract_free[lo_i:hi_i, lo_j:hi_j]] = (210, 240, 210)
            image[contract.pending_cells[lo_i:hi_i, lo_j:hi_j] > 0] = (0, 140, 255)
            image[contract.overhead[lo_i:hi_i, lo_j:hi_j] > 0] = (200, 120, 40)
            image[contract.wall[lo_i:hi_i, lo_j:hi_j]] = (25, 25, 25)
        return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "comparison.png"),
                np.hstack([panel(True), panel(False)]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
