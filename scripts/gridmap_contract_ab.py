"""Same-observation A/B for the production GridMap: old fallback vs pending ground.

The baseline subclass restores the previous robot-foot height fallback only for
this experiment; the runtime GridMap has no new/old switch.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.mapping import GridMap
from go2_rescue.public_task import PublicTask
from go2_rescue.types import Pose
from contract_map_compare import metrics, truth_grid


def yaw_of(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


ENTRANCE = {"start": (10.55, 6.23), "end": (11.85, 6.23),
            "low": (10.45, 5.75), "high": (11.95, 6.75)}


def cell_reason(grid, cell):
    if not grid.inside(cell):
        return "outside_map"
    if not grid.seen[cell]:
        return "unobserved"
    if grid.logodds[cell] > .5:
        return "occupied"
    if not np.isfinite(grid.floor[cell]):
        return "no_ground_support"
    if grid.headroom[cell] < grid.c.body_clearance:
        return "insufficient_clearance"
    if grid.hazard_votes[cell] >= 2:
        return "hazard"
    return "steep_or_inflated"


def local_path(grid):
    traversable = grid.traversable()
    lo = np.floor((np.asarray(ENTRANCE["low"]) - grid.origin) / grid.c.resolution).astype(int)
    hi = np.ceil((np.asarray(ENTRANCE["high"]) - grid.origin) / grid.c.resolution).astype(int)
    start = grid.cell(ENTRANCE["start"])
    end = grid.cell(ENTRANCE["end"])
    info = {"start_cell": list(start), "end_cell": list(end),
            "start_traversable": bool(grid.inside(start) and traversable[start]),
            "end_traversable": bool(grid.inside(end) and traversable[end]),
            "start_reason": cell_reason(grid, start),
            "end_reason": cell_reason(grid, end),
            "box": [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]}
    if not (info["start_traversable"] and info["end_traversable"]):
        return info
    queue = deque([start])
    parent = {start: None}
    found = False
    while queue:
        cell = queue.popleft()
        if cell == end:
            found = True
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (cell[0] + di, cell[1] + dj)
            if not (lo[0] <= other[0] <= hi[0] and lo[1] <= other[1] <= hi[1]):
                continue
            if other in parent or not traversable[other]:
                continue
            parent[other] = cell
            queue.append(other)
    info["connected"] = bool(found)
    if found:
        length = 0
        cell = end
        while parent[cell] is not None:
            length += 1
            cell = parent[cell]
        info["path_cells"] = length
    return info


class BaselineGridMap(GridMap):
    def _local_ground(self, xy):
        # Previous behaviour: fall back to the robot-foot height estimate.
        return self.ground_height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    task = PublicTask.load(args.task)
    config = task.config()
    config.map_size = 30.
    data = np.load(args.capture / "frames.npz")
    lidar = data["lidar"]
    selected = data["selected_points"]
    poses = data["pose"]
    quats = data["quaternion"]
    baseline = BaselineGridMap(config)
    current = GridMap(config)
    for index, (obs, points, pose, quat) in enumerate(zip(lidar, selected, poses, quats)):
        points = points @ gravity_rotation(quat).T
        ranges = np.linalg.norm(points, axis=1)
        points = points[(ranges > .25) & (ranges < config.max_range)]
        pose_obj = Pose(x=float(pose[0]), y=float(pose[1]), yaw=yaw_of(quat))
        baseline.update(points, pose_obj, quat, index * .1)
        current.update(points, pose_obj, quat, index * .1)
    model = mujoco.MjModel.from_binary_path(str(task.scene_path))
    origin, resolution = baseline.origin, baseline.c.resolution
    cached = args.output / "gridmap_ab.npz"
    if cached.exists():
        with np.load(cached) as previous:
            truth = (previous["truth_floor"], previous["truth_occupied"],
                     previous["truth_body"], previous["truth_height"])
    else:
        truth = truth_grid(model, origin, resolution, baseline.logodds.shape)
    lo = np.floor((poses.min(axis=0)[:2] - origin - 2.) / resolution).astype(int)
    hi = np.ceil((poses.max(axis=0)[:2] - origin + 2.) / resolution).astype(int)
    region = np.zeros(baseline.logodds.shape, bool)
    region[max(0, lo[0]):hi[0], max(0, lo[1]):hi[1]] = True
    baseline_free = baseline.seen & (baseline.logodds < 0)
    current_free = current.seen & (current.logodds < 0)
    baseline_occupied = (baseline.logodds > .5) | (baseline.structure_votes >= 2)
    current_occupied = (current.logodds > .5) | (current.structure_votes >= 2)
    baseline_entry = local_path(baseline)
    current_entry = local_path(current)
    summary = {
        "frames": int(len(lidar)),
        "baseline": metrics(baseline_occupied, baseline_free, truth, origin, resolution, region),
        "current": metrics(current_occupied, current_free, truth, origin, resolution, region),
        "baseline_occupied_cells": int(baseline_occupied.sum()),
        "current_occupied_cells": int(current_occupied.sum()),
        "baseline_free_cells": int(baseline_free.sum()),
        "current_free_cells": int(current_free.sum()),
        "current_pending_endpoints": int(len(current.pending_endpoints)),
        "baseline_pending_endpoints": int(len(baseline.pending_endpoints)),
        "current_structure_cells": int((current.structure_votes >= 2).sum()),
        "baseline_structure_cells": int((baseline.structure_votes >= 2).sum()),
        "entrance": {"id": "tunnel3_east", "start": list(ENTRANCE["start"]),
                     "end": list(ENTRANCE["end"]),
                     "baseline": baseline_entry, "current": current_entry},
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    baseline_navigation = baseline.logodds > .5
    current_navigation = current.logodds > .5
    np.savez_compressed(out / "gridmap_ab.npz", origin=origin, resolution=resolution,
                        baseline_occupied=baseline_occupied, baseline_free=baseline_free,
                        current_occupied=current_occupied, current_free=current_free,
                        baseline_navigation_occupied=baseline_navigation,
                        current_navigation_occupied=current_navigation,
                        baseline_structure=baseline.structure_votes >= 2,
                        current_structure=current.structure_votes >= 2,
                        baseline_traversable=baseline.traversable(),
                        current_traversable=current.traversable(),
                        current_pending=np.asarray(len(current.pending_endpoints)),
                        truth_floor=truth[0], truth_occupied=truth[1],
                        truth_body=truth[2], truth_height=truth[3])
    (out / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
