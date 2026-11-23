"""One short entrance segment: sample LiDAR while moving across a roofed entrance
and keep reading the same live GridMap's traversable() mask every frame.

Poses are scripted (external estimate) because this is a map-interface check,
not a controller or localization acceptance. No full-course rerun.
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
from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.public_task import PublicTask
from go2_rescue.types import Pose

JOINT_NAMES = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
               "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", type=float, nargs=2, default=[11.40, 7.80])
    ap.add_argument("--end", type=float, nargs=2, default=[11.40, 6.00])
    ap.add_argument("--steps", type=int, default=36)
    args = ap.parse_args()
    task = PublicTask.load(args.task)
    config = task.config()
    config.map_size = 30.
    model = mujoco.MjModel.from_binary_path(str(task.scene_path))
    data = mujoco.MjData(model)
    joint_q = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    home = model.key_qpos[0][joint_q].copy() if model.nkey else model.qpos0[joint_q].copy()
    sensors = SimSensors(model, data, config, seed=3101)
    grid = GridMap(config)
    start = np.asarray(args.start, float)
    end = np.asarray(args.end, float)
    timeline = []
    maps = []
    try:
        for index, fraction in enumerate(np.linspace(0., 1., args.steps)):
            xyz = start + (end - start) * fraction
            yaw = float(np.arctan2(*(end - start)[::-1]))
            quat = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)
            data.qpos[:3] = [xyz[0], xyz[1], .27]
            data.qpos[3:7] = quat
            data.qpos[joint_q] = home
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            points = sensors.lidar()
            points = points @ gravity_rotation(quat).T
            ranges = np.linalg.norm(points, axis=1)
            points = points[(ranges > .25) & (ranges < config.max_range)]
            pose = Pose(x=float(xyz[0]), y=float(xyz[1]), yaw=yaw)
            grid.update(points, pose, quat, index * .1)
            if index % 5 == 0 or index == args.steps - 1:
                trav = grid.traversable()
                timeline.append({"frame": index, "t": index * .1,
                                 "traversable_cells": int(trav.sum())})
                maps.append(trav.copy())
    finally:
        sensors.close()

    trav = grid.traversable()
    lo = np.floor((np.array([11.1, 5.8]) - grid.origin) / grid.c.resolution).astype(int)
    hi = np.ceil((np.array([11.7, 7.9]) - grid.origin) / grid.c.resolution).astype(int)
    start_cell = grid.cell(start)
    end_cell = grid.cell(end)
    def reason(cell):
        if not grid.inside(cell):
            return "outside_map"
        if trav[cell]:
            return "traversable"
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
    queue = deque([start_cell])
    parent = {start_cell: None}
    found = False
    while queue:
        cell = queue.popleft()
        if cell == end_cell:
            found = True
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (cell[0] + di, cell[1] + dj)
            if not (lo[0] <= other[0] <= hi[0] and lo[1] <= other[1] <= hi[1]):
                continue
            if other in parent or not trav[other]:
                continue
            parent[other] = cell
            queue.append(other)
    result = {
        "frames": args.steps,
        "start": start.tolist(), "end": end.tolist(),
        "box": [lo.tolist(), hi.tolist()],
        "start_cell": list(start_cell), "end_cell": list(end_cell),
        "start_traversable": bool(grid.inside(start_cell) and trav[start_cell]),
        "end_traversable": bool(grid.inside(end_cell) and trav[end_cell]),
        "start_reason": reason(start_cell), "end_reason": reason(end_cell),
        "connected": bool(found),
        "traversable_cells": int(trav.sum()),
        "timeline": timeline,
        "same_grid_updated": len(timeline) > 1,
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "short_segment.npz", origin=grid.origin,
                        resolution=grid.c.resolution, traversable=trav,
                        logodds=grid.logodds, floor=grid.floor,
                        headroom=grid.headroom, hazard_votes=grid.hazard_votes,
                        structure=grid.structure_votes)
    (out / "short_segment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
