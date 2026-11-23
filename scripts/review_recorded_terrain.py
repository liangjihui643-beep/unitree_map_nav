"""Rebuild a readable terrain layer and comparison figure from a recorded run."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import transform_xy
from go2_rescue.known_map_navigation import KnownMapNavigator
from go2_rescue.mapping import GridMap, spatial_frame_support
from go2_rescue.terrain_layer import collect_cell_heights, terrain_snapshot, write_terrain_snapshot
from go2_rescue.types import Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config.load(args.run / "config.json")
    config.navigation_prior = str((args.run / "navigation_prior.npz").resolve())
    navigator = KnownMapNavigator(config)
    grid = GridMap(config)
    with np.load(args.run / "map.npz") as archive:
        grid.origin, grid.seen, grid.logodds = (archive[k] for k in ("origin", "seen", "logodds"))
    with (args.run / "trajectory.jsonl").open() as stream:
        poses = {round(row["timestamp"], 5): Pose(**row["pose"]) for row in map(json.loads, stream)}
    visual = Counter()
    support_frames = []
    stamp = 0.
    with (args.run / "rubble_observations.jsonl").open() as stream:
        for row in map(json.loads, stream):
            stamp = row["t"]
            pose = poses.get(round(stamp, 5))
            visual_points, lidar_points, avoidance_points = [], [], []
            for region in row["regions"]:
                visual_points.extend(region["map_xy"])
                lidar_points.extend(region["lidar_map_xy"])
                if len(region["map_xy"]) >= 3:
                    avoidance_points.extend(region["map_xy"])
            visual.update({grid.cell(xy) for xy in visual_points})
            support_frames.append({grid.cell(xy) for xy in lidar_points})
            if pose is not None and pose.quality >= config.min_pose_quality and avoidance_points:
                xy = np.asarray(avoidance_points)
                navigator.observe_visual_rubble(xy[np.linalg.norm(xy - pose.xy(), axis=1) < 2.], pose)
    observations = json.loads((args.run / "terrain_heights.json").read_text())
    heights = collect_cell_heights(grid, (
        (np.asarray(row["points_level"]), row["height_m"], Pose(**row["pose"]))
        for row in observations), transform_xy)
    layer = terrain_snapshot(grid, visual, spatial_frame_support(support_frames), heights, stamp, navigator.grid)
    layer["provenance"] = "offline reconstruction from recorded estimated poses; no true scene labels used"
    write_terrain_snapshot(args.output / "terrain_layer.json", layer)

    base = np.full((*grid.seen.shape, 3), 78, np.uint8)
    base[grid.seen] = (185, 185, 185)
    base[grid.seen & (grid.logodds < -.5)] = (232, 232, 232)
    base[grid.logodds > .5] = (22, 22, 22)
    overlay = base.copy()
    avoid = overlay.copy()
    nav_grid = navigator.grid
    for index in np.argwhere(nav_grid.visual_votes >= 2):
        cell = grid.cell(nav_grid.world(index))
        if grid.inside(cell):
            avoid[cell] = (184, 68, 146)
    for item in layer["cells"]:
        cell = tuple(item["ij"])
        overlay[cell] = (53, 166, 35) if item["height_observed_min_max_m"] else (0, 150, 250)
        avoid[cell] = overlay[cell]
    # Focus the comparison on terrain that was actually detected.
    candidates = np.asarray([item["ij"] for item in layer["cells"]], dtype=int).reshape(-1, 2)
    lo, hi = (np.maximum(0, candidates.min(0)-6), np.minimum(grid.n, candidates.max(0)+7)) if len(candidates) else (np.zeros(2, int), np.full(2, grid.n))
    crop = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
    panels = []
    for title, raster in [
        ("LiDAR occupancy only", base),
        ("Terrain map: amber RGB / green measured height", overlay),
        ("Planner: purple visual avoidance", avoid),
    ]:
        patch = raster[crop].transpose(1, 0, 2)[::-1]
        scale = min(460 / patch.shape[1], 400 / patch.shape[0])
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        panel = np.full((460, 480, 3), 34, np.uint8)
        panel[40:40+patch.shape[0], 10:10+patch.shape[1]] = patch
        cv2.putText(panel, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .49, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    (args.output / "terrain_map_comparison.png").write_bytes(cv2.imencode(".png", np.hstack(panels))[1].tobytes())
    print(json.dumps({
        "terrain_cells": len(layer["cells"]),
        "over_lidar_ray_free": sum(cell["lidar_ray_free"] for cell in layer["cells"]),
        "measured_height_cells": sum(cell["height_observed_min_max_m"] is not None for cell in layer["cells"]),
        "reconstructed_visual_avoid_cells": int((nav_grid.visual_votes >= 2).sum()),
    }))


if __name__ == "__main__":
    main()
