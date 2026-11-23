"""Offline compare repeat vertical endpoints against the existing occupancy map."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation, transform_xy
from go2_rescue.types import Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    run, out = args.run, args.out
    out.mkdir(parents=True, exist_ok=True)
    config = Config.load(run / "config.json")
    with np.load(run / "map.npz") as grid:
        origin, resolution = grid["origin"], float(grid["resolution"])
        odds, floors, seen = grid["logodds"], grid["floor"], grid["seen"]
    rows = {round(d["timestamp"], 5): d for d in map(json.loads, (run / "trajectory.jsonl").open())}
    views = defaultdict(set)
    heights = defaultdict(list)
    for frame in map(json.loads, (run / "recording/frames.jsonl").open()):
        row = rows.get(round(frame["timestamp"], 5))
        if row is None or row["pose"]["quality"] < config.min_pose_quality:
            continue
        pose = Pose(**row["pose"])
        with np.load(run / "recording" / frame["file"]) as packet:
            points = packet["points"] @ gravity_rotation(packet["quaternion"]).T
        distance = np.linalg.norm(points, axis=1)
        points = points[(distance > .40) & (distance < config.max_range)]
        xy = transform_xy(points, pose)
        view = tuple(np.floor(pose.xy()/.25).astype(int))
        ground = row["diagnostics"]["ground_height"]
        for cell, z in zip(np.floor((xy-origin)/resolution).astype(int), points[:, 2]):
            if (cell < 0).any() or (cell >= odds.shape).any():
                continue
            ij = tuple(cell)
            elevation = z + config.base_height + ground
            floor = floors[ij] if np.isfinite(floors[ij]) else ground
            if elevation-floor < .24 or elevation-floor > .85:
                continue
            views[ij].add(view)
            heights[ij].append(elevation)
    stable = np.zeros_like(seen)
    for cell in views:
        if len(views[cell]) >= 3 and np.ptp(heights[cell]) >= .09:
            stable[cell] = True
    np.savez_compressed(out / "structural_mask.npz", mask=stable, origin=origin, resolution=resolution)
    if args.evidence_only:
        return
    with np.load(run / "navigation_prior.npz") as prior:
        truth = prior["reference_xy"]
    baseline = odds > .5
    after = baseline | stable
    tree = cKDTree(truth)
    result = {}
    for name, mask in (("baseline", baseline), ("repeat_vertical_union", after)):
        xy = origin + (np.argwhere(mask)+.5)*resolution
        valid = ((xy >= [-.9,-.9]) & (xy <= [11.3,9.3])).all(axis=1)
        xy = xy[valid]
        d = tree.query(xy)[0]
        reverse = cKDTree(xy).query(truth)[0]
        result[name] = {
            "count": len(xy), "reference_within_20cm": float((reverse <= .20).mean()),
            "distance_p90_m": float(np.quantile(d,.9)),
            "points_more_than_20cm_from_structure": int((d > .20).sum()),
        }
    result["stable_new_cells"] = int((stable & ~baseline).sum())
    (out / "structure_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    panels = []
    lo = np.maximum(0, np.floor((np.array([-.9,-.9])-origin)/resolution).astype(int))
    hi = np.minimum(odds.shape, np.ceil((np.array([11.3,9.3])-origin)/resolution).astype(int))
    for after_layer in (False, True):
        panel = np.full((*seen.shape, 3), 80, np.uint8)
        panel[seen] = (225,225,225)
        panel[baseline] = (22,22,22)
        if after_layer:
            panel[stable & ~baseline] = (0,100,230)
        region = panel[lo[0]:hi[0],lo[1]:hi[1]].transpose(1,0,2)[::-1]
        panels.append(cv2.resize(region,None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST))
    (out/"structure_comparison.png").write_bytes(cv2.imencode(".png",np.hstack(panels))[1].tobytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
