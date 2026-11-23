"""Build an independent repeated-vertical-endpoint structure layer.

No online occupancy is modified. The output NPZ is scored by
scripts/evaluate_scene_coverage.py --structure-layer, which reports baseline,
structure-only and union against the same reference/classification/threshold.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation, transform_xy
from go2_rescue.types import Pose


def build_structure(run: Path, config: Config) -> tuple[np.ndarray, np.ndarray, float, int]:
    with np.load(run / "map.npz") as grid:
        origin, resolution = grid["origin"], float(grid["resolution"])
        shape = grid["seen"].shape
    rows = {}
    with (run / "trajectory.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            rows[round(record["timestamp"], 5)] = record
    views = defaultdict(set)
    heights = defaultdict(list)
    used = 0
    for record in map(json.loads, (run / "recording/frames.jsonl").open()):
        row = rows.get(round(record["timestamp"], 5))
        if row is None or row["pose"]["quality"] < config.min_pose_quality:
            continue
        used += 1
        pose = Pose(**row["pose"])
        with np.load(run / "recording" / record["file"]) as packet:
            points = packet["points"] @ gravity_rotation(packet["quaternion"]).T
        distance = np.linalg.norm(points, axis=1)
        points = points[(distance > .40) & (distance < config.max_range)]
        xy = transform_xy(points, pose)
        view = tuple(np.floor(pose.xy() / .25).astype(int))
        ground = float(row["diagnostics"].get("ground_height", 0.))
        for cell, z in zip(np.floor((xy - origin) / resolution).astype(int), points[:, 2]):
            if (cell < 0).any() or (cell >= np.asarray(shape)).any():
                continue
            elevation = z + config.base_height + ground
            if elevation - ground < .24 or elevation - ground > .85:
                continue
            views[tuple(cell)].add(view)
            heights[tuple(cell)].append(elevation)
    stable = np.zeros(shape, dtype=bool)
    for cell, seen_views in views.items():
        if len(seen_views) >= 3 and np.ptp(heights[cell]) >= .09:
            stable[cell] = True
    return stable, origin, resolution, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run, args.output
    out.mkdir(parents=True, exist_ok=True)
    config = Config.load(run / "config.json")
    stable, origin, resolution, used = build_structure(run, config)
    np.savez_compressed(out / "structure_layer.npz", mask=stable, origin=origin, resolution=resolution)
    meta = {"schema": "go2-structure-layer/1", "frames_used": used,
            "stable_cells": int(stable.sum()), "origin": origin.tolist(),
            "resolution_m": resolution,
            "rule": "repeated vertical endpoints: >=3 view bins (0.25 m), height spread >=0.09 m",
            "note": "score with scripts/evaluate_scene_coverage.py --structure-layer"}
    (out / "structure_layer.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
