"""Ablate a sensor-only vertical surface filter on previously recorded maps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from go2_rescue.geometry import gravity_rotation, transform_xy
from go2_rescue.types import Pose


def collect(run, origin, resolution):
    trace = {round(d["timestamp"],5):d for d in map(json.loads,(run/"trajectory.jsonl").open())}
    config = json.loads((run/"config.json").read_text())
    voxels = defaultdict(set)
    for item in map(json.loads,(run/"recording/frames.jsonl").open()):
        row = trace.get(round(item["timestamp"],5))
        if row is None or row["pose"]["quality"] < config["min_pose_quality"]:
            continue
        pose = Pose(**row["pose"])
        with np.load(run/"recording"/item["file"],allow_pickle=False) as packet:
            points = packet["points"]@gravity_rotation(packet["quaternion"]).T
        ranges = np.linalg.norm(points,axis=1)
        points = points[(ranges>.35)&(ranges<config["max_range"])]
        xy = transform_xy(points,pose)
        height_offset = row["diagnostics"]["ground_height"]+config["base_height"]
        cells = np.floor((xy-origin)/resolution).astype(int)
        z = np.floor((points[:,2]+height_offset)/.07).astype(int)
        view = tuple(np.floor(pose.xy()/.25).astype(int))
        for cell, level in zip(cells,z):
            voxels[(*cell,level)].add(view)
    return voxels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("runs",type=Path,nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.base/"map.npz") as base:
        arrays = {key:base[key] for key in base.files}
    origin,resolution,odds = arrays["origin"],float(arrays["resolution"]),arrays["logodds"]
    accumulated = defaultdict(set)
    for run in args.runs:
        for voxel,views in collect(run,origin,resolution).items():
            accumulated[voxel].update(views)
    repeat = [key for key,views in accumulated.items() if len(views)>=2]
    samples = np.asarray(repeat,dtype=int).reshape(-1,3)
    xyz = np.column_stack((origin+(samples[:,:2]+.5)*resolution,(samples[:,2]+.5)*.07))
    horizontal = xyz[:,:2]
    tree = cKDTree(horizontal)
    stable = np.zeros_like(odds,dtype=bool)
    for cell in np.argwhere(odds>.5):
        ij = tuple(cell)
        neighbours = np.asarray(tree.query_ball_point(origin+(cell+.5)*resolution,.225))
        if len(neighbours)<6:
            continue
        point = xyz[neighbours]
        mean = np.median(point,axis=0)
        close = abs(point[:,2]-mean[2])<.28
        if close.sum()<6:
            continue
        point=point[close]
        values,axes=np.linalg.eigh(np.cov(point.T))
        if values[1]>.0025 and values[0]<.015 and abs(axes[2,0])<.62 and np.ptp(point[:,2])>=.12:
            stable[ij]=True
    # Preserve all original occupancy as data and write only a disposable
    # scoring variant. No truth reference enters the classification above.
    flags = [(args.out/"vertical_only",stable),
             (args.out/"vertical_or_strong",stable|(odds>2.5))]
    for output,mask in flags:
        output.mkdir(exist_ok=True)
        for name in ("summary.json","evaluation_truth.json","lidar_diagnostics.jsonl","terrain_layer.json"):
            (output/name).write_bytes((args.base/name).read_bytes())
        np.savez_compressed(output/"map.npz",**dict(arrays,logodds=np.where(mask,odds,np.minimum(odds,.5))))
        print(output, int(mask.sum()))


if __name__ == "__main__":
    main()
