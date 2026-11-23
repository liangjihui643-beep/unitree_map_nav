"""Estimate bounded rough-surface interiors between multi-view observations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


def refine(layer, grid_odds, views):
    resolution = layer["resolution_m"]
    origin = np.asarray(layer["origin_xy_m"])
    shape = tuple(layer["shape"])
    terrain = np.zeros(shape, dtype=bool)
    for item in layer["cells"]:
        if not item.get("inferred_footprint"):
            terrain[tuple(item["ij"])] = True
    if not terrain.any():
        return []
    distance = distance_transform_edt(~terrain)*resolution
    # Nearby occupied laser cells anchor the sides of a patch. The mask
    # never uses a public prior or links remote regions across a walkway.
    seeds = terrain | ((grid_odds > 1.5) & (distance <= .15))
    components, n = label(binary_dilation(seeds,iterations=1))
    filled = np.zeros(shape, dtype=bool)
    for group in range(1,n+1):
        points = np.argwhere((components==group)&seeds)
        visual_points = np.argwhere((components==group)&terrain)
        if len(visual_points)<7 or np.ptp(points,axis=0).max()*resolution>2.4:
            continue
        distinct = set().union(*(views.get(tuple(cell),set()) for cell in visual_points))
        if len(distinct)<2:
            continue
        hull = cv2.convexHull(points[:,::-1].astype(np.int32))
        patch = np.zeros(shape,np.uint8)
        cv2.fillConvexPoly(patch,hull,1)
        filled |= patch.astype(bool)&(distance<=.25)&~terrain
    result = []
    for index in np.argwhere(filled):
        result.append({
            "ij":index.tolist(),
            "xy_m":(origin+(index+.5)*resolution).tolist(),
            "rgb_projection_frames":0,
            "lidar_neighbourhood_frames":0,
            "height_observed_min_max_m":None,
            "lidar_ray_free":bool(grid_odds[tuple(index)] < -.5),
            "inferred_footprint":True,
        })
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run,out=args.run,args.out
    layer=json.loads((run/"terrain_layer.json").read_text())
    if "inference_cpu_wall_ms" in layer:
        # Keep the online snapshot and its visual height evidence intact.
        out.mkdir(parents=True,exist_ok=True)
        (out/"terrain_layer.json").write_text(json.dumps(layer,allow_nan=False),encoding="utf-8")
        return
    origin=np.asarray(layer["origin_xy_m"])
    poses={round(row["timestamp"],6): row["pose"]
           for row in map(json.loads,(run/"trajectory.jsonl").open())}
    views=defaultdict(set)
    for row in map(json.loads,(run/"rubble_observations.jsonl").open()):
        pose=poses.get(round(row["t"],6))
        if pose is None:
            continue
        view=tuple(np.floor(np.array([pose["x"],pose["y"]])/.25).astype(int))
        for region in row["regions"]:
            for xy in region["map_xy"]+region["lidar_map_xy"]:
                views[tuple(np.floor((np.asarray(xy)-origin)/layer["resolution_m"]).astype(int))].add(view)
    with np.load(run/"map.npz",allow_pickle=False) as grid:
        new=refine(layer,grid["logodds"],views)
    observed=[item for item in layer["cells"] if not item.get("inferred_footprint")]
    layer["cells"]=observed+new
    layer["footprint_refinement"]={"direct_cells":len(observed),"inferred_cells":len(new),
                                    "provenance":"multi-view bounded interpolation, no measured height"}
    out.mkdir(parents=True,exist_ok=True)
    (out/"terrain_layer.json").write_text(json.dumps(layer,allow_nan=False),encoding="utf-8")
    print(json.dumps(layer["footprint_refinement"],indent=2))


if __name__=="__main__":
    main()
