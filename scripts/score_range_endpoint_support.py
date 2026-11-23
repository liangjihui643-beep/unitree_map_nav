"""Check whether projected endpoints are recurring and agree with held-out XYZ."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation, transform_xy
from go2_rescue.lidar_obs import N_AZIMUTH, RangeOnlyEvidence, decode_lidar_obs
from go2_rescue.mapping import GridMap
from go2_rescue.recording import replay
from go2_rescue.types import Pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("record", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root = a.record
    cfg = Config.load(root / "config.json")
    grid = GridMap(cfg)
    layer = RangeOnlyEvidence(grid,cfg.lidar_elevation_center_deg,cfg.lidar_vertical_fov_deg,
                              height_floor_m=cfg.lidar_height_floor)
    with (root / "lidar_diagnostics.jsonl").open() as file:
        packets = {round(x["t"],3):x for x in map(json.loads,file)}
    observations = []
    votes = np.zeros_like(grid.logodds, dtype=np.int32)
    matches = total = corner_misses = invalid_contract = 0
    for frame in replay(root / "recording"):
        row = packets.get(round(frame.timestamp,3))
        if row is None or row.get("agent_obs") is None:
            continue
        pose = Pose(**row["pose"])
        if pose.quality < cfg.min_pose_quality:
            continue
        # Only the compressed packet and estimated pose enter the evidence.
        layer.unambiguous_hit.fill(False)
        layer.update(grid,row["agent_obs"],frame.quaternion,pose,cfg.min_range,cfg.max_range)
        mask = layer.unambiguous_hit
        votes += mask
        observations.append(int(mask.sum()))

        # Raw XYZ points are exclusively an evaluation oracle for planar bins.
        ids, meters = decode_lidar_obs(row["agent_obs"], cfg.min_range, cfg.max_range)
        local = frame.points - layer.origin_body
        az = (np.rint((np.arctan2(local[:,1], local[:,0])+np.pi) * N_AZIMUTH/(2*np.pi))
              .astype(int) % N_AZIMUTH)
        if sorted(az.tolist()) != ids.tolist() or not np.allclose(
                np.linalg.norm(local,axis=1)[np.argsort(az)], meters, atol=.002, rtol=0.):
            invalid_contract += 1
            continue
        oracle = {grid.cell(pt) for pt in transform_xy(frame.points @ gravity_rotation(frame.quaternion).T, pose)}
        for cell in map(tuple,np.argwhere(mask)):
            total += 1
            if cell in oracle:
                matches += 1
            elif any(abs(cell[0]-i)<=1 and abs(cell[1]-j)<=1 for i,j in oracle):
                corner_misses += 1
    if not observations:
        p.error("No synchronized Agent observations")
    result = {"frames":len(observations),"frames_with_at_least_12":int((np.asarray(observations)>=12).sum()),
              "median_endpoints_per_frame":float(np.median(observations)), "max_endpoints_per_frame":max(observations),
              "distinct_cells":int((votes>0).sum()),"cells_at_least_3_frames":int((votes>=3).sum()),
              "contract_mismatches":invalid_contract,
              "scored_frame_cells":total,"same_xy_grid":matches,"adjacent_xy_grid_only":corner_misses,
              "scope":"Sensor-plane consistency at the same frozen estimated poses; not ground obstacle accuracy"}
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
