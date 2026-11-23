"""Replay two elevation envelopes on identical logged Agent packets."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.lidar_obs import RangeOnlyEvidence
from go2_rescue.mapping import GridMap
from go2_rescue.recording import replay
from go2_rescue.types import Pose


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    cfg=Config.load(a.run/"config.json")
    grid=GridMap(cfg)
    with (a.run/"lidar_diagnostics.jsonl").open() as f:
        sensor={round(x["t"],3): x for x in map(json.loads,f)}
    envelopes={k:RangeOnlyEvidence(grid,cfg.lidar_elevation_center_deg,cfg.lidar_vertical_fov_deg,
                                   height_floor_m=hf)
               for k,hf in (("all_elevations",None),("contract_filtered",cfg.lidar_height_floor))}
    count=0
    for f in replay(a.run/"recording"):
        row=sensor.get(round(f.timestamp,3))
        if row is None or row.get("agent_obs") is None:
            continue
        pose=Pose(**row["pose"])
        if pose.quality<cfg.min_pose_quality:
            continue
        for estimator in envelopes.values():
            estimator.update(grid,row["agent_obs"],f.quaternion,pose,cfg.min_range,cfg.max_range)
        count+=1
    if not count:
        p.error("Run lacks logged 144-value Agent observations and valid poses")
    result={"frames":count,"scope":"recorded range plus mask, IMU and frozen estimated pose; no independent ground"}
    for name,e in envelopes.items():
        result[name]={"possible_hit_cells":int(e.possible_hit.sum()),
                      "common_ray_cells":int(e.ray_prefix.sum()),
                      "unambiguous_hit_cells":int(e.unambiguous_hit.sum()),
                      "ground_band_cells":int(e.ground_band_clear.sum())}
    for k in ("possible_hit","ray_prefix"):
        # Union of smaller feasible endpoint sets and intersection of fewer
        # rays: tightening the contract narrows hits and can extend agreement.
        key,ref=(("contract_filtered","all_elevations") if k=="possible_hit"
                 else ("all_elevations","contract_filtered"))
        if np.any(getattr(envelopes[key],k) & ~getattr(envelopes[ref],k)):
            raise RuntimeError(f"Unexpected envelope relationship: {k}")
    if grid.logodds.any() or grid.seen.any():
        raise RuntimeError("Map was mutated by evidence-only processing")
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
