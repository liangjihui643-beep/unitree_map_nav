"""Score compressed range shadow odometry and velocity on identical frames."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import rot2
from go2_rescue.lidar_obs import encode_lidar_obs
from go2_rescue.localization import Localization
from go2_rescue.recording import replay
from go2_rescue.sparse_range_odometry import SparseRangeOdometry


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--start",type=float,default=0.)
    p.add_argument("--end",type=float,default=1e9)
    a=p.parse_args()
    r=a.run
    cfg=Config.load(r/"config.json")
    estimator=SparseRangeOdometry(cfg)
    data={round(x["t"],3):x for x in map(json.loads,(r/"lidar_diagnostics.jsonl").open())}
    truth={round(x["t"],3):np.asarray(x["base_world_xy"]) for x in json.loads((r/"evaluation_truth.json").read_text())}
    zero=rot=None
    last=None
    velocity=np.zeros(2)
    errors=[]
    counts=Counter()
    for f in replay(r/"recording"):
        t=round(f.timestamp,3)
        if t>a.end:
            break
        if t<a.start or t not in truth or t not in data:
            continue
        obs=data[t].get("agent_obs")
        if obs is None:
            # Old logs have only the already selected return. Re-encode its
            # scalar range and azimuth; never send elevation or XYZ to odometry.
            offset=f.points-[0.,0.,.10]
            az=np.rint((np.arctan2(offset[:,1],offset[:,0])+np.pi)*72/(2*np.pi)).astype(int)%72
            depth=np.full(72,np.inf)
            np.minimum.at(depth,az,np.linalg.norm(offset,axis=1))
            mask=np.isfinite(depth)
            obs=encode_lidar_obs(mask,np.clip(depth[mask],cfg.min_range,cfg.max_range),cfg.max_range)
            counts["reencoded"]+=1
        pose=estimator.update(obs,f.timestamp,f.quaternion,f.velocity)
        if zero is None:
            zero=truth[t]
            rot=rot2(Localization._imu_yaw(f.quaternion))
        if last is not None and 0 < f.timestamp-last <= .5:
            velocity += rot2(pose.yaw) @ f.velocity.array()[:2]*(f.timestamp-last)
        last=f.timestamp
        true=(truth[t]-zero) @ rot
        errors.append([np.linalg.norm(velocity-true),np.linalg.norm(pose.xy()-true)])
        counts["corrections"]+=bool(estimator.diagnostics["accepted"])
    if not errors:
        p.error("empty evaluation")
    arr=np.array(errors)
    result={"frames":len(arr),"window_s":[a.start,a.end],**counts,
            "p90_m":dict(zip(("velocity","sparse_range"), np.quantile(arr,.9,axis=0).tolist())),
            "final_m":dict(zip(("velocity","sparse_range"), arr[-1].tolist()))}
    a.output.write_text(json.dumps(result,indent=2))
    print(result)


if __name__=="__main__":
    main()
