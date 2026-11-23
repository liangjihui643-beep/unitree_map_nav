"""Encode recorded selected returns, then map without their elevation indices."""
import json,sys
from go2_rescue.range_odometry import RangeOdometry
from pathlib import Path
import numpy as np
from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.lidar_obs import encode_lidar_obs,RangeOnlyEvidence
from go2_rescue.mapping import GridMap
from go2_rescue.recording import replay
from go2_rescue.types import Pose
r=Path(sys.argv[1]); cfg=Config.load(r/'config.json')
rows={round(q['timestamp'],3):q for line in (r/'trajectory.jsonl').open() if (q:=json.loads(line))}
old=GridMap(cfg); layer=RangeOnlyEvidence(old,cfg.lidar_elevation_center_deg,cfg.lidar_vertical_fov_deg)
count=0
odom=RangeOdometry(cfg.min_range,cfg.max_range)
errors=[]; dead_errors=[]; valid=0; predicted=[]; gt=[]
dead=np.zeros(2); last=None
anchor=None
for f in replay(r/'recording'):
    if f.timestamp<90.: continue
    if f.timestamp>125.: break
    row=rows.get(round(f.timestamp,3))
    if row is None: continue
    pose=Pose(**row['pose'])
    if pose.quality<cfg.min_pose_quality: continue
    level=gravity_rotation(f.quaternion)
    old.ground_height=row['diagnostics']['ground_height']
    old.update(f.points@level.T,pose)
    d=f.points-[0,0,.10]
    ids=np.rint((np.arctan2(d[:,1],d[:,0])+np.pi)*72/(2*np.pi)).astype(int)%72
    full=np.full(72,np.inf);np.minimum.at(full,ids,np.linalg.norm(d,axis=1))
    ok=np.isfinite(full)
    obs=encode_lidar_obs(ok,np.clip(full[ok],cfg.min_range,cfg.max_range),cfg.max_range)
    layer.update(old,obs,f.quaternion,pose,cfg.min_range,cfg.max_range)
    own=odom.update(obs,f.timestamp,f.quaternion,f.velocity)
    xy=row.get("diagnostics_only_world_position")
    if xy is not None:
        if anchor is None:
            anchor=np.asarray(xy[:2])
            theta=pose.yaw
            R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        target=(np.asarray(xy[:2])-anchor) @ R
        if last is not None:
            yaw=own.yaw
            heading=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            dead+=heading@f.velocity.array()[:2]*(f.timestamp-last)
        last=f.timestamp
        dead_errors.append(float(np.linalg.norm(dead-target)))
        predicted.append(own.xy().tolist())
        gt.append(target.tolist())
        if own.quality>=cfg.min_pose_quality:
            errors.append(float(np.linalg.norm(own.xy()-target)))
            valid+=1
    count+=1
legacy_free=old.logodds<-.5
n={'frames':count,'legacy_free_cells':int(legacy_free.sum()),
   'legacy_free_without_floor':int((legacy_free&~np.isfinite(old.floor)).sum()),
   'range_common_projected_cells':int(layer.ray_prefix.sum()),
   'possible_hit_cells':int(layer.possible_hit.sum()),
   'near_ground_band_cells_no_independent_floor':int(layer.ground_band_clear.sum())}
n["range_pose"]={"accepted":valid,"scored":len(gt),
                 "accepted_p90_m":float(np.quantile(errors,.9)) if errors else None,
                 "integrated_velocity_p90_m":float(np.quantile(dead_errors,.9)) if dead_errors else None,
                 "final_error_m":float(np.linalg.norm(np.asarray(predicted[-1])-gt[-1])) if len(gt) else None}
Path(sys.argv[2]).write_text(json.dumps(n,indent=2)); print(n)
