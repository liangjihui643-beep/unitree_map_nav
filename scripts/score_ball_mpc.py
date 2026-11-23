"""Compare event-level tracker results on a single labelled walking fixture."""
import json,sys
from go2_rescue.geometry import gravity_rotation
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.ball_multiview import BallTwoView
from pathlib import Path
import numpy as np
from go2_rescue.config import Config
from go2_rescue.perception import Perception
from go2_rescue.perception_demo import confirmable_ball
from go2_rescue.recording import replay
from go2_rescue.tracking import Tracker
from go2_rescue.types import BallObservation,Pose

p=Path(sys.argv[1])
read=lambda name:{round(row.get('t',row.get('timestamp')),3):row
                  for v in (p/name).read_text().splitlines() if (row:=json.loads(v))}
t,logs=read('trajectory.jsonl'),read('target_tracks.jsonl')
loc=read('lidar_diagnostics.jsonl')
truth=json.loads((p/'evaluation_truth.json').read_text())
truth_by_time={round(x['t'],3):x for x in truth}
anchor=np.array(truth[0]['base_world_xyz'][:2])
cfg=Config.load(p/'config.json')
cfg.target_quota=1
model=Perception(cfg)
tracker=Tracker(cfg,static_targets=True)
old=[]; new=[]; size=[]; candidates=0
surface=StanceSurface()
ball=BallTwoView(model.k,ball_radius=cfg.ball_radius, consensus=True)
other=Tracker(cfg,static_targets=True)
max_hits=0; solved=[]
for frame in replay(p/'recording'):
    plane=surface.update(frame)
    k=round(frame.timestamp,3)
    if k not in logs or k not in loc:
        continue
    pose=Pose(**loc[k]['pose'])
    if pose.quality<cfg.min_pose_quality: continue
    row=logs[k]
    proposals=[BallObservation(tuple(d['center_px']),d['radius_px'],d['confidence'],0)
               for d in row['detections'] if d['confidence']>=.85]
    candidates+=len(proposals)
    placed=[model.locate_ball(x,frame,pose) for x in proposals]
    for event in tracker.update([d for d in placed if confirmable_ball(d,.85)],pose,frame.timestamp):
        old.append({'t':k,'xy':event.payload['position']})
    for d in proposals:
        d.position=None
        d.range_source="unknown"
    attitude=gravity_rotation(frame.quaternion)
    z=-plane[2] if plane is not None else np.nan
    ball.update(proposals,pose,z,attitude@model.t[:3,:3],attitude@model.t[:3,3],frame.timestamp,.85)
    solved.extend({'t':k,'xy':d.position.tolist()} for d in proposals if d.position is not None)
    for event in other.update([d for d in proposals if confirmable_ball(d,.85)],pose,frame.timestamp):
        size.append({'t':k,'xy':event.payload['position']})
    max_hits=max([max_hits,*[track.hits for track in other.tracks]])
    new.extend({'t':k,'xy':x['map_xy']} for x in row['new_confirmations'])
for sequence in (old,new,size):
    for row in sequence:
        expected=truth_by_time[row['t']]['football_world_xy']
        row['error_m']=float(np.linalg.norm(np.array(row['xy'])+anchor-expected))
    # Last-frame and init truth are never used by the estimators.
xyz=np.asarray([v['base_world_xyz'] for v in truth])
out={'duration_s':truth[-1]['t']-truth[0]['t'],
     'world_translation_xy_m':np.ptp(xyz[:,:2],axis=0).tolist(),
     'world_displacement_m':float(np.linalg.norm(xyz[-1,:2]-xyz[0,:2])),
     'classified_candidates':candidates,'old':old,'new':new,'size_validated':size,'consensus_max_hits':max_hits,'consensus_candidates':solved}
(p/'events_ab.json').write_text(json.dumps(out,indent=2))
print(out)
