"""Compare fixed-plane and multi-view localisation on recorded classifier proposals."""
import argparse
import json
from pathlib import Path
import numpy as np
from dataclasses import replace
from go2_rescue.ball_multiview import BallTwoView, fuse_target_position
from go2_rescue.geometry import gravity_rotation
from go2_rescue.perception import Perception
from go2_rescue.config import Config
from go2_rescue.types import Pose, BallObservation, Frame

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('out', type=Path)
args=parser.parse_args()
r=args.run
read=lambda name: (json.loads(l) for l in (r/name).read_text().splitlines())
cfg=Config.load(r/'config.json')
traj={round(x['timestamp'],3):x for x in read('trajectory.jsonl')}
records={round(x['timestamp'],3):x for x in read('recording/frames.jsonl')}
ball=np.array(list(json.loads((r/'scene_provenance.json').read_text()).get('targets',[])))
truth=json.loads((r/'evaluation_truth.json').read_text())
anchor=truth[0]['base_world_xyz'][:2]
vision=Perception(cfg)
solver=BallTwoView(vision.k)
old,new,fused=[],[],[]
out={'accepted_classifier_proposals':0,'tolerance_m':.5}
for x in read('target_tracks.jsonl'):
    t=round(x['t'],3)
    if t not in traj or t not in records:
        continue
    row=traj[t]
    pose=Pose(**row['pose'])
    if pose.quality<cfg.min_pose_quality:
        continue
    with np.load(r/'recording'/records[t]['file']) as data:
        q=data['quaternion']
    detections=[BallObservation(tuple(d['center_px']), d['radius_px'],d['confidence'],0.)
                for d in x['detections'] if d['confidence']>=.85]
    out['accepted_classifier_proposals']+=len(detections)
    level=gravity_rotation(q)
    single=[]
    for d in detections:
        result=vision.locate_ball(d, Frame(t,np.empty((0,3)),q),pose)
        if result.position is not None:
            old.append((t,result.position.tolist()))
        single.append(replace(d))
        d.position=None
    support=row['diagnostics'].get('support',{})
    z=(cfg.base_height+row['diagnostics']['ground_height']
       if 'base_to_support_m' in support else np.nan)
    solver.update(detections,pose,z,level@vision.t[:3,:3],level@vision.t[:3,3],t,.85)
    new.extend((t,d.position.tolist()) for d in detections if d.position is not None)
    for d, ref in zip(detections, single):
        fuse_target_position(d, ref)
    fused.extend((t,d.position.tolist()) for d in detections if d.position is not None)
out['fused']=fused
out['map_anchor_world_xy']=anchor
out['old']=old
out['new']=new
args.out.write_text(json.dumps(out))
print({k:len(out[k]) for k in ['old','new']}, 'input',out['accepted_classifier_proposals'])
