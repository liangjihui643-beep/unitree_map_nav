"""Rendered RGB regression: same fixture ball, floor and elevated."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from go2_rescue.ball_classifier import BallProposalClassifier
from go2_rescue.ball_multiview import BallTwoView, fuse_target_position
from dataclasses import replace
from go2_rescue.mujoco_scene import task_scene
from go2_rescue.mujoco_sim import SimSensors, simulation_config
from go2_rescue.perception import Perception
from go2_rescue.types import Frame, Pose
from go2_rescue.tracking import Tracker

root=Path.cwd()
out=Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
model=task_scene(root,out,(.8,.8), 'near-start', save_model=False)
data=mujoco.MjData(model)
config=simulation_config(task_fixture=True, task_layout='near-start')
visual=Perception(config)
classifier=BallProposalClassifier(root/'models/ball_proposal.onnx')
sensor=SimSensors(model,data,config)
joint=int(model.jnt_qposadr[model.joint('local_football_free').id])
rows={}
for scenario, ball_z in (('nominal', .11),('raised',.42)):
    solver=BallTwoView(visual.k)
    tracker=Tracker(config,static_targets=True)
    result=[]
    for i, dy in enumerate(np.linspace(-.30,.30,9)):
        data.qpos[:3]=[.8,.8+dy,.33]
        data.qpos[3:7]=[1,0,0,0]
        data.qpos[joint:joint+7]=[2.,.8,ball_z,1,0,0,0]
        mujoco.mj_forward(model,data)
        rgb=sensor.image()
        __import__('cv2').imwrite(str(out/f'{scenario}-{i}.png'),rgb)
        candidates=classifier.filter(rgb, visual.detect_balls(rgb,small_targets=True))
        pose=Pose(.8,.8+dy,quality=1.,sigma=.02)
        old=[]
        single=[]
        for d in candidates:
            visual.locate_ball(d,Frame(i*.2,np.empty((0,3))),pose)
            if d.position is not None:
                old.append(float(np.linalg.norm(d.position-[2.,.8])))
            single.append(replace(d))
            d.position=None
            d.range_source='unknown'
        solver.update(candidates,pose,.33,visual.t[:3,:3],visual.t[:3,3],i*.2,.85)
        for d, ref in zip(candidates, single):
            fuse_target_position(d, ref)
        new=[float(np.linalg.norm(d.position-[2.,.8])) for d in candidates if d.position is not None]
        events=tracker.update([d for d in candidates if d.position is not None
                               and d.confidence>=.85 and d.sigma<=.2],pose,i*.2)
        result.append({'old_errors_m':old,'new_errors_m':new,'classifier_candidates':len(candidates),
                       'published_target_events':[e.payload for e in events]})
    rows[scenario]=result
sensor.close()
(out/'ball_metrics.json').write_text(json.dumps(rows,indent=2))
for scenario, seq in rows.items():
    print(scenario, 'events', [e for row in seq for e in row['published_target_events']], 'proposals',sum(row['classifier_candidates'] for row in seq))
    for mode in ('old','new'):
        v=[x for row in seq for x in row[mode+'_errors_m']]
        print(mode,len(v),'p50',float(np.median(v)) if v else None,'p90',float(np.quantile(v,.9)) if v else None)
