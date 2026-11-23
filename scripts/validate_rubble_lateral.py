"""Controlled rendered lateral motion, no MPC and no production truth input."""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors, simulation_config
from go2_rescue.rubble_visual_height import RubbleVisualHeight
from go2_rescue.types import Pose

out=Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(1)
cfg=simulation_config()
model=mujoco.MjModel.from_binary_path("scene/mujoco/competition-course.mjb")
data=mujoco.MjData(model)
sensor=SimSensors(model,data,cfg)
seg=LowObstacleDetector("models/seg_rubble_scene.onnx",cfg.camera_matrix,cfg.body_from_camera,cfg.base_height)
mount=np.asarray(cfg.body_from_camera)
result={}
mode=sys.argv[2] if len(sys.argv)>2 else 'base'
assert mode in {'base','separate','wider'}
for name,x,y,yaw in [('side',6.2,2.56,-np.pi/2),('opposite',7.3,2.56,np.pi/2)]:
    solver=RubbleVisualHeight(np.asarray(cfg.camera_matrix), independent_ground=mode!='base')
    points=[]; counts=[]; masks=[]
    sweep = [(d,0.) for d in np.linspace(-.34,.34,18)]
    if mode != 'base':
        sweep = ([(d,-.60) for d in np.linspace(-.34,.34,18)] if mode=='wider' else []) + sweep
    for i,(d,f) in enumerate(sweep):
        here=np.array([x,y])+d*np.array([-np.sin(yaw),np.cos(yaw)])+f*np.array([np.cos(yaw),np.sin(yaw)])
        data.qpos[:3]=[*here,.33]
        data.qpos[3:7]=np.roll(Rotation.from_euler('z',yaw).as_quat(),1)
        mujoco.mj_forward(model,data)
        image=sensor.image()
        prob=seg.segment(image)
        mask=((prob.argmax(axis=0)==1)&(prob[1]>=.70)).astype(np.uint8)
        structure=((prob.argmax(axis=0)==2)&(prob[2]>=.45)).astype(np.uint8)
        mask[cv2.dilate(structure,np.ones((3,3),np.uint8))!=0]=0
        p=solver.observe(image,mask,mount[:3,:3],mount[:3,3],Pose(*here,yaw),0.,None,i*.15,
                         ground_mask=(prob.argmax(axis=0)==0)&(prob[0]>=.65),base_z=.33)
        if f == 0.:
            points.extend(p.tolist())
        counts.append(solver.diagnostics.copy()); masks.append(int(mask.sum()))
        if i in (0,8,17):
            cv2.imwrite(str(out/f'{name}-{i}.png'),image)
    cells={tuple(np.floor(np.asarray(p[:2])/.1).astype(int)) for p in points}
    with_height=[p for p in points if np.isfinite(p[4])]
    result[name]={"frames":len(masks),"rubble_pixels":masks,"point_observations":len(points),
                  "cells":len(cells),"height_observations":len(with_height),
                  "height_cells":len({tuple(np.floor(np.asarray(p[:2])/.1).astype(int)) for p in with_height}),
                  "geometry":{k:sum(c.get(k,0) for c in counts) for k in set().union(*counts)}}
    (out/f'{name}-points.json').write_text(json.dumps([[v if np.isfinite(v) else None for v in p] for p in points]))
sensor.close()
(out/'metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
