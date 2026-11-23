"""Offline-only supervised floor data; runtime never imports this generator."""
import os,json
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import cv2,mujoco,numpy as np
from go2_rescue.mujoco_sim import SimSensors,simulation_config
root=Path.cwd();out=root/'data/navigation-floor';(out/'images').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
m=mujoco.MjModel.from_binary_path(str(root/'scene/mujoco/competition-course.mjb'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
c=simulation_config();s=SimSensors(m,d,c);lookup=np.full(m.ngeom,2,np.uint8)
for g in range(m.ngeom):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '';body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
 if 'ground' in name.lower() or m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE or body.startswith('ramp'):lookup[g]=0
 if body.startswith('pile'):lookup[g]=1
robot=np.array([i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in s.robot_bodies]) if hasattr(s,'robot_bodies') else s.robot_geoms
rows=[];rng=np.random.default_rng(71);positions=[(.8,.8),(1.1,2),(1.1,3.5),(1.1,4.5),(1.1,5.5),(1.1,6.7),(1.1,8),(2.,9),(3.,9),(3.5,7),(4.,5),(4.8,3.5),(5.5,2),(6.5,1),(7.5,1),(9,1),(10.5,1),(11,3),(11,5),(10.7,7),(10.5,8.8),(8.5,8.8),(6.5,8.8),(5,7.5),(6.5,6.5),(6.5,5),(8,4),(9.5,3),(4.5,1),(2.5,.8)]
try:
 for loc,(x,y) in enumerate(positions):
  old=m.geom_group[robot].copy();m.geom_group[robot]=5
  gid=np.array([-1],np.int32);height=mujoco.mj_ray(m,d,np.array([x,y,3.]),np.array([0.,0.,-1.]),np.array([1,1,1,1,1,0],np.uint8),True,-1,gid)
  m.geom_group[robot]=old
  floor=3-height
  if height<0 or floor>1.2:continue
  for view in range(8):
   yaw=view*np.pi/4+rng.uniform(-.1,.1)
   d.qpos[:7]=[x,y,floor+.33,np.cos(yaw/2),0,0,np.sin(yaw/2)];mujoco.mj_forward(m,d)
   image=s.image();ids=s.segmentation()[...,0];label=np.full(ids.shape,2,np.uint8);valid=(ids>=0)&(ids<m.ngeom);label[valid]=lookup[ids[valid]]
   stem=f'loc{loc:02d}_{view}';cv2.imwrite(str(out/'images'/f'{stem}.png'),image);cv2.imwrite(str(out/'masks'/f'{stem}.png'),label)
   rows.append({'name':stem,'split':'val' if loc%5==3 else 'train','world_viewpoint':[x,y,floor+.33,yaw]})
  print('generated',loc,len(rows),flush=True)
finally:s.close()
(out/'meta.json').write_text(json.dumps({'frames':rows,'scope':'offline scene-supervised RGB only; location-group holdout; no truth input to runtime'},indent=2))
