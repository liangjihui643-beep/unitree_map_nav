"""Offline public-scene prior; never called by the online navigator."""
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.ndimage import binary_dilation

root=Path(__file__).resolve().parents[1]
source=root/'scene/mujoco/competition-course.mjb'
model=mujoco.MjModel.from_binary_path(str(source));data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
base=model.body('base_link').id
model.geom_group[model.body_rootid[model.geom_bodyid]==base]=5
resolution=.05;n=240
floor=np.zeros((n,n),bool);rock=floor.copy();wall=floor.copy()
gid=np.array([-1],np.int32);groups=np.array([1,1,1,1,1,0],np.uint8)
for i in range(240):
 for j in range(200):
  hit=mujoco.mj_ray(model,data,np.array([(i+.5)*resolution,(j+.5)*resolution,3.]),np.array([0.,0.,-1.]),groups,True,-1,gid)
  if hit<0:continue
  g=int(gid[0]);name=model.geom(g).name or '';body=model.body(int(model.geom_bodyid[g])).name or ''
  rock[i,j]=body.startswith('pile')
  floor[i,j]=bool(not rock[i,j] and ('ground' in name.lower() or model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE or body.startswith('ramp')) and 3.-hit<1.)
  if not floor[i,j] and ('roof' in name.lower() or 'ceiling' in name.lower()):
   below=mujoco.mj_ray(model,data,np.array([(i+.5)*resolution,(j+.5)*resolution,.50]),np.array([0.,0.,-1.]),groups,True,-1,gid)
   if below>=0:
    ground_z=.50-below
    under=int(gid[0]);under_name=(model.geom(under).name or '').lower()
    up=mujoco.mj_ray(model,data,np.array([(i+.5)*resolution,(j+.5)*resolution,ground_z+.02]),np.array([0.,0.,1.]),groups,True,-1,gid)
    declared_ground=('ground' in under_name or model.geom_type[under]==mujoco.mjtGeom.mjGEOM_PLANE)
    floor[i,j]=bool(declared_ground and abs(ground_z)<.04 and (up<0 or up+.02>=.50))
  wall[i,j]=not floor[i,j] and not rock[i,j]
# Thin vertical source walls can lie BETWEEN top-down raster centres. Sample
# horizontal public-map rays too; otherwise a 5cm raster loses those walls
# entirely and provides a false localization/planning reference.
angle=np.arange(360)*np.pi/180.
directions=np.column_stack([np.cos(angle),np.sin(angle),np.zeros(len(angle))])
reference_points=[]
for x in np.arange(.25,12.,.5):
 for y in np.arange(.25,10.,.5):
  for z in [.25,.50,.80]:
   ids=np.full(len(angle),-1,np.int32);ranges=np.full(len(angle),-1.,float)
   origin=np.array([x,y,z])
   mujoco.mj_multiRay(model,data,origin,directions.ravel(),groups,True,-1,ids,ranges,None,len(angle),12.)
   for k in np.flatnonzero((ids>=0)&(ranges>0)&(ranges<12.)):
    g=int(ids[k]);name=(model.geom(g).name or '').lower();body=model.body(int(model.geom_bodyid[g])).name or ''
    p=origin[:2]+ranges[k]*directions[k,:2];cell=np.floor(p/resolution).astype(int)
    # Supplied outer-wall inner faces are at x/y=-.009m, just outside
    # the nominal 0..12 raster. Retain them in the metric reference;
    # clip only the navigation border cell, never the physical XY point.
    if not ((p>=-.10).all() and (p<=12.10).all()):continue
    ij=tuple(np.clip(cell,0,n-1))
    if body.startswith('pile'):
     rock[ij]=True
    elif (not body.startswith('ramp') and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE
          and not any(word in name for word in ['roof','ceiling'])
          and (any(word in name for word in ['wall','boundary','partition'])
               or not any(word in name for word in ['ground','floor','ramp','slope']))):
     wall[ij]=True;reference_points.append(p)
floor &= ~(wall|rock)
reference=np.asarray(reference_points)-[.8,.8]
_,unique=np.unique(np.floor(reference/.03).astype(int),axis=0,return_index=True)
reference=reference[unique]
out=root/'configs/public-navigation-prior.npz'
if out.exists():
 old_sha=hashlib.sha256(out.read_bytes()).hexdigest()
 for manifest in (root/'output').glob('coverage-planned-survey-*/run_manifest.json'):
  if json.loads(manifest.read_text()).get('navigation_prior_sha256')==old_sha:
   snapshot=manifest.parent/'navigation_prior.npz'
   if not snapshot.exists():shutil.copyfile(out,snapshot)
np.savez_compressed(out,floor=floor,rock=rock,wall=wall,reference_xy=reference,
                    origin=np.array([-.8,-.8]),resolution=resolution)
meta={'source':'offline public scene geometry; NOT measured map','scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'coordinate_frame':'map_start; declared start world(.8,.8), yaw0','floor_cells':int(floor.sum()),'rock_cells':int(rock.sum()),'reference_points':len(reference)}
out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
image=np.full((n,n,3),60,np.uint8);image[floor]=(210,210,210);image[rock]=(40,150,210);image[wall]=(20,20,20)
cv2.imwrite(str(out.with_suffix('.png')),cv2.flip(image.transpose(1,0,2),0))
print(json.dumps(meta,indent=2))
