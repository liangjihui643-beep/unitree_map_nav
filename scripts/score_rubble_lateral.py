"""Truth scores for controlled world-frame lateral render. Offline only."""
import json
import sys
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

root=Path(sys.argv[1])
model=mujoco.MjModel.from_binary_path("scene/mujoco/competition-course.mjb")
data=mujoco.MjData(model); mujoco.mj_forward(model,data)
mask=np.zeros((100,120),np.uint8)
for g in range(model.ngeom):
    body=model.body(int(model.geom_bodyid[g])).name
    if not body.startswith("pile") or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH: continue
    n=model.geom_dataid[g]; a,l=model.mesh_vertadr[n],model.mesh_vertnum[n]
    xyz=model.mesh_vert[a:a+l]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
    cv2.fillConvexPoly(mask,cv2.convexHull(np.floor(xyz[:,:2]/.1).astype(np.int32)),1)
dist=distance_transform_edt(~mask.astype(bool))*.1
out={}
for f in root.glob("*-points.json"):
    points=np.asarray(json.loads(f.read_text()),dtype=float).reshape(-1,5)
    cells=np.unique(np.floor(points[:,:2]/.1).astype(int),axis=0)
    good=(cells>=0).all(1)&(cells[:,0]<mask.shape[1])&(cells[:,1]<mask.shape[0])
    x,y=cells[good].T
    errors=dist[y,x]
    out[f.stem]={"cells":len(cells),"inside":int((errors==0).sum()),
                 "outside":len(cells)-int((errors==0).sum()),
                 "distance_to_mesh_footprint_p90_m":float(np.quantile(errors,.9)) if len(errors) else None}
(root/"footprint_score.json").write_text(json.dumps(out,indent=2))
print(out)
