"""Generate training-only views of the local proxy sphere at varied distances."""
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from go2_rescue.config import Config
from go2_rescue.mujoco_scene import task_scene
from go2_rescue.mujoco_sim import SimSensors

root=Path(__file__).resolve().parents[1]
out=root/"data/ball-views";out.mkdir(parents=True,exist_ok=False)
model=task_scene(root,out,(5.3,2.25),layout="perception",save_model=False,pipe_ramp="official")
data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
ball=model.body("local_football").id
geom=model.geom("local_football_geom").id
j=model.body_jntadr[ball];q=model.jnt_qposadr[j]
center=np.array([7.2,2.7,.11])
cfg=Config();sensors=SimSensors(model,data,cfg)
rows=[]
try:
    for index,(dist,angle,yaw) in enumerate((d,a,y) for d in (1.2,2.3,3.5) for a in (0.,.20,-.20,np.pi) for y in (0.,1.4)):
        xy=center[:2]-dist*np.array([np.cos(angle),np.sin(angle)])
        data.qpos[:7]=[*xy,.33,np.cos(angle/2),0,0,np.sin(angle/2)]
        data.qpos[q:q+7]=[*center,np.cos(yaw/2),0,0,np.sin(yaw/2)]
        mujoco.mj_forward(model,data)
        image=sensors.image();ids=sensors.segmentation()
        mask=(ids[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)&(ids[:,:,0]==geom)
        ys,xs=np.nonzero(mask)
        if len(xs)<60:continue
        if xs.min()<3 or ys.min()<3 or xs.max()>316 or ys.max()>236:continue
        name=f"view-{index:03d}"
        cv2.imwrite(str(out/(name+".png")),image)
        cv2.imwrite(str(out/(name+"-mask.png")),mask.astype(np.uint8)*255)
        rows.append({"image":name+".png","mask":name+"-mask.png","distance":dist,"yaw":yaw,"view_angle":angle,
                     "body_world_xy":xy.tolist(),"ball_world_xyz":center.tolist()})
finally:sensors.close()
(out/"manifest.json").write_text(json.dumps({"use":"training only","views":rows},indent=2))
print("training views",len(rows))
