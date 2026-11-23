"""Check support-plane estimation during an existing torque-controlled ramp approach."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import cv2
import mujoco
import numpy as np
from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation, rot2
from go2_rescue.mpc_locomotion import ConvexMPC, LEGS
from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.types import Frame, Velocity


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    model=mujoco.MjModel.from_binary_path(str(root/"scene/mujoco/competition-course.mjb"))
    data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
    data.qpos[:7]=[1.1,4.,.30,np.sqrt(.5),0,0,np.sqrt(.5)]
    mujoco.mj_forward(model,data)
    control=ConvexMPC(model,data,root)
    config=Config();sensors=SimSensors(model,data,config);estimator=StanceSurface()
    robot=np.flatnonzero(model.body_rootid[model.geom_bodyid]==model.body("base_link").id)
    rows=[];images=[]
    try:
        for tick in range(4300):
            t=float(data.time)
            if tick%50==0:
                scan=sensors.lidar()
                control.observe_terrain(scan)
                frame=Frame(t,scan,data.sensor("imu_quat").data.copy(),control.measured_velocity(),
                            support_points=control.measured_support_points())
                frame.validate()
                plane=estimator.update(frame)
                # Independent reference, evaluated after the estimate.
                feet=np.array([data.xpos[model.body(f"{leg}_foot").id] for leg in LEGS])
                heights=[];previous=model.geom_group[robot].copy()
                try:
                    model.geom_group[robot]=5
                    for foot in feet:
                        gid=np.array([-1],dtype=np.int32)
                        distance=mujoco.mj_ray(model,data,foot+[0,0,.5],np.array([0.,0.,-1.]),
                                               np.array([1,1,1,1,1,0],dtype=np.uint8),True,-1,gid)
                        heights.append(foot[2]+.5-distance if distance>=0 else np.nan)
                finally:model.geom_group[robot]=previous
                height=np.array(heights);truth_plane=None
                if np.isfinite(height).all():
                    design=np.column_stack((feet[:,:2],np.ones(4)))
                    fit=np.linalg.lstsq(design,height,rcond=None)[0]
                    residual=float(np.max(abs(design@fit-height)))
                    if residual<.015:truth_plane=fit
                yaw=np.arctan2(2*(frame.quaternion[0]*frame.quaternion[3]+frame.quaternion[1]*frame.quaternion[2]),
                               1-2*(frame.quaternion[2]**2+frame.quaternion[3]**2))
                predicted=(rot2(yaw)@plane[:2]).tolist() if plane is not None else None
                row={"t":t,"status":estimator.status,"plane":plane.tolist() if plane is not None else None,
                     "estimated_global_gradient":predicted,
                     "truth_global_gradient":truth_plane[:2].tolist() if truth_plane is not None else None,
                     "diagnostics_only_world_xy":data.qpos[:2].tolist()}
                rows.append(row)
                if tick in (300,4000):
                    image=sensors.image()
                    cv2.putText(image,f"t={t:.1f}s / estimated slope {estimator.status.get('slope_deg')}",(5,20),0,.40,(255,255,255),1)
                    images.append(image)
            if tick%3==0:
                data.ctrl[:]=control.update(t,Velocity(.14) if t>=.6 else Velocity())
            mujoco.mj_step(model,data)
            if data.qpos[2]<.13:break
    finally:sensors.close()
    evaluated=[r for r in rows if r["estimated_global_gradient"] is not None
               and r["truth_global_gradient"] is not None and np.linalg.norm(r["truth_global_gradient"])>.04
               and r["status"]["source"]=="joint_imu_stance_plane"]
    error=[np.rad2deg(np.arctan(np.linalg.norm(np.asarray(r["estimated_global_gradient"])-r["truth_global_gradient"])))
           for r in evaluated]
    result={"physical_seconds":float(data.time),"end_world_xy":data.qpos[:2].tolist(),
            "frames":len(rows),"estimated_plane_frames":sum(r["status"]["source"]=="joint_imu_stance_plane" for r in rows),
            "sloped_reference_frames":len(evaluated),
            "slope_gradient_error_p50_p90_deg":np.quantile(error,[.5,.9]).tolist() if error else None,
            "source":"torque-controlled approach with unchanged controller; truth rays scored after estimation"}
    (args.output/"support_frames.json").write_text(json.dumps(rows,indent=2))
    (args.output/"support_acceptance.json").write_text(json.dumps(result,indent=2))
    if images:cv2.imwrite(str(args.output/"ramp-support.png"),np.hstack(images))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
