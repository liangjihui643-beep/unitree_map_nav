"""Render estimated map in full-scene coordinates, for independent coverage review."""
import argparse
import json
from pathlib import Path
import cv2
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from go2_rescue.geometry import rot2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    run=args.run
    summary=json.loads((run/"summary.json").read_text())
    truth=json.loads((run/"evaluation_truth.json").read_text())
    anchor=np.asarray(truth[0]["base_world_xy"])
    rotation=rot2(summary["initial_yaw"])
    model=mujoco.MjModel.from_binary_path(str(Path(__file__).resolve().parents[1]/"scene/mujoco/competition-course.mjb"))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    canvas=np.full((600,600,3),230,np.uint8)
    ground=model.geom("base_ground_flat_independent_12x10").id
    aabb=model.geom_aabb[ground]
    basis=data.geom_xmat[ground].reshape(3,3)
    center=data.geom_xpos[ground]+basis@aabb[:3]
    extent=abs(basis)@aabb[3:]
    lower=np.round((center-extent)[:2],5)
    upper=np.round((center+extent)[:2],5)
    span=upper-lower

    def pix(points):
        points=np.asarray(points).reshape(-1,2)
        points=(points-lower)/span
        return np.column_stack((points[:,0]*600,(1-points[:,1])*600)).astype(np.int32)

    model.vis.global_.offwidth = max(600, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(600, model.vis.global_.offheight)
    model.vis.global_.fovy = 60.
    model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(model,height=600,width=600)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [*(lower+upper)/2,0.]
    camera.distance = 13.
    camera.azimuth = 90.
    camera.elevation = -90.
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    try:
        renderer.update_scene(data,camera=camera,scene_option=options)
        raw = cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        cam = renderer.scene.camera[0]
        origin = np.array(cam.pos)
        forward = np.array(cam.forward)
        up = np.array(cam.up)
        right = np.cross(forward,up)
        def project(points):
            delta = np.asarray(points)-origin
            depth = delta@forward
            focal=300/np.tan(np.deg2rad(30))
            return np.column_stack((300+focal*(delta@right)/depth,300-focal*(delta@up)/depth)).astype(int)
        corners=project([[lower[0],lower[1],0],[upper[0],lower[1],0],
                         [upper[0],upper[1],0],[lower[0],upper[1],0]]).astype(np.float32)
        target=np.array([[0,600],[600,600],[600,0],[0,0]],dtype=np.float32)
        homography=cv2.getPerspectiveTransform(corners,target)
        canvas=cv2.warpPerspective(raw,homography,(600,600))
    finally:
        renderer.close()
    with np.load(run/"map.npz",allow_pickle=False) as grid:
        xy=np.argwhere(grid["seen"])
        world=(grid["origin"]+(xy+.5)*float(grid["resolution"]))@rotation.T+anchor
        occupied=grid["logodds"][tuple(xy.T)]>.5
    in_arena=((world>=lower)&(world<upper)).all(1)
    cells=np.unique(np.floor((world[in_arena]-lower)/.1).astype(int),axis=0)
    mapped=np.full_like(canvas,75)
    for point,obstacle in zip(pix(world[in_arena]),occupied[in_arena]):
        color=(20,20,20) if obstacle else (215,215,215)
        cv2.rectangle(mapped,tuple(point-[2,2]),tuple(point+[2,2]),color,-1)
    poses=[json.loads(line)["pose"] for line in (run/"lidar_diagnostics.jsonl").read_text().splitlines()]
    estimated=np.asarray([[p["x"],p["y"]] for p in poses])@rotation.T+anchor
    cv2.polylines(mapped,[pix(estimated)],False,(255,170,30),2)
    cv2.polylines(canvas,[pix([p["base_world_xy"] for p in truth])],False,(255,170,30),2)
    reference=canvas.copy()
    comparison=np.full((680,1240,3),25,np.uint8)
    comparison[60:660,10:610]=reference
    comparison[60:660,630:1230]=mapped
    for label,x in [("Source scene + evaluated path",15),("Estimated map; dark gray is UNKNOWN",635)]:
        cv2.putText(comparison,label,(x,28),0,.67,(240,240,240),1,cv2.LINE_AA)
    denominator=int(np.prod(np.ceil(span/.1)))
    coverage=len(cells)/denominator
    cv2.putText(comparison,f"Known sampled cells in {span[0]:g}m x {span[1]:g}m main-ground rectangle: {coverage:.1%}",(15,676),0,.6,(240,240,240),1)
    cv2.imwrite(str(run/"scene-coverage.png"),comparison)
    result={ "rectangle_world_xy_m":[float(lower[0]),float(upper[0]),float(lower[1]),float(upper[1])],
            "reference_geom":"base_ground_flat_independent_12x10",
            "denominator_cells":denominator,"known_0_10m_cells":len(cells),
            "rectangle_known_fraction":coverage,"scope":"bounding rectangle includes walls and occluded cells",
            "whole_scene_coverage_proven":False,
            "reason":"route does not visit all connected scene areas; rectangle is a visualization denominator"}
    (run/"scene_coverage.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
