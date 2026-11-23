"""Offline build and test a metric surface prior for the tunnel structures."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.portal_localization import SurfaceReference
from go2_rescue.types import Pose


def surfaces(model, data):
    all_points, all_normals = [], []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name.startswith(("tunnel2_", "tunnel3_", "finish_south_", "east_boundary")):
            continue
        kind = model.geom_type[gid]
        matrix, center = data.geom_xmat[gid].reshape(3,3), data.geom_xpos[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            offset, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[offset:offset+count]@matrix.T+center
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces = model.mesh_face[start:start+count]
            for i,j,k in faces:
                a,b = verts[j]-verts[i],verts[k]-verts[i]
                normal = np.cross(a,b)
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue
                n = max(1, int(np.ceil(max(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(a-b))/.05)))
                points = np.array([verts[i]+u/n*a+v/n*b for u in range(n+1) for v in range(n+1-u)])
                all_points.append(points)
                all_normals.append(np.broadcast_to(normal/norm,points.shape))
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            half = model.geom_size[gid]
            for axis in range(3):
                u,v = [i for i in range(3) if i !=axis]
                dimensions = np.meshgrid(*[np.linspace(-half[i],half[i],int(np.ceil(2*half[i]/.05))+1) for i in (u,v)])
                for side in (-1.,1.):
                    points = np.zeros((dimensions[0].size,3))
                    points[:,axis]=side*half[axis]
                    points[:,u],points[:,v]=[arr.ravel() for arr in dimensions]
                    all_points.append(points@matrix.T+center)
                    all_normals.append(np.broadcast_to(side*matrix[:,axis],points.shape))
    points = np.vstack(all_points)-[.8,.8,0.]
    normals = np.vstack(all_normals)
    _, unique = np.unique(np.floor(points/.027).astype(int),axis=0,return_index=True)
    return points[unique], normals[unique]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--write-reference",type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root/"scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    points,normals=surfaces(model,data)
    reference=SurfaceReference(points,normals)
    if args.write_reference:
        np.savez_compressed(args.write_reference,points=points,normals=normals)
    config=Config.load(args.run/"config.json")
    trace={round(r["timestamp"],5):r for r in map(json.loads,(args.run/"trajectory.jsonl").open())}
    for failed_only in (False,True):
        tested=0
        errors=[]
        for r in map(json.loads,(args.run/"recording/frames.jsonl").open()):
            log=trace.get(round(r["timestamp"],5))
            if log is None:
                continue
            pose=Pose(**log["pose"])
            if not (10.3<pose.x<11.1 and 4.7<pose.y<6.0):
                continue
            if failed_only and pose.quality>=config.min_pose_quality:
                continue
            with np.load(args.run/"recording"/r["file"]) as packet:
                scan=packet["points"]@gravity_rotation(packet["quaternion"]).T
                quat=packet["quaternion"]
            # The replay stores the actual frozen XY estimate, but heading
            # measurements must continue through a failed XY registration.
            from go2_rescue.localization import Localization
            yaw=Localization._imu_yaw(quat)
            guess=Pose(pose.x,pose.y,yaw)
            fit=reference.fit(scan,guess,.33)
            tested+=1
            if fit:
                truth=np.array(log["diagnostics_only_world_position"][:2])-.8
                errors.append(float(np.linalg.norm(fit[0].xy()-truth)))
        print("failed_only",failed_only,"tested",tested,"fits",len(errors),
              "p90/max",np.quantile(errors,.9) if errors else None,max(errors,default=None))


if __name__=="__main__":
    main()
