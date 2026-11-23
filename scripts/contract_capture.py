"""Capture a short contract-profile LiDAR sequence along the public reference route.

Poses are scripted from the public manifest route (external estimate for a map
A/B); the same poses are used by every map variant. Simulator truth is kept
separate for scoring only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.public_task import PublicTask

JOINT_NAMES = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
               "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]


def selected_points(meta, config):
    """Diagnostic XYZ for the selected echo per azimuth (baseline input only)."""
    elevation = np.deg2rad(np.linspace(-config.lidar_vertical_fov_deg / 2,
                                       config.lidar_vertical_fov_deg / 2, 9))
    points = np.zeros((72, 3))
    mount = np.asarray(config.lidar_mount, float)
    for azimuth, elevation_index, distance in zip(meta["azimuth_index"],
                                                  meta["elevation_index"], meta["range"]):
        angle = -np.pi + azimuth * (2 * np.pi / 72)
        ele = elevation[elevation_index]
        direction = np.array([np.cos(ele) * np.cos(angle),
                              np.cos(ele) * np.sin(angle), np.sin(ele)])
        points[azimuth] = mount + direction * distance
    return points


def samples(route, initial, spacing=.15):
    points = [np.asarray(initial, float)] + [np.asarray(p, float) for p in route]
    result = []
    for a, b in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(b[:2] - a[:2]))
        count = max(1, int(np.ceil(distance / spacing)))
        for i in range(count):
            fraction = i / count
            xyz = a + (b - a) * fraction
            yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            result.append((xyz, yaw))
    result.append((points[-1], float(np.arctan2(points[-1][1] - points[-2][1],
                                                 points[-1][0] - points[-2][0]))))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--catalog", type=Path)
    ap.add_argument("--spacing", type=float, default=.15)
    args = ap.parse_args()
    task = PublicTask.load(args.task, args.catalog)
    config = task.config()
    model = mujoco.MjModel.from_binary_path(str(task.scene_path))
    data = mujoco.MjData(model)
    joint_q = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    home = model.key_qpos[0][joint_q].copy() if model.nkey else model.qpos0[joint_q].copy()
    sensors = SimSensors(model, data, config, seed=3101)
    rows = []
    lidar = []
    selected = []
    poses = []
    quaternions = []
    times = []
    try:
        initial = task.manifest["initial_position"]
        for index, (xyz, yaw) in enumerate(samples(task.manifest["route"], initial, args.spacing)):
            quaternion = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)
            data.qpos[:3] = xyz
            data.qpos[3:7] = quaternion
            data.qpos[joint_q] = home
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            points = sensors.lidar()
            meta = sensors.last_lidar_metadata
            obs = np.asarray(meta["agent_obs"], np.float32)
            lidar.append(obs)
            selected.append(selected_points(meta, config))
            poses.append(np.asarray(xyz, float))
            quaternions.append(np.asarray(quaternion, float))
            times.append(index * .1)
            rows.append({"index": index, "t": index * .1,
                         "pose": np.asarray(xyz, float).tolist(), "yaw": yaw,
                         "truth_pose": np.asarray(data.qpos[:3], float).tolist(),
                         "valid_azimuths": int(np.asarray(meta["valid"]).sum()),
                         "raw_valid_fraction": float(meta["raw_valid_fraction"])})
    finally:
        sensors.close()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz", lidar=np.stack(lidar), pose=np.stack(poses),
                        quaternion=np.stack(quaternions), time=np.asarray(times, np.float32),
                        selected_points=np.stack(selected))
    (out / "frames.jsonl").write_text("\n".join(json.dumps(v) for v in rows) + "\n", encoding="utf-8")
    meta = {"schema": "go2-contract-capture/1", "task": task.manifest["id"],
            "task_version": task.manifest["version"], "frames": len(rows),
            "spacing_m": args.spacing, "scene_sha256": task.verified["scene_sha256"],
            "sensors_sha256": task.verified["sensors_sha256"],
            "pose_source": "public manifest route, external estimate; same pose reused by both maps",
            "note": "truth_pose is separated for scoring; tensors contain only LiDAR contract obs"}
    (out / "capture.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
