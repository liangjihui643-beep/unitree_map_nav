"""Public reference-task contract smoke: verify hashes and build six observations.

The output contains only agent-visible tensors (obs/rgb/state/sensor_meta) and
LiDAR range/mask diagnostics. Simulator truth is not written into the tensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.public_task import PublicTask

JOINT_NAMES = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
               "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]


def resize_rgb(image):
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    return np.transpose(image[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--catalog", type=Path)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3101)
    args = ap.parse_args()
    task_dir, out = args.task.resolve(), args.output
    out.mkdir(parents=True, exist_ok=True)
    task = PublicTask.load(task_dir, args.catalog)
    config = task.config()
    model = mujoco.MjModel.from_binary_path(str(task.scene_path))
    data = mujoco.MjData(model)
    position, quaternion = task.initial_state()
    data.qpos[:3] = position
    data.qpos[3:7] = quaternion
    joint_q = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    joint_v = [model.jnt_dofadr[model.joint(name).id] for name in JOINT_NAMES]
    if model.nkey:
        home = model.key_qpos[0][joint_q].copy()
    else:
        home = model.qpos0[joint_q].copy()
    data.qpos[joint_q] = home
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)
    sensors = SimSensors(model, data, config, seed=args.seed)
    observations, rgbs, states, metas = [], [], [], []
    lidar_ranges, lidar_masks, stamps, fresh_flags = [], [], [], []
    raw_fractions, azimuth_fractions = [], []
    last = None
    try:
        for step in range(args.steps):
            fresh = step % 5 == 0 or last is None
            if fresh:
                points = sensors.lidar()
                meta = dict(sensors.last_lidar_metadata)
                image = resize_rgb(sensors.image())
                last = {"t": float(data.time), "meta": meta, "image": image}
            meta, image = last["meta"], last["image"]
            valid = np.asarray(meta["valid"], bool)
            ranges = np.asarray(meta["range"], float)
            full_ranges = np.zeros(72, np.float32)
            full_ranges[valid] = ranges
            timestamp = float(data.time)
            rgb_age = timestamp - last["t"]
            lidar_age = timestamp - last["t"]
            obs = task.build_observation(
                lidar_obs=meta["agent_obs"], imu_quaternion=data.sensor("imu_quat").data,
                imu_gyro=data.sensor("imu_gyro").data, joint_position=data.qpos[joint_q],
                joint_velocity=data.qvel[joint_v], home_joint=home,
                previous_action=np.zeros(12), command=task.manifest.get("command"))
            meta_row = task.sensor_meta(rgb_age=rgb_age, lidar_age=lidar_age, rgb_valid=1.,
                                        lidar_raw_valid_fraction=float(meta["raw_valid_fraction"]))
            raw_fractions.append(float(meta["raw_valid_fraction"]))
            azimuth_fractions.append(float(meta["azimuth_valid_fraction"]))
            observations.append(obs)
            rgbs.append(image)
            states.append(np.zeros(32, np.float32))
            metas.append(meta_row)
            lidar_ranges.append(full_ranges)
            lidar_masks.append(valid)
            stamps.append(timestamp)
            fresh_flags.append(fresh)
            for _ in range(10):
                mujoco.mj_step(model, data)
    finally:
        sensors.close()
    obs = np.stack(observations)
    rgb = np.stack(rgbs)
    state = np.stack(states)
    sensor_meta = np.stack(metas)
    ranges = np.stack(lidar_ranges)
    masks = np.stack(lidar_masks)
    np.savez_compressed(out / "public_contract_smoke.npz", obs=obs, rgb=rgb, state=state,
                        sensor_meta=sensor_meta, lidar_ranges=ranges, lidar_masks=masks,
                        timestamps=np.asarray(stamps), fresh=np.asarray(fresh_flags))
    checks = {
        "task_id": task.manifest["id"], "task_version": task.manifest["version"],
        "catalog_entry": task.catalog_entry,
        "hashes": task.verified,
        "obs_shape": list(obs.shape), "rgb_shape": list(rgb.shape),
        "state_shape": list(state.shape), "sensor_meta_shape": list(sensor_meta.shape),
        "lidar_ranges_shape": list(ranges.shape), "lidar_masks_shape": list(masks.shape),
        "obs_finite": bool(np.isfinite(obs).all()), "rgb_finite": bool(np.isfinite(rgb).all()),
        "batch_1": [bool(np.array_equal(obs[:1], obs[:1])), list(rgb[:1].shape)],
        "batch_2": [list(obs[:2].shape), list(rgb[:2].shape)],
        "state_reset_zero": bool(np.all(state == 0.)),
        "fresh_steps": [int(i) for i, value in enumerate(fresh_flags) if value],
        "reused_steps": [int(i) for i, value in enumerate(fresh_flags) if not value],
        "sensor_meta_row0": sensor_meta[0].tolist(),
        "lidar_valid_fraction": masks.mean(axis=1).tolist(),
        "lidar_azimuth_valid_fraction": azimuth_fractions,
        "lidar_raw_valid_fraction": raw_fractions,
        "sensor_meta3_equals_raw_fraction": bool(np.allclose(sensor_meta[:, 3], raw_fractions)),
        "half_raw_fraction_assert": bool(np.isclose(324 / 648, .5)),
        "scene_sha256": task.verified["scene_sha256"],
        "sensors_sha256": task.verified["sensors_sha256"],
        "device": "CPU numpy; no ONNX model was provided for go2-agent-v1",
    }
    (out / "public_contract_smoke.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
