"""Public Go2 task profile: manifest/sensors verification and contract observations.

This does not modify the legacy demo profile. It only turns a task directory
(reference-tasks/<id>) into a sensor/configuration contract and builds the
agent-visible tensors; no simulator ground truth enters the observation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PublicTask:
    task_dir: Path
    manifest: dict
    sensors: dict
    catalog_entry: dict | None = None
    verified: dict | None = None

    @classmethod
    def load(cls, task_dir, catalog=None, verify=True):
        task_dir = Path(task_dir).resolve()
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        sensors = json.loads((task_dir / "sensors.json").read_text(encoding="utf-8"))
        if (task_dir / manifest["scene"]).resolve().parent != task_dir:
            raise ValueError("scene path escapes task directory")
        if (task_dir / manifest["sensor_file"]).resolve().parent != task_dir:
            raise ValueError("sensor path escapes task directory")
        catalog_entry = None
        if catalog is not None:
            entries = json.loads(Path(catalog).read_text(encoding="utf-8"))
            catalog_entry = next((v for v in entries if v.get("id") == manifest.get("id")
                                  and str(v.get("version")) == str(manifest.get("version"))), None)
        task = cls(task_dir, manifest, sensors, catalog_entry)
        if verify:
            result = task.verify()
            if catalog is not None:
                if catalog_entry is None:
                    result["mismatches"].append("catalog_entry_missing")
                else:
                    result["catalog_sha256"] = sha256(task_dir / "manifest.json")
                    if result["catalog_sha256"] != catalog_entry.get("sha256"):
                        result["mismatches"].append("catalog_manifest_sha256")
            task.verified = result
            if result["mismatches"]:
                raise ValueError("public task verification failed: " + ", ".join(sorted(set(result["mismatches"]))))
        return task

    def verify(self):
        mismatches = []
        expected = dict(self.manifest.get("files", {}))
        for name, digest in expected.items():
            path = self.task_dir / name
            if not path.is_file():
                mismatches.append(name)
            elif sha256(path) != digest:
                mismatches.append(name)
        sensors_digest = expected.get("sensors.json")
        if sensors_digest and sha256(self.task_dir / self.manifest["sensor_file"]) != sensors_digest:
            mismatches.append("sensors.json")
        scene_digest = expected.get(self.manifest["scene"])
        if scene_digest and sha256(self.task_dir / self.manifest["scene"]) != scene_digest:
            mismatches.append(self.manifest["scene"])
        return {"files_checked": len(expected), "mismatches": sorted(set(mismatches)),
                "scene_sha256": sha256(self.task_dir / self.manifest["scene"]),
                "sensors_sha256": sha256(self.task_dir / self.manifest["sensor_file"])}

    @property
    def scene_path(self) -> Path:
        return self.task_dir / self.manifest["scene"]

    def body_from_camera(self) -> np.ndarray:
        camera = self.sensors["camera"]
        axes = np.asarray(camera["xyaxes"], float).reshape(2, 3)
        x_axis, y_axis = axes[0], -axes[1]
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        matrix = np.eye(4)
        matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = x_axis, y_axis, z_axis
        matrix[:3, 3] = np.asarray(camera["position_m"], float)
        return matrix

    def camera_matrix(self) -> list:
        camera = self.sensors["camera"]
        width, height = camera["width"], camera["height"]
        f = height / (2 * np.tan(np.deg2rad(camera["fovy_deg"]) / 2))
        return [[f, 0., width / 2], [0., f, height / 2], [0., 0., 1.]]

    def config(self) -> Config:
        camera = self.sensors["camera"]
        lidar = self.sensors["lidar"]
        config = Config()
        config.camera_size = [int(camera["width"]), int(camera["height"])]
        config.camera_matrix = self.camera_matrix()
        config.body_from_camera = self.body_from_camera().tolist()
        config.lidar_mount = list(lidar["position_m"])
        config.lidar_vertical_fov_deg = float(lidar["vertical_fov_deg"])
        config.lidar_elevation_center_deg = 0.
        config.lidar_noise_std = float(lidar.get("noise_std_m", 0.))
        config.lidar_dropout = float(lidar.get("dropout_probability", 0.))
        config.lidar_contract_order = True
        config.min_range = float(lidar["range_min_m"])
        config.max_range = float(lidar["range_max_m"])
        config.lidar_height_floor = float(self.manifest.get("lidar_min_relative_height_m", -.22))
        return config

    def initial_state(self):
        position = np.asarray(self.manifest["initial_position"], float)
        yaw = float(self.manifest["initial_yaw"])
        quaternion = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)  # wxyz
        return position, quaternion

    def build_observation(self, *, lidar_obs, imu_quaternion, imu_gyro,
                          joint_position, joint_velocity, home_joint,
                          previous_action=None, command=None) -> np.ndarray:
        gyro = np.asarray(imu_gyro, float).reshape(3)
        quat = np.asarray(imu_quaternion, float).reshape(4)
        rotation = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        gravity = rotation.T @ np.array([0., 0., -1.])
        command = np.zeros(3) if command is None else np.asarray(command, float).reshape(3)
        previous_action = (np.zeros(12) if previous_action is None
                           else np.asarray(previous_action, float).reshape(12))
        joint_position = np.asarray(joint_position, float).reshape(12)
        joint_velocity = np.asarray(joint_velocity, float).reshape(12)
        home_joint = np.asarray(home_joint, float).reshape(12)
        obs = np.concatenate([gyro * .25, gravity, command,
                              joint_position - home_joint, joint_velocity * .05,
                              previous_action, np.asarray(lidar_obs, float).reshape(144)])
        if obs.shape != (189,):
            raise ValueError(f"contract obs shape {obs.shape}, expected (189,)")
        return obs.astype(np.float32)

    def sensor_meta(self, *, rgb_age, lidar_age, rgb_valid, lidar_raw_valid_fraction,
                    imu_age=0., joint_age=0., imu_valid=1., joint_valid=1.) -> np.ndarray:
        return np.asarray([rgb_age, lidar_age, float(rgb_valid), lidar_raw_valid_fraction,
                           imu_age, joint_age, float(imu_valid), float(joint_valid)],
                          dtype=np.float32)
