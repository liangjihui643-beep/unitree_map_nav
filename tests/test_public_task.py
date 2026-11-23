import hashlib
import json

import numpy as np
import pytest

from go2_rescue.public_task import PublicTask


def build_task(root):
    sensors = {
        "schema_version": "go2-sensors/1", "id": "go2-scene-sensors", "version": "1",
        "camera": {"width": 320, "height": 240, "fovy_deg": 60,
                   "position_m": [.36, 0., .08], "xyaxes": [0, -1, 0, 0, 0, 1]},
        "lidar": {"horizontal_samples": 72, "vertical_channels": 9, "vertical_fov_deg": 96,
                  "range_min_m": .05, "range_max_m": 12, "position_m": [.32, 0., .05],
                  "period_s": .1, "noise_std_m": .01, "dropout_probability": .01},
    }
    scene = (root / "scene.mjb").write_bytes(b"scene")
    (root / "sensors.json").write_text(json.dumps(sensors), encoding="utf-8")
    files = {}
    for name in ("scene.mjb", "sensors.json", "NOTICE.md"):
        path = root / name
        if not path.exists():
            path.write_text("x", encoding="utf-8")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": "go2-scene-task/1", "id": "public-course-agent", "version": "1",
                "scene": "scene.mjb", "sensor_file": "sensors.json", "files": files,
                "initial_position": [.95, 1.15, .27], "initial_yaw": np.pi / 2,
                "command": [0., 0., 0.], "lidar_min_relative_height_m": -.22}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_public_task_verifies_and_builds_contract(tmp_path):
    task = PublicTask.load(build_task(tmp_path))
    assert task.verified["mismatches"] == []
    config = task.config()
    assert config.camera_size == [320, 240]
    assert np.isclose(config.camera_matrix[0][0], 207.846, atol=1e-3)
    assert np.allclose(config.lidar_mount, [.32, 0., .05])
    assert config.lidar_elevation_center_deg == 0.
    assert config.lidar_contract_order is True
    matrix = np.asarray(config.body_from_camera)
    assert np.allclose(matrix[:3, 3], [.36, 0., .08])
    assert np.allclose(matrix[:3, 2], [1., 0., 0.])


def test_public_task_builds_observation_and_meta(tmp_path):
    task = PublicTask.load(build_task(tmp_path))
    lidar = np.zeros(144, np.float32)
    lidar[72:] = 1.
    obs = task.build_observation(
        lidar_obs=lidar, imu_quaternion=[1., 0., 0., 0.], imu_gyro=[1., 2., 3.],
        joint_position=np.zeros(12), joint_velocity=np.zeros(12), home_joint=np.zeros(12))
    assert obs.shape == (189,)
    assert np.isclose(obs[0], .25)
    assert np.isclose(obs[5], -1.)
    meta = task.sensor_meta(rgb_age=.1, lidar_age=.2, rgb_valid=1.,
                            lidar_raw_valid_fraction=324 / 648)
    assert meta.shape == (8,)
    assert np.isclose(meta[3], .5)
    assert np.allclose(meta[:4], [.1, .2, 1., .5])


def test_public_task_reports_hash_mismatch(tmp_path):
    root = build_task(tmp_path)
    (root / "sensors.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sensors.json"):
        PublicTask.load(root)
