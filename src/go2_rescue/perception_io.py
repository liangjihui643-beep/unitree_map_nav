"""Project-local perception handoff and public observation input adapter."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial.transform import Rotation

from .lidar_obs import RangeOnlyEvidence, decode_lidar_obs


@dataclass(frozen=True)
class PublicSensorStep:
    time_s: float
    lidar_time_s: float
    rgb_time_s: float
    lidar_obs: np.ndarray | None
    image_bgr: np.ndarray | None
    gravity_body: np.ndarray
    roll_pitch_wxyz: np.ndarray
    gyro_rad_s: np.ndarray
    joint_delta_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    raw_lidar_valid_fraction: float
    imu_valid: bool
    joint_valid: bool


class PublicSensorStream:
    """Decode one public task episode without fabricating XYZ, yaw or velocity."""

    def __init__(self, config):
        self.config = config
        self.reset()

    @property
    def camera_matrix_64(self):
        """Intrinsics after the public contract's pixel-centre resize."""
        sx, sy = 64. / np.asarray(self.config.camera_size, dtype=float)
        resize = np.array([[sx, 0., (sx - 1.) / 2.],
                           [0., sy, (sy - 1.) / 2.],
                           [0., 0., 1.]])
        return resize @ np.asarray(self.config.camera_matrix, dtype=float)

    def reset(self):
        self._time = self._lidar_stamp = self._rgb_stamp = -np.inf

    def decode(self, timestamp, obs, rgb, sensor_meta):
        time_s = float(timestamp)
        obs, rgb, meta = (np.asarray(value, dtype=float) for value in (obs, rgb, sensor_meta))
        if (obs.shape != (189,) or rgb.shape != (3, 64, 64) or meta.shape != (8,)
                or not (np.isfinite(obs).all() and np.isfinite(rgb).all() and np.isfinite(meta).all())):
            raise ValueError("expected finite public obs[189], rgb[3,64,64], sensor_meta[8]")
        if not np.isfinite(time_s) or time_s < 0 or time_s <= self._time:
            raise ValueError("control time must increase within an episode; reset at episode start")
        if ((meta[[0, 1, 4, 5]] < 0).any()
                or not np.isin(meta[[2, 6, 7]], (0, 1)).all()
                or not 0 <= meta[3] <= 1 or (rgb < 0).any() or (rgb > 1).any()):
            raise ValueError("invalid public sensor ages, flags or RGB range")
        packet = obs[45:189]
        decode_lidar_obs(packet, self.config.min_range, self.config.max_range)
        gravity = obs[3:6].copy()
        magnitude = np.linalg.norm(gravity)
        if meta[6] and not np.isclose(magnitude, 1., atol=1e-3):
            raise ValueError("valid IMU requires unit projected gravity")
        if meta[6]:
            x, y, z = -gravity / magnitude
            pitch = np.arctan2(-x, np.hypot(y, z))
            roll = np.arctan2(y, z)
            quaternion = np.roll(Rotation.from_euler("xyz", [roll, pitch, 0.]).as_quat(), 1)
        else:
            quaternion = np.array([1., 0., 0., 0.])

        rgb_stamp, lidar_stamp = time_s - meta[:2]
        tolerance = 1e-6  # cached frame age is encoded as float32
        if min(rgb_stamp, lidar_stamp) < -tolerance or lidar_stamp < self._lidar_stamp - tolerance:
            raise ValueError("sensor timestamps must be nonnegative and nondecreasing")
        if meta[2] and rgb_stamp < self._rgb_stamp - tolerance:
            raise ValueError("valid RGB timestamps must not move backward")
        new_lidar = lidar_stamp > self._lidar_stamp + tolerance
        new_rgb = bool(meta[2]) and rgb_stamp > self._rgb_stamp + tolerance

        self._time = time_s
        if new_lidar:
            self._lidar_stamp = lidar_stamp
        if new_rgb:
            self._rgb_stamp = rgb_stamp
        image = np.rint(rgb[[2, 1, 0]].transpose(1, 2, 0) * 255).astype(np.uint8) if new_rgb else None
        return PublicSensorStep(time_s, float(lidar_stamp), float(rgb_stamp),
                                packet.astype(np.float32) if new_lidar else None, image, gravity,
                                quaternion, obs[:3] * 4., obs[9:21].copy(), obs[21:33] * 20.,
                                float(meta[3]), bool(meta[6]), bool(meta[7]))


def public_range_evidence(grid):
    """Create a diagnostic overlay for the compressed input on this grid geometry."""
    config = grid.c
    return RangeOnlyEvidence(grid, config.lidar_elevation_center_deg, config.lidar_vertical_fov_deg,
                             height_floor_m=config.lidar_height_floor, origin_body=config.lidar_mount)


def perception_snapshot(grid, terrain, confirmed_positions, timestamp, *, detections=None):
    """One complete XYZ-pipeline snapshot in map_start coordinates."""
    occupied = grid.logodds > .5
    free = grid.seen & (grid.logodds < 0.)
    return {
        "schema": "go2-perception-snapshot/1",
        "frame": "map_start",
        "time_s": float(timestamp),
        "lidar_input": "body_flu_xyz",
        "map": {
            "origin_xy_m": grid.origin.tolist(),
            "resolution_m": float(grid.c.resolution),
            "shape": list(occupied.shape),
            "occupied_ij": np.argwhere(occupied).tolist(),
            "ray_free_ij": np.argwhere(free).tolist(),
            "traversable_ij": np.argwhere(grid.traversable()).tolist(),
        },
        "terrain": terrain,
        "confirmed_target_xy_m": [np.asarray(xy, dtype=float).tolist() for xy in confirmed_positions],
        "target_observation": detections,
    }


def publish_perception_snapshot(path: Path, snapshot: dict):
    """Publish by atomic replacement so a polling reader always gets one version."""
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
