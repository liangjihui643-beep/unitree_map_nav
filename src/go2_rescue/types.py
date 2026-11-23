from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.wz], dtype=float)


@dataclass
class Frame:
    """One synchronized observation. All vectors use body FLU, metres/seconds.

    timestamp is monotonic seconds in the recording/source time domain.
    points are body-frame XYZ, never map points or simulator ground truth.
    quaternion is body orientation in an IMU reference frame, wxyz.
    The controller uses roll/pitch only, so absolute IMU yaw is not map yaw.
    """

    timestamp: float
    points: np.ndarray
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    velocity: Velocity = field(default_factory=Velocity)
    image: np.ndarray | None = None
    point_times: np.ndarray | None = None
    state_age: float = 0.0
    image_age: float = 0.0
    support_points: np.ndarray | None = None  # body FLU contact soles from joint/IMU kinematics

    def validate(self) -> None:
        if self.support_points is not None:
            self.support_points = np.asarray(self.support_points, dtype=float)
            if (self.support_points.ndim != 2 or self.support_points.shape[1] != 3
                    or len(self.support_points) > 4 or not np.isfinite(self.support_points).all()):
                raise ValueError("support_points must contain up to four finite body-frame soles")
        if not math.isfinite(self.timestamp):
            raise ValueError("Non-finite sensor timestamp")
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be Nx3 body-frame XYZ")
        if len(self.points) > 250_000:
            raise ValueError("Point cloud exceeds frame size limit")
        finite = np.isfinite(self.points).all(axis=1)
        if self.point_times is not None:
            self.point_times = np.asarray(self.point_times, dtype=float)
            if self.point_times.shape != (len(self.points),) or not np.isfinite(self.point_times).all():
                raise ValueError("point_times must be finite and match points")
            self.point_times = self.point_times[finite]
        self.points = self.points[finite]
        q = np.asarray(self.quaternion, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
            raise ValueError("Invalid wxyz quaternion")
        self.quaternion = q / np.linalg.norm(q)
        if not np.isfinite(self.velocity.array()).all():
            raise ValueError("Invalid measured body velocity")
        if not np.isfinite([self.state_age, self.image_age]).all() or min(self.state_age, self.image_age) < 0:
            raise ValueError("Invalid sensor ages")


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    quality: float = 0.0
    sigma: float = 1.0

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class BallObservation:
    center: tuple[float, float]
    radius: float
    confidence: float
    arc_support: float
    position: np.ndarray | None = None
    sigma: float = 1.0
    range_source: str = "unknown"


@dataclass
class Event:
    kind: str
    timestamp: float
    payload: dict = field(default_factory=dict)


@dataclass
class Step:
    velocity: Velocity
    pose: Pose
    state: str
    events: list[Event]
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
