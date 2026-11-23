"""Metric odometry from near-ground RGB correspondences and measured stance.

Simulator XYZ returns are deliberately discarded before vision processing.
The velocity prior bridges camera gaps; neither source provides loop closure.
"""
from dataclasses import replace
import cv2
import numpy as np

from .geometry import gravity_rotation, rot2, wrap
from .localization import Localization
from .stance_surface import StanceSurface
from .visual_odometry import MetricVisualOdometry


class GroundMaskedOpticalOdometry(MetricVisualOdometry):
    """Allocate feature budget to nearby ground, not higher-contrast walls."""

    def depths(self, frame, plane, probability, gray):
        empty = (np.empty((0, 2), np.float32), np.empty((0, 3)))
        if plane is None:
            return empty
        h, w = gray.shape
        mask = cv2.erode(((probability.argmax(axis=0) == 0) &
                          (probability[0] >= .75)).astype(np.uint8),
                         np.ones((5, 5), np.uint8))
        level = gravity_rotation(frame.quaternion)
        origin = level @ self.mount[:3, 3]
        y, x = np.indices((h, w), dtype=np.float32)
        directions = np.stack(((x-self.k[0, 2])/self.k[0, 0],
                               (y-self.k[1, 2])/self.k[1, 1],
                               np.ones_like(x)), axis=-1)
        ground_rays = directions @ (level @ self.mount[:3, :3]).T
        denominator = ground_rays[..., 2] - ground_rays[..., :2] @ plane[:2]
        dz = plane[2] + origin[:2] @ plane[:2] - origin[2]
        depth = np.divide(dz, denominator, out=np.full((h, w), np.inf),
                          where=denominator < -.15)
        mask[(depth < .30) | (depth > 1.45)] = 0
        features = cv2.goodFeaturesToTrack(gray, 220, .006, 5, mask=mask, blockSize=5)
        if features is None:
            return empty
        xy = features[:, 0, :]
        ij = np.rint(xy).astype(int)
        z = depth[ij[:, 1], ij[:, 0]]
        valid = np.isfinite(z) & (z >= .30) & (z <= 1.45)
        rays = np.column_stack(((xy[:, 0]-self.k[0, 2])/self.k[0, 0],
                               (xy[:, 1]-self.k[1, 2])/self.k[1, 1],
                               np.ones(len(xy))))
        return xy[valid], rays[valid] * z[valid, None]


class VisualStanceOdometry:
    def __init__(self, config, segmenter, *, visual_weight=.65, retain_keyframe=False, ground_first=False, along_prior=False):
        self.vo = (GroundMaskedOpticalOdometry if ground_first else MetricVisualOdometry)(config)
        self.segmenter = segmenter
        self.surface = StanceSurface()
        self.visual_weight = visual_weight
        self.retain_keyframe = retain_keyframe
        self.along_prior = along_prior
        self.xyz = np.array([0., 0., config.base_height])
        self.stamp = self.heading0 = self.camera = self.baseline_height = None
        self.status = {}

    def update(self, frame):
        yaw = Localization._imu_yaw(frame.quaternion)
        if self.heading0 is None:
            self.heading0 = yaw
        h = wrap(yaw - self.heading0)
        dt = 0. if self.stamp is None else frame.timestamp - self.stamp
        self.stamp = frame.timestamp
        plane = self.surface.update(frame)
        height = -plane[2] if plane is not None else None
        if height is not None and self.baseline_height is None:
            self.baseline_height = height
        if 0. < dt <= .5:
            local = frame.velocity.array()[:2] * dt
            self.xyz[:2] += rot2(h) @ local
            if plane is not None:
                self.xyz[2] += plane[:2] @ local + height - self.baseline_height
        if height is not None:
            self.baseline_height = height
        self.status = {"visual_used": False, "stance": self.surface.status.get("source")}
        if frame.image is None:
            return self.xyz.copy(), h
        isolated = replace(frame, points=np.empty((0, 3)), point_times=None)
        previous_visual, previous_camera = self.vo.previous, self.camera
        delta = self.vo.update(isolated, plane, self.segmenter.segment(frame.image))
        self.status.update(self.vo.status)
        if self.status.get("source") == "lidar_scaled_visual_imu":
            self.status["source"] = "stance_scaled_visual_imu"
        if delta is not None and self.camera is not None:
            stamp, origin, heading = self.camera
            if abs(stamp - delta["from_stamp"]) < 1e-5:
                translation = np.asarray(delta["delta_gravity"])
                estimate = origin[:2] + rot2(heading) @ translation[:2]
                innovation = estimate - self.xyz[:2]
                if self.along_prior:
                    predicted_motion = self.xyz[:2] - origin[:2]
                    norm = np.linalg.norm(predicted_motion)
                    if norm < .006:
                        innovation = np.full(2, np.inf)
                    else:
                        direction = predicted_motion / norm
                        correction = innovation @ direction
                        cross = np.linalg.norm(innovation - correction*direction)
                        # Diagnostic ablation: prevent correlated sideways
                        # errors and bound scale corrections to 15 percent.
                        if cross > .008:
                            innovation = np.full(2, np.inf)
                        else:
                            innovation = direction*np.clip(correction, -.15*norm, .15*norm)
                # Remote slopes are not allowed to inject arbitrary changes.
                if np.linalg.norm(innovation) <= .12:
                    self.xyz[:2] += self.visual_weight * innovation
                    self.status["visual_used"] = True
        self.camera = (frame.timestamp, self.xyz.copy(), h)
        if (self.retain_keyframe and not self.status["visual_used"]
                and previous_visual is not None and previous_camera is not None
                and len(previous_visual["uv"]) >= 15
                and 0. < frame.timestamp-previous_camera[0] < 1.05
                and np.linalg.norm(self.xyz[:2]-previous_camera[1][:2]) < .22):
            self.vo.previous, self.camera = previous_visual, previous_camera
            self.status["retained_reference"] = True
        return self.xyz.copy(), h
