"""No learned weights: black/white football arc detection and calibrated geometry."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .config import Config
from .geometry import body_to_map, gravity_rotation
from .types import BallObservation, Frame, Pose


def circle_from_three(p: np.ndarray):
    a = 2 * (p[1:] - p[0])
    if abs(np.linalg.det(a)) < 1e-5:
        return None
    center = np.linalg.solve(a, (p[1:] ** 2).sum(axis=1) - (p[0] ** 2).sum())
    return center, float(np.linalg.norm(center - p[0]))


def ransac_arc(points: np.ndarray, seed: int = 0):
    if len(points) < 12:
        return None
    rng = np.random.default_rng(seed)
    # Preserve the same 100 sampled triples/tie order, but solve and score them
    # together. Thousands of tiny Python/linear-algebra calls dominated each
    # camera frame on the target laptop.
    triples = points[np.array([rng.choice(len(points), 3, replace=False) for _ in range(100)])]
    a = 2*(triples[:, 1:]-triples[:, :1])
    b = (triples[:, 1:]**2).sum(axis=2)-(triples[:, :1]**2).sum(axis=2)
    nonsingular = abs(np.linalg.det(a)) >= 1e-5
    if not nonsingular.any():
        return None
    centers = np.linalg.solve(a[nonsingular], b[nonsingular, :, None])[..., 0]
    radii = np.linalg.norm(centers-triples[nonsingular, 0], axis=1)
    eligible = (radii >= 5)&(radii <= 100)
    if not eligible.any():
        return None
    centers, radii = centers[eligible], radii[eligible]
    residual = abs(np.linalg.norm(points[None]-centers[:, None], axis=2)-radii[:, None])
    inliers = residual < np.maximum(1.5, radii*.05)[:, None]
    scores = inliers.sum(axis=1)
    winner = int(np.argmax(scores))
    if scores[winner] < 12:
        return None
    center, radius, valid = centers[winner], radii[winner], inliers[winner]
    selected = points[valid]
    fit = least_squares(lambda x: np.linalg.norm(selected - x[:2], axis=1) - x[2], [*center, radius], loss="soft_l1")
    return fit.x[:2], float(fit.x[2])


class Perception:
    def __init__(self, config: Config):
        self.c = config
        self.k = np.asarray(config.camera_matrix, dtype=float)
        self.t = np.asarray(config.body_from_camera, dtype=float)
        self.aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    def undistort(self, image):
        expected = tuple(self.c.camera_size)
        if (image.shape[1], image.shape[0]) != expected:
            raise ValueError(f"Image size differs from calibration {expected}")
        distortion=np.asarray(self.c.distortion)
        if not np.any(distortion):
            return image
        return cv2.undistort(image, self.k, distortion)

    def detect_balls(self, image: np.ndarray, small_targets=False) -> list[BallObservation]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Fiducial corners/patterns are strong false-circle distractors. Exclude
        # recognized markers before testing football texture and arc support.
        marker_corners, marker_ids, _ = self.aruco.detectMarkers(image)
        marker_rects = []
        if marker_ids is not None:
            gray = gray.copy()
            for corners in marker_corners:
                marker_rects.append((corners[0].min(axis=0) - 3, corners[0].max(axis=0) + 3))
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, corners[0].astype(np.int32), 255)
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                gray[mask > 0] = 112
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.)
        edges = cv2.Canny(blurred, 45, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates = []
        hough = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1., 12., param1=100, param2=8 if small_targets else 16, minRadius=5, maxRadius=90)
        if hough is not None:
            candidates.extend((p[:2], float(p[2])) for p in hough[0])
        for i, contour in enumerate(sorted(contours, key=len, reverse=True)[:35]):
            points = contour[:, 0, :].astype(float)
            if len(points) < 20:
                continue
            fit = ransac_arc(points[::max(1, len(points) // 300)], seed=i)
            if fit is not None:
                candidates.append(fit)
        output = []
        height, width = gray.shape
        yy, xx = np.ogrid[:height, :width]
        edge_y, edge_x = np.nonzero(edges)
        edge_points = np.column_stack([edge_x, edge_y])
        for center, radius in candidates:
            x, y = map(float, center)
            if any(np.all(center >= low) and np.all(center <= high) for low, high in marker_rects):
                continue
            if not 5 <= radius <= 90 or not (0 <= x < width and 0 <= y < height):
                continue
            ring = abs(np.linalg.norm(edge_points - center, axis=1) - radius) < max(1.8, radius * .06)
            on_arc = edge_points[ring] - center
            if len(on_arc) < 12:
                continue
            bins = np.floor((np.arctan2(on_arc[:, 1], on_arc[:, 0]) + np.pi) / (2 * np.pi) * 36).astype(int) % 36
            support = len(np.unique(bins)) / 36.
            interior = gray[(xx - x) ** 2 + (yy - y) ** 2 < (radius * .85) ** 2]
            if len(interior) < 25:
                continue
            # MuJoCo/real-camera exposure can put a white patch below 165.
            # Measure the two tones relative to robust local contrast while
            # retaining a minimum absolute contrast/variance requirement.
            low, high = np.quantile(interior, [.1, .9])
            contrast = float(high - low)
            dark = float((interior < low + .3 * contrast).mean())
            light = float((interior > low + .7 * contrast).mean())
            texture = min(1., float(interior.std()) / 60.)
            if dark < .08 or light < .08 or contrast < 55 or texture < .35 or support < .28:
                continue
            confidence = float(np.clip(.20 + .45 * support + .20 * texture + .15 * min(1., min(dark, light) * 4), 0., 1.))
            if confidence >= self.c.min_ball_confidence:
                output.append(BallObservation((x, y), radius, confidence, support))
        # Pixel-space NMS, retaining strongest geometric evidence.
        kept = []
        for detection in sorted(output, key=lambda d: d.confidence, reverse=True):
            if all(np.linalg.norm(np.array(detection.center) - k.center) > .8 * max(detection.radius, k.radius) for k in kept):
                kept.append(detection)
        return kept

    def locate_ball(self, detection: BallObservation, frame: Frame, pose: Pose) -> BallObservation:
        detection.position = None
        detection.range_source = "unknown"
        direction = np.linalg.solve(self.k, [*detection.center, 1.])
        level = gravity_rotation(frame.quaternion)
        ray = level @ self.t[:3, :3] @ direction
        origin = level @ self.t[:3, 3]
        # A rolling football lies on the local support plane. Its partial bounding
        # box width is deliberately NOT used to estimate distance.
        target_z = self.c.ball_radius - self.c.base_height
        if ray[2] >= -.025:
            return detection
        scale = (target_z - origin[2]) / ray[2]
        center = origin + scale * ray
        distance = np.linalg.norm(center[:2])
        if scale <= 0 or not .35 <= distance <= self.c.max_range:
            return detection
        # The ground-ray hypothesis determines depth independently of apparent
        # size. Use the known football radius only to reject inconsistent wall/
        # rock arcs, never to range from a partly occluded bounding-box width.
        expected_radius = .5 * (self.k[0, 0] + self.k[1, 1]) * self.c.ball_radius / scale
        if not .50 <= detection.radius / max(expected_radius, 1e-6) <= 1.65:
            return detection
        detection.sigma = float(max(.12, distance * .07 + pose.sigma))
        # Lidar checks are constrained to the predicted sphere neighbourhood,
        # rather than assigning an arbitrary background return inside the box.
        detection.range_source = "nominal_flat_ground"
        lidar = frame.points @ level.T
        close = lidar[np.linalg.norm(lidar - center, axis=1) < self.c.ball_radius + .10]
        if len(close) >= 6 and np.linalg.norm(np.std(close, axis=0)) > .035:
            fit = least_squares(lambda x: np.linalg.norm(close - x, axis=1) - self.c.ball_radius,
                                center, loss="soft_l1", f_scale=.02)
            rmse = np.sqrt(np.mean(fit.fun ** 2))
            if rmse < .035 and abs(fit.x[2] - target_z) < .06 and np.linalg.norm(fit.x - center) < .3:
                center, detection.sigma = fit.x, float(max(.07, rmse + pose.sigma))
                detection.range_source = "lidar_sphere"
        detection.position = body_to_map(center, pose)
        return detection

    def finish(self, image: np.ndarray, frame: Frame, pose: Pose) -> tuple[np.ndarray, np.ndarray] | None:
        corners, ids, _ = self.aruco.detectMarkers(image)
        if ids is None:
            return None
        size = self.c.finish_marker_size / 2
        object_points = np.array([[-size, size, 0], [size, size, 0], [size, -size, 0], [-size, -size, 0]], dtype=float)
        for i, marker in enumerate(ids.ravel()):
            if marker != self.c.finish_marker_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(object_points, corners[i][0].astype(float), self.k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or not np.isfinite(tvec).all() or not np.isfinite(rvec).all():
                ok, rvec, tvec = cv2.solvePnP(object_points, corners[i][0].astype(float), self.k, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not np.isfinite(tvec).all() or not np.isfinite(rvec).all() or tvec[2, 0] <= 0:
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.k, None)
            if np.linalg.norm(projected.reshape(-1, 2) - corners[i][0], axis=1).mean() > 2.:
                continue
            center = self.t[:3, :3] @ tvec.ravel() + self.t[:3, 3]
            center = gravity_rotation(frame.quaternion) @ center
            if np.linalg.norm(center[:2]) > self.c.max_range:
                continue
            marker_xy = body_to_map(center, pose)
            towards = marker_xy - pose.xy()
            norm = np.linalg.norm(towards)
            goal = marker_xy - towards / max(norm, 1e-6) * self.c.finish_standoff
            return marker_xy, goal
        return None
