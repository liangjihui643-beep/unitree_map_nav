"""Bounded two-view metric positioning from classifier-accepted ball centres."""
from collections import deque
import cv2
import numpy as np

from .geometry import rot2


class BallTwoView:
    def __init__(self, camera_matrix, *, ball_radius=None, consensus=False):
        self.k = camera_matrix
        self.ball_radius = ball_radius
        self.consensus = consensus
        self.views = deque(maxlen=25)

    def update(self, detections, pose, base_z, rotation, origin, stamp, minimum_confidence):
        good = [d for d in detections if d.confidence >= minimum_confidence]
        if not good or not np.isfinite(base_z):
            return
        world = np.eye(3)
        world[:2, :2] = rot2(pose.yaw)
        r = world @ rotation
        o = np.r_[pose.xy(), base_z] + world @ origin
        p = self.k @ np.column_stack((r.T, -r.T @ o))
        xy = np.array([d.center for d in good])
        radii = np.array([d.radius for d in good])
        for i, d in enumerate(good):
            estimates = []
            for time, old_o, old_p, old_xy, old_radii in self.views:
                if stamp-time > 3.:
                    continue
                if not .15 <= np.linalg.norm(o[:2]-old_o[:2]) <= .8:
                    continue
                uv = np.repeat(xy[i, :, None], len(old_xy), axis=1)
                raw = cv2.triangulatePoints(old_p, p, old_xy.T, uv).T
                candidates = []
                for j, q in enumerate(raw):
                    if abs(q[3]) < 1e-8:
                        continue
                    hom = q/q[3]
                    a, b = old_p @ hom, p @ hom
                    if min(a[2], b[2]) <= .25 or np.linalg.norm(hom[:2]-o[:2]) > 5.:
                        continue
                    if max(np.linalg.norm(a[:2]/a[2]-old_xy[j]), np.linalg.norm(b[:2]/b[2]-xy[i])) > 1.5:
                        continue
                    if self.ball_radius is not None:
                        # Check apparent size after triangulation. This chooses
                        # associations without presuming the ball is on a floor.
                        expected = np.mean(np.diag(self.k)[:2])*self.ball_radius
                        ratio = np.array([old_radii[j]*a[2], radii[i]*b[2]])/expected
                        if np.any((ratio < .65) | (ratio > 1.45)):
                            continue
                    da, db = hom[:3]-old_o, hom[:3]-o
                    if np.dot(da, db) / (np.linalg.norm(da)*np.linalg.norm(db)) > np.cos(np.deg2rad(4)):
                        continue
                    candidates.append(hom[:3])
                if len(candidates) == 1 or self.consensus:
                    estimates.extend((old_o, candidate) for candidate in candidates)
            if self.consensus and estimates:
                # Keep ambiguous pairs until independent viewpoints agree.
                anchors = np.array([o for o, _ in estimates])
                values = np.array([p for _, p in estimates])
                bins = np.floor(anchors[:, :2] / .075).astype(int)
                clusters = []
                for point in values:
                    close = np.linalg.norm(values-point, axis=1) <= .11
                    _, independent = np.unique(bins[close], axis=0, return_index=True)
                    if len(independent) < 2:
                        continue
                    witnesses = anchors[close][independent, :2]
                    if np.linalg.norm(np.ptp(witnesses, axis=0)) < .10:
                        continue
                    clusters.append((len(independent), np.median(values[close][independent], axis=0)))
                if clusters:
                    best, point = max(clusters, key=lambda pair: pair[0])
                    ambiguous = any(n >= best and np.linalg.norm(other-point) > .17
                                    for n, other in clusters)
                    if not ambiguous:
                        d.position = point[:2]
                        d.sigma = float(np.hypot(.10, pose.sigma))
                        d.range_source = "visual_multiview_consensus"
                continue
            # Two separated previous views must agree; a single epipolar pair
            # can silently associate two different balls.
            for j, (first, point) in enumerate(estimates):
                agree = [other for second, other in estimates[j+1:]
                         if np.linalg.norm(first[:2]-second[:2]) >= .07
                         and np.linalg.norm(point-other) <= .12]
                if agree:
                    d.position = np.median(np.vstack((point, agree)), axis=0)[:2]
                    d.sigma = float(np.hypot(.10, pose.sigma))
                    d.range_source = "visual_multiview"
                    break
        if not self.views or np.linalg.norm(self.views[-1][1][:2]-o[:2]) >= .075:
            self.views.append((stamp, o, p, xy, radii))
def fuse_target_position(detection, single, *, agreement=.20):
    """Use a corroborated single-frame result, or the independent 3-view rescue."""
    if detection.position is None:
        # Do not promote an uncorroborated floor-plane guess in slope mode.
        return
    if (single.position is not None and single.sigma <= .20
            and np.linalg.norm(detection.position-single.position) <= agreement):
        detection.position = single.position.copy()
        detection.sigma = min(detection.sigma, single.sigma)
        detection.range_source = "visual_multiview_corroborated_plane"
