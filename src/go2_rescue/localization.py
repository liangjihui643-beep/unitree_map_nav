"""Classical lidar odometry, conservative loop detection and a planar pose graph.

The planar backend targets floor-level navigation, not arbitrary 6-DoF climbing.
KISS-ICP is available for 3-D odometry; both expose the same quality gate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .config import Config
from .geometry import gravity_rotation, rot2, transform_xy, wrap
from .types import Frame, Pose
from .terrain import ground_surface


def register(source: np.ndarray, target: np.ndarray, guess: Pose, limit: float = .65):
    """Trimmed point-to-point SE(2) ICP, source -> target."""
    if len(source) < 12 or len(target) < 12:
        return guess, 0.0, float("inf")
    tree = cKDTree(target[:, :2])
    r, t = rot2(guess.yaw), guess.xy()
    for _ in range(24):
        current = source[:, :2] @ r.T + t
        distance, indices = tree.query(current)
        valid = distance < limit
        if valid.sum() < 12:
            return guess, 0.0, float("inf")
        threshold = max(.035, np.quantile(distance[valid], .85))
        valid &= distance <= threshold
        a, b = current[valid], target[indices[valid], :2]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        dr = vt.T @ u.T
        if np.linalg.det(dr) < 0:
            vt[-1] *= -1
            dr = vt.T @ u.T
        dt = cb - dr @ ca
        r, t = dr @ r, dr @ t + dt
        if np.linalg.norm(dt) < 1e-4 and abs(np.arctan2(dr[1, 0], dr[0, 0])) < 1e-4:
            break
    distance, _ = tree.query(source[:, :2] @ r.T + t)
    inliers = distance < min(limit, .25)
    overlap = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(distance[inliers] ** 2))) if inliers.any() else float("inf")
    pose = Pose(float(t[0]), float(t[1]), float(np.arctan2(r[1, 0], r[0, 0])))
    return pose, overlap, rmse


def relative(a: Pose, b: Pose) -> np.ndarray:
    xy = rot2(a.yaw).T @ (b.xy() - a.xy())
    return np.array([*xy, wrap(b.yaw - a.yaw)])


def line_normals(target):
    tree = cKDTree(target)
    _, neighbours = tree.query(target, k=min(7, len(target)))
    patches = target[neighbours]
    centered = patches-patches.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    line = (eigenvalues[:, 1] > .001)&(eigenvalues[:, 0] < .22*eigenvalues[:, 1])
    return tree, eigenvectors[:, :, 0], line


def register_lines(source, target, guess, limit=.65, yaw_hint=None, motion_prior_weight=.7,
                   sparse_geometry=False):
    """Robust point-to-line ICP with a weak measured-motion prior.

    Along a featureless wall, point-to-point correspondences artificially pin
    the robot to the wall's sampling pattern. Line normals leave that direction
    unobserved, so measured joint/IMU motion supplies it until corners reappear.
    """
    if len(source) < 12 or len(target) < 12:
        return guess, 0., float("inf")
    target = target[:, :2]
    tree, normals, line = line_normals(target)
    if line.sum() < 12:
        if sparse_geometry:
            return guess, 0., float("inf")
        return register(source, target, guess, limit)
    minimum = 8 if sparse_geometry else 12
    if sparse_geometry:
        target, normals = target[line], normals[line]
        tree = cKDTree(target)
        line = np.ones(len(target), bool)
    prior = np.array([guess.x, guess.y, guess.yaw])
    if yaw_hint is not None:
        prior[2] = yaw_hint
    value = prior.copy()
    regularization = np.array([motion_prior_weight, motion_prior_weight, 2.])
    if yaw_hint is not None:
        regularization[2] = 1./.025**2
    for _ in range(18):
        rotated = source[:, :2]@rot2(value[2]).T
        current = rotated+value[:2]
        distance, indices = tree.query(current)
        normal = normals[indices]
        residual = np.sum(normal*(current-target[indices]), axis=1)
        valid = (distance < limit)&line[indices]&(abs(residual) < .25)
        if valid.sum() < minimum:
            return guess, 0., float("inf")
        n, r, q = normal[valid], residual[valid], rotated[valid]
        jacobian = np.column_stack([n, -n[:, 0]*q[:, 1]+n[:, 1]*q[:, 0]])
        weight = np.minimum(1., .04/np.maximum(abs(r), 1e-9))
        hessian = jacobian.T@(jacobian*weight[:, None])+np.diag(regularization)
        offset = value-prior; offset[2] = wrap(offset[2])
        gradient = jacobian.T@(weight*r)+regularization*offset
        delta = -np.linalg.solve(hessian, gradient)
        delta[:2] = np.clip(delta[:2], -.15, .15)
        delta[2] = np.clip(delta[2], -.15, .15)
        value += delta
        if np.linalg.norm(delta) < 1e-5:
            break
    distance, indices = tree.query(source[:, :2]@rot2(value[2]).T+value[:2])
    normal = normals[indices]
    residual = np.sum(normal*(source[:, :2]@rot2(value[2]).T+value[:2]-target[indices]), axis=1)
    inliers = (distance < .4)&(abs(residual) < .12)&line[indices]
    support = (distance < limit)&line[indices]
    # Judge residual agreement on usable line correspondences. Horizontal or
    # irregular surfaces are deliberately outside this metric, not failed wall
    # matches. Retain a coverage penalty so a tiny matching subset cannot make
    # an otherwise unrelated scan look reliable.
    coverage = min(1., float(support.mean())/.60)
    overlap = float(inliers.sum()/max(1, support.sum())*coverage) if inliers.sum() >= minimum else 0.
    rmse = float(np.sqrt(np.mean(residual[inliers]**2))) if inliers.any() else float("inf")
    if sparse_geometry and overlap:
        # A visually sparse scan must still constrain both horizontal axes.
        # Only accept compact corrections backed by precise, nonparallel lines.
        supported_normals = normal[inliers]
        observability = np.linalg.eigvalsh(supported_normals.T@supported_normals)
        if observability[0] < 1.5 or rmse > .055 or np.linalg.norm(value[:2]-prior[:2]) > .07:
            return guess, 0., float("inf")
    return Pose(*value), overlap, rmse


@dataclass
class Keyframe:
    pose: Pose
    points: np.ndarray
    scan: np.ndarray
    timestamp: float | None = None
    quaternion: np.ndarray | None = None


class PoseGraph:
    def __init__(self):
        self.nodes: list[Pose] = []
        self.edges: list[tuple[int, int, np.ndarray, float]] = []

    def optimize(self, heading_priors=None) -> list[Pose]:
        if len(self.nodes) < 3:
            return self.nodes
        anchor = np.array([self.nodes[0].x, self.nodes[0].y, self.nodes[0].yaw])
        initial = np.array([[p.x, p.y, p.yaw] for p in self.nodes[1:]]).ravel()

        def residual(x):
            values = np.vstack([anchor, x.reshape(-1, 3)])
            result = []
            for i, j, measurement, weight in self.edges:
                a, b = values[i], values[j]
                delta = rot2(a[2]).T @ (b[:2] - a[:2])
                error = np.array([*(delta - measurement[:2]), wrap(b[2] - a[2] - measurement[2])])
                result.extend(error * np.array([1., 1., 2.]) * weight)
            if heading_priors is not None:
                result.extend([wrap(v[2]-heading)/.025 for v, heading in zip(values[1:], heading_priors[1:])])
            return result

        result = least_squares(residual, initial, loss="soft_l1", f_scale=.1, max_nfev=60)
        if result.success and np.isfinite(result.x).all():
            self.nodes = [Pose(*v, quality=p.quality, sigma=p.sigma) for v, p in zip(
                np.vstack([anchor, result.x.reshape(-1, 3)]), self.nodes)]
        return self.nodes


def coarse_submap_seed(scan, target, guess, fixed_heading):
    """Evaluate a bounded local distance field around measured odometry."""
    shifts = np.array([(x, y) for x in (-.10, 0., .10) for y in (-.10, 0., .10)])
    angles = [guess.yaw] if fixed_heading else [guess.yaw-.06, guess.yaw, guess.yaw+.06]
    tree = cKDTree(target)
    scored = []
    for yaw in angles:
        xy = scan[:, :2] @ rot2(yaw).T + guess.xy()
        distances, _ = tree.query(xy[None] + shifts[:, None])
        costs = np.minimum(distances, .25).mean(axis=1) + .18*np.linalg.norm(shifts, axis=1)
        scored.extend((float(cost), shift, yaw) for cost, shift in zip(costs, shifts))
    scored.sort(key=lambda option: option[0])
    best, shift, yaw = scored[0]
    margin = scored[1][0] - best
    # Flat scores in a corridor leave the measured-motion prior in charge.
    return (Pose(*(guess.xy()+shift), yaw) if margin >= .008 else guess), best, margin


class Localization:
    public_reference_radius = 6.
    use_sparse_geometry_fallback = True

    def __init__(self, config: Config):
        self.c = config
        self.pose = Pose()
        self.last_timestamp: float | None = None
        self.keyframes: list[Keyframe] = []
        self.graph = PoseGraph()
        self.revision = 0
        self.corrections: list[tuple[Pose, Pose]] = []
        self.kiss = None
        self.heading_origin = None
        self.keyframe_headings = []
        self.sensor_heading = None
        self.use_local_submap = False
        self.last_registration = {}
        self.public_reference = None
        self.portal_reference = None
        self.declared_start_pose = Pose()
        self.recent_structural_scans = deque(maxlen=16)
        self.scan_odometry_xy = np.zeros(2)
        if config.navigation_prior:
            from pathlib import Path
            path=Path(__file__).resolve().parents[2]/config.navigation_prior
            with np.load(path) as prior:
                self.public_reference=prior['reference_xy'].copy()
            if config.navigation_surface_reference:
                from .portal_localization import SurfaceReference
                path = Path(__file__).resolve().parents[2]/config.navigation_surface_reference
                with np.load(path, allow_pickle=False) as surface:
                    self.portal_reference = SurfaceReference(surface["points"], surface["normals"])
        if config.localization_backend == "kiss":
            from kiss_icp.config import KISSConfig
            from kiss_icp.kiss_icp import KissICP
            settings = KISSConfig()
            settings.data.max_range = config.max_range
            settings.data.min_range = .25
            settings.data.deskew = False  # Enabled per frame only with real point times.
            settings.mapping.voxel_size = .12
            self.kiss = KissICP(settings)

    def accumulated_public_fit(self, scan, target, guess, stamp, kwargs):
        """Fit a bounded, motion-compensated window during a map-match loss."""
        scans = [scan]
        for t, xy, heading, points in reversed(self.recent_structural_scans):
            if stamp-t > 1.2:
                break
            displacement = xy-self.scan_odometry_xy
            distance = np.linalg.norm(displacement)
            if distance > .24 or (distance < .035 and abs(wrap(heading-guess.yaw)) < .07):
                continue
            transformed = points.copy()
            transformed[:, :2] = (points[:, :2] @ rot2(heading).T + displacement) @ rot2(guess.yaw)
            scans.append(transformed)
        if len(scans) < 2:
            return None
        combined = np.vstack(scans)
        _, unique = np.unique(np.floor(combined[:, :2]/.07).astype(int), axis=0, return_index=True)
        fit, overlap, rmse = register_lines(combined[unique], target, guess, **kwargs)
        if (overlap*np.exp(-rmse/.20) < self.c.min_pose_quality or rmse > .06
                or np.linalg.norm(fit.xy()-guess.xy()) > .10 or abs(wrap(fit.yaw-guess.yaw)) > .10):
            return None
        tree, normals, line = line_normals(target)
        current = transform_xy(scan, fit)
        distances, indices = tree.query(current)
        residual = np.sum(normals[indices]*(current-target[indices]), axis=1)
        agrees = (distances < .25) & line[indices] & (abs(residual) < .08)
        if agrees.sum() < max(6, int(np.ceil(.25*len(scan)))):
            return None
        self.last_registration.update(accumulated_frames=len(scans), accumulated_points=len(unique),
                                      current_scan_agreement=int(agrees.sum()))
        return fit, overlap, rmse

    def structural_scan(self, points, pose=None):
        pose = self.pose if pose is None else pose
        limit = lambda xy: self.c.slope_limit(xy @ rot2(pose.yaw).T+pose.xy())
        samples = ground_surface(points + [0., 0., self.c.base_height], 0.,
                                 self.c.resolution, self.c.ground_max_gap, limit)
        ground = np.zeros(len(points), bool)
        if len(samples):
            distance, indices = cKDTree(samples[:, :2]).query(points[:, :2])
            ground = (distance < self.c.resolution) & (abs(points[:, 2] + self.c.base_height - samples[indices, 2]) < .04)
        if self.c.navigation_support:
            plane=getattr(self,"support_plane",None)
            if plane is not None:
                ground |= abs(points[:,2]-(points[:,:2]@plane[:2]+plane[2]))<.06
            if getattr(self,"ground_filter",None) is not None:
                ground |= self.ground_filter(points)
        scan = points[(~ground) & (points[:, 2] > -.22) & (points[:, 2] < .75)]
        if self.c.localization_metric == "point_to_line" and len(scan) >= 12:
            # Horizontal overhead planes have no stable XY correspondence for
            # planar odometry. Keep them in the returned cloud for clearance,
            # but do not let ceiling sampling drown out vertical wall matches.
            _, indices = cKDTree(scan).query(scan, k=min(10, len(scan)))
            patches = scan[indices]
            centered = patches-patches.mean(axis=1, keepdims=True)
            values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
            horizontal = ((scan[:, 2] > .16)&(abs(vectors[:, 2, 0]) > .90)
                          &(values[:, 1] > .0005)&(values[:, 0] < .20*values[:, 1]))
            scan = scan[~horizontal]
        if len(scan):
            _, indices = np.unique(np.floor(scan[:, :2]/.07).astype(int), axis=0, return_index=True)
            scan = scan[indices]
        return scan

    def update(self, frame: Frame) -> tuple[Pose, np.ndarray]:
        frame.validate()
        self._last_quaternion = frame.quaternion
        self.corrections = []
        points = frame.points @ gravity_rotation(frame.quaternion).T
        ranges = np.linalg.norm(points, axis=1)
        points = points[(ranges > .25) & (ranges < self.c.max_range)]
        scan = self.structural_scan(points)
        self.last_registration = {"structural_points": len(scan)}
        if self.last_timestamp is not None and frame.timestamp <= self.last_timestamp:
            raise ValueError("Sensor timestamps must increase strictly")
        dt = 0. if self.last_timestamp is None else frame.timestamp - self.last_timestamp
        self.last_timestamp = frame.timestamp
        if self.c.imu_heading_reference and self.heading_origin is not None:
            self.sensor_heading = wrap(self._imu_yaw(frame.quaternion)-self.heading_origin)
        if self.public_reference is not None and self.c.experimental_scan_window:
            if dt > self.c.max_frame_gap:
                self.recent_structural_scans.clear()
            elif self.heading_origin is not None:
                heading = wrap(self._imu_yaw(frame.quaternion)-self.heading_origin)
                self.scan_odometry_xy += rot2(heading) @ (frame.velocity.array()[:2]*dt)
        plane = getattr(self, "support_plane", None)
        portal_available = (self.portal_reference is not None and self.heading_origin is not None
                            and 10.25 < self.pose.x < 11.15 and 4.65 < self.pose.y < 6.2
                            and plane is not None and .26 < -plane[2] < .39
                            and np.linalg.norm(plane[:2]) < .08 and len(points) >= 12)
        if (len(scan) < self.c.min_points and not portal_available) or dt > self.c.max_frame_gap:
            hint=getattr(self,"visual_prior",None)
            if self.c.navigation_support and hint is not None and self.heading_origin is not None and dt<=self.c.max_frame_gap:
                yaw=wrap(self._imu_yaw(frame.quaternion)-self.heading_origin)
                self.pose=Pose(*hint,yaw,quality=.65,sigma=min(.3,self.pose.sigma+.02))
                self.last_registration['source']='visual_imu_without_scan'
                return self.pose,points
            self.pose.quality = 0.
            return self.pose, points
        spread = np.linalg.eigvalsh(np.cov(scan[:, :2].T)) if len(scan) > 2 else np.zeros(2)
        geometry = float(np.clip(spread[0] / max(.12, spread[-1] * .08), 0., 1.))
        if not self.keyframes:
            start = self.declared_start_pose
            self.heading_origin = self._imu_yaw(frame.quaternion)-start.yaw
            self.sensor_heading = start.yaw
            # The initial relative origin is defined here, not inferred from
            # scan spread. Degeneracy matters for subsequent motion only.
            self.pose = Pose(start.x,start.y,start.yaw,quality=.8 if self.c.navigation_support else .8 * geometry, sigma=.08)
            if self.kiss:
                self.kiss.register_frame(points, np.zeros(len(points)))
            self._keyframe(points)
            return self.pose, points
        old = self.pose
        displacement = rot2(old.yaw) @ (frame.velocity.array()[:2] * dt)
        guess = Pose(*(old.xy() + displacement), wrap(old.yaw + frame.velocity.wz * dt))
        if self.c.imu_heading_reference:
            guess.yaw = self.sensor_heading
        visual_hint=getattr(self,"visual_prior",None) if self.c.navigation_support else None
        if visual_hint is not None:
            guess.x,guess.y=.85*np.asarray(visual_hint)+.15*guess.xy()
            self.last_registration['visual_prior_xy']=np.asarray(visual_hint).tolist()
        keys = self.keyframes[-12:]
        if self.use_local_submap and self.kiss is None:
            radius=4.
            keys = [key for key in keys if np.linalg.norm(key.pose.xy() - guess.xy()) <= radius] or keys[-1:]
        target = np.vstack([transform_xy(k.scan, k.pose) for k in keys])
        if self.use_local_submap and self.kiss is None:
            local = target[np.linalg.norm(target - guess.xy(), axis=1) <= 5.]
            if len(local) >= 12:
                _, indices = np.unique(np.floor(local/.10).astype(int), axis=0, return_index=True)
                if len(indices) >= 12:
                    target = local[indices]
            if not self.c.navigation_support:
                guess, cost, margin = coarse_submap_seed(scan, target, guess, self.c.imu_heading_reference)
                self.last_registration.update(coarse_cost_m=cost, coarse_margin_m=margin)
        self.last_registration["submap_points"] = len(target)
        if self.public_reference is not None:
            reference=self.public_reference
            target=reference[np.linalg.norm(reference-guess.xy(),axis=1)<self.public_reference_radius]
            if len(target)<12:
                self.pose.quality=0.
                return self.pose,points
            self.last_registration.update(source='scan_to_declared_public_map',submap_points=len(target))
        if self.kiss is not None:
            # KISS expects timestamps matching the filtered cloud. Motion compensation
            # stays disabled here because gravity/range filtering changes indices.
            self.kiss.register_frame(points, np.zeros(len(points)))
            t = self.kiss.last_pose
            candidate = Pose(float(t[0, 3]), float(t[1, 3]), float(np.arctan2(t[1, 0], t[0, 0])))
            distances, _ = cKDTree(target).query(transform_xy(scan, candidate))
            inside = distances < .25
            overlap = float(inside.mean())
            rmse = float(np.sqrt(np.mean(distances[inside] ** 2))) if inside.any() else 1.
        else:
            registration = register_lines if self.c.localization_metric == "point_to_line" else register
            kwargs = {"yaw_hint": self.sensor_heading} if self.c.imu_heading_reference and registration is register_lines else {}
            if registration is register_lines and self.c.navigation_support:
                kwargs["motion_prior_weight"] = 64. if visual_hint is not None else 16.
            if len(scan) < self.c.min_points:
                candidate, overlap, rmse = guess, 0., float("inf")
            else:
                candidate, overlap, rmse = registration(scan, target, guess, **kwargs)
            if (self.public_reference is not None and registration is register_lines
                    and self.use_sparse_geometry_fallback and plane is not None
                    and np.linalg.norm(plane[:2]) < .08
                    and overlap*np.exp(-rmse/.20) < self.c.min_pose_quality):
                alternative = register_lines(scan, target, guess, sparse_geometry=True, **kwargs)
                if alternative[1]*np.exp(-alternative[2]/.20) >= self.c.min_pose_quality:
                    candidate, overlap, rmse = alternative
                    self.last_registration["source"] = "observable_sparse_lines"
            if (self.public_reference is not None and self.c.experimental_scan_window
                    and registration is register_lines and self.c.imu_heading_reference):
                if overlap*np.exp(-rmse/.20) < self.c.min_pose_quality:
                    accumulated = self.accumulated_public_fit(scan, target, guess, frame.timestamp, kwargs)
                    if accumulated is not None:
                        candidate, overlap, rmse = accumulated
                        self.last_registration["source"] = "short_accumulated_scan_to_public_map"
                self.recent_structural_scans.append(
                    (frame.timestamp, self.scan_odometry_xy.copy(), guess.yaw, scan))
        if portal_available and overlap*np.exp(-rmse/.20) < self.c.min_pose_quality:
            surface_fit = self.portal_reference.fit(points, guess, -plane[2])
            if surface_fit is not None:
                candidate, inliers, rmse = surface_fit
                overlap = min(1., inliers/20.)
                self.last_registration.update(source="portal_surface_reference", surface_inliers=inliers)
        # In the fused profile, IMU fixes heading and measured leg odometry
        # supplies along-wall motion. XY covariance is not an independent
        # failure probability for that fused estimate.
        quality = (1. if self.c.navigation_support and self.c.imu_heading_reference else geometry) * overlap * np.exp(-rmse / .20)
        self.last_registration.update(overlap=float(overlap), rmse_m=float(rmse) if np.isfinite(rmse) else None,
                                      geometry=float(geometry), correction_m=float(np.linalg.norm(candidate.xy()-guess.xy())))
        yaw_baseline = guess.yaw if self.public_reference is not None and self.c.imu_heading_reference else old.yaw
        yaw_limit = .18 if self.public_reference is not None and self.c.imu_heading_reference else max(.4, dt * 2.)
        if np.linalg.norm(candidate.xy() - old.xy()) > max(.25, dt * 1.2) or abs(wrap(candidate.yaw-yaw_baseline)) > yaw_limit:
            quality = 0.
        if quality < self.c.min_pose_quality and visual_hint is not None:
            self.pose=Pose(*visual_hint,guess.yaw,quality=.65,sigma=min(.3,old.sigma+.02))
            self.last_registration['source']='visual_imu_scan_disagreement'
            return self.pose,points
        if quality < self.c.min_pose_quality:
            self.pose = Pose(old.x, old.y, old.yaw, float(quality), min(2., old.sigma + .05))
            return self.pose, points
        candidate.quality, candidate.sigma = float(quality), float(max(.03, rmse))
        self.pose = candidate
        last = self.keyframes[-1].pose
        if np.linalg.norm(last.xy() - candidate.xy()) > .45 or abs(wrap(last.yaw - candidate.yaw)) > .3:
            self._keyframe(points)
        return self.pose, points

    @staticmethod
    def _imu_yaw(quaternion):
        w, x, y, z = quaternion
        return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

    def _keyframe(self, points: np.ndarray):
        pose = Pose(**vars(self.pose))
        self.keyframes.append(Keyframe(pose, points.copy(), self.structural_scan(points),
                                       self.last_timestamp,
                                       getattr(self, "_last_quaternion", None)))
        self.graph.nodes.append(pose)
        self.keyframe_headings.append(self.sensor_heading if self.c.imu_heading_reference else pose.yaw)
        n = len(self.keyframes) - 1
        if n:
            self.graph.edges.append((n - 1, n, relative(self.keyframes[n - 1].pose, pose), 1.))
        # The 3-D backend maintains its own odometry map; do not apply an SE(2)
        # correction to that map without a corresponding backend reset.
        if n < 18 or self.kiss is not None or self.public_reference is not None:
            return
        candidates = [(np.linalg.norm(k.pose.xy() - pose.xy()), i) for i, k in enumerate(self.keyframes[:-15])]
        distance, i = min(candidates)
        if distance > .75:
            return
        old = self.keyframes[i]
        source = self.structural_scan(points)[:, :2]
        target = old.scan[:, :2]
        delta = relative(old.pose, pose)
        if self.c.localization_metric == "point_to_line" and len(target) >= 12:
            _, normals, valid = line_normals(target)
            spectrum = np.linalg.eigvalsh(normals[valid].T@normals[valid])
            # Parallel walls do not constrain longitudinal loop displacement.
            # Such a loop must not fold an otherwise valid odometry trajectory.
            if valid.sum() < 12 or spectrum[0] < .05*max(spectrum[-1], 1.):
                return
        registration = register_lines if self.c.localization_metric == "point_to_line" else register
        fit, overlap, rmse = registration(source, target, Pose(*delta), limit=.35)
        if overlap < .85 or rmse > .06 or np.linalg.norm(fit.xy() - delta[:2]) > .4:
            return
        self.graph.edges.append((i, n, np.array([fit.x, fit.y, fit.yaw]), 2.))
        previous = [Pose(**vars(k.pose)) for k in self.keyframes]
        optimized = self.graph.optimize(self.keyframe_headings if self.c.imu_heading_reference else None)
        self.corrections = list(zip(previous, optimized))
        for key, updated in zip(self.keyframes, optimized):
            key.pose = updated
        self.pose = self.keyframes[-1].pose
        self.revision += 1
