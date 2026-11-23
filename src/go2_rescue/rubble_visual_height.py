"""Sparse two-view rubble height estimates with no depth-camera dependency."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class View:
    stamp: float
    gray: np.ndarray
    features: np.ndarray | None
    camera: np.ndarray
    center: np.ndarray
    base_xyz: np.ndarray
    from_level: np.ndarray
    plane: np.ndarray
    labels: np.ndarray


class RubbleVisualHeight:
    def __init__(self, intrinsics, *, interpolate_stone=False, wider_baseline=False, independent_ground=False,
                 use_contact_ground=False, dense=False, multiview=False):
        self.k = intrinsics
        self.interpolate_stone = interpolate_stone
        self.wider_baseline = wider_baseline
        self.independent_ground = independent_ground
        self.use_contact_ground = use_contact_ground
        self.dense = dense
        self.multiview = multiview
        self.views = deque(maxlen=20)
        self.multiview_history = deque(maxlen=30)
        self.ground_history = deque(maxlen=36 if independent_ground else 12)
        self.dense_ground_history = deque(maxlen=24)
        self.extra_surface = np.empty((0, 5))
        self.diagnostics = {}
        self.contact_ground = deque(maxlen=180)

    def clear(self):
        self.views.clear()
        self.multiview_history.clear()
        self.ground_history.clear()
        self.dense_ground_history.clear()
        self.contact_ground.clear()
        self.extra_surface = np.empty((0, 5))

    def observe_contacts(self, leveled_soles, pose, base_z, stamp):
        """Cache recent foot contacts, not heights extrapolated from stance."""
        from .geometry import transform_xy
        if not (self.use_contact_ground or self.multiview) or not np.isfinite(base_z) or len(leveled_soles) < 2:
            return
        xy = transform_xy(leveled_soles, pose)
        xyz = np.column_stack((xy, leveled_soles[:, 2]+base_z))
        self.contact_ground.append((stamp,xyz))

    def observe(self, image, mask, rotation, origin, pose, ground_z, plane, stamp,
                *, ground_mask=None, base_z=None, floor_mask=None):
        self.diagnostics = {}
        self.extra_surface = np.empty((0, 5))
        if ground_mask is not None and (base_z is None or not np.isfinite(base_z)):
            return np.empty((0, 5))
        enough_rubble = np.count_nonzero(mask) >= 80
        floor_only = ground_mask is not None and (self.independent_ground or self.multiview)
        if not enough_rubble and not floor_only:
            return np.empty((0, 5 if ground_mask is not None else 3))
        if ground_mask is None and (plane is None or not np.isfinite(ground_z)):
            return np.empty((0, 5 if ground_mask is not None else 3))
        if plane is None:
            plane = np.zeros(3)  # unused for geometry with a supplied base_z
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        semantic = interior.copy()
        if ground_mask is not None:
            kernel = np.ones((5, 5), np.uint8)
            near = cv2.dilate(interior, np.ones((101, 101), np.uint8)) != 0
            ground = cv2.erode(ground_mask.astype(np.uint8), kernel).astype(bool)
            semantic[(ground if self.independent_ground else (near & ground)) & (interior == 0)] = 2
        yaw = pose.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        from_level = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        base_xyz = np.array([pose.x, pose.y, ground_z-plane[2] if base_z is None else base_z])
        world_rotation = from_level @ rotation
        center = base_xyz + from_level @ origin
        camera = self.k @ np.column_stack((world_rotation.T, -world_rotation.T @ center))
        if self.multiview and ground_mask is not None:
            from .rubble_multiview import select_sources, sweep_multiview

            # The wider floor mask and older views are private to visual reconstruction.
            # The sparse pose and ground stream still uses its original sampling rules.
            full_semantic = interior.copy()
            floor = ground if floor_mask is None else cv2.erode(floor_mask.astype(np.uint8), kernel) != 0
            full_semantic[floor & (interior == 0)] = 2
            current = View(stamp, gray, None, camera, center, base_xyz, from_level, plane, full_semantic)
            sources = select_sources(current, self.multiview_history)
            if (not self.multiview_history
                    or np.linalg.norm(center - self.multiview_history[-1].center) >= .075):
                self.multiview_history.append(current)
            points, classes = sweep_multiview(self.k, sources, current)
            self._count(multiview_sources=len(sources))
            self._dense_result(points, classes, stamp)
            if not enough_rubble and not self.independent_ground:
                return np.empty((0, 5))
        features = cv2.goodFeaturesToTrack(gray, 120, .008, 4, mask=interior)
        if ground_mask is not None:
            ground_features = cv2.goodFeaturesToTrack(gray, 200, .003, 4,
                                                       mask=(semantic == 2).astype(np.uint8))
            if ground_features is not None:
                features = ground_features if features is None else np.concatenate((features, ground_features))
        if features is None:
            return np.empty((0, 5 if ground_mask is not None else 3))
        view = View(stamp, gray, features, camera, center, base_xyz, from_level, plane.copy(), semantic)
        candidates = []
        for old in self.views:
            delta = base_xyz - old.base_xyz
            baseline = np.linalg.norm(delta[:2])
            old_plane_at_base = old.base_xyz[2] + old.plane[2] + (old.from_level.T @ delta)[:2] @ old.plane[:2]
            if (0. < stamp-old.stamp <= (8. if self.wider_baseline else 5.)
                    and .18 <= baseline <= (.9 if self.wider_baseline else .48)
                    and (ground_mask is not None or abs(old_plane_at_base-ground_z) <= .035)
                    and np.trace(old.from_level.T @ from_level) > 1 + 2*np.cos(np.deg2rad(28))):
                candidates.append((abs(baseline-(.58 if self.wider_baseline else .30)), old, baseline))
        if not self.views or np.linalg.norm(base_xyz[:2]-self.views[-1].base_xyz[:2]) >= .075:
            self.views.append(view)
        if not candidates:
            return np.empty((0, 5 if ground_mask is not None else 3))
        if ground_mask is None:
            chosen = [min(candidates, key=lambda entry: entry[0])]
        else:
            chosen = sorted(candidates, key=lambda entry: entry[0])[:3]
        results = [self._pair(old, baseline, view, ground_mask) for _, old, baseline in chosen]
        if self.dense and not self.multiview and ground_mask is not None:
            from .rubble_dense import sweep_surface

            points, classes = sweep_surface(self.k, chosen[0][1], view)
            self._dense_result(points, classes, stamp)
        sparse = np.vstack(results)
        if self.multiview and ground_mask is not None:
            sparse = self._relative(sparse[:, :3], self._dense_support(stamp), stamp,
                                    contacts=False, keep_contour=True)
        return sparse

    def _dense_support(self, stamp):
        ground = [cloud for time, cloud in self.dense_ground_history if 0 <= stamp-time <= 6.]
        if not self.multiview:
            ground.extend(cloud for time, cloud in self.ground_history if 0 <= stamp-time <= 3.)
        else:
            ground.extend(cloud for time, cloud in self.contact_ground if 0 <= stamp-time <= 10.)
        return ground

    def _dense_result(self, points, classes, stamp):
        self._count(dense_stone=int((classes == 1).sum()), dense_floor=int((classes == 2).sum()))
        support = points[classes == 2]
        if len(support):
            self.dense_ground_history.append((stamp, support))
        # Keep dense floor estimates independent of the existing range evidence.
        self.extra_surface = self._relative(points[classes == 1], self._dense_support(stamp), stamp,
                                            contacts=False, keep_contour=self.multiview)

    def _count(self, **fields):
        for name, value in fields.items():
            self.diagnostics[name] = self.diagnostics.get(name, 0) + value

    def _pair(self, old, baseline, view, ground_mask):
        stamp, gray, camera, center, base_xyz, from_level, plane, semantic = (
            view.stamp, view.gray, view.camera, view.center, view.base_xyz,
            view.from_level, view.plane, view.labels)
        tracked, found, error = cv2.calcOpticalFlowPyrLK(old.gray, gray, old.features, None)
        if tracked is None:
            return np.empty((0, 5 if ground_mask is not None else 3))
        backwards, found_back, _ = cv2.calcOpticalFlowPyrLK(gray, old.gray, tracked, None)
        if backwards is None:
            return np.empty((0, 5 if ground_mask is not None else 3))
        previous = old.features.reshape(-1, 2)
        current = tracked.reshape(-1, 2)
        pixels = np.rint(current).astype(int)
        rows, cols = gray.shape
        inside = ((pixels >= 0) & (pixels < [cols, rows])).all(axis=1)
        pixels = np.clip(pixels, 0, [cols-1,rows-1])
        old_pixels = np.rint(previous).astype(int)
        tags = old.labels[old_pixels[:, 1], old_pixels[:, 0]]
        good = (inside & found.ravel().astype(bool) & found_back.ravel().astype(bool)
                & (error.ravel() < 15) & (np.linalg.norm(previous-backwards.reshape(-1,2),axis=1)
                                         < np.where(tags == 2, .65, .8))
                & (tags != 0) & (semantic[pixels[:,1],pixels[:,0]] == tags))
        self._count(matched_ground=int((good & (tags == 2)).sum()),
                    matched_stone=int((good & (tags == 1)).sum()))
        if not good.any():
            return np.empty((0, 5 if ground_mask is not None else 3))
        lhs, rhs = previous[good], current[good]
        raw = cv2.triangulatePoints(old.camera, camera, lhs.T, rhs.T).T
        finite = np.isfinite(raw).all(axis=1) & (np.abs(raw[:,3]) > 1e-7)
        raw, lhs, rhs = (values[finite] for values in (raw, lhs, rhs))
        points = raw[:,:3]/raw[:,3,None]
        hom = np.column_stack((points,np.ones(len(points))))
        before, now = hom @ old.camera.T, hom @ camera.T
        tags = tags[good][finite]
        depth = (before[:,2] > .2) & (now[:,2] > .2)
        if ground_mask is not None:
            self._count(stone_positive_depth=int((depth & (tags == 1)).sum()))
        points, before, now, lhs, rhs, tags = (values[depth] for values in (points,before,now,lhs,rhs,tags))
        reproj = (np.linalg.norm(before[:,:2]/before[:,2,None]-lhs,axis=1) < 1.25
                  ) & (np.linalg.norm(now[:,:2]/now[:,2,None]-rhs,axis=1) < 1.25)
        local = (points-base_xyz) @ from_level
        relative = local[:,2] - (local[:,:2] @ plane[:2] + plane[2])
        distance = np.linalg.norm(local[:,:2],axis=1)
        a, b = points-old.center, points-center
        cosine = np.sum(a*b,axis=1) / (np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        if ground_mask is not None:
            good = reproj & (cosine < np.cos(np.deg2rad(3))) & (distance < min(4., baseline/.085))
            # Ground can lie near the focus of expansion. It gets its own
            # range limit and tighter pixel residual before fitting a patch.
            ground_reproj = (np.linalg.norm(before[:,:2]/before[:,2,None]-lhs, axis=1) < .75
                             ) & (np.linalg.norm(now[:,:2]/now[:,2,None]-rhs, axis=1) < .75)
            ground_good = ground_reproj & (cosine < np.cos(np.deg2rad(1.5))) & (distance < 4.)
            # Ground supports a height only inside a small slope-consistent
            # triangle. No remote projection onto the robot's stance plane.
            self._count(
                stone_reprojected=int((reproj & (tags == 1)).sum()),
                stone_with_parallax=int((reproj & (tags == 1)
                                         & (cosine < np.cos(np.deg2rad(3)))).sum()),
                ground_reprojected=int((ground_reproj & (tags == 2)).sum()))
            stone = good & (tags == 1)
            surface = points[stone]
            self._count(triangulated_stone=len(surface))
            if self.interpolate_stone:
                fill = bounded_stone_surface(surface, rhs[stone], now[stone, 2], semantic == 1)
                self._count(interpolated_surface=len(fill))
                surface = np.vstack((surface, fill))
            support = points[ground_good & (tags == 2)]
            self._count(triangulated_ground=len(support))
            return self._resolve(surface, support, stamp)
        good = (reproj & (cosine < np.cos(np.deg2rad(3))) & (distance < min(2.5,baseline/.10))
                & (relative >= .035) & (relative <= .42))
        return np.column_stack((points[good,:2],relative[good]))

    def _resolve(self, surface, support, stamp):
        if len(support):
            self.ground_history.append((stamp, support))
        compatible = [cloud for time, cloud in self.ground_history if 0 <= stamp-time <= (10. if self.independent_ground else 3.)]
        return self._relative(surface, compatible, stamp, contacts=self.use_contact_ground)

    def _relative(self, surface, compatible, stamp, *, contacts, keep_contour=False):
        support = np.vstack(compatible) if compatible else np.empty((0, 3))
        ground = estimate_ground_under_stone(support, surface[:, :2])
        if contacts and len(support) and np.any(~np.isfinite(ground)):
            contact = [cloud for t, cloud in self.contact_ground if 0 <= stamp-t <= 15.]
            if contact:
                anchored = contacts_with_visual_support(np.vstack(contact), support)
                self._count(compatible_contact_points=len(anchored))
                if len(anchored):
                    missing = ~np.isfinite(ground)
                    ground[missing] = estimate_ground_under_stone(np.vstack((support, anchored)), surface[missing, :2])
                    self._count(height_from_contact_support=int(np.isfinite(ground[missing]).sum()))
        self._count(stone_with_ground=int(np.isfinite(ground).sum()))
        relative = surface[:, 2] - ground
        keep = ~np.isfinite(relative) | ((relative >= .035) & (relative <= .42))
        if keep_contour:
            relative[~keep] = np.nan
            return np.column_stack((surface, ground, relative))
        return np.column_stack((surface[keep], ground[keep], relative[keep]))



def contacts_with_visual_support(contacts, visual_ground):
    """Only use contact samples corroborated by nearby visual floor."""
    from scipy.spatial import cKDTree

    if not len(contacts) or not len(visual_ground):
        return np.empty((0, 3))
    tree = cKDTree(visual_ground[:, :2])
    k = min(3, len(visual_ground))
    distance, ix = tree.query(contacts[:, :2], k=k)
    if k<3:
        return np.empty((0, 3))
    heights = visual_ground[ix,2]
    # A foot on rubble, and a distant floor on a different level, should not
    # seed a fictitious supporting plane beneath the reconstructed stones.
    valid = (distance[:, -1] <= .18) & (np.ptp(heights,axis=1) <= .06)
    valid &= np.abs(contacts[:,2] - np.median(heights,axis=1)) <= .045
    return contacts[valid]


def estimate_ground_under_stone(support, xy):
    """Bounded interpolation, then one-sided continuation of a coherent plane."""
    from scipy.spatial import cKDTree
    from .terrain import interpolate_surface

    z = np.full(len(xy), np.nan)
    if len(support) < 3 or not len(xy):
        return z
    cells = np.floor(support[:, :2] / .06).astype(int)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    support = np.array([np.median(support[groups == i], axis=0)
                        for i in range(groups.max()+1)])
    tree = cKDTree(support[:, :2])
    z = interpolate_surface(support, xy, .28, 28.)
    for i in np.flatnonzero(~np.isfinite(z)):
        neighbours = tree.query_ball_point(xy[i], .42)
        if len(neighbours) < 8:
            continue
        patch = support[neighbours]
        offsets = patch[:, :2] - xy[i]
        if np.min(np.linalg.norm(offsets, axis=1)) > .21:
            continue
        # Reject narrow support strips and any unresolved step/edge in the patch.
        if np.linalg.eigvalsh(np.cov(offsets.T))[0] < .0012:
            continue
        A = np.column_stack((offsets, np.ones(len(offsets))))
        coef = np.linalg.lstsq(A, patch[:, 2], rcond=None)[0]
        if (np.linalg.norm(coef[:2]) <= .53
                and np.max(np.abs(A @ coef - patch[:, 2])) <= .035):
            z[i] = coef[2]
    return z


def bounded_stone_surface(xyz, uv, depth, mask):
    """Fill small gaps using perspective-correct surface triangles.

    Only interpolate inside the same view's eroded semantic mask. Large
    screen/metric edges and optical depth discontinuities are vetoes.
    """
    from scipy.spatial import Delaunay, QhullError
    if len(uv) < 3:
        return np.empty((0, 3))
    try:
        tri = Delaunay(uv)
    except QhullError:
        return np.empty((0, 3))
    filled = []
    for simplex in tri.simplices:
        screen, surface, z = uv[simplex], xyz[simplex], depth[simplex]
        edges = ((0, 1), (1, 2), (2, 0))
        if any(np.linalg.norm(screen[i]-screen[j]) > 20.
               or np.linalg.norm(surface[i]-surface[j]) > .26
               or abs(z[i]-z[j]) > .13 for i, j in edges):
            continue
        lo = np.floor(screen.min(axis=0)).astype(int)
        hi = np.ceil(screen.max(axis=0)).astype(int)
        if np.any(lo < 0) or np.any(hi >= np.array(mask.shape[::-1])):
            continue
        roi = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), np.uint8)
        cv2.fillConvexPoly(roi, np.rint(screen-lo).astype(np.int32), 1)
        if not mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1][roi != 0].all():
            continue
        for bary in ([1/3,1/3,1/3], [.5,.25,.25], [.25,.5,.25], [.25,.25,.5]):
            weights = bary/z
            filled.append(weights @ surface / weights.sum())
    return np.asarray(filled).reshape(-1, 3)
