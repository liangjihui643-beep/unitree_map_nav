from __future__ import annotations

from collections import deque

import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .config import Config
from .geometry import bresenham, rot2, transform_xy
from .types import Pose
from .terrain import ground_surface, interpolate_surface


def spatial_frame_support(frames, radius_cells=1):
    """Count nearby distinct frames at measured cells without filling unseen cells."""
    frames = [set(frame) for frame in frames]
    observed = set().union(*frames) if frames else set()
    votes = dict.fromkeys(observed, 0)
    for frame in frames:
        neighbourhood = {(i+di,j+dj) for i,j in frame
                         for di in range(-radius_cells,radius_cells+1)
                         for dj in range(-radius_cells,radius_cells+1)
                         if di*di+dj*dj <= 2*radius_cells*radius_cells}
        for cell in neighbourhood & observed:
            votes[cell] += 1
    return votes


class GridMap:
    """Observed free/occupied space plus floor/ceiling evidence.

    Empty rays never establish safe floor support. A cell needs ground returns
    before it becomes traversable; this conservatively rejects unobserved drops.
    """

    def __init__(self, config: Config):
        self.c = config
        self.n = int(np.ceil(config.map_size / config.resolution))
        self.origin = np.array([-config.map_size / 2, -config.map_size / 2])
        self.clear()

    def clear(self):
        shape = (self.n, self.n)
        self.logodds = np.zeros(shape, dtype=np.float32)
        self.seen = np.zeros(shape, dtype=bool)
        self.floor = np.full(shape, np.nan, dtype=np.float32)
        # 0 unknown, 1 footprint only, 2 interpolated, 3 measured.
        self.floor_source = np.zeros(shape, dtype=np.uint8)
        self.ceiling = np.full(shape, np.inf, dtype=np.float32)
        self.headroom = np.full(shape, np.inf, dtype=np.float32)
        self.hazard_votes = np.zeros(shape, dtype=np.uint8)
        self.structure_votes = np.zeros(shape, dtype=np.uint16)
        self.overhead_votes = np.zeros(shape, dtype=np.uint16)
        self.pending_endpoints = []
        self.surface_scans = deque(maxlen=12)
        self.pending_ray_cells = {}
        self.free_evidence = np.zeros(shape, dtype=np.float32)
        self.hit_evidence_total = np.zeros(shape, dtype=np.float32)
        self.occupancy_contributions = {}
        self.cell_frames = {}
        self.evidence_window = 200
        self._has_hazards = False
        self.ground_height = 0.
        self.support_plane=None
        self.support_contacts=np.empty((0,3))
        self.visual_ground_mask=None
        self._free_cache = None
        self._surface_safe = None
        self._frame_seq = 0
        self.wall_diagnostics = []
        self._clear_evidence()

    def _clear_evidence(self):
        """Optional per-cell evidence counters (no effect on occupancy by default)."""
        self._evidence_enabled = bool(self.c.map_evidence or self.c.map_ray_height_clear
                                      or self.c.map_evidence_positions)
        self.evidence_positions = {} if self.c.map_evidence_positions else None
        if not self._evidence_enabled:
            return
        shape = (self.n, self.n)
        self.evidence_endpoint = np.zeros(shape, dtype=np.uint16)
        self.evidence_prefix = np.zeros(shape, dtype=np.uint16)
        self.evidence_near_clear = np.zeros(shape, dtype=np.uint16)  # full band crossing
        self.evidence_near_partial = np.zeros(shape, dtype=np.uint16)
        self.evidence_collision_partial = np.zeros(shape, dtype=np.uint16)
        self.evidence_collision_full = np.zeros(shape, dtype=np.uint16)
        self.evidence_unknown_ground_prefix = np.zeros(shape, dtype=np.uint16)
        self.evidence_structure_endpoint = np.zeros(shape, dtype=np.uint16)
        self.evidence_obstacle_endpoint = np.zeros(shape, dtype=np.uint16)
        self.evidence_frame_support = np.zeros(shape, dtype=np.uint16)
        self.evidence_floor_footprint = np.zeros(shape, dtype=np.uint16)
        self.evidence_floor_interpolated = np.zeros(shape, dtype=np.uint16)
        self.evidence_floor_measured = np.zeros(shape, dtype=np.uint16)
        self._last_endpoint = np.full(shape, -1, dtype=np.int32)
        self._last_prefix = np.full(shape, -1, dtype=np.int32)
        self._last_near_partial = np.full(shape, -1, dtype=np.int32)
        self._last_near_full = np.full(shape, -1, dtype=np.int32)
        self._last_collision_partial = np.full(shape, -1, dtype=np.int32)
        self._last_collision_full = np.full(shape, -1, dtype=np.int32)
        self._last_support = np.full(shape, -1, dtype=np.int32)
        self._last_floor = np.full(shape, -1, dtype=np.int32)
        self._last_obstacle = np.full(shape, -1, dtype=np.int32)
        self._last_structure = np.full(shape, -1, dtype=np.int32)

    def evidence_summary(self):
        """Compact counts for map_evidence.json; empty when evidence is disabled."""
        if not getattr(self, "_evidence_enabled", False):
            return {}
        views = self.evidence_positions or {}
        return {
            "frames": int(self._frame_seq),
            "endpoint": int(self.evidence_endpoint.sum()),
            "prefix": int(self.evidence_prefix.sum()),
            "near_ground_full": int(self.evidence_near_clear.sum()),
            "near_ground_partial": int(self.evidence_near_partial.sum()),
            "collision_partial": int(self.evidence_collision_partial.sum()),
            "collision_full": int(self.evidence_collision_full.sum()),
            "unknown_ground_prefix": int(self.evidence_unknown_ground_prefix.sum()),
            "structure_endpoint": int(self.evidence_structure_endpoint.sum()),
            "obstacle_endpoint": int(self.evidence_obstacle_endpoint.sum()),
            "frame_support": int(self.evidence_frame_support.sum()),
            "cells_endpoint": int((self.evidence_endpoint > 0).sum()),
            "cells_near_ground_full": int((self.evidence_near_clear > 0).sum()),
            "cells_near_ground_partial": int((self.evidence_near_partial > 0).sum()),
            "cells_unknown_ground_prefix": int((self.evidence_unknown_ground_prefix > 0).sum()),
            "cells_obstacle": int((self.evidence_obstacle_endpoint > 0).sum()),
            "cells_structure": int((self.evidence_structure_endpoint > 0).sum()),
            "floor_footprint": int(self.evidence_floor_footprint.sum()),
            "floor_interpolated": int(self.evidence_floor_interpolated.sum()),
            "floor_measured": int(self.evidence_floor_measured.sum()),
            "floor_source_now": np.bincount(self.floor_source.ravel(), minlength=6).astype(int).tolist(),
            "near_band_m": [float(self.c.map_near_band_low), float(self.c.min_obstacle_height)],
            "collision_band_m": [float(self.c.min_obstacle_height), float(self.c.body_clearance)],
            "position_bins": {f"{k[0]},{k[1]}": len(v) for k, v in views.items()} if views else {},
            "ray_height_clear": bool(self.c.map_ray_height_clear),
        }

    def cell(self, xy) -> tuple[int, int]:
        if not np.isfinite(np.asarray(xy)[:2]).all():
            raise ValueError("Non-finite map coordinate")
        ij = np.floor((np.asarray(xy)[:2] - self.origin) / self.c.resolution).astype(int)
        return int(ij[0]), int(ij[1])

    def world(self, cell) -> np.ndarray:
        return self.origin + (np.asarray(cell) + .5) * self.c.resolution

    def inside(self, cell) -> bool:
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n

    def _record_view(self, cell, xy):
        """Keep bounded independent observer positions (>=0.15 m apart, max 4)."""
        if self.evidence_positions is None:
            return
        views = self.evidence_positions.setdefault(tuple(cell), [])
        if len(views) >= 4:
            return
        xy = np.asarray(xy, float)
        if any(np.hypot(*(xy - np.asarray(view))) < .15 for view in views):
            return
        views.append((float(xy[0]), float(xy[1])))

    def update(self, points: np.ndarray, pose: Pose, quaternion=None, timestamp=None):
        self._frame_seq += 1
        fid = self._frame_seq
        start = self.cell(pose.xy())
        if not self.inside(start):
            return
        # Stable wall evidence from previous frames only; this frame's hits are
        # not allowed to become a wall that truncates their own ray.
        wall_cells = None
        if self.c.map_wall_truncate:
            wall_cells = self.logodds >= float(self.c.map_wall_logodds)
            if self._evidence_enabled:
                wall_cells |= self.evidence_obstacle_endpoint >= 2
        # Ground reference for near-ground evidence is the previous frame's
        # floor (with a trusted source), never the floor updated by this scan.
        floor_ref = self.floor.copy() if (self._evidence_enabled or self.c.map_skip_high_clear
                                          or self.c.map_skip_above_near_clear) else None
        source_ref = self.floor_source.copy() if floor_ref is not None else None
        mount = np.asarray(self.c.lidar_mount, float).copy()
        if quaternion is not None:
            from .geometry import gravity_rotation
            mount = gravity_rotation(np.asarray(quaternion, float)) @ mount
        xy = transform_xy(points, pose)
        raw_heights = points[:, 2] + self.c.base_height
        cells = np.floor((xy - self.origin) / self.c.resolution).astype(int)
        inside = (cells >= 0).all(axis=1) & (cells < self.n).all(axis=1)
        # Align successive local elevation observations to overlapping measured
        # terrain, rather than pinning every scan's floor to world z=0.
        indices = np.flatnonzero(inside)
        if len(indices):
            previous = self.floor[tuple(cells[indices].T)]
            offsets = previous - raw_heights[indices]
            consistent = np.isfinite(offsets) & (abs(offsets - self.ground_height) < .08)
            if consistent.sum() >= 5 and not self.c.navigation_support:
                self.ground_height = float(np.median(offsets[consistent]))
        heights = raw_heights + self.ground_height
        ev = getattr(self, "_evidence_enabled", False)
        ground_inside=inside.copy()
        if self.c.navigation_support and getattr(self,"support_plane",None) is not None:
            plane=self.support_plane
            residual=points[:,2]-(points[:,:2]@plane[:2]+plane[2])
            rgb=getattr(self,"visual_ground_mask",np.zeros(len(points),bool))
            if rgb is None or len(rgb)!=len(points):rgb=np.zeros(len(points),bool)
            ground_inside &= ((abs(residual)<.045)|rgb)&(np.linalg.norm(points[:,:2],axis=1)<2.5)
        ground_xyz=np.column_stack([xy[ground_inside],heights[ground_inside]])
        if self.c.navigation_support:
            ground_xyz=np.vstack([ground_xyz,getattr(self,"support_contacts",np.empty((0,3)))])
        samples = ground_surface(ground_xyz,
                                 self.ground_height, self.c.resolution,
                                 self.c.ground_max_gap, self.c.slope_limit)
        if self.c.navigation_support and self.support_plane is not None:
            rgb=self.visual_ground_mask
            if rgb is not None and len(rgb)==len(points):
                distance=np.linalg.norm(points[:,:2],axis=1)
                if self.c.map_connected_ramp_ground:
                    # Classification checked the measured surface. Traversability
                    # checks slope independently.
                    direct = inside & rgb & (distance < 2.5)
                else:
                    limit = self.c.slope_limit(xy)
                    plausible = abs(points[:, 2] - self.support_plane[2]) <= np.tan(np.deg2rad(limit)) * distance + .04
                    direct = inside & rgb & plausible
                # Sparse formal rays can observe the ramp 2m ahead without
                # any intermediate floor echo. RGB labels that measured hit;
                # do not turn it into an obstacle just because no short
                # triangulation connects it to the standing feet. Interpolation
                # remains limited to small measured triangles below.
                samples=np.vstack([samples,np.column_stack([xy[direct],heights[direct]])])
        recovered = self._connected_surface(xy[inside], heights[inside], pose.xy(), start)
        if len(recovered):
            samples = np.vstack((samples, recovered))
        if len(samples):
            lo = np.maximum(0, np.floor((samples[:, :2].min(axis=0) - self.origin) / self.c.resolution).astype(int))
            hi = np.minimum(self.n - 1, np.floor((samples[:, :2].max(axis=0) - self.origin) / self.c.resolution).astype(int))
            ii, jj = np.mgrid[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
            query_cells = np.column_stack([ii.ravel(), jj.ravel()])
            query_xy = self.origin + (query_cells + .5) * self.c.resolution
            z = interpolate_surface(samples, query_xy, self.c.ground_max_gap, self.c.slope_limit)
            valid = np.isfinite(z)
            observed_cells = np.floor((samples[:, :2] - self.origin) / self.c.resolution).astype(int)
            floor_cells = np.vstack([query_cells[valid], observed_cells])
            floor_z = np.r_[z[valid], samples[:, 2]]
            local_limits = self.c.slope_limit(self.origin+(floor_cells+.5)*self.c.resolution)
            interpolated_count = int(valid.sum())
            for index, (cell, height, local_limit) in enumerate(zip(floor_cells, floor_z, local_limits)):
                ij = tuple(cell)
                old = self.floor[ij]
                # An occluded floor does not move up to a later roof return.
                # Keep established lower-envelope support when a connected
                # curved wall/roof is mistakenly included in ground samples.
                # The higher hit is still classified for obstacles/headroom.
                if local_limit > self.c.max_slope_deg and np.isfinite(old) and height > old+self.c.max_step_height:
                    continue
                source = 2 if index < interpolated_count else 3
                self.floor[ij] = height if not np.isfinite(old) else .5 * old + .5 * height
                previous_source = self.floor_source[ij]
                self.floor_source[ij] = max(previous_source, source)
                if ev and self._last_floor[ij] != fid:
                    self._last_floor[ij] = fid
                    if source == 2 and previous_source < 2:
                        self.evidence_floor_interpolated[ij] += 1
                    elif source == 3 and previous_source < 3:
                        self.evidence_floor_measured[ij] += 1
                # A return previously classified without floor support may
                # later be confirmed as the ramp itself, not an overhead roof.
                if self.ceiling[ij] <= height + .025:
                    self.ceiling[ij] = self.headroom[ij] = np.inf
                else:
                    self.headroom[ij] = self.ceiling[ij]-self.floor[ij]
        self._free_cache = None
        self._surface_safe = None
        floor_at_hit = np.full(len(points), np.nan)
        if len(indices):
            floor_at_hit[indices] = self.floor[tuple(cells[indices].T)]
        # Only a nearby measured/interpolated local ground may replace a missing
        # cell ground; never fall back to the robot-foot height.
        local_ground = np.full(len(points), np.nan)
        for point_index in np.flatnonzero(inside & ~np.isfinite(floor_at_hit)):
            value = self._local_ground(xy[point_index])
            if value is not None:
                local_ground[point_index] = value
        reference = np.where(np.isfinite(floor_at_hit), floor_at_hit, local_ground)
        relative_height = heights - reference
        # Distinct 3D rays may traverse one cell: free updates must not erase a
        # same-scan obstacle endpoint. Apply free evidence before hit evidence.
        free, hits = set(), set()
        frame_hits = {}
        self._reprocess_pending_rays(free)
        ray_clear = bool(self.c.map_ray_height_clear)
        start_xy = np.asarray(pose.xy(), float) + rot2(pose.yaw) @ mount[:2]
        start_z = self.c.base_height + self.ground_height + mount[2]
        near_low = float(self.c.map_near_band_low)
        near_high = float(self.c.min_obstacle_height)
        col_low, col_high = near_high, float(self.c.body_clearance)
        trusted_sources = (1, 3, 4, 5)
        view_xy = np.asarray(pose.xy(), float)
        rgb_ground = getattr(self, "visual_ground_mask", None)
        high_clear = set() if self.c.map_skip_high_clear else None
        above_near = set() if self.c.map_skip_above_near_clear else None
        for point_index, (p, height, relative) in enumerate(zip(xy, heights, relative_height)):
            end = self.cell(p)
            if not self.inside(end):
                continue
            ray = list(bresenham(self.cell(start_xy), end))
            prefix_free, unresolved = self._prefix_free(ray, np.asarray(start_xy, float),
                                                        start_z, np.asarray(p, float), height)
            free.update(prefix_free)
            for cell in unresolved:
                self.pending_ray_cells.setdefault((fid, cell),
                                                  (np.asarray(start_xy, float).copy(),
                                                   float(start_z), np.asarray(p, float).copy(),
                                                   float(height)))
            if not np.isfinite(relative):
                # Endpoint unknown, but the prefix evidence above still stands.
                if len(self.pending_endpoints) < 50000:
                    self.pending_endpoints.append((float(p[0]), float(p[1]), float(height),
                                                   None if timestamp is None else float(timestamp), fid))
                self.seen[end] = True
                continue
            obstacle = self.c.min_obstacle_height < relative < self.c.body_clearance
            if relative >= self.c.body_clearance and np.isfinite(floor_at_hit[point_index]):
                self.structure_votes[end] += 1
            # Suppress a false obstacle vote only when the same laser return
            # projects into a high-confidence RGB ground patch AND the aligned
            # return was verified against the local gravity/stance plane by
            # NavigationSupport.classify_ground(). Raised rocks and walls fail
            # that geometric check and remain occupied.
            if rgb_ground is not None and len(rgb_ground) == len(points):
                obstacle &= not bool(rgb_ground[point_index])
            ray_evidence = (ray if not (ev or ray_clear or self.c.map_skip_high_clear)
                            else list(bresenham(self.cell(start_xy), end)))
            if ev:
                if self._last_endpoint[end] != fid:
                    self._last_endpoint[end] = fid
                    self.evidence_endpoint[end] += 1
                    self._last_support[end] = fid
                    self.evidence_frame_support[end] += 1
                if obstacle and self._last_obstacle[end] != fid:
                    self._last_obstacle[end] = fid
                    self.evidence_obstacle_endpoint[end] += 1
                    self._record_view(end, view_xy)
                elif (not obstacle and relative >= self.c.body_clearance
                      and self._last_structure[end] != fid):
                    self._last_structure[end] = fid
                    self.evidence_structure_endpoint[end] += 1
            if (ev or ray_clear or high_clear is not None or above_near is not None) and len(ray_evidence) > 1:
                direction = np.asarray(p, float) - start_xy
                dz = height - start_z
                for cell in ray_evidence[:-1]:
                    if ev and self._last_prefix[cell] != fid:
                        self._last_prefix[cell] = fid
                        self.evidence_prefix[cell] += 1
                    if (floor_ref is None or not np.isfinite(floor_ref[cell])
                            or int(source_ref[cell]) not in trusted_sources):
                        if ev and self._last_near_partial[cell] != fid:
                            self._last_near_partial[cell] = fid
                            self.evidence_unknown_ground_prefix[cell] += 1
                        continue
                    floor_cell = float(floor_ref[cell])
                    lower = self.origin + np.asarray(cell) * self.c.resolution
                    upper = lower + self.c.resolution
                    parallel = np.abs(direction) < 1e-9
                    if np.any(parallel & ((start_xy < lower) | (start_xy >= upper))):
                        continue
                    safe = np.where(parallel, 1., direction)
                    t_low = (lower - start_xy) / safe
                    t_high = (upper - start_xy) / safe
                    entry = max(0., float(np.max(np.where(parallel, -np.inf, np.minimum(t_low, t_high)))))
                    exit_ = min(1., float(np.min(np.where(parallel, np.inf, np.maximum(t_low, t_high)))))
                    if exit_ <= entry:
                        continue
                    z_entry = start_z + entry * dz
                    z_exit = start_z + exit_ * dz
                    clearance_min = min(z_entry, z_exit) - floor_cell
                    clearance_max = max(z_entry, z_exit) - floor_cell
                    if high_clear is not None and clearance_min > self.c.body_clearance:
                        # Selective rule: this prefix cell was seen only from a
                        # height band the robot cannot occupy (e.g. over a wall
                        # top); do not let it clear near-ground occupancy.
                        high_clear.add(cell)
                    near_partial = clearance_max >= near_low and clearance_min <= near_high
                    near_full = (clearance_min <= near_low + .01
                                 and clearance_max >= near_high - .01)
                    collision_partial = clearance_max >= col_low and clearance_min <= col_high
                    collision_full = (clearance_min <= col_low + .01
                                      and clearance_max >= col_high - .01)
                    if above_near is not None and collision_partial and not near_partial:
                        # The ray sampled the collision band but never the
                        # near-ground band: it cannot prove the floor level
                        # itself is clear (e.g. skimming over a low wall).
                        above_near.add(cell)
                    if ev:
                        if near_partial and self._last_near_partial[cell] != fid:
                            self._last_near_partial[cell] = fid
                            self.evidence_near_partial[cell] += 1
                        if near_full and self._last_near_full[cell] != fid:
                            self._last_near_full[cell] = fid
                            self.evidence_near_clear[cell] += 1
                            self._record_view(cell, view_xy)
                        if collision_partial and self._last_collision_partial[cell] != fid:
                            self._last_collision_partial[cell] = fid
                            self.evidence_collision_partial[cell] += 1
                        if collision_full and self._last_collision_full[cell] != fid:
                            self._last_collision_full[cell] = fid
                            self.evidence_collision_full[cell] += 1
            # Only robot-height returns establish horizontal visibility. Ground
            # rays are also useful, but overhead rays cannot clear low obstacles.
            # Near-ground evidence is diagnostic: the full-run A/B rejected
            # replacing prefix clearing with it, so occupancy clearing keeps
            # its original criterion until a selective layer is validated.
            if np.isfinite(relative) and relative < self.c.body_clearance and not obstacle:
                free.add(end)
            if obstacle:
                hits.add(end)
                frame_hits.setdefault(end, []).append((float(p[0]), float(p[1]), float(height)))
            self.seen[end] = True
            if relative > max(self.c.min_obstacle_height, self.c.base_height-.05):
                # Clearance is relative to a measured support surface. A far
                # unclassified ramp must not leave a permanent fictitious roof.
                # Low obstacles belong in the reversible occupancy layer.
                # Treating a 10cm rock/ramp edge as a roof left an irreversible
                # low ceiling after later rays had correctly cleared the cell.
                if np.isfinite(self.floor[end]):
                    self.ceiling[end] = min(self.ceiling[end], height)
                    self.headroom[end] = self.ceiling[end]-self.floor[end]
        if high_clear:
            free.difference_update(high_clear)
        if above_near:
            free.difference_update(above_near)
        for cell in free - set(frame_hits):
            self._record_evidence(fid, cell, "free")
            self.seen[cell] = True
        for cell, entries in frame_hits.items():
            self._record_evidence(fid, cell, "obstacle")
            self.occupancy_contributions.setdefault((fid, cell), list(entries))
        # The standing footprint is direct evidence of local support.
        radius = int(np.ceil(self.c.robot_radius / self.c.resolution))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ij = (start[0] + dx, start[1] + dy)
                if self.inside(ij) and np.hypot(dx, dy) * self.c.resolution <= self.c.robot_radius:
                    self.seen[ij] = True
                    if ij not in hits:
                        self._record_evidence(fid, ij, "free")
                        if not np.isfinite(self.floor[ij]):
                            self.floor[ij] = self.ground_height
                            self.floor_source[ij] = 1
                        if ev and self._last_floor[ij] != fid:
                            self._last_floor[ij] = fid
                            self.evidence_floor_footprint[ij] += 1
        self._reclassify_pending()

    def _connected_surface(self, xy, heights, centre, start):
        """Recover a short measured ramp that meets an established floor downhill."""
        empty = np.empty((0, 3))
        span = 1.5
        local = (np.linalg.norm(xy - centre, axis=1) <= span) & (abs(heights - self.ground_height) < .32)
        scans = np.column_stack((xy[local], heights[local], np.full(local.sum(), self._frame_seq)))
        self.surface_scans.append(scans)
        if len(self.surface_scans) < 3:
            return empty
        points = np.concatenate(self.surface_scans)
        points = points[np.linalg.norm(points[:, :2] - centre, axis=1) <= span]
        if len(points) < 8:
            return empty
        radius = int(np.ceil(span / self.c.resolution))
        lo = np.maximum(0, np.asarray(start) - radius)
        hi = np.minimum(self.n, np.asarray(start) + radius + 1)
        floor = self.floor[lo[0]:hi[0], lo[1]:hi[1]]
        known = np.argwhere(np.isfinite(floor))
        if len(known) < 3:
            return empty
        anchors_xy = self.origin + (known + lo + .5) * self.c.resolution
        anchors_z = floor[tuple(known.T)]
        cells = np.floor((points[:, :2] - self.origin) / self.c.resolution).astype(int)
        _, cell_ids = np.unique(cells, axis=0, return_inverse=True)
        lower = np.full(cell_ids.max() + 1, np.inf)
        np.minimum.at(lower, cell_ids, points[:, 2])
        unknown = ~np.isfinite(self.floor[tuple(cells.T)])
        points = points[unknown & (points[:, 2] <= lower[cell_ids] + .035)]
        if len(points) < 8:
            return empty
        cells = np.floor((points[:, :2] - self.origin) / self.c.resolution).astype(int)
        _, candidate_ids = np.unique(cells, axis=0, return_index=True)
        radius = min(self.c.ground_max_gap, .30)
        neighbourhoods = cKDTree(points[:, :2]).query_ball_point(points[candidate_ids, :2], radius)
        recovered = []
        for index, neighbours in zip(candidate_ids, neighbourhoods):
            cluster = points[neighbours]
            if len(cluster) < 8 or len(np.unique(cluster[:, 3])) < 3:
                continue
            xy = cluster[:, :2] - points[index, :2]
            if np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)[-1] < .025 * np.sqrt(len(xy)):
                continue
            design = np.column_stack((xy, np.ones(len(xy))))
            plane = np.linalg.lstsq(design, cluster[:, 2], rcond=None)[0]
            slope = np.linalg.norm(plane[:2])
            if not .13 <= slope <= np.tan(np.deg2rad(23.)):
                continue
            if np.quantile(abs(cluster[:, 2] - design @ plane), .85) > .024:
                continue
            if np.ptp(xy @ (plane[:2] / slope)) < .075:
                continue
            offset = anchors_xy - points[index, :2]
            downhill = offset @ (-plane[:2] / slope)
            lateral = np.linalg.norm(offset + downhill[:, None] * (plane[:2] / slope), axis=1)
            prediction = offset @ plane[:2] + plane[2]
            # Permit one short gap at the toe of a measured ramp. A flat
            # raised surface or a wall cannot meet these slope/coverage tests.
            attached = ((downhill > .06) & (downhill <= .48) & (lateral < .16)
                        & (abs(prediction - anchors_z) < .075)
                        & (anchors_z < plane[2] - .045))
            if np.count_nonzero(attached) >= 2:
                recovered.append(np.r_[points[index, :2], plane[2]])
        return np.asarray(recovered).reshape(-1, 3)

    def _relative_for_endpoints(self, heights, floor_at_hit):
        """Local-ground-relative height; NaN means pending (no fallback)."""
        return heights - floor_at_hit

    def _record_evidence(self, frame, cell, kind):
        frames = self.cell_frames.setdefault(cell, {})
        previous = frames.get(frame)
        if previous == "obstacle" and kind == "free":
            return  # obstacle wins inside the same original frame
        frames[frame] = kind
        cutoff = frame - self.evidence_window
        for key in [key for key in frames if key < cutoff]:
            frames.pop(key)
        self._publish_cell(cell)

    def _publish_cell(self, cell):
        frames = self.cell_frames.get(cell, {})
        free = sum(-.45 for kind in frames.values() if kind == "free")
        hit = sum(1.25 for kind in frames.values() if kind == "obstacle")
        self.free_evidence[cell] = free
        self.hit_evidence_total[cell] = hit
        self.logodds[cell] = float(np.clip(free + hit, -4., 4.))

    def _local_ground(self, xy):
        """Local plane through connected measured floor, never robot-foot height."""
        cell = self.cell(xy)
        radius = max(2, int(np.ceil(2.5 / self.c.resolution)))
        i0, i1 = max(0, cell[0] - radius), min(self.n, cell[0] + radius + 1)
        j0, j1 = max(0, cell[1] - radius), min(self.n, cell[1] + radius + 1)
        patch = self.floor[i0:i1, j0:j1]
        sources = self.floor_source[i0:i1, j0:j1]
        finite = np.isfinite(patch) & (sources >= 1)
        if finite.sum() < 4:
            return None
        ii, jj = np.nonzero(finite)
        world = self.origin + (np.column_stack([ii + i0, jj + j0]) + .5) * self.c.resolution
        design = np.column_stack([world, np.ones(len(world))])
        coefficients, *_ = np.linalg.lstsq(design, patch[finite], rcond=None)
        residual = patch[finite] - design @ coefficients
        if float(np.max(np.abs(residual))) > .08:
            return None
        query = self.origin + (np.asarray(cell) + .5) * self.c.resolution
        return float(np.r_[query, 1.] @ coefficients)

    def _reprocess_pending_rays(self, free):
        remaining = {}
        for (frame, cell), (start_xy, start_z, end_xy, end_z) in self.pending_ray_cells.items():
            ground = self.floor[cell]
            if not np.isfinite(ground):
                ground = self._local_ground(self.world(cell))
            if ground is None or not np.isfinite(ground):
                remaining[(frame, cell)] = (start_xy, start_z, end_xy, end_z)
                continue
            # Settle this cell once for this original echo; never repeat it.
            ray = list(bresenham(self.cell(start_xy), self.cell(end_xy)))
            if cell not in ray[:-1]:
                continue
            index = ray.index(cell)
            lower = self.origin + np.asarray(cell) * self.c.resolution
            upper = lower + self.c.resolution
            direction = end_xy - start_xy
            dz = end_z - start_z
            parallel = np.abs(direction) < 1e-9
            if np.any(parallel & ((start_xy < lower) | (start_xy >= upper))):
                continue
            safe = np.where(parallel, 1., direction)
            t_low = (lower - start_xy) / safe
            t_high = (upper - start_xy) / safe
            entry = max(0., float(np.max(np.where(parallel, -np.inf, np.minimum(t_low, t_high)))))
            exit_ = min(1., float(np.min(np.where(parallel, np.inf, np.maximum(t_low, t_high)))))
            if exit_ <= entry:
                continue
            clearance_min = min(start_z + entry * dz, start_z + exit_ * dz) - ground
            clearance_max = max(start_z + entry * dz, start_z + exit_ * dz) - ground
            if clearance_min < self.c.body_clearance and clearance_max > 0.:
                self._record_evidence(frame, cell, "free")
        self.pending_ray_cells = remaining

    def _prefix_free(self, ray, start_xy, start_z, end_xy, end_z):
        """Free evidence along a ray using each cell's own ground; defer gaps."""
        free = set()
        unresolved = []
        direction = np.asarray(end_xy, float) - start_xy
        dz = float(end_z) - start_z
        for cell in ray[:-1]:
            ground = self.floor[cell]
            if not np.isfinite(ground):
                ground = self._local_ground(self.world(cell))
            if ground is None or not np.isfinite(ground):
                unresolved.append(cell)
                continue
            lower = self.origin + np.asarray(cell) * self.c.resolution
            upper = lower + self.c.resolution
            parallel = np.abs(direction) < 1e-9
            if np.any(parallel & ((start_xy < lower) | (start_xy >= upper))):
                continue
            safe = np.where(parallel, 1., direction)
            t_low = (lower - start_xy) / safe
            t_high = (upper - start_xy) / safe
            entry = max(0., float(np.max(np.where(parallel, -np.inf, np.minimum(t_low, t_high)))))
            exit_ = min(1., float(np.min(np.where(parallel, np.inf, np.maximum(t_low, t_high)))))
            if exit_ <= entry:
                continue
            z_entry = start_z + entry * dz
            z_exit = start_z + exit_ * dz
            clearance_min = min(z_entry, z_exit) - ground
            clearance_max = max(z_entry, z_exit) - ground
            # Only a ray that actually passes through the robot's height band
            # can clear near-ground cells; a ray over a wall only sees above it.
            if clearance_min < self.c.body_clearance and clearance_max > 0.:
                free.add(cell)
        return free, unresolved

    def _reclassify_contributions(self):
        """Undo one original frame-cell contribution only when no endpoint supports it."""
        for key, entries in list(self.occupancy_contributions.items()):
            cell = key[1]
            ground = self.floor[cell]
            if not np.isfinite(ground):
                ground = self._local_ground(self.world(cell))
            if ground is None or not np.isfinite(ground):
                continue
            keep = [entry for entry in entries if
                    self.c.min_obstacle_height < entry[2] - ground < self.c.body_clearance]
            if keep:
                self.occupancy_contributions[key] = keep
                continue
            frame = key[0]
            self.occupancy_contributions.pop(key, None)
            frames = self.cell_frames.get(cell)
            if frames and frames.get(frame) == "obstacle":
                del frames[frame]
                if any(abs(entry[2] - ground) <= .04 for entry in entries):
                    self._record_evidence(frame, cell, "free")
                else:
                    self._publish_cell(cell)

    def _reclassify_pending(self):
        """Re-score stored endpoints once a local ground exists."""
        remaining = []
        for x, y, height, timestamp, frame in self.pending_endpoints:
            cell = self.cell((x, y))
            if not self.inside(cell):
                continue
            ground = self.floor[cell]
            if not np.isfinite(ground):
                ground = self._local_ground((x, y))
            if ground is None or not np.isfinite(ground):
                remaining.append((x, y, height, timestamp, frame))
                continue
            relative = height - ground
            if relative >= self.c.body_clearance:
                self.structure_votes[cell] += 1
            elif relative > self.c.min_obstacle_height:
                self._record_evidence(frame, cell, "obstacle")
                self.occupancy_contributions.setdefault((frame, cell), []).append((x, y, height))
            # Ground-level pending returns confirm the floor via the existing
            # ground evidence; they do not silently become free space.
        self.pending_endpoints = remaining

    def obstacle_points(self, points, pose):
        """Use the same local elevation model in DWA as in the global map."""
        xy = transform_xy(points, pose)
        cells = np.floor((xy - self.origin) / self.c.resolution).astype(int)
        inside = (cells >= 0).all(axis=1) & (cells < self.n).all(axis=1)
        floor = np.full(len(points), np.nan)
        ix = np.flatnonzero(inside)
        observed = self.floor[tuple(cells[ix].T)]
        floor[ix] = observed
        for index in ix[~np.isfinite(observed)]:
            local = self._local_ground(xy[index])
            if local is not None:
                floor[index] = local
        heights = points[:, 2] + self.c.base_height + self.ground_height
        height = heights - floor
        return points[np.isfinite(height) & (height > self.c.min_obstacle_height)
                      & (height < self.c.body_clearance)]

    def add_hazard(self, polygon):
        cells = np.array([self.cell(p) for p in polygon])
        mask = np.zeros_like(self.hazard_votes)
        cv2.fillPoly(mask, [cells[:, ::-1].astype(np.int32)], 1)
        # Account for monocular boundary projection and sparse lidar sampling.
        margin = max(1, int(np.ceil(.08 / self.c.resolution)))
        mask = cv2.dilate(mask, np.ones((2*margin+1, 2*margin+1), np.uint8))
        self.hazard_votes = np.minimum(10, self.hazard_votes + mask)
        self._has_hazards = bool((self.hazard_votes >= 2).any())
        self._free_cache = self._surface_safe = None

    def traversable(self) -> np.ndarray:
        if self._free_cache is not None:
            return self._free_cache
        occupied = self.logodds > .5
        hazards = self.hazard_votes >= 2
        # Fill small sampling gaps only next to measured floor; never invent a
        # broad supported surface from a wall-only scan.
        support = binary_dilation(np.isfinite(self.floor), iterations=1)
        # Fit the tangent around each cell without blending a neighbouring wall
        # or raised rock into the flat floor beside it. Otherwise slope masking
        # followed by robot inflation effectively counts the wall twice.
        # Unobserved map padding has no height gradients. Work on the measured
        # bounding window plus the full two-cell stencil, preserving the same
        # NaN boundaries while avoiding millions of empty-cell operations.
        rows, columns = np.nonzero(np.isfinite(self.floor))
        if len(rows):
            window = (slice(max(0, rows.min()-2), min(self.n, rows.max()+3)),
                      slice(max(0, columns.min()-2), min(self.n, columns.max()+3)))
        else:
            window = (slice(0, 1), slice(0, 1))
        floor = self.floor[window]
        ii, jj = np.mgrid[window[0], window[1]]
        xy = self.origin+(np.stack([ii, jj], axis=-1)+.5)*self.c.resolution
        slope_limits = self.c.slope_limit(xy)
        cxx = np.zeros_like(floor, dtype=float)
        cyy, cxy, cxz, cyz = [cxx.copy() for _ in range(4)]
        step_edge = np.zeros_like(floor, dtype=bool)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if not (dx or dy):
                    continue
                neighbour = np.roll(floor, (-dx, -dy), axis=(0, 1))
                if dx > 0:
                    neighbour[-dx:, :] = np.nan
                elif dx < 0:
                    neighbour[:-dx, :] = np.nan
                if dy > 0:
                    neighbour[:, -dy:] = np.nan
                elif dy < 0:
                    neighbour[:, :-dy] = np.nan
                delta = neighbour - floor
                distance = np.hypot(dx, dy) * self.c.resolution
                local_limit = np.minimum(slope_limits, np.roll(slope_limits, (-dx, -dy), axis=(0, 1)))
                continuous = np.isfinite(delta) & (abs(delta) <= np.tan(np.deg2rad(local_limit)) * distance + .008)
                x, y = dx * self.c.resolution, dy * self.c.resolution
                cxx += continuous * x*x
                cyy += continuous * y*y
                cxy += continuous * x*y
                dz = np.where(continuous, delta, 0.)
                cxz += dz * x
                cyz += dz * y
                if max(abs(dx), abs(dy)) == 1:
                    step_edge |= np.isfinite(delta) & (delta < -self.c.max_step_height)
        determinant = cxx * cyy - cxy * cxy
        valid = determinant > 1e-8
        gx = np.divide(cxz * cyy - cyz * cxy, determinant, out=np.zeros_like(cxx), where=valid)
        gy = np.divide(cyz * cxx - cxz * cxy, determinant, out=np.zeros_like(cxx), where=valid)
        steep = np.zeros_like(self.seen)
        steep[window] = (np.hypot(gx, gy) > np.tan(np.deg2rad(slope_limits))) | step_edge
        insufficient_headroom = self.headroom < self.c.body_clearance
        body_radius = min(self.c.footprint_half_length, self.c.footprint_half_width) if self.c.footprint_half_length else self.c.robot_radius
        clearance_radius = body_radius + self.c.safety_margin
        radius = int(np.ceil(clearance_radius / self.c.resolution))
        xx, yy = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = (xx*xx + yy*yy) * self.c.resolution**2 <= clearance_radius**2
        inflated = binary_dilation(occupied | insufficient_headroom, structure=disk)
        # A ground grade constrains where the feet can land, not the body
        # clearance from a solid wall. Exact footprint checks still reject a
        # step whose contacts would land on the slope.
        configured_foot_radius = min(self.c.hazard_foot_half_length, self.c.hazard_foot_half_width)
        step_radius = configured_foot_radius or body_radius
        step_disk = (xx*xx + yy*yy) * self.c.resolution**2 <= step_radius**2
        inflated |= binary_dilation(steep, structure=step_disk)
        foot_radius = min(self.c.hazard_foot_half_length, self.c.hazard_foot_half_width) if self.c.hazard_foot_half_length else clearance_radius
        fr = int(np.ceil(foot_radius/self.c.resolution))
        fx, fy = np.ogrid[-fr:fr+1, -fr:fr+1]
        foot_disk = (fx*fx+fy*fy)*self.c.resolution**2 <= foot_radius**2
        inflated_hazards = binary_dilation(hazards, structure=foot_disk)
        measured_safe = self.seen & (self.logodds < 0.) & support & ~(occupied | steep | insufficient_headroom)
        self._surface_safe = measured_safe
        allowed = self.seen & (self.logodds < 0.) & support & ~inflated & ~inflated_hazards
        allowed[[0, -1], :] = False
        allowed[:, [0, -1]] = False
        self._free_cache = allowed
        return allowed

    def footprint_safe(self, xy, yaw, offsets):
        """Check the complete oriented footprint, including ground support."""
        from .geometry import rot2
        if self._surface_safe is None:
            self.traversable()
        points = offsets @ rot2(yaw).T + xy
        cells = np.floor((points - self.origin) / self.c.resolution).astype(int)
        if (cells < 0).any() or (cells >= self.n).any():
            return False
        if not self._surface_safe[tuple(cells.T)].all():
            return False
        return bool(self.hazard_contacts_safe(np.asarray(xy)[None], np.array([yaw]), offsets)[0])

    def hazard_contacts_safe(self, xy, yaw, body_offsets):
        if not self._has_hazards:
            return np.ones(len(xy), bool)
        offsets = body_offsets
        if self.c.hazard_foot_half_length:
            xx, yy = np.meshgrid(np.linspace(-self.c.hazard_foot_half_length, self.c.hazard_foot_half_length, 7),
                                 np.linspace(-self.c.hazard_foot_half_width, self.c.hazard_foot_half_width, 7))
            offsets = np.column_stack([xx.ravel(), yy.ravel()])
        cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        x = xy[:, 0, None]+cs*offsets[None, :, 0]-sn*offsets[None, :, 1]
        y = xy[:, 1, None]+sn*offsets[None, :, 0]+cs*offsets[None, :, 1]
        i = np.floor((x-self.origin[0])/self.c.resolution).astype(int)
        j = np.floor((y-self.origin[1])/self.c.resolution).astype(int)
        inside = ((i >= 0)&(i < self.n)&(j >= 0)&(j < self.n)).all(axis=1)
        return inside & (self.hazard_votes[np.clip(i, 0, self.n-1), np.clip(j, 0, self.n-1)] < 2).all(axis=1)

    def footprints_safe(self, xy, yaw, offsets):
        if self._surface_safe is None:
            self.traversable()
        cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        x = xy[:, 0, None] + cs * offsets[None, :, 0] - sn * offsets[None, :, 1]
        y = xy[:, 1, None] + sn * offsets[None, :, 0] + cs * offsets[None, :, 1]
        i = np.floor((x - self.origin[0]) / self.c.resolution).astype(int)
        j = np.floor((y - self.origin[1]) / self.c.resolution).astype(int)
        inside = ((i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)).all(axis=1)
        return inside & self._surface_safe[np.clip(i, 0, self.n-1), np.clip(j, 0, self.n-1)].all(axis=1) & self.hazard_contacts_safe(xy, yaw, offsets)

    def save(self, path):
        arrays = dict(logodds=self.logodds, seen=self.seen, floor=self.floor,
                      ceiling=self.ceiling, headroom=self.headroom,
                      hazard_votes=self.hazard_votes, floor_source=self.floor_source,
                      structure_votes=self.structure_votes,
                      overhead_votes=self.overhead_votes,
                      free_evidence=self.free_evidence,
                      hit_evidence_total=self.hit_evidence_total,
                      pending_endpoint_count=np.asarray(len(self.pending_endpoints)),
                      origin=self.origin, resolution=self.c.resolution)
        if getattr(self, "_evidence_enabled", False):
            for name in ("evidence_endpoint", "evidence_prefix", "evidence_near_clear",
                         "evidence_near_partial", "evidence_collision_partial",
                         "evidence_collision_full", "evidence_unknown_ground_prefix",
                         "evidence_structure_endpoint", "evidence_obstacle_endpoint",
                         "evidence_frame_support", "evidence_floor_footprint",
                         "evidence_floor_interpolated", "evidence_floor_measured"):
                arrays[name] = getattr(self, name)
        np.savez_compressed(path, **arrays)
