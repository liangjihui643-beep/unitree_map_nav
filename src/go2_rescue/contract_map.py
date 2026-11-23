"""Candidate-based contract map prototype for a same-observation A/B.

Rules enforced here:
- one azimuth's nine pitch candidates are mutually exclusive interpretations;
  a committed echo contributes at most one cell of one evidence class;
- echoes whose ground is not yet known are stored as pending;
- when ground later becomes known, pending echoes are re-classified and may
  then contribute exactly one evidence, never nine;
- trusted ground grows by bounded local completion from near-body seeds;
- walls are confirmed independently from multi-view vertical candidate bundles,
  one vote per bundle, and kept separate from ground/free layers.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial.transform import Rotation

from .config import Config


class ContractMap:
    def __init__(self, config: Config, size=20., resolution=.1):
        self.c = config
        self.resolution = resolution
        self.origin = np.array([-2., -2.])
        self.shape = (int(round(size / resolution)), int(round(size / resolution)))
        self.ground = np.full(self.shape, np.nan, dtype=np.float32)
        self.ground_hits = np.zeros(self.shape, dtype=np.uint8)
        self.near_body_seed = np.zeros(self.shape, dtype=bool)
        self.support_hits = np.zeros(self.shape, dtype=np.uint8)
        self.last_support_frame = np.full(self.shape, -1, dtype=np.int32)
        self.ground_views = {}
        self.wall = np.zeros(self.shape, dtype=bool)
        self.wall_votes = np.zeros(self.shape, dtype=np.uint16)
        self.wall_segments = []
        self.occupied = np.zeros(self.shape, dtype=np.uint16)
        self.free = np.zeros(self.shape, dtype=bool)
        self.seen = np.zeros(self.shape, dtype=bool)
        self.overhead = np.zeros(self.shape, dtype=np.uint16)
        self.undetermined = np.zeros(self.shape, dtype=np.uint16)
        self.pending = []
        self.pending_cells = np.zeros(self.shape, dtype=np.uint16)
        self.pending_dropped = 0
        self.max_pending = 50000
        self.frame_index = 0
        self.elevations = np.deg2rad(np.linspace(
            -config.lidar_vertical_fov_deg / 2, config.lidar_vertical_fov_deg / 2, 9))

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy, float)[:2] - self.origin) / self.resolution).astype(int))

    def inside(self, cell):
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def _endpoint(self, azimuth, elevation, distance, pose, rotation, mount):
        direction = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth),
                              np.sin(elevation)])
        return np.asarray(pose, float) + rotation @ (mount + direction * distance)

    def _classify(self, cell, world_z, pose_z):
        ground = self.ground[cell]
        if not np.isfinite(ground):
            return "unknown"
        relative = world_z - ground
        return "ground" if relative < self.c.min_obstacle_height else (
            "obstacle" if relative < self.c.body_clearance else "overhead")

    def published_free(self):
        """Free for planning; mutually exclusive with occupancy and wall."""
        return ((self.free | self.near_body_seed) & (self.occupied == 0)
                & (self.wall == 0) & (self.overhead == 0))

    def published_occupied(self):
        return (self.occupied > 0) | self.wall

    def free_occupied_conflicts(self):
        return int((self.published_free() & self.published_occupied()).sum())

    def _commit(self, cell, kind):
        if kind == "ground":
            self.free[cell] = True
        elif kind == "obstacle":
            self.occupied[cell] += 1
        elif kind == "overhead":
            self.overhead[cell] += 1
        else:
            self.undetermined[cell] += 1

    def _store_pending(self, record):
        if len(self.pending) >= self.max_pending:
            self.pending.pop(0)
            self.pending_dropped += 1
        self.pending.append(record)

    def _refresh_pending_cells(self):
        self.pending_cells.fill(0)
        for record in self.pending:
            for cell, _ in record["candidates"]:
                self.pending_cells[cell] += 1

    def _recompute_pending(self):
        remaining = []
        for record in self.pending:
            kinds = [self._classify(cell, z, record["pose_z"])
                     for cell, z in record["candidates"]]
            cells = {cell for cell, _ in record["candidates"]}
            if "unknown" in kinds or len(cells) != 1 or len(set(kinds)) != 1:
                remaining.append(record)
                continue
            cell = next(iter(cells))
            self._commit(cell, kinds[0])
        self.pending = remaining

    def update(self, obs, pose, quaternion, frame_index=None):
        self.frame_index = int(self.frame_index + 1 if frame_index is None else frame_index)
        obs = np.asarray(obs, float).reshape(144)
        normalized, mask = obs[:72], obs[72:].astype(bool)
        rotation = Rotation.from_quat(np.roll(np.asarray(quaternion, float), -1)).as_matrix()
        mount = np.asarray(self.c.lidar_mount, float)
        pose = np.asarray(pose, float)
        azimuths = -np.pi + np.flatnonzero(mask) * (2 * np.pi / 72)
        ranges = normalized[mask] * self.c.max_range
        for azimuth, distance in zip(azimuths, ranges):
            candidates = []
            for elevation in self.elevations:
                endpoint = self._endpoint(azimuth, elevation, distance, pose, rotation, mount)
                height = endpoint[2] - pose[2]
                if height < self.c.lidar_height_floor:
                    continue
                cell = self.cell(endpoint[:2])
                if not self.inside(cell):
                    continue
                self.seen[cell] = True
                candidates.append((cell, float(endpoint[2])))
            if not candidates:
                continue
            self._ground_evidence(candidates, pose, azimuth)
            kinds = [self._classify(cell, z, pose[2]) for cell, z in candidates]
            cells = {cell for cell, _ in candidates}
            if len(cells) == 1 and kinds and "unknown" not in kinds and len(set(kinds)) == 1:
                # Exactly one exclusive interpretation: one evidence.
                self._commit(next(iter(cells)), kinds[0])
            else:
                # Ambiguous or ground-unknown: store the whole echo, do not
                # let nine candidates vote independently.
                self._store_pending({"candidates": candidates, "pose_z": float(pose[2]),
                                     "azimuth": float(azimuth), "range": float(distance),
                                     "frame": int(self.frame_index),
                                     "view_xy": np.asarray(pose[:2], float).copy()})
        self._complete_ground()
        self._recompute_pending()
        if self.frame_index % 10 == 0:
            self.wall.fill(False)
            self.wall_votes.fill(0)
            self.wall_segments.clear()
            self._confirm_walls()
        self._refresh_pending_cells()

    def _has_ground_neighbor(self, cell):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if not (di or dj):
                    continue
                other = (cell[0] + di, cell[1] + dj)
                if self.inside(other) and np.isfinite(self.ground[other]):
                    return True
        return False

    def _ground_evidence(self, candidates, pose, azimuth):
        """One echo contributes at most one ground vote, and only from support
        or a consistent multi-view low surface connected to known ground."""
        cell, z = min(candidates, key=lambda item: item[1])
        if z - pose[2] > .20:
            return
        key = (self.frame_index, round(float(azimuth), 6))
        views = self.ground_views.setdefault(cell, {})
        if key in views:
            return
        views[key] = (np.asarray(pose[:2], float).copy(), float(z))
        self.ground_hits[cell] = min(255, len(views))
        if np.isfinite(self.ground[cell]):
            return
        world = self.origin + (np.asarray(cell) + .5) * self.resolution
        if np.linalg.norm(world - pose[:2]) <= min(float(self.c.robot_radius), .22):
            # One nearby echo is only a hypothesis; require support in at least
            # two distinct frames before calling it verifiable body support.
            if self.last_support_frame[cell] != self.frame_index:
                self.last_support_frame[cell] = self.frame_index
                self.support_hits[cell] = min(255, int(self.support_hits[cell]) + 1)
            if self.support_hits[cell] >= 2:
                self.ground[cell] = float(z)
                self.near_body_seed[cell] = True
            return
        if len(views) < 3:
            return
        positions, heights = [], []
        for view, height in views.values():
            if all(np.linalg.norm(view - other) >= .15 for other in positions):
                positions.append(view)
                heights.append(height)
        if len(positions) >= 3 and np.ptp(heights) <= .06 and self._has_ground_neighbor(cell):
            # Different views, consistent low surface, connected to known ground;
            # this is not free-space propagation by itself.
            self.ground[cell] = float(np.median(heights))

    def _complete_ground(self):
        """Bounded local completion from seeds; stop at jumps or missing data."""
        queue = [tuple(cell) for cell in np.argwhere(np.isfinite(self.ground))]
        for cell in queue:
            current = float(self.ground[cell])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not (di or dj):
                        continue
                    other = (cell[0] + di, cell[1] + dj)
                    if not self.inside(other) or np.isfinite(self.ground[other]):
                        continue
                    views = self.ground_views.get(other)
                    if not views:
                        continue
                    heights = np.asarray([value[1] for value in views.values()], float)
                    if len(heights) < 2 or np.ptp(heights) > .10:
                        continue
                    height = float(np.median(heights))
                    if abs(height - current) > .12:
                        continue
                    self.ground[other] = height
                    queue.append(other)

    @staticmethod
    def _slope_explains(xy, z):
        """Return True when one bundle's candidates fit a continuous slope."""
        spread = np.ptp(xy, axis=0)
        if max(spread) < .10:
            return False
        design = np.column_stack([xy, np.ones(len(xy))])
        coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
        residual = z - design @ coefficients
        slope = float(np.hypot(coefficients[0], coefficients[1]))
        return float(np.max(np.abs(residual))) < .04 and slope < .45

    def _confirm_walls(self):
        """Fit a surface from multi-view bundles, then take one candidate per bundle.

        Height span alone is not a wall criterion: ramp-like bundles that fit a
        continuous slope are excluded before fitting. A wall is only rasterized
        over supported inlier spans; gaps split segments instead of being
        bridged, and two frames from the same 0.25 m view bin count once.
        """
        observations = []
        for record in self.pending:
            candidates = record["candidates"]
            if len(candidates) < 2:
                continue
            provisional = record["pose_z"] - self.c.base_height
            filtered = list(candidates)
            xy = np.asarray([self.origin + (np.asarray(cell) + .5) * self.resolution
                             for cell, _ in filtered])
            z = np.asarray([value for _, value in filtered])
            if np.ptp(xy, axis=0).max() > .25:
                continue
            # Per-bundle slope fitting is not used as a wall exclusion: a wall
            # seen obliquely can also look locally planar. Ground competition is
            # applied later through the local ground and height band.
            represented = int(np.argmin(np.abs(z - np.median(z))))
            observations.append({"candidates": filtered, "xy": xy, "z": z,
                                 "rep": filtered[represented][0],
                                 "rep_xy": xy[represented],
                                 "pose_z": float(record["pose_z"]),
                                 "frame": int(record.get("frame", -1)),
                                 "view": np.asarray(record.get("view_xy", [0., 0.]), float)})
        if len(observations) < 2:
            return
        # Independent per-cell support: the same cell must be seen from at
        # least two different 0.25 m view bins. This keeps short wall pieces
        # that do not form a long PCA segment.
        cell_views = {}
        cell_heights = {}
        for observation in observations:
            provisional = observation["pose_z"] - self.c.base_height
            options = []
            for cell, height in observation["candidates"]:
                ground = self.ground[cell]
                reference = ground if np.isfinite(ground) else provisional
                if reference + .08 <= height <= reference + .60:
                    options.append((cell, height))
            if not options:
                continue
            cell, height = min(options, key=lambda item: abs(item[1] - (provisional + .30)))
            view_bin = tuple(np.floor(observation["view"] / .25).astype(int))
            cell_views.setdefault(cell, set()).add(view_bin)
            cell_heights.setdefault(cell, []).append(height)
        for cell, bins in cell_views.items():
            if len(bins) < 2:
                continue
            ground = self.ground[cell]
            if np.isfinite(ground) and min(cell_heights[cell]) - ground < self.c.min_obstacle_height:
                continue
            self.wall[cell] = True
        grid = np.zeros(self.shape, bool)
        for observation in observations:
            grid[observation["rep"]] = True
        grid = binary_dilation(grid, iterations=1)
        groups, count = label(grid)
        for group in range(1, count + 1):
            members = [item for item in observations if groups[item["rep"]] == group]
            views = {tuple(np.floor(item["view"] / .25).astype(int)) for item in members}
            if len(members) < 2 or len(views) < 2:
                continue
            all_xy = np.vstack([item["rep_xy"] for item in members])
            centre = all_xy.mean(axis=0)
            covariance = np.cov((all_xy - centre).T)
            values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
            direction = vectors[:, int(np.argmax(values))]
            normal = np.array([-direction[1], direction[0]])
            support = []
            for item in members:
                offsets = item["xy"] - centre
                residual = np.abs(offsets @ normal)
                index = int(np.argmin(residual))
                if residual[index] > .25:
                    continue
                cell, height = item["candidates"][index]
                ground = self.ground[cell]
                if np.isfinite(ground) and height - ground < self.c.min_obstacle_height:
                    # A ground/slope explanation wins over the wall hypothesis.
                    continue
                support.append({"cell": cell, "z": float(height),
                                "frame": item["frame"], "view": item["view"]})
            if len(support) < 2:
                continue
            inliers = np.asarray([self.origin + (np.asarray(item["cell"]) + .5) * self.resolution
                                  for item in support])
            centre = inliers.mean(axis=0)
            covariance = np.cov((inliers - centre).T)
            values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
            direction = vectors[:, int(np.argmax(values))]
            normal = np.array([-direction[1], direction[0]])
            t_values = (inliers - centre) @ direction
            order = np.argsort(t_values)
            anchors = sorted(((t_values[index], support[index]) for index in order),
                             key=lambda item: item[0])
            spans = []
            start = previous = anchors[0]
            for anchor in anchors[1:]:
                if anchor[0] - previous[0] > .20:
                    spans.append((start[0], previous[0]))
                    start = anchor
                previous = anchor
            spans.append((start[0], previous[0]))
            zmin = min(item["z"] for item in support)
            zmax = max(item["z"] for item in support)
            segment_cells = []
            for t0, t1 in spans:
                lo = np.floor((inliers.min(axis=0) - self.origin - .3) / self.resolution).astype(int)
                hi = np.ceil((inliers.max(axis=0) - self.origin + .3) / self.resolution).astype(int)
                for i in range(max(0, lo[0]), min(self.shape[0], hi[0] + 1)):
                    for j in range(max(0, lo[1]), min(self.shape[1], hi[1] + 1)):
                        point = self.origin + (np.array([i, j]) + .5) * self.resolution
                        offset = point - centre
                        t = float(offset @ direction)
                        if t < t0 - .05 or t > t1 + .05:
                            continue
                        if abs(offset @ normal) > .15:
                            continue
                        ground = self.ground[i, j]
                        if np.isfinite(ground):
                            if zmin - ground > self.c.body_clearance:
                                self.overhead[i, j] += 1
                                continue
                            if zmax - ground < self.c.min_obstacle_height:
                                continue
                        self.wall[i, j] = True
                        segment_cells.append((i, j))
                if segment_cells:
                    self.wall_segments.append({"cells": segment_cells, "views": len(views),
                                               "height_m": [zmin, zmax],
                                               "span_m": [float(t0), float(t1)]})
            for item in support:
                cell = item["cell"]
                ground = self.ground[cell]
                if np.isfinite(ground):
                    if item["z"] - ground > self.c.body_clearance:
                        self.overhead[cell] += 1
                        continue
                    if item["z"] - ground < self.c.min_obstacle_height:
                        continue
                self.wall[cell] = True
                self.wall_votes[cell] += 1
                if cell not in segment_cells:
                    segment_cells.append(cell)
            if segment_cells:
                self.wall_segments.append({"cells": segment_cells, "views": len(views),
                                           "height_m": [zmin, zmax],
                                           "span_m": [float(spans[0][0]), float(spans[-1][1])]})

    def finalize(self):
        self._complete_ground()
        self._recompute_pending()
        self.wall.fill(False)
        self.wall_votes.fill(0)
        self.wall_segments.clear()
        self._confirm_walls()
        self._refresh_pending_cells()


class LegacyMap(ContractMap):
    """Fixed-centre-elevation baseline: one point per valid azimuth."""

    def update(self, obs, pose, quaternion):
        obs = np.asarray(obs, float).reshape(144)
        normalized, mask = obs[:72], obs[72:].astype(bool)
        rotation = Rotation.from_quat(np.roll(np.asarray(quaternion, float), -1)).as_matrix()
        mount = np.asarray(self.c.lidar_mount, float)
        pose = np.asarray(pose, float)
        estimate = pose[2] - self.c.base_height
        for azimuth, distance in zip(-np.pi + np.flatnonzero(mask) * (2 * np.pi / 72),
                                     normalized[mask] * self.c.max_range):
            endpoint = self._endpoint(azimuth, 0., distance, pose, rotation, mount)
            cell = self.cell(endpoint[:2])
            if not self.inside(cell):
                continue
            relative = endpoint[2] - estimate
            self.seen[cell] = True
            if relative < self.c.min_obstacle_height:
                self.free[cell] = True
                self.ground[cell] = estimate
            elif relative < self.c.body_clearance:
                self.occupied[cell] += 1
            else:
                self.overhead[cell] += 1
