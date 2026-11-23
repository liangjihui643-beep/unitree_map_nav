"""Track local rubble surfaces and publish their current map contributions."""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import rot2, transform_xy
from .rubble_revision import RevisableRubbleMap, RubbleCell, nearby_depth, project


@dataclass
class Contribution(RubbleCell):
    score: float = 0.
    measured: bool = False
    stereo_views: list = field(default_factory=list)

    @property
    def supported(self):
        return self.score >= 1.5 and not self.ground_views


class RegionRubbleMap(RevisableRubbleMap):
    def __init__(self, k):
        super().__init__(k, footprint=True)
        self.regions = {}
        self.next_region = 0
        self.version = 0
        self.removed = []
        self.stamp = None
        self.replaced = 0
        self._planning = (set(), set())

    def planning_parts(self):
        return self._planning

    @staticmethod
    def _components(keys):
        remaining = set(keys)
        while remaining:
            component = [remaining.pop()]
            for x,y in component:
                for key in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if key in remaining:
                        remaining.remove(key)
                        component.append(key)
            yield component

    @classmethod
    def _keep_unbridged(cls, footprint, additions):
        owner = {key: index for index, group in enumerate(cls._components(footprint)) for key in group}
        retained = set()
        for group in cls._components(additions):
            touched = {owner[key] for x,y in group
                       for key in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)) if key in owner}
            if len(touched) == 1:
                retained.update(group)
        return retained

    def _refresh(self):
        previous = self.supported_keys()
        merged = {}
        for contributions in self.regions.values():
            for key, cell in contributions.items():
                prior = merged.get(key)
                if prior is None or (cell.supported, cell.score) > (prior.supported, prior.score):
                    merged[key] = cell
        self.cells = merged
        core, edges = super().planning_parts()
        stereo_patches = set()
        for contributions in self.regions.values():
            repeated = {key for key, cell in contributions.items()
                        if cell.supported and len(cell.stereo_views) >= 2}
            for part in self._components(repeated):
                if len(part) < 3:
                    continue
                xy = np.asarray(part)
                if np.linalg.matrix_rank(xy-xy[0]) == 2:
                    stereo_patches.update(part)
        footprint = core | edges
        owner = {key:index for index, group in enumerate(self._components(footprint)) for key in group}
        # A compact measured patch can initialise its own region. A patch
        # spanning two existing components waits for more geometry.
        for group in self._components(stereo_patches-footprint):
            touched = {owner[key] for x,y in group
                       for key in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)) if key in owner}
            if len(touched) <= 1:
                edges.update(group)
        # Keep new metric additions close to current measured evidence.
        # The reference set is frozen before admitting any additions.
        anchors = {key for key in core if merged[key].sources & {1,4}}
        if anchors:
            stable = cKDTree(np.asarray(sorted(anchors)))
            extra = [key for key, cell in merged.items()
                     if 7 in cell.sources and cell.supported and key not in core and key not in edges]
            gaps = stable.query(np.asarray(extra).reshape(-1,2),k=2)[0]
            admissible = {key for key, gap in zip(extra,gaps) if gap[0] <= 1.5 and gap[1] <= 2.5}
            edges.update(self._keep_unbridged(core|edges,admissible))
        # Every edge refers to the current measured component, never to
        # another interpolated edge or a previously published footprint.
        self._planning = core, edges
        current = core | edges
        self.removed = [tuple(map(int,key)) for key in sorted(previous-current)]
        if current != previous:
            self.version += 1

    def correct(self, corrections):
        if not corrections:
            return
        for region, cells in self.regions.items():
            moved = {}
            for cell in cells.values():
                old, new = min(corrections, key=lambda pair: np.linalg.norm(pair[0].xy()-cell.view))
                rotation = rot2(new.yaw-old.yaw)
                def transform(xy):
                    return new.xy() + rotation @ (xy-old.xy())
                cell.xyz[:2] = transform(cell.xyz[:2])
                cell.view = transform(cell.view)
                cell.positive_views = [transform(view) for view in cell.positive_views]
                cell.ground_views = [transform(view) for view in cell.ground_views]
                cell.stereo_views = [transform(view) for view in cell.stereo_views]
                key = self.key(cell.xyz[:2])
                if key not in moved or moved[key].score < cell.score:
                    moved[key] = cell
            self.regions[region] = moved
        self._refresh()

    @staticmethod
    def _patches(points, components):
        for label in np.unique(components):
            indices = np.flatnonzero(components == label)
            local = points[indices] * [1., 1., 1.7]
            graph = cKDTree(local).sparse_distance_matrix(cKDTree(local), .19, output_type="coo_matrix")
            _, parts = connected_components(graph, directed=False)
            for part in np.unique(parts):
                yield indices[parts == part]

    def observe_regions(self, points, sources, components, pose, base_z, origin, rotation,
                        floor, depth, observed_floor, stamp):
        self.stamp = stamp
        if not np.isfinite(base_z):
            return
        view = pose.xy()
        base = np.r_[view, base_z]
        world = np.column_stack((transform_xy(points, pose), points[:, 2]+base_z))
        current_keys = {self.key(point[:2]) for point in world}
        # Project the old contributions once. Negative evidence is local to
        # the range of the hypothetical rock surface, not the planar scan.
        visible = {}
        for region, contributions in self.regions.items():
            keys = list(contributions)
            xyz = np.asarray([contributions[key].xyz for key in keys])
            if len(xyz) == 0:
                continue
            uv, optical_z = project(xyz, base, pose, origin, rotation, self.k)
            u, v = np.rint(uv).astype(int).T
            height, width = floor.shape
            inside = (.35 < optical_z) & (optical_z < 4.)
            inside &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
            visible[region] = (keys, xyz, uv, optical_z, inside)
            checks = np.flatnonzero(inside)
            checks = checks[floor[v[checks], u[checks]]]
            metric = nearby_depth(*observed_floor, uv[checks], 7., .18)
            clear = np.where(np.isfinite(metric), metric >= optical_z[checks]-.06,
                             depth[v[checks], u[checks]]*.88 >= optical_z[checks])
            for index in checks[clear]:
                key = keys[index]
                cell = contributions[key]
                if key in current_keys or any(np.linalg.norm(old-view)<.075 for old in cell.ground_views):
                    continue
                cell.ground_views.append(view.copy())
                self.rejections += 1
                cell.score = max(-4., cell.score-2.)
                if cell.score <= -2.:
                    del contributions[key]
                    self.withdrawn += 1

        patch_uv, patch_depth = project(world, base, pose, origin, rotation, self.k)
        claimed = set()
        for indices in self._patches(world, components):
            positions = world[indices]
            uv_tree = cKDTree(patch_uv[indices])
            spatial_tree = cKDTree(positions[:, :2])
            best = None
            for region, (keys, xyz, pixels, optical_z, inside) in visible.items():
                if region in claimed:
                    continue
                visible_ids = np.flatnonzero(inside)
                if not len(visible_ids):
                    continue
                near = spatial_tree.query(xyz[visible_ids, :2])[0] <= .30
                visible_ids = visible_ids[near]
                gap, nearest = uv_tree.query(pixels[visible_ids])
                residual = patch_depth[indices[nearest]]-optical_z[visible_ids]
                matches = (gap <= 5.) & (abs(residual) <= .28)
                count = int(matches.sum())
                if count and (best is None or count > best[0]):
                    best = (count, region, visible_ids[matches], indices[nearest[matches]], residual[matches])
            if best is None:
                region = self.next_region
                self.next_region += 1
                self.regions[region] = {}
            else:
                _, region, old_ids, new_ids, residual = best
                old_keys = visible[region][0]
                for old_id, new_id, delta in zip(old_ids, new_ids, residual):
                    key = old_keys[old_id]
                    old = self.regions[region].get(key)
                    if (old is not None and not old.measured and key not in current_keys
                            and sources[new_id] in (1,4,7) and abs(delta) >= .10):
                        del self.regions[region][key]
                        self.replaced += 1
            claimed.add(region)
            groups = defaultdict(list)
            for index in indices:
                groups[self.key(world[index, :2])].append(index)
            for key, ids in groups.items():
                self._update(self.regions[region], key, world[ids], sources[ids], view)
        self.regions = {region: cells for region, cells in self.regions.items() if cells}
        self._refresh()

    def _update(self, contributions, key, xyz, sources, view):
        measured = np.isin(sources, (1,4,7))
        position = np.median(xyz[measured] if measured.any() else xyz, axis=0)
        cell = contributions.get(key)
        if cell is None:
            cell = contributions[key] = Contribution(position, view.copy())
        if (len(cell.stereo_views)<2 and (sources == 4).any()
                and all(np.linalg.norm(old-view)>=.15 for old in cell.stereo_views)):
            cell.stereo_views.append(view.copy())
        independent = all(np.linalg.norm(old-view)>=.10 for old in cell.positive_views)
        if independent:
            cell.positive_views = (cell.positive_views+[view.copy()])[-8:]
            increment = 2. if np.isin(sources,(1,5,7)).any() else .85
            cell.score = min(4., cell.score+increment)
            if cell.ground_views and cell.score >= 2. and measured.any():
                cell.ground_views.pop(0)
        if measured.any() or not cell.measured:
            weight = .60 if measured.any() and not cell.measured else .35
            cell.xyz += weight*(position-cell.xyz)
            cell.view = view.copy()
        cell.measured |= bool(measured.any())
        cell.sources.update(sources.tolist())
        cell.frames += 1

    def snapshot(self, grid):
        result = super().snapshot(grid)
        result.update(revision=self.version, observation_stamp=self.stamp,
                      tracked_regions=len(self.regions), corrected_contributions=self.replaced,
                      removed_cells=[list(key) for key in self.removed])
        return result
