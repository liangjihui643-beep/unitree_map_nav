"""Independently observed geometry and revisable rubble map evidence."""
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot2, transform_xy
from .rubble_visual_height import RubbleVisualHeight


def project(points, base, pose, origin, rotation, k):
    local = points - base
    local[:, :2] = local[:, :2] @ rot2(pose.yaw)
    optical = (local - origin) @ rotation
    depth = optical[:, 2]
    uv = np.divide((optical @ k.T)[:, :2], depth[:, None],
                   out=np.full((len(points), 2), -1.), where=depth[:, None] > .1)
    return uv, depth


def nearby_depth(uv, depth, target, radius, spread):
    result = np.full(len(target), np.nan)
    if len(uv) < 2 or not len(target):
        return result
    distance, index = cKDTree(uv).query(target, k=2, distance_upper_bound=radius)
    indices = np.flatnonzero(np.isfinite(distance[:, 1]))
    pair = depth[index[indices]]
    good = np.ptp(pair, axis=1) <= spread
    result[indices[good]] = pair[good].mean(axis=1)
    return result


class _SurfaceGeometry(RubbleVisualHeight):
    def _resolve(self, surface, support, stamp):
        # Completion checks reconstructed floor separately against each seed.
        # Avoid fitting the same surrounding floor for each individual pair.
        if len(support):
            self.ground_history.append((stamp,support))
        return np.column_stack((surface,np.full((len(surface),2),np.nan)))


class RubbleGeometry:
    def __init__(self, k, small_k):
        self.stereo = _SurfaceGeometry(k, wider_baseline=True, independent_ground=True)
        self.k = small_k

    def clear(self):
        self.stereo.clear()

    def observe(self, image, mask, floor, pose, origin, rotation, plane, base_z, stamp):
        empty = (np.empty((0, 2)), np.empty(0))
        if floor is None or not np.isfinite(base_z):
            return empty, empty, np.empty((0, 3))
        size = (image.shape[1], image.shape[0])
        rock = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        ground = cv2.resize(floor.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        surface = self.stereo.observe(image, rock, rotation, origin, pose, base_z, plane, stamp,
                                      ground_mask=ground & ~rock, base_z=base_z)[:, :3]
        recent = [cloud for time, cloud in self.stereo.ground_history if 0 <= stamp-time <= 2.]
        floor_cloud = np.vstack(recent) if recent else np.empty((0, 3))
        base = np.r_[pose.xy(), base_z]
        def projected(points):
            uv, depth = project(points, base, pose, origin, rotation, self.k)
            valid = (depth > .35) & (depth < 4.)
            return uv[valid], depth[valid]
        return projected(surface), projected(floor_cloud), floor_cloud

    @staticmethod
    def surface_depths(pixels, shape, observed):
        uv, depth = observed
        rounded = np.rint(uv).astype(int)
        _, index = np.unique(rounded,axis=0,return_index=True)
        uv,depth = uv[index],depth[index]
        # A pair of consistent features can support nearby samples. Preserve
        # individual features as seeds at their own pixels.
        result = nearby_depth(uv, depth, pixels, 4., .15)
        seeds = np.full(shape, np.nan)
        rounded = np.rint(uv).astype(int)
        u, v = rounded.T
        inside = (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])
        seeds[v[inside], u[inside]] = depth[inside]
        return result, seeds

    @staticmethod
    def ground_z(points, floor_cloud):
        ground = np.full(len(points), np.nan)
        if len(floor_cloud)<3 or not len(points):
            return ground
        distance, indices = cKDTree(floor_cloud[:,:2]).query(points[:,:2],k=3)
        samples = floor_cloud[indices, 2]
        valid = (distance[:, -1] <= .19) & (np.ptp(samples, axis=1) <= .075)
        ground[valid] = np.median(samples[valid], axis=1)
        return ground


@dataclass
class RubbleCell:
    xyz: np.ndarray
    view: np.ndarray
    frames: int = 0
    sources: set = field(default_factory=set)
    heights: list = field(default_factory=list)
    positive_views: list = field(default_factory=list)
    ground_views: list = field(default_factory=list)

    @property
    def supported(self):
        return not self.ground_views and (bool({1,5}&self.sources) or len(self.positive_views) >= 2)


class RevisableRubbleMap:
    """Age out contradicted cells only when the camera sees clear ground."""
    def __init__(self, k, resolution=.10, *, footprint=False):
        self.k = k
        self.resolution = resolution
        self.cells = {}
        self.withdrawn = 0
        self.rejections = 0
        self.footprint = footprint

    def key(self, xy):
        return tuple(np.floor(xy / self.resolution).astype(int))

    def _core_components(self):
        supported={key for key,cell in self.cells.items() if cell.supported}
        if not self.footprint:
            return {key:0 for key in supported}
        core={key for key in supported if 1 in self.cells[key].sources}
        if not core:
            return {}
        expanded=np.asarray(sorted(supported-core)).reshape(-1,2)
        distances=cKDTree(np.asarray(sorted(core))).query(expanded)[0]
        remaining=core | {tuple(cell) for cell in expanded[distances<=3.15]}
        result={}
        group=0
        while remaining:
            component=[remaining.pop()]
            for x,y in component:
                for point in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if point in remaining:
                        remaining.remove(point)
                        component.append(point)
            # A tiny single-view patch is unlikely to be worth a detour.
            # Retain it in the live evidence map until enough is seen.
            if len(component)>=9 or sum(len(self.cells[key].positive_views)>=2 for key in component)>=2:
                result.update((key,group) for key in component)
                group+=1
        return result

    def planning_parts(self):
        """Keep evidence-backed edges within one grid cell of the original core."""
        components=self._core_components()
        core=set(components)
        edges=set()
        if not self.footprint:
            return core,edges
        axes=((0,1),(1,0),(0,-1),(-1,0))
        diagonal=((1,1),(1,-1),(-1,1),(-1,-1))
        for (x,y),cell in self.cells.items():
            key=(x,y)
            if key in core or cell.ground_views:
                continue
            touches=[(x+dx,y+dy) for dx,dy in axes if (x+dx,y+dy) in core]
            if not touches:
                continue
            local=[(x+dx,y+dy) for dx,dy in axes+diagonal if (x+dx,y+dy) in core]
            owner=components[touches[0]]
            if any(components[p]!=owner for p in local):
                continue
            enclosed=any(components.get((x+dx,y+dy))==owner and components.get((x-dx,y-dy))==owner
                         for dx,dy in ((1,0),(0,1),(1,1),(1,-1)))
            nearby=len(local)+sum(components.get((x+2*dx,y+2*dy))==owner for dx,dy in axes)
            verified=len(cell.positive_views)>=2
            if cell.sources=={6} and not verified:
                continue
            if enclosed or len(local)>=4 or (verified and nearby>=3):
                edges.add(key)
        return core,edges

    def supported_keys(self):
        core,edges=self.planning_parts()
        return core | edges

    def correct(self, corrections):
        if not corrections:
            return
        restored = {}
        for cell in self.cells.values():
            old, new = min(corrections,key=lambda pair: np.linalg.norm(pair[0].xy()-cell.view))
            rot = rot2(new.yaw-old.yaw)
            cell.xyz[:2] = new.xy() + rot @ (cell.xyz[:2]-old.xy())
            cell.view = new.xy() + rot @ (cell.view-old.xy())
            cell.positive_views = [new.xy() + rot @ (view-old.xy()) for view in cell.positive_views]
            cell.ground_views.clear()
            key = self.key(cell.xyz[:2])
            if key not in restored or restored[key].frames < cell.frames:
                restored[key] = cell
        self.cells = restored

    def observe(self, points, heights, sources, pose, base_z, origin, rotation, floor, floor_depth,
                observed_floor):
        xy = transform_xy(points,pose)
        xyz = np.column_stack((xy,points[:,2]+base_z))
        groups = defaultdict(list)
        for i, p in enumerate(xy):
            groups[self.key(p)].append(i)
        view = pose.xy()
        base = np.r_[view,base_z]
        keys = [key for key in self.cells if key not in groups]
        if keys and floor is not None and np.isfinite(base_z):
            cloud = np.array([self.cells[key].xyz for key in keys])
            uv, depth = project(cloud,base,pose,origin,rotation,self.k)
            rounded = np.rint(uv).astype(int)
            u,v = rounded.T
            h,w = floor.shape
            indices = np.flatnonzero((depth > .35)&(depth < 4.)&(u >= 0)&(v >= 0)&(u < w)&(v < h))
            indices = indices[np.asarray(floor[v[indices],u[indices]],dtype=bool)]
            metric = nearby_depth(*observed_floor, uv[indices], 7., .18)
            learned = floor_depth[v[indices],u[indices]]
            # A closer surface may hide the rock. Use visual ground depth
            # alone only when its conservative lower bound clears the point.
            clear = np.where(np.isfinite(metric),metric >= depth[indices]-.06,
                             learned*.88 >= depth[indices])
            for index, is_metric in zip(indices[clear],np.isfinite(metric[clear])):
                key = keys[index]
                cell = self.cells.get(key)
                if cell is None or any(np.linalg.norm(old-view)<.075 for old in cell.ground_views):
                    continue
                cell.ground_views.append(view.copy())
                self.rejections += 1
                required = 2 if is_metric and not ({1,4}&cell.sources) else 3
                if len(cell.ground_views) >= required:
                    del self.cells[key]
                    self.withdrawn += 1
        for key, ids in groups.items():
            sources_here = set(sources[ids].tolist())
            reliable = np.isin(sources[ids],[1,4])
            selected = np.asarray(ids)[reliable] if reliable.any() else ids
            cell = self.cells.get(key)
            latest = np.median(xyz[selected],axis=0)
            if cell is None:
                cell = self.cells[key] = RubbleCell(latest,view.copy())
            elif np.isfinite(latest[2]) and (reliable.any() or not ({1,4} & cell.sources)
                                            or not np.isfinite(cell.xyz[2])):
                cell.xyz,cell.view = latest,view.copy()
            cell.sources.update(sources_here)
            cell.frames += 1
            cell.ground_views.clear()
            if len(cell.positive_views)<2 and all(np.linalg.norm(old-view)>=.10 for old in cell.positive_views):
                cell.positive_views.append(view.copy())
            if heights is not None:
                heights_here = heights[ids]
                finite = heights_here[np.isfinite(heights_here)]
                if len(finite):
                    cell.heights = (cell.heights + [float(np.median(finite))])[-12:]

    def snapshot(self, grid):
        result = {}
        core,edges=self.planning_parts()
        for key,cell in self.cells.items():
            ij = grid.cell(cell.xyz[:2])
            if not grid.inside(ij):
                continue
            element = result.setdefault(ij,dict(ij=list(ij), xy_m=grid.world(ij).tolist(), frames=0,
                                                depth_sources=set(), heights=[],
                                                range_or_repeat_support=False))
            element["depth_sources"].update(cell.sources)
            element["frames"] = max(element["frames"],cell.frames)
            element["range_or_repeat_support"] |= key in core or key in edges
            if self.footprint and (key in core or "planning_role" not in element
                                   or (key in edges and element["planning_role"]=="provisional")):
                element["planning_role"]="core" if key in core else "fused_edge" if key in edges else "provisional"
            if not self.footprint:
                element["heights"].extend(cell.heights)
        for element in result.values():
            heights = element.pop("heights")
            if not self.footprint:
                element["height_observations"] = len(heights)
                element["height_above_local_ground_m"] = float(np.median(heights)) if len(heights)>=2 else None
            element["depth_sources"] = sorted(element["depth_sources"])
        return dict(status="revisable_rubble_footprint" if self.footprint else "revisable_rubble_surface",
                    cells=list(result.values()),
                    withdrawn_cells=self.withdrawn, ground_contradictions=self.rejections)
