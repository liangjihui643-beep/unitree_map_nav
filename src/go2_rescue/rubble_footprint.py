"""Build a revisable 2-D rubble footprint from semantic and metric support."""
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot2, transform_xy
from .rubble_depth_runtime import RGBRubbleDepth, SIZE
from .rubble_revision import RevisableRubbleMap, RubbleGeometry
from .rubble_single_view import RubbleRangeAnchors
from .rubble_multiview import select_sources, sweep_multiview
from .rubble_region_tracks import RegionRubbleMap
from .rubble_visual_height import View


class RubbleFootprint:
    def __init__(self, config, intrinsics, model, *, region_mapping=False):
        self.depth = RGBRubbleDepth(model, intrinsics, config.camera_size, threshold=.75)
        self.ranges = RubbleRangeAnchors(config, intrinsics)
        self.small_k = intrinsics.copy()
        self.small_k[:2] *= .5
        self.small_k[:2,2] -= .25
        self.geometry = RubbleGeometry(intrinsics,self.small_k)
        self.revision = (RegionRubbleMap(self.small_k) if region_mapping
                         else RevisableRubbleMap(self.small_k, footprint=True))
        self.keyframes = deque(maxlen=20)

    def correct(self, corrections):
        self.geometry.clear()
        self.keyframes.clear()
        self.revision.correct(corrections)

    def planning_centers(self):
        cells = np.asarray(sorted(self.revision.supported_keys())).reshape(-1,2)
        return (cells+.5)*self.revision.resolution

    def observe(self, image, mask, floor_mask, obs, level, rotation, origin, plane, pose, base_z, stamp):
        start=time.perf_counter()
        small=cv2.resize(mask.astype(np.uint8),SIZE,interpolation=cv2.INTER_NEAREST_EXACT)
        floor=cv2.resize(floor_mask.astype(np.uint8),SIZE,interpolation=cv2.INTER_NEAREST_EXACT)
        floor=(cv2.erode(floor,np.ones((5,5),np.uint8))!=0) & (small==0)
        stones,observed_floor,floor_cloud=self.geometry.observe(image,small!=0,floor,pose,origin,rotation,
                                                                 plane,base_z,stamp)
        if small.any() or self.revision.cells:
            depth,_,_=self.depth.infer(image,rotation[2],origin[2]-plane[2])
        else:
            depth=np.full(small.shape,np.nan)
        _,labels=cv2.connectedComponents(mask.astype(np.uint8))
        anchors=self.ranges.associate(obs,level,rotation,origin,labels)
        y,x=np.nonzero(small[1::3,1::3])
        y,x=y*3+1,x*3+1
        pixels=np.column_stack((x,y))
        screen=(pixels+.5)*np.asarray(mask.shape[::-1])/SIZE-.5
        components=labels[np.rint(screen[:,1]).astype(int),np.rint(screen[:,0]).astype(int)]
        measured=self.ranges.depth_at(screen,components,anchors)
        stereo,_=self.geometry.surface_depths(pixels,small.shape,stones)
        rays=self.depth.rays[y,x] @ rotation.T
        distance,sources=self.expand(pixels,components,depth[y,x],rays,measured,stereo)
        self.complete_surface(labels,pixels,components,depth[y,x],rays,measured,stereo,distance,sources)
        if isinstance(self.revision, RegionRubbleMap):
            missing = ~np.isfinite(distance)
            dense = self._fill_depth(image, mask, pixels[missing], pose, base_z, origin, rotation, stamp)
            resolved = np.flatnonzero(missing)[np.isfinite(dense)]
            distance[resolved] = dense[np.isfinite(dense)]
            sources[resolved] = 7
        good=np.isfinite(distance)
        stone=origin+rays[good]*distance[good,None]
        ground=self.geometry.ground_z(transform_xy(stone,pose),floor_cloud)
        obstructs=~np.isfinite(ground) | (stone[:,2]+base_z>ground+.020)
        good[np.flatnonzero(good)[~obstructs]]=False
        stone=stone[obstructs]
        sources=sources[good]
        # A row of echoes along a wall/floor seam may overlap a falsely
        # segmented strip. Let a second view resolve line-shaped geometry.
        for region in np.unique(components[good]):
            part=components[good]==region
            core=stone[part & (sources==1),:2]
            if self.is_line(core):
                sources[part & np.isin(sources,[1,5])]=6
        if isinstance(self.revision, RegionRubbleMap):
            self.revision.observe_regions(stone, sources, components[good], pose, base_z, origin,
                                          rotation, floor, depth, observed_floor, stamp)
        else:
            self.revision.observe(stone,None,sources,pose,base_z,origin,rotation,floor,depth,observed_floor)
        return {
            "t":stamp, "completed_uv":pixels[good].tolist(), "completed_xy_m":transform_xy(stone,pose).tolist(),
            "completion_source":sources.tolist(), "active_count":len(self.revision.cells),
            "planning_count":len(self.revision.supported_keys()),
            "withdrawn_cells":self.revision.withdrawn, "ground_contradictions":self.revision.rejections,
            **({"revision":self.revision.version, "removed_cells":self.revision.removed,
                "corrected_contributions":self.revision.replaced}
               if isinstance(self.revision, RegionRubbleMap) else {}),
            "processing_ms":1000.*(time.perf_counter()-start),
        }

    def _fill_depth(self, image, mask, missing, pose, base_z, origin, rotation, stamp):
        result = np.full(len(missing), np.nan)
        if not np.isfinite(base_z) or np.count_nonzero(mask)<80:
            return result
        yaw = np.eye(3)
        yaw[:2,:2] = rot2(pose.yaw)
        base = np.r_[pose.xy(), base_z]
        center = base+yaw@origin
        if self.keyframes and (stamp-self.keyframes[-1].stamp<.45
                               or np.linalg.norm(center-self.keyframes[-1].center)<.075):
            return result
        optical = yaw@rotation
        camera = self.geometry.stereo.k @ np.column_stack((optical.T, -optical.T@center))
        labels = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8))
        view = View(stamp, cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), None, camera, center,
                    base, yaw, np.zeros(3), labels)
        sources = select_sources(view, self.keyframes)
        self.keyframes.append(view)
        if len(missing)<6 or len(sources)<2:
            return result
        # RGB range is a search hint. Visual correspondence still chooses the
        # depth, and samples at the search endpoints are rejected.
        planes = np.linspace(1/3.6, 1/.45, 32)
        points, _ = sweep_multiview(self.geometry.stereo.k, sources[:2], view,
                                    inverse_depths=planes, target_pixels=missing)
        if not len(points):
            return result
        coords = (points-center)@optical
        depth = coords[:,2]
        uvw = coords@self.small_k.T
        uv = uvw[:,:2]/depth[:,None]
        gap, index = cKDTree(uv).query(missing)
        good = (gap<=2.) & (depth[index]>.35) & (depth[index]<4.)
        result[good] = depth[index[good]]
        return result

    @staticmethod
    def is_line(core):
        if len(core)<4:
            return False
        _,vectors=np.linalg.eigh(np.cov(core.T))
        spread=np.ptp(core @ vectors,axis=0)
        return spread[1]>.38 and spread[0]<.14 and spread[0]<.27*spread[1]

    @staticmethod
    def complete_surface(labels, pixels, components, predicted, rays, measured, stereo, distance, sources):
        """Extend a visible stone face only where separated metric anchors agree."""
        anchors = np.where(np.isfinite(measured), measured, stereo)
        for component in np.unique(components[~np.isfinite(distance)]):
            if component == 0:
                continue
            same_face = components == component
            core = np.flatnonzero(same_face & np.isfinite(anchors) & np.isfinite(predicted))
            query = np.flatnonzero(same_face & ~np.isfinite(distance) & np.isfinite(predicted))
            if len(core) < 2 or not len(query):
                continue
            gaps, nearest = cKDTree(pixels[core]).query(
                pixels[query], k=np.arange(1,min(40,len(core))+1), distance_upper_bound=42.,
            )
            seeds = core[np.minimum(nearest,len(core)-1)]
            shape = predicted[query,None] - predicted[seeds]
            estimate = anchors[seeds] + np.clip(.8*shape,-.38,.38)
            offset = np.linalg.norm(
                rays[query,None,:2]*estimate[...,None] - rays[seeds,:2]*anchors[seeds,None], axis=2,
            )
            eligible = (np.isfinite(gaps) & (abs(shape)<=.56) & (offset<=.53)
                        & (estimate>.35) & (estimate<4.))
            step = pixels[seeds]-pixels[query,None]
            # A straight image path cannot cross background and reconnect to
            # a different side of the same mask through a distant neck.
            for fraction in (.25,.5,.75):
                path = np.rint(2.*(pixels[query,None] + fraction*step)+.5).astype(int)
                eligible &= labels[path[:,:,1],path[:,:,0]] == component
            primary = eligible & (offset<=.40)
            first = primary.argmax(axis=1)
            row = np.arange(len(query))
            separated = np.linalg.norm(pixels[seeds]-pixels[seeds[row,first],None],axis=2)>=13.
            consistent = abs(estimate-estimate[row,first,None])<=.14
            corroborating = eligible & separated & consistent
            accepted = primary.any(axis=1) & corroborating.any(axis=1)
            row = row[accepted]
            first_seed = first[accepted]
            second_seed = corroborating[accepted].argmax(axis=1)
            indices = query[accepted]
            first_gap,second_gap = gaps[row,first_seed],gaps[row,second_seed]
            first_weight = (2.+second_gap)/(4.+first_gap+second_gap)
            distance[indices] = first_weight*estimate[row,first_seed] + (1.-first_weight)*estimate[row,second_seed]
            has_echo = np.isfinite(measured[seeds[row,first_seed]]) | np.isfinite(measured[seeds[row,second_seed]])
            sources[indices] = np.where(has_echo,5,2)

    @staticmethod
    def expand(pixels, components, predicted, rays, measured, stereo):
        distance=np.where(np.isfinite(measured),measured,stereo)
        sources=np.where(np.isfinite(measured),1,4)
        for component in np.unique(components):
            if component==0:
                continue
            part=np.flatnonzero(components==component)
            core=part[np.isfinite(distance[part])]
            if not len(core):
                continue
            remaining=part[~np.isfinite(distance[part])]
            gaps,nearest=cKDTree(pixels[core]).query(pixels[remaining],distance_upper_bound=25.)
            close=np.isfinite(gaps)
            index=remaining[close]
            seed=core[nearest[close]]
            gap=gaps[close]
            echo=sources[seed]==1
            shape=predicted[index]-predicted[seed]
            resolved=distance[seed]+np.clip(.8*shape,-.23,.26)
            offset=np.linalg.norm(rays[index,:2]*resolved[:,None]-rays[seed,:2]*distance[seed,None],axis=1)
            connected=(abs(shape)<.42) & (offset<=np.where(echo,.31,.23)) & (gap<=np.where(echo,24.,14.))
            connected &= (resolved>.35)&(resolved<4.)
            keep=index[connected]
            distance[keep]=resolved[connected]
            # A strongly tied expansion can be used immediately. Wider rays
            # and geometry-only regions need a second independent view.
            immediate=echo & (gap<=12.) & (offset<=.19)
            sources[keep]=np.where(immediate[connected],5,2)
        distance[components==0]=np.nan
        return distance,sources
