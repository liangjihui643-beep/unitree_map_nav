"""Complete observed rubble surfaces from metric seeds and RGB boundaries."""
from collections import deque

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .geometry import rot2, transform_xy
from .rubble_depth_runtime import RGBRubbleDepth, SIZE


def propagate(mask, rgb, prior, seeds):
    """Use short, image-consistent paths from the closest measured seed."""
    ids = np.arange(mask.size).reshape(mask.shape)
    origins, targets, costs = [], [], []
    for first, second in ((np.s_[:,:-1], np.s_[:,1:]), (np.s_[:-1,:], np.s_[1:,:])):
        inside = mask[first] & mask[second]
        color_jump = np.linalg.norm(rgb[first]-rgb[second], axis=2)
        depth_jump = abs(prior[first]-prior[second])
        join = inside & (color_jump <= 45.) & (depth_jump <= .20)
        origins.append(ids[first][join])
        targets.append(ids[second][join])
        costs.append(1.+color_jump[join]/21.+depth_jump[join]/.06)
    graph = coo_matrix((np.concatenate(costs), (np.concatenate(origins), np.concatenate(targets))),
                       shape=(ids.size, ids.size)).tocsr()
    seed_ids = np.flatnonzero(np.isfinite(seeds))
    distances, _, source = dijkstra(graph, directed=False, indices=seed_ids, limit=23.,
                                    min_only=True, return_predecessors=True)
    valid = np.flatnonzero(np.isfinite(distances))
    nearby = source[valid]
    shape = prior.ravel()[valid]-prior.ravel()[nearby]
    limit = np.minimum(.18, distances[valid]*.015)
    completed = np.full(ids.size, np.nan)
    completed[valid] = seeds.ravel()[nearby] + np.clip(.6*shape,-limit,limit)
    return completed.reshape(mask.shape), source.reshape(mask.shape), distances.reshape(mask.shape)


class RubbleCompletion:
    def __init__(self, intrinsics, size, model, *, revise=False):
        self.model = RGBRubbleDepth(model, intrinsics, size, threshold=.75) if model else None
        self.k = intrinsics.copy()
        self.k[:2] *= .5
        self.k[:2,2] -= .25
        self.history = deque(maxlen=14)
        self.geometry = None
        self.revision = None
        if revise:
            from .rubble_revision import RevisableRubbleMap, RubbleGeometry

            self.geometry = RubbleGeometry(intrinsics,self.k)
            self.revision = RevisableRubbleMap(self.k)

    def clear(self):
        self.history.clear()
        if self.geometry is not None:
            self.geometry.clear()

    @staticmethod
    def appearance(color,u,v):
        height,width = color.shape[:2]
        patches = [color[np.clip(v+dv,0,height-1),np.clip(u+du,0,width-1)]
                   for du,dv in ((0,0),(-2,0),(2,0),(0,-2),(0,2))]
        return np.stack(patches,axis=1)

    def recover(self, mask, color, pose, rotation, origin, base_z, stamp, current):
        recovered = np.full(mask.shape,np.nan)
        if not np.isfinite(base_z):
            return recovered
        base = np.r_[pose.xy(),base_z]
        yaw = pose.yaw
        for time, previous, world, appearance in reversed(self.history):
            baseline = np.linalg.norm(base-previous)
            if not (0 < stamp-time <= 4. and .035 < baseline <= .70):
                continue
            local = world-base
            local[:,:2] = local[:,:2] @ rot2(yaw)
            optical = (local-origin) @ rotation
            depth = optical[:,2]
            valid = (depth>.35)&(depth<4.)
            uv = np.rint(optical[valid] @ self.k.T / depth[valid,None]).astype(int)[:,:2]
            u,v = uv.T
            inside = (u>=0)&(v>=0)&(u<mask.shape[1])&(v<mask.shape[0])
            u,v = u[inside],v[inside]
            z = depth[valid][inside]
            patch = self.appearance(color,u,v)
            old_patch = appearance[valid][inside]
            same = np.sqrt(np.mean((patch-old_patch)**2,axis=(1,2))) < 10.
            texture = np.std(old_patch,axis=1).mean(1) > 2.5
            good = same & texture & mask[v,u] & ~np.isfinite(recovered[v,u]) & ~np.isfinite(current[v,u])
            recovered[v[good],u[good]] = z[good]
        return recovered

    def remember(self, points, u, v, color, pose, base_z, stamp):
        if not len(points) or not np.isfinite(base_z):
            return
        stride = max(1, int(np.ceil(len(points)/100)))
        xyz = points[::stride].copy()
        xyz[:,:2] = transform_xy(xyz,pose)
        xyz[:,2] += base_z
        self.history.append((stamp,np.r_[pose.xy(),base_z],xyz,self.appearance(color,u[::stride],v[::stride])))

    @staticmethod
    def heights(points, ground):
        """Fit nearby visible ground before extrapolating to the stone sample."""
        relief = np.full(len(points),np.nan)
        if len(ground)<8:
            return relief
        for i, point in enumerate(points):
            delta = ground[:,:2]-point[:2]
            distances = np.linalg.norm(delta,axis=1)
            nearest = np.argsort(distances)[:24]
            nearest = nearest[distances[nearest]<.40]
            if len(nearest)<8 or distances[nearest[0]]>.19:
                continue
            xy = delta[nearest]
            if np.linalg.eigvalsh(np.cov(xy.T))[0]<.0012:
                continue
            matrix = np.column_stack((xy,np.ones(len(xy))))
            z = ground[nearest,2]
            coeff = np.linalg.lstsq(matrix,z,rcond=None)[0]
            error = abs(z-matrix@coeff)
            height = point[2]-coeff[2]
            if np.quantile(error,.8)<.035 and np.linalg.norm(coeff[:2])<.5 and 0 <= height <= .50:
                relief[i]=height
        return relief

    def complete(self, mask, image, prior, rays, rotation, origin, plane, pose,
                 stamp, base_z, floor_mask, pixels, measured):
        color = cv2.GaussianBlur(cv2.resize(image,SIZE,interpolation=cv2.INTER_AREA),(3,3),0).astype(np.float32)
        depth, floor_depth, probability = prior, prior, None
        if self.model is not None:
            depth, _, probability = self.model.infer(image,rotation[2],origin[2]-plane[2])
            floor_depth = depth
        u,v = pixels.T
        floor = None
        if floor_mask is not None:
            small = cv2.resize(floor_mask.astype(np.uint8),SIZE,interpolation=cv2.INTER_NEAREST_EXACT)
            floor = (cv2.erode(small,np.ones((5,5),np.uint8)) != 0) & ~mask
        direct = np.full(mask.shape,np.nan)
        direct[v,u] = measured
        history = self.recover(mask,color,pose,rotation,origin,base_z,stamp,direct)
        stereo = np.full(mask.shape,np.nan)
        floor_cloud = np.empty((0,3))
        if self.geometry is not None:
            surface, observed_floor, floor_cloud = self.geometry.observe(
                image,mask,floor,pose,origin,rotation,plane,base_z,stamp)
            nearby, stereo = self.geometry.surface_depths(pixels,mask.shape,surface)
            stereo[~mask] = np.nan
            stereo[v[np.isfinite(nearby)],u[np.isfinite(nearby)]] = nearby[np.isfinite(nearby)]
            disagree = np.isfinite(stereo) & (abs(history-stereo) > .14)
            history[disagree] = np.nan
            for seeds in (history, direct, stereo):
                y,x = np.nonzero(np.isfinite(seeds))
                points = origin+rays[y,x]*seeds[y,x,None]
                xy = transform_xy(points,pose)
                ground = self.geometry.ground_z(xy,floor_cloud)
                relative = points[:,2]+base_z-ground
                invalid = (relative < -.05) | (relative > .52)
                seeds[y[invalid],x[invalid]] = np.nan
            # Let consistent stereo features replace a conflicting range association
            # at sample pixels, while keeping range support in weakly textured areas.
            supported = np.zeros(mask.shape,bool)
            supported[v,u] = np.isfinite(nearby)
            replace = supported & np.isfinite(stereo) & (abs(stereo-direct)>.12)
            direct[replace] = np.nan
        seeds = np.where(np.isfinite(direct),direct,history)
        use_stereo = ~np.isfinite(direct) & np.isfinite(stereo)
        seeds[use_stereo] = stereo[use_stereo]
        measured = direct[v,u]
        accepted = np.isfinite(measured)
        direct_xyz = origin+rays[v[accepted],u[accepted]]*measured[accepted,None]
        self.remember(direct_xyz,u[accepted],v[accepted],color,pose,base_z,stamp)
        output, provenance, distance = (propagate(mask,color,depth,seeds) if np.isfinite(seeds).any()
                                        else (direct, np.zeros(mask.shape,int), np.full(mask.shape,np.inf)))
        if self.geometry is not None:
            reached = np.isfinite(output)
            weak = ~np.isfinite(direct.ravel()[provenance[reached]])
            output[reached] = np.where(weak & (distance[reached] > 12.),np.nan,output[reached])
        d = output[v,u]
        keep = np.isfinite(d)
        uu,vv = u[keep],v[keep]
        stone = origin+rays[vv,uu]*d[keep,None]
        source = np.where(np.isfinite(direct[vv,uu]),1,
                          np.where(np.isfinite(direct.ravel()[provenance[vv,uu]]),2,3))
        source[use_stereo.ravel()[provenance[vv,uu]]] = 4
        if np.any(keep) and floor_mask is not None:
            relief_floor = cv2.resize(floor_mask.astype(np.uint8),SIZE,interpolation=cv2.INTER_NEAREST_EXACT)
            border = cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8)) != 0
            ground_mask = (relief_floor!=0) & border & ~mask & (floor_depth>.35) & (floor_depth<4.)
            ground_mask &= cv2.erode(relief_floor,np.ones((3,3),np.uint8)) != 0
            # Align the local shape's depth scale using only same-frame seeds.
            if accepted.sum() >= 5:
                scale = measured[accepted]/depth[v[accepted],u[accepted]]
                median = np.median(scale)
                consistent = np.quantile(abs(scale-median),.8) <= .09
            else:
                median,consistent = 1.,False
            gy,gx = np.nonzero(ground_mask[1::3,1::3])
            gy,gx = gy*3+1,gx*3+1
            ground = origin+rays[gy,gx]*floor_depth[gy,gx,None]*median
            relief = self.heights(stone,ground) if consistent else np.full(len(stone),np.nan)
        else:
            relief = np.full(len(stone),np.nan)
        if self.geometry is not None:
            xy = transform_xy(stone,pose)
            ground = self.geometry.ground_z(xy,floor_cloud)
            relative = stone[:,2]+base_z-ground
            relief[(relative >= 0.) & (relative <= .50)] = relative[(relative >= 0.) & (relative <= .50)]
            okay = ~np.isfinite(relative) | ((relative >= -.05) & (relative <= .52))
            kept_indices = np.flatnonzero(keep)
            keep[kept_indices[~okay]] = False
            stone, relief, source = stone[okay], relief[okay], source[okay]
            vv,uu = vv[okay],uu[okay]
            self.revision.observe(stone,relief,source,pose,base_z,origin,rotation,floor,floor_depth,observed_floor)
        component_mask = cv2.connectedComponents(mask.astype(np.uint8))[1]
        anchored_labels = np.unique(component_mask[np.isfinite(seeds)])
        unscaled = mask[v,u] & ~np.isin(component_mask[v,u],anchored_labels)
        mono = (unscaled & (probability[v,u] >= .75) & (depth[v,u] >= .35) & (depth[v,u] <= 4.)
                if probability is not None else np.zeros(len(u),bool))
        mono_xyz = origin+rays[v[mono],u[mono]]*depth[v[mono],u[mono],None]
        comparisons = dict(
            completed_uv=pixels[keep].tolist(), completed_xy_m=transform_xy(stone,pose).tolist(),
            completion_source=source.tolist(), propagation_cost=distance[vv,uu].tolist(),
            estimated_relief_m=[float(h) if np.isfinite(h) else None for h in relief],
            unanchored_uv=pixels[mono].tolist(), unanchored_xy_m=transform_xy(mono_xyz,pose).tolist(),
        )
        if self.revision is not None:
            comparisons["withdrawn_cells"] = self.revision.withdrawn
            comparisons["ground_contradictions"] = self.revision.rejections
        return stone[:,:2], relief, source, comparisons
