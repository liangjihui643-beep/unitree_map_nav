"""Optional rubble precision and height-candidate evidence on the existing map."""
from collections import defaultdict

import numpy as np

from .geometry import rot2, transform_xy
from .rubble_consensus import RubbleConsensus
from .rubble_depth_runtime import RGBRubbleDepth, SIZE


class RubblePrecisionLayer:
    def __init__(self, model, intrinsics, camera_size):
        self.k = np.asarray(intrinsics)
        self.camera_size = np.asarray(camera_size)
        trained = np.array([[207.8461,0.,160.],[0.,207.8461,120.],[0.,0.,1.]])
        if tuple(self.camera_size) != (320,240) or not np.allclose(self.k,trained,atol=.05,rtol=0):
            raise ValueError('Rubble precision model was trained for the Go2 320x240 camera calibration')
        self.depth = RGBRubbleDepth(model, self.k, camera_size, threshold=.35)
        if self.depth.size != (64,64):
            raise ValueError('The precision layer requires a 64x64 RGB model')
        self.agreement = RubbleConsensus()

    def clear(self):
        self.agreement.history.clear()

    def observe(self, image, pose, rotation, origin, base_z, plane, stamp, sparse, dense):
        cloud = np.vstack((sparse,dense))
        if not len(cloud):
            return np.empty((0,5)), np.empty((0,5))
        heading = rot2(pose.yaw)
        camera = np.r_[heading@origin[:2]+pose.xy(),base_z+origin[2]]
        dense_mask = np.arange(len(cloud)) >= len(sparse)
        consistent = self.agreement.update(cloud,dense_mask,stamp,camera)
        precise = cloud[consistent]
        if plane is None or not np.all(np.isfinite(plane)):
            return precise, np.empty((0,5))
        unknown = cloud[~np.isfinite(cloud[:,4])].copy()
        local = np.column_stack(((unknown[:,:2]-pose.xy()) @ heading, unknown[:,2]-base_z))
        optical = (local-origin) @ rotation
        in_front = optical[:,2] > .1
        optical,unknown = optical[in_front],unknown[in_front]
        projected = optical @ self.k.T
        uv = projected[:,:2] / projected[:,2,None]
        pixels = np.rint((uv+.5)*np.asarray(SIZE)/self.camera_size-.5).astype(int)
        inside = ((pixels>=0)&(pixels<np.asarray(SIZE))).all(axis=1)
        if not inside.any():
            return precise, np.empty((0,5))
        unknown,optical = unknown[inside],optical[inside]
        x,y = pixels[inside].T
        depth,relief,score = self.depth.infer(image,rotation[2],origin[2]-plane[2])
        agree = ((score[y,x] >= self.depth.threshold) & (abs(depth[y,x]-optical[:,2]) <= .30)
                 & np.isfinite(relief[y,x]))
        unknown = unknown[agree]
        unknown[:,4] = relief[y[agree],x[agree]]
        unknown[:,3] = unknown[:,2]-unknown[:,4]
        return precise, unknown


def snapshot_precision(layer, history, grid):
    masks = {
        'rgb_precision_points': defaultdict(lambda: defaultdict(list)),
        'rgb_height_candidates': defaultdict(lambda: defaultdict(list)),
    }
    for row in history:
        pose = row['pose']
        view = tuple(np.floor(pose.xy()/.15).astype(int))
        for name,index in masks.items():
            values = row.get(name)
            if values is None or not len(values):
                continue
            xy = transform_xy(values[:,:2],pose)
            groups = defaultdict(list)
            for point,z,height in zip(xy,values[:,2],values[:,4]):
                groups[grid.cell(point)].append([*point,z,height])
            for cell, samples in groups.items():
                sample = np.asarray(samples)
                height = sample[np.isfinite(sample[:,3]),3]
                index[cell][view].append(np.r_[np.median(sample[:,:3],axis=0),
                                              np.quantile(height,.8) if len(height) else np.nan])
    for item in layer['cells']:
        cell = tuple(item['ij'])
        positions = masks['rgb_precision_points'].get(cell,{})
        possible = masks['rgb_height_candidates'].get(cell,{})
        if len(positions) >= 2:
            views = [np.median(np.asarray(values)[:,:3],axis=0) for values in positions.values()]
            item['rgb_consensus_xyz_m'] = np.median(views,axis=0).tolist()
            item['rgb_consensus_view_bins'] = len(positions)
            height, count = height_across_views(positions)
            if height is not None:
                item['rgb_consensus_relief_m'] = height
                item['rgb_consensus_relief_view_bins'] = count
        if item['height_observed_min_max_m'] is None and 'height_estimated_m' not in item:
            height,count = height_across_views(possible)
            if height is not None:
                item['rgb_relief_candidate_m'] = height
                item['rgb_relief_candidate_view_bins'] = count
    layer['rgb_precision'] = {
        'source':'rgb_multiview_consensus_and_rgb64_relief_candidate',
        'consensus_cells':sum('rgb_consensus_xyz_m' in item for item in layer['cells']),
        'consensus_relief_cells':sum('rgb_consensus_relief_m' in item for item in layer['cells']),
        'additional_relief_candidate_cells':sum('rgb_relief_candidate_m' in item for item in layer['cells']),
    }


def height_across_views(views):
    estimates = []
    for observations in views.values():
        heights = np.asarray(observations)[:,3]
        heights = heights[np.isfinite(heights)]
        if len(heights):
            estimates.append(np.median(heights))
    if len(estimates) < 2:
        return None,0
    estimates = np.asarray(estimates)
    reliable = estimates[np.abs(estimates-np.median(estimates)) <= .035]
    return (float(np.median(reliable)),len(reliable)) if len(reliable) >= 2 else (None,0)
