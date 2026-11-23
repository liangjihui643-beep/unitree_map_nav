"""Sparse map-frame terrain snapshot for display and downstream planning."""
import json
from collections import defaultdict
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label


def infer_terrain(layer, logodds, views):
    """Fill only short gaps inside a bounded, multi-view footprint."""
    observed = {tuple(item['ij']): item for item in layer['cells']}
    if not observed:
        return
    resolution = layer['resolution_m']
    origin = np.asarray(layer['origin_xy_m'])
    shape = tuple(layer['shape'])
    direct = np.zeros(shape, dtype=bool)
    indices = np.asarray(list(observed))
    direct[tuple(indices.T)] = True
    distance = distance_transform_edt(~direct) * resolution
    seeds = direct | ((logodds > 1.5) & (distance <= .15 + 1e-9))
    groups, _ = label(binary_dilation(seeds))
    inferred_groups = np.zeros(shape, dtype=groups.dtype)
    eligible = (distance <= .25) & ~direct
    for group, region in enumerate(find_objects(groups), 1):
        local_groups = groups[region]
        local_direct = direct[region]
        points = np.argwhere((local_groups == group) & seeds[region])
        visual = points[local_direct[tuple(points.T)]]
        if len(visual) < 7 or np.ptp(points, axis=0).max() * resolution > 2.4:
            continue
        offset = np.array([extent.start for extent in region])
        if len(set().union(*(views.get(tuple(cell), set()) for cell in visual+offset))) < 2:
            continue
        hull = cv2.convexHull(points[:, ::-1].astype(np.int32))
        mask = np.zeros(local_groups.shape, np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        supported = mask.astype(bool) & eligible[region] & ((local_groups == group) | (local_groups == 0))
        assigned = inferred_groups[region]
        assigned[supported & (assigned == 0)] = group
    for index in np.argwhere(inferred_groups):
        cell = tuple(index)
        observed[cell] = {
            'ij': index.tolist(), 'xy_m': (origin + (index + .5) * resolution).tolist(),
            'rgb_projection_frames': 0, 'lidar_neighbourhood_frames': 0,
            'height_observed_min_max_m': None, 'lidar_ray_free': bool(logodds[cell] < -.5),
            'inferred_footprint': True,
        }
    layer['cells'] = list(observed.values())


def add_visual_height(layer, samples, view_quantile=.8):
    for item in layer['cells']:
        if item['height_observed_min_max_m'] is not None:
            continue
        views = samples.get(tuple(item['ij']), {})
        if len(views) < 2:
            continue
        estimates = np.asarray([np.quantile(heights, view_quantile) for heights in views.values()])
        center = np.median(estimates)
        consistent = estimates[np.abs(estimates-center) <= .035]
        if len(consistent) < 2:
            continue
        item['height_estimated_m'] = float(np.median(consistent))
        item['height_estimate_source'] = 'rgb_two_view'
        item['height_estimate_support_views'] = len(consistent)


def terrain_snapshot(grid, visual, confirmed, heights, stamp, navigation=None, *, views=None,
                     visual_heights=None, reconstructed=False, visual_height_quantile=.8):
    cells = []
    for cell in sorted(visual.keys() | confirmed.keys()):
        rgb_frames = visual.get(cell, 0)
        lidar_frames = confirmed.get(cell, 0)
        if not grid.inside(cell) or (rgb_frames < (1 if reconstructed else 3) and lidar_frames < 2):
            continue
        samples = heights.get(cell, ())
        cells.append({
            "ij": [int(v) for v in cell],
            "xy_m": grid.world(cell).tolist(),
            "rgb_projection_frames": 0 if reconstructed else int(rgb_frames),
            "rgb_reconstructed_pairs": int(rgb_frames) if reconstructed else 0,
            "lidar_neighbourhood_frames": int(lidar_frames),
            "height_observed_min_max_m": [float(min(samples)), float(max(samples))]
            if lidar_frames >= 2 and len(samples) else None,
            "lidar_ray_free": bool(grid.seen[cell] and grid.logodds[cell] < -.5),
        })
    avoidance = None
    if navigation is not None:
        avoidance = {
            "origin_xy_m": navigation.origin.tolist(),
            "resolution_m": navigation.c.resolution,
            "shape": list(navigation.visual_votes.shape),
            "visual_avoid_indices": np.argwhere(navigation.visual_votes >= 2).tolist(),
            "target_avoid_indices": np.argwhere(navigation.target_keepout).tolist(),
        }
    layer = {
        "schema": "go2-terrain-layer/1",
        "frame": "map_start",
        "time_s": float(stamp),
        "origin_xy_m": grid.origin.tolist(),
        "resolution_m": grid.c.resolution,
        "shape": list(grid.seen.shape),
        "cells": cells,
        "navigation_visual_avoidance": avoidance,
    }
    started = perf_counter()
    if views is not None:
        infer_terrain(layer, grid.logodds, views)
    if visual_heights:
        add_visual_height(layer, visual_heights, visual_height_quantile)
    layer['inference_cpu_wall_ms'] = (perf_counter()-started)*1000
    return layer


def collect_cell_heights(grid, height_observations, transform_xy):
    heights = defaultdict(list)
    for points, residuals, pose in height_observations:
        for xy, residual in zip(transform_xy(points, pose), residuals):
            heights[grid.cell(xy)].append(residual)
    return heights


def write_terrain_snapshot(path, layer):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(layer, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
