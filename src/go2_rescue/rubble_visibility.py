"""Reject ground hypotheses hidden behind established occupied returns."""
import numpy as np

from .geometry import bresenham


def visible_ground_hypotheses(grid, origin_xy, points_xy):
    start = grid.cell(origin_xy)
    reach_pad = max(.15, 1.5*grid.c.resolution)
    visible = np.ones(len(points_xy), dtype=bool)
    if not grid.inside(start):
        return np.zeros_like(visible)
    for index, point in enumerate(points_xy):
        end = grid.cell(point)
        if not grid.inside(end):
            visible[index] = False
            continue
        for cell in bresenham(start, end):
            if grid.logodds[cell] < 2.:
                continue
            if np.linalg.norm(grid.world(cell)-point) > reach_pad:
                visible[index] = False
                break
    return visible
