#!/usr/bin/env python3
"""A* 全局规划（对应 Gazebo 侧 GlobalPlanner 的移植精简版）+ 视线平滑。"""
import heapq
import math

import numpy as np


def astar(grid, start_w, goal_w, inflation=0.30):
    """grid: OccupancyGrid。返回世界坐标路点列表（含起终点），失败返回 None。"""
    blocked = grid.inflate_cells(inflation)
    sx, sy = grid.to_cell(*start_w)
    gx, gy = grid.to_cell(*goal_w)
    if not grid.in_bounds(sx, sy) or not grid.in_bounds(gx, gy):
        return None
    if blocked[gy, gx]:
        # 目标被膨胀区覆盖：就近找空闲格
        found = _nearest_free(blocked, gx, gy)
        if found is None:
            return None
        gx, gy = found
    openq = [(0.0, sx, sy)]
    gs = {(sx, sy): 0.0}
    parent = {}
    while openq:
        f, cx, cy = heapq.heappop(openq)
        if (cx, cy) == (gx, gy):
            path = []
            node = (gx, gy)
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append((sx, sy))
            path.reverse()
            return _smooth(grid, blocked, path)
        for dx, dy, c in ((1,0,1),(-1,0,1),(0,1,1),(0,-1,1),
                          (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)):
            nx, ny = cx + dx, cy + dy
            if not grid.in_bounds(nx, ny) or blocked[ny, nx]:
                continue
            ng = gs[(cx, cy)] + c
            if ng < gs.get((nx, ny), float('inf')):
                gs[(nx, ny)] = ng
                parent[(nx, ny)] = (cx, cy)
                h = math.hypot(gx - nx, gy - ny)
                heapq.heappush(openq, (ng + h, nx, ny))
    return None


def _nearest_free(blocked, gx, gy):
    for r in range(1, 12):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < blocked.shape[0] and 0 <= nx < blocked.shape[1] \
                        and not blocked[ny, nx]:
                    return nx, ny
    return None


def _smooth(grid, blocked, path_cells):
    """视线法捷径"""
    pts = [grid.to_world(cx, cy) for cx, cy in path_cells]
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _los(grid, blocked, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _los(grid, blocked, a, b):
    x0, y0 = grid.to_cell(*a)
    x1, y1 = grid.to_cell(*b)
    from mapping import _bresenham
    for cx, cy in _bresenham(x0, y0, x1, y1):
        if not grid.in_bounds(cx, cy) or blocked[cy, cx]:
            return False
    return True
