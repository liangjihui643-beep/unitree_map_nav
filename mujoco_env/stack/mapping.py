#!/usr/bin/env python3
"""占据栅格建图（对应 Gazebo 侧 Gmapping+costmap 的轻量移植）。

只用契约传感器：360° 激光水平环（9×72 的 0° 行）+ 里程计位姿。
log-odds 更新，含膨胀层。评测无 GPS——这里的位姿用仿真里程计代替，
正式评测应由策略用 IMU/编码器自定位（README 已声明）。
"""
import math
from collections import deque

import numpy as np


class OccupancyGrid:
    def __init__(self, x0, y0, w, h, res=0.1):
        self.x0, self.y0, self.res = x0, y0, res
        self.w, self.h = int(w / res), int(h / res)
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self.hit = np.zeros((self.h, self.w), dtype=np.uint32)

    def to_cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def to_world(self, cx, cy):
        return self.x0 + (cx + 0.5) * self.res, self.y0 + (cy + 0.5) * self.res

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h

    LO_OCC, LO_FREE = 0.9, -0.5     # 阈值：单圈扫描即定型（一次 miss=-0.6 < -0.5）

    def value(self, cx, cy):
        """0=未知 1=空闲 2=占用"""
        if not self.in_bounds(cx, cy):
            return 2
        lo = self.logodds[cy, cx]
        if lo > self.LO_OCC:
            return 2
        if lo < self.LO_FREE:
            return 1
        return 0

    def update(self, px, py, yaw, ring, max_range=8.0, miss=0.6, hit_inc=1.4):
        n = len(ring)
        for i, r in enumerate(ring):
            if not math.isfinite(r) or r >= max_range:
                self._ray(px, py, yaw + 2 * math.pi * i / n, max_range, miss, False)
            else:
                self._ray(px, py, yaw + 2 * math.pi * i / n, r, miss, True, hit_inc)
        np.clip(self.logodds, -8, 8, out=self.logodds)

    def _ray(self, px, py, ang, dist, miss, is_hit, hit_inc=1.4):
        ex, ey = px + dist * math.cos(ang), py + dist * math.sin(ang)
        x0, y0 = self.to_cell(px, py)
        x1, y1 = self.to_cell(ex, ey)
        for cx, cy in _bresenham(x0, y0, x1, y1):
            if not self.in_bounds(cx, cy):
                break
            self.logodds[cy, cx] -= miss
        if is_hit and self.in_bounds(x1, y1):
            self.logodds[y1, x1] += hit_inc + miss
            self.hit[y1, x1] += 1

    def inflate_cells(self, radius):
        """返回膨胀后的占用 mask（机器人半径+安全距离）"""
        occ = self.logodds > self.LO_OCC
        r = int(radius / self.res)
        mask = occ.copy()
        ys, xs = np.nonzero(occ)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                ny, nx = ys + dy, xs + dx
                ok = (ny >= 0) & (ny < self.h) & (nx >= 0) & (nx < self.w)
                mask[ny[ok], nx[ok]] = True
        return mask

    def known_fraction(self):
        return float(np.mean(np.abs(self.logodds) > 0.5))

    def frontier_goals(self, px, py, min_size=4):
        """前沿检测（explore_lite 的移植精简版）：从机器人出发的可达空闲区
        与未知的边界聚类，返回候选目标（世界坐标）列表。"""
        start = self.to_cell(px, py)
        if not self.in_bounds(*start):
            return []
        free = self.logodds < self.LO_FREE
        seen = np.zeros_like(free)
        q = deque([start])
        seen[start[1], start[0]] = True
        front = np.zeros_like(free)
        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                if not self.in_bounds(nx, ny) or seen[ny, nx]:
                    continue
                seen[ny, nx] = True
                if free[ny, nx]:
                    q.append((nx, ny))
                elif abs(self.logodds[ny, nx]) <= 1.0:
                    front[cy, cx] = True   # 可达空闲格，邻接未知 -> 前沿
        # 聚类（4 连通）
        goals = []
        visited = np.zeros_like(front)
        ys, xs = np.nonzero(front)
        for sx, sy in zip(xs, ys):
            if visited[sy, sx]:
                continue
            q = deque([(sx, sy)])
            visited[sy, sx] = True
            cells = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                    if self.in_bounds(nx, ny) and front[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(cells) >= min_size:
                mx = sum(c[0] for c in cells) / len(cells)
                my = sum(c[1] for c in cells) / len(cells)
                goals.append(self.to_world(mx, my))
        goals.sort(key=lambda g: math.hypot(g[0]-px, g[1]-py))
        return goals

    def to_image(self):
        img = np.full((self.h, self.w, 3), 128, dtype=np.uint8)
        img[self.logodds < -0.5] = (235, 235, 235)
        img[self.logodds > 0.9] = (40, 40, 40)
        return img[::-1]


def _bresenham(x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
