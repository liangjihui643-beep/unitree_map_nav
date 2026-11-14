#!/usr/bin/env python3
"""局部导航：纯追踪 + VFH 避障（对应 Gazebo 侧 move_base/DWA 的移植精简版）。

输入：全局路点、当前 360° 激光环（72 线，5° 分辨率）、位姿。
输出：(vx, wz) 速度指令给步态控制器。
"""
import math

import numpy as np


class LocalNav:
    def __init__(self, max_vx=0.18, max_wz=0.5, robot_radius=0.28,
                 lookahead=0.45, safety=0.55):
        self.max_vx = max_vx
        self.max_wz = max_wz
        self.robot_radius = robot_radius
        self.lookahead = lookahead
        self.safety = safety

    def cmd(self, path, px, py, yaw, ring):
        """ring: 72 个距离（nan=无回波）。返回 (vx, wz, target)。"""
        # 1) 纯追踪找前瞻点
        target = None
        for wx, wy in path:
            if math.hypot(wx - px, wy - py) > self.lookahead:
                target = (wx, wy)
                break
        if target is None:
            target = path[-1]
        desired = math.atan2(target[1] - py, target[0] - px)
        err = _wrap(desired - yaw)

        # 2) VFH：前方扇区障碍检查，必要时绕行
        n = len(ring)
        steer = err
        vx = self.max_vx * max(0.15, 1.0 - abs(err) * 1.2)
        danger = False
        for i, r in enumerate(ring):
            if not math.isfinite(r) or r > self.safety + self.robot_radius:
                continue
            bearing = 2 * math.pi * i / n
            d = _wrap(bearing - err)          # 相对期望航向
            if abs(d) < 0.6:                   # 正前 ±34°
                danger = True
                gap = self._widest_gap(ring, bearing)
                steer = err + _wrap(gap - bearing)
                vx *= 0.45
                break
        wz = float(np.clip(steer * 2.2, -self.max_wz, self.max_wz))
        if math.hypot(target[0] - px, target[1] - py) < 0.25:
            vx = 0.0
        return vx, wz, target

    def _widest_gap(self, ring, blocked_bearing):
        """在受阻方向附近找最宽的空隙中心"""
        n = len(ring)
        best_gap, best_center = None, 0.0
        i = 0
        while i < n:
            r = ring[i]
            if math.isfinite(r) and r < self.safety + self.robot_radius:
                i += 1
                continue
            j = i
            while j < n and (not math.isfinite(ring[j]) or ring[j] >= self.safety + self.robot_radius):
                j += 1
            width = j - i
            if best_gap is None or width > best_gap:
                center = (i + j - 1) / 2.0
                best_gap, best_center = width, 2 * math.pi * center / n
            i = j
        if best_gap is None:
            return blocked_bearing + math.pi
        return best_center


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
