#!/usr/bin/env python3
"""伤员（足球）识别：RGB 颜色阈值检测 + 激光测距融合。

对应比赛任务 2"生命体探测"的最小可用实现：在 320x240 前视图中找
橙色球团，得到方位角；用该方位附近最近的一束激光距离作为测距，
融合出目标世界坐标。评测时可替换为训练的检测网络（同样走 ONNX）。
"""
import math

import numpy as np


def detect_ball(rgb, fov_deg=90.0):
    """返回 (bearing_rad, pixel_area) 或 None。bearing>0 表示目标在左。"""
    if rgb is None:
        return None
    h, w, _ = rgb.shape
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (r > 150) & (g > 60) & (g < 160) & (b < 110) & (r - b > 70)
    area = int(mask.sum())
    if area < 30:                        # ~0.03% 画面，滤噪
        return None
    ys, xs = np.nonzero(mask)
    cx = xs.mean()
    # 图像 x -> 方位角（画面中心为 0，左正）
    bearing = math.radians((0.5 - cx / w) * fov_deg)
    return float(bearing), area


def estimate_position(px, py, yaw, bearing, ring, max_range=6.0):
    """激光辅助测距：取方位 ±10° 内最近的有效回波作为目标距离。"""
    n = len(ring)
    world_ang = yaw + bearing
    center = int(((world_ang % (2 * math.pi)) / (2 * math.pi)) * n) % n
    best = None
    for k in range(-2, 3):
        r = ring[(center + k) % n]
        if math.isfinite(r) and r < max_range and (best is None or r < best):
            best = r
    if best is None:
        return None
    return px + best * math.cos(world_ang), py + best * math.sin(world_ang)
