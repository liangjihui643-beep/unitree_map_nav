"""Optional 2-D lidar scan preprocessing migrated from the Gazebo stack.

The Gazebo mapping pipeline used three filters before scan matching:
range clamping, a speckle (isolated-return) filter, and a shadow filter that
drops returns at occlusion edges.  In MuJoCo the ray caster already clamps
ranges, so only the last two are relevant here.  Both are disabled by default
so existing simulations and tests keep their behaviour.
"""
from __future__ import annotations

import numpy as np

INVALID = -1.0


def speckle_filter(ranges: np.ndarray, window: int, delta: float) -> np.ndarray:
    """Drop isolated returns whose neighbours disagree by more than ``delta``.

    ``ranges`` is a 2-D array (elevation rows x azimuth columns) of measured
    distances; invalid entries are <= 0.  A return is kept when at least half
    of its in-window azimuth neighbours are within ``delta`` metres.
    """
    if window <= 0 or delta <= 0:
        return ranges
    out = ranges.copy()
    rows, cols = ranges.shape
    for r in range(rows):
        row = ranges[r]
        valid = row > 0
        if valid.sum() < 3:
            continue
        keep = np.zeros(cols, dtype=bool)
        for c in range(cols):
            if not valid[c]:
                continue
            lo, hi = max(0, c - window), min(cols, c + window + 1)
            neigh = [k for k in range(lo, hi) if k != c and valid[k]]
            if not neigh:
                keep[c] = True
                continue
            close = sum(abs(row[k] - row[c]) <= delta for k in neigh)
            keep[c] = close >= max(1, len(neigh) // 2)
        out[r, ~keep] = INVALID
    return out


def shadow_filter(ranges: np.ndarray, azimuth_step: float, min_angle_deg: float,
                  neighbours: int = 1, min_gap: float = 0.25) -> np.ndarray:
    """Drop occlusion-edge returns (classic lidar shadow filter).

    For each return the angle between the segments to its two azimuth
    neighbours is computed; a small angle means the ray grazed an obstacle
    edge and the return is discarded, matching ``laser_filters``' behaviour.
    """
    if min_angle_deg <= 0 or azimuth_step <= 0 or neighbours <= 0:
        return ranges
    limit = np.deg2rad(min_angle_deg)
    out = ranges.copy()
    rows, cols = ranges.shape
    angles = np.arange(cols) * azimuth_step
    for r in range(rows):
        row = ranges[r]
        if (row > 0).sum() < 3:
            continue
        x = np.where(row > 0, row * np.cos(angles), 0.0)
        y = np.where(row > 0, row * np.sin(angles), 0.0)
        for c in range(neighbours, cols - neighbours):
            if row[c] <= 0:
                continue
            a = np.array([x[c] - x[c - neighbours], y[c] - y[c - neighbours]])
            b = np.array([x[c + neighbours] - x[c], y[c + neighbours] - y[c]])
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                out[r, c] = INVALID
                continue
            gap = max(abs(row[c] - row[c - neighbours]), abs(row[c + neighbours] - row[c]))
            cos_angle = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            if np.arccos(cos_angle) < limit and gap > min_gap:
                out[r, c] = INVALID
    return out
