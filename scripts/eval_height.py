#!/usr/bin/env python3
"""Height accuracy audit with depth-based ground truth.

Rubble pixels are identified by toggling the rubble visibility group; their
metric depth gives a world-frame surface height. A detection is matched to the
highest rubble surface within 0.15 m of its world xy, which avoids the old
AABB-top bias (that counted rocks hidden behind the visible surface).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors

ROOT = Path(__file__).resolve().parents[1]
RUBBLE_GROUP = 4


def main():
    model = mujoco.MjModel.from_binary_path(str(ROOT / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    piles = {}
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("pile"):
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
            piles[n] = np.array([data.geom_xpos[g] for g in gs]).mean(0)
    rubble = [g for g in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("pile")]
    orig = model.geom_group[rubble].copy()
    model.geom_group[rubble] = RUBBLE_GROUP
    cfg = Config()
    det = LowObstacleDetector(str(ROOT / "models/seg_low_obstacle_v4.onnx"),
                              cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
    k = np.asarray(cfg.camera_matrix, float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    t = np.asarray(cfg.body_from_camera, float)
    rot, tr = t[:3, :3], t[:3, 3]
    sensors = SimSensors(model, data, cfg)
    rows = []
    try:
        for name, dist, ang in (("pile_01", 1.2, -0.2), ("pile_01", 1.2, 0.25),
                                ("pile_04", 1.3, 0.15)):
            c = piles[name]
            sx, sy = c[0] - dist * math.cos(ang), c[1] - dist * math.sin(ang)
            data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0., 0., math.sin(ang / 2)]
            mujoco.mj_forward(model, data)
            sensors.option.geomgroup[RUBBLE_GROUP] = 1
            rgb = sensors.image()
            depth = sensors.depth()
            sensors.option.geomgroup[RUBBLE_GROUP] = 0
            plain = sensors.image()
            sensors.option.geomgroup[RUBBLE_GROUP] = 0
            diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
            mask = (diff > 12) & (depth > .05) & (depth < 12)
            vs, us = np.nonzero(mask)
            z = depth[vs, us]
            cam = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
            body = cam @ rot.T + tr
            cc, ss = math.cos(ang), math.sin(ang)
            bx, by = body[:, 0], body[:, 1]
            wx = sx + cc * bx - ss * by
            wy = sy + ss * bx + cc * by
            wz = body[:, 2] + .35
            dets = det.detect(rgb)
            for d in dets:
                dwx = sx + cc * d.x - ss * d.y
                dwy = sy + ss * d.x + cc * d.y
                near = (np.abs(wx - dwx) < .15) & (np.abs(wy - dwy) < .15)
                if near.sum() < 3:
                    continue
                gt = float(np.percentile(wz[near], 90))
                rows.append((name, d.height, gt, d.height - gt, d.clipped))
    finally:
        model.geom_group[rubble] = orig
        sensors.close()
    if not rows:
        print(json.dumps({"n": 0}))
        return
    err = np.array([r[3] for r in rows])
    absg = np.abs(err)
    clean = np.abs(np.array([r[3] for r in rows if not r[4]]))
    print(json.dumps({
        "n": len(rows),
        "n_clipped": int(sum(1 for r in rows if r[4])),
        "clean_median_abs_cm": round(float(np.median(clean)) * 100, 1) if len(clean) else None,
        "clean_p90_abs_cm": round(float(np.percentile(clean, 90)) * 100, 1) if len(clean) else None,
        "median_abs_cm": round(float(np.median(absg)) * 100, 1),
        "p90_abs_cm": round(float(np.percentile(absg, 90)) * 100, 1),
        "bias_cm": round(float(err.mean()) * 100, 1),
        "by_view": {v: int(sum(1 for r in rows if r[0] == v)) for v in {r[0] for r in rows}},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
