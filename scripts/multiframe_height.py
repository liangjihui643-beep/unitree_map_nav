#!/usr/bin/env python3
"""Multi-view height transfer: can a farther view fix clipped near-field heights?

For each clipped detection in view A, find the nearest detection in view B
(farther view, likely unclipped) and use its measured height; compare both
against depth-based rubble surface height at the view-A location.
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


def view_frame(model, data, sensors, det, pose, pile_center):
    sx, sy, ang = pose
    data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0., 0., math.sin(ang / 2)]
    mujoco.mj_forward(model, data)
    sensors.option.geomgroup[RUBBLE_GROUP] = 1
    rgb = sensors.image()
    depth = sensors.depth()
    sensors.option.geomgroup[RUBBLE_GROUP] = 0
    plain = sensors.image()
    diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
    mask = (diff > 12) & (depth > .05) & (depth < 12)
    cfg = Config()
    k = np.asarray(cfg.camera_matrix, float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    t = np.asarray(cfg.body_from_camera, float)
    rot, tr = t[:3, :3], t[:3, 3]
    vs, us = np.nonzero(mask)
    z = depth[vs, us]
    cam = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
    body = cam @ rot.T + tr
    cc, ss = math.cos(ang), math.sin(ang)
    wx = sx + cc * body[:, 0] - ss * body[:, 1]
    wy = sy + ss * body[:, 0] + cc * body[:, 1]
    wz = body[:, 2] + .35
    dets = det.detect(rgb)
    out = []
    for d in dets:
        dwx = sx + cc * d.x - ss * d.y
        dwy = sy + ss * d.x + cc * d.y
        near = (np.abs(wx - dwx) < .15) & (np.abs(wy - dwy) < .15)
        gt = float(np.percentile(wz[near], 90)) if near.sum() >= 3 else None
        out.append(dict(x=dwx, y=dwy, h=d.height, clipped=d.clipped, gt=gt))
    return out


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
    rubric = [g for g in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("pile")]
    orig = model.geom_group[rubric].copy()
    model.geom_group[rubric] = RUBBLE_GROUP
    cfg = Config()
    det = LowObstacleDetector(str(ROOT / "models/seg_low_obstacle_v4.onnx"),
                              cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
    sensors = SimSensors(model, data, cfg)
    c = piles["pile_04"]
    near_pose = (c[0] - 1.3 * math.cos(.15), c[1] - 1.3 * math.sin(.15), .15)
    far_pose = (c[0] - 2.4 * math.cos(.15), c[1] - 2.4 * math.sin(.15), .15)
    try:
        near = view_frame(model, data, sensors, det, near_pose, c)
        far = view_frame(model, data, sensors, det, far_pose, c)
    finally:
        model.geom_group[rubric] = orig
        sensors.close()
    rows = []
    for a in near:
        if not a["clipped"] or a["gt"] is None:
            continue
        cands = [b for b in far if not b["clipped"] and b["gt"] is not None]
        if not cands:
            continue
        b = min(cands, key=lambda o: (o["x"] - a["x"]) ** 2 + (o["y"] - a["y"]) ** 2)
        dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        if dist > .3:
            continue
        rows.append((a["gt"], a["h"], b["h"], b["h"] - a["gt"], abs(a["h"] - a["gt"])))
    if not rows:
        print(json.dumps({"matched": 0, "near_view_dets": len(near), "far_view_dets": len(far)}))
        return
    transfer_err = np.abs(np.array([r[3] for r in rows]))
    placeholder_err = np.array([r[4] for r in rows])
    print(json.dumps({
        "near_view_dets": len(near), "far_view_dets": len(far), "matched": len(rows),
        "transfer_median_abs_cm": round(float(np.median(transfer_err)) * 100, 1),
        "transfer_p90_abs_cm": round(float(np.percentile(transfer_err, 90)) * 100, 1),
        "placeholder_median_abs_cm": round(float(np.median(placeholder_err)) * 100, 1),
        "placeholder_p90_abs_cm": round(float(np.percentile(placeholder_err, 90)) * 100, 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
