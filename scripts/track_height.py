#!/usr/bin/env python3
"""Object-level tracking + multi-view height estimate for clipped near-field rubble.

Robot approaches pile_04 in small steps. Each frame's detections are associated
into world-frame tracks; a track's height is the median of its *unclipped*
measurements. Clipped detections in the final (nearest) frame borrow their
track's height. Compared against depth-based surface height.

Exit criterion: if the tracked-height median error is not below 10 cm, record
the negative result and stop this line instead of iterating blindly.
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


def frame(model, data, sensors, det, cfg, pose):
    sx, sy, ang = pose
    data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0., 0., math.sin(ang / 2)]
    mujoco.mj_forward(model, data)
    sensors.option.geomgroup[RUBBLE_GROUP] = 1
    rgb = sensors.image()
    depth = sensors.depth()
    sensors.option.geomgroup[RUBBLE_GROUP] = 0
    plain = sensors.image()
    sensors.option.geomgroup[RUBBLE_GROUP] = 1
    diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
    mask = (diff > 12) & (depth > .05) & (depth < 12)
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
    out = []
    for d in det.detect(rgb):
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
    rubble = [g for g in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("pile")]
    orig = model.geom_group[rubble].copy()
    cfg = Config()
    det = LowObstacleDetector(str(ROOT / "models/seg_low_obstacle_v4.onnx"),
                              cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
    sensors = SimSensors(model, data, cfg)
    c = piles["pile_04"]
    ang = .15
    poses = [(c[0] - d * math.cos(ang), c[1] - d * math.sin(ang), ang) for d in (2.6, 2.3, 2.0, 1.7, 1.4, 1.1)]
    frames = []
    try:
        for p in poses:
            model.geom_group[rubble] = RUBBLE_GROUP
            frames.append(frame(model, data, sensors, det, cfg, p))
    finally:
        model.geom_group[rubble] = orig
        sensors.close()
    # associate world-frame detections across frames (greedy nearest)
    tracks = []  # dict(pts: [(x,y,unclipped_heights)], )
    for fi, f in enumerate(frames):
        for d in f:
            best = None
            for tr in tracks:
                x, y = tr["x"], tr["y"]
                dist = math.hypot(x - d["x"], y - d["y"])
                if dist < .2 and (best is None or dist < best[0]):
                    best = (dist, tr)
            if best is None:
                tracks.append(dict(x=d["x"], y=d["y"], hs=[], clips=0, seen=1))
            else:
                tr = best[1]
                tr["x"], tr["y"] = d["x"], d["y"]
                tr["seen"] += 1
                if d["clipped"]:
                    tr["clips"] += 1
                elif d["h"] is not None:
                    tr["hs"].append(d["h"])
    last = frames[-1]
    rows = []
    for d in last:
        if not d["clipped"] or d["gt"] is None:
            continue
        cands = [tr for tr in tracks if tr["hs"] and math.hypot(tr["x"] - d["x"], tr["y"] - d["y"]) < .25]
        if not cands:
            continue
        tr = min(cands, key=lambda t: math.hypot(t["x"] - d["x"], t["y"] - d["y"]))
        est = float(np.median(tr["hs"]))
        rows.append(dict(gt=d["gt"], est=est, err=est - d["gt"], placeholder=abs(d["h"] - d["gt"])))
    if not rows:
        print(json.dumps({"tracks": len(tracks), "matched": 0}))
        return
    err = np.abs(np.array([r["err"] for r in rows]))
    ph = np.abs(np.array([r["placeholder"] for r in rows]))
    print(json.dumps({
        "tracks": len(tracks), "near_clipped": sum(1 for d in last if d["clipped"]),
        "matched": len(rows),
        "tracked_median_abs_cm": round(float(np.median(err)) * 100, 1),
        "tracked_p90_abs_cm": round(float(np.percentile(err, 90)) * 100, 1),
        "placeholder_median_abs_cm": round(float(np.median(ph)) * 100, 1),
        "pass_10cm": bool(np.median(err) < .10),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
