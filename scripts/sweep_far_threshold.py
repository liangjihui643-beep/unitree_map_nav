#!/usr/bin/env python3
"""Sweep the far-pass probability threshold on held-out rubble views."""
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


def main():
    model = mujoco.MjModel.from_binary_path(str(ROOT / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    centres = {}
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("pile"):
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
            centres[n] = np.array([data.geom_xpos[g] for g in gs]).mean(0)
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    views = [("pile_01", 1.2, -0.2), ("pile_01", 1.2, 0.25),
             ("pile_01", 2.0, -0.2), ("pile_01", 2.0, 0.25),
             ("pile_04", 1.3, 0.15)]
    frames = []
    try:
        for name, dist, ang in views:
            c = centres[name]
            sx, sy = c[0] - dist * math.cos(ang), c[1] - dist * math.sin(ang)
            data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
            mujoco.mj_forward(model, data)
            frames.append((name, dist, ang, sensors.image(), c, (sx, sy, ang)))
    finally:
        sensors.close()
    models = ["models/seg_low_obstacle_v4.onnx", str(ROOT / "models/seg_low_obstacle_v6.pt")]
    onnx_paths = [str(ROOT / "models/seg_low_obstacle_v4.onnx"), "/tmp/seg_v6snap.onnx"]
    for path in onnx_paths:
        det = LowObstacleDetector(path, cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
        for th in (0.30, 0.40, 0.50):
            det.min_prob_far = th
            rows = []
            for name, dist, ang, rgb, c, (sx, sy, a) in frames:
                ds = det.detect(rgb)
                cc, ss = math.cos(a), math.sin(a)
                near = far = 0
                for d in ds:
                    wx, wy = sx + cc * d.x - ss * d.y, sy + ss * d.x + cc * d.y
                    if math.hypot(wx - c[0], wy - c[1]) < 0.9:
                        near += 1
                    else:
                        far += 1
                rows.append((f"{name}_{dist}_{ang}", len(ds), near, far))
            print(Path(path).stem, "th", th, json.dumps(
                {r[0]: {"det": r[1], "near": r[2], "far": r[3]} for r in rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
