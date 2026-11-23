#!/usr/bin/env python3
"""Compare low-obstacle ONNX models on held-out MuJoCo rubble views."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors

ROOT = Path(__file__).resolve().parents[1]
RUBBLE_GROUP = 4


def pile_centres(model, data):
    piles, bodies = {}, set()
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not name.startswith("pile"):
            continue
        bodies.add(b)
        gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        piles[name] = np.array([data.geom_xpos[g] for g in gs]).mean(0)
    return piles, bodies


def gt_mask(sensors, model):
    sensors.option.geomgroup[RUBBLE_GROUP] = 1
    a = sensors.image()
    sensors.option.geomgroup[RUBBLE_GROUP] = 0
    b = sensors.image()
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(2)
    mask = (diff > 12).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", default=[str(ROOT / "models/seg_low_obstacle.onnx")])
    ap.add_argument("--scene", default=str(ROOT / "scene/mujoco/competition-course.mjb"))
    ap.add_argument("--out", default="/tmp/eval_low_obstacle.json")
    args = ap.parse_args()
    model = mujoco.MjModel.from_binary_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    piles, bodies = pile_centres(model, data)
    rubble = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
    orig = model.geom_group[rubble].copy()
    model.geom_group[rubble] = RUBBLE_GROUP
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    # held-out views: pile_01 with distances/angles absent from training
    views = []
    for dist in (1.2, 2.0):
        for ang in (-0.2, 0.25):
            views.append(("pile_01", dist, ang))
    views.append(("pile_04", 1.3, 0.15))
    results = {}
    try:
        images = {}
        for name, dist, ang in views:
            c = piles[name]
            sx, sy = c[0] - dist * math.cos(ang), c[1] - dist * math.sin(ang)
            data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
            mujoco.mj_forward(model, data)
            mask, rgb = gt_mask(sensors, model)
            images[(name, dist, ang)] = (rgb, mask, c, (sx, sy, ang))
        for onnx in args.onnx:
            det = LowObstacleDetector(onnx, cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
            tag = Path(onnx).stem
            results[tag] = {}
            for key, (rgb, mask, c, pose) in images.items():
                prob = det.segment(rgb)
                pred = (prob.argmax(0) == 1).astype(np.uint8)
                tp = int((pred & mask).sum())
                recall = tp / max(1, int(mask.sum()))
                prec = tp / max(1, int(pred.sum()))
                iou = tp / max(1, int(((pred | mask) > 0).sum()))
                dets = det.detect(rgb)
                sx, sy, ang = pose
                cc, ss = math.cos(ang), math.sin(ang)
                near = sum(1 for d in dets
                           if math.hypot(sx + cc * d.x - ss * d.y - c[0],
                                         sy + ss * d.x + cc * d.y - c[1]) < 0.9)
                results[tag]["%s_d%.1f_a%+.2f" % (key[0], key[1], key[2])] = {
                    "pixel_recall": round(recall, 3), "pixel_precision": round(prec, 3),
                    "pixel_iou": round(iou, 3), "detections": len(dets),
                    "detections_near_pile": near,
                    "detections_far": len(dets) - near,
                }
    finally:
        model.geom_group[rubble] = orig
        sensors.close()
    Path(args.out).write_text(json.dumps(results, indent=1))
    for tag, rows in results.items():
        for key, v in rows.items():
            print(tag, key, json.dumps(v))
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
