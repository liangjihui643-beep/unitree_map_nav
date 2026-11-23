#!/usr/bin/env python3
"""Monocular height estimation (paper-inspired): MiDaS disparity -> metric depth
via ground-plane calibration -> obstacle height, evaluated against MuJoCo depth.

Reference: Ahn et al., "Monocular Vision-Based Obstacle Height Estimation for
Mobile Robot", Applied Sciences 15(23):12711, 2025 (depth -> height stage).
We keep their height-from-depth idea but use MiDaS-small (local ONNX) as the
depth source because the paper's R-Depth Net needs defocus/optical-flow inputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import onnxruntime as ort

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors

ROOT = Path(__file__).resolve().parents[1]
RUBBLE_GROUP = 4
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)


def midas_depth(image_bgr, session):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    disp = out[0] if out.ndim == 4 else out
    if disp.shape[0] == 1:
        disp = disp[0]
    disp = cv2.resize(disp, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return disp


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
    det = LowObstacleDetector(str(ROOT / "models/seg_low_obstacle_v7.onnx"),
                              cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
    k = np.asarray(cfg.camera_matrix, float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    t = np.asarray(cfg.body_from_camera, float)
    rot, tr = t[:3, :3], t[:3, 3]
    session = ort.InferenceSession("/home/ros/models/midas_small.onnx", providers=["CPUExecutionProvider"])
    sensors = SimSensors(model, data, cfg)
    rows, cals = [], []
    try:
        for name, dist, ang in (("pile_01", 1.2, -0.2), ("pile_01", 1.2, 0.25), ("pile_04", 1.3, 0.15)):
            c = piles[name]
            sx, sy = c[0] - dist * math.cos(ang), c[1] - dist * math.sin(ang)
            data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0., 0., math.sin(ang / 2)]
            mujoco.mj_forward(model, data)
            sensors.option.geomgroup[RUBBLE_GROUP] = 1
            rgb = sensors.image()
            depth = sensors.depth()
            sensors.option.geomgroup[RUBBLE_GROUP] = 0
            plain = sensors.image()
            diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
            rubble_mask = (diff > 12)
            disp = midas_depth(rgb, session)
            cc, ss = math.cos(ang), math.sin(ang)
            R_yaw = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1.]])
            origin = np.array([sx, sy, .35]) + R_yaw @ tr

            def unproject(uu, vv, zz):
                cam = np.stack([(uu - cx) / fx * zz, (vv - cy) / fy * zz, zz], axis=-1)
                body = cam @ rot.T + tr
                world = body @ R_yaw.T + np.array([sx, sy, .35])
                return world

            # ground calibration on non-rubble pixels whose ray hits the ground
            vs, us = np.nonzero(~rubble_mask)
            sample = np.arange(0, len(vs), 7)
            vs, us = vs[sample], us[sample]
            rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=1)
            dirs_world = (rays @ rot.T) @ R_yaw.T
            zs = np.maximum(0.05, dirs_world[:, 2])
            dists = origin[2] / zs
            ok = (dists > .6) & (dists < 5.0)
            if ok.sum() < 50:
                continue
            d = disp[vs[ok], us[ok]]
            A = np.stack([d, np.ones_like(d)], axis=1)
            coef, *_ = np.linalg.lstsq(A, 1.0 / dists[ok], rcond=None)
            a, b = float(coef[0]), float(coef[1])
            cals.append((a, b))

            # rubble pixel heights from calibrated metric depth
            rv, ru = np.nonzero(rubble_mask)
            dd = disp[rv, ru]
            inv = a * dd + b
            good = inv > 1e-3
            rv, ru, inv = rv[good], ru[good], inv[good]
            zz = 1.0 / inv
            world = unproject(ru.astype(float), rv.astype(float), zz)
            wz = world[:, 2]
            keep = (wz > -.2) & (wz < .8)
            world, wz = world[keep], wz[keep]
            if len(wz) < 20:
                continue
            # for each detection, GT from renderer depth and estimate from MiDaS
            gt_v, gt_u = np.nonzero(rubble_mask & (depth > .05) & (depth < 12))
            gt_pts = unproject(gt_u.astype(float), gt_v.astype(float), depth[gt_v, gt_u])
            dets = det.detect(rgb)
            for o in dets:
                dwx = sx + cc * o.x - ss * o.y
                dwy = sy + ss * o.x + cc * o.y
                near = (np.abs(gt_pts[:, 0] - dwx) < .15) & (np.abs(gt_pts[:, 1] - dwy) < .15)
                if near.sum() < 3:
                    continue
                gt = float(np.percentile(gt_pts[near, 2], 90))
                # nearest MiDaS height point
                dist2 = (world[:, 0] - dwx) ** 2 + (world[:, 1] - dwy) ** 2
                if dist2.min() > .15 ** 2:
                    continue
                est = float(wz[dist2.argmin()])
                rows.append((name, est, gt, est - gt))
    finally:
        model.geom_group[rubble] = orig
        sensors.close()
    if not rows:
        print(json.dumps({"n": 0}))
        return
    err = np.abs(np.array([r[3] for r in rows]))
    print(json.dumps({
        "n": len(rows),
        "median_abs_cm": round(float(np.median(err)) * 100, 1),
        "p90_abs_cm": round(float(np.percentile(err, 90)) * 100, 1),
        "bias_cm": round(float(np.array([r[3] for r in rows]).mean()) * 100, 1),
        "calibration": [tuple(round(x, 3) for x in c) for c in cals],
        "pass_10cm": bool(np.median(err) < .10),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
