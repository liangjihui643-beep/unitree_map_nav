#!/usr/bin/env python3
"""Stress-test the low-obstacle detector: rubble precision audit + non-rubble (wood) piles."""
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
OBSTACLE_GROUP = 4


def pile_geoms(model):
    centres, body_of = [], []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("pile"):
            ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
            centres.append(ids)
    return centres


def wood_scene(seed=0, n=14):
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(str(ROOT / "robot/go2/physics/go2.xml"))
    spec.worldbody.add_light(pos=[1.0, 0.0, 2.5], dir=[0.2, 0.0, -1.0])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4.0, 4.0, 0.1],
                            pos=[0.0, 0.0, 0.0], rgba=[0.35, 0.35, 0.35, 1.0])
    centres = []
    for i in range(n):
        x = rng.uniform(1.0, 2.4)
        y = rng.uniform(-0.6, 0.6)
        h = rng.uniform(0.06, 0.24)
        is_box = i % 2 == 0
        if is_box:
            geom = spec.worldbody.add_geom(name="stress_%d" % i,
                                           type=mujoco.mjtGeom.mjGEOM_BOX,
                                           size=[rng.uniform(0.10, 0.30), rng.uniform(0.06, 0.12), h / 2],
                                           pos=[x, y, h / 2],
                                           rgba=[0.45, 0.30, 0.15, 1.0])
        else:
            geom = spec.worldbody.add_geom(name="stress_%d" % i,
                                           type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                           size=[rng.uniform(0.05, 0.10), h / 2],
                                           pos=[x, y, h / 2],
                                           rgba=[0.40, 0.28, 0.16, 1.0])
        centres.append([x, y])
    model = spec.compile()
    return model, np.array(centres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/seg_low_obstacle_v8.onnx"))
    ap.add_argument("--out", default="/tmp/stress_low_obstacle.json")
    args = ap.parse_args()
    cfg = Config()
    det = LowObstacleDetector(args.model, cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
    report = {}

    # --- A. rubble precision audit on official scene, held-out pile_01 views ---
    scene = mujoco.MjModel.from_binary_path(str(ROOT / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(scene)
    if scene.nkey:
        mujoco.mj_resetDataKeyframe(scene, data, 0)
    mujoco.mj_forward(scene, data)
    pile_centres = {}
    for b in range(scene.nbody):
        name = mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("pile"):
            ids = [g for g in range(scene.ngeom) if scene.geom_bodyid[g] == b]
            pile_centres[name] = np.array([data.geom_xpos[g] for g in ids])[:, :2]
    rubble_ids = [g for b in range(scene.nbody)
                  for g in range(scene.ngeom)
                  if (mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_BODY, scene.geom_bodyid[g]) or "").startswith("pile")]
    orig_group = scene.geom_group[rubble_ids].copy()
    scene.geom_group[rubble_ids] = OBSTACLE_GROUP
    sensors = SimSensors(scene, data, cfg)
    rock_xy = np.vstack([v for v in pile_centres.values()])
    try:
        score = {"tp": 0, "fp": 0, "views": {}}
        for dist, ang in ((1.2, -0.2), (1.2, 0.25), (2.0, -0.2), (2.0, 0.25)):
            c = pile_centres["pile_01"].mean(0)
            sx, sy = c[0] - dist * math.cos(ang), c[1] - dist * math.sin(ang)
            data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
            mujoco.mj_forward(scene, data)
            sensors.option.geomgroup[OBSTACLE_GROUP] = 1
            rgb = sensors.image()
            sensors.option.geomgroup[OBSTACLE_GROUP] = 0
            dets = det.detect(rgb)
            cc, ss = math.cos(ang), math.sin(ang)
            tp = fp = 0
            for d in dets:
                wx = sx + cc * d.x - ss * d.y
                wy = sy + ss * d.x + cc * d.y
                if np.min(np.linalg.norm(rock_xy - np.array([wx, wy]), axis=1)) < 0.35:
                    tp += 1
                else:
                    fp += 1
            score["tp"] += tp
            score["fp"] += fp
            score["views"]["pile_01_d%.1f_a%+.2f" % (dist, ang)] = {"det": len(dets), "tp": tp, "fp": fp}
        score["precision"] = round(score["tp"] / max(1, score["tp"] + score["fp"]), 3)
        report["rubble"] = score
    finally:
        scene.geom_group[rubble_ids] = orig_group
        sensors.close()

    # --- B. non-rubble (wood/box/cylinder) piles, held-out random seeds ---
    wood = {}
    for seed in (11, 23):
        model, centres = wood_scene(seed=seed)
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        mujoco.mj_forward(model, d)
        # move robot 0.8 m back from the pile centroid along -x
        cx, cy = centres.mean(0)
        d.qpos[:7] = [cx - 1.3, cy, .35, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, d)
        ids = [g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("stress_")]
        orig = model.geom_group[ids].copy()
        model.geom_group[ids] = OBSTACLE_GROUP
        s = SimSensors(model, d, cfg)
        try:
            s.option.geomgroup[OBSTACLE_GROUP] = 1
            rgb = s.image()
            s.option.geomgroup[OBSTACLE_GROUP] = 0
            gt = s.image()
            diff = np.abs(rgb.astype(np.int16) - gt.astype(np.int16)).sum(2)
            mask = (diff > 12).astype(np.uint8)
            dets = det.detect(rgb)
            tp = sum(1 for o in dets if np.min(np.linalg.norm(centres - np.array([o.x, o.y]), axis=1)) < 0.35)
            wood["seed%d" % seed] = {"gt_pixels": int(mask.sum()), "det": len(dets),
                                     "tp": tp, "fp": len(dets) - tp}
        finally:
            model.geom_group[ids] = orig
            s.close()
    report["wood"] = wood
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
