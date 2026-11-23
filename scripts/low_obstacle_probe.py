#!/usr/bin/env python3
"""Render one MuJoCo frame and run the migrated RGB low-obstacle detector.

Two modes:
  * default scene: official competition MJB, robot at its home keyframe;
  * ``--synthetic-rock``: Go2 MJCF with a single 20 cm box 1.5 m ahead, used
    only to exercise the detector's positive path (render/diagnostic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors

ROOT = Path(__file__).resolve().parents[1]


def build_synthetic_rock(robot_xml: Path, distance=1.5, height=0.20):
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.worldbody.add_geom(
        name="synthetic_rock",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.25, 0.25, height / 2],
        pos=[distance, 0.0, height / 2],
        rgba=[0.55, 0.52, 0.48, 1.0],
    )
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "scene/mujoco/competition-course.mjb"))
    ap.add_argument("--model", default=str(ROOT / "models/seg_low_obstacle_v2.onnx"))
    ap.add_argument("--image-out", default="/tmp/low_obstacle_frame.png")
    ap.add_argument("--spawn", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW"), help="diagnostic render pose only")
    ap.add_argument("--synthetic-rock", action="store_true")
    args = ap.parse_args()

    if args.synthetic_rock:
        model = build_synthetic_rock(ROOT / "robot/go2/physics/go2.xml")
    else:
        model = mujoco.MjModel.from_binary_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if args.spawn is not None:
        x, y, yaw = args.spawn
        data.qpos[:7] = [x, y, .35, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    try:
        image = sensors.image()
        detector = LowObstacleDetector(args.model, cfg.camera_matrix,
                                       cfg.body_from_camera, cfg.base_height)
        detections = detector.detect(image)
    finally:
        sensors.close()
    cv2.imwrite(args.image_out, image)
    world = None
    if args.spawn is not None:
        sx, sy, syaw = args.spawn
        c, s = np.cos(syaw), np.sin(syaw)
        world = [(sx + c * d.x - s * d.y, sy + s * d.x + c * d.y) for d in detections]
    stats = None
    if args.spawn is not None and world is not None:
        piles = {}
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not name.startswith("pile"):
                continue
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
            pos = np.array([data.geom_xpos[g] for g in gs])
            piles[name] = pos.mean(0)[:2]
        centres = np.array(list(piles.values()))
        wp = np.array(world)
        if len(wp):
            dist = np.linalg.norm(wp[:, None, :] - centres[None, :, :], axis=2)
            near = dist.min(axis=1) < 0.9
            stats = {
                "piles": {k: [round(float(v[0]), 2), round(float(v[1]), 2)] for k, v in piles.items()},
                "detections_near_pile": int(near.sum()),
                "detections_far_from_pile": int((~near).sum()),
                "near_ratio": round(float(near.mean()), 3),
                "per_pile": {k: int((np.linalg.norm(wp - centres[i], axis=1) < 0.9).sum())
                             for i, k in enumerate(piles)},
            }
    print(json.dumps({
        "image": args.image_out,
        "detections": len(detections),
        "top": [{"x": round(d.x, 3), "y": round(d.y, 3), "h": round(d.height, 3),
                 "score": round(d.score, 3)} for d in detections[:5]],
        "world_top": [{"x": round(x, 2), "y": round(y, 2)} for x, y in (world[:5] if world else [])],
        "stats": stats,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
