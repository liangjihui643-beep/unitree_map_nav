#!/usr/bin/env python3
"""Collect RGB + difference-rendered rubble masks around official rubble piles.

MuJoCo's segmentation renderer can fail with an internal geom-id table error
on this scene, so labels are produced by rendering the scene twice: once with
the rubble geoms assigned to a dedicated visibility group, once with that
group hidden.  Pixels that change belong to rubble.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.config import Config
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/mujoco_rubble"))
    ap.add_argument("--scene", default=str(ROOT / "scene/mujoco/competition-course.mjb"))
    ap.add_argument("--distances", nargs="+", type=float, default=[1.0, 1.6, 2.4])
    ap.add_argument("--angles", nargs="+", type=float, default=[-0.4, 0.0, 0.4])
    ap.add_argument("--test-pile", default="pile_01")
    args = ap.parse_args()

    model = mujoco.MjModel.from_binary_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    piles, pile_bodies = pile_centres(model, data)
    rubble_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in pile_bodies]
    original_group = model.geom_group[rubble_geoms].copy()
    model.geom_group[rubble_geoms] = RUBBLE_GROUP
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta = {"train": [], "test": [], "piles": {k: v.tolist() for k, v in piles.items()}}
    try:
        for name, centre in piles.items():
            split = "test" if name == args.test_pile else "train"
            for dist in args.distances:
                for ang in args.angles:
                    yaw = ang
                    sx = centre[0] - dist * math.cos(yaw)
                    sy = centre[1] - dist * math.sin(yaw)
                    data.qpos[:7] = [sx, sy, .35, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
                    mujoco.mj_forward(model, data)
                    sensors.option.geomgroup[RUBBLE_GROUP] = 1
                    with_rubble = sensors.image()
                    sensors.option.geomgroup[RUBBLE_GROUP] = 0
                    without = sensors.image()
                    sensors.option.geomgroup[RUBBLE_GROUP] = 0
                    diff = np.abs(with_rubble.astype(np.int16) - without.astype(np.int16)).sum(2)
                    mask = (diff > 12).astype(np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    if mask.sum() < 200:
                        continue
                    name_i = "%s_d%.1f_a%+.2f" % (name, dist, ang)
                    cv2.imwrite(str(out / "images" / (name_i + ".png")), with_rubble)
                    cv2.imwrite(str(out / "masks" / (name_i + ".png")), mask)
                    meta[split].append(name_i)
    finally:
        model.geom_group[rubble_geoms] = original_group
        sensors.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"train": len(meta["train"]), "test": len(meta["test"]), "out": str(out)}))


if __name__ == "__main__":
    main()
