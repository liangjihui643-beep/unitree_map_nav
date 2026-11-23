#!/usr/bin/env python3
"""Collect RGB + geom-id segmentation labels around the official rubble piles.

Rubble pixels come from MuJoCo segmentation rendering, so the labels are
self-supervised: geom_bodyid in pile_* => rubble (1); world ground => 0;
everything else (tunnels, ramps, robot) => structure (2).
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
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta = {"train": [], "test": [], "piles": {k: v.tolist() for k, v in piles.items()}}
    idx = 0
    try:
        for name, centre in piles.items():
            split = "test" if name == args.test_pile else "train"
            for dist in args.distances:
                for ang in args.angles:
                    # robot stands `dist` away from the pile, facing it
                    yaw = ang
                    sx = centre[0] - dist * math.cos(yaw)
                    sy = centre[1] - dist * math.sin(yaw)
                    data.qpos[:7] = [sx, sy, .35, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
                    mujoco.mj_forward(model, data)
                    rgb = sensors.image()
                    try:
                        seg = sensors.segmentation()
                    except Exception:
                        continue
                    ids = np.asarray(seg[..., 0], dtype=np.int32)
                    classes = np.zeros(ids.shape, np.uint8)
                    for b in pile_bodies:
                        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
                        classes[np.isin(ids, geoms)] = 1
                    structure = ~np.isin(ids, [0]) & (classes == 0)
                    body_of_geom = model.geom_bodyid[ids]
                    world_body = 0
                    structure &= body_of_geom != world_body
                    classes[structure] = 2
                    if (classes == 1).sum() < 200:
                        continue
                    name_i = "%s_d%.1f_a%+.2f" % (name, dist, ang)
                    cv2.imwrite(str(out / "images" / (name_i + ".png")), rgb)
                    cv2.imwrite(str(out / "masks" / (name_i + ".png")), classes)
                    meta[split].append(name_i)
                    idx += 1
    finally:
        sensors.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"frames": idx, "train": len(meta["train"]), "test": len(meta["test"]),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
