#!/usr/bin/env python3
"""MuJoCo rubble dataset v3: three-class masks (ground/rubble/structure) + empty negatives.

Rubble and structure pixels come from visibility-group differencing: rubble
geoms in group 4, structure geoms in group 5, each toggled independently.
Robot geoms are never labelled (self-observation is not an obstacle).
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
STRUCT_GROUP = 5


def classify_geoms(model):
    rubble, struct = [], []
    ground = None
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.startswith("pile"):
            rubble.append(g)
        elif name.startswith(("ramp", "tunnel")):
            struct.append(g)
    # world-body geoms: everything except the ground plane counts as structure
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "ground" in gname.lower() or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground = g
        else:
            struct.append(g)
    return rubble, struct


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
    ap.add_argument("--out", default=str(ROOT / "data/mujoco_rubble3"))
    ap.add_argument("--scene", default=str(ROOT / "scene/mujoco/competition-course.mjb"))
    ap.add_argument("--distances", nargs="+", type=float, default=[1.0, 1.6, 2.4])
    ap.add_argument("--angles", nargs="+", type=float, default=[-0.4, 0.0, 0.4])
    ap.add_argument("--test-pile", default="pile_01")
    ap.add_argument("--empty", type=int, default=10)
    args = ap.parse_args()
    model = mujoco.MjModel.from_binary_path(args.scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    piles, _ = pile_centres(model, data)
    rubble_ids, struct_ids = classify_geoms(model)
    orig_r, orig_s = model.geom_group[rubble_ids].copy(), model.geom_group[struct_ids].copy()
    model.geom_group[rubble_ids] = RUBBLE_GROUP
    model.geom_group[struct_ids] = STRUCT_GROUP
    cfg = Config()
    sensors = SimSensors(model, data, cfg)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta = {"train": [], "test": [], "piles": {k: v.tolist() for k, v in piles.items()}}

    def render_labels():
        sensors.option.geomgroup[RUBBLE_GROUP] = 1
        sensors.option.geomgroup[STRUCT_GROUP] = 1
        full = sensors.image()
        sensors.option.geomgroup[RUBBLE_GROUP] = 0
        no_r = sensors.image()
        sensors.option.geomgroup[STRUCT_GROUP] = 0
        no_rs = sensors.image()
        d_r = np.abs(full.astype(np.int16) - no_r.astype(np.int16)).sum(2)
        d_s = np.abs(no_r.astype(np.int16) - no_rs.astype(np.int16)).sum(2)
        mask = np.zeros(d_r.shape, np.uint8)
        mask[cv2.morphologyEx((d_s > 12).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0] = 2
        mask[cv2.morphologyEx((d_r > 12).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0] = 1
        return full, mask

    try:
        for name, centre in piles.items():
            split = "test" if name == args.test_pile else "train"
            for dist in args.distances:
                for ang in args.angles:
                    sx, sy = centre[0] - dist * math.cos(ang), centre[1] - dist * math.sin(ang)
                    data.qpos[:7] = [sx, sy, .35, math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
                    mujoco.mj_forward(model, data)
                    rgb, mask = render_labels()
                    if (mask == 1).sum() < 150:
                        continue
                    name_i = "%s_d%.1f_a%+.2f" % (name, dist, ang)
                    cv2.imwrite(str(out / "images" / (name_i + ".png")), rgb)
                    cv2.imwrite(str(out / "masks" / (name_i + ".png")), mask)
                    meta[split].append(name_i)
        # empty negatives: robot away from every pile, facing empty space
        rng = np.random.default_rng(7)
        for k in range(args.empty):
            x, y = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
            ang = rng.uniform(-math.pi, math.pi)
            data.qpos[:7] = [x, y, .35, math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
            mujoco.mj_forward(model, data)
            rgb, mask = render_labels()
            if (mask > 0).sum() > 500:
                continue
            name_i = "empty_%02d" % k
            cv2.imwrite(str(out / "images" / (name_i + ".png")), rgb)
            cv2.imwrite(str(out / "masks" / (name_i + ".png")), mask)
            meta["train"].append(name_i)
    finally:
        model.geom_group[rubble_ids] = orig_r
        model.geom_group[struct_ids] = orig_s
        sensors.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"train": len(meta["train"]), "test": len(meta["test"]), "out": str(out)}))


if __name__ == "__main__":
    main()
