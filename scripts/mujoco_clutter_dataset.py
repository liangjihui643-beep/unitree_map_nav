#!/usr/bin/env python3
"""Procedural low-obstacle clutter dataset (wood boxes, logs, crates) with self-supervised masks.

Each frame: Go2 robot standing in front of a random pile of boxes/cylinders/spheres.
Labels come from rendering with the clutter geoms hidden versus visible.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.mujoco_sim import SimSensors

ROOT = Path(__file__).resolve().parents[1]
GROUP = 4


def build_scene(seed: int, n_obs: int):
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(str(ROOT / "robot/go2/physics/go2.xml"))
    spec.worldbody.add_light(pos=[1.2, 0.0, 2.4], dir=[0.1, 0.0, -1.0],
                             diffuse=[1.0, 1.0, 1.0])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6.0, 6.0, 0.1],
                            pos=[0.0, 0.0, 0.0], rgba=[0.33, 0.32, 0.30, 1.0])
    palette = [
        [0.45, 0.30, 0.15, 1.0],  # wood brown
        [0.30, 0.22, 0.12, 1.0],  # dark wood
        [0.55, 0.42, 0.25, 1.0],  # light wood
        [0.42, 0.42, 0.42, 1.0],  # concrete
        [0.50, 0.48, 0.45, 1.0],  # stone
    ]
    names = []
    for i in range(n_obs):
        x = rng.uniform(0.9, 2.6)
        y = rng.uniform(-0.95, 0.95)
        h = rng.uniform(0.05, 0.26)
        kind = i % 3
        rgba = palette[int(rng.integers(0, len(palette)))]
        name = "clutter_%d" % i
        if kind == 0:
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[rng.uniform(0.10, 0.28), rng.uniform(0.06, 0.12), h / 2],
                                    pos=[x, y, h / 2], rgba=rgba)
        elif kind == 1:
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=[rng.uniform(0.05, 0.09), h / 2],
                                    pos=[x, y, h / 2], rgba=rgba)
        else:
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=[h / 2], pos=[x, y, h / 2], rgba=rgba)
        names.append(name)
    return spec.compile(), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/mujoco_clutter"))
    ap.add_argument("--train-seeds", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--test-seeds", nargs="+", type=int, default=[91, 92, 93])
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta = {"train": [], "test": []}
    cfg = Config()
    for split, seeds in (("train", args.train_seeds), ("test", args.test_seeds)):
        for seed in seeds:
            model, names = build_scene(seed, n_obs=int(8 + seed % 9))
            data = mujoco.MjData(model)
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            ids = [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("clutter_")]
            orig = model.geom_group[ids].copy()
            model.geom_group[ids] = GROUP
            sensors = SimSensors(model, data, cfg)
            try:
                # vary brightness a little per seed
                for light in range(model.nlight):
                    k = 0.7 + 0.6 * ((seed % 7) / 6.0)
                    model.light_diffuse[light][:3] = np.clip(np.ones(3) * k, 0, 1)
                sensors.option.geomgroup[GROUP] = 1
                rgb = sensors.image()
                sensors.option.geomgroup[GROUP] = 0
                plain = sensors.image()
                diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
                mask = (diff > 12).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                if mask.sum() < 150:
                    continue
                name = "seed%03d" % seed
                cv2.imwrite(str(out / "images" / (name + ".png")), rgb)
                cv2.imwrite(str(out / "masks" / (name + ".png")), mask)
                meta[split].append(name)
            finally:
                model.geom_group[ids] = orig
                sensors.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"train": len(meta["train"]), "test": len(meta["test"]), "out": str(out)}))


if __name__ == "__main__":
    main()
