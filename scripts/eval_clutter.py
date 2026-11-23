#!/usr/bin/env python3
"""Evaluate low-obstacle ONNX models on the procedural clutter test seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.mujoco_sim import SimSensors
from mujoco_clutter_dataset import GROUP, build_scene

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/eval_clutter.json")
    args = ap.parse_args()
    meta = json.loads((ROOT / "data/mujoco_clutter/meta.json").read_text())
    seeds = [int(n.replace("seed", "")) for n in meta["test"]]
    cfg = Config()
    results = {}
    for onnx in args.onnx:
        det = LowObstacleDetector(onnx, cfg.camera_matrix, cfg.body_from_camera, cfg.base_height)
        rows = []
        for seed in seeds:
            model, _ = build_scene(seed, n_obs=int(8 + seed % 9))
            data = mujoco.MjData(model)
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            ids = [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("clutter_")]
            centres = np.array([[model.geom_pos[g][0], model.geom_pos[g][1]] for g in ids])
            orig = model.geom_group[ids].copy()
            model.geom_group[ids] = GROUP
            sensors = SimSensors(model, data, cfg)
            try:
                sensors.option.geomgroup[GROUP] = 1
                rgb = sensors.image()
                sensors.option.geomgroup[GROUP] = 0
                plain = sensors.image()
            finally:
                model.geom_group[ids] = orig
                sensors.close()
            diff = np.abs(rgb.astype(np.int16) - plain.astype(np.int16)).sum(2)
            mask = (diff > 12).astype(np.uint8)
            prob = det.segment(rgb)
            pred = (prob.argmax(0) == 1).astype(np.uint8)
            tp = int((pred & mask).sum())
            rows.append({
                "seed": seed,
                "pixel_recall": round(tp / max(1, int(mask.sum())), 3),
                "pixel_iou": round(tp / max(1, int(((pred | mask) > 0).sum())), 3),
                "detections": len(det.detect(rgb)),
            })
        results[Path(onnx).stem] = rows
    Path(args.out).write_text(json.dumps(results, indent=1))
    for tag, rows in results.items():
        recalls = [r["pixel_recall"] for r in rows]
        ious = [r["pixel_iou"] for r in rows]
        print(tag, "pixel_recall_mean", round(float(np.mean(recalls)), 3),
              "iou_mean", round(float(np.mean(ious)), 3),
              "det_mean", round(float(np.mean([r["detections"] for r in rows])), 1))
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
