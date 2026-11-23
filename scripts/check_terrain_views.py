"""Read-only view challenges for walls, slope and rubble."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector, rubble_region_observable
from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.perception import Perception
from go2_rescue.perception_demo import write_png
from go2_rescue.types import Frame, Pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    cfg = Config()
    detector = LowObstacleDetector(root / "models/seg_low_obstacle_perception.onnx", cfg.camera_matrix,
                                   cfg.body_from_camera, cfg.base_height)
    vision = Perception(cfg)
    rocks = [g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.startswith("pile")]
    walls = [g for g in range(model.ngeom) if any(
        token in model.geom(g).name.lower() for token in ("wall", "partition"))]
    views = [
        ("start_forward", .8, .8, 0.),
        ("start_wall", .8, .8, np.pi),
        ("ramp_front", 1.1, 4., np.pi/2),
        ("ramp_side", 1.7, 4.3, np.pi),
        ("rubble_side", 6.2, 2.56, -np.pi/2),
        ("rubble_opposite", 7.3, 2.56, np.pi/2),
        ("far_wall", 9., 2.56, 0.),
        ("corridor_back", 8.4, 2.56, np.pi),
    ]
    sensors = SimSensors(model, data, cfg)
    reports, panels = [], []
    try:
        for name, x, y, yaw in views:
            data.qpos[:7] = [x, y, .33, np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            mujoco.mj_forward(model, data)
            image = sensors.image()
            labels = sensors.segmentation()
            geom = np.where(labels[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, labels[:, :, 0], -1)
            truth = np.isin(geom, rocks)
            wall = np.isin(geom, walls)
            probability = detector.segment(image)
            pred = (probability[1] >= .70) & (probability.argmax(0) == 1)
            structure = ((probability.argmax(0) == 2) & (probability[2] >= .45)).astype(np.uint8)
            pred[cv2.dilate(structure, np.ones((3, 3), np.uint8)) != 0] = False
            count, regions, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8))
            pred = np.isin(regions, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 25
                                   and rubble_region_observable(*stats[i, :4], image.shape[1])])
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            report = {"name": name, "spawn": [x,y,yaw], "true_rock_pixels": int(truth.sum()),
                      "predicted_rock_pixels": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
                      "wall_pixels": int(wall.sum()), "wall_fp": int((pred & wall).sum()),
                      "precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn)}
            reports.append(report)
            write_png(args.output / (name+"-raw.png"), image)
            write_png(args.output / (name+"-truth.png"), truth.astype(np.uint8) + 2*wall.astype(np.uint8))
            panel = image.copy()
            panel[pred] = (.5*panel[pred] + .5*np.array([20,180,255])).astype(np.uint8)
            cv2.putText(panel, name, (5, 18), 0, .45, (255,255,255), 1)
            panels.append(panel)
            write_png(args.output / (name+".png"), panel)
    finally:
        sensors.close()
    write_png(args.output / "challenge_views.png", np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
    report = {"views": reports, "scope": "independent static RGB views; reset for read-only render only"}
    (args.output / "challenge.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
