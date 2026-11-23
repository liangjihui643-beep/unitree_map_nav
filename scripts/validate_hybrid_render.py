"""Rendered target-height sweep; all modes see the same classifier candidates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from go2_rescue.ball_classifier import BallProposalClassifier
from go2_rescue.hybrid_ball import HybridBallRange, usable_ball
from go2_rescue.mujoco_scene import task_scene
from go2_rescue.mujoco_sim import SimSensors, simulation_config
from go2_rescue.perception import Perception
from go2_rescue.tracking import Tracker
from go2_rescue.types import Frame, Pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-unconfirmed-flat", action="store_true")
    a = parser.parse_args()
    root = Path.cwd()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    model = task_scene(root, out, (.8, .8), "near-start", save_model=False)
    data = mujoco.MjData(model)
    config = simulation_config(task_fixture=True, task_layout="near-start")
    config.target_quota = 0
    vision = Perception(config)
    classifier = BallProposalClassifier(root/"models/ball_proposal.onnx")
    sensor = SimSensors(model, data, config)
    joint = int(model.jnt_qposadr[model.joint("local_football_free").id])
    rows = {}
    try:
        for rise in (0., .10, .20, .31):
            hybrid = HybridBallRange(vision, allow_unconfirmed_flat=a.allow_unconfirmed_flat)
            trackers = {mode: Tracker(config, static_targets=True) for mode in ("legacy", "hybrid")}
            seq = {mode: [] for mode in trackers}
            true_xy = np.array([2., .8])
            all_proposals = 0
            for i, dy in enumerate(np.linspace(-.34,.34,17)):
                data.qpos[:3] = [.8, .8+dy, config.base_height]
                data.qpos[3:7] = [1, 0, 0, 0]
                data.qpos[joint:joint+7] = [*true_xy, .11+rise, 1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                image = sensor.image()
                candidates = classifier.filter(image, vision.detect_balls(image, small_targets=True))
                all_proposals += len(candidates)
                pose = Pose(.8, .8+dy, quality=1., sigma=.04)
                frame = Frame(i*.2,np.empty((0,3)))
                old = [vision.locate_ball(replace(d),frame,pose) for d in candidates]
                hybrid.update(candidates,image,frame,pose,config.base_height,
                              np.array([0.,0.,-config.base_height]),.85)
                for mode, observations in (("legacy",old),("hybrid",candidates)):
                    for event in trackers[mode].update([d for d in observations if usable_ball(d,.85)],
                                                        pose,frame.timestamp):
                        track = next(t for t in trackers[mode].tracks if t.id==event.payload["id"])
                        xy = event.payload["position"]
                        seq[mode].append({"t": frame.timestamp,"xy":xy,
                                          "error_m":float(np.linalg.norm(xy-true_xy)),
                                          "source":track.observation.range_source})
            rows[str(rise)] = {"classified_proposals":all_proposals, **seq}
    finally:
        sensor.close()
    (out/"summary.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
