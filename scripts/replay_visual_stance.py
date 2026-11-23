"""Compare no-XYZ localization to velocity and truth using fixed windows."""
import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2

from go2_rescue.config import Config
from go2_rescue.geometry import rot2
from go2_rescue.localization import Localization
from go2_rescue.low_obstacle import LowObstacleDetector
from go2_rescue.recording import replay
from go2_rescue.visual_stance_odometry import VisualStanceOdometry

p = argparse.ArgumentParser()
p.add_argument("record", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--start", type=float, required=True)
p.add_argument("--end", type=float, required=True)
p.add_argument("--retain-keyframe", action="store_true")
p.add_argument("--image-stride", type=int, default=1)
p.add_argument("--ground-first", action="store_true")
p.add_argument("--along-prior", action="store_true")
a = p.parse_args()
if a.image_stride < 1:
    p.error("--image-stride must be >= 1")
cv2.setNumThreads(1)
cfg = Config.load(a.record/"config.json")
segmenter = LowObstacleDetector("models/navigation_floor.onnx", cfg.camera_matrix,
                               cfg.body_from_camera, cfg.base_height)
vo = VisualStanceOdometry(cfg, segmenter, retain_keyframe=a.retain_keyframe, ground_first=a.ground_first, along_prior=a.along_prior)
truth = {round(v["t"],3): np.asarray(v["base_world_xyz"]) for v in
         json.loads((a.record/"evaluation_truth.json").read_text())}
anchor = None
stats = Counter()
rows = []
dead = np.zeros(2)
last = None
image_count = 0
for frame in replay(a.record/"recording"):
    t = round(frame.timestamp,3)
    if t > a.end:
        break
    if t < a.start:
        continue
    if frame.image is not None:
        if image_count % a.image_stride:
            frame = replace(frame, image=None)
        image_count += 1
    xyz, yaw = vo.update(frame)
    if last is not None and 0 < frame.timestamp-last <= .5:
        dead += rot2(yaw) @ frame.velocity.array()[:2] * (frame.timestamp-last)
    last = frame.timestamp
    if t not in truth:
        continue
    if anchor is None:
        anchor = truth[t]
        alignment = rot2(Localization._imu_yaw(frame.quaternion))
        offset = xyz[2]
    expected = truth[t]-anchor
    expected[:2] = expected[:2] @ alignment
    ez = xyz[2]-offset-expected[2]
    rows.append([t,*xyz,*expected,np.linalg.norm(xyz[:2]-expected[:2]),
                 np.linalg.norm(dead-expected[:2]),abs(ez)])
    if frame.image is not None:
        stats["images"] += 1
        stats[vo.status.get("reason",vo.status.get("source","unknown"))] += 1
        stats["visual_used"] += vo.status["visual_used"]
errors=np.asarray(rows)[:,-3:]
summary={"window":[a.start,a.end],"n":len(errors),"camera":dict(stats),
         "along_prior":a.along_prior,
         "ground_first":a.ground_first,
         "image_stride": a.image_stride,
         "retain_keyframe": a.retain_keyframe,
         "p50":dict(zip(("visual_xy","velocity_xy","stance_z"),np.median(errors,axis=0).tolist())),
         "p90":dict(zip(("visual_xy","velocity_xy","stance_z"),np.quantile(errors,.9,axis=0).tolist())),
         "final_xy": errors[-1,:2].tolist()}
a.out.mkdir(parents=True,exist_ok=True)
(a.out/"metrics.json").write_text(json.dumps(summary,indent=2))
np.save(a.out/"trajectory.npy",rows)
print(json.dumps(summary,indent=2))
