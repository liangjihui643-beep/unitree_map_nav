"""Replay confirmation from actual logged proposals, poses and raw range input."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import rot2
from go2_rescue.perception import Perception
from go2_rescue.perception_demo import confirmable_ball
from go2_rescue.tracking import Tracker
from go2_rescue.types import BallObservation, Frame, Pose, Velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    config = Config.load(run/"config.json")
    detector = Perception(config)
    trace = {round(row["t"],6):row for row in map(json.loads,(run/"lidar_diagnostics.jsonl").open())}
    frames = {round(row["timestamp"],6):row for row in map(json.loads,(run/"recording/frames.jsonl").open())}
    truth = {round(row["t"],6):row for row in json.loads((run/"evaluation_truth.json").read_text())}
    summary = json.loads((run/"summary.json").read_text())
    fixture = summary["scene"].endswith("plus_task_objects") or summary["scene"].endswith("plus_local_task_objects")
    updated = replace(config,target_quota=1) if fixture else config
    trackers = Tracker(config, static_targets=True), Tracker(updated, static_targets=True)
    if summary.get("planned_public_map_survey",False):
        anchor,rotation = np.array([.8,.8]),np.eye(2)
    else:
        anchor = np.asarray(next(iter(truth.values()))["base_world_xy"])
        rotation = rot2(summary["initial_yaw"])
    confirmations = [[],[]]
    for row in map(json.loads,(run/"target_tracks.jsonl").open()):
        t=row["t"]
        key=round(t,6)
        pose=Pose(**trace[key]["pose"])
        if pose.quality < config.min_pose_quality:
            continue
        record=frames[key]
        detections=[]
        with np.load(run/"recording"/record["file"],allow_pickle=False) as packet:
            frame=Frame(t,packet["points"],packet["quaternion"],Velocity(**record["velocity"]))
            for item in row["detections"]:
                if item["map_xy"] is None or item["confidence"] < .85:
                    continue
                detection=BallObservation(tuple(item["center_px"]),item["radius_px"],item["confidence"],0.)
                detections.append(detector.locate_ball(detection,frame,pose))
        for i,tracker in enumerate(trackers):
            eligible = detections if i==0 else [d for d in detections if confirmable_ball(d,.85)]
            for event in tracker.update(eligible,pose,t):
                track=next(x for x in tracker.tracks if x.id==event.payload["id"])
                detection=track.observation
                obs_truth=truth[key]
                ball=np.asarray(obs_truth["football_world_xy"])
                range_m=float(np.linalg.norm(detection.position-pose.xy()))
                range_truth=float(np.linalg.norm(ball-obs_truth["base_world_xy"]))
                confirmations[i].append({
                    "t":t,"map_xy":event.payload["position"],"range_m":range_m,
                    "range_error_m":abs(range_m-range_truth),
                    "marker_error_m":float(np.linalg.norm(anchor+rotation@track.position-ball)),
                    "source":detection.range_source,"sigma_m":detection.sigma,
                })
    result=dict(zip(("baseline","uncertainty_gated"),confirmations))
    (run/"target_confirmation_replay.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
