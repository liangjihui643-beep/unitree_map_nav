"""Replay the recorded rubble cut from a shared pre-failure pose."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.localization import Localization
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.types import Frame, Pose, Velocity


class SingleScan(Localization):
    use_sparse_geometry_fallback = False

    def accumulated_public_fit(self, *args):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--start", type=float, default=270.)
    parser.add_argument("--stop", type=float, default=309.)
    parser.add_argument("--after", type=float, default=278.6)
    parser.add_argument("--compare-range", type=float)
    parser.add_argument("--compare-sparse", action="store_true")
    args = parser.parse_args()
    run = args.run
    config = Config.load(run / "config.json")
    config.experimental_scan_window = True
    config.navigation_prior = str((run / "navigation_prior.npz").resolve())
    trace = {round(row["timestamp"], 5): row for row in map(json.loads, (run / "trajectory.jsonl").open())}
    estimators = [SingleScan(config), Localization(config)]
    estimators[1].use_sparse_geometry_fallback = False
    if args.compare_range is not None:
        estimators[1] = SingleScan(config)
        estimators[1].public_reference_radius = args.compare_range
    if args.compare_sparse:
        estimators[1] = SingleScan(config)
        estimators[1].use_sparse_geometry_fallback = True
    scores = [[], []]
    sources = [Counter(), Counter()]
    surface = StanceSurface()
    for row in map(json.loads, (run / "recording/frames.jsonl").open()):
        t = row["timestamp"]
        if not args.start-2 <= t <= args.stop:
            continue
        with np.load(run / "recording" / row["file"]) as scan:
            frame = Frame(t, scan["points"], scan["quaternion"], Velocity(**row["velocity"]),
                          support_points=scan["support_points"] if "support_points" in scan else None)
        plane = surface.update(frame)
        trace_row = trace.get(round(t, 5))
        if t < args.start or trace_row is None:
            continue
        truth = np.asarray(trace_row["diagnostics_only_world_position"][:2]) - .8
        for loc, values, counts in zip(estimators, scores, sources):
            loc.support_plane = plane
            if loc.last_timestamp is None:
                loc.pose = Pose(**trace_row["pose"])
                if loc.pose.quality < config.min_pose_quality:
                    raise ValueError("Choose a valid pre-failure initialization")
                loc.heading_origin = loc._imu_yaw(frame.quaternion)-loc.pose.yaw
                loc.sensor_heading = loc.pose.yaw
                loc.last_timestamp = t
                loc._keyframe(frame.points @ gravity_rotation(frame.quaternion).T)
                continue
            pose, _ = loc.update(frame)
            if pose.quality >= config.min_pose_quality:
                counts[loc.last_registration.get("source", "unknown")] += 1
            values.append((t, pose.quality >= config.min_pose_quality, float(np.linalg.norm(pose.xy()-truth))))
    second = "sparse_observable" if args.compare_sparse else (
        "window" if args.compare_range is None else f"single_range_{args.compare_range:g}")
    for name, records, counts in zip(("single", second), scores, sources):
        for label, subset in (("all", records), (f"after_{args.after:g}", [r for r in records if r[0] >= args.after])):
            errors = [err for _, valid, err in subset if valid]
            print(name, label, "frames", len(subset), "valid", len(errors),
                  "error_p90/max_m", np.quantile(errors, .90) if errors else None, max(errors, default=None))
        print(name, dict(counts))


if __name__ == "__main__":
    main()
