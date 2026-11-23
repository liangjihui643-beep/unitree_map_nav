"""Evaluate fused relief and map consistency from archived observations."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from scipy.spatial.transform import Rotation

from go2_rescue.geometry import rot2
from go2_rescue.mapping import spatial_frame_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    truth = json.loads((root / "evaluation_truth.json").read_text())
    config = json.loads((root / "config.json").read_text())
    observations = json.loads((root / "terrain_heights.json").read_text())
    height_errors, heights, low, high = [], [], 0, 0
    sources = {}
    on_rock, other, missing = 0, 0, 0
    physical_rock, physical_other = 0, 0
    source = Path(__file__).resolve().parents[1] / "scene/mujoco/competition-course.mjb"
    model = mujoco.MjModel.from_binary_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body("base_link").id
    model.geom_group[model.body_rootid[model.geom_bodyid] == base] = 5
    needed_stamps = {round(row["t"],6) for row in observations}
    quaternions = {}
    recording = root/"recording"
    for row in map(json.loads, (recording/"frames.jsonl").open()):
        stamp = round(row["timestamp"],6)
        if stamp in needed_stamps:
            with np.load(recording/row["file"], allow_pickle=False) as packet:
                quaternions[stamp] = packet["quaternion"].copy()
    truth_by_stamp = {round(row["t"],6):row for row in truth}
    physical_rows = []
    camera = np.asarray(config["body_from_camera"])
    k = np.asarray(config["camera_matrix"])
    cells = {}
    measured_frames = []
    for row in observations:
        frame = truth_by_stamp.get(round(row["t"],6))
        if frame is None or abs(frame["t"] - row["t"]) > 1e-6:
            raise ValueError("Missing same-stamp evaluation pose")
        pts = np.asarray(row["points_level"])
        measured = np.asarray(row["height_m"])
        selected = (measured >= .035) & (measured <= .30)
        measured, pts = measured[selected], pts[selected]
        body = np.asarray(row["points_body"])[selected]
        rotation = Rotation.from_quat(np.roll(quaternions[round(row["t"],6)], -1)).as_matrix()
        offset = np.array([0.,0.,.10])
        laser_origin = np.asarray(frame["base_world_xyz"]) + rotation @ offset
        directions = (body-offset) @ rotation.T
        measured_ranges = np.linalg.norm(directions, axis=1)
        directions /= measured_ranges[:,None]
        ids = np.full(len(body),-1,dtype=np.int32)
        ranges = np.full(len(body),-1.)
        mujoco.mj_multiRay(model,data,laser_origin,directions.ravel(),np.array([1,1,1,1,1,0],dtype=np.uint8),
                          True,-1,ids,ranges,None,len(body),config["max_range"])
        # The source MJB contains static course geometry. Add the evaluated
        # local proxy sphere analytically when it is closer than that geometry.
        kind_ball = np.zeros(len(body),dtype=bool)
        if frame["football_world_xy"] is not None:
            delta = np.r_[frame["football_world_xy"],.11] - laser_origin
            projection = directions @ delta
            disc = projection**2 - (delta @ delta - .11**2)
            valid = (disc >= 0) & (projection > 0)
            ball_range = projection - np.sqrt(np.maximum(0,disc))
            kind_ball = valid & (ball_range > 0) & ((ranges < 0) | (ball_range < ranges))
            ranges = np.where(kind_ball, ball_range, ranges)
        for index,gid in enumerate(ids):
            matches_range = abs(ranges[index]-measured_ranges[index]) <= .02
            rock = bool(matches_range and not kind_ball[index] and gid>=0
                        and model.body(int(model.geom_bodyid[gid])).name.startswith("pile"))
            physical_rock += int(rock)
            physical_other += int(not rock)
            physical_rows.append({"t":row["t"],"source_geom":model.geom(int(gid)).name if gid>=0 else None,
                                  "sphere_closer":bool(kind_ball[index]),"rock":rock,
                                  "range_disagreement_m":float(abs(ranges[index]-measured_ranges[index]))})
        image = cv2.imread(str(root / "evaluation_masks" / f"{row['t']:.6f}.png"), 0)
        if image is None:
            missing += len(body)
        else:
            optical = (body-camera[:3, 3]) @ camera[:3, :3]
            pixels = optical @ k.T
            uv = np.rint(pixels[:, :2]/pixels[:, 2:]).astype(int)
            for u,v in uv:
                if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
                    if image[v,u] & 1:
                        on_rock += 1
                    else:
                        other += 1
                else:
                    missing += 1
        true_height = pts[:, 2] + frame["base_world_xyz"][2]
        height_errors.extend(abs(measured-true_height).tolist())
        heights.extend(measured.tolist())
        low += int((measured <= .30).sum())
        high += int((measured > .30).sum())
        sources[row["plane_source"]] = sources.get(row["plane_source"], 0) + len(pts)
        xy = pts[:, :2] @ rot2(row["pose"]["yaw"]).T + [row["pose"]["x"], row["pose"]["y"]]
        per_frame = {}
        for point, height in zip(xy, measured):
            key = tuple(np.floor(point/config["resolution"]).astype(int))
            per_frame.setdefault(key, []).append(float(height))
        measured_frames.append(set(per_frame))
        for key, values in per_frame.items():
            cell = cells.setdefault(key, {"hits": 0, "low": .65, "high": 0., "views": set()})
            cell["hits"] += 1
            cell["low"] = min(cell["low"], min(values))
            cell["high"] = max(cell["high"], max(values))
            cell["views"].add(tuple(np.floor(np.array([row["pose"]["x"], row["pose"]["y"]])/.25).astype(int)))
    support = spatial_frame_support(measured_frames)
    neighbouring_views = {}
    for key in cells:
        views = set()
        for di in (-1,0,1):
            for dj in (-1,0,1):
                adjacent_cell = cells.get((key[0]+di,key[1]+dj))
                if adjacent_cell:
                    views.update(adjacent_cell["views"])
        neighbouring_views[key] = len(views)
    error = np.asarray(height_errors)
    stats = {
        "physical_rock_samples": physical_rock, "physical_other_samples": physical_other,
        "physical_rock_precision": physical_rock/max(1,physical_rock+physical_other),
        "physical_sample_details": physical_rows,
        "on_visible_rock_samples": on_rock, "other_samples": other, "missing_pixel_truth": missing,
        "visible_rock_sample_precision": on_rock/max(1,on_rock+other),
        "observations": len(observations), "valid_height_samples": len(heights),
        "height_min_max_m": [min(heights), max(heights)] if heights else None,
        "height_error_p50_p90_max_m": np.quantile(error, [.5, .9, 1]).tolist() if len(error) else None,
        "source": sources, "low_relief_samples_0_035_to_0_30_m": low,
        "higher_relief_samples_0_30_to_0_65_m": high,
         "stable_cells_2_frames": sum(n >= 2 for n in support.values()),
        "nearby_frame_support_radius_m": config["resolution"] * 2**.5,
         "cross_view_cells_2_positions": sum(n >= 2 for n in neighbouring_views.values()),
        "evaluation_reference": "world z above the flat support strip; fixed joint/IMU contact assumption",
        "pass": bool(physical_rock/max(1,physical_rock+physical_other) >= .80 and len(error) >= 5 and np.quantile(error, .9) <= .08 and low >= 5
                     and any(n >= 2 for n in support.values())),
    }
    keys = np.asarray(list(cells), dtype=int).reshape(-1, 2)
    np.savez_compressed(root / "terrain_map.npz", cells=keys, resolution=config["resolution"],
                        origin=np.zeros(2), frame="map_start",
                        neighbouring_frame_hits=[support[key] for key in cells],
                        neighbouring_distinct_views=[neighbouring_views[key] for key in cells],
                        min_height_m=[c["low"] for c in cells.values()],
                        max_height_m=[c["high"] for c in cells.values()],
                        frame_hits=[c["hits"] for c in cells.values()],
                        distinct_views=[len(c["views"]) for c in cells.values()])
    (root / "terrain_acceptance.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
