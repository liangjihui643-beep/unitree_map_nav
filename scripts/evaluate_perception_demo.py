"""Score the saved moving demonstration without sending truth to perception."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from go2_rescue.config import Config
from go2_rescue.geometry import rot2
from go2_rescue.localization import Localization
from go2_rescue.mapping import spatial_frame_support
from go2_rescue.recording import replay


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def error_stats(errors):
    if not len(errors):
        return None
    return {k: float(v) for k, v in zip(("p50_m", "p90_m", "max_m"),
                                      np.quantile(errors, [.5, .9, 1]))}


def rubble_map_precision(run, anchor, rotation, field='map_xy', min_frames=3, mapped_xy=None):
    """Compare stable map cells to projected individual source rock meshes."""
    with np.load(run / "map.npz", allow_pickle=False) as cells:
        resolution = float(cells["resolution"])
        origin = cells["origin"]
        mask = np.zeros(cells["seen"].shape, dtype=np.uint8)
    counts = Counter()
    measured_frames = []
    for frame in rows(run / "rubble_observations.jsonl"):
        xy = [point for region in frame["regions"] for point in region.get(field, [])]
        measured = {tuple(cell) for cell in np.floor((np.asarray(xy)-origin)/resolution).astype(int)} if xy else set()
        counts.update(measured)
        measured_frames.append(measured)
    if field == "lidar_map_xy":
        counts = spatial_frame_support(measured_frames)
    predictions = [cell for cell, n in counts.items() if n >= min_frames and all(
        0 <= index < size for index, size in zip(cell, mask.shape))]
    if mapped_xy is not None:
        predictions = [tuple(cell) for cell in np.floor((np.asarray(mapped_xy).reshape(-1,2)-origin)/resolution).astype(int)
                       if ((cell >= 0) & (cell < np.asarray(mask.shape))).all()]
    file = Path(__file__).resolve().parents[1] / "scene/mujoco/competition-course.mjb"
    model = mujoco.MjModel.from_binary_path(str(file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for geom in range(model.ngeom):
        if not model.body(int(model.geom_bodyid[geom])).name.startswith("pile"):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        offset, length = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[offset:offset + length]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        xy = (world[:, :2]-anchor) @ rotation
        pixels = np.floor((xy-origin)/resolution).astype(np.int32)
        hull = cv2.convexHull(pixels[:, ::-1].copy())
        cv2.fillConvexPoly(mask, hull, 1)
    distances = distance_transform_edt(mask == 0) * resolution
    deviation = distances[tuple(np.asarray(predictions).T)] if predictions else np.empty(0)
    return {"evaluated_cells": len(predictions), "in_mesh_footprint_cells": int((deviation == 0).sum()),
            "within_0_20m_cells": int((deviation <= .20+1e-8).sum()),
            "distance_to_nearest_rock": error_stats(deviation),
            "scope": "stable cells from logged observations, source rock mesh hulls, no visibility denominator"}


def evaluate(run, localization_ab=True):
    summary = json.loads((run / "summary.json").read_text())
    config = Config.load(run / "config.json")
    truth = json.loads((run / "evaluation_truth.json").read_text())
    lidar = rows(run / "lidar_diagnostics.jsonl")
    estimated = np.array([[r["pose"]["x"], r["pose"]["y"]] for r in lidar])
    if len(truth) != len(lidar) or not np.allclose([r["t"] for r in truth], [r["t"] for r in lidar]):
        raise ValueError("Truth and estimate timestamps differ")
    world = np.array([r["base_world_xy"] for r in truth])
    anchor = np.array([.8,.8]) if summary.get('planned_public_map_survey',False) else world[0]
    rotation = rot2(0. if summary.get("planned_public_map_survey", False) else summary["initial_yaw"])
    relative = (world - anchor) @ rotation
    result = {"run": run.name, "status": summary["status"], "seconds": summary["simulation_seconds"],
              "wall_seconds": summary["wall_seconds"], "scene": summary["scene"],
              "path_m": float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum()),
              "displacement_m": float(np.linalg.norm(world[-1]-world[0])),
              "fell": summary["fell"],
              "obstacle_contact_episodes": summary["evaluation"]["obstacle_contact_episodes"],
              "live_position_error": error_stats(np.linalg.norm(estimated-relative, axis=1)),
              "pose_quality_pass_fraction": float(np.mean([r["pose"]["quality"] >= config.min_pose_quality for r in lidar])),
              "perception": summary["perception_demo"]}
    result["rubble_map_agreement"] = rubble_map_precision(run, anchor, rotation)
    result["low_height_map_agreement"] = rubble_map_precision(run, anchor, rotation, field="lidar_map_xy", min_frames=2)
    if localization_ab and (run / "recording").is_dir():
        frames = list(replay(run / "recording"))
        if not np.allclose([f.timestamp for f in frames], [r["t"] for r in truth]):
            raise ValueError("Replay and truth timestamps differ")
        result["same_scan_localization_ab"] = {}
        for label, metric, imu, submap in [
            ("original_point_to_point", "point_to_point", False, False),
            ("point_to_line_imu", "point_to_line", True, False),
            ("point_to_line_imu_submap", "point_to_line", True, True),
        ]:
            config.localization_metric = metric
            config.imu_heading_reference = imu
            loc = Localization(config)
            loc.use_local_submap = submap
            poses = np.array([loc.update(frame)[0].xy() for frame in frames])
            result["same_scan_localization_ab"][label] = error_stats(np.linalg.norm(poses-relative, axis=1))
    targets = np.asarray(summary["perception_demo"]["confirmed_targets"], dtype=float).reshape(-1, 2)
    if truth[-1]["football_world_xy"] is not None:
        football = np.array([r["football_world_xy"] for r in truth])
        result["football_motion_m"] = float(np.linalg.norm(football-football[0], axis=1).max())
        # Compare every published marker to the one scene football. Do not
        # drop unmatched markers or claim per-frame detection recall.
        ball_xy = (football[-1] - anchor) @ rotation
        result["football_final_map_truth_xy"] = ball_xy.tolist()
        result["football_confirmed_marker_errors_m"] = np.linalg.norm(targets-ball_xy, axis=1).tolist()
        result["football_extra_markers"] = max(0, len(targets)-1)
    evidence = summary["perception_demo"].get("showcase")
    if evidence:
        distance = result["rubble_map_agreement"]
        result["map_known_area_m2"] = summary["perception_demo"]["seen_cells"] * config.resolution**2
        result["map_unknown_area_m2"] = float((np.ceil(config.map_size/config.resolution)**2
            - summary["perception_demo"]["seen_cells"]) * config.resolution**2)
        result["wall_false_positive_rate"] = (evidence["rock_on_wall_pixels"]
            / max(1, evidence["wall_visible_pixels"]))
        range_errors = []
        for confirmation in evidence["ball_confirmations"]:
            frame = min(truth, key=lambda row: abs(row["t"]-confirmation["t"]))
            if abs(frame["t"]-confirmation["t"]) > 1e-6 or frame["football_world_xy"] is None:
                continue
            true_range = np.linalg.norm(np.asarray(frame["base_world_xy"]) - frame["football_world_xy"])
            range_errors.append(float(abs(confirmation["range_m"]-true_range)))
        result["ball_confirmed_range_errors_m"] = range_errors
        target_expected = truth[-1]["football_world_xy"] is not None
        target_proof = (len(evidence["ball_confirmations"]) == 1
                        and evidence["ball_confirmations"][0]["true_ball_mask_iou"] >= .45
                        and evidence["ball_visible_frames"] >= 5
                        and evidence["ball_visible_recall"] >= .70
                        and len(result.get("football_confirmed_marker_errors_m", [])) == 1
                        and max(result["football_confirmed_marker_errors_m"]) <= .35)
        result["acceptance"] = {
            "physical_route": result["path_m"] >= 3 and not result["fell"] and result["obstacle_contact_episodes"] == 0,
            "lidar_map": (summary["perception_demo"]["lidar_contract_violations"] == 0
                          and result["live_position_error"]["p90_m"] <= .30
                          and summary["perception_demo"]["seen_cells"] > 0),
            "rubble_image": evidence["rock_precision"] >= .80 and evidence["rock_recall"] >= .70,
            "rubble_map_visible": (distance["evaluated_cells"] > 0
                                  and distance["distance_to_nearest_rock"]["p90_m"] <= .30),
            "ball_image_range_map": target_proof if target_expected else None,
            "rubble_height_confirmed": summary["perception_demo"]["rough_lidar_cells"] > 0,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--no-localization-ab",action="store_true",help="Score the actual run without repeating three offline localizers")
    args = parser.parse_args()
    result = evaluate(args.run,localization_ab=not args.no_localization_ab)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.run / "perception_evaluation.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
