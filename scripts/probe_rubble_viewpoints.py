"""Offline geometry probe, never feeds simulated poses to live navigation."""
import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.known_map_navigation import KnownMapNavigator
from go2_rescue.geometry import transform_xy
from go2_rescue.localization import Localization, line_normals, register_lines
from go2_rescue.mujoco_sim import SimSensors
from go2_rescue.types import Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--east-corridor", action="store_true")
    parser.add_argument("--inner-line", action="store_true")
    parser.add_argument("--explain-returns", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = Config.load(args.run / "config.json")
    config.navigation_prior = str((args.run / "navigation_prior.npz").resolve())
    loc = Localization(config)
    nav = KnownMapNavigator(config)
    nav.grid.traversable()
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    sensor = SimSensors(model, data, config, 0)
    if args.east_corridor:
        pose = Pose(10.75, 4.9, np.pi/2)
        columns = (10.36,10.42,10.48,10.54) if args.inner_line else (10.66,10.73,10.80,10.87,10.94,11.01)
        offsets = [(x-10.75, y-4.9) for y in np.arange(4.35, 6.9, .25) for x in columns]
        angles = (0.,)
    else:
        trace = [json.loads(line) for line in (args.run / "trajectory.jsonl").open()]
        failure = next(row for row in trace if row["timestamp"] > 279 and row["reason"] == "localization_unreliable")
        pose = Pose(**failure["pose"])
        offsets = ((0., 0.), (-.25, .15), (0., .25), (-.45, .2))
        angles = (0., -.25, .25, -.5, .5, 1., -1.)
    loc.support_plane = np.array([0., 0., -config.base_height])
    half = [nav.grid.c.footprint_half_length+nav.grid.c.safety_margin,
            nav.grid.c.footprint_half_width+nav.grid.c.safety_margin]
    corners = np.asarray(np.meshgrid(np.linspace(-half[0], half[0], 11),
                                    np.linspace(-half[1], half[1], 9))).reshape(2, -1).T
    for dx, dy in offsets:
        xy = pose.xy() + [dx, dy]
        target = loc.public_reference[np.linalg.norm(loc.public_reference-xy, axis=1)<6.]
        for turn in angles:
            yaw = pose.yaw+turn
            if not nav.grid.footprints_safe(xy[None], np.array([yaw]), corners).all():
                continue
            data.qpos[:3] = [*(xy+.8), config.base_height]
            data.qpos[3:7] = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)
            mujoco.mj_forward(model, data)
            points = sensor.lidar()
            ranges = np.linalg.norm(points, axis=1)
            guess = Pose(*xy, yaw)
            scan = loc.structural_scan(points[(ranges > .25) & (ranges < config.max_range)], guess)
            candidate, overlap, rmse = register_lines(scan, target, guess, yaw_hint=yaw, motion_prior_weight=16.)
            _, sparse_overlap, sparse_rmse = register_lines(
                scan, target, guess, yaw_hint=yaw, motion_prior_weight=16., sparse_geometry=True)
            tree, normals, is_line = line_normals(target)
            mapped = transform_xy(scan, guess)
            distance, index = tree.query(mapped)
            lateral = (distance < .14) & is_line[index] & (abs(normals[index, 0]) > .9)
            error = np.linalg.norm(candidate.xy()-xy)
            print(f"map=({xy[0]:.2f},{xy[1]:.2f}) turn={turn:+.2f} "
                  f"points={len(scan)} quality={overlap*np.exp(-rmse/.2):.3f} "
                  f"sparse={sparse_overlap*np.exp(-sparse_rmse/.2):.3f} lateral={lateral.sum()} error_m={error:.3f}")
            if args.explain_returns and 5.05 < xy[1] < 5.95 and 10.35 < xy[0] < 10.45:
                mount = np.array([0., 0., .10])
                rotation = Rotation.from_euler("z", yaw).as_matrix()
                lidar_origin = data.qpos[:3]+mount
                previous = model.geom_group[sensor.robot_geoms].copy()
                model.geom_group[sensor.robot_geoms] = 5
                type_counts = Counter()
                try:
                    groups = np.array([1,1,1,1,1,0], dtype=np.uint8)
                    hit_id = np.array([-1], dtype=np.int32)
                    for point, d in zip(scan, distance):
                        ray = rotation@(point-mount)
                        mujoco.mj_ray(model,data,lidar_origin,ray/np.linalg.norm(ray),groups,True,-1,hit_id)
                        if hit_id[0] >= 0:
                            geom = model.geom(int(hit_id[0]))
                            type_counts[(geom.name or model.body(int(geom.bodyid[0])).name, d < .14)] += 1
                finally:
                    model.geom_group[sensor.robot_geoms] = previous
                print("  return objects, near map line:", type_counts)


if __name__ == "__main__":
    main()
