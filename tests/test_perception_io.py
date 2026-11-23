import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.mapping import GridMap
from go2_rescue.perception_io import (
    PublicSensorStream,
    perception_snapshot,
    public_range_evidence,
    publish_perception_snapshot,
)
from go2_rescue.types import Pose


def test_contract_samples_deduplicate_and_preserve_gravity():
    config = Config(lidar_mount=[.32, 0, .05], lidar_elevation_center_deg=0.,
                    camera_size=[320, 240], camera_matrix=[[200., 0., 160.],
                                                        [0., 200., 120.], [0., 0., 1.]])
    stream = PublicSensorStream(config)
    np.testing.assert_allclose(stream.camera_matrix_64,
                               [[40., 0., 31.6], [0., 160. / 3., 31.6333333333], [0., 0., 1.]])
    rotation = Rotation.from_euler("xyz", [.17, -.21, .9]).as_matrix()
    obs = np.zeros(189, np.float32)
    obs[3:6] = -rotation[2]
    obs[0] = .25
    obs[21] = .05
    obs[45:117] = 1.
    rgb = np.zeros((3, 64, 64), np.float32)
    rgb[0] = 1.
    for index in range(6):
        age = (index % 5) * .02
        step = stream.decode(index * .02, obs, rgb,
                             np.array([age, age, 1, .5, 0, 0, 1, 1], np.float32))
        assert (step.lidar_obs is not None) == (index in (0, 5))
        assert (step.image_bgr is not None) == (index in (0, 5))
        np.testing.assert_allclose(gravity_rotation(step.roll_pitch_wxyz)[2], rotation[2])
    assert step.image_bgr[0, 0].tolist() == [0, 0, 255]
    assert step.gyro_rad_s[0] == pytest.approx(1.)
    assert step.joint_velocity_rad_s[0] == pytest.approx(1.)
    assert step.raw_lidar_valid_fraction == .5

    grid = GridMap(config)
    evidence = public_range_evidence(grid)
    np.testing.assert_allclose(evidence.origin_body, [.32, 0, .05])
    evidence.update(grid, step.lidar_obs, step.roll_pitch_wxyz, Pose(), .05, 12.)
    assert not evidence.possible_hit.any()

    stream.reset()
    obs[45] = 0.
    with pytest.raises(ValueError, match="distance/mask"):
        stream.decode(0., obs, rgb, [0, 0, 1, .5, 0, 0, 1, 1])


def test_snapshot_is_replaceable_and_exports_distinct_map_layers(tmp_path):
    grid = GridMap(Config(map_size=4.))
    free, obstacle = grid.cell((.2, 0.)), grid.cell((1., 0.))
    grid.seen[free] = grid.seen[obstacle] = True
    grid.floor[free] = 0.
    grid.logodds[free], grid.logodds[obstacle] = -2., 2.
    path = tmp_path / "perception_snapshot.json"
    terrain = {"schema": "go2-terrain-layer/1", "cells": [{"ij": list(free)}]}
    detections = {"time_s": .9, "detections": [{"range_from_base_xy_m": 1.4}]}
    snapshot = perception_snapshot(grid, terrain, [[1., 2.]], 1., detections=detections)
    publish_perception_snapshot(path, snapshot)
    read = json.loads(path.read_text())
    assert read["frame"] == "map_start"
    assert read["map"]["occupied_ij"] == [list(obstacle)]
    assert read["map"]["ray_free_ij"] == [list(free)]
    assert read["map"]["traversable_ij"] == np.argwhere(grid.traversable()).tolist()
    assert read["confirmed_target_xy_m"] == [[1., 2.]]
    assert read["terrain"] == terrain
    assert read["target_observation"] == detections

    publish_perception_snapshot(path, perception_snapshot(grid, None, [], 2.))
    read = json.loads(path.read_text())
    assert read["time_s"] == 2.
    assert read["terrain"] is None
    assert read["confirmed_target_xy_m"] == []
    assert list(tmp_path.iterdir()) == [path]
