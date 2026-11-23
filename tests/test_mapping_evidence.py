import numpy as np

from go2_rescue.config import Config
from go2_rescue.mapping import GridMap
from go2_rescue.types import Pose


def prepared_grid(**kwargs):
    cfg = Config(map_size=10., resolution=.1, **kwargs)
    grid = GridMap(cfg)
    for x in np.arange(.5, 3.05, .1):
        cell = grid.cell((x, 0.))
        grid.floor[cell] = 0.
        grid.floor_source[cell] = 3
        grid.seen[cell] = True
    return grid


def logodds_at(grid, x, y=0.):
    return float(grid.logodds[grid.cell((x, y))])


def test_high_prefix_has_no_near_ground_evidence():
    grid = prepared_grid(map_evidence=True, map_ray_height_clear=True)
    # Simulated mount is 0.10 m above the base; endpoint body z=.10 -> world
    # .45 m, above the near band [.02, .10], so no near evidence.
    grid.update(np.array([[3., 0., .10]]), Pose())
    mid = grid.cell((1.5, 0.))
    assert grid.evidence_prefix[mid] > 0
    assert grid.evidence_near_clear[mid] == 0
    assert grid.evidence_near_partial[mid] == 0
    assert logodds_at(grid, 1.5) < 0.


def test_near_ground_ray_produces_band_evidence():
    grid = prepared_grid(map_evidence=True, map_ray_height_clear=True)
    grid.update(np.array([[3., 0., -.35]]), Pose())
    near = grid.cell((2.5, 0.))
    assert grid.evidence_near_partial[near] > 0
    assert logodds_at(grid, 2.5) < 0.


def test_unknown_floor_endpoint_is_stored_pending():
    grid = GridMap(Config(map_size=10., resolution=.1, map_evidence=True, map_ray_height_clear=True))
    grid.update(np.array([[3., 0., .10]]), Pose())
    mid = grid.cell((1.5, 0.))
    assert grid.evidence_prefix[mid] == 0
    assert len(grid.pending_endpoints) == 1
    assert not np.isfinite(grid.floor[grid.cell((3., 0.))])


def test_endpoint_counted_once_per_frame():
    grid = prepared_grid(map_evidence=True)
    point = np.array([[1., .2, .0]])
    cell = grid.cell((1., .2))
    grid.floor[cell] = 0.
    grid.floor_source[cell] = 3
    grid.update(np.vstack([point, point, point]), Pose())
    assert grid.evidence_endpoint[cell] == 1
    assert grid.evidence_frame_support[cell] == 1


def test_evidence_flags_do_not_change_occupancy():
    plain = prepared_grid()
    plain.update(np.array([[3., 0., .10]]), Pose())
    instrumented = prepared_grid(map_evidence=True, map_ray_height_clear=True)
    instrumented.update(np.array([[3., 0., .10]]), Pose())
    assert np.array_equal(plain.logodds, instrumented.logodds)
    assert np.array_equal(plain.seen, instrumented.seen)


def test_obstacle_frame_uses_final_obstacle_decision():
    grid = prepared_grid(map_evidence=True)
    # Body z=.02 -> world .37 m, inside the obstacle band [.10, .55].
    grid.update(np.array([[1., 0., .02]]), Pose())
    cell = grid.cell((1., 0.))
    assert grid.evidence_obstacle_endpoint[cell] == 1
    assert grid.evidence_structure_endpoint[cell] == 0


def test_quaternion_tilts_mount_point():
    from scipy.spatial.transform import Rotation
    level = Rotation.from_euler("x", 0.).as_quat()
    tilted = Rotation.from_euler("x", .4).as_quat()[..., [3, 0, 1, 2]]
    grid = prepared_grid(map_evidence=True, map_ray_height_clear=True)
    grid.update(np.array([[3., 0., .0]]), Pose(), tilted)
    assert grid._frame_seq == 1
    # The tilted mount moves the ray start; evidence must still be recorded.
    assert int(grid.evidence_prefix.sum()) > 0


def test_single_hit_is_not_a_stable_wall():
    grid = prepared_grid(map_wall_truncate=True)
    wall = grid.cell((2.0, 0.))
    grid.logodds[wall] = 1.25
    grid.update(np.array([[3., 0., -.35]]), Pose())
    behind = grid.cell((2.5, 0.))
    assert grid.logodds[behind] < 0.


def test_wall_truncation_can_be_disabled():
    grid = prepared_grid(map_wall_truncate=False)
    wall = grid.cell((2.0, 0.))
    grid.logodds[wall] = 2.5
    grid.update(np.array([[3., 0., -.35]]), Pose())
    behind = grid.cell((2.5, 0.))
    assert grid.logodds[behind] < 0.
