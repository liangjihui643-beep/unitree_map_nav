import numpy as np

from go2_rescue.config import Config
from go2_rescue.contract_map import ContractMap, LegacyMap


def observation(azimuth=0, distance=2.0):
    obs = np.ones(144, np.float32)
    obs[:72] = 1.
    obs[72:] = 0.
    index = int((azimuth + np.pi) / (2 * np.pi) * 72) % 72
    obs[72 + index] = 1.
    obs[index] = distance / 12.
    return obs


def test_unknown_ground_echoes_are_stored_as_pending():
    config = Config()
    config.lidar_mount = [.32, 0., .05]
    config.lidar_vertical_fov_deg = 96.
    contract = ContractMap(config)
    contract.update(observation(), np.array([0., 0., .3]), np.array([1., 0., 0., 0.]))
    assert int(contract.occupied.sum()) == 0
    assert len(contract.pending) > 0
    assert int(contract.pending_cells.sum()) > 0


def test_one_echo_can_commit_only_one_evidence():
    config = Config()
    config.lidar_mount = [.32, 0., .05]
    config.lidar_vertical_fov_deg = 96.
    contract = ContractMap(config)
    cell = contract.cell((1.0, 0.))
    contract.ground[cell] = 0.
    contract.pending = [{"candidates": [(cell, .30)], "pose_z": .3, "azimuth": 0., "range": 1.}]
    contract._recompute_pending()
    assert int(contract.occupied[cell]) == 1
    assert not contract.pending


def test_multi_cell_pending_never_commits_multiple_cells():
    config = Config()
    contract = ContractMap(config)
    a = contract.cell((1.0, 0.))
    b = contract.cell((1.2, 0.))
    contract.ground[a] = contract.ground[b] = 0.
    contract.pending = [{"candidates": [(a, .30), (b, .30)], "pose_z": .3,
                         "azimuth": 0., "range": 1.}]
    contract._recompute_pending()
    assert int(contract.occupied.sum()) == 0
    assert len(contract.pending) == 1


def test_single_035m_echo_stays_pending():
    config = Config()
    config.lidar_mount = [.32, 0., .05]
    config.lidar_vertical_fov_deg = 96.
    contract = ContractMap(config)
    contract.update(observation(distance=.35), np.array([0., 0., .3]),
                    np.array([1., 0., 0., 0.]), frame_index=0)
    assert not np.isfinite(contract.ground).any()
    assert not contract.near_body_seed.any()
    assert not contract.free.any()
    assert len(contract.pending) > 0
    assert int(contract.ground_hits.max()) <= 1


def test_published_free_and_occupied_are_exclusive():
    config = Config()
    contract = ContractMap(config)
    cell = contract.cell((1., 1.))
    contract.free[cell] = True
    contract.occupied[cell] = 1
    assert not contract.published_free()[cell]
    assert contract.published_occupied()[cell]
    assert contract.free_occupied_conflicts() == 0


def test_bounded_ground_completion_stops_at_jump():
    config = Config()
    contract = ContractMap(config)
    a = contract.cell((1.0, 1.0))
    b = contract.cell((1.1, 1.0))
    c = contract.cell((1.2, 1.0))
    contract.ground[a] = 0.
    contract.ground_views[b] = {0: (np.array([0., 0.]), .02),
                                1: (np.array([.2, 0.]), .03)}
    contract.ground_views[c] = {2: (np.array([.4, 0.]), .50),
                                3: (np.array([.5, 0.]), .51)}
    contract.finalize()
    assert np.isfinite(contract.ground[b])
    assert not np.isfinite(contract.ground[c])


def test_multi_view_wall_segment_gets_one_vote_per_echo():
    config = Config()
    contract = ContractMap(config)
    cell = contract.cell((2.0, 1.0))
    records = [{"candidates": [(cell, .30), (cell, .50)], "pose_z": .3,
                "azimuth": 0., "range": 2., "frame": i,
                "view_xy": np.array([2.0 - .2 * i, 1.0])} for i in range(3)]
    contract.pending = records
    contract._confirm_walls()
    assert contract.wall[cell]
    assert int(contract.wall_votes[cell]) == 3
    assert contract.published_occupied()[cell]


def test_legacy_fixed_elevation_can_claim_occupancy():
    config = Config()
    config.lidar_mount = [.32, 0., .05]
    legacy = LegacyMap(config)
    legacy.update(observation(), np.array([0., 0., .3]), np.array([1., 0., 0., 0.]))
    assert int(legacy.occupied.sum()) > 0
