import numpy as np

from go2_rescue.scan_filter import INVALID, shadow_filter, speckle_filter


def test_speckle_removes_isolated_return():
    scan = np.full((2, 20), 2.0)
    scan[0, 5] = 5.0  # isolated outlier
    out = speckle_filter(scan, window=1, delta=0.1)
    assert out[0, 5] == INVALID
    assert out[0, 4] == 2.0 and out[0, 6] == 2.0


def test_speckle_disabled_is_identity():
    scan = np.array([[1.0, 5.0, 1.0]])
    out = speckle_filter(scan, window=0, delta=0.0)
    assert np.array_equal(out, scan)


def test_shadow_requires_range_jump():
    # Flat surfaces have no range discontinuity, so a small segment angle alone
    # must NOT remove them (the v1 filter wiped out 85% of planar returns).
    step = np.deg2rad(5.0)
    scan = np.full((1, 9), 2.0)
    out = shadow_filter(scan, step, min_angle_deg=10.0, neighbours=1)
    assert (out[0, 1:-1] == 2.0).all()


def test_shadow_keeps_flat_wall():
    step = np.deg2rad(5.0)
    scan = np.full((1, 9), 2.0)
    out = shadow_filter(scan, step, min_angle_deg=2.0, neighbours=1)
    assert (out[0, 1:-1] == 2.0).all()
