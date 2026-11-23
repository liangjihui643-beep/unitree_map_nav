import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import _project, detect_from_probability


def make_prob(block=None, h=240, w=320):
    prob = np.zeros((3, h, w), np.float32)
    prob[0] = 1.0  # ground everywhere
    if block is not None:
        u0, u1, v0, v1 = block
        prob[1, v0:v1, u0:u1] = 0.95
        prob[0, v0:v1, u0:u1] = 0.03
    return prob


def paint_rock(cfg, distance=1.5, lateral=0.0, height=0.10, half_width=0.15):
    k = np.asarray(cfg.camera_matrix, float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    t = np.asarray(cfg.body_from_camera, float)
    rot, tr = t[:3, :3], t[:3, 3]
    ground = np.array([distance, lateral, -cfg.base_height])
    left = _project(ground + [0, half_width, 0], rot, tr, fx, fy, cx, cy)
    right = _project(ground - [0, half_width, 0], rot, tr, fx, fy, cx, cy)
    top = _project(ground + [0, 0, height], rot, tr, fx, fy, cx, cy)
    u0, u1 = int(min(left[0], right[0])), int(max(left[0], right[0]))
    v_top, v_bottom = int(top[1]), int(_project(ground, rot, tr, fx, fy, cx, cy)[1])
    return (u0, u1, v_top, v_bottom)


def test_detects_low_rock_height():
    cfg = Config()
    block = paint_rock(cfg, distance=1.5, height=0.10)
    dets = detect_from_probability(make_prob(block), cfg.camera_matrix,
                                   cfg.body_from_camera, cfg.base_height)
    assert dets, "expected one low obstacle"
    best = dets[0]
    assert abs(best.height - 0.10) < 0.05
    assert abs(best.x - 1.5) < 0.4


def test_empty_ground_has_no_detection():
    cfg = Config()
    dets = detect_from_probability(make_prob(), cfg.camera_matrix,
                                   cfg.body_from_camera, cfg.base_height)
    assert dets == []


def test_taller_rock_is_not_lower():
    cfg = Config()
    low = detect_from_probability(make_prob(paint_rock(cfg, height=0.06)), cfg.camera_matrix,
                                  cfg.body_from_camera, cfg.base_height)
    high = detect_from_probability(make_prob(paint_rock(cfg, height=0.14)), cfg.camera_matrix,
                                   cfg.body_from_camera, cfg.base_height)
    assert low and high
    assert high[0].height > low[0].height
