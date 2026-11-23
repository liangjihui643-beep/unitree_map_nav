import numpy as np

from go2_rescue.ball_multiview import BallTwoView
from go2_rescue.config import Config
from go2_rescue.lidar_obs import RangeOnlyEvidence, decode_lidar_obs, encode_lidar_obs
from go2_rescue.mapping import GridMap
from go2_rescue.types import BallObservation, Pose


def test_range_only_does_not_claim_ground():
    grid = GridMap(Config())
    layer = RangeOnlyEvidence(grid, 0., 96.)
    mask = np.zeros(72, bool)
    mask[36] = True
    obs = encode_lidar_obs(mask, [3.])
    ids, distances = decode_lidar_obs(obs)
    np.testing.assert_array_equal(ids, [36])
    np.testing.assert_allclose(distances, [3.])
    assert obs[35] == 1 and obs[107] == 0
    q = np.array([1., 0., 0., 0.])
    layer.update(grid, encode_lidar_obs(np.zeros(72, bool), []), q, Pose(), .05, 12.)
    assert not layer.ray_prefix.any() and not layer.possible_hit.any()
    layer.update(grid, obs, q, Pose(), .05, 12.)
    assert layer.ray_prefix[grid.cell((1.6, 0.))]
    assert layer.possible_hit.sum() > 1
    assert not grid.seen.any() and not grid.logodds.any()


def test_raised_ball_does_not_need_ground_intersection():
    k = np.array([[220., 0., 160.], [0., 220., 120.], [0., 0., 1.]])
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    solver = BallTwoView(k)
    for i, y in enumerate((-.29, -.18, 0.)):
        pose = Pose(y=y, sigma=.04)
        image = k @ (optical.T @ (np.array([1.2, .10, .55]) - np.array([0., y, .4])))
        d = BallObservation(tuple(image[:2]/image[2]), 20., .95, .9)
        solver.update([d], pose, .4, optical, np.zeros(3), i*.2, .85)
        if i < 2:
            assert d.position is None
    assert d.range_source == "visual_multiview"
    np.testing.assert_allclose(d.position, [1.2, .1], atol=1e-6)
def test_overhead_ray_not_counted_as_ground_band_clear():
    grid = GridMap(Config())
    layer = RangeOnlyEvidence(grid, 0., 96.)
    mask = np.zeros(72, bool)
    mask[36] = True
    obs = encode_lidar_obs(mask, [3.])
    q = np.array([1., 0., 0., 0.])
    ground = np.zeros_like(grid.floor)
    layer.update(grid, obs, q, Pose(), .05, 12., base_z=.8, ground=ground)
    index = grid.cell((1.5, 0.))
    assert layer.ray_prefix[index]
    assert not layer.ground_band_clear[index]

    # Same folded observation can certify the near-floor band only when
    # every possible elevation agrees at the observed patch.
    layer = RangeOnlyEvidence(grid, 0., 1.)
    layer.update(grid, obs, q, Pose(), .05, 12., base_z=-.035, ground=ground)
    assert layer.ground_band_clear[index]

def test_interpolation_requires_mask_and_continuous_depth():
    from go2_rescue.rubble_visual_height import bounded_stone_surface
    xyz = np.array([[1., 0., .20], [1.1, 0., .21], [1., .12, .22]])
    uv = np.array([[5., 5.], [13., 5.], [5., 13.]])
    mask = np.ones((20,20), bool)
    assert len(bounded_stone_surface(xyz, uv, np.ones(3), mask)) == 4
    assert not len(bounded_stone_surface(xyz, uv, np.array([1.,1.,1.25]), mask))
    mask[6:9,6:9] = False
    assert not len(bounded_stone_surface(xyz, uv, np.ones(3), mask))

def test_visual_stance_input_discards_simulator_xyz():
    from go2_rescue.types import Frame
    from go2_rescue.visual_stance_odometry import VisualStanceOdometry

    class Segment:
        @staticmethod
        def segment(image):
            return None

    class Audit:
        status = {}

        @staticmethod
        def update(frame, plane, probability):
            assert frame.points.shape == (0, 3)
            assert frame.point_times is None
            return None

    odom = VisualStanceOdometry(Config(), Segment())
    auditor = Audit()
    auditor.previous = None
    odom.vo = auditor
    image = np.zeros((20, 20, 3), np.uint8)
    xyz, yaw = odom.update(Frame(0., np.array([[.5, .5, .5]]), image=image,
                                 point_times=np.array([0.])))
    assert np.isfinite(xyz).all() and np.isfinite(yaw)

def test_ground_survives_without_rubble_and_clears_on_pose_reset():
    from go2_rescue.rubble_visual_height import RubbleVisualHeight
    solver = RubbleVisualHeight(np.eye(3), independent_ground=True)
    image = np.random.default_rng(3).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), bool)
    solver.observe(image, mask, np.eye(3), np.zeros(3), Pose(), 0., None, 1.,
                   ground_mask=~mask, base_z=.33)
    assert len(solver.views) == 1
    solver.ground_history.append((1., np.array([[1., 1., .1]])))
    solver.clear()
    assert not solver.views and not solver.ground_history

def test_outline_rejects_partial_pattern_and_refines_full_circle():
    import cv2
    from go2_rescue.ball_outline import refine_ball_outline
    image = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(image, (50, 50), 18, (220, 220, 220), -1)
    full = BallObservation((52., 49.), 17., .95, .85)
    assert refine_ball_outline(image, full)
    assert np.linalg.norm(np.asarray(full.center)-[50, 50]) < 1.
    assert abs(full.radius-18.) < 1.5
    image[:, 52:] = 0
    partial = BallObservation((50., 50.), 18., .95, .85)
    assert not refine_ball_outline(image, partial)

def test_hybrid_prefers_independent_geometry_and_counts_only_one_hit(monkeypatch):
    from go2_rescue import hybrid_ball
    from go2_rescue.tracking import Tracker
    from go2_rescue.types import Frame

    class Vision:
        k = np.eye(3)
        t = np.eye(4)
        c = Config()

        @staticmethod
        def locate_ball(detection, frame, pose):
            assert len(frame.points) == 0 and frame.point_times is None
            detection.position = np.array([1.1, 0.])
            detection.sigma = .18
            detection.range_source = "nominal_flat_ground"
            return detection

    solver = hybrid_ball.HybridBallRange(Vision())
    monkeypatch.setattr(hybrid_ball, "refine_ball_outline", lambda image, d: d.center[0] == 30.)
    monkeypatch.setattr(solver, "flat_range_consistent", lambda *args: True)
    def solved(detections, *args):
        assert len(detections) == 1
        detection = detections[0]
        detection.position = np.array([1.3, 0.1])
        detection.sigma = .12
        detection.range_source = "visual_image_track"
    monkeypatch.setattr(solver.outline, "update", solved)
    pose = Pose(quality=.9)
    flat = np.array([0., 0., -Vision.c.base_height])
    tracker = Tracker(Vision.c, static_targets=True)
    events = []
    for stamp in (.1, .3, .5, .7):
        same_ball = [BallObservation((30., 40.), 15., .97, .9),
                     BallObservation((34., 41.), 13., .98, .9)]
        frame = Frame(stamp, np.ones((1,3)), point_times=np.array([stamp]))
        solver.update(same_ball, None, frame, pose, .33, flat, .85)
        valid = [d for d in same_ball if hybrid_ball.usable_ball(d,.85)]
        assert len(valid) == 1 and valid[0].range_source == "visual_image_track"
        events.extend(tracker.update(valid, pose, stamp))
    assert len(events) == 1
    assert tracker.tracks[0].hits == 4


def test_hybrid_flat_fallback_is_disabled_on_slope_or_missing_support(monkeypatch):
    from go2_rescue import hybrid_ball
    from go2_rescue.perception import Perception
    from go2_rescue.types import Frame

    config = Config()
    solver = hybrid_ball.HybridBallRange(Perception(config))
    monkeypatch.setattr(hybrid_ball, "refine_ball_outline", lambda image, d: False)
    candidate = BallObservation((160., 200.), 45., .95, .9)
    frame = Frame(.5, np.empty((0,3)))
    for support in (None, np.array([.24,0.,-config.base_height])):
        solver.update([candidate], None, frame, Pose(quality=.9), .33, support, .85)
        assert candidate.position is None


def test_hybrid_tightens_scale_without_changing_the_ground_ray():
    from go2_rescue.hybrid_ball import HybridBallRange
    from go2_rescue.perception import Perception
    from go2_rescue.types import Frame

    solver = HybridBallRange(Perception(Config()))
    vision = solver.vision
    frame = Frame(0., np.empty((0, 3)))
    pose = Pose(quality=1., sigma=.03)
    optical = (np.array([1.25, 0., vision.c.ball_radius-vision.c.base_height])
               - vision.t[:3, 3]) @ vision.t[:3, :3]
    pixel = vision.k @ optical
    radius = np.mean(np.diag(vision.k)[:2])*vision.c.ball_radius/optical[2]
    for ratio, passes in ((1., True), (.68, False), (1.4, False)):
        detection = BallObservation(tuple(pixel[:2]/pixel[2]), radius*ratio, .97, .9)
        vision.locate_ball(detection, frame, pose)
        assert detection.position is not None  # Legacy accepts all three.
        np.testing.assert_allclose(detection.position, [1.25, 0.], atol=1e-8)
        assert bool(solver.flat_range_consistent(detection, pose, np.eye(3))) is passes

def test_hybrid_keeps_unconfirmed_plane_out_of_tracker(monkeypatch):
    from go2_rescue import hybrid_ball
    from go2_rescue.types import Frame

    class Vision:
        k = np.eye(3)
        t = np.eye(4)
        c = Config()

        @staticmethod
        def locate_ball(detection, frame, pose):
            detection.position = np.array([1., 0.])
            detection.sigma = .15
            return detection

    solver = hybrid_ball.HybridBallRange(Vision())
    monkeypatch.setattr(hybrid_ball, "refine_ball_outline", lambda *args: False)
    monkeypatch.setattr(solver, "flat_range_consistent", lambda *args: True)
    candidate = BallObservation((30., 40.), 15., .97, .9)
    solver.update([candidate], None, Frame(.1, np.empty((0,3))), Pose(quality=.9),
                  .33, np.array([0.,0.,-Vision.c.base_height]), .85)
    assert len(solver.provisional)==1
    assert candidate.position is None

def test_image_track_association_survives_yaw_without_inventing_depth():
    from go2_rescue.ball_image_tracks import BallImageTracks
    from go2_rescue.geometry import rot2

    k = np.array([[205., 0., 160.], [0., 205., 120.], [0., 0., 1.]])
    mount = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    target = np.array([2., 0., 0.])
    tracks = BallImageTracks(k, compensate_rotation=True)
    for i, yaw in enumerate(np.deg2rad([-16., -8., 0., 8., 16.])):
        world = np.eye(3)
        world[:2,:2] = rot2(yaw)
        optical = target @ (world @ mount)
        pixel = k @ optical
        d = BallObservation(tuple(pixel[:2]/pixel[2]), 13., .95, .8)
        tracks.update([d], Pose(yaw=yaw, quality=1.), 0., mount, np.zeros(3), i*.2, .85)
        assert d.position is None
    assert len(tracks.tracks) == 1
    assert len(tracks.tracks[0]) == 5

def test_range_envelope_uses_height_floor_but_does_not_promote_free():
    grid = GridMap(Config())
    mask = np.zeros(72, bool)
    mask[36] = True
    obs = encode_lidar_obs(mask, [3.])
    q = np.array([1.,0.,0.,0.])
    wide = RangeOnlyEvidence(grid, -22.5, 96.)
    bounded = RangeOnlyEvidence(grid, -22.5, 96., height_floor_m=-.22)
    for layer in (wide, bounded):
        layer.update(grid, obs, q, Pose(.05,.05), .05, 12., base_z=.5)
    assert bounded.possible_hit.sum() < wide.possible_hit.sum()
    assert np.all(~bounded.possible_hit | wide.possible_hit)
    assert not bounded.ground_band_clear.any()
    assert not grid.seen.any()


def test_cell_crossing_checks_whole_segment_and_rejects_a_corner():
    from go2_rescue.lidar_obs import cell_crossing_fractions
    d = np.array([[2.,0.],[2.,.1]])
    hit = cell_crossing_fractions(np.array([0., .1]),d,np.array([.5,0.]),np.array([1.,.3]))
    np.testing.assert_allclose(hit, [[.25,.25],[.5,.5]])
    assert cell_crossing_fractions(np.zeros(2),np.array([[1.,1.]]),
                                   np.array([.5,0.]),np.array([1.,.5])) is None

def test_range_only_known_endpoint_still_is_not_an_obstacle():
    grid = GridMap(Config())
    mask = np.zeros(72, bool)
    mask[36] = True
    obs = encode_lidar_obs(mask,[3.])
    layer = RangeOnlyEvidence(grid, 0., .1, height_floor_m=-.22)
    layer.update(grid, obs, np.array([1.,0.,0.,0.]), Pose(.025,.025),
                 .05,12.)
    assert layer.unambiguous_hit[grid.cell((3.025,.025))]
    assert layer.unambiguous_hit.sum() == 1
    assert not grid.logodds.any()

def test_visual_ground_rejects_contacts_on_wrong_level():
    from go2_rescue.rubble_visual_height import contacts_with_visual_support
    visual = np.array([[.90,.90,.03],[1.0,.92,.03],[1.08,1.,.04],[1.,1.08,.03]])
    contacts = np.array([[1.,1.,.033],[1.,1.,.16],[3.,1.,.03]])
    np.testing.assert_allclose(contacts_with_visual_support(contacts, visual), contacts[:1])
