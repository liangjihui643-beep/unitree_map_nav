import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.controller import Autonomy
from go2_rescue.geometry import gravity_rotation
from go2_rescue.hazards import WaterDetector
from go2_rescue.mapping import GridMap
from go2_rescue.planning import DWA
from go2_rescue.synthetic import SensorWorld
from go2_rescue.types import Pose, Velocity, Frame


def test_quota_does_not_drive_towards_unreachable_marker():
    c = Config(map_size=10., target_quota=1)
    world, autonomy = SensorWorld(c), Autonomy(c)
    autonomy.step(world.frame())
    autonomy.tracker.reported_positions.append(np.array([1., 0.]))
    autonomy.finish_goal = np.array([4., 4.])
    world.advance(Velocity())
    frame = world.frame()
    frame.image = None
    result = autonomy.step(frame)
    assert result.state != "to_finish"
    assert not any(e.kind == "finish" for e in result.events)


def test_timeout_remains_effective_during_localization_failure():
    c = Config(map_size=10., mission_seconds=1., return_reserve=.4)
    world, autonomy = SensorWorld(c), Autonomy(c)
    autonomy.step(world.frame())
    result = autonomy.step(Frame(1.2, np.empty((0, 3))))
    assert result.state == "timed_out" and result.velocity == Velocity()
    assert [e.kind for e in result.events] == ["timeout"]


def test_recovery_timeout_defers_the_failed_frontier_instead_of_retrying_it():
    c = Config(map_size=10.)
    world, autonomy = SensorWorld(c), Autonomy(c)
    autonomy.step(world.frame())
    old_goal = np.array([1., .5])
    autonomy.state, autonomy.resume_goal = "recovering", old_goal.copy()
    autonomy.recovery_path = np.array([[0., 0., 0.], [1., 0., 0.]])
    autonomy.recovery_index, autonomy.recovery_until = 1, .05
    world.advance(Velocity())
    frame = world.frame()
    frame.image = None
    autonomy.step(frame)
    assert autonomy.explorer.blacklist[autonomy.grid.cell(old_goal)] > frame.timestamp
    assert autonomy.resume_goal is None


def test_diagnostic_map_origin_is_first_accepted_scan_after_warmup():
    mujoco = pytest.importorskip("mujoco")
    from go2_rescue.evaluation import SimulationEvaluator
    from go2_rescue.types import Event
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    evaluator = SimulationEvaluator(model, model, [.8, .8], 0.)
    data.qpos[:3] = [1.05, .85, .33]
    data.qpos[3:7] = np.roll(Rotation.from_euler("z", .2).as_quat(), 1)
    data.time = .6
    mujoco.mj_forward(model, data)
    evaluator.update(data, [Event("start", .6)])
    assert np.allclose(evaluator.initial, [1.05, .85])
    assert np.allclose(evaluator.map_rotation, Rotation.from_euler("z", .2).as_matrix()[:2, :2])
    assert evaluator.map_anchor_timestamp == .6


def test_dynamic_window_never_exceeds_limits_when_measured_speed_overshoots():
    c = Config(map_size=10.)
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-2, 2, .08), np.arange(-2, 2, .08))
    grid.update(np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)]), Pose())
    path = np.array([[0., 0.], [1., 0.]])
    command = DWA(c).command(Pose(), Velocity(1., -.9, 2.), path, np.empty((0, 3)), grid)
    assert abs(command.vx) <= c.max_vx
    assert abs(command.vy) <= c.max_vy
    assert abs(command.wz) <= c.max_wz


def test_rendered_puddle_rgb_keepout_candidate_blocks_planning():
    mujoco = pytest.importorskip("mujoco")
    from go2_rescue.mujoco_sim import SimSensors, simulation_config
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Sensor calibration fixture, not a navigation shortcut.
    data.qpos[:3] = [.75, 5.4, .51]
    data.qpos[3:7] = np.roll(Rotation.from_euler("xyz", [0., -.21, np.pi/2]).as_quat(), 1)
    mujoco.mj_forward(model, data)
    c = simulation_config()
    c.water_detection = True
    c.navigation_support = True
    sensors = SimSensors(model, data, c)
    try:
        frame = sensors.observe(Velocity())
    finally:
        sensors.close()
    grid = GridMap(c)
    grid.update(frame.points @ gravity_rotation(frame.quaternion).T, Pose())
    # This fixture is level at the map origin; make the RGB/stance branch's
    # prerequisite explicit instead of asking GridMap to invent a plane.
    grid.support_plane = np.zeros(3)
    detector = WaterDetector(c)
    polygons = detector.detect(frame.image, frame, Pose(), grid)
    assert polygons
    assert detector.last_sources == ["rgb_stance_keepout_candidate"]
    for polygon in polygons:
        grid.add_hazard(polygon)
        grid.add_hazard(polygon)
    assert (grid.hazard_votes >= 2).any()
    assert not grid.traversable()[grid.cell(polygons[0].mean(axis=0))]
    c.water_detection = False
    assert not WaterDetector(c).detect(frame.image, frame, Pose(), grid)


def test_laser_ground_colour_confirms_puddle_without_rgb_keepout():
    from go2_rescue.hazards import WaterDetector
    from go2_rescue.mujoco_sim import simulation_config
    from go2_rescue.types import Frame

    c = simulation_config()
    c.water_detection = True
    c.navigation_support = False
    k = np.asarray(c.camera_matrix, float)
    body_from_camera = np.asarray(c.body_from_camera, float)
    # Build body-frame ground returns that project into a small cyan image
    # patch. This exercises the lidar-confirmed branch, not the RGB keepout
    # candidate that requires a stance plane.
    us, vs = np.arange(150., 171., 2.), np.arange(170., 191., 2.)
    uv = np.array(np.meshgrid(us, vs)).reshape(2, -1).T
    x = (uv[:, 0] - k[0, 2]) / k[0, 0]
    y = (uv[:, 1] - k[1, 2]) / k[1, 1]
    target = -c.base_height
    denominator = body_from_camera[2, 0] * x + body_from_camera[2, 1] * y + body_from_camera[2, 2]
    depth = (target - body_from_camera[2, 3]) / denominator
    camera_points = np.column_stack([x * depth, y * depth, depth])
    body = camera_points @ body_from_camera[:3, :3].T + body_from_camera[:3, 3]
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(image, (int(us.min()), int(vs.min())),
                  (int(us.max()), int(vs.max())), (255, 255, 0), -1)
    grid = GridMap(c)
    grid.update(body, Pose())
    frame = Frame(0., body, image=image)
    detector = WaterDetector(c)
    polygons = detector.detect(image, frame, Pose(), grid)
    assert len(polygons) == 1
    assert detector.last_sources == ["lidar_ground_colour"]
    assert .004 < cv2.contourArea(polygons[0]) < 1.5


def test_evaluator_water_episodes_are_diagnostic_and_debounced():
    mujoco = pytest.importorskip("mujoco")
    from go2_rescue.evaluation import SimulationEvaluator
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [.8, .8, .3]
    mujoco.mj_forward(model, data)
    evaluator = SimulationEvaluator(model, model, [.8, .8], 0.)
    evaluator.update(data)
    assert evaluator.water_episodes == 0
    puddle = data.geom("shallow_pale_puddle").xpos.copy()
    data.qpos[:3] = puddle + [0., 0., .30]
    mujoco.mj_forward(model, data)
    evaluator.update(data)
    evaluator.update(data)
    assert evaluator.water_episodes == 1
    assert not evaluator.report(data, False, False)["local_task_pass"]


def test_sparse_observed_corner_refuses_unsafe_escape():
    mujoco = pytest.importorskip("mujoco")
    from go2_rescue.mujoco_sim import SimSensors, simulation_config
    from go2_rescue.planning import Explorer
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [1.09, 3.72, .33]
    data.qpos[3:7] = np.roll(Rotation.from_euler("z", .514).as_quat(), 1)
    mujoco.mj_forward(model, data)
    c = simulation_config()
    sensors = SimSensors(model, data, c)
    try:
        points = sensors.lidar()
    finally:
        sensors.close()
    grid = GridMap(c)
    grid.update(points, Pose())
    route = Explorer().maneuver(grid, Pose())
    assert route.shape == (0, 3)
    half = np.array([c.footprint_half_length, c.footprint_half_width]) + c.safety_margin
    x, y = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
    offsets = np.column_stack([x.ravel(), y.ravel()])
    assert not grid.footprints_safe(np.zeros((1, 2)), np.zeros(1), offsets)[0]


def test_observed_map_allows_collision_checked_se2_escape():
    from go2_rescue.mujoco_sim import simulation_config
    from go2_rescue.planning import Explorer

    c = simulation_config()
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-2., 3.01, .08), np.arange(-2., 2.01, .08))
    ground = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)])
    grid.update(ground, Pose())
    wall_y = np.arange(-1., 1.01, .06)
    wall = np.column_stack([np.full(len(wall_y), 1.), wall_y,
                            np.full(len(wall_y), -.05)])
    grid.update(np.vstack([ground, wall]), Pose())
    route = Explorer().maneuver(grid, Pose())
    assert len(route) > 2 and np.linalg.norm(route[-1, :2]) > .45
    assert abs(route[-1, 2]) > .35
    half = np.array([c.footprint_half_length, c.footprint_half_width]) + c.safety_margin
    x, y = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
    offsets = np.column_stack([x.ravel(), y.ravel()])
    for a, b in zip(route[:-1], route[1:]):
        fraction = np.linspace(0., 1., 9)
        positions = a[:2] + (b[:2] - a[:2]) * fraction[:, None]
        angle = (b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
        assert grid.footprints_safe(positions, a[2] + angle * fraction, offsets).all()


@pytest.mark.parametrize("roof,allowed", [(.45, False), (.65, True)])
def test_tunnel_clearance_distinguishes_low_and_passable_roofs(roof, allowed):
    c = Config(map_size=10., body_clearance=.48)
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-1., 3., .075), np.arange(-1., 1., .075))
    ground = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)])
    grid.update(ground, Pose())
    rx, ry = np.meshgrid(np.arange(.7, 2.5, .075), np.arange(-.7, .7, .075))
    ceiling = np.column_stack([rx.ravel(), ry.ravel(), np.full(rx.size, roof-c.base_height)])
    grid.update(np.vstack([ground, ceiling]), Pose())
    assert bool(grid.traversable()[grid.cell([1.5, 0.])]) == allowed


def test_water_uses_contact_envelope_while_walls_use_full_body():
    c = Config(map_size=10., robot_radius=.23, safety_margin=.05,
               footprint_half_length=.35, footprint_half_width=.23,
               hazard_foot_half_length=.28, hazard_foot_half_width=.22)
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-2., 2., .075), np.arange(-2., 2., .075))
    grid.update(np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)]), Pose())
    polygon = np.array([[.45, -.1], [.65, -.1], [.65, .1], [.45, .1]])
    grid.add_hazard(polygon)
    grid.add_hazard(polygon)
    xx, yy = np.meshgrid(np.linspace(-.40, .40, 9), np.linspace(-.28, .28, 7))
    body = np.column_stack([xx.ravel(), yy.ravel()])
    assert grid.footprints_safe(np.array([[0., 0.]]), np.array([0.]), body)[0]
    assert not grid.footprints_safe(np.array([[.15, 0.]]), np.array([0.]), body)[0]


def test_cleared_low_obstacle_does_not_leave_a_permanent_false_roof():
    c = Config(map_size=10., body_clearance=.48, base_height=.33)
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-1., 3., .075), np.arange(-1., 1., .075))
    ground = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)])
    grid.update(ground, Pose())
    low_obstacle = np.array([[1.05, .05, .12-c.base_height]])
    grid.update(np.vstack([ground, low_obstacle]), Pose())
    assert not grid.traversable()[grid.cell([1.05, .05])]
    for _ in range(12):
        grid.update(ground, Pose())
    assert grid.traversable()[grid.cell([1.05, .05])]
    # A genuine low overhead surface remains a clearance constraint; ground
    # returns below a roof do not disprove that roof.
    rx, ry = np.meshgrid(np.arange(.7, 2.5, .075), np.arange(-.7, .7, .075))
    roof = np.column_stack([rx.ravel(), ry.ravel(), np.full(rx.size, .45-c.base_height)])
    grid.update(np.vstack([ground, roof]), Pose())
    for _ in range(12):
        grid.update(ground, Pose())
    assert not grid.traversable()[grid.cell([1.5, 0.])]


def test_frontier_goal_has_room_for_whole_body_before_unknown_edge():
    from go2_rescue.planning import Explorer
    c = Config(map_size=10., robot_radius=.23, safety_margin=.05,
               footprint_half_length=.35, footprint_half_width=.23)
    grid = GridMap(c)
    x, y = np.meshgrid(np.arange(-.6, 3., .075), np.arange(-.6, .6, .075))
    grid.update(np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -c.base_height)]), Pose())
    goal = Explorer().choose(grid, Pose(), 0.)
    assert goal is not None and np.linalg.norm(goal) > .25
    x, y = np.meshgrid(np.linspace(-.4, .4, 9), np.linspace(-.28, .28, 7))
    offsets = np.column_stack([x.ravel(), y.ravel()])
    headings = np.linspace(0., np.pi, 12, endpoint=False)
    assert grid.footprints_safe(np.repeat(goal[None], len(headings), axis=0), headings, offsets).any()


def test_loop_correction_cancels_recovery_route_in_previous_map_frame(monkeypatch):
    c = Config(map_size=10.)
    world, autonomy = SensorWorld(c), Autonomy(c)
    autonomy.step(world.frame())
    old = autonomy.localization.pose
    corrected = Pose(.3, .1, 0., quality=.8, sigma=.03)
    autonomy.localization.keyframes[0].pose = corrected
    autonomy.localization.pose = corrected
    autonomy.localization.corrections = [(old, corrected)]
    monkeypatch.setattr(autonomy.localization, "update", lambda f: (corrected, f.points))
    autonomy.state, autonomy.resume_goal = "recovering", np.array([2., 1.])
    autonomy.recovery_path = np.array([[0., 0., 0.], [1., 0., 0.]])
    autonomy.recovery_index, autonomy.recovery_until = 1, 30.
    world.advance(Velocity())
    frame = world.frame()
    frame.image = None
    result = autonomy.step(frame)
    assert result.state != "recovering"
    assert autonomy.recovery_path.shape == (0, 3)
    assert autonomy.resume_goal is None
