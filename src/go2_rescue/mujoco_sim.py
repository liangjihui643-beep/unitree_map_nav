"""Real MuJoCo dynamics + open-source MPC + the existing autonomy pipeline."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from collections import deque

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .controller import Autonomy
from .mpc_locomotion import ConvexMPC
from .recording import Recorder
from .scan_filter import shadow_filter, speckle_filter
from .types import Frame, Pose, Step, Velocity


class SimSensors:
    """Sensor renderer/raycaster; plant truth is internal to sensor generation.

    The public output is Frame, with joint-kinematic measured velocity. It does
    not expose plant qpos, geom IDs, ball coordinates or a precomputed map.
    """
    def __init__(self, model, data, config: Config, seed=0):
        self.model, self.data, self.config = model, data, config
        self.base = model.body("base_link").id
        self.rng = np.random.default_rng(seed)
        self.robot_geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == self.base)
        azimuth = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        # Guide v1.0.02: 72 azimuth x 9 elevation, vertical FOV 96 deg; the exact
        # elevation list/offsets belong to sensors.json (not published yet), so a
        # symmetric 96 deg sweep is the documented placeholder.
        elevation = np.deg2rad(np.linspace(
            config.lidar_elevation_center_deg - config.lidar_vertical_fov_deg / 2,
            config.lidar_elevation_center_deg + config.lidar_vertical_fov_deg / 2, 9))
        self.n_az, self.n_el = len(azimuth), len(elevation)
        self.az_step = float(azimuth[1] - azimuth[0])
        self.directions = np.array([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
                                    for e in elevation for a in azimuth])
        # Multisampled integer-ID edges blend to invalid IDs on WSL D3D12.
        # Disable offscreen MSAA for RGB and evaluation masks consistently.
        self.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(model, height=config.camera_size[1],
                                        width=config.camera_size[0])
        from OpenGL.GL import glGetString, GL_RENDERER
        self.graphics_device = (glGetString(GL_RENDERER) or b"unknown").decode(errors="replace")
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 1  # Official course collision surfaces are group 3.
        self.camera = mujoco.MjvCamera()
        fy = float(config.camera_matrix[1][1])
        self.model.vis.global_.fovy = float(np.degrees(
            2 * np.arctan(config.camera_size[1] / (2 * fy))))

    def lidar(self):
        model, data = self.model, self.data
        rotation = data.xmat[self.base].reshape(3, 3)
        offset = np.asarray(self.config.lidar_mount, float)
        origin = data.xpos[self.base] + rotation @ offset
        directions = self.directions @ rotation.T
        ids = np.full(len(directions), -1, dtype=np.int32)
        distances = np.full(len(directions), -1., dtype=float)
        previous = model.geom_group[self.robot_geoms].copy()
        try:
            # Exclude the complete robot, not only the base-link geom. Restore
            # immediately; rendering and collision masks are unchanged.
            model.geom_group[self.robot_geoms] = 5
            groups = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)
            mujoco.mj_multiRay(model, data, origin, directions.ravel(), groups, True, -1,
                              ids, distances, None, len(directions), self.config.max_range)
        finally:
            model.geom_group[self.robot_geoms] = previous
        dirs = self.directions.reshape(self.n_el, self.n_az, 3)
        up_body = rotation.T @ np.array([0., 0., 1.])
        if self.config.lidar_contract_order:
            # Public contract order: range clip, dropout and noise on the 9x72
            # grid BEFORE gravity-height filtering and nearest-per-azimuth.
            raw = distances.reshape(self.n_el, self.n_az).copy()
            valid = np.isfinite(raw) & (raw >= self.config.min_range) & (raw <= self.config.max_range)
            if self.config.lidar_dropout > 0:
                valid &= self.rng.random(raw.shape) >= self.config.lidar_dropout
            noisy = raw + self.rng.normal(0., self.config.lidar_noise_std, raw.shape)
            noisy = np.clip(noisy, self.config.min_range, self.config.max_range)
            grid = np.where(valid, noisy, -1.)
        else:
            # Legacy profile: optional scan filters and noise after selection.
            scan = distances.reshape(self.n_el, self.n_az)
            if self.config.lidar_speckle_window > 0:
                scan = speckle_filter(scan, self.config.lidar_speckle_window,
                                      self.config.lidar_speckle_delta)
            if self.config.lidar_shadow_deg > 0:
                scan = shadow_filter(scan, self.az_step, self.config.lidar_shadow_deg,
                                     self.config.lidar_shadow_neighbours)
            grid = scan.reshape(self.n_el, self.n_az)
        pts = offset + dirs * np.where(grid > 0, grid, 0.)[:, :, None]
        height = pts @ up_body
        in_range = np.isfinite(grid) & (grid >= self.config.min_range) & (grid <= self.config.max_range)
        raw_valid_count = int(in_range.sum())
        eligible = in_range & (height >= self.config.lidar_height_floor)
        best_range = np.full(self.n_az, np.inf)
        best_el = np.full(self.n_az, -1, dtype=int)
        for el in range(self.n_el):
            better = eligible[el] & (grid[el] < best_range)
            best_range[better] = grid[el][better]
            best_el[better] = el
        chosen = best_el >= 0
        directions_selected = dirs[best_el[chosen], np.flatnonzero(chosen)]
        if self.config.lidar_contract_order:
            ranges = grid[best_el[chosen], np.flatnonzero(chosen)].copy()
        else:
            ranges = best_range[chosen].copy()
            ranges = ranges + self.rng.normal(0., self.config.lidar_noise_std, chosen.sum())
            ranges = np.clip(ranges, self.config.min_range, self.config.max_range)
            # Clip additive range noise at the eligibility boundary.
            vertical = directions_selected @ up_body
            height_range = np.full_like(vertical, np.inf)
            np.divide(self.config.lidar_height_floor - offset @ up_body, vertical,
                      out=height_range, where=vertical != 0)
            ranges = np.where(vertical < 0, np.minimum(ranges, height_range), ranges)
            ranges = np.where(vertical > 0, np.maximum(ranges, height_range), ranges)
        pts_out = offset + directions_selected * ranges[:, None]
        from .lidar_obs import encode_lidar_obs
        self.last_lidar_metadata = {
            "agent_obs": encode_lidar_obs(chosen, ranges, self.config.max_range),
            "valid": chosen.copy(), "azimuth_index": np.flatnonzero(chosen),
            "elevation_index": best_el[chosen].copy(), "range": ranges.copy(),
            "raw_ranges": grid.copy(), "up_body": up_body.copy(),
            "raw_valid_count": raw_valid_count,
            "raw_valid_fraction": raw_valid_count / float(self.n_el * self.n_az),
            "azimuth_valid_fraction": float(chosen.sum()) / float(self.n_az),
        }
        return pts_out

    def _camera_pose(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        transform = np.asarray(self.config.body_from_camera)
        origin = self.data.xpos[self.base] + rotation @ transform[:3, 3]
        optical = rotation @ transform[:3, :3]
        return origin, optical

    def image(self):
        origin, optical = self._camera_pose()
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = origin
            camera.forward[:] = optical[:, 2]
            camera.up[:] = -optical[:, 1]
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def depth(self):
        """Metric depth (metres) with the same camera; rendering/diagnostics only."""
        origin, optical = self._camera_pose()
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = origin
            camera.forward[:] = optical[:, 2]
            camera.up[:] = -optical[:, 1]
        self.renderer.enable_depth_rendering()
        try:
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

    def segmentation(self):
        """Render geom ids with the same camera; diagnostics/self-supervised labels only."""
        origin, optical = self._camera_pose()
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = origin
            camera.forward[:] = optical[:, 2]
            camera.up[:] = -optical[:, 1]
        self.renderer.enable_segmentation_rendering()
        try:
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def observe(self, measured: Velocity):
        return Frame(float(self.data.time), self.lidar(),
                     self.data.sensor("imu_quat").data.copy(), measured, self.image())

    def close(self):
        self.renderer.close()


def simulation_config(path=None, task_fixture=False, task_layout="finish-area", max_speed=None, max_reverse_speed=None):
    c = Config.load(path)
    c.base_height = .33
    c.camera_size = [320, 240]
    focal = 120. / np.tan(np.deg2rad(30.))
    c.camera_matrix = [[focal, 0., 160.], [0., focal, 120.], [0., 0., 1.]]
    c.distortion = [0., 0., 0., 0., 0.]
    c.body_from_camera = [[0., 0., 1., .24], [-1., 0., 0., 0.], [0., -1., 0., .10], [0., 0., 0., 1.]]
    # The MJB contains no camera. Mount this documented simulated front camera
    # 15deg downward so the ground ahead remains visible while climbing.
    camera_transform = np.asarray(c.body_from_camera)
    camera_transform[:3, :3] = Rotation.from_euler("y", 15., degrees=True).as_matrix() @ camera_transform[:3, :3]
    c.body_from_camera = camera_transform.tolist()
    if path is None:
        c.max_vx, c.max_vy, c.max_wz = .50, .08, .35
        c.passage_max_vx = .50
        c.max_reverse_vx = .25
    if max_speed is not None:
        c.max_vx = c.passage_max_vx = max_speed
    if max_reverse_speed is not None:
        c.max_reverse_vx = max_reverse_speed
    c.resolution = .10
    # Simulator Go2 dimensions are known. Use the body's oriented envelope in
    # DWA; a circumscribed disk plus the old 12cm margin sealed the start lane.
    # Unknown real hardware keeps its original conservative circular defaults.
    c.robot_radius, c.safety_margin = .23, .05
    c.footprint_half_length, c.footprint_half_width = .35, .23
    c.hazard_foot_half_length, c.hazard_foot_half_width = .28, .22
    c.body_clearance = .48  # Known simulated standing envelope plus head margin.
    # Official ramp measurements are about 12 degrees. This simulator-only
    # bound is backed by the sensor-driven MPC ascent regression; real hardware
    # keeps its calibrated/default terrain limit.
    c.max_slope_deg = 14. if path is None else c.max_slope_deg
    if path is None:
        c.water_detection = True
    if task_fixture and path is None:
        c.target_quota = 0 if task_layout == "perception" else 1  # Benchmark all output marks in the perception fixture.
        if task_layout == "near-start":
            c.mission_seconds, c.return_reserve = 20., 12.
        else:
            c.mission_seconds, c.return_reserve = 600., 60.
    c.validate()
    return c


def smoke_command(t, profile="smoke", max_speed=.5):
    if profile == "perception-survey":
        speed = max(0., min(max_speed, .35*(t-1.), .35*(14.-t)))
        if t < 15.:
            return Velocity(speed)
        return Velocity(wz=.40 if t < 31. else 0.)
    if profile == "perception-walk":
        speed = max(0., min(max_speed, .35*(t-1.), .35*(10.-t)))
        return Velocity(speed)
    if profile == "speed-smoke":
        # Acceleration and braking are time profiles, independent of plant
        # truth. This is a controller test, not autonomous navigation.
        speed = max(0., min(max_speed, .45*(t-1.), .45*(6.-t)))
        return Velocity(speed)
    if profile == "ramp-smoke":
        return Velocity(.14) if .6 <= t < 18. else Velocity()
    if t < 1.:
        return Velocity()
    if t < 4.:
        return Velocity(.15)
    if t < 6.:
        return Velocity(wz=.25)
    return Velocity()


class RuntimeLog:
    """Flush diagnostics to both the terminal and an inspectable JSONL file."""
    def __init__(self, path):
        self.file = Path(path).open("w")

    def emit(self, kind, timestamp, message, **details):
        record = {"kind": kind, "simulation_seconds": timestamp, "message": message, **details}
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()
        print(f"[go2 sim={timestamp:6.2f}s] {message}", flush=True)

    def close(self):
        self.file.close()


def close_viewer(viewer):
    """Keep model/image storage alive until the passive render thread exits.

    MuJoCo3.17 Handle.close() only requests exit. Its pinned Handle owns a
    weakref to Simulate, which expires after render_loop/destroy complete.
    Waiting here avoids GLFW teardown racing that daemon at Python exit.
    """
    viewer.clear_images()
    viewer.close()
    reference = getattr(viewer, "_sim", None)
    deadline = time.monotonic()+3.
    while reference is not None and reference() is not None and time.monotonic() < deadline:
        time.sleep(.01)


def finish_viewer(viewer, keep_open):
    """Review a frozen final frame without stepping physics or sending commands."""
    if viewer is None:
        return
    try:
        if keep_open and viewer.is_running():
            print("[go2] 仿真已停止，保留最终画面。关闭窗口或按 Ctrl+C 退出；不会继续物理步进。", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        close_viewer(viewer)


def camera_inset(front_bgr, viewport):
    """Reuse the sensor image as a live RGB overlay; no rendering/recording."""
    width = min(480, max(320, viewport.width//3), max(1, viewport.width-32), max(1, int((viewport.height-64)*4/3)))
    height = max(1, width*3//4)
    rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rgb, (0, 0), (width-1, min(21, height-1)), (15, 15, 15), -1)
    cv2.putText(rgb, "ROBOT FRONT CAMERA | 10 Hz", (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                .38, (245, 245, 245), 1, cv2.LINE_AA)
    return mujoco.MjrRect(viewport.left+max(0, viewport.width-width-16), viewport.bottom+16, width, height), rgb


def run_simulation(args):
    # Go2 uses tiny 3x3/18x18 systems. BLAS worker-pool dispatch costs more
    # than these solves and competes with the viewer. Restore limits on exit.
    from threadpoolctl import threadpool_limits
    old_cv_threads=cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            return _run_simulation(args)
    finally:
        cv2.setNumThreads(old_cv_threads)


def _run_simulation(args):
    root = Path(__file__).resolve().parents[2]
    pipe_ramp = getattr(args, "pipe_ramp", None) or ("entry" if args.task_fixture else "official")
    if pipe_ramp == "entry" and not args.task_fixture:
        raise ValueError("--pipe-ramp entry requires --task-fixture (a labelled local modified scene)")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Simulation output must be a new/empty directory")
    maximum=1800. if getattr(args,"coverage",False) else 600.
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > maximum:
        raise ValueError(f"--seconds must be in (0,{maximum:g}]")
    if not np.isfinite(args.spawn).all():
        raise ValueError("--spawn must contain finite initial coordinates")
    initial_yaw = getattr(args, "spawn_yaw", 0.)
    if not np.isfinite(initial_yaw):
        raise ValueError("--spawn-yaw must be finite")
    video = getattr(args, "video", False)
    video_fps = getattr(args,"video_fps",10.)
    if not np.isfinite(video_fps) or not 1. <= video_fps <= 30.:
        raise ValueError("--video-fps must be within 1..30")
    save_physics = getattr(args, "save_physics", False)
    snapshots = getattr(args, "snapshots", False)
    viewer_fps = getattr(args, "viewer_fps", 60.)
    if not np.isfinite(viewer_fps) or not 1 <= viewer_fps <= 120:
        raise ValueError("--viewer-fps must be in [1,120]")
    output.mkdir(parents=True, exist_ok=True)
    print(f"[go2] 加载场景和 MPC；运行上限 {args.seconds:g} 仿真秒；输出 {output.resolve()}", flush=True)
    task_layout = getattr(args, "task_layout", "finish-area")
    config = simulation_config(args.config, args.task_fixture, task_layout, getattr(args, "max_speed", None),
                               getattr(args, "max_reverse_speed", None))
    course_policy = getattr(args, "course_policy", "auto")
    survey_segment = getattr(args, "survey_segment", None)
    if survey_segment is not None and (not args.planned_survey or args.control != "autonomy"):
        raise ValueError("survey-segment requires planned autonomous survey")
    if course_policy == "auto":
        course_policy = "explore" if task_layout == "near-start" or args.control != "autonomy" else "ordered"
    if survey_segment is not None and course_policy != "ordered":
        raise ValueError("survey-segment requires the ordered course policy")
    if course_policy == "ordered" and args.control == "autonomy":
        if survey_segment is None and (not np.allclose(args.spawn, [.8, .8], atol=1e-6, rtol=0.) or abs(initial_yaw) > 1e-6):
            raise ValueError("Ordered full course must start at (0.8,0.8), yaw0; stage tests require --course-policy explore")
        default_route = "configs/local-entry-ramp-route.json" if pipe_ramp == "entry" else "configs/official-course-route.json"
        config.course_route = getattr(args, "course_route", None) or config.course_route or default_route
        if args.config is None:
            config.localization_metric = "point_to_line"
            config.imu_heading_reference = True
            config.allow_margin_escape = True
            # The U-exit is only about 0.62m wide. A 0.10m cell can consume
            # most of its remaining body clearance on the two boundary cells.
            config.course_portal_resolution = .05
            config.map_size = 30.
    else:
        config.course_route = ""
    if getattr(args, "coverage", False):
        if course_policy == "ordered" and not getattr(args,"course_route",None):
            config.course_route = "configs/perception-coverage-tour.json"
        config.navigation_support = True
        config.kinematic_velocity_frame="gravity_body"
        config.localization_metric = "point_to_line"
        config.imu_heading_reference = True
        config.min_points = 18
        config.ground_max_gap = .6
        config.mission_seconds = max(600., args.seconds+60.)
        config.target_quota = 0
        config.allow_margin_escape = True
    if getattr(args, "planned_survey", False):
        if not getattr(args, "coverage", False) or getattr(args, "known_route_diagnostic", False) or pipe_ramp != "official":
            raise ValueError("planned-survey requires coverage and excludes truth diagnostic")
        config.navigation_prior="configs/public-navigation-prior.npz"
        if not (root/config.navigation_prior).exists():
            raise ValueError("Build the explicit offline prior with scripts/build_public_navigation_prior.py first")
        import shutil
        snapshot=(output/'navigation_prior.npz').resolve()
        shutil.copyfile(root/config.navigation_prior,snapshot)
        config.navigation_prior=str(snapshot)
        surface = root/"configs/portal-surface-reference.npz"
        if surface.is_file():
            surface_snapshot = (output/surface.name).resolve()
            shutil.copyfile(surface,surface_snapshot)
            config.navigation_surface_reference = str(surface_snapshot)
        config.course_portal_resolution=0.
    if args.task_fixture:
        from .mujoco_scene import task_scene
        model = task_scene(root, output, tuple(args.spawn), task_layout,
                           save_model=save_physics or snapshots, pipe_ramp=pipe_ramp)
        scene_kind = "local_entry_ramp_plus_task_objects" if pipe_ramp == "entry" else "official_geometry_plus_local_task_objects"
    else:
        model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
        scene_kind = "official_mjb"
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [*args.spawn, .30]  # Initial reset only; never set root pose in the stepping loop.
    data.qpos[3:7] = np.roll(Rotation.from_euler("z", initial_yaw).as_quat(), 1)
    mujoco.mj_forward(model, data)
    controller = ConvexMPC(model, data, root)
    controller.raw_odometry = bool(getattr(args,"coverage",False))
    if getattr(args, "camera_size", None):
        # Profiling-only override; the competition baseline is 320x240.
        w_new, h_new = int(args.camera_size[0]), int(args.camera_size[1])
        k = config.camera_matrix
        sx, sy = w_new / 320., h_new / 240.
        config.camera_matrix = [[k[0][0] * sx, 0., k[0][2] * sx],
                                [0., k[1][1] * sy, k[1][2] * sy],
                                [0., 0., 1.]]
        config.camera_size = [w_new, h_new]
    if getattr(args, "perception_demo", False) and args.config is None and not args.baseline_localizer:
        config.localization_metric = "point_to_line"
        config.imu_heading_reference = True
    sensors = SimSensors(model, data, config, args.seed)
    autonomy = Autonomy(config)
    if survey_segment is not None:
        first, stop = survey_segment
        navigator = autonomy.known_map_navigator
        if not 0 <= first < stop <= len(navigator.points):
            raise ValueError("Invalid survey segment waypoint bounds")
        navigator.index = first
        navigator.points = navigator.points[:stop]
        autonomy.localization.declared_start_pose = Pose(*(np.asarray(args.spawn)-.8),initial_yaw)
    diagnostic_route = None
    if getattr(args, "known_route_diagnostic", False):
        if not getattr(args, "coverage", False) or args.control != "autonomy":
            raise ValueError("known-route diagnostic requires coverage and autonomy control wrapper")
        from .route_diagnostic import KnownRouteDiagnostic
        diagnostic_route = KnownRouteDiagnostic(root / config.course_route, data.qpos[:2].copy(), initial_yaw)
    if getattr(args, "coverage", False):
        autonomy.localization.use_local_submap = True
    ros2_bridge = None
    if getattr(args, "ros2", False):
        from .ros2_bridge import SimSocketClient
        ros2_bridge = SimSocketClient(port=int(getattr(args, "ros2_sim_port", 8765)))
    if getattr(args, "slope_perception", False) and not (
            args.perception_demo and config.navigation_support and args.control == "autonomy"
            and not args.ros2 and not getattr(args, "known_route_diagnostic", False)):
        raise ValueError("--slope-perception requires --perception-demo, navigation support (--coverage), and autonomy")
    if getattr(args, "outline_ball", False) and not getattr(args, "slope_perception", False):
        raise ValueError("--outline-ball requires --slope-perception")
    if getattr(args, "dense_rubble", False) and not getattr(args, "slope_perception", False):
        raise ValueError("--dense-rubble requires --slope-perception")
    if getattr(args, "multiview_rubble", False) and not getattr(args, "slope_perception", False):
        raise ValueError("--multiview-rubble requires --slope-perception")
    precision_model = getattr(args, "rubble_precision_model", None)
    single_view_model = getattr(args, "rubble_single_view", None)
    completion_model = getattr(args, "rubble_completion_model", None)
    region_mapping = getattr(args, "rubble_region_mapping", False)
    if region_mapping and (completion_model is None or getattr(args, "slope_perception", False)):
        raise ValueError("--rubble-region-mapping requires the footprint completion path")
    if (single_view_model is not None or completion_model is not None) and not getattr(args, "perception_demo", False):
        raise ValueError("The rubble camera models require --perception-demo")
    if completion_model is not None and single_view_model is None and getattr(args,"slope_perception",False):
        raise ValueError("The slope-perception comparison mode requires --rubble-single-view")
    if precision_model is not None and not getattr(args, "multiview_rubble", False):
        raise ValueError("--rubble-precision-model requires --multiview-rubble")
    if getattr(args, "hybrid_ball", False) and (
            not getattr(args, "slope_perception", False) or getattr(args, "outline_ball", False)):
        raise ValueError("--hybrid-ball requires --slope-perception and excludes --outline-ball")
    config.save(output / "config.json")
    demo = None
    if getattr(args, "perception_demo", False):
        from .perception_demo import PerceptionDemo
        default_demo_model="models/seg_rubble_scene.onnx" if config.navigation_support else "models/seg_low_obstacle_perception.onnx"
        demo_model = Path(args.low_obstacle_model or root / default_demo_model)
        autonomy.localization.use_local_submap = not args.baseline_localizer
        demo = PerceptionDemo(autonomy, demo_model, output, known_target_count=1 if args.task_fixture else None)
        demo.slope_perception = getattr(args, "slope_perception", False)
        demo.outline_ball = getattr(args, "outline_ball", False)
        demo.hybrid_ball = getattr(args, "hybrid_ball", False)
        demo.visual_height.dense = getattr(args, "dense_rubble", False)
        demo.visual_height.multiview = getattr(args, "multiview_rubble", False)
        if precision_model is not None:
            from .rubble_precision_layer import RubblePrecisionLayer
            demo.rubble_precision = RubblePrecisionLayer(precision_model, demo.vision.k, config.camera_size)
        if completion_model is not None and not demo.slope_perception:
            from .rubble_footprint import RubbleFootprint
            demo.rubble_footprint = RubbleFootprint(config,demo.vision.k,completion_model,
                                                   region_mapping=region_mapping)
        elif single_view_model is not None:
            from .rubble_single_view import SingleViewRubble
            demo.single_view_rubble = SingleViewRubble(single_view_model, demo.vision.k, config.camera_size)
            if completion_model is not None:
                from .rubble_completion import RubbleCompletion
                demo.single_view_rubble.completion = RubbleCompletion(
                    demo.vision.k,config.camera_size,completion_model,revise=True)
        paths = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (root / "src/go2_rescue").glob("*.py")}
        (output / "run_manifest.json").write_text(json.dumps({
            "source_sha256": paths, "model": str(demo_model), "model_sha256": hashlib.sha256(demo_model.read_bytes()).hexdigest(),
            "target_model_sha256": hashlib.sha256((root / "models/ball_proposal.onnx").read_bytes()).hexdigest()
                if (root / "models/ball_proposal.onnx").exists() else None,
            "rubble_precision_model_sha256": hashlib.sha256(precision_model.read_bytes()).hexdigest()
                if precision_model is not None else None,
            "rubble_single_view_sha256": hashlib.sha256(single_view_model.read_bytes()).hexdigest()
                if single_view_model is not None else None,
            "rubble_completion_model_sha256": hashlib.sha256(completion_model.read_bytes()).hexdigest()
                if completion_model is not None else None,
            "navigation_floor_model_sha256": hashlib.sha256((root/"models/navigation_floor.onnx").read_bytes()).hexdigest() if config.navigation_support else None,
            "navigation_prior_sha256": hashlib.sha256((root/config.navigation_prior).read_bytes()).hexdigest() if config.navigation_prior else None,
            "navigation_surface_reference_sha256": (
                hashlib.sha256((root/config.navigation_surface_reference).read_bytes()).hexdigest()
                if config.navigation_surface_reference else None),
            "source_scene_sha256": hashlib.sha256((root / "scene/mujoco/competition-course.mjb").read_bytes()).hexdigest(),
            "scene": scene_kind, "args": vars(args), "config": "config.json",
        }, indent=2, default=str) + "\n")
    heading_at_start = None
    coverage_last_progress = .6
    coverage_known = 0
    coverage_checkpoint = np.zeros(2)
    coverage_scan_start = None
    low_det = low_sha = None
    if getattr(args, "low_obstacle", False):
        from .low_obstacle import LowObstacleDetector
        low_path = Path(args.low_obstacle_model) if getattr(args, "low_obstacle_model", None) else root / "models/seg_low_obstacle_v8.onnx"
        if not low_path.exists():
            raise FileNotFoundError(f"low-obstacle model not found: {low_path}")
        low_det = LowObstacleDetector(str(low_path), config.camera_matrix, config.body_from_camera, config.base_height)
        low_sha = hashlib.sha256(low_path.read_bytes()).hexdigest()
        print(f"LOW_OBSTACLE enabled model={low_path} sha256={low_sha[:16]}", flush=True)
    if autonomy.course is not None:
        (output / "ordered-course.json").write_text(json.dumps(autonomy.course.definition, indent=2, ensure_ascii=False)+"\n")
    recorder = Recorder(output / "recording") if args.record else None
    overview = mujoco.Renderer(model, height=480, width=640) if video else None
    overview_camera = mujoco.MjvCamera()
    overview_camera.distance, overview_camera.azimuth, overview_camera.elevation = 3.8, 140., -30.
    writer = cv2.VideoWriter(str(output / "simulation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (960, 480)) if video else None
    if writer is not None and not writer.isOpened():
        raise RuntimeError("Could not open MuJoCo video encoder")
    viewer = None
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.opt.geomgroup[3] = 1
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4., 140., -30.
    initial = data.qpos[:3].copy()
    from .evaluation import SimulationEvaluator
    official_model = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    evaluator = SimulationEvaluator(model, official_model, initial[:2], initial_yaw)
    last_step = Step(Velocity(), controller_pose(), "warming_up", [])
    observations, all_events, history, command_history = 0, [], [], []
    control_times, nav_times, sensor_times, physics_times = [deque(maxlen=10000) for _ in range(4)]
    command, next_nav, next_video, next_physics = Velocity(), .6, 0., 0.
    next_viewer, front, viewer_syncs = 0., None, 0
    overlay_key, camera_updates = None, 0
    ticks, fell = 0, False
    start = time.monotonic()
    status, error, stop_reason = "time_limit", None, "simulation_time_limit"
    runtime = RuntimeLog(output / "runtime.jsonl")
    original_start = np.allclose(args.spawn, [.8, .8], atol=1e-6, rtol=0.)
    run_scope = "original_start" if original_start and survey_segment is None else "custom_start_stage_test"
    start_label = f"{'ORIGINAL' if original_start else 'STAGE'} START ({args.spawn[0]:g}, {args.spawn[1]:g})"
    scene_label = "LOCAL: RAMP AT ENTRY" if pipe_ramp == "entry" else "SOURCE COURSE"
    runtime.emit("startup", 0., f"{start_label} yaw={initial_yaw:g}; policy={course_policy}; scene={scene_kind} control={args.control}; limit={args.seconds:g}s",
                 scene=scene_kind, control=args.control, time_limit=args.seconds,
                 run_scope=run_scope, initial_world_xy=list(args.spawn), initial_yaw=initial_yaw, course_policy=course_policy)
    runtime.emit("limits", 0., f"前进上限={config.max_vx:g}m/s；倒退上限={config.max_reverse_vx:g}m/s；窄口前进上限={min(config.max_vx, config.passage_max_vx):g}m/s；实际指令仍由避障和制动检查决定",
                 max_forward_mps=config.max_vx, max_reverse_mps=config.max_reverse_vx,
                 max_passage_forward_mps=min(config.max_vx, config.passage_max_vx))
    runtime.emit("resources", 0., f"物理/MPC/导航=CPU；OpenGL={sensors.graphics_device}；视频={video} 原始录制={args.record} 物理归档={save_physics} 快照={snapshots}；窗口刷新上限={viewer_fps:g}Hz",
                 video=video, raw_recording=args.record, physics_archive=save_physics, snapshots=snapshots,
                 graphics_device=sensors.graphics_device)
    previous_state, next_progress, next_snapshot = None, 0., 20.
    lidar_times, image_times, terrain_times = [], [], []
    nav_counter = 0
    demo_truth = []
    showcase = None
    if demo is not None:
        from .perception_showcase import Showcase
        showcase = Showcase(output, model, official_model, args.spawn, initial_yaw)
    camera_every = max(1, int(getattr(args, "camera_every", 1) or 1))
    next_low, low_ms, low_last = 0., [], []
    low_frames = low_total = 0
    low_interval = float(getattr(args, "low_obstacle_every", .5) or .5)
    try:
        with (output / "trajectory.jsonl").open("w") as log, (output / "low_obstacle.jsonl").open("w") as lolog:
            while data.time < args.seconds - 1e-9:
                t = float(data.time)
                if viewer is not None and not viewer.is_running():
                    status = "viewer_closed"
                    stop_reason = "viewer_closed_by_user"
                    break
                if t + 1e-9 >= next_nav:
                    begin = time.monotonic()
                    begin_lidar = time.monotonic()
                    lidar_points = sensors.lidar()
                    lidar_times.append(time.monotonic() - begin_lidar)
                    nav_counter += 1
                    if nav_counter % camera_every == 0:
                        begin_image = time.monotonic()
                        image_bgr = sensors.image()
                        image_times.append(time.monotonic() - begin_image)
                    else:
                        image_bgr = None
                    frame = Frame(float(data.time), lidar_points,
                                  data.sensor("imu_quat").data.copy(),
                                  controller.measured_velocity(), image_bgr,
                                  support_points=controller.measured_support_points() if demo is not None or config.navigation_support else None)
                    controller.terrain_slope_limit_deg = max(15., float(config.slope_limit(last_step.pose.xy())))
                    begin_terrain = time.monotonic()
                    controller.observe_terrain(frame.points)
                    terrain_times.append(time.monotonic() - begin_terrain)
                    sensor_times.append(time.monotonic()-begin)
                    front = frame.image
                    begin = time.monotonic()
                    if ros2_bridge is not None:
                        ros2_bridge.publish(frame, controller.measured_velocity(), t,
                                             config.camera_matrix, frame.image, frame.points)
                        got = ros2_bridge.latest_command(timeout=2.0)
                        vel = Velocity(float(got[0]), float(got[1]), float(got[2])) if got else Velocity(0., 0., 0.)
                        last_step = Step(vel, controller_pose(), "ros2_external_command", [], "external ROS2 agent")
                        command = last_step.velocity
                    elif diagnostic_route is not None:
                            # Mapping/localization remain a shadow observer;
                            # neither their goals nor truth poses enter each
                            # other's state. Only this diagnostic drives MPC.
                            autonomy.navigation_support.prepare(frame, autonomy.localization.pose, autonomy.grid)
                            pose, points = autonomy.localization.update(frame)
                            autonomy.grid.update(points, pose)
                            controller.observe_support(frame)
                            from .localization import Localization
                            command = diagnostic_route.command(t, data.qpos[:2], Localization._imu_yaw(data.qpos[3:7]))
                            last_step = Step(command, pose, "known_route_diagnostic", [], "external route diagnostic")
                            last_step.velocity = command
                            last_step.state = "known_route_diagnostic"
                            last_step.reason = "external known route and simulator pose; NOT autonomous acceptance"
                            last_step.diagnostics["external_route"] = diagnostic_route.status
                            if diagnostic_route.index == len(diagnostic_route.points) or diagnostic_route.stalled:
                                status = "diagnostic_complete" if not diagnostic_route.stalled else "diagnostic_stalled"
                                stop_reason = status
                                break
                    elif args.control == "autonomy":
                        last_step = autonomy.step(frame)
                        if config.navigation_support:
                            # classify_ground consumes raw body-frame lidar
                            # points and performs the timestamp/IMU transform
                            # internally. Passing a pre-rotated cloud here
                            # applies roll/pitch twice in MPC terrain support.
                            selected=autonomy.navigation_support.classify_ground(frame.points,frame)
                            controller.observe_support(frame,frame.points[selected])
                        controller.observe_forward_motion(last_step.pose, frame.timestamp)
                        command = last_step.velocity
                    else:
                        command = smoke_command(t, args.control, config.max_vx)
                        if args.control in {"perception-walk", "perception-survey"}:
                            from .geometry import wrap
                            from .localization import Localization
                            heading = Localization._imu_yaw(frame.quaternion)
                            if heading_at_start is None:
                                heading_at_start = heading
                            if command.vx > 0:
                                command = Velocity(command.vx, 0., float(np.clip(
                                    1.8 * wrap(heading_at_start - heading), -.25, .25)))
                        last_step = Step(command, controller_pose(), f"mpc_{args.control}_profile", [], "scripted MPC test, not autonomous navigation")
                    if demo is not None:
                        pose = demo.observe(frame, sensors.last_lidar_metadata,
                                            estimated=last_step.pose if args.control == "autonomy" and ros2_bridge is None else None)
                        last_step.pose = pose
                        if autonomy.known_map_navigator is not None and frame.image is not None:
                            autonomy.known_map_navigator.avoid_confirmed_targets(demo.tracker.reported_positions)
                        if (demo.rubble_footprint is not None and autonomy.known_map_navigator is not None
                                and frame.image is not None):
                            footprint=demo.rubble_footprint
                            autonomy.known_map_navigator.replace_visual_rubble(
                                footprint.planning_centers(),footprint.revision.resolution)
                        elif (not demo.slope_perception and autonomy.known_map_navigator is not None
                                and frame.image is not None and demo.history):
                            row=demo.history[-1]
                            if abs(row['t']-frame.timestamp)<1e-6 and pose.quality>=config.min_pose_quality:
                                from .geometry import transform_xy
                                visual=[p for p,measured in row['rough'] if not measured and len(p)>=3]
                                if visual:
                                    local=np.vstack(visual)
                                    local=local[np.linalg.norm(local,axis=1)<2.]
                                    autonomy.known_map_navigator.observe_visual_rubble(transform_xy(local,pose),pose)
                        # Read-only evaluation record, downstream of all estimates.
                        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "local_football")
                        target = data.xpos[ball_id, :2].tolist() if ball_id >= 0 else None
                        demo_truth.append({"t": t, "base_world_xy": data.qpos[:2].tolist(), "base_world_xyz": data.qpos[:3].tolist(), "football_world_xy": target})
                        if frame.image is not None:
                            showcase.observe(frame, demo, sensors)
                    if getattr(args,"coverage",False) and diagnostic_route is None:
                        known = int(autonomy.grid.seen.sum())
                        if known > coverage_known+30 or np.linalg.norm(last_step.pose.xy()-coverage_checkpoint) > .20:
                            coverage_last_progress,coverage_known = t,known
                            coverage_checkpoint = last_step.pose.xy().copy()
                        # A bounded 360-degree observation scan is an active
                        # exploration action, not a deadlock. The controller
                        # caps one scan at 2*pi or 30 s; allow that scan to
                        # finish before the no-map/no-translation watchdog.
                        scanning = (last_step.reason == "search_scan_for_new_view"
                                    and abs(command.wz) > .01)
                        if scanning:
                            if coverage_scan_start is None:
                                coverage_scan_start = t
                            if t - coverage_scan_start <= 28.:
                                coverage_last_progress = t
                        else:
                            coverage_scan_start = None
                        if t-coverage_last_progress > 30.:
                            status,stop_reason = "stalled","coverage_no_mapping_or_translation_progress"
                            break
                    nav_times.append(time.monotonic() - begin)
                    if recorder:
                        recorder.write(frame)
                    all_events.extend(asdict(e) for e in last_step.events)
                    evaluator.update(data, last_step.events)
                    if last_step.state != previous_state or t >= next_progress:
                        slope = float(np.rad2deg(np.arctan(np.linalg.norm(controller.terrain_plane[:2]))))
                        runtime.emit("progress", t,
                                     f"state={last_step.state} reason={last_step.reason or '-'} "
                                     f"cmd=({command.vx:.2f},{command.vy:.2f},{command.wz:.2f}) "
                                     f"pose_quality={last_step.pose.quality:.2f} terrain_slope={slope:.1f}deg",
                                     state=last_step.state, reason=last_step.reason,
                                     command=asdict(command), pose_quality=last_step.pose.quality,
                                     observed_support_slope_deg=slope)
                        if getattr(args,"coverage",False):
                            names={"L1_ascent":"上坡区","L1_descent":"下坡区","S_curve":"S形通道","dead_end":"死胡同观察与退出","rubble":"碎石区","pipe_external_bypass":"管道外绕行","north_sector":"北侧区域","rubble_views":"碎石多视角"}
                            stage=autonomy.course.stage if autonomy.course else "自由探索"
                            if autonomy.known_map_navigator is not None:
                                stage=autonomy.known_map_navigator.status.get('stage','预规划巡查')
                            targets=len(demo.tracker.reported_positions) if demo else len(autonomy.tracker.reported_positions)
                            row={"t":t,"stage":names.get(stage,stage),"known_area_m2":float(autonomy.grid.seen.sum()*config.resolution**2),"targets":targets,"message":"运行中："+last_step.reason+"。已观测面积不是整场完成率；结束后单独验收。"}
                            (output/"coverage_status.js").write_text("window.coverageStatus("+json.dumps(row,ensure_ascii=True)+");",encoding="utf-8")
                        previous_state, next_progress = last_step.state, t + 2.
                    for event in last_step.events:
                        runtime.emit("event", t, f"event={event.kind} " + json.dumps(event.payload, ensure_ascii=False),
                                     event=asdict(event))
                    observations += 1
                    from .cli import step_record
                    record = step_record(last_step, t, nav_times[-1])
                    # Read-only diagnostics, kept outside Step/Autonomy inputs.
                    record["diagnostics_only_world_position"] = data.qpos[:3].tolist()
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    if low_det is not None and frame.image is not None and t + 1e-9 >= next_low:
                        begin_low = time.monotonic()
                        low_dets = low_det.detect(frame.image)
                        low_ms.append(time.monotonic() - begin_low)
                        low_frames += 1
                        low_total += len(low_dets)
                        low_last[:] = low_dets
                        lolog.write(json.dumps({
                            "t": t, "model_sha256": low_sha, "inference_ms": low_ms[-1] * 1000.,
                            "pose": asdict(last_step.pose),
                            "detections": [{"x": d.x, "y": d.y, "height": d.height,
                                            "quality": d.height_quality, "score": d.score}
                                           for d in low_dets]}, allow_nan=False) + "\n")
                        next_low = t + low_interval
                    if snapshots and t >= next_snapshot:
                        from .cli import save_map
                        save_map(autonomy, output)
                        if low_det is not None and low_last:
                            overlay = cv2.imread(str(output / "map.png"))
                            if overlay is not None:
                                rows = overlay.shape[0]
                                pose_l = last_step.pose
                                c_l, s_l = np.cos(pose_l.yaw), np.sin(pose_l.yaw)
                                for d in low_last:
                                    wx = pose_l.x + c_l * d.x - s_l * d.y
                                    wy = pose_l.y + s_l * d.x + c_l * d.y
                                    cell_l = autonomy.grid.cell((wx, wy))
                                    if autonomy.grid.inside(cell_l):
                                        y_img = rows - 1 - cell_l[0]
                                        color = (0, 255, 0) if d.height_quality == "measured" else (0, 165, 255)
                                        cv2.circle(overlay, (cell_l[1], y_img), 2, color, -1)
                                cv2.imwrite(str(output / "low_obstacle_overlay.png"), overlay)
                        (output / "progress.json").write_text(json.dumps({"simulation_seconds": t,
                            "state": last_step.state, "diagnostics": last_step.diagnostics,
                            "evaluation": evaluator.report(data, False, fell)}, indent=2)+"\n")
                        if "canvas" in locals():
                            cv2.imwrite(str(output / "live-preview.png"), canvas)
                        next_snapshot += 20.
                    next_nav += .1
                if ticks % 3 == 0:
                    begin = time.monotonic()
                    data.ctrl[:] = controller.update(t, command)
                    control_times.append(time.monotonic() - begin)
                # This is the sole state transition during a run. No qpos/qvel
                # overwrite, root forces, mocap following, or kinematic plant.
                begin = time.monotonic()
                mujoco.mj_step(model, data)
                physics_times.append(time.monotonic()-begin)
                ticks += 1
                if ticks % 5 == 0 or args.control == "speed-smoke":
                    evaluator.update(data)
                if save_physics and data.time + 1e-9 >= next_physics:
                    history.append(np.r_[data.time, data.qpos.copy()])
                    command_history.append(command.array())
                    next_physics += .1
                if writer is not None and data.time + 1e-9 >= next_video:
                    base = data.xpos[model.body("base_link").id]
                    overview_camera.lookat[:] = base
                    overview.update_scene(data, camera=overview_camera, scene_option=sensors.option)
                    canvas = np.zeros((480, 960, 3), np.uint8)
                    canvas[:, :640] = cv2.cvtColor(overview.render(), cv2.COLOR_RGB2BGR)
                    if front is not None:
                        canvas[:240, 640:] = front
                    lines = ["MuJoCo 3.17 | Convex MPC", start_label, f"scene: {scene_kind[:29]}",
                             f"t={data.time:.1f}s {last_step.state}",
                             f"cmd {command.vx:.2f} {command.vy:.2f} {command.wz:.2f}",
                             f"localization {last_step.pose.quality:.2f}",
                             "events: " + ",".join(e["kind"] for e in all_events[-4:]),
                             last_step.reason[:40]]
                    if autonomy.course is not None:
                        lines[2] = f"stage: {autonomy.course.stage} wp {autonomy.course.waypoint_index}"
                    if autonomy.known_map_navigator is not None:
                        state=autonomy.known_map_navigator.status
                        lines[2]=f"survey: {state.get('stage','-')} wp {state.get('index',0)}"
                    for i, text in enumerate(lines):
                        cv2.putText(canvas, text, (650, 265 + 27 * i), cv2.FONT_HERSHEY_SIMPLEX, .40, (220, 230, 220), 1, cv2.LINE_AA)
                    writer.write(canvas)
                    next_video += 1./video_fps
                now = time.monotonic()
                if viewer is not None and now >= next_viewer:
                    viewport = viewer.viewport
                    if viewport is None or not viewer.is_running():
                        status, stop_reason = "viewer_closed", "viewer_closed_by_user"
                        break
                    viewer.cam.lookat[:] = data.xpos[model.body("base_link").id]
                    phase = autonomy.course.stage if autonomy.course is not None else "exploration"
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      "Scene\nStart\nStage\nState\nSim time",
                                      f"{scene_label}\n{start_label}\n{phase}\n{last_step.state}\n{data.time:.1f}s"))
                    if front is not None and not getattr(args, "hide_camera", False):
                        key = (observations, viewport.width, viewport.height)
                        if key != overlay_key and viewport.width > 0 and viewport.height > 64:
                            viewer.set_images(camera_inset(front, viewport))
                            overlay_key, camera_updates = key, camera_updates+1
                    viewer.sync()
                    viewer_syncs += 1
                    next_viewer = now+1./viewer_fps
                rpy = Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_euler("xyz")
                if data.qpos[2] < .13 or max(abs(rpy[:2])) > .8 or not np.isfinite(data.qpos).all():
                    fell, status = True, "fell"
                    stop_reason = "fall_or_invalid_physics_state"
                    break
                if last_step.state in {"finished", "timed_out"}:
                    status = "completed" if last_step.state == "finished" else "timed_out"
                    stop_reason = "mission_finished" if last_step.state == "finished" else "mission_budget_exhausted"
                    if autonomy.known_map_navigator is not None and autonomy.known_map_navigator.unreachable:
                        status="partial_survey"
                        stop_reason="reachable_survey_finished_with_unvisited_viewpoints"
                    break
                if args.realtime:
                    delay = float(data.time) - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(min(.01, delay))
    except KeyboardInterrupt:
        status, stop_reason = "interrupted", "keyboard_interrupt"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        stop_reason = "exception"
        raise
    finally:
        # Navigation runs after a physics/render tick. Include its terminal
        # event in the video even when it falls between two 10 Hz render ticks.
        if "canvas" in locals():
            canvas[240:, 640:] = 0
            terminal_lines = ["MuJoCo 3.17 | Convex MPC", start_label, f"t={data.time:.1f}s {last_step.state}",
                              "events: " + ",".join(e["kind"] for e in all_events[-4:]),
                              f"cmd {command.vx:.2f} {command.vy:.2f} {command.wz:.2f}",
                              last_step.reason[:37]]
            if autonomy.course is not None:
                terminal_lines.append(f"stage: {autonomy.course.stage} wp {autonomy.course.waypoint_index}")
            for i, text in enumerate(terminal_lines):
                cv2.putText(canvas, text, (650, 270 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, .40, (220, 230, 220), 1, cv2.LINE_AA)
            for _ in range(10):
                writer.write(canvas)
        if writer is not None:
            writer.release()
        if viewer and status in {"failed", "interrupted", "viewer_closed"}:
            close_viewer(viewer)
        if overview is not None:
            overview.close()
        sensors.close()
        if recorder:
            recorder.close()
        if snapshots and "canvas" in locals():
            cv2.imwrite(str(output / "preview.png"), canvas)
        if save_physics:
            np.savez_compressed(output / "physics.npz", states=np.asarray(history), commands=np.asarray(command_history))
        if snapshots or getattr(args,"coverage",False):
            from .cli import save_map
            save_map(autonomy, output)
        demo_summary = demo.finish(float(data.time)) if demo is not None else None
        if demo_summary is not None:
            demo_summary["showcase"] = showcase.finish()
            (output / "evaluation_truth.json").write_text(json.dumps(demo_truth) + "\n")
        summary = {"status": status, "stop_reason": stop_reason, "perception_demo": demo_summary,
                   "run_scope": run_scope, "initial_yaw": initial_yaw, "survey_segment": survey_segment,
                   "course_policy": course_policy, "ordered_course": None if autonomy.course is None else autonomy.course.diagnostics(),
                   "task_layout": task_layout if args.task_fixture else None,
                   "mission_complete": last_step.state == "finished", "time_limit_seconds": args.seconds,
                   "error": error, "scene": scene_kind, "control": args.control,
                   "pipe_ramp": pipe_ramp, "modified_course_geometry": pipe_ramp != "official",
                   "physics": "MuJoCo rigid-body dynamics; torque actuators; mj_step only",
                   "controller": "go2-convex-mpc@1c63c6a762779887ab0431fd60db681dede6cb32",
                   "state_estimation": "joint/IMU contact-scheduled kinematics, no plant root pose input",
                   "mujoco": mujoco.__version__, "simulation_seconds": float(data.time),
                   "wall_seconds": time.monotonic() - start, "fell": fell,
                   "mpc_solves": controller.solves, "events": all_events, "mission_state": last_step.state,
                   "mpc_terrain_updates": controller.terrain_updates,
                   "observations": observations, "nav_p95_seconds": float(np.quantile(nav_times, .95)) if nav_times else None,
                   "recording": {"video": video, "raw": args.record, "physics": save_physics, "snapshots": snapshots},
                   "compute": {"physics": "CPU", "mpc": "CPU", "autonomy": "CPU", "blas_threads": 1,
                               "graphics_device": sensors.graphics_device, "viewer_syncs": viewer_syncs,
                               "front_camera_updates": camera_updates},
                   "timing_p95_seconds": {name: float(np.quantile(values, .95)) if values else None for name, values in
                       [("sensors_and_terrain", sensor_times), ("autonomy", nav_times), ("mpc_update", control_times), ("physics_step", physics_times)]},
                   "timing_breakdown_p95_seconds": {name: float(np.quantile(values, .95)) if values else None for name, values in
                       [("lidar", lidar_times), ("camera_render", image_times), ("terrain", terrain_times)]},
                   "final_map_resolution_m": config.resolution,
                   "ros2": ros2_bridge is not None,
                   "low_obstacle": {"enabled": low_det is not None, "model_sha256": low_sha,
                                    "inferences": len(low_ms), "frames": low_frames,
                                    "mean_ms": float(np.mean(low_ms) * 1000.) if low_ms else None,
                                    "p95_ms": float(np.quantile(low_ms, .95) * 1000.) if low_ms else None,
                                    "detections": low_total},
                   "mpc_qp_p95_seconds": float(np.quantile(controller.solve_times, .95)) if controller.solve_times else None,
                   "diagnostics_only": {"initial_world_position": initial.tolist(), "final_world_position": data.qpos[:3].tolist(),
                                        "displacement_m": float(np.linalg.norm(data.qpos[:2] - initial[:2]))},
                   "evaluation": evaluator.report(data, last_step.state == "finished", fell),
                   "official_scene_sha256": hashlib.sha256((root / "scene/mujoco/competition-course.mjb").read_bytes()).hexdigest(),
                   "not_a_competition_score": True}
        summary["full_course_pass"] = bool(pipe_ramp == "official" and original_start and args.control == "autonomy"
                                            and autonomy.course is not None and not autonomy.course.coverage_tour and autonomy.course.route_complete
                                            and summary["evaluation"]["ordered_traversal_verified"]
                                            and summary["evaluation"]["local_task_pass"])
        summary["known_route_diagnostic"] = diagnostic_route is not None
        summary["planned_public_map_survey"] = bool(getattr(args,"planned_survey",False))
        if autonomy.known_map_navigator is not None:
            summary["survey_route"] = autonomy.known_map_navigator.status
            summary["survey_adjusted_viewpoints"] = autonomy.known_map_navigator.adjusted
            summary["survey_unvisited_viewpoints"] = autonomy.known_map_navigator.unreachable
            summary["mission_complete"] = bool(autonomy.known_map_navigator.status.get('route_complete') and not autonomy.known_map_navigator.unreachable)
            summary["mission_scope"] = "public-map-prior survey; not unknown-layout exploration or full competition"
        if diagnostic_route is not None:
            summary["external_route"] = diagnostic_route.status
            summary["full_course_pass"] = False
            summary["mission_complete"] = False
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        runtime.emit("stop", float(data.time),
                     f"STOP: {stop_reason}; mission={last_step.state}; fell={fell}; "
                     f"events={[e['kind'] for e in all_events]}; results={output.resolve()}",
                     status=status, stop_reason=stop_reason, mission_complete=last_step.state == "finished", error=error)
        if stop_reason == "mission_finished":
            print("[go2] 任务正常完成：已发出 finish，结束自主控制。", flush=True)
        elif stop_reason == "simulation_time_limit":
            print("[go2] 已到 --seconds 仿真时间上限；这不代表任务完成，请查看 mission_state。", flush=True)
        runtime.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if viewer is not None and viewer.is_running():
        phase = autonomy.course.stage if autonomy.course is not None else "exploration"
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "Scene\nStart\nStage\nStopped", f"{scene_label}\n{start_label}\n{phase}\n{stop_reason}"))
        viewer.sync()
    finish_viewer(viewer, not getattr(args, "close_viewer_on_stop", False)
                  and status not in {"failed", "interrupted", "viewer_closed"})
    return summary


def controller_pose():
    from .types import Pose
    return Pose()
