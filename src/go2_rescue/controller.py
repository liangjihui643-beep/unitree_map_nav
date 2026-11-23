from __future__ import annotations

from collections import deque
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .geometry import rot2, wrap
from .localization import Localization
from .mapping import GridMap
from .perception import Perception
from .planning import DWA, Explorer, astar
from .tracking import Tracker
from .types import Event, Frame, Step, Velocity
from .hazards import WaterDetector
from .course import OrderedCourse


class Autonomy:
    def __init__(self, config: Config):
        self.c = config
        self.course = OrderedCourse(config.course_route) if config.course_route else None
        if self.course is not None:
            for region in self.course.definition.get("terrain_slope_regions", []):
                if region not in config.terrain_slope_regions:
                    config.terrain_slope_regions.append(region)
            config.validate()
        self.localization = Localization(config)
        self.grid = GridMap(config)
        self.known_map_navigator = None
        if config.navigation_prior:
            from .known_map_navigation import KnownMapNavigator
            self.known_map_navigator = KnownMapNavigator(config)
        self.navigation_support = None
        if config.navigation_support:
            from pathlib import Path
            from .navigation_support import NavigationSupport
            from .low_obstacle import LowObstacleDetector
            model = Path(__file__).resolve().parents[2] / "models/navigation_floor.onnx"
            self.navigation_support = NavigationSupport(config, LowObstacleDetector(
                model, config.camera_matrix, config.body_from_camera, config.base_height))
        self.corridor_global_grid = None
        self.corridor_keyframe_start = None
        self.perception = Perception(config)
        self.tracker = Tracker(config)
        self.explorer, self.dwa = Explorer(), DWA(config)
        self.state = "initializing"
        self.started: float | None = None
        self.goal: np.ndarray | None = None
        self.finish_goal: np.ndarray | None = None
        self.finish_marker: np.ndarray | None = None
        self.finish_observations = 0
        self.last_finish = -np.inf
        self.still_since: float | None = None
        self.last_progress = 0.
        self.progress_position = np.zeros(2)
        self.progress_yaw = 0.
        self.breadcrumbs = deque(maxlen=60)
        self.explored_empty_since: float | None = None
        self.verifying: set[int] = set()
        self.verify_until = -np.inf
        self.last_timestamp: float | None = None
        self.current_path = np.empty((0, 2))
        self.water = WaterDetector(config)
        self.hazard_observations = []
        # RGB-only keep-out regions are provisional. Repeated camera frames
        # from one stance are not independent evidence and must not become a
        # permanent hard obstacle through add_hazard()'s vote accumulator.
        self.rgb_hazard_candidates = []
        self.scan_start = None
        self.scan_yaw = None
        self.scan_angle = 0.
        self.scan_attempts = 0
        self.resume_goal = None
        self.recovery_until = -np.inf
        self.slow_since = None
        self.recovery_path = np.empty((0, 3))
        self.recovery_index = 0
        self.ordered_probe_goal = None
        self.ordered_probe_attempt = -np.inf
        self.ordered_no_path_since = None
        self.coverage_exploring = False
        self.margin_escape_active = False
        self.escape_dwa = DWA(replace(config, safety_margin=0., max_vx=min(config.max_vx, .05),
                                     max_reverse_vx=min(config.max_reverse_vx, .05),
                                     max_vy=min(config.max_vy, .05), max_wz=min(config.max_wz, .12)))
        # Only explicitly marked portals use this extra buffer. The oriented
        # physical body and foot/water envelope remain full size.
        self.passage_dwa = DWA(replace(config, safety_margin=min(config.safety_margin, .02),
                                      max_reverse_vx=min(config.max_reverse_vx, .15),
                                      max_vx=min(config.max_vx, config.passage_max_vx), max_vy=min(config.max_vy, .05),
                                      max_wz=min(config.max_wz, .15)))

    def _confirm_rgb_hazard(self, polygon, pose):
        """Require a second spatially distinct view before hard-blocking RGB-only cues."""
        center = np.asarray(polygon, dtype=float).mean(axis=0)
        position = pose.xy().copy()
        yaw = float(pose.yaw)
        cluster = None
        for item in self.rgb_hazard_candidates:
            if np.linalg.norm(item["center"] - center) <= .40:
                cluster = item
                break
        if cluster is None:
            self.rgb_hazard_candidates.append({
                "center": center, "views": [(position, yaw)], "confirmed": False})
            self.rgb_hazard_candidates = self.rgb_hazard_candidates[-100:]
            return False
        if cluster["confirmed"]:
            return False
        independent = any(
            np.linalg.norm(position - old_position) >= .25
            or abs(wrap(yaw - old_yaw)) >= np.deg2rad(25.)
            for old_position, old_yaw in cluster["views"])
        if not independent:
            return False
        cluster["views"].append((position, yaw))
        cluster["center"] = .5 * cluster["center"] + .5 * center
        cluster["confirmed"] = True
        return True

    def step(self, frame: Frame) -> Step:
        result = self._step(frame)
        result.diagnostics = {"goal": None if self.goal is None else self.goal.tolist(),
                              "measured_velocity": frame.velocity.array().tolist(),
                              "path_cells": len(self.current_path),
                              "ground_height": self.grid.ground_height,
                              "hazard_cells": int((self.grid.hazard_votes >= 2).sum()),
                              "reported_targets": len(self.tracker.reported_positions),
                              "registration": self.localization.last_registration,
                              "support": self.navigation_support.status if self.navigation_support else None,
                              "hazard_sources": getattr(self.water,"last_sources",[])}
        if self.course is not None:
            result.diagnostics["course"] = self.course.diagnostics()
            result.diagnostics["coverage_frontier_mode"] = self.coverage_exploring
        if self.known_map_navigator is not None:
            result.diagnostics["survey_route"] = self.known_map_navigator.status
        return result

    def ordered_step(self, frame, pose, points, events, dt):
        """Mandatory route owns goals; it cannot be overridden by exploration."""
        course = self.course
        target = course.target
        passage = bool(course.waypoint.get("narrow_passage")) and self.c.footprint_half_length > 0
        planner = self.passage_dwa if passage else self.dwa
        self.state = "ordered_route"
        required_step = course.stages[course.stage_index].get("required_step_height_m", 0.)
        if required_step > self.c.max_step_height:
            # Route intent is not evidence that the configured locomotion can
            # cross its curb. Never silently detour through a different pipe.
            self.state, self.goal = "blocked", target.copy()
            self.current_path = np.empty((0, 2))
            return Step(Velocity(), pose, self.state, events, "ordered_pipe2_requires_validated_step_capability")
        if course.waypoint.get("required_drop_height_m", 0.) > self.c.max_step_height:
            self.state, self.goal = "blocked", target.copy()
            return Step(Velocity(), pose, self.state, events, "ordered_pipe2_exit_requires_validated_descent_capability")
        if course.diagnostics()["awaiting_sensor_evidence"]:
            return Step(Velocity(), pose, self.state, events, "ordered_awaiting_sensor_evidence")
        if self.margin_escape_active:
            half = np.array([self.c.footprint_half_length, self.c.footprint_half_width])+self.c.safety_margin
            x, y = np.meshgrid(np.linspace(-half[0], half[0], 11), np.linspace(-half[1], half[1], 9))
            offsets = np.column_stack([x.ravel(), y.ravel()])
            if self.grid.footprint_safe(pose.xy(), pose.yaw, offsets):
                # Recovered clearance is the goal. Do not keep chasing tiny
                # interpolation points after safe normal operation is possible.
                self.recovery_path, self.recovery_index = np.empty((0, 3)), 0
                self.margin_escape_active = False
        if not self.margin_escape_active and not passage:
            route = self.explorer.margin_escape(self.grid, pose)
            if len(route):
                self.recovery_path, self.recovery_index = route, 1
                self.recovery_until = frame.timestamp+20.
                self.margin_escape_active = True
                events.append(Event("margin_escape", frame.timestamp, {"speed_limit_mps": .05,
                    "physical_body_checks": True, "water_contact_envelope_unchanged": True}))
        # Keep a previously planned safe SE(2) manoeuvre until it finishes.
        if self.recovery_index < len(self.recovery_path):
            while self.recovery_index < len(self.recovery_path):
                waypoint = self.recovery_path[self.recovery_index]
                position_tolerance = .012 if self.margin_escape_active else .11
                yaw_tolerance = .04 if self.margin_escape_active else .16
                if np.linalg.norm(waypoint[:2]-pose.xy()) >= position_tolerance or abs(wrap(waypoint[2]-pose.yaw)) >= yaw_tolerance:
                    break
                self.recovery_index += 1
            if self.recovery_index < len(self.recovery_path) and frame.timestamp < self.recovery_until:
                waypoint = self.recovery_path[self.recovery_index]
                planner = self.escape_dwa if self.margin_escape_active else self.dwa
                if self.margin_escape_active:
                    # The route's intermediate poses have already been checked.
                    # Millimetre-scale subgoals made the finite DWA horizon choose
                    # zero motion indefinitely; use the short exit endpoint.
                    waypoint = self.recovery_path[-1]
                command = planner.command(pose, frame.velocity, np.array([pose.xy(), waypoint[:2]]),
                    self.grid.obstacle_points(points, pose), self.grid, dt, desired_yaw=float(waypoint[2]), reverse_limit=.08)
                return Step(command, pose, "recovering", events, "ordered_margin_escape" if self.margin_escape_active else "ordered_se2_recovery")
            self.recovery_path, self.recovery_index = np.empty((0, 3)), 0
            self.margin_escape_active = False
            self.resume_goal = None
        near = course.checkpoint_near(pose)
        desired = course.waypoint.get("yaw")
        strict_heading = any(course.waypoint.get(k) for k in ["strict", "reverse", "inspect_front", "hold_seconds"])
        aligned = not strict_heading or desired is None or abs(wrap(desired-pose.yaw)) < course.waypoint.get("yaw_tolerance", .20)
        if near and aligned:
            return Step(Velocity(), pose, self.state, events, "ordered_checkpoint_hold")
        if near and desired is not None:
            # safe_turn checks a 1.5s sweep. Do not reject a reachable heading
            # because that sweep continues past the requested final angle.
            command = planner.safe_turn(pose, float(np.clip(wrap(desired-pose.yaw)/1.5, -.15 if passage else -.3, .15 if passage else .3)), self.grid)
            if not np.any(command.array()) and frame.timestamp-self.last_progress > 3.:
                route = self.explorer.maneuver(self.grid, pose, target, goal_yaw=desired,
                                              position_tolerance=course.waypoint.get("tolerance", .20))
                self.last_progress = frame.timestamp
                if len(route) > 1:
                    self.recovery_path, self.recovery_index = route, 1
                    self.recovery_until = frame.timestamp+30.
                    return Step(Velocity(), pose, "recovering", events, "ordered_plan_pose_alignment")
            return Step(command, pose, self.state, events, "ordered_align_checkpoint")
        self.current_path = course.local_path(self.grid, pose)
        # Intermediate goals returned by local_path are safe, but the best
        # cell can move left/right as sparse evidence changes. Hold that goal
        # until reached (or invalidated) instead of dithering between cells.
        direct_route = (len(self.current_path) > 0 and
                        np.linalg.norm(self.current_path[-1]-target) < .12)
        if direct_route:
            self.ordered_probe_goal = None
        else:
            if self.ordered_probe_goal is not None:
                distance = np.linalg.norm(self.ordered_probe_goal-pose.xy())
                if distance < .18:
                    self.explorer.arrived(self.grid,self.ordered_probe_goal,frame.timestamp)
                    self.ordered_probe_goal = None
                elif frame.timestamp-self.ordered_probe_attempt > 15.:
                    self.explorer.reject(self.grid,self.ordered_probe_goal,frame.timestamp)
                    self.ordered_probe_goal = None
            if self.ordered_probe_goal is None:
                if len(self.current_path):
                    self.ordered_probe_goal=self.current_path[-1].copy()
                    self.ordered_probe_attempt=frame.timestamp
                elif frame.timestamp-self.ordered_probe_attempt >= 8.:
                    self.ordered_probe_attempt=frame.timestamp
                    candidate=self.explorer.choose(self.grid,pose,frame.timestamp,toward=target)
                    if candidate is not None and np.linalg.norm(candidate-pose.xy()) > .24:
                        self.ordered_probe_goal=np.asarray(candidate,dtype=float).copy()
                        events.append(Event("ordered_frontier_probe",frame.timestamp,{
                            "goal_xy":self.ordered_probe_goal.tolist(),
                            "checkpoint_xy":target.tolist(),
                            "unknown_and_hazard_cells_remain_blocked":True}))
            if self.ordered_probe_goal is not None:
                path=astar(self.grid.traversable(),self.grid.cell(pose.xy()),
                           self.grid.cell(self.ordered_probe_goal))
                if path:
                    self.current_path=np.array([self.grid.world(cell) for cell in path])
                    self.goal=self.ordered_probe_goal.copy()
                    command=planner.command(pose,frame.velocity,self.current_path,
                        self.grid.obstacle_points(points,pose),self.grid,dt,reverse_limit=.08)
                    if np.linalg.norm(command.array())>1e-6:
                        return Step(command,pose,"ordered_frontier_probe",events,
                                    "ordered_safe_frontier_probe")
                    return Step(Velocity(),pose,"ordered_frontier_probe",events,
                                "ordered_frontier_probe_wait")
                self.explorer.reject(self.grid,self.ordered_probe_goal,frame.timestamp)
                self.ordered_probe_goal=None
        if not len(self.current_path):
            self.goal = target.copy()
            if course.coverage_tour and frame.timestamp-self.started > 20.:
                self.ordered_no_path_since = (frame.timestamp if self.ordered_no_path_since is None
                                              else self.ordered_no_path_since)
                if frame.timestamp-self.ordered_no_path_since >= 8.:
                    self.coverage_exploring = True
                    self.state = "exploring"
                    events.append(Event("coverage_route_deferred",frame.timestamp,{
                        "stage":course.stage,"waypoint_index":course.waypoint_index,
                        "checkpoint_xy":target.tolist(),
                        "reason":"no safe observed route for 8s",
                        "unknown_occupied_and_hazard_cells_remain_blocked":True,
                        "course_checkpoint_remains_unverified":True}))
                    return Step(Velocity(),pose,self.state,events,"coverage_switch_to_safe_frontiers")
            if self.navigation_support is not None and not course.waypoint.get("reverse"):
                # A fixed forward camera must first face an unobserved route.
                # Rotation still passes the full measured body/hazard check.
                direction = target-pose.xy()
                error = wrap(np.arctan2(direction[1],direction[0])-pose.yaw)
                if abs(error) > .15:
                    look = planner.safe_turn(pose,float(np.clip(error,-.25,.25)),self.grid)
                    if abs(look.wz) > .01:
                        return Step(look,pose,"scanning",events,"ordered_observe_route_heading")
            if frame.timestamp-self.last_progress > 5. and self.begin_recovery(pose, frame.timestamp):
                return Step(Velocity(), pose, "recovering", events, "ordered_plan_recovery")
            return Step(Velocity(), pose, self.state, events, "ordered_no_observed_route")
        self.ordered_no_path_since = None
        self.goal = self.current_path[-1].copy()
        heading = desired if strict_heading or np.linalg.norm(target-pose.xy()) < .45 else None
        planning_velocity = frame.velocity
        if course.waypoint.get("observed_corridor") and len(course.motion_window) >= 3:
            window = course.motion_window
            duration = window[-1][0]-window[0][0]
            if .19 <= duration <= .6:
                world_velocity = (window[-1][1]-window[0][1])/duration
                if np.linalg.norm(world_velocity) < .6:
                    body_velocity = rot2(pose.yaw).T@world_velocity
                    # Instantaneous stance odometry aliases the slow trot at
                    # 10Hz. Accepted scan displacement supplies the actual
                    # planning window, rather than alternating foot swings.
                    planning_velocity = Velocity(*body_velocity,frame.velocity.wz)
        command = planner.command(pose, planning_velocity, self.current_path,
            self.grid.obstacle_points(points, pose), self.grid, dt, desired_yaw=heading,
            reverse_limit=None if course.waypoint.get("reverse") else .08,
            forward_limit=course.waypoint.get("max_forward_mps"))
        if np.hypot(command.vx, command.vy) < .025 and abs(command.wz) < .06:
            self.slow_since = frame.timestamp if self.slow_since is None else self.slow_since
            if frame.timestamp-self.slow_since > 2. and self.begin_recovery(pose, frame.timestamp):
                return Step(Velocity(), pose, "recovering", events, "ordered_plan_recovery")
        else:
            self.slow_since = None
        reason = "ordered_local_planner_blocked" if np.linalg.norm(command.array()) < 1e-6 else "ordered_follow_checkpoint"
        return Step(command, pose, self.state, events, reason)

    def begin_recovery(self, pose, timestamp):
        if float(self.c.slope_limit(pose.xy())) > 15.:
            # Flat SE(2) recovery does not validate rotating/support changes
            # on the steep local ramp. Stop and refresh observations instead.
            return False
        route = np.empty((0, 3))
        resume = None
        if self.goal is not None:
            path = astar(self.grid.traversable(), self.grid.cell(pose.xy()), self.grid.cell(self.goal))
            if path:
                positions = np.array([self.grid.world(p) for p in path])
                distances = np.linalg.norm(positions-pose.xy(), axis=1)
                for advance in [1., .65, .4]:
                    far = np.flatnonzero(distances >= advance)
                    subgoal = positions[far[0]] if len(far) else self.goal
                    route = self.explorer.maneuver(self.grid, pose, subgoal)
                    if len(route):
                        break
                if len(route):
                    resume = self.goal.copy()
        if resume is None and self.goal is not None:
            self.explorer.reject(self.grid, self.goal, timestamp)
        if not len(route):
            route = self.explorer.maneuver(self.grid, pose)
        if not len(route):
            bay = self.explorer.turning_bay(self.grid, pose)
            if bay is None:
                return False
            path = astar(self.grid.traversable(), self.grid.cell(pose.xy()), self.grid.cell(bay))
            route = np.array([[*self.grid.world(p), pose.yaw] for p in path])
        self.recovery_path, self.recovery_index = route, 1
        self.resume_goal, self.goal = resume, route[-1, :2].copy()
        self.state, self.recovery_until = "recovering", timestamp+30.
        self.last_progress, self.slow_since = timestamp, None
        return True

    def _step(self, frame: Frame) -> Step:
        # Reject stale packets before localization, so they never contaminate
        # scan matching or advance mission time through the normal state machine.
        frame.validate()
        if frame.state_age > self.c.max_sensor_age:
            return Step(Velocity(), self.localization.pose, self.state, [], "stale_state")
        previous_time = self.last_timestamp
        if self.navigation_support is not None:
            self.navigation_support.observe(frame)
            self.localization.support_plane=self.navigation_support.surface.plane
            self.localization.visual_prior=self.navigation_support.visual_prior(frame)
            self.localization.ground_filter=lambda points:self.navigation_support.classify_ground(points,frame,points_frame="gravity")
        pose, points = self.localization.update(frame)
        if self.navigation_support is not None:
            self.navigation_support.accept_pose(frame,pose)
        self.last_timestamp = frame.timestamp
        dt = .1 if previous_time is None else frame.timestamp - previous_time
        if self.state in {"finished", "timed_out"}:
            return Step(Velocity(), pose, self.state, [], "terminal")
        if self.started is not None and frame.timestamp-self.started > self.c.mission_seconds:
            self.state = "timed_out"
            return Step(Velocity(), pose, self.state, [Event("timeout", frame.timestamp)], "budget_exhausted_without_finish")
        roll, pitch, _ = Rotation.from_quat(np.roll(frame.quaternion, -1)).as_euler("xyz")
        if max(abs(roll), abs(pitch)) > np.deg2rad(25):
            return Step(Velocity(), pose, self.state, [], "excessive_tilt")
        if pose.quality < self.c.min_pose_quality:
            if self.known_map_navigator is not None:
                plane = self.navigation_support.surface.plane if self.navigation_support is not None else None
                command = self.known_map_navigator.reacquire(frame,self.localization.sensor_heading,points,plane)
                if command is not None and abs(command.wz) > .01:
                    return Step(command,pose,self.state,[],"stationary_localization_scan")
            return Step(Velocity(), pose, self.state, [], "localization_unreliable")
        if self.localization.corrections:
            # Candidate centroids are in the old map frame. Drop unconfirmed
            # RGB evidence after a map correction; confirmed hazards are
            # already retained in hazard_observations and transformed below.
            self.rgb_hazard_candidates.clear()
            corrected_hazards = []
            for polygon in self.hazard_observations:
                center = polygon.mean(axis=0)
                old, new = min(self.localization.corrections, key=lambda pair: np.linalg.norm(pair[0].xy() - center))
                corrected_hazards.append((polygon - old.xy()) @ rot2(new.yaw - old.yaw).T + new.xy())
            self.hazard_observations = corrected_hazards
            self.grid.clear()
            keys = self.localization.keyframes if self.corridor_keyframe_start is None else self.localization.keyframes[self.corridor_keyframe_start:]
            if self.navigation_support is not None:
                self.navigation_support.rebuild(self.grid,keys,self.localization.corrections)
            else:
                for key in keys:
                    self.grid.update(key.points, key.pose, getattr(key, "quaternion", None), getattr(key, "timestamp", None))
            for polygon in self.hazard_observations:
                self.grid.add_hazard(polygon)
            self.tracker.correct_map(self.localization.corrections)
            # Reacquire the visual finish after map changes; don't reuse a goal
            # expressed in the old map frame.
            self.goal = self.finish_goal = self.finish_marker = None
            self.finish_observations = 0
            self.breadcrumbs.clear()
            # A stored SE(2) route belongs to the previous graph frame too.
            # Cancel it and replan in the corrected map; DWA must never keep
            # chasing old coordinates after a loop adjustment.
            self.recovery_path = np.empty((0, 3))
            self.recovery_index, self.resume_goal = 0, None
            self.margin_escape_active = False
            self.slow_since = self.scan_start = None
            if self.state in {"recovering", "verifying"}:
                self.state = "exploring"
        if self.navigation_support is not None:
            self.navigation_support.prepare(frame,pose,self.grid)
            self.grid.visual_ground_mask = self.navigation_support.classify_ground(
                points, frame, points_frame="gravity", allow_ramp=self.c.map_connected_ramp_ground)
        self.grid.update(points, pose, frame.quaternion, frame.timestamp)
        if self.navigation_support is not None:
            self.navigation_support.update(frame, pose, self.grid)
            if self.localization.keyframes:
                self.navigation_support.capture_keyframe(self.localization.keyframes[-1],frame,self.grid)
        self.explorer.observe(pose, frame.timestamp)
        events = []
        if self.started is None:
            self.started = frame.timestamp
            self.last_progress = frame.timestamp
            self.progress_position = pose.xy().copy()
            self.progress_yaw = pose.yaw
            self.state = "exploring"
            events.append(Event("start", frame.timestamp, {"coordinate_frame": "map_start"}))
        elapsed = frame.timestamp - self.started
        if elapsed > self.c.mission_seconds:
            self.state = "timed_out"
            events.append(Event("timeout", frame.timestamp))
            return Step(Velocity(), pose, self.state, events, "budget_exhausted_without_finish")
        if self.course is not None and self.known_map_navigator is None:
            if self.course.waypoint.get("observed_corridor") and self.corridor_keyframe_start is None:
                # A height map accumulated across ascent/descent carries a
                # drifting vertical reference. Start a measured local terrain
                # submap for this corridor; retain the global map, and never
                # mark unseen support free. Pose/route/target state is unchanged.
                self.corridor_global_grid = self.grid
                self.corridor_keyframe_start = max(0,len(self.localization.keyframes)-1)
                self.grid = GridMap(self.c)
                self.grid.update(points, pose, frame.quaternion, frame.timestamp)
                for polygon in self.hazard_observations:
                    self.grid.add_hazard(polygon)
                events.append(Event("corridor_terrain_submap_started",frame.timestamp,{
                    "source":"current and subsequent sensor observations",
                    "global_map_retained":True,"unknown_support_blocked":True}))
            course_events = self.course.update(frame, pose, self.grid.ground_height)
            events.extend(course_events)
            if any(e.kind == "course_waypoint" for e in course_events):
                self.recovery_path, self.recovery_index = np.empty((0, 3)), 0
                self.margin_escape_active = False
            resolution = self.c.course_portal_resolution
            if self.course.stage == "rubble" and 0 < resolution < self.c.resolution:
                old_resolution = self.c.resolution
                self.c.resolution = resolution
                self.escape_dwa.c.resolution = self.passage_dwa.c.resolution = resolution
                # Rebuild from stored sensor observations and estimated poses.
                # Fine cells are needed at the flat portal, whereas the ramp's
                # sparse ground returns benefit from the coarser terrain grid.
                self.grid = GridMap(self.c)
                if self.navigation_support is not None:
                    self.navigation_support.rebuild(self.grid,self.localization.keyframes)
                else:
                    for key in self.localization.keyframes:
                        self.grid.update(key.points, key.pose, getattr(key, "quaternion", None), getattr(key, "timestamp", None))
                        key.scan=self.localization.structural_scan(key.points,key.pose)
                self.grid.update(points, pose, frame.quaternion, frame.timestamp)
                for polygon in self.hazard_observations:
                    self.grid.add_hazard(polygon)
                self.recovery_path, self.recovery_index = np.empty((0, 3)), 0
                self.margin_escape_active = False
                events.append(Event("course_map_refined", frame.timestamp, {
                    "from_m": old_resolution, "to_m": resolution,
                    "source": "recorded keyframe observations and estimated poses"}))
        marker_visible = False
        detections = []
        if frame.image is not None and frame.image_age <= self.c.max_sensor_age:
            image = self.perception.undistort(frame.image)
            polygons = self.water.detect(image, frame, pose, self.grid)
            sources = getattr(self.water, "last_sources", [])
            for index, polygon in enumerate(polygons):
                source = sources[index] if index < len(sources) else "confirmed_sensor_hazard"
                if source == "rgb_stance_keepout_candidate":
                    if not self._confirm_rgb_hazard(polygon, pose):
                        continue
                self.grid.add_hazard(polygon)
                self.hazard_observations.append(polygon)
            self.hazard_observations = self.hazard_observations[-200:]
            if self.course is None or self.course.stage == "finish_search":
                detections = [self.perception.locate_ball(d, frame, pose) for d in self.perception.detect_balls(image)]
                events.extend(self.tracker.update(detections, pose, frame.timestamp))
            finish = self.perception.finish(image, frame, pose)
            if finish is not None:
                self.finish_marker, observed_goal = finish
                self.finish_goal = observed_goal if self.finish_goal is None else .8 * self.finish_goal + .2 * observed_goal
                self.finish_observations += 1
                self.last_finish, marker_visible = frame.timestamp, True
        else:
            # Also expire tracks during camera gaps, without manufacturing hits.
            self.tracker.update([], pose, frame.timestamp)
        if not self.breadcrumbs or np.linalg.norm(pose.xy() - self.breadcrumbs[-1]) > .35:
            self.breadcrumbs.append(pose.xy().copy())
        if np.linalg.norm(pose.xy() - self.progress_position) > .12 or abs(wrap(pose.yaw - self.progress_yaw)) > .25:
            self.progress_position, self.last_progress = pose.xy().copy(), frame.timestamp
            self.progress_yaw = pose.yaw
        if self.known_map_navigator is not None:
            self.known_map_navigator.sync_hazards(self.grid)
            result = self.known_map_navigator.step(frame,pose,points,events,dt,
                self.navigation_support.surface.plane if self.navigation_support else None)
            self.goal=self.known_map_navigator.goal
            self.current_path=self.known_map_navigator.path
            self.state=result.state
            return result
        if self.course is not None and not self.course.route_complete and not self.coverage_exploring:
            return self.ordered_step(frame, pose, points, events, dt)
        quota_met = self.c.target_quota > 0 and len(self.tracker.reported_positions) >= self.c.target_quota
        returning = elapsed >= self.c.mission_seconds - self.c.return_reserve or quota_met
        finish_reachable = self.finish_goal is not None and bool(astar(self.grid.traversable(), self.grid.cell(pose.xy()), self.grid.cell(self.finish_goal)))
        if self.course is not None and not self.course.coverage_tour and not quota_met:
            # Mandatory traversal is finished; search here rather than restarting
            # global exploration or declaring finish without the required target.
            self.state = "ordered_search"
            turn = self.dwa.safe_turn(pose, .20, self.grid)
            return Step(turn, pose, self.state, events, "ordered_search_finish_target")
        if returning and finish_reachable and self.state != "recovering":
            self.state, self.goal = "to_finish", self.finish_goal.copy()
        elif self.state == "to_finish" and not finish_reachable:
            # Seeing a marker through an impassable opening does not establish
            # a route. Continue exploring to find a reachable entrance.
            self.state, self.goal = "exploring", None
        if self.state == "to_finish":
            self.goal = None if self.finish_goal is None else self.finish_goal.copy()
            near_finish = self.goal is not None and np.linalg.norm(self.goal - pose.xy()) < .18
            if near_finish and self.finish_marker is not None and not marker_visible:
                # Once at the standoff, navigation's path heading is ambiguous.
                # Face the previously observed marker to reacquire it, rather
                # than circling a nearly coincident positional goal indefinitely.
                direction = self.finish_marker - pose.xy()
                error = wrap(np.arctan2(direction[1], direction[0]) - pose.yaw)
                self.still_since = None
                turn = self.dwa.safe_turn(pose, float(np.clip(1.2*error, -.3, .3)), self.grid)
                return Step(turn, pose, self.state, events, "finish_reacquire")
            if near_finish and marker_visible and self.finish_observations >= 3:
                stopped = np.linalg.norm(frame.velocity.array()) < .06
                self.still_since = frame.timestamp if stopped and self.still_since is None else self.still_since
                if not stopped:
                    self.still_since = None
                if self.still_since is not None and frame.timestamp - self.still_since >= 1.:
                    self.state = "finished"
                    events.append(Event("finish", frame.timestamp, {"method": "local_aruco_standoff", "marker_id": self.c.finish_marker_id}))
                return Step(Velocity(), pose, self.state, events, "finish_confirmation")
            self.still_since = None
        if self.state == "verifying" and (frame.timestamp > self.verify_until or self.goal is None or np.linalg.norm(self.goal - pose.xy()) < .1):
            self.state, self.goal = "exploring", None
        if self.state == "recovering":
            while self.recovery_index < len(self.recovery_path):
                waypoint = self.recovery_path[self.recovery_index]
                if np.linalg.norm(waypoint[:2]-pose.xy()) >= .11 or abs(wrap(waypoint[2]-pose.yaw)) >= .18:
                    break
                self.recovery_index += 1
            if self.recovery_index >= len(self.recovery_path) or frame.timestamp > self.recovery_until:
                timed_out = self.recovery_index < len(self.recovery_path)
                if timed_out and self.resume_goal is not None:
                    self.explorer.reject(self.grid, self.resume_goal, frame.timestamp)
                self.state, self.goal, self.slow_since = "exploring", None if timed_out else self.resume_goal, None
                self.resume_goal = None
            else:
                waypoint = self.recovery_path[self.recovery_index]
                short_path = np.array([pose.xy(), waypoint[:2]])
                command = self.dwa.command(pose, frame.velocity, short_path, self.grid.obstacle_points(points, pose),
                                           self.grid, dt, desired_yaw=float(waypoint[2]))
                return Step(command, pose, self.state, events, "follow_observed_se2_recovery")
        if self.state in {"exploring", "waiting_finish"}:
            pending = [t for t in self.tracker.tracks if not t.reported and t.id not in self.verifying and t.hits >= 2]
            if pending and not returning:
                # Try a small lateral viewpoint only through known supported free
                # space. The path planner below rejects blocked candidates.
                self.verifying.add(pending[0].id)
                self.goal = pose.xy() + rot2(pose.yaw) @ np.array([0., .25])
                self.state, self.verify_until = "verifying", frame.timestamp + 2.
            elif self.goal is None or np.linalg.norm(self.goal - pose.xy()) < .18:
                if self.goal is not None:
                    self.explorer.arrived(self.grid, self.goal, frame.timestamp)
                # After a coverage checkpoint is temporarily unreachable,
                # keep frontier exploration biased toward that checkpoint.
                # Purely global information-gain scoring repeatedly revisits
                # the already mapped start bay and never returns to the route.
                toward = (self.course.target if self.course is not None
                          and self.course.coverage_tour else None)
                self.goal = self.explorer.choose(self.grid, pose, frame.timestamp, toward=toward)
                if self.goal is None:
                    if self.explored_empty_since is None:
                        self.explored_empty_since = frame.timestamp
                    if finish_reachable and frame.timestamp - self.explored_empty_since > 2.:
                        self.state, self.goal = "to_finish", self.finish_goal.copy()
                    elif frame.timestamp - self.explored_empty_since > 2.:
                        self.state = "waiting_finish"
                else:
                    self.explored_empty_since = None
                    self.state = "exploring"
        if self.goal is None:
            if self.scan_start is None or (frame.timestamp-self.scan_start > 35. and self.scan_attempts < 3):
                self.scan_start, self.scan_yaw, self.scan_angle = frame.timestamp, pose.yaw, 0.
                self.scan_attempts += 1
            self.scan_angle += abs(wrap(pose.yaw-self.scan_yaw))
            self.scan_yaw = pose.yaw
            if self.scan_angle < 2*np.pi and frame.timestamp-self.scan_start < 30.:
                turn = self.dwa.safe_turn(pose, .25, self.grid)
                if turn == Velocity():
                    turn = self.dwa.safe_turn(pose, -.25, self.grid)
                if turn != Velocity():
                    return Step(turn, pose, "scanning", events, "search_scan_for_new_view")
            self.begin_recovery(pose, frame.timestamp)
            return Step(Velocity(), pose, self.state, events, "no_reachable_frontier_or_finish")
        self.scan_start = None
        free = self.grid.traversable()
        start, end = self.grid.cell(pose.xy()), self.grid.cell(self.goal)
        path = astar(free, start, end)
        if not path or frame.timestamp - self.last_progress > 5.:
            if self.state != "recovering":
                if self.begin_recovery(pose, frame.timestamp):
                    return Step(Velocity(), pose, self.state, events, "planned_observed_se2_recovery")
            self.explorer.reject(self.grid, self.goal, frame.timestamp)
            self.goal = None
            self.last_progress = frame.timestamp
            self.state = "exploring" if self.state != "to_finish" else "to_finish"
            path = []
            # Backtrack to the most recent reachable breadcrumb. Never issue an
            # open-loop reverse command based only on a stuck timer.
            for point in reversed(list(self.breadcrumbs)[:-2]):
                if np.linalg.norm(point - pose.xy()) < .25:
                    continue
                backtrack = astar(free, start, self.grid.cell(point))
                if backtrack:
                    path, self.goal = backtrack, point.copy()
                    break
            if not path:
                self.current_path = np.empty((0, 2))
                return Step(Velocity(), pose, self.state, events, "blocked_replanning")
        self.current_path = np.array([self.grid.world(p) for p in path])
        obstacles = self.grid.obstacle_points(points, pose)
        command = self.dwa.command(pose, frame.velocity, self.current_path, obstacles, self.grid, dt, face_path=self.state != "recovering")
        if np.hypot(command.vx, command.vy) < .025 and abs(command.wz) < .06:
            self.slow_since = frame.timestamp if self.slow_since is None else self.slow_since
            if frame.timestamp-self.slow_since > 2. and self.begin_recovery(pose, frame.timestamp):
                return Step(Velocity(), pose, self.state, events, "planned_observed_se2_recovery")
        else:
            self.slow_since = None
        reason = "local_planner_no_safe_motion" if np.linalg.norm(command.array()) < 1e-6 else ""
        return Step(command, pose, self.state, events, reason)
