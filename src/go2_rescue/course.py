"""Ordered fixed-map mission prior; online inputs are estimated pose/sensors only.

This module deliberately has no simulator imports, geom IDs or world-state API.
The explicit route is a manually authored prior for this public map, not an
unknown-layout exploration solution or the official competition I/O contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial.transform import Rotation

from .geometry import wrap
from .planning import astar
from .types import Event

STAGES = ["L1_ascent", "L1_descent", "S_curve", "dead_end", "rubble", "pipe", "finish_search"]


class OrderedCourse:
    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        self.definition = json.loads(path.read_text())
        self.stages = self.definition.get("stages", [])
        self.coverage_tour = self.definition.get("schema") == "go2-coverage-tour/1"
        if self.definition.get("frame") != "map_start" or (not self.coverage_tour and [s.get("name") for s in self.stages] != STAGES):
            raise ValueError("Ordered course requires map_start and the complete mandatory stage order")
        for stage in self.stages:
            if not stage.get("waypoints"):
                raise ValueError("Every mandatory stage needs explicit waypoints")
            for waypoint in stage["waypoints"]:
                xy = np.asarray(waypoint.get("xy"))
                values = [waypoint.get("tolerance", .20), waypoint.get("hold_seconds", 0.), waypoint.get("yaw", 0.)]
                if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(values).all():
                    raise ValueError("Invalid course waypoint")
                if not .08 <= values[0] <= .35 or values[1] < 0:
                    raise ValueError("Invalid course tolerance/hold")
                width = waypoint.get("gate_half_width", .32)
                if not np.isfinite(width) or not .1 <= width <= .6:
                    raise ValueError("Invalid course transit gate width")
                for key in ("max_forward_mps", "required_drop_height_m"):
                    value = waypoint.get(key, 0.)
                    if not np.isfinite(value) or value < 0 or (key in waypoint and key == "max_forward_mps" and value == 0):
                        raise ValueError("Invalid course speed/drop capability")
        self.stage_index = self.waypoint_index = 0
        self.completed = []
        self.route_complete = False
        self.entered = False
        self.hold_since = None
        self.start_height = None
        self.min_height, self.max_height = np.inf, -np.inf
        self.max_pitch = 0.
        self.roof_observed = self.front_observed = False
        self.last_pose = None
        self.distance = 0.
        self.completed_waypoints = 0
        self.entry_xy = None
        self.front_confirmations = 0
        self.motion_window = deque(maxlen=10)

    @property
    def stage(self):
        return self.stages[self.stage_index]["name"]

    @property
    def waypoint(self):
        points = self.stages[self.stage_index]["waypoints"]
        return points[min(self.waypoint_index, len(points)-1)]

    @property
    def target(self):
        return np.asarray(self.waypoint["xy"], dtype=float)

    def checkpoint_near(self, pose):
        # Once stopped-inspection hold has begun, millimetre lidar jitter at
        # the arrival boundary must not restart it or issue another move.
        margin = .02 if self.hold_since is not None and self.waypoint.get("inspect_front") else 0.
        return np.linalg.norm(self.target-pose.xy()) <= self.waypoint.get("tolerance", .20)+margin

    def evidence(self):
        if self.coverage_tour:
            return True
        if self.start_height is None:
            return False
        if self.stage == "L1_ascent":
            return self.max_height-self.start_height >= .25 and self.max_pitch >= np.deg2rad(4.)
        if self.stage == "L1_descent":
            return self.start_height-self.min_height >= .22 and self.max_pitch >= np.deg2rad(4.)
        if self.stage == "dead_end":
            return self.front_observed
        if self.stage == "pipe":
            stage = self.stages[self.stage_index]
            return (self.roof_observed and self.distance >= stage.get("required_distance_m", 1.)
                    and self.max_height-self.min_height >= stage.get("required_height_gain_m", 0.))
        return True

    def update(self, frame, pose, ground_height):
        events = []
        if self.route_complete:
            return events
        if not self.entered:
            self.entered = True
            self.start_height = ground_height
            self.min_height = self.max_height = ground_height
            self.entry_xy = pose.xy().copy()
            events.append(Event("course_stage_enter", frame.timestamp, {"stage": self.stage}))
        self.min_height, self.max_height = min(self.min_height, ground_height), max(self.max_height, ground_height)
        _, pitch, _ = Rotation.from_quat(np.roll(frame.quaternion, -1)).as_euler("xyz")
        self.max_pitch = max(self.max_pitch, abs(pitch))
        if self.last_pose is not None:
            self.distance += min(.2, np.linalg.norm(pose.xy()-self.last_pose))
        self.last_pose = pose.xy().copy()
        self.motion_window.append((frame.timestamp, pose.xy().copy(),
                                   np.r_[Rotation.from_euler("z", pose.yaw).as_matrix()[:2, :2] @ frame.velocity.array()[:2],
                                         frame.velocity.wz]))
        while self.motion_window and frame.timestamp-self.motion_window[0][0] > .45:
            self.motion_window.popleft()
        p = frame.points
        roof = (np.linalg.norm(p[:, :2], axis=1) < .8)&(p[:, 2] > .18)&(p[:, 2] < .9)
        self.roof_observed |= int(roof.sum()) >= 4
        waypoint = self.waypoint
        near = self.checkpoint_near(pose)
        strict_heading = any(waypoint.get(k) for k in ["strict", "reverse", "inspect_front", "hold_seconds"])
        aligned = not strict_heading or "yaw" not in waypoint or abs(wrap(waypoint["yaw"]-pose.yaw)) < waypoint.get("yaw_tolerance", .20)
        previous = self.entry_xy if self.waypoint_index == 0 else np.asarray(
            self.stages[self.stage_index]["waypoints"][min(self.waypoint_index-1, len(self.stages[self.stage_index]["waypoints"])-1)]["xy"])
        segment = self.target-previous
        length = np.linalg.norm(segment)
        passed = False
        # Transit points are gates, not mandatory centimeter-precise landing
        # spots. A safe water detour can pass beside one. Inspection, reversing,
        # alignment and narrow-portal checkpoints stay strict.
        if length > .25 and not any(waypoint.get(k) for k in ["strict", "inspect_front", "reverse", "hold_seconds"]):
            direction = segment/length
            relative = pose.xy()-previous
            cross_track = abs(relative[0]*direction[1]-relative[1]*direction[0])
            passed = bool(relative@direction >= length and cross_track <= waypoint.get("gate_half_width", .32)
                          and np.linalg.norm(self.target-pose.xy()) < .70)
        if waypoint.get("inspect_front") and aligned and np.linalg.norm(self.target-pose.xy()) < .45:
            front = (p[:, 0] > .35)&(p[:, 0] < 1.15)&(abs(p[:, 1]) < .45)&(p[:, 2] > -.12)&(p[:, 2] < .6)
            self.front_confirmations = self.front_confirmations+1 if int(front.sum()) >= 4 else 0
            if self.front_confirmations >= 3 and not self.front_observed:
                self.front_observed = True
                events.append(Event("dead_end_confirmed", frame.timestamp, {"source": "consecutive frontal lidar observations",
                                    "observations": self.front_confirmations, "frame": "body"}))
        elif not self.front_observed:
            self.front_confirmations = 0
        hold = waypoint.get("hold_seconds", 0.)
        stopped = np.linalg.norm(frame.velocity.array()) < .06
        if waypoint.get("inspect_front"):
            # Trotting-in-place produces alternating joint-odometry spikes.
            # Require a short stable estimated-pose window AND small mean
            # measured motion, rather than restarting hold on every spike.
            motion = np.array([v[2] for v in self.motion_window])
            displacement = np.array([v[1]-pose.xy() for v in self.motion_window])
            mean = motion.mean(axis=0)
            stopped = (len(motion) >= 3 and frame.timestamp-self.motion_window[0][0] >= .19
                       and np.linalg.norm(mean[:2]) < .06 and abs(mean[2]) < .12
                       and np.linalg.norm(displacement, axis=1).max() < .025)
        ready = (near and aligned or passed) and (not hold or stopped)
        if waypoint.get("inspect_front"):
            ready &= self.front_observed
        if ready:
            self.hold_since = frame.timestamp if self.hold_since is None else self.hold_since
        else:
            self.hold_since = None
        if self.waypoint_index < len(self.stages[self.stage_index]["waypoints"]) and ready and frame.timestamp-self.hold_since >= hold:
            events.append(Event("course_waypoint", frame.timestamp, {"stage": self.stage, "index": self.waypoint_index,
                "frame": "map_start", "xy": self.target.tolist(), "method": "passed_gate" if passed else "reached_pose"}))
            self.waypoint_index += 1
            self.completed_waypoints += 1
            self.hold_since = None
        if self.waypoint_index == len(self.stages[self.stage_index]["waypoints"]) and self.evidence():
            events.append(Event("course_stage_complete", frame.timestamp, {"stage": self.stage,
                "observed_height_range_m": [float(self.min_height), float(self.max_height)],
                "max_imu_pitch_deg": float(np.rad2deg(self.max_pitch)), "roof_observed": bool(self.roof_observed),
                "front_observed": bool(self.front_observed), "estimated_distance_m": float(self.distance)}))
            self.completed.append(self.stage)
            if self.stage_index == len(self.stages)-1:
                self.route_complete = True
                events.append(Event("course_route_complete", frame.timestamp, {"stages": self.completed.copy()}))
            else:
                self.stage_index += 1
                self.waypoint_index = 0
                self.entered = False
                self.hold_since = self.last_pose = None
                self.distance, self.max_pitch = 0., 0.
                self.roof_observed = self.front_observed = False
                self.front_confirmations = 0
                self.motion_window.clear()
        return events

    def diagnostics(self):
        return {"stage": self.stage, "waypoint_index": self.waypoint_index,
                "front_observed": bool(self.front_observed), "front_confirmations": self.front_confirmations,
                "waypoint_count": len(self.stages[self.stage_index]["waypoints"]),
                "target": self.target.tolist(), "completed_stages": self.completed.copy(),
                "route_complete": self.route_complete, "awaiting_sensor_evidence": bool(
                    self.waypoint_index == len(self.stages[self.stage_index]["waypoints"]) and not self.evidence())}

    def local_path(self, grid, pose):
        """Reach the configured target through observed support, never clear it.

        A not-yet-observed waypoint is a direction prior. Advance only to a
        supported, footprint-checked intermediate goal on the current map.
        Clearance-weighted A* avoids cutting the inside of narrow corners.
        """
        free = grid.traversable()
        corridor = bool(self.waypoint.get("observed_corridor")) and grid.c.footprint_half_length > 0
        corridor_offsets = None
        if corridor:
            # In a declared straight passage, plan with the same full oriented
            # body envelope as local DWA. Disk inflation around ramp side edges
            # can reject a body pose that fits; no support/obstacle cells change.
            half = np.array([grid.c.footprint_half_length,grid.c.footprint_half_width])+min(grid.c.safety_margin,.02)
            x,y = np.meshgrid(np.linspace(-half[0],half[0],int(np.ceil(2*half[0]/grid.c.resolution))+1),
                              np.linspace(-half[1],half[1],int(np.ceil(2*half[1]/grid.c.resolution))+1))
            corridor_offsets = np.column_stack([x.ravel(),y.ravel()])
            heading = self.waypoint.get("yaw",pose.yaw)
            lo = np.maximum(0,np.array(grid.cell(pose.xy()-1.6)))
            hi = np.minimum(grid.n-1,np.array(grid.cell(pose.xy()+1.6)))
            i,j = np.mgrid[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
            cells = np.column_stack([i.ravel(),j.ravel()])
            xy = grid.origin+(cells+.5)*grid.c.resolution
            free = np.zeros_like(free)
            free[tuple(cells.T)] = grid.footprints_safe(xy,np.full(len(xy),heading),corridor_offsets)
        start = grid.cell(pose.xy())
        if not grid.inside(start) or not free[start]:
            return np.empty((0, 2))
        room = distance_transform_edt(free)*grid.c.resolution
        costs = 1.+2.*np.exp(-room/.22)
        target = self.target
        end = grid.cell(target)
        path = astar(free, start, end, costs)
        if path:
            return np.array([grid.world(p) for p in path])
        components, _ = label(free)
        cells = np.argwhere(free & (components == components[start]))
        xy = grid.origin+(cells+.5)*grid.c.resolution
        distance = np.linalg.norm(xy-pose.xy(), axis=1)
        remaining = np.linalg.norm(target-pose.xy())
        reduction = remaining-np.linalg.norm(xy-target, axis=1)
        minimum_advance = min(.25,max(.05,.3*remaining)) if corridor else .25
        minimum_reduction = min(.08,.2*remaining) if corridor else .08
        eligible = (distance > minimum_advance)&(distance < 1.35)&(reduction > minimum_reduction)
        cells, xy, distance, reduction = cells[eligible], xy[eligible], distance[eligible], reduction[eligible]
        if not len(cells):
            return np.empty((0, 2))
        score = 2.*reduction-.3*distance+np.minimum(room[tuple(cells.T)], .6)
        if grid.c.footprint_half_length:
            half = np.array([grid.c.footprint_half_length, grid.c.footprint_half_width])+grid.c.safety_margin
            x, y = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
            offsets = np.column_stack([x.ravel(), y.ravel()])
            headings = np.linspace(0., np.pi, 12, endpoint=False)
            if corridor:
                offsets = corridor_offsets
                headings = np.array([self.waypoint.get("yaw",pose.yaw)])
            safe = grid.footprints_safe(np.repeat(xy, len(headings), axis=0), np.tile(headings, len(xy)), offsets)
            score[~safe.reshape(-1, len(headings)).any(axis=1)] = -np.inf
        for i in np.argsort(score)[::-1][:12]:
            if not np.isfinite(score[i]):
                break
            path = astar(free, start, tuple(cells[i]), costs)
            if path:
                return np.array([grid.world(p) for p in path])
        return np.empty((0, 2))
