from __future__ import annotations

import heapq
import math
import numpy as np
from scipy.ndimage import binary_dilation, label, distance_transform_edt
from scipy.spatial import cKDTree

from .config import Config
from .geometry import rot2, wrap
from .mapping import GridMap
from .types import Pose, Velocity


def astar(free: np.ndarray, start, goal, traversal_cost=None) -> list[tuple[int, int]]:
    start, goal = tuple(start), tuple(goal)
    if any(v < 0 or v >= free.shape[k] for p in (start, goal) for k, v in enumerate(p)):
        return []
    if not free[start] or not free[goal]:
        return []
    if traversal_cost is not None and (traversal_cost.shape != free.shape or
                                      not np.isfinite(traversal_cost).all() or (traversal_cost < 1.).any()):
        raise ValueError("Traversal costs must match the map and be finite and >= 1")
    queue = [(math.dist(start, goal), 0., start)]
    costs, parent = {start: 0.}, {}
    moves = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > costs[current] + 1e-9:
            continue
        if current == goal:
            result = [current]
            while current in parent:
                current = parent[current]
                result.append(current)
            return result[::-1]
        for dx, dy in moves:
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            if dx and dy and (not free[current[0] + dx, current[1]] or not free[current[0], current[1] + dy]):
                continue
            factor = 1. if traversal_cost is None else float(traversal_cost[nxt])
            total = cost + math.hypot(dx, dy)*factor
            if total < costs.get(nxt, float("inf")):
                costs[nxt], parent[nxt] = total, current
                heapq.heappush(queue, (total + math.dist(nxt, goal), total, nxt))
    return []


class Explorer:
    def __init__(self):
        self.blacklist: dict[tuple[int, int], float] = {}
        self.trail = []

    def observe(self, pose, timestamp):
        if not self.trail or np.linalg.norm(pose.xy() - self.trail[-1]) > .20:
            self.trail.append(pose.xy().copy())
            self.trail = self.trail[-2000:]

    def reject(self, grid: GridMap, xy, timestamp: float):
        cell = grid.cell(xy)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                self.blacklist[cell[0] + dx, cell[1] + dy] = timestamp + 90.

    def arrived(self, grid: GridMap, xy, timestamp: float):
        # A tiny unobserved sliver behind a wall may remain a frontier after
        # visiting it. Give another region a turn before revisiting this view.
        cell = grid.cell(xy)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.blacklist[cell[0]+dx, cell[1]+dy] = timestamp+45.

    def turning_bay(self, grid, pose):
        """Find observed room reachable with the current body orientation."""
        free = grid.traversable()
        start = grid.cell(pose.xy())
        if not grid.inside(start) or not free[start]:
            return None
        components, _ = label(free)
        cells = np.argwhere((components == components[start]) & free)
        xy = grid.origin + (cells + .5) * grid.c.resolution
        distance = np.linalg.norm(xy-pose.xy(), axis=1)
        valid = (distance > .35) & (distance < 1.6)
        cells, xy, distance = cells[valid], xy[valid], distance[valid]
        if not len(cells):
            return None
        half = np.array([grid.c.footprint_half_length or grid.c.robot_radius,
                         grid.c.footprint_half_width or grid.c.robot_radius]) + grid.c.safety_margin
        xx, yy = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
        offsets = np.column_stack([xx.ravel(), yy.ravel()])
        clearance = distance_transform_edt(grid._surface_safe) * grid.c.resolution
        score = clearance[tuple(cells.T)] - .12*distance
        if self.trail:
            proximity = cKDTree(self.trail).query(xy)[0]
            score += .4*np.minimum(proximity, .8)
        for index in np.argsort(score)[::-1][:30]:
            cell = tuple(cells[index])
            path = astar(free, start, cell)
            if not path:
                continue
            positions = np.array([grid.world(c) for c in path])
            if not grid.footprints_safe(positions, np.full(len(path), pose.yaw), offsets).all():
                continue
            return grid.world(cell)
        return None

    def maneuver(self, grid, pose, goal=None, goal_yaw=None, position_tolerance=.12):
        """Local SE(2) search permits simultaneous sideways motion and turning.

        A quadruped wedged between two obstacles can have no safe pure turn or
        fixed-heading translation, yet still have a safe combined manoeuvre.
        Every edge is checked at intermediate poses against the observed map.
        """
        grid.traversable()
        half = np.array([grid.c.footprint_half_length or grid.c.robot_radius,
                         grid.c.footprint_half_width or grid.c.robot_radius]) + grid.c.safety_margin
        xx, yy = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
        offsets = np.column_stack([xx.ravel(), yy.ravel()])
        actions = np.array([(x, y, h) for x in [-1, 0, 1] for y in [-1, 0, 1]
                            for h in [-1, 0, 1] if x or y or h])
        angle = np.pi/12
        start = (0, 0, 0)
        goal = None if goal is None else np.asarray(goal)
        bound = 12 if goal is None else min(35, int(np.ceil((np.linalg.norm(goal-pose.xy())+.8)/grid.c.resolution)))
        queue, costs, parent = [(0., 0., start)], {start: 0.}, {}
        for _ in range(1200 if goal is None else 5000):
            if not queue:
                break
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]+1e-6:
                continue
            delta = np.array(node[:2])*grid.c.resolution
            turn = (node[2]*angle+np.pi)%(2*np.pi)-np.pi
            reached = np.linalg.norm(delta) > .45 and abs(turn) > .35 if goal is None else np.linalg.norm(pose.xy()+delta-goal) < position_tolerance
            if goal_yaw is not None:
                reached &= abs(wrap(pose.yaw+turn-goal_yaw)) < angle/2+.001
            if reached:
                result = [node]
                while result[-1] != start:
                    result.append(parent[result[-1]])
                return np.array([[*(pose.xy()+np.array(n[:2])*grid.c.resolution), pose.yaw+n[2]*angle]
                                 for n in result[::-1]])
            following = np.asarray(node)+actions
            following[:, 2] %= 24
            bounded = (abs(following[:, :2]) <= bound).all(axis=1)
            origin = pose.xy()+delta
            fractions = np.linspace(0., 1., 9)
            positions = origin + actions[:, None, :2]*grid.c.resolution*fractions[None, :, None]
            headings = pose.yaw+node[2]*angle + actions[:, None, 2]*angle*fractions
            safe = grid.footprints_safe(positions.reshape(-1, 2), headings.ravel(), offsets).reshape(-1, len(fractions)).all(axis=1)
            for i in np.flatnonzero(bounded & safe):
                nxt = tuple(following[i])
                step = max(np.linalg.norm(actions[i, :2])*grid.c.resolution/.12, abs(actions[i, 2])*angle/.25)
                total = cost + step
                if total < costs.get(nxt, np.inf):
                    costs[nxt], parent[nxt] = total, node
                    heuristic = 0. if goal is None else np.linalg.norm(pose.xy()+following[i, :2]*grid.c.resolution-goal)/.12
                    if goal_yaw is not None:
                        heuristic += abs(wrap(pose.yaw+following[i, 2]*angle-goal_yaw))/.25
                    heapq.heappush(queue, (total+heuristic, total, nxt))
        return np.empty((0, 3))

    def margin_escape(self, grid, pose):
        """Leave an infringed safety buffer without crossing the physical body.

        A controller's small tracking lag can put the buffer inside a wall cell,
        while the physical body still fits. Requiring the buffer on the first
        recovery step then deadlocks. Only a fully checked short exit to a pose
        with restored normal clearance is allowed. Water contact checks retain
        their normal foot envelope throughout.
        """
        c = grid.c
        if not c.allow_margin_escape or not c.footprint_half_length:
            return np.empty((0, 3))
        grid.traversable()

        def offsets(margin):
            x, y = np.meshgrid(np.linspace(-c.footprint_half_length-margin, c.footprint_half_length+margin, 11),
                               np.linspace(-c.footprint_half_width-margin, c.footprint_half_width+margin, 9))
            return np.column_stack([x.ravel(), y.ravel()])

        body, buffered = offsets(0.), offsets(c.safety_margin)
        xy, yaw = pose.xy()[None], np.array([pose.yaw])
        if not grid.footprints_safe(xy, yaw, body)[0] or grid.footprints_safe(xy, yaw, buffered)[0]:
            return np.empty((0, 3))
        candidates = np.array([(x, y, angle) for x in np.arange(-.4, .41, .05)
                               for y in np.arange(-.4, .41, .05) for angle in [-.10, 0., .10]
                               if .025 < np.hypot(x, y) < .45])
        fraction = np.linspace(0., 1., 15)
        positions = pose.xy()+candidates[:, None, :2]*fraction[None, :, None]
        headings = pose.yaw+candidates[:, None, 2]*fraction
        physical = grid.footprints_safe(positions.reshape(-1, 2), headings.ravel(), body).reshape(-1, len(fraction)).all(axis=1)
        normal_end = grid.footprints_safe(positions[:, -1], headings[:, -1], buffered)
        valid = np.flatnonzero(physical & normal_end)
        if not len(valid):
            return np.empty((0, 3))
        costs = np.linalg.norm(candidates[valid, :2], axis=1)+.3*abs(candidates[valid, 2])
        i = valid[int(np.argmin(costs))]
        return np.column_stack([positions[i], headings[i]])

    def choose(self, grid: GridMap, pose: Pose, timestamp: float, toward=None) -> np.ndarray | None:
        free = grid.traversable()
        start = grid.cell(pose.xy())
        if not grid.inside(start) or not free[start]:
            return None
        components, _ = label(free)
        reachable = components == components[start]
        # A frontier can border unobserved visibility OR unconfirmed support.
        frontier = free & reachable & binary_dilation(~(grid.seen & np.isfinite(grid.floor)))
        groups, count = label(frontier)
        candidates = []
        if grid.c.footprint_half_length:
            half = np.array([grid.c.footprint_half_length, grid.c.footprint_half_width])+grid.c.safety_margin
            x, y = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
            offsets = np.column_stack([x.ravel(), y.ravel()])
        else:
            angles = np.linspace(0., 2*np.pi, 32, endpoint=False)
            offsets = np.vstack([np.zeros(2), np.column_stack([np.cos(angles), np.sin(angles)])*(grid.c.robot_radius+grid.c.safety_margin)])
        headings = np.linspace(0., np.pi, 12, endpoint=False)
        self.blacklist = {k: expiry for k, expiry in self.blacklist.items() if expiry > timestamp}
        for i in range(1, count + 1):
            cells = np.argwhere(groups == i)
            # Sparse lidar can leave a valid frontier only one or two cells
            # wide. Reachability and robot inflation already enforce clearance;
            # dropping these components can falsely declare exploration done.
            eligible = [tuple(c) for c in cells if tuple(c) not in self.blacklist]
            valid = [c for c in eligible if np.linalg.norm(grid.world(c)-pose.xy()) > .65]
            if not valid:
                # Narrow passages may only expose a short safe advance. Do not
                # demand a 65cm subgoal when 25cm is all the sensor confirms.
                valid = [c for c in eligible if np.linalg.norm(grid.world(c)-pose.xy()) > .25]
            if not valid:
                continue
            center = cells.mean(axis=0)
            valid.sort(key=lambda cell: np.linalg.norm(np.array(cell) - center))
            for cell in valid:
                xy = grid.world(cell)
                path = astar(free, start, cell)
                if not path:
                    continue
                # The frontier center borders unknown support. A center-only
                # goal makes DWA stop a body-length before it forever. Find an
                # observed approach pose along its path where the WHOLE body
                # can stand, then advance the map from fresh observations.
                positions = np.array([grid.world(p) for p in path])
                eligible_indices = [j for j in range(len(path)-1, 0, -1)
                                    if np.linalg.norm(positions[j]-xy) <= .8
                                    and np.linalg.norm(positions[j]-pose.xy()) > .25
                                    and path[j] not in self.blacklist]
                if not eligible_indices:
                    continue
                approaches = positions[eligible_indices]
                safe = grid.footprints_safe(np.repeat(approaches, len(headings), axis=0),
                                            np.tile(headings, len(approaches)), offsets)
                valid_approaches = np.flatnonzero(safe.reshape(-1, len(headings)).any(axis=1))
                if not len(valid_approaches):
                    continue
                index = eligible_indices[valid_approaches[0]]
                distance = max(.2, index * grid.c.resolution)
                score=min(len(cells), 60)*.04-distance
                if toward is not None:
                    # When recovering an ordered survey, prefer safe views
                    # that also make progress toward its checkpoint. A modest
                    # retreat is allowed to look around a blocked bend.
                    advance=np.linalg.norm(np.asarray(toward)-pose.xy())-np.linalg.norm(
                        np.asarray(toward)-positions[index])
                    if advance < -.35:
                        continue
                    score += 1.5*advance
                candidates.append((score, positions[index]))
                break
        return max(candidates, key=lambda v: v[0])[1] if candidates else None


class DWA:
    def __init__(self, config: Config):
        self.c = config

    def safe_turn(self, pose, desired_rate, grid):
        if not grid.inside(grid.cell(pose.xy())):
            return Velocity()
        free = grid.traversable()
        if not self.c.footprint_half_length and not free[grid.cell(pose.xy())]:
            return Velocity()
        rate = float(np.clip(desired_rate, -self.c.max_wz, self.c.max_wz))
        if self.c.footprint_half_length:
            half = np.array([self.c.footprint_half_length, self.c.footprint_half_width]) + self.c.safety_margin
            xx, yy = np.meshgrid(np.linspace(-half[0], half[0], 9), np.linspace(-half[1], half[1], 7))
            offsets = np.column_stack([xx.ravel(), yy.ravel()])
            headings = pose.yaw + rate * np.linspace(0., 1.5, 12)
            if not grid.footprints_safe(np.repeat(pose.xy()[None], len(headings), axis=0), headings, offsets).all():
                return Velocity()
        return Velocity(wz=rate)

    def command(self, pose: Pose, measured: Velocity, path: np.ndarray,
                obstacles: np.ndarray, grid: GridMap, dt: float = .1, face_path: bool = True,
                desired_yaw: float | None = None, reverse_limit: float | None = None,
                forward_limit: float | None = None) -> Velocity:
        if len(path) == 0:
            return Velocity()
        free = grid.traversable()
        # Pick a lookahead on the path, not an obstacle-crossing shortcut.
        distances = np.linalg.norm(path - pose.xy(), axis=1)
        measured_speed = np.hypot(measured.vx, measured.vy)
        lookahead = float(np.clip(.30 + .8*measured_speed + measured_speed**2/(2*self.c.braking), .30, 1.20))
        ahead = np.flatnonzero(distances > lookahead)
        target = path[ahead[0]] if len(ahead) else path[-1]
        world_obstacles = obstacles[:, :2] @ rot2(pose.yaw).T + pose.xy() if len(obstacles) else np.empty((0, 2))
        tree = cKDTree(world_obstacles) if len(world_obstacles) else None
        dt = float(np.clip(dt, .02, .2))
        reverse_speed = self.c.max_reverse_vx if reverse_limit is None else min(self.c.max_reverse_vx, reverse_limit)
        forward_speed = self.c.max_vx if forward_limit is None else min(self.c.max_vx, forward_limit)
        velocities = []
        # Keep the established forward-window sampling unchanged. The reverse
        # bound is a candidate rejection below; changing the sampling endpoints
        # also perturbs normal ramp/corner commands near zero velocity.
        for value, minimum, maximum, accel in zip(measured.array(), [-max(self.c.max_vx, reverse_speed), -self.c.max_vy, -self.c.max_wz],
                                         [forward_speed, self.c.max_vy, self.c.max_wz],
                                         [self.c.acceleration, self.c.acceleration, self.c.angular_acceleration]):
            lo = float(np.clip(value - accel*dt, minimum, maximum))
            hi = float(np.clip(value + accel*dt, minimum, maximum))
            velocities.append(np.linspace(lo, hi, 5))
        best, best_cost = Velocity(), float("inf")
        radius = self.c.robot_radius + self.c.safety_margin
        rectangular = self.c.footprint_half_length > 0
        if rectangular:
            half = np.array([self.c.footprint_half_length, self.c.footprint_half_width]) + self.c.safety_margin
            xx, yy = np.meshgrid(np.linspace(-half[0], half[0], int(np.ceil(2*half[0]/self.c.resolution))+1),
                                 np.linspace(-half[1], half[1], int(np.ceil(2*half[1]/self.c.resolution))+1))
            offsets = np.column_stack([xx.ravel(), yy.ravel()])
        vx, vy, wz = [a.ravel() for a in np.meshgrid(*velocities, indexing="ij")]
        count = len(vx)
        xy = np.repeat(pose.xy()[None, :], count, axis=0)
        yaw = np.full(count, pose.yaw)
        valid = (vx >= -reverse_speed)&(vx <= forward_speed)
        clearance = np.full(count, self.c.max_range)
        speed = np.hypot(vx, vy)
        horizon = 1. + speed / self.c.braking
        for t in np.arange(.1, horizon.max() + .05, .1):
            active = t <= horizon + .05
            factor = np.ones(count) if t <= 1. else np.maximum(0., 1. - (t - 1.) * self.c.braking / np.maximum(speed, 1e-6))
            factor *= active
            cs, sn = np.cos(yaw), np.sin(yaw)
            xy[:, 0] += (cs * vx - sn * vy) * .1 * factor
            xy[:, 1] += (sn * vx + cs * vy) * .1 * factor
            yaw = (yaw + wz * .1 * factor + np.pi) % (2*np.pi) - np.pi
            cells = np.floor((xy - grid.origin) / self.c.resolution).astype(int)
            inside = (cells >= 0).all(axis=1) & (cells < grid.n).all(axis=1)
            clipped = np.clip(cells, 0, grid.n-1)
            safe = inside if rectangular else inside & free[tuple(clipped.T)]
            if rectangular:
                safe &= grid.footprints_safe(xy, yaw, offsets)
            if tree:
                distances, _ = tree.query(xy)
                clearance = np.where(active, np.minimum(clearance, distances), clearance)
                if rectangular:
                    dx = world_obstacles[None, :, 0] - xy[:, 0, None]
                    dy = world_obstacles[None, :, 1] - xy[:, 1, None]
                    cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
                    collision = ((abs(cs*dx + sn*dy) < half[0]) & (abs(-sn*dx + cs*dy) < half[1])).any(axis=1)
                else:
                    collision = distances < radius
                safe &= ~collision
            valid &= ~active | safe
        if not valid.any():
            return Velocity()
        heading = abs((np.arctan2(target[1]-xy[:, 1], target[0]-xy[:, 0])-yaw+np.pi) % (2*np.pi)-np.pi)
        if desired_yaw is not None:
            heading = abs((desired_yaw-yaw+np.pi)%(2*np.pi)-np.pi)
        # Safety is a hard footprint constraint above. A reciprocal potential
        # with an unbounded cost made narrow but safe passages unattractive
        # enough that the optimum was permanently zero velocity.
        comfort = .15 / (1. + np.maximum(0., clearance-radius)/.10)
        costs = 3.*np.linalg.norm(target-xy, axis=1) + (.6 if face_path else 0.)*heading + comfort - .3*vx
        costs[~valid] = np.inf
        best = int(np.argmin(costs))
        return Velocity(float(vx[best]), float(vy[best]), float(wz[best]))
