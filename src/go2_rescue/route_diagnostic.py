"""User-requested external-route locomotion isolation; uses truth explicitly.

Never used by Autonomy or hardware. Results cannot pass autonomous acceptance.
"""
import json
import numpy as np
from .geometry import rot2, wrap
from .types import Velocity


class KnownRouteDiagnostic:
    def __init__(self, path, initial_xy, initial_yaw):
        route = json.loads(path.read_text(encoding="utf-8"))
        self.points = [(stage["name"], w) for stage in route["stages"] for w in stage["waypoints"]]
        self.anchor = np.asarray(initial_xy, float)
        self.rotation = rot2(initial_yaw)
        self.index = 0
        self.last_progress = 0.
        self.best = np.inf
        self.status = {}
        self.stalled = False

    def command(self, timestamp, world_xy, world_yaw):
        if self.index == len(self.points):
            return Velocity()
        stage, w = self.points[self.index]
        target = self.anchor + self.rotation @ np.asarray(w["xy"])
        delta = target - world_xy
        distance = float(np.linalg.norm(delta))
        if distance < .18:
            self.index += 1
            self.best = np.inf
            self.last_progress = timestamp
            return Velocity()
        if distance < self.best - .08:
            self.best = distance
            self.last_progress = timestamp
        self.stalled = timestamp - self.last_progress > 35.
        reverse = bool(w.get("reverse"))
        heading = float(np.arctan2(delta[1], delta[0])) + (np.pi if reverse else 0.)
        error = wrap(heading-world_yaw)
        speed = min(.20, .6*distance)
        if stage == "L1_descent":
            speed = min(speed, .12)
        body = rot2(world_yaw).T @ delta
        vx = (-speed if reverse else speed) * max(0., np.cos(error)) if abs(error) < .65 else 0.
        vy = float(np.clip(.35*body[1], -.04, .04)) if abs(error) < .65 else 0.
        self.status = {"uses_simulator_pose": True, "autonomous": False, "stage": stage,
                       "index": self.index, "count": len(self.points), "target_world_xy": target.tolist(),
                       "distance_m": distance, "stalled": self.stalled}
        return Velocity(vx, vy, float(np.clip(1.2*error, -.25, .25)))
