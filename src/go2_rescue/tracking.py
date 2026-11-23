from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Config
from .geometry import rot2
from .types import BallObservation, Event, Pose


@dataclass
class Track:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    variance: float
    first_seen: float
    last_seen: float
    viewpoint: np.ndarray
    hits: int = 1
    confidence: float = 0.0
    reported: bool = False
    diverse_view: bool = False
    update_time: float = 0.0
    confirming_hits: int = 0
    last_confirming: float = -np.inf
    observation: BallObservation | None = None


class Tracker:
    def __init__(self, config: Config, static_targets=False):
        self.c = config
        self.static_targets = static_targets
        self.tracks: list[Track] = []
        self.next_id = 1
        self.reported_positions: list[np.ndarray] = []

    def update(self, detections: list[BallObservation], pose: Pose, timestamp: float) -> list[Event]:
        detections = [d for d in detections if d.position is not None and np.isfinite(d.position).all()]
        self.tracks = [t for t in self.tracks if timestamp - t.last_seen < (30. if t.reported else 3.)]
        for track in self.tracks:
            dt = max(0., timestamp - track.update_time)
            if self.static_targets:
                track.velocity[:] = 0.
            track.position = track.position + track.velocity * dt
            track.variance = min(4., track.variance + .015 * dt)
            track.update_time = timestamp
        assigned = set()
        if self.tracks and detections:
            costs = np.array([[np.linalg.norm(t.position - d.position) for d in detections] for t in self.tracks])
            rows, columns = linear_sum_assignment(costs)
            for row, col in zip(rows, columns):
                track, detection = self.tracks[row], detections[col]
                gate = min(.75, max(.3, 2.5 * np.sqrt(track.variance + detection.sigma ** 2)))
                if costs[row, col] > gate:
                    continue
                assigned.add(col)
                dt = max(.05, timestamp - track.last_seen)
                innovation = detection.position - track.position
                gain = track.variance / (track.variance + detection.sigma ** 2)
                track.position = track.position + gain * innovation
                track.velocity = .7 * track.velocity + .3 * np.clip(innovation / dt, -.5, .5)
                track.variance = max(.003, (1 - gain) * track.variance)
                if self.static_targets:
                    track.velocity[:] = 0.
                track.last_seen, track.hits = timestamp, track.hits + 1
                track.observation = detection
                track.confidence = .5 * (track.confidence + detection.confidence)
                track.diverse_view |= np.linalg.norm(pose.xy() - track.viewpoint) > .12
                if detection.confidence >= .8:
                    track.confirming_hits += 1
                    track.last_confirming = timestamp
        for i, detection in enumerate(detections):
            if i in assigned:
                continue
            self.tracks.append(Track(self.next_id, detection.position.copy(), np.zeros(2), detection.sigma ** 2,
                                     timestamp, timestamp, pose.xy().copy(), confidence=detection.confidence,
                                     update_time=timestamp, confirming_hits=int(detection.confidence >= .8),
                                     last_confirming=timestamp if detection.confidence >= .8 else -np.inf,
                                     observation=detection))
            self.next_id += 1
        events = []
        for track in self.tracks:
            if track.reported or timestamp != track.last_seen:
                continue
            stable = timestamp - track.first_seen >= .35
            # Static observations can confirm high-confidence geometry; a weak,
            # partial arc requires translation to a distinct camera viewpoint.
            if track.hits < self.c.ball_confirmations or not stable or (track.confidence < .8 and not track.diverse_view):
                continue
            # Repeated weak arcs on a stationary rock are correlated evidence,
            # even when the camera translates. Retain them for active inspection,
            # but require two recent strong observations before a report.
            if track.confirming_hits < 2 or timestamp-track.last_confirming > 1.:
                continue
            if np.sqrt(track.variance) > .35:
                continue
            duplicate = any(np.linalg.norm(track.position - p) < .4 for p in self.reported_positions)
            track.reported = True
            if duplicate:
                continue
            # A declared task count is also a reporting budget. Continue
            # tracking, but never emit more targets than the task contains.
            # Zero retains unknown-count behaviour on unconfigured hardware.
            if self.c.target_quota > 0 and len(self.reported_positions) >= self.c.target_quota:
                continue
            self.reported_positions.append(track.position.copy())
            events.append(Event("target", timestamp, {"id": track.id, "position": track.position.tolist(),
                                                      "frame": "map_start", "confidence": track.confidence,
                                                      "sigma_m": float(np.sqrt(track.variance))}))
        return events

    def correct_map(self, corrections: list[tuple[Pose, Pose]]):
        if not corrections:
            return
        def correct(point):
            old, new = min(corrections, key=lambda pair: np.linalg.norm(pair[0].xy() - point))
            return new.xy() + rot2(new.yaw - old.yaw) @ (point - old.xy())
        for track in self.tracks:
            track.position = correct(track.position)
            track.viewpoint = correct(track.viewpoint)
            track.velocity[:] = 0.
        self.reported_positions = [correct(p) for p in self.reported_positions]
