"""RGB low-obstacle detection migrated from the Gazebo perception stack.

The Gazebo node used a LRASPP segmentation model plus per-column geometry:
the bottom pixel of a rubble run gives the ground contact point, the top
pixel gives the obstacle height.  This module keeps that algorithm but runs
ONNX Runtime directly on the MuJoCo front camera image, with no ROS.

The module is observation-only: it returns detections in the robot body frame
and does not feed the planner unless a caller chooses to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class LowObstacle:
    x: float  # body-frame forward (m)
    y: float  # body-frame left (m)
    height: float  # metres above the local ground plane
    score: float
    clipped: bool = False
    height_quality: str = "measured"  # "measured" | "clipped" (0.30 m upper bound, not a measurement)


def _project(point, rotation, translation, fx, fy, cx, cy):
    camera = rotation.T @ (np.asarray(point, float) - translation)
    if camera[2] <= 1e-3:
        return None
    return fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy


def detect_from_probability(prob: np.ndarray, camera_matrix, body_from_camera,
                            base_height: float, min_prob: float = 0.60,
                            min_height: float = 0.01, max_height: float = 0.30,
                            height_offset: float = 0.0, cell: float = 0.05,
                            col_step: int = 2) -> list[LowObstacle]:
    """Convert a segmentation probability map into low-obstacle detections.

    ``prob`` is (3, H, W) with class 0 ground, 1 rubble, 2 structure.
    Runs touching the image border are solved by borrowing the ground-contact
    distance from a nearby column that is fully visible; only when no neighbour
    exists does the height fall back to the upper bound (flagged ``clipped``).
    """
    k = np.asarray(camera_matrix, float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    t = np.asarray(body_from_camera, float)
    rotation, translation = t[:3, :3], t[:3, 3]
    h, w = prob.shape[1:]
    mask = (prob[1] > min_prob).astype(np.uint8)
    struct = ((prob.argmax(0) == 2) & (prob[2] > 0.5)).astype(np.uint8)
    if struct.any():
        mask[cv2.dilate(struct, np.ones((7, 7), np.uint8)).astype(bool)] = 0
    ground_z = -float(base_height)

    def ray_ground(ray):
        d_body = rotation @ ray
        if d_body[2] >= -1e-3:
            return None
        dist = (ground_z - translation[2]) / d_body[2]
        if dist <= 0.15 or dist > 8.0:
            return None
        return dist, d_body

    columns = {}
    for u in range(0, w, col_step):
        col = mask[:, u]
        if not col.any():
            continue
        ys = np.nonzero(col)[0]
        splits = np.where(np.diff(ys) > 2)[0]
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [len(ys) - 1]))
        runs = []
        for s, e in zip(starts, ends):
            vb, vt = int(ys[e]), int(ys[s])
            if vb - vt < 2:
                continue
            clipped = vb > h - 4 or vt < 3
            vb_c = min(vb, h - 2)
            ray = np.array([(u - cx) / fx, (vb_c - cy) / fy, 1.0])
            runs.append((vb_c, vt, clipped, ray_ground(ray)))
        if runs:
            columns[u] = runs

    visible_ground = {}
    for u, runs in columns.items():
        for _, _, clipped, g in runs:
            if not clipped and g is not None:
                visible_ground[u] = g[0]

    cells = {}
    run_id = 0
    for u in range(0, w, col_step):
        for vb, vt, clipped, g in columns.get(u, []):
            run_id += 1
            if g is None:
                continue
            dist, d_body = g
            ground = translation + dist * d_body
            placeholder = clipped
            if placeholder:
                height = max_height
            else:
                lo, hi = 0.0, max(0.5, max_height + 0.2)
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    q = _project(ground + np.array([0.0, 0.0, mid]), rotation, translation,
                                 fx, fy, cx, cy)
                    if q is None:
                        break
                    if q[1] > vt:
                        lo = mid
                    else:
                        hi = mid
                height = 0.5 * (lo + hi) - height_offset
            if height < min_height or height > max_height:
                continue
            key = ("clip", run_id) if placeholder else (int(round(ground[0] / cell)),
                                                        int(round(ground[1] / cell)))
            score = float(prob[1, int((vt + vb) / 2), u])
            old = cells.get(key)
            if old is None or height > old.height:
                cells[key] = LowObstacle(float(ground[0]), float(ground[1]), float(height),
                                         score, placeholder,
                                         "clipped" if placeholder else "measured")
    return sorted(cells.values(), key=lambda o: -o.height)


def rubble_region_observable(x, y, width, height, image_width):
    """Reject thin side fragments with most horizontal extent out of view."""
    side_cut = x == 0 or x + width >= image_width
    return not (side_cut and width < .25 * height)


class LowObstacleDetector:
    """ONNX-backed low-obstacle detector for the MuJoCo front camera."""

    def __init__(self, model_path: str | Path, camera_matrix, body_from_camera,
                 base_height: float, min_prob: float = 0.60, max_height: float = 0.30,
                 height_offset: float = 0.0, session=None,
                 min_prob_far: float = 0.30, far_distance: float = 0.0):
        import onnxruntime as ort
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.session = session
        interface = self.session.get_inputs()[0]
        self.input = interface.name
        shape = getattr(interface, "shape", [1,3,320,480])
        self.input_size = (int(shape[3]), int(shape[2])) if isinstance(shape[2],int) and isinstance(shape[3],int) else (480,320)
        self.k = np.asarray(camera_matrix, float)
        self.t = np.asarray(body_from_camera, float)
        self.base_height = float(base_height)
        self.min_prob = min_prob
        self.max_height = max_height
        self.height_offset = height_offset
        self.min_prob_far = min_prob_far
        self.far_distance = far_distance

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, self.input_size).astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input: x})[0][0]
        prob = cv2.resize(logits.transpose(1, 2, 0), (image_bgr.shape[1], image_bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        e = np.exp(prob - prob.max(axis=2, keepdims=True))
        return (e / e.sum(axis=2, keepdims=True)).transpose(2, 0, 1)

    def detect(self, image_bgr: np.ndarray) -> list[LowObstacle]:
        prob = self.segment(image_bgr)
        near = detect_from_probability(prob, self.k, self.t, self.base_height,
                                       self.min_prob, max_height=self.max_height,
                                       height_offset=self.height_offset)
        if self.min_prob_far >= self.min_prob - 1e-6:
            return near
        far = [o for o in detect_from_probability(prob, self.k, self.t, self.base_height,
                                                  self.min_prob_far, max_height=self.max_height,
                                                  height_offset=self.height_offset)
               if o.x > self.far_distance]
        seen = {(round(o.x / 0.05), round(o.y / 0.05)) for o in near}
        extra = [o for o in far if (round(o.x / 0.05), round(o.y / 0.05)) not in seen]
        return sorted(near + extra, key=lambda o: -o.height)
