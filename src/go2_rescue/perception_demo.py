"""Observation-only perception for the scripted MuJoCo demonstration."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .geometry import gravity_rotation, rot2, transform_xy
from .low_obstacle import LowObstacleDetector, rubble_region_observable
from .perception import Perception
from .ball_classifier import BallProposalClassifier
from .tracking import Tracker
from .stance_surface import StanceSurface
from .mapping import spatial_frame_support
from .terrain_layer import collect_cell_heights, terrain_snapshot, write_terrain_snapshot
from .rubble_visibility import visible_ground_hypotheses
from .perception_io import perception_snapshot, public_range_evidence, publish_perception_snapshot
from .hybrid_ball import HybridBallRange
from .ball_outline import refine_ball_outline
from .ball_image_tracks import BallImageTracks
from .ball_multiview import BallTwoView, fuse_target_position
from .rubble_visual_height import RubbleVisualHeight


def emit(path, row):
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, allow_nan=False) + "\n")


def write_png(path, image):
    # Atomic update so the monitor never reads a half-written PNG.
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    tmp.replace(path)


def confirmable_ball(detection, minimum_confidence):
    # A distant centre projected onto nominal ground is an observation.
    # Wait for a tighter range estimate before publishing an immutable mark.
    return (detection.position is not None and detection.confidence >= minimum_confidence
            and detection.sigma <= .20)


def support_plane(points, base_height):
    """Fit near-field ground compatible with the expected contact height."""
    distance = np.linalg.norm(points[:, :2], axis=1)
    ground = points[(distance < 2.5) & (points[:, 2] <= -base_height + .08 + .40*distance)]
    if len(ground) < 6 or np.linalg.eigvalsh(np.cov(ground[:, :2].T))[0] <= .008:
        return None
    design = np.column_stack((ground[:, :2], np.ones(len(ground))))
    fit = np.linalg.lstsq(design, ground[:, 2], rcond=None)[0]
    if (np.linalg.norm(fit[:2]) >= .40 or abs(fit[2] + base_height) >= .065
            or np.quantile(abs(design @ fit - ground[:, 2]), .8) >= .035):
        return None
    return fit


class PerceptionDemo:
    def __init__(self, autonomy, model_path, output, known_target_count=None):
        self.autonomy = autonomy
        self.grid = autonomy.grid
        self.loc = autonomy.localization
        self.c = self.grid.c
        self.output = Path(output)
        self.segmenter = LowObstacleDetector(model_path, self.c.camera_matrix,
                                            self.c.body_from_camera, self.c.base_height)
        self.vision = Perception(self.c)
        self.slope_perception = False
        self.outline_ball = False
        self.hybrid_ball = False
        self.range_evidence = None
        self.range_grid = None
        self.ball_two_view = BallTwoView(self.vision.k)
        self.hybrid_ball_range = HybridBallRange(self.vision, compensate_rotation=True)
        self.ball_image_tracks = BallImageTracks(self.vision.k)
        self.visual_height = RubbleVisualHeight(self.vision.k)
        self.rubble_precision = None
        self.single_view_rubble = None
        self.rubble_footprint = None
        target_model = Path(model_path).parent / "ball_proposal.onnx"
        self.ball_classifier = BallProposalClassifier(target_model) if target_model.exists() else None
        self.ball_min_confidence = .85 if self.ball_classifier is not None else .86
        tracker_config = self.c if known_target_count is None else replace(self.c,target_quota=known_target_count)
        self.tracker = Tracker(tracker_config, static_targets=True)
        self.history = []
        self.last_front = None
        self.front_rubble = None
        self.latest_detections = []
        self.target_observation = None
        self.confirmation_frames = {}
        self.latest_confirmations = []
        self.lidar = np.empty((0, 3))
        self.violations = 0
        self.camera_frames = 0
        self.geometry_frames = 0
        self.visual_components = 0
        self.ball_candidates = 0
        self.next_preview = 0.
        self.support_height = None
        self.stance_surface = StanceSurface()
        self.terrain_observations = 0
        self.terrain_samples = []
        self.terrain_layer = None
        self.recent_lidar = []
        self.video = cv2.VideoWriter(str(self.output / "perception.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 5., (1056, 480))
        if not self.video.isOpened():
            raise RuntimeError("Cannot create perception.mp4")
        self.last_video = None

    def estimated_base_z(self):
        if self.c.navigation_support and self.grid.support_plane is not None:
            return self.grid.ground_height + self.c.base_height
        return np.nan

    def observe(self, frame, scan_metadata, estimated=None):
        self.grid = self.autonomy.grid
        if estimated is None:
            pose, points = self.loc.update(frame)
            if self.loc.corrections:
                self.grid.clear()
                for key in self.loc.keyframes:
                    self.grid.update(key.points, key.pose)
            if pose.quality >= self.c.min_pose_quality:
                self.grid.update(points, pose)
        else:
            pose = estimated
        if self.loc.corrections:
            self.visual_height.clear()
            if self.rubble_footprint is not None:
                self.rubble_footprint.correct(self.loc.corrections)
            if self.single_view_rubble is not None and self.single_view_rubble.completion is not None:
                completion = self.single_view_rubble.completion
                completion.clear()
                if completion.revision is not None:
                    completion.revision.correct(self.loc.corrections)
            if self.rubble_precision is not None:
                self.rubble_precision.clear()
            self.ball_two_view.views.clear()
            self.ball_image_tracks.clear()
            self.hybrid_ball_range.clear()
            for row in self.history:
                previous = row["pose"]
                old, new = min(self.loc.corrections, key=lambda pair: np.linalg.norm(pair[0].xy() - previous.xy()))
                previous.x, previous.y = new.xy() + rot2(new.yaw - old.yaw) @ (previous.xy() - old.xy())
                previous.yaw += new.yaw - old.yaw
            self.tracker.correct_map(self.loc.corrections)
        self.stance_surface.update(frame)
        if self.visual_height.multiview:
            self.observe_terrain_contacts(frame, pose)
        self.lidar = frame.points
        self.azimuths = len(scan_metadata["azimuth_index"])
        ids = scan_metadata["azimuth_index"]
        violation = (len(ids) > 72 or len(np.unique(ids)) != len(ids) or len(ids) != len(frame.points)
                     or bool(np.any(frame.points @ scan_metadata["up_body"] < self.c.lidar_height_floor - 1e-8)))
        self.violations += violation
        emit(self.output / "lidar_diagnostics.jsonl", {
            "t": frame.timestamp, "returns": self.azimuths, "contract_violation": violation,
            "pose": asdict(pose), "localization": self.loc.last_registration,
            "agent_obs": scan_metadata["agent_obs"].tolist() if self.slope_perception else None,
        })
        if self.slope_perception:
            if self.range_grid is not self.grid or self.loc.corrections:
                self.range_grid = self.grid
                self.range_evidence = public_range_evidence(self.grid)
            if pose.quality >= self.c.min_pose_quality:
                # Use recent triangulated RGB ground only. Legacy XYZ floors
                # cannot certify a range-only map product.
                ground = np.full(self.grid.seen.shape, np.nan, dtype=np.float32)
                heights = defaultdict(list)
                for time, cloud in self.visual_height.ground_history:
                    if 0 <= frame.timestamp-time <= 3.:
                        for x, y, z in cloud:
                            heights[self.grid.cell((x, y))].append(z)
                for cell, values in heights.items():
                    if self.grid.inside(cell) and len(values) >= 2 and np.ptp(values) <= .035:
                        ground[cell] = np.median(values)
                self.range_evidence.update(self.grid, scan_metadata["agent_obs"], frame.quaternion,
                                           pose, self.c.min_range, self.c.max_range,
                                           base_z=self.estimated_base_z(), ground=ground)
        if frame.image is not None:
            self.observe_image(frame, pose, scan_metadata["agent_obs"])
        if frame.timestamp >= self.next_preview - 1e-6:
            self.snapshot(frame.timestamp)
            self.next_preview = frame.timestamp + .5
        return pose

    def observe_terrain_contacts(self, frame, pose):
        soles = frame.support_points
        if (soles is not None and self.grid.support_plane is not None
                and pose.quality >= self.c.min_pose_quality):
            self.visual_height.observe_contacts(soles @ gravity_rotation(frame.quaternion).T,
                                                pose, self.estimated_base_z(), frame.timestamp)

    def observe_image(self, frame, pose, lidar_obs=None):
        self.camera_frames += 1
        image = self.vision.undistort(frame.image)
        front = image.copy()
        valid_pose = pose.quality >= self.c.min_pose_quality
        level = gravity_rotation(frame.quaternion)
        transform = self.vision.t
        rotation = level @ transform[:3,:3]
        origin = level @ transform[:3,3]
        depth_layers = None
        if self.single_view_rubble is not None and valid_pose:
            stance = self.stance_surface.plane
            reference = stance if stance is not None else np.array([0.,0.,-self.c.base_height])
            depth_layers = self.single_view_rubble.estimate(image,rotation,origin,reference)
        probability = self.segmenter.segment(image)
        detections = self.vision.detect_balls(image, small_targets=self.ball_classifier is not None)
        if self.ball_classifier is not None:
            detections = self.ball_classifier.filter(image, detections)
        if not self.slope_perception:
            detections = [self.vision.locate_ball(d, frame, pose) for d in detections]
        labels = ((probability[1] >= .70) & (probability.argmax(axis=0) == 1)).astype(np.uint8)
        support=self.autonomy.navigation_support
        if support is not None and support.probability_stamp==frame.timestamp:
            other=support.probability
            labels[(other[0]>.95)|(other[2]>.95)]=0
        for d in detections:
            if (self.slope_perception or d.position is not None) and d.confidence >= self.ball_min_confidence:
                cv2.circle(labels, tuple(np.rint(d.center).astype(int)), int(np.ceil(d.radius))+2, 0, -1)
        # Keep a confirmed static target out of the rough-surface layer even
        # when its proposal is missed in this particular camera frame.
        if not self.slope_perception and pose.quality >= self.c.min_pose_quality:
            level_now = gravity_rotation(frame.quaternion)
            optical_rotation = level_now @ self.vision.t[:3,:3]
            optical_origin = level_now @ self.vision.t[:3,3]
            for target in self.tracker.reported_positions:
                observed = any(d.position is not None and d.confidence >= self.ball_min_confidence
                               and np.linalg.norm(d.position-target) < .35 for d in detections)
                if observed:
                    continue
                xy = rot2(pose.yaw).T @ (target-pose.xy())
                center = np.r_[xy, self.c.ball_radius - (self.support_height or self.c.base_height)]
                camera = (center-optical_origin) @ optical_rotation
                if camera[2] > .15:
                    pixel = self.vision.k @ camera
                    uv = np.rint(pixel[:2]/pixel[2]).astype(int)
                    radius = int(np.ceil(self.vision.k[0,0]*(self.c.ball_radius+.015)/camera[2]))
                    cv2.circle(labels,tuple(uv),radius,0,-1)
        structure = ((probability.argmax(0) == 2) & (probability[2] >= .45)).astype(np.uint8)
        labels[cv2.dilate(structure, np.ones((3, 3), np.uint8)) != 0] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(labels)
        points = frame.points @ level.T
        # The experimental path must not use the simulator's hidden elevation
        # for lidar/camera association or for terrain height.
        if self.slope_perception or self.rubble_footprint is not None:
            points = np.empty((0, 3))
        optical = (points - origin) @ rotation
        in_front = optical[:, 2] > .05
        projected = optical[in_front] @ self.vision.k.T
        uv = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
        h, w = labels.shape
        on_screen = ((uv >= 0) & (uv < [w, h])).all(axis=1)
        hits = points[in_front][on_screen]
        u, v = uv[on_screen].T
        hit_labels = labels[v, u]
        plane = support_plane(points, self.c.base_height)
        plane_source = "lidar_plane" if plane is not None else "unavailable"
        if self.stance_surface.plane is not None:
            self.support_height = -float(self.stance_surface.plane[2])
            if plane is None:
                plane = self.stance_surface.plane.copy()
                plane_source = self.stance_surface.status["source"]
        row = {"pose": type(pose)(**vars(pose)), "t": frame.timestamp, "rough": [], "heights": []}
        observations = []
        self.front_rubble = np.zeros(labels.shape, dtype=bool)
        for i in range(1, count):
            x, y, width, height, area = stats[i]
            if area < 25 or not rubble_region_observable(x, y, width, height, w):
                continue
            self.visual_components += 1
            mask = labels == i
            self.front_rubble |= mask
            front[mask] = (.52 * front[mask] + .48 * np.array([0, 200, 255])).astype(np.uint8)
            if self.rubble_footprint is not None:
                continue
            actual_hits = hits[hit_labels == i]
            above = np.empty((0, 3))
            if plane is not None and len(actual_hits):
                residual = actual_hits[:, 2] - (actual_hits[:, :2] @ plane[:2] + plane[2])
                above = actual_hits[(residual >= .035) & (residual <= .30)]
                residual = residual[(residual >= .035) & (residual <= .30)]
            if valid_pose and len(above) and self.tracker.reported_positions:
                coordinates = transform_xy(above,pose)
                target_distance = np.linalg.norm(coordinates[:,None,:]
                    - np.asarray(self.tracker.reported_positions)[None,:,:],axis=2).min(axis=1)
                keep = target_distance > self.c.ball_radius + .10
                above, residual = above[keep], residual[keep]
            confirmed = len(above) >= 1
            if confirmed and valid_pose:
                self.terrain_observations += 1
                self.terrain_samples.append({
                    "t": frame.timestamp, "plane_source": plane_source,
                    "points_body": (above @ level).tolist(),
                    "points_level": above.tolist(), "height_m": residual.tolist(),
                    "pose": asdict(pose), "plane": plane.tolist(),
                    "support_status": dict(self.stance_surface.status),
                })
            if confirmed:
                if valid_pose:
                    row["rough"].append((above[:, :2], True))
                    row["heights"].append((above[:, :2], residual))
                interval = [float(residual.min()), float(residual.max())]
            else:
                interval = None
            if not self.slope_perception and y + height < h and x > 0 and x + width < w:
                # Ground-contact rays are position hypotheses. They never
                # manufacture measured height or certify traversability.
                plane_used = plane if plane is not None else np.array([0., 0., -self.c.base_height])
                pixels = []
                for col in range(x, x + width, 3):
                    ys = np.flatnonzero(mask[:, col])
                    if len(ys):
                        pixels.append([col, ys[-1], 1])
                rays = np.asarray(pixels) @ np.linalg.inv(self.vision.k).T @ rotation.T
                denominator = rays[:, 2] - rays[:, :2] @ plane_used[:2]
                scale = np.divide(plane_used[2] + origin[:2] @ plane_used[:2] - origin[2],
                                  denominator, out=np.zeros_like(denominator), where=denominator < -.03)
                samples = origin[:2] + rays[(scale > .25) & (scale < 4), :2] * scale[(scale > .25) & (scale < 4), None]
            else:
                samples = np.empty((0, 2))
            occluded_samples = 0
            if valid_pose and len(samples):
                world_samples = transform_xy(samples, pose)
                camera_xy = transform_xy(origin[None, :], pose)[0]
                visible = visible_ground_hypotheses(self.grid, camera_xy, world_samples)
                occluded_samples = int((~visible).sum())
                samples = samples[visible]
            if confirmed and valid_pose:
                self.geometry_frames += 1
            if valid_pose and len(samples):
                row["rough"].append((samples, False))
            observations.append({"box_xywh": [int(x), int(y), int(width), int(height)], "mask_pixels": int(area),
                                 "source": "lidar_residual" if confirmed else
                                 (plane_source if plane is not None else "nominal_flat_ground"),
                                 "plane_source": plane_source, "height_interval_m": interval, "lidar_hits": len(actual_hits),
                                 "occluded_visual_samples": occluded_samples,
                                 "lidar_map_xy": transform_xy(above[:, :2], pose).tolist() if valid_pose else [],
                                 "map_xy": transform_xy(samples, pose).tolist() if valid_pose else []})
        if self.rubble_footprint is not None:
            if valid_pose and lidar_obs is not None:
                reference = self.stance_surface.plane
                if reference is None:
                    reference = np.array([0.,0.,-self.c.base_height])
                floor = (probability[0]>=.80)&(probability.argmax(0)==0)
                comparison = self.rubble_footprint.observe(
                    image,self.front_rubble,floor,lidar_obs,level,rotation,origin,
                    reference,pose,self.estimated_base_z(),frame.timestamp)
                emit(self.output / "rubble_footprint.jsonl", comparison)
        elif valid_pose and self.slope_perception:
            # NavigationSupport defines ground_height as body map Z minus
            # nominal base height. Do not treat stance alone as a map datum.
            base_z = self.estimated_base_z()
            if self.hybrid_ball:
                self.hybrid_ball_range.update(detections, image, frame, pose, base_z,
                                              self.stance_surface.plane, self.ball_min_confidence)
            elif self.outline_ball:
                refined = [d for d in detections
                           if d.confidence >= self.ball_min_confidence and refine_ball_outline(image, d)]
                self.ball_image_tracks.update(refined, pose, base_z, rotation, origin,
                                              frame.timestamp, self.ball_min_confidence)
            else:
                self.ball_two_view.update(detections, pose, base_z, rotation, origin,
                                          frame.timestamp, self.ball_min_confidence)
            for detection in detections:
                if self.hybrid_ball or detection.position is None:
                    continue
                # Do not let simulator-only XYZ influence the single-view check.
                image_only = replace(frame, points=np.empty((0, 3)), point_times=None)
                candidate = self.vision.locate_ball(replace(detection), image_only, pose)
                fuse_target_position(detection, candidate)
            ground_mask = (probability[0] >= .65) & (probability.argmax(axis=0) == 0)
            floor_mask = None
            if self.visual_height.multiview:
                floor_mask = np.zeros_like(ground_mask)
                if support is not None and support.probability_stamp == frame.timestamp:
                    floor_mask = support.probability[0] >= .85
                    if np.any(self.c.distortion):
                        floor_mask = cv2.undistort(floor_mask.astype(np.float32), self.vision.k,
                                                   np.asarray(self.c.distortion)) >= .95
            started = perf_counter()
            stereo = self.visual_height.observe(image, self.front_rubble, rotation, origin, pose,
                                                self.grid.ground_height, self.stance_surface.plane,
                                                frame.timestamp, ground_mask=ground_mask, base_z=base_z,
                                                floor_mask=floor_mask)
            if len(stereo):
                xy_level = (stereo[:, :2] - pose.xy()) @ rot2(pose.yaw)
                row["rough"].append((xy_level, False))
                valid_height = np.isfinite(stereo[:, 4])
                row["visual_height"] = (xy_level[valid_height], stereo[valid_height, 4])
            dense = self.visual_height.extra_surface
            if len(dense):
                xy_level = (dense[:, :2] - pose.xy()) @ rot2(pose.yaw)
                row["dense_surface"] = (xy_level, dense[:, 4])
            precision_log = {}
            if self.rubble_precision is not None and np.isfinite(base_z):
                precise, candidates = self.rubble_precision.observe(
                    image, pose, rotation, origin, base_z, self.stance_surface.plane, frame.timestamp, stereo, dense)
                for name,points in (("rgb_precision_points", precise), ("rgb_height_candidates", candidates)):
                    if len(points):
                        local = points.copy()
                        local[:, :2] = (local[:, :2]-pose.xy()) @ rot2(pose.yaw)
                        row[name] = local
                    precision_log[name] = [[float(v) if np.isfinite(v) else None for v in p] for p in points]
            emit(self.output / "visual_pseudocloud.jsonl", {
                "t": frame.timestamp, "frame": "map_start",
                "columns": ["x", "y", "surface_z", "ground_z", "relief_m"],
                "points": [[float(v) if np.isfinite(v) else None for v in p] for p in stereo],
                "dense_points": [[float(v) if np.isfinite(v) else None for v in p] for p in dense],
                "processing_ms": (perf_counter()-started)*1000.,
                "base_z_available": bool(np.isfinite(base_z)),
                "geometry": self.visual_height.diagnostics,
                **precision_log,
            })
        elif valid_pose:
            stereo = self.visual_height.observe(image, self.front_rubble, rotation, origin, pose,
                                                self.grid.ground_height, self.stance_surface.plane,
                                                frame.timestamp)
            if len(stereo):
                xy_level = (stereo[:, :2] - pose.xy()) @ rot2(pose.yaw)
                row['visual_height'] = (xy_level, stereo[:, 2])
        if depth_layers is not None:
            from .rubble_single_view import RubbleRangeAnchors
            if not hasattr(self, "rubble_range_anchors"):
                self.rubble_range_anchors = RubbleRangeAnchors(self.c, self.vision.k)
            ranging = ((self.rubble_range_anchors, lidar_obs, level) if lidar_obs is not None else None)
            candidate_xy, comparison = self.single_view_rubble.label(
                self.front_rubble,depth_layers,pose,rotation,origin,reference,ranging=ranging,
                context=(image,frame.timestamp,self.estimated_base_z(),
                         (probability[0]>=.80)&(probability.argmax(0)==0)))
            comparison.update(t=frame.timestamp,scale_reference="stance" if stance is not None else "nominal")
            emit(self.output / "single_view_rubble.jsonl", comparison)
            completion = self.single_view_rubble.completion
            if ranging is not None and len(candidate_xy) and (completion is None or completion.revision is None):
                row["single_view_rubble"] = candidate_xy
                if "estimated_relief_m" in comparison:
                    row["completion_relief"] = np.asarray(comparison["estimated_relief_m"],dtype=float)
                    row["completion_source"] = comparison["completion_source"]
        if row['rough'] or 'visual_height' in row or 'dense_surface' in row or 'single_view_rubble' in row:
            self.history.append(row)
        emit(self.output / "rubble_observations.jsonl", {"t": frame.timestamp, "regions": observations})
        self.ball_candidates += len(detections)
        self.latest_detections = detections
        self.latest_confirmations = []
        if valid_pose:
            eligible = [d for d in detections if confirmable_ball(d, self.ball_min_confidence)]
            events = self.tracker.update(eligible, pose, frame.timestamp)
            by_id = {track.id: track for track in self.tracker.tracks}
            for event in events:
                track = by_id[event.payload["id"]]
                d = track.observation
                item = {"id": track.id, "t": frame.timestamp, "center_px": list(d.center),
                        "radius_px": d.radius, "range_m": float(np.linalg.norm(d.position-pose.xy())),
                        "source": d.range_source, "map_xy": event.payload["position"], "hits": track.hits}
                self.confirmation_frames[track.id] = item
                self.latest_confirmations.append(item)
        boxes = []
        for d in detections:
            center = tuple(np.rint(d.center).astype(int))
            if d.position is not None and d.confidence >= self.ball_min_confidence:
                cv2.circle(front, center, int(round(d.radius)) + 3, (255, 220, 30), 2)
            boxes.append({"center_px": d.center, "radius_px": d.radius, "confidence": d.confidence,
                          "source": d.range_source, "map_xy": d.position.tolist()
                          if d.position is not None and valid_pose else None})
        self.target_observation = {
            "time_s": float(frame.timestamp),
            "frame": "camera_image",
            "image_size_wh": [int(v) for v in frame.image.shape[:2][::-1]],
            "detections": [
                {"center_px": [float(v) for v in d.center], "radius_px": float(d.radius),
                 "confidence": float(d.confidence), "range_source": d.range_source,
                 "range_from_base_xy_m": float(np.linalg.norm(d.position - pose.xy()))
                 if d.position is not None and valid_pose else None}
                for d in detections
            ],
        }
        tracks = [{"id": t.id, "seen": t.last_seen, "map_xy": t.position.tolist(),
                   "reported": t.reported, "hits": t.hits} for t in self.tracker.tracks]
        provisional = [{"map_xy": d.position.tolist(), "center_px": list(d.center), "sigma_m": d.sigma}
                       for d in self.hybrid_ball_range.provisional] if self.hybrid_ball and valid_pose else []
        emit(self.output / "target_tracks.jsonl", {"t": frame.timestamp, "detections": boxes, "tracks": tracks,
                                                    "provisional_flat_hypotheses": provisional,
                                                    "new_confirmations": self.latest_confirmations,
                                                    "confirmed_map_xy": [p.tolist() for p in self.tracker.reported_positions]})
        self.last_front = front

    def snapshot(self, stamp):
        grid = self.grid
        confirmed_frames = []
        visual = Counter()
        all_evidence = Counter()
        views = defaultdict(set)
        visual_heights = defaultdict(dict)
        dense_views = defaultdict(set)
        dense_hits = Counter()
        for row in self.history:
            view = tuple(np.floor(row['pose'].xy() / .25).astype(int))
            visual_cells, confirmed_cells = set(), set()
            for points, reliable in row["rough"]:
                cells = {grid.cell(p) for p in transform_xy(points, row["pose"])}
                (confirmed_cells if reliable else visual_cells).update(cells)
            visual.update(visual_cells)
            all_evidence.update(visual_cells | confirmed_cells)
            for cell in visual_cells | confirmed_cells:
                views[cell].add(view)
            if 'visual_height' in row:
                xy, values = row['visual_height']
                viewpoint = tuple(np.floor(row['pose'].xy() / .15).astype(int))
                for point, height in zip(transform_xy(xy, row['pose']), values):
                    visual_heights[grid.cell(point)].setdefault(viewpoint, []).append(height)
            if 'dense_surface' in row:
                xy, values = row['dense_surface']
                viewpoint = tuple(np.floor(row['pose'].xy() / .15).astype(int))
                cells = [grid.cell(p) for p in transform_xy(xy, row['pose'])]
                dense_hits.update(set(cells))
                for cell, height in zip(cells, values):
                    dense_views[cell].add(viewpoint)
                    if np.isfinite(height):
                        visual_heights[cell].setdefault(viewpoint, []).append(height)
            confirmed_frames.append(confirmed_cells)
        for cell, view_bins in dense_views.items():
            if len(view_bins) >= 2:
                visual[cell] = max(visual[cell], dense_hits[cell])
                all_evidence[cell] = max(all_evidence[cell], dense_hits[cell])
        confirmed = spatial_frame_support(confirmed_frames)
        heights = collect_cell_heights(grid, (
            (xy, residual, row["pose"]) for row in self.history
            for xy, residual in row.get("heights", ())), transform_xy)
        navigator = self.autonomy.known_map_navigator
        self.terrain_layer = terrain_snapshot(
            grid, visual, confirmed, heights, stamp, navigator.grid if navigator is not None else None,
            views=None if self.slope_perception else views, visual_heights=visual_heights,
            reconstructed=self.slope_perception,
            visual_height_quantile=.5 if self.visual_height.multiview else .8)
        if self.visual_height.dense or self.visual_height.multiview:
            mode = 'rgb_multiview' if self.visual_height.multiview else 'rgb_two_view'
            self.terrain_layer['visual_surface_mode'] = mode
            self.terrain_layer['height_view_quantile'] = .5 if self.visual_height.multiview else .8
            for item in self.terrain_layer['cells']:
                cell = tuple(item['ij'])
                item['rgb_dense_view_bins'] = len(dense_views.get(cell, ()))
                item['rgb_dense_frames'] = dense_hits[cell]
                if self.visual_height.multiview and 'height_estimated_m' in item:
                    item['height_estimate_source'] = 'rgb_surface_local_support'
        if self.range_evidence is not None:
            evidence = self.range_evidence
            np.savez_compressed(self.output / "lidar_range_evidence.npz",
                                ray_prefix=evidence.ray_prefix, possible_hit=evidence.possible_hit,
                                unambiguous_hit=evidence.unambiguous_hit,
                                ground_band_clear=evidence.ground_band_clear,
                                origin=grid.origin, resolution=self.c.resolution)
            for item in self.terrain_layer["cells"]:
                item["lidar_ray_free"] = None  # range-only evidence has no height label
                item["lidar_near_ground_band_sampled"] = bool(evidence.ground_band_clear[tuple(item["ij"])])
            self.terrain_layer["lidar_mode"] = "range_only_evidence_estimated_pose"
        if self.rubble_precision is not None:
            from .rubble_precision_layer import snapshot_precision
            snapshot_precision(self.terrain_layer,self.history,grid)
        completion = self.single_view_rubble.completion if self.single_view_rubble is not None else None
        revision = completion.revision if completion is not None else None
        if self.rubble_footprint is not None:
            revision = self.rubble_footprint.revision
        if revision is not None:
            self.terrain_layer["single_view_rubble"] = revision.snapshot(grid)
        elif self.single_view_rubble is not None:
            frame_votes = Counter()
            relief_votes = defaultdict(list)
            source_votes = defaultdict(set)
            for row in self.history:
                if "single_view_rubble" in row:
                    xy = transform_xy(row["single_view_rubble"],row["pose"])
                    cells = [grid.cell(point) for point in xy]
                    frame_votes.update({cell for cell in cells if grid.inside(cell)})
                    available = defaultdict(list)
                    for cell, height, source in zip(cells,row.get("completion_relief",[]),row.get("completion_source",[])):
                        if grid.inside(cell):
                            source_votes[cell].add(source)
                            if np.isfinite(height):
                                available[cell].append(height)
                    for cell, heights in available.items():
                        relief_votes[cell].append(float(np.median(heights)))
            self.terrain_layer["single_view_rubble"] = {
                "status": "range_supported_depth_completion" if self.single_view_rubble.completion is not None
                          else "camera_mask_range_supported_candidates",
                "cells": [{"ij":list(cell), "xy_m":grid.world(cell).tolist(), "frames":hits,
                           "depth_sources":sorted(source_votes[cell]),
                           "range_or_repeat_support":1 in source_votes[cell] or hits >= 2,
                           "height_observations":len(relief_votes[cell]),
                           "height_above_local_ground_m":float(np.median(relief_votes[cell]))
                           if len(relief_votes[cell]) >= 2 else None} for cell,hits in sorted(frame_votes.items())],
            }
        write_terrain_snapshot(self.output / "terrain_layer.json", self.terrain_layer)
        snapshot = perception_snapshot(grid, self.terrain_layer, self.tracker.reported_positions,
                                       stamp, detections=self.target_observation)
        publish_perception_snapshot(self.output / "perception_snapshot.json", snapshot)
        base = np.full((*grid.seen.shape, 3), 85, np.uint8)
        base[grid.seen] = (210, 210, 210)
        base[grid.floor_source == 1] = (185, 185, 185)
        base[grid.floor_source == 2] = (160, 220, 175)
        base[grid.floor_source == 3] = (135, 185, 155)
        base[grid.logodds > .5] = (20, 20, 20)
        if self.slope_perception:
            # Never display the legacy 3D map as an Agent observation product.
            base[:] = 85
            if self.range_evidence is not None:
                base[self.range_evidence.ray_prefix] = (135, 135, 135)
                base[self.range_evidence.possible_hit] = (145, 125, 90)
                base[self.range_evidence.unambiguous_hit] = (103, 70, 35)
                base[self.range_evidence.ground_band_clear] = (150, 180, 140)
        base[grid.hazard_votes >= 2]=(60,65,220)
        overlay = base.copy()
        for item in self.terrain_layer["cells"] if revision is None else []:
            color = (0, 175, 255)
            if item['height_observed_min_max_m'] is not None:
                color = (40, 175, 30)
            elif 'height_estimated_m' in item:
                color = (185, 95, 182)
            elif item.get('inferred_footprint'):
                color = (100, 190, 240)
            overlay[tuple(item["ij"])] = color
        if revision is not None:
            for item in self.terrain_layer["single_view_rubble"]["cells"]:
                overlay[tuple(item["ij"])] = (85,205,103) if item["range_or_repeat_support"] else (85,157,110)
        pose_cell = grid.cell(self.loc.pose.xy())
        trail = np.asarray([grid.cell(key.pose.xy())[::-1] for key in self.loc.keyframes] +
                           [pose_cell[::-1]], dtype=np.int32)
        if len(trail) >= 2:
            cv2.polylines(overlay, [trail], False, (200, 90, 30), 1)
        cv2.circle(overlay, pose_cell[::-1], 2, (230, 90, 30), -1)
        for point in self.tracker.reported_positions:
            cv2.drawMarker(overlay, grid.cell(point)[::-1], (255, 170, 0), cv2.MARKER_STAR, 9, 1)
        seen = np.argwhere(grid.seen)
        color_cells = np.argwhere(np.any(overlay != base, axis=2))
        bounds = np.concatenate([seen, color_cells, np.array([pose_cell])])
        lo = np.maximum(0, bounds.min(axis=0) - 6)
        hi = np.minimum(grid.n, bounds.max(axis=0) + 7)
        crop = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
        base = cv2.flip(base[crop], 0)
        overlay = cv2.flip(overlay[crop], 0)
        write_png(self.output / "map.png", cv2.resize(base, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        self.map_view = overlay.copy()
        self.map_origin_cell = lo.copy()
        write_png(self.output / "perception_overlay.png", cv2.resize(overlay, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        panel = np.full((480, 1056, 3), 28, np.uint8)
        front = self.last_front
        if front is not None:
            panel[34:274, 12:332] = cv2.resize(front, (320, 240))
        scan = np.rint(-self.lidar[:, [1, 0]] * 24 + [155, 85]).astype(int)
        patch = panel[290:468, 12:332]
        cv2.circle(patch, (155, 85), 3, (0, 130, 255), -1)
        for x, y in scan:
            cv2.circle(patch, (int(x), int(y)), 2, (200, 225, 225), -1)
        factor = min(690 / overlay.shape[1], 410 / overlay.shape[0])
        scaled = cv2.resize(overlay, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        panel[34:34+scaled.shape[0], 356:356+scaled.shape[1]] = scaled
        terrain_legend = [
            ("Amber: observed | pale: inferred footprint", (356, 466), (0, 175, 255)),
            ("Green: measured | purple: estimated height", (680, 466), (185, 95, 182)),
        ] if revision is None else [
            ("Bright green: supported rubble", (356, 466), (85, 205, 103)),
            ("Dim green: provisional rubble", (680, 466), (85, 157, 110)),
        ]
        for label, position, color in [
            ("Front RGB and detected regions", (12, 23), (225, 225, 225)),
            (f"LiDAR: {len(self.lidar)}/72", (12, 292), (225, 225, 225)),
            (f"{'Range evidence' if self.slope_perception else 'Estimated map'}  |  t={stamp:.1f}s",
             (356, 23), (225, 225, 225)),
            *terrain_legend,
        ]:
            cv2.putText(panel, label, position, cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1, cv2.LINE_AA)
        write_png(self.output / "perception_live.png", panel)
        repeats = 1 if self.last_video is None else max(1, round((stamp - self.last_video) * 5))
        for _ in range(repeats):
            self.video.write(panel)
        self.last_video = stamp
        self.counts = {"rough_visual_cells": sum(n >= 3 and grid.inside(c) for c, n in all_evidence.items()),
                       "rough_lidar_cells": sum(n >= 2 and grid.inside(c) for c, n in confirmed.items()),
                       "rough_inferred_footprint_cells": sum(bool(item.get('inferred_footprint')) for item in self.terrain_layer['cells']),
                       "rough_estimated_height_cells": sum('height_estimated_m' in item for item in self.terrain_layer['cells']),
                       "confirmed_targets": [p.tolist() for p in self.tracker.reported_positions],
                       "occupied_cells": int((grid.logodds > .5).sum()), "seen_cells": int(grid.seen.sum()),
                       "terrain_height_observations": self.terrain_observations,
                       "low_map_frame_support_radius_m": self.c.resolution * 2**.5,
                       "map_resolution_m": self.c.resolution, "floor_sources": np.bincount(
                           grid.floor_source.ravel(), minlength=4).tolist()}
        if revision is not None:
            layer = self.terrain_layer["single_view_rubble"]
            self.counts.update(rubble_active_cells=len(layer["cells"]),
                               rubble_supported_cells=sum(cell["range_or_repeat_support"] for cell in layer["cells"]),
                               rubble_withdrawals=revision.withdrawn)

    def finish(self, stamp):
        try:
            self.snapshot(stamp)
            filename = "diagnostic_legacy_xyz_map.npz" if self.slope_perception else "map.npz"
            self.grid.save(self.output / filename)
            (self.output / "terrain_heights.json").write_text(json.dumps(self.terrain_samples, indent=2))
        finally:
            self.video.release()
        return dict(self.counts, camera_frames=self.camera_frames, ball_candidates=self.ball_candidates,
                    rough_components=self.visual_components, lidar_supported_components=self.geometry_frames,
                    lidar_contract_violations=self.violations)
