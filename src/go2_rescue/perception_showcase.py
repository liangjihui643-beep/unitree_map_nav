"""Presentation and independent pixel-level checks, downstream of perception."""
from collections import Counter
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .perception_demo import emit, write_png


INK = (238, 244, 248)
BLUE = (255, 201, 74)
GOLD = (42, 182, 255)


def text(image, label, x, y, size=.63, color=INK):
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def fit_to(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


class Showcase:
    def __init__(self, output, runtime, source, spawn, yaw):
        self.output = Path(output)
        self.ball = mujoco.mj_name2id(runtime, mujoco.mjtObj.mjOBJ_GEOM, "local_football_geom")
        self.rocks = []
        self.walls = []
        for index in range(source.ngeom):
            name = source.body(int(source.geom_bodyid[index])).name
            geom = mujoco.mj_name2id(runtime, mujoco.mjtObj.mjOBJ_GEOM, f"course_geom_{index}")
            if geom < 0:
                geom = index if runtime is source or runtime.ngeom == source.ngeom else -1
            if geom < 0:
                continue
            if name.startswith("pile"):
                self.rocks.append(geom)
            if any(token in source.geom(index).name.lower() for token in ("wall", "partition")):
                self.walls.append(geom)
        self.score = Counter()
        self.mask_directory = self.output / "evaluation_masks"
        self.mask_directory.mkdir()
        self.confirmations = []
        self.height_evidence = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [spawn[0] + 1.8, spawn[1] + .1, 0.]
        self.camera.distance = 5.6
        self.camera.azimuth = 90 + np.rad2deg(yaw)
        self.camera.elevation = -56.
        self.last_overview = None
        self.next_overview = 0.
        self.video = cv2.VideoWriter(str(self.output / "showcase.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     5., (1280, 720))
        if not self.video.isOpened():
            raise RuntimeError("Cannot write showcase video")
        self.video_frames = 0
        self.proof = None

    @staticmethod
    def overlap(center, radius, mask):
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        disk = (x-center[0])**2 + (y-center[1])**2 <= (radius + 2)**2
        inter = (disk & mask).sum()
        union = (disk | mask).sum()
        return float(inter / max(1, union))

    def observe(self, frame, demo, sensors):
        from time import perf_counter
        start = perf_counter()
        ids = sensors.segmentation()
        geom = np.where(ids[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, ids[:, :, 0], -1)
        ball = geom == self.ball if self.ball >= 0 else np.zeros(geom.shape, dtype=bool)
        rocks = np.isin(geom, self.rocks)
        walls = np.isin(geom, self.walls)
        pred = demo.front_rubble
        visible = int(ball.sum()) >= 30
        scores = [self.overlap(d.center, d.radius, ball) for d in demo.latest_detections if d.confidence >= demo.ball_min_confidence]
        matched = bool(visible and max(scores, default=0.) >= .45)
        values = {"rgb_frames": 1, "ball_visible_frames": int(visible), "ball_hit_frames": int(matched),
                  "rock_tp": int((pred & rocks).sum()), "rock_fp": int((pred & ~rocks).sum()),
                  "rock_fn": int((~pred & rocks).sum()), "rock_on_wall_pixels": int((pred & walls).sum()),
                  "wall_visible_pixels": int(walls.sum())}
        mask_name = f"{frame.timestamp:.6f}.png"
        write_png(self.mask_directory / mask_name, rocks.astype(np.uint8) + 2*ball.astype(np.uint8)
                  + 4*walls.astype(np.uint8) + 8*pred.astype(np.uint8))
        self.score.update(values)
        emit(self.output / "pixel_acceptance.jsonl", {"t": frame.timestamp, "visible_ball_pixels": int(ball.sum()),
                                                      "best_ball_mask_iou": max(scores, default=0.), "mask": mask_name, "audit_seconds": perf_counter()-start, **values})
        if frame.timestamp >= self.next_overview - 1e-6:
            # Read-only presentation camera, downstream of all estimation.
            self.camera.lookat[:]=sensors.data.xpos[sensors.base]
            sensors.renderer.update_scene(sensors.data, camera=self.camera, scene_option=sensors.option)
            self.last_overview = cv2.cvtColor(sensors.renderer.render(), cv2.COLOR_RGB2BGR)
            self.next_overview = frame.timestamp + .6
        if demo.terrain_samples and self.height_evidence is None:
            sample = demo.terrain_samples[-1]
            if abs(sample["t"] - frame.timestamp) < 1e-6:
                board = demo.last_front.copy()
                text(board, "Laser + joint/IMU support", 5, 20, .43, (40, 230, 90))
                text(board, f"observed relief {min(sample['height_m']):.2f}..{max(sample['height_m']):.2f} m",
                     5, 42, .43, (40, 230, 90))
                write_png(self.output / "terrain-evidence.png", board)
                self.height_evidence = sample["t"]
        latest = demo.latest_confirmations
        if latest:
            demo.snapshot(frame.timestamp)
            for item in latest:
                mask_iou = self.overlap(item["center_px"], item["radius_px"], ball)
                self.confirmations.append(dict(item, true_ball_mask_iou=mask_iou, visible_ball_pixels=int(ball.sum())))
                if self.proof is None and mask_iou >= .45:
                    self.proof = self.make_proof(frame.image, demo, item, mask_iou)
                    write_png(self.output / "ball-evidence.png", self.proof)
        if self.last_overview is None:
            return
        board = np.full((720, 1280, 3), (30, 27, 24), np.uint8)
        text(board, "GO2  /  MOVING PERCEPTION", 24, 34, .88)
        text(board, f"SIM TIME {frame.timestamp:04.1f} s   |   CAMERA, LiDAR & ESTIMATED MAP", 24, 66, .52)
        self.picture(board, self.last_overview, 24, 107, 396, 220)
        self.picture(board, demo.last_front, 24, 390, 396, 265)
        text(board, "1  MuJoCo: watch the robot move", 24, 94)
        text(board, "2  Front camera: ball and rubble", 24, 375)
        text(board, "3  Map: estimated route and objects", 466, 95, .72)
        self.draw_map(board, demo, (467, 109, 782, 490))
        text(board, "COURSE SCAN", 28, 685, .48)
        text(board, f"{len(frame.points)}/72 returns", 160, 685, .52)
        text(board, "Blue line: robot route", 469, 630, .56, BLUE)
        text(board, "Orange: RGB rubble", 720, 630, .52, GOLD)
        text(board, "Green: laser height", 1005, 630, .52, (40, 200, 90))
        text(board, "Black: occupied   |   light gray: ray observed   |   dark gray: unknown", 469, 660, .52)
        if self.confirmations:
            first = self.confirmations[0]
            label = f"Ball B{first['id']} | at {first['t']:.1f}s | range {first['range_m']:.2f} m (flat-ground model)"
        else:
            label = "Waiting for consistent ball detections before placing a star"
        text(board, label, 469, 691, .56, BLUE)
        write_png(self.output / "showcase_live.png", board)
        desired = round(frame.timestamp * 5)
        for _ in range(max(1, desired - self.video_frames)):
            self.video.write(board)
            self.video_frames += 1

    @staticmethod
    def picture(board, image, x, y, w, h):
        image = fit_to(image, w, h)
        xp = x + (w-image.shape[1])//2
        board[y:y+image.shape[0], xp:xp+image.shape[1]] = image
        return xp, y, image.shape[1], image.shape[0]

    def draw_map(self, board, demo, box):
        xp, yp, w, h = self.picture(board, demo.map_view, *box)
        for item, point in zip(demo.confirmation_frames.values(), demo.tracker.reported_positions):
            gx, gy = np.array(demo.grid.cell(point)) - demo.map_origin_cell
            px = int(xp + (gy+.5)*w/demo.map_view.shape[1])
            py = int(yp + (demo.map_view.shape[0]-gx-.5)*h/demo.map_view.shape[0])
            cv2.drawMarker(board, (px, py), BLUE, cv2.MARKER_STAR, 23, 2)
            text(board, f"B{item['id']}", px+12, py-10, .67, BLUE)

    def make_proof(self, image, demo, item, iou):
        board = np.full((720, 1280, 3), (30, 27, 24), np.uint8)
        cx, cy = map(int, item["center_px"])
        radius = int(np.ceil(item["radius_px"]))
        margin = max(18, 2*radius)
        crop = image[max(0, cy-margin):cy+margin+1, max(0, cx-margin):cx+margin+1]
        outlined = image.copy()
        cv2.circle(outlined, (cx, cy), radius+4, BLUE, 2)
        text(board, f"BALL B{item['id']}  /  CAMERA TO MAP", 28, 43, .9)
        text(board, f"3. Estimated map at {item['t']:.1f} s", 500, 99)
        text(board, "1. Original camera frame", 28, 99)
        text(board, "2. Raw pixels enlarged", 28, 380)
        self.picture(board, outlined, 28, 118, 435, 230)
        self.picture(board, crop, 28, 400, 435, 228)
        self.draw_map(board, demo, (500, 118, 752, 474))
        text(board, f"Track B{item['id']}  |  {item['hits']} frames  |  estimated range {item['range_m']:.2f} m", 28, 662)
        text(board, f"Range: {item['source'].replace("_", " ")}   |   true-ball mask IoU: {iou:.0%}", 28, 697, .59)
        return board

    def finish(self):
        if self.proof is not None:
            for _ in range(18):
                self.video.write(self.proof)
        self.video.release()
        t = self.score
        report = dict(t, ball_confirmations=self.confirmations,
                      ball_visible_recall=t["ball_hit_frames"]/max(1, t["ball_visible_frames"]),
                      rock_precision=t["rock_tp"]/max(1, t["rock_tp"]+t["rock_fp"]),
                      rock_recall=t["rock_tp"]/max(1, t["rock_tp"]+t["rock_fn"]))
        (self.output / "showcase_acceptance.json").write_text(json.dumps(report, indent=2))
        return report
