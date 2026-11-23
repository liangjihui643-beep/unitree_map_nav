from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import base64
import json
import socket
import struct

import cv2
import numpy as np

from .types import Frame, Velocity


class Recorder:
    def __init__(self, directory):
        self.root = Path(directory)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("Recording destination must be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = (self.root / "frames.jsonl").open("w")
        self.index = 0

    def write(self, frame: Frame):
        frame.validate()
        name = f"frame_{self.index:06d}.npz"
        arrays = {"points": frame.points, "quaternion": frame.quaternion}
        if frame.support_points is not None:
            arrays["support_points"] = frame.support_points
        if frame.image is not None:
            arrays["image"] = frame.image
        if frame.point_times is not None:
            arrays["point_times"] = frame.point_times
        np.savez_compressed(self.root / name, **arrays)
        record = {"schema": "go2-rescue-frame/1", "file": name, "timestamp": frame.timestamp,
                  "velocity": asdict(frame.velocity), "state_age": frame.state_age, "image_age": frame.image_age}
        self.manifest.write(json.dumps(record, allow_nan=False) + "\n")
        self.manifest.flush()
        self.index += 1

    def close(self):
        self.manifest.close()


def replay(directory):
    root = Path(directory).resolve()
    with (root / "frames.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("schema") != "go2-rescue-frame/1":
                raise ValueError("Unknown recording schema")
            path = (root / record["file"]).resolve()
            if path.parent != root or path.suffix != ".npz":
                raise ValueError("Recording path escapes recording directory")
            with np.load(path, allow_pickle=False) as data:
                frame = Frame(record["timestamp"], data["points"], data["quaternion"],
                              Velocity(**record["velocity"]),
                              data["image"] if "image" in data else None,
                              data["point_times"] if "point_times" in data else None,
                              record.get("state_age", 0.), record.get("image_age", 0.),
                              data["support_points"] if "support_points" in data else None)
            frame.validate()
            yield frame


def decode_packet(packet: bytes) -> Frame:
    message = json.loads(packet)
    if message.get("schema") != "go2-rescue-frame/1" or message.get("point_frame") != "body_flu":
        raise ValueError("Bridge requires schema go2-rescue-frame/1 and point_frame body_flu")
    if message.get("clock") != "host_monotonic":
        raise ValueError("Live bridge timestamps must use this host's monotonic clock")
    image = None
    if message.get("image_jpeg"):
        image = cv2.imdecode(np.frombuffer(base64.b64decode(message["image_jpeg"], validate=True), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid JPEG in sensor bridge")
    frame = Frame(float(message["timestamp"]), np.asarray(message["points"], dtype=float),
                  np.asarray(message["quaternion_wxyz"], dtype=float), Velocity(**message["velocity"]),
                  image, np.asarray(message["point_times"], dtype=float) if "point_times" in message else None,
                  float(message["state_age"]), float(message.get("image_age", 0.)))
    frame.validate()
    return frame


def socket_frames(endpoint: str):
    """Client for an on-host sensor driver bridge, network byte order length + JSON."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
        raise ValueError("Sensor bridge must be tcp://127.0.0.1:PORT on the same host")
    with socket.create_connection((parsed.hostname, parsed.port), timeout=.3) as connection:
        connection.settimeout(.3)
        def receive(count):
            pieces = bytearray()
            while len(pieces) < count:
                chunk = connection.recv(count - len(pieces))
                if not chunk:
                    raise EOFError("Sensor bridge disconnected")
                pieces.extend(chunk)
            return bytes(pieces)
        while True:
            size = struct.unpack("!I", receive(4))[0]
            if size < 2 or size > 16 * 1024 * 1024:
                raise ValueError("Sensor bridge frame outside size limits")
            yield decode_packet(receive(size))
