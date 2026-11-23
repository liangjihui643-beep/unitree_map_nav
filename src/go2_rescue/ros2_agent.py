"""ROS 2 autonomy-side runner through the socket bridge.

Usage:
    # terminal 1 (ROS 2 bridge, python3.8):
    source /opt/ros/foxy/setup.bash
    python3 -m go2_rescue.ros2_bridge --sim-port 8765 --agent-port 8766
    # terminal 2 (autonomy, python3.12 venv):
    go2-rescue ros2-agent --config <cfg.json> --output <dir> --seconds 20
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .controller import Autonomy
from .ros2_bridge import AgentSocketClient
from .types import Frame, Velocity


def run_agent(config_path=None, output="output/ros2_agent", seconds=20.0, rate=20.0, port=8766):
    config = Config.load(config_path) if config_path else Config()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    client = AgentSocketClient(port=port)
    autonomy = Autonomy(config)
    wall_deadline = time.monotonic() + max(60.0, seconds * 60.0)
    start_scan = None
    n = 0
    t_scan = None
    try:
        with (out / "agent_commands.jsonl").open("w") as log:
            while time.monotonic() < wall_deadline:
                obs = client.recv_obs(timeout=1.0)
                if obs is None:
                    continue
                t_scan = float(obs["t"])
                if start_scan is None:
                    start_scan = t_scan
                if t_scan - start_scan > seconds:
                    break
                pts = np.asarray(obs.get("points") or [], float).reshape(-1, 3)
                img = None
                if obs.get("image_b64"):
                    buf = np.frombuffer(base64.b64decode(obs["image_b64"]), np.uint8)
                    img = buf.reshape(int(obs["image_h"]), int(obs["image_w"]), 3)
                frame = Frame(t_scan, pts, np.asarray(obs.get("quat") or [1., 0., 0., 0.], float),
                              Velocity(*(obs.get("vel") or [0., 0., 0.])), img)
                step = autonomy.step(frame)
                v = step.velocity
                client.send_cmd(v.vx, v.vy, v.wz)
                n += 1
                log.write(json.dumps({
                    "t": t_scan, "scan_points": int(len(pts)),
                    "command": {"vx": v.vx, "vy": v.vy, "wz": v.wz},
                    "state": step.state, "reason": step.reason,
                }) + "\n")
                time.sleep(max(0.0, 1.0 / max(1.0, rate) - 0.001))
    finally:
        client.sock.close()
    summary = {"status": "done", "observations": n,
               "sim_seconds": (t_scan - start_scan) if (t_scan is not None and start_scan is not None) else 0.0,
               "output": str(out)}
    (out / "agent_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    return summary
