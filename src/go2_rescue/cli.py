from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import Config
from .controller import Autonomy
from .motion import DryRunMotion, UnitreeMotion
from .recording import Recorder, replay, socket_frames
from .synthetic import SensorWorld


def step_record(step, timestamp, duration):
    return {"timestamp": timestamp, "state": step.state, "velocity": asdict(step.velocity),
            "pose": asdict(step.pose), "reason": step.reason,
            "events": [asdict(e) for e in step.events], "processing_seconds": duration,
            "diagnostics": step.diagnostics}


def save_map(autonomy: Autonomy, output: Path):
    autonomy.grid.save(output / "map.npz")
    summary = autonomy.grid.evidence_summary()
    if summary:
        (output / "map_evidence.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if getattr(autonomy.grid, "wall_diagnostics", None):
        with (output / "map_wall_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
            for item in autonomy.grid.wall_diagnostics:
                stream.write(json.dumps(item) + "\n")
    grid = autonomy.grid
    canvas = np.full((*grid.seen.shape, 3), 125, np.uint8)
    canvas[grid.seen] = (225, 225, 225)
    canvas[grid.traversable()] = (190, 230, 190)
    canvas[grid.logodds > .5] = (30, 30, 30)
    cell = grid.cell(autonomy.localization.pose.xy())
    if grid.inside(cell):
        cv2.circle(canvas, (cell[1], cell[0]), 3, (0, 0, 255), -1)
    cv2.imwrite(str(output / "map.png"), cv2.flip(canvas, 0))


def run(args):
    config = Config.load(args.config)
    # Refuse incompatible options before loading SDK or connecting any source.
    if args.execute and args.mode not in {"live", "native"}:
        raise ValueError("--execute is available only with live sensor input")
    if args.execute and (not config.calibrated or config.hardware_model == "unknown" or not config.network_interface):
        raise ValueError("Verify hardware_model, network_interface and calibration before live execution")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    world = SensorWorld(config, args.seed, args.mission_fixture) if args.mode == "demo" else None
    if world:
        frames = (world.frame() for _ in range(args.steps))
    elif args.mode == "replay":
        frames = replay(args.input)
    elif args.mode == "native":
        from .unitree_sensors import native_frames
        frames = native_frames(config, args.lidar_provider, args.sport_velocity_frame)
    else:
        frames = socket_frames(args.endpoint)
    autonomy = Autonomy(config)
    motion = UnitreeMotion(config, execute=True) if args.execute else DryRunMotion()
    recorder = Recorder(output / "recording") if args.record else None
    count, times, event_count, command_count = 0, [], 0, 0
    status, error = "complete", None
    try:
        with (output / "trajectory.jsonl").open("w") as log:
            for frame in frames:
                received = time.monotonic()
                if args.mode in {"live", "native"}:
                    age = received - frame.timestamp
                    if age < -.02 or age > config.max_sensor_age:
                        raise ValueError("Live lidar timestamp stale or not in this host's monotonic clock")
                    frame.state_age += age
                    frame.image_age += age
                if recorder:
                    recorder.write(frame)
                begin = time.monotonic()
                step = autonomy.step(frame)
                duration = time.monotonic() - begin
                # A newly computed command can already be stale. The SDK watchdog
                # covers stalled computation; this check covers the first packet.
                if args.execute and time.monotonic() - frame.timestamp > config.max_sensor_age:
                    from .types import Velocity
                    step.velocity, step.reason = Velocity(), "processing_deadline_missed"
                motion.send(step.velocity)
                log.write(json.dumps(step_record(step, frame.timestamp, duration), allow_nan=False) + "\n")
                log.flush()
                times.append(duration)
                event_count += len(step.events)
                command_count += int(np.linalg.norm(step.velocity.array()) > 0)
                count += 1
                if world:
                    world.advance(step.velocity)
                if count >= args.steps or step.state in {"finished", "timed_out"}:
                    break
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            motion.close()
        finally:
            if hasattr(frames, "close"):
                frames.close()
            if recorder:
                recorder.close()
            save_map(autonomy, output)
            summary = {"status": status, "error": error, "mode": args.mode, "robot_commands_enabled": args.execute,
                       "frames": count, "nonzero_commands": command_count, "events": event_count,
                       "state": autonomy.state, "processing_p50_seconds": float(np.median(times)) if times else None,
                       "processing_p95_seconds": float(np.quantile(times, .95)) if times else None,
                       "evidence": "synthetic kinematic software fixture only" if world else "sensor replay/live; not a competition score"}
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description="Go2 V1: classical autonomy, no ROS/RL/training dependency")
    sub = parser.add_subparsers(dest="mode", required=True)
    init = sub.add_parser("init-config", help="Write uncalibrated hardware template")
    init.add_argument("path")
    marker = sub.add_parser("finish-marker", help="Generate a local test ArUco marker; not an official competition marker")
    marker.add_argument("path")
    marker.add_argument("--id", type=int, default=42)
    sim = sub.add_parser("mujoco", help="Original/local MuJoCo scenes + pinned open-source convex MPC")
    sim.add_argument("--output", required=True)
    sim.add_argument("--config")
    sim.add_argument("--seconds", type=float, default=600., help="Simulation time limit (default600s for ordered course); a finished mission may stop earlier")
    sim.add_argument("--spawn", type=float, nargs=2, default=[.8, .8], metavar=("X", "Y"),
                     help="Initial reset XY; default 0.8 0.8 before the ramp route. Other locations are labelled stage tests")
    sim.add_argument("--spawn-yaw", type=float, default=0., help="Initial heading in radians; scene reset only")
    sim.add_argument("--control", choices=["autonomy", "smoke", "ramp-smoke", "speed-smoke", "perception-walk", "perception-survey"], default="autonomy")
    sim.add_argument("--max-speed", type=float, help="Forward speed limit in m/s for ordinary walking and narrow passages; overrides config")
    sim.add_argument("--max-reverse-speed", type=float, help="Independent reverse limit in m/s; simulation default .25, overrides config")
    sim.add_argument("--course-policy", choices=["auto", "ordered", "explore"], default="auto",
                     help="Ordered fixed-map stages by default; explore is a diagnostic mode")
    sim.add_argument("--course-route", help="Explicit ordered map_start route JSON; never simulator state")
    sim.add_argument("--task-fixture", action="store_true", help="Local football/finish scene; entry-ramp variant by default, --pipe-ramp official preserves source geometry")
    sim.add_argument("--pipe-ramp", choices=["official", "entry"],
                     help="Task fixture ramp layout: entry(default with --task-fixture) moves the exit ramp to the west entrance; official preserves source geometry")
    sim.add_argument("--task-layout", choices=["finish-area", "near-start", "perception"], default="finish-area",
                     help="Place the task in the far finish area (default); near-start is the old short regression fixture")
    sim.add_argument("--viewer", action="store_true", help="Open MuJoCo interactive viewer")
    sim.add_argument("--hide-camera", action="store_true", help="Hide the default live front-camera inset in the viewer")
    sim.add_argument("--close-viewer-on-stop", action="store_true", help="Close viewer after the run; by default keep the final scene open")
    sim.add_argument("--realtime", action="store_true")
    sim.add_argument("--record", action="store_true", help="Opt in to raw sensor recording (off by default)")
    sim.add_argument("--video", action="store_true", help="Opt in to offscreen video encoding (off by default)")
    sim.add_argument("--video-fps", type=float, default=10., help="Recorded overview frames per simulation second (1..30); sensors unchanged")
    sim.add_argument("--save-physics", action="store_true", help="Opt in to sampled physics.npz diagnostics")
    sim.add_argument("--snapshots", action="store_true", help="Opt in to map/image snapshots")
    sim.add_argument("--low-obstacle", action="store_true",
                     help="Bypass observation: run the RGB low-obstacle model on live frames; never fed to the planner")
    sim.add_argument("--low-obstacle-model", default=None, help="Path to the low-obstacle ONNX (default models/seg_low_obstacle_v8.onnx; perception demo uses v10)")
    sim.add_argument("--low-obstacle-every", type=float, default=.5, help="Seconds between low-obstacle inferences")
    sim.add_argument("--camera-every", type=int, default=1,
                     help="Render the front camera every N-th nav step (1=10Hz); decoding/ball roles assume 1")
    sim.add_argument("--camera-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                     help="Profiling override for the render resolution; scales nothing, so geometry roles must keep 320x240")
    sim.add_argument("--ros2", action="store_true",
                     help="Transport observations/commands over the ROS 2 bridge; commands come from an external agent (go2-rescue ros2-agent)")
    sim.add_argument("--ros2-sim-port", type=int, default=8765, help="Bridge port the simulator connects to")
    sim.add_argument("--viewer-fps", type=float, default=60., help="Maximum viewer sync rate, independent of sensor/video frequency")
    sim.add_argument("--coverage", action="store_true", help="Autonomous observed-floor coverage with joint/IMU/RGB support; no truth navigation")
    sim.add_argument("--known-route-diagnostic", action="store_true", help="Diagnostic only: follow public route using simulator pose; never autonomous acceptance")
    sim.add_argument("--planned-survey", action="store_true", help="Public-map survey route with estimated sensor pose; avoids known rocks; no frontier exploration")
    sim.add_argument("--survey-segment", type=int, nargs=2, metavar=("FIRST", "STOP"),
                     help="Local planned-survey test, with a known initial pose; stop index is exclusive")
    sim.add_argument("--hybrid-ball", action="store_true", help="Combine the flat-floor hypothesis and silhouette multi-view rescue (requires --slope-perception)")
    sim.add_argument("--outline-ball", action="store_true", help="Test silhouette refinement and 2-D tracks (requires --slope-perception)")
    sim.add_argument("--slope-perception", action="store_true", help="Experimental slope-aware visual and range-only outputs (requires --perception-demo and navigation support)")
    rubble = sim.add_mutually_exclusive_group()
    rubble.add_argument("--dense-rubble", action="store_true", help="Include repeated RGB stereo estimates in terrain_layer.json (requires --slope-perception)")
    rubble.add_argument("--multiview-rubble", action="store_true", help="Use joint keyframe depths and local ground support for the rubble layer (requires --slope-perception)")
    sim.add_argument("--rubble-precision-model", type=Path,
                     help="Add RGB consensus and inferred relief candidates (requires --multiview-rubble)")
    sim.add_argument("--rubble-single-view", type=Path,
                     help="Publish single RGB frame range-layer rubble candidates and a comparison baseline")
    sim.add_argument("--rubble-completion-model", type=Path,
                     help="Estimate a revisable rubble footprint for planning (requires --perception-demo)")
    sim.add_argument("--rubble-region-mapping", action="store_true",
                     help="Track and replace rubble surface contributions (requires --rubble-completion-model)")
    sim.add_argument("--perception-demo", action="store_true", help="Build and record perception layers in a moving MPC diagnostic")
    sim.add_argument("--baseline-localizer", action="store_true", help="Use the original ICP window in the perception diagnostic")
    sim.add_argument("--seed", type=int, default=0)
    for name in ("demo", "replay", "live", "native"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--output", required=True)
        p.add_argument("--steps", type=int, default=200)
        p.add_argument("--execute", action="store_true", help="Enable real SDK commands; live mode only")
        p.add_argument("--record", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        if name == "demo":
            p.add_argument("--mission-fixture", action="store_true", help="Project a nearby local marker and ball into the software fixture")
        if name == "replay":
            p.add_argument("--input", required=True)
        if name == "live":
            p.add_argument("--endpoint", default="tcp://127.0.0.1:8765")
        if name == "native":
            p.add_argument("--lidar-provider", required=True, help="module:factory implementing the actual scanner driver")
            p.add_argument("--sport-velocity-frame", required=True, choices=["body", "imu_world"])
    agent = sub.add_parser("ros2-agent")
    agent.add_argument("--config")
    agent.add_argument("--output", default="output/ros2_agent")
    agent.add_argument("--seconds", type=float, default=20.)
    agent.add_argument("--rate", type=float, default=20.)
    agent.add_argument("--ros2-agent-port", type=int, default=8766)
    args = parser.parse_args()
    if args.mode == "mujoco":
        import os
        if not args.viewer:
            os.environ.setdefault("MUJOCO_GL", "egl")
        from .mujoco_sim import run_simulation
        try:
            run_simulation(args)
        except (ValueError, OSError, RuntimeError) as error:
            parser.exit(1, f"go2-rescue mujoco: {error}\n")
    elif args.mode == "ros2-agent":
        from .ros2_agent import run_agent
        try:
            run_agent(args.config, args.output, args.seconds, args.rate, args.ros2_agent_port)
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            parser.exit(1, f"go2-rescue ros2-agent: {error}\n")
    elif args.mode == "init-config":
        if Path(args.path).exists():
            parser.error("Config already exists")
        Config().save(args.path)
    elif args.mode == "finish-marker":
        if not 0 <= args.id < 50:
            parser.error("Marker id must be in [0,49]")
        if Path(args.path).exists():
            parser.error("Marker image already exists")
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), args.id, 400)
        marker = cv2.copyMakeBorder(marker, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        if not cv2.imwrite(args.path, marker):
            parser.error("Could not write marker image")
    else:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        try:
            run(args)
        except (ValueError, OSError, ImportError, EOFError, RuntimeError) as error:
            parser.exit(1, f"go2-rescue: {error}\n")


if __name__ == "__main__":
    main()
