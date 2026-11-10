#!/usr/bin/env python3
"""Start/stop only the roslaunch process group owned by this workspace."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(__file__).resolve().parent.parent
record = root / ".runtime/session.json"


def start_time(pid):
    try:
        return Path("/proc/%d/stat" % pid).read_text().split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("mode", nargs="?", default="mapping", choices=["mapping", "navigation", "explore"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--scene", choices=["room", "competition"], default="room")
    args = parser.parse_args()
    active = json.loads(record.read_text()) if record.exists() else None
    alive = bool(active and start_time(active["pid"]) == active["start_time"])
    if args.action == "status":
        print(json.dumps({"running": alive, "session": active}, indent=2))
    elif args.action == "stop":
        if alive:
            os.killpg(active["pid"], signal.SIGINT)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and start_time(active["pid"]) == active["start_time"]:
                time.sleep(0.2)
            if start_time(active["pid"]) == active["start_time"]:
                os.killpg(active["pid"], signal.SIGTERM)
        record.unlink(missing_ok=True)
        print("Workspace demo stopped")
    else:
        if alive:
            raise SystemExit("A workspace demo is already running; stop it first")
        launcher = "run_competition.sh" if args.scene == "competition" else "run_demo.sh"
        command = [str(root / launcher), args.mode,
                   "gui:=" + str(not args.headless).lower(),
                   "rviz:=" + str(not args.no_rviz).lower()]
        log = root / "logs" / (args.scene + "_" + args.mode + ".log")
        with log.open("w") as output:
            process = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        active = {"pid": process.pid, "start_time": start_time(process.pid),
                  "mode": args.mode, "scene": args.scene, "log": str(log)}
        record.write_text(json.dumps(active, indent=2) + "\n")
        time.sleep(2)
        if process.poll() is not None:
            raise SystemExit("Launch failed; see " + str(log))
        print(json.dumps(active, indent=2))


if __name__ == "__main__":
    main()
