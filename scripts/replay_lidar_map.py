"""Replay a recorded run into GridMap for laser-mapping A/B comparisons.

Only laser observations and estimated poses are used; no simulator truth is fed
to the map. The recorded configuration is loaded but navigation_support is
disabled so the comparison isolates GridMap ray criteria.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.mapping import GridMap
from go2_rescue.recording import replay
from go2_rescue.types import Pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ray-height-clear", action="store_true")
    ap.add_argument("--evidence", action="store_true", default=True)
    ap.add_argument("--no-evidence", dest="evidence", action="store_false")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--skip-high-clear", action="store_true",
                    help="selective rule: do not clear cells the ray only crossed above body clearance")
    ap.add_argument("--skip-above-near-clear", action="store_true",
                    help="selective rule: do not clear cells where the ray missed the near-ground band")
    ap.add_argument("--no-wall-truncate", action="store_true",
                    help="disable stopping ray free updates at a stable wall (A/B switch)")
    ap.add_argument("--autonomy", action="store_true",
                    help="step Autonomy (localization + navigation_support) instead of GridMap only")
    ap.add_argument("--start-seconds", type=float, default=0.)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    source, out = args.source, args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    config = Config.load(source / "config.json")
    config.map_evidence = bool(args.evidence)
    config.map_evidence_positions = bool(args.positions)
    config.map_ray_height_clear = bool(args.ray_height_clear)
    config.map_skip_high_clear = bool(args.skip_high_clear)
    config.map_skip_above_near_clear = bool(args.skip_above_near_clear)
    if args.no_wall_truncate:
        config.map_wall_truncate = False

    rows = {}
    with (source / "trajectory.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            rows[round(record["timestamp"], 5)] = record
    used = skipped = 0
    if args.autonomy:
        # Faithful module probe: original configuration, localization and
        # navigation_support steps run; no simulator truth is injected.
        from go2_rescue.controller import Autonomy
        config.navigation_support = True
        autonomy = Autonomy(config)
        grid = autonomy.grid
        for frame in replay(source / "recording"):
            if frame.timestamp < args.start_seconds:
                continue
            if args.limit and used >= args.limit:
                break
            if rows.get(round(frame.timestamp, 5)) is None:
                skipped += 1
                continue
            autonomy.step(frame)
            used += 1
    else:
        config.navigation_support = False
        grid = GridMap(config)
        for frame in replay(source / "recording"):
            if frame.timestamp < args.start_seconds:
                continue
            if args.limit and used >= args.limit:
                break
            row = rows.get(round(frame.timestamp, 5))
            if row is None or row["pose"]["quality"] < config.min_pose_quality:
                skipped += 1
                continue
            points = frame.points @ gravity_rotation(frame.quaternion).T
            ranges = np.linalg.norm(points, axis=1)
            points = points[(ranges > .25) & (ranges < config.max_range)]
            if not len(points):
                skipped += 1
                continue
            grid.update(points, Pose(**row["pose"]), frame.quaternion, frame.timestamp)
            used += 1

    grid.save(out / "map.npz")
    summary = grid.evidence_summary()
    if summary:
        (out / "map_evidence.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if grid.evidence_positions:
        positions = {f"{cell[0]},{cell[1]}": views for cell, views in grid.evidence_positions.items()}
        (out / "map_evidence_positions.json").write_text(json.dumps(positions, indent=0), encoding="utf-8")
    if getattr(grid, "wall_diagnostics", None):
        with (out / "map_wall_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
            for item in grid.wall_diagnostics:
                stream.write(json.dumps(item) + "\n")
    for name in ("summary.json", "evaluation_truth.json", "lidar_diagnostics.jsonl",
                 "config.json", "navigation_prior.npz"):
        asset = source / name
        if asset.exists():
            shutil.copyfile(asset, out / name)
    meta = {"source": str(source), "frames_used": used, "frames_skipped": skipped,
            "mode": "autonomy" if args.autonomy else "gridmap",
            "navigation_support": bool(config.navigation_support),
            "ray_height_clear": bool(args.ray_height_clear),
            "evidence": bool(args.evidence), "seen": int(grid.seen.sum()),
            "occupied": int((grid.logodds > .5).sum()),
            "free": int(((grid.logodds < -.5) & grid.seen).sum()),
            "floor_source": np.bincount(grid.floor_source.ravel(), minlength=6).astype(int).tolist(),
            "wall_truncated_rays": len(getattr(grid, "wall_diagnostics", [])),
            "map_options": {"wall_truncate": bool(config.map_wall_truncate),
                            "wall_logodds": float(config.map_wall_logodds),
                            "evidence": bool(config.map_evidence),
                            "ray_height_clear": bool(config.map_ray_height_clear),
                            "skip_high_clear": bool(config.map_skip_high_clear),
                            "skip_above_near_clear": bool(config.map_skip_above_near_clear)},
            "evidence_summary": summary}
    (out / "replay_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("frames_used", "frames_skipped", "seen", "occupied", "free")}))


if __name__ == "__main__":
    main()
