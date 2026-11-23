"""Check target ranging replay and online fixture against explicit local gates."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("replays", type=Path, nargs="+")
p.add_argument("--target-counts", nargs="+", type=int, required=True)
p.add_argument("--online", type=Path, nargs="*", default=[])
p.add_argument("--max-error-m", type=float, default=.10)
p.add_argument("--max-online-first-s", type=float, default=9.)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if len(a.replays) != len(a.target_counts):
    p.error("Supply a target count for every replay")
rows = []
for file, expected in zip(a.replays, a.target_counts):
    result = json.loads(file.read_text(encoding="utf-8"))["hybrid"]
    errors = [x["error_m"] for x in result["events"]]
    accepted = (len(result["events"]) == result["unique_matched"] == expected
                and result["unmatched"] == result["duplicate"] == 0
                and all(x <= a.max_error_m for x in errors))
    rows.append({"fixture": file.stem, "mode": "replay", "passed": accepted,
                 "errors_m": errors, "reported": len(result["events"])})
for run in a.online:
    meta = json.loads((run / "scene_provenance.json").read_text(encoding="utf-8"))
    seq = json.loads((run / "online_event_score.json").read_text(encoding="utf-8"))["events"]
    errors = [x["error_m"] for x in seq]
    usable = len(seq) == 1 and bool(meta.get("unlimited_reports"))
    accepted = bool(usable and errors[0] <= a.max_error_m and seq[0]["stamp"] <= a.max_online_first_s)
    rows.append({"fixture": run.name, "mode": "online MPC", "passed": accepted,
                 "errors_m": errors, "reported": len(seq), "quota_disabled": meta.get("unlimited_reports"),
                 "stamps_s": [x["stamp"] for x in seq]})
result = {"passed": all(row["passed"] for row in rows),
          "scope": "local target geometry, replay and MPC fixtures, not competition acceptance",
          "max_error_m": a.max_error_m,
          "max_online_first_s": a.max_online_first_s, "results": rows}
a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
