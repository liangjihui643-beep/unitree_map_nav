"""Consolidate local validation without treating independent artifacts as one run."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("out", type=Path)
    a=p.parse_args()
    root=a.evidence

    def load(name):
        path=root/name
        return json.loads(path.read_text(encoding="utf-8-sig"))

    target=load("hybrid-target-gate.json")
    sparse=[load(n) for n in ("sparse-range-ramp.json","sparse-range-old.json")]
    footprint=load("rubble-fixed-wide-200/slope_metrics.json")
    heights=load("replay-iter4/height_score.json")["estimated"]
    # Prior delivery plan caps trajectory p90 at .20 m and terrain p90 at .08 m.
    # Retain the harder independent evidence requirements introduced for slopes.
    gates={
        "local_ball":bool(target["passed"]),
        "independent_trajectory_xy_p90_at_most_20cm":all(
            r["p90_m"]["sparse_range"]<=.20 for r in sparse),
        "mapping_compressed_input_full_run":False,
        "rubble_height_p90_at_most_8cm":int(heights["valid_truth_cells"])>=5
            and heights["p90_m"] is not None and heights["p90_m"]<=.08,
        "holdout_rubble_recovers_observed_footprint":(
            footprint["legacy"]["inside_pile_footprint"]>0
            and footprint["estimated"]["inside_pile_footprint"]>=.85*footprint["legacy"]["inside_pile_footprint"]),
        "new_route_and_map_verified_in_one_run":False,
    }
    result={
        "accepted":all(gates.values()),
        "gates":gates,
        "scope":"Current experimental perception; no end-to-end acceptance inferred from separate recordings",
        "notes":{
            "rubble_height":"Existing estimated-motion baseline, which beats the contact-ground ablation on p90",
            "rubble_footprint":"Empirical holdout non-regression gate; not an official competition metric",
            "mapping":"Range evidence still takes legacy estimated pose online; it does not form a complete independent map",
        },
    }
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__=="__main__":
    main()
