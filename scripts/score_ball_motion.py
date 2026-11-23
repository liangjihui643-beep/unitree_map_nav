"""Reproducible comparison of nearest-reference target candidates, not events."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("observations",type=Path)
p.add_argument("provenance",type=Path)
a=p.parse_args()
r=json.loads(a.observations.read_text())
scene=json.loads(a.provenance.read_text())
placements=scene.get("diagnostics_only_placements",{})
primary=placements.get("football_world_xyz")
target_xyz=[*scene.get("test_fixture_extra_balls_world_xyz",[]), *([primary] if primary else [])]
if not target_xyz:
    raise ValueError("No labelled football positions in provenance")
targets=np.asarray(target_xyz)[:,:2]-r["map_anchor_world_xy"]
groups={};report={}
def percentile(values):
    return float(np.quantile(values,.9)) if len(values) else None
for mode in ("old","new","fused"):
    if mode not in r:
        continue
    errors=[]; group={}
    for time, xy in r[mode]:
        distances=np.linalg.norm(targets-xy,axis=1)
        who=int(distances.argmin())
        err=float(distances[who])
        errors.append(err)
        if err<=.5:
            group[time,who]=min(err,group.get((time,who),np.inf))
    groups[mode]=group
    report[mode]={"solutions":len(errors),"p90_m":percentile(errors),
                  "matched_within_0_5m":int((np.array(errors)<=.5).sum())}
common=set.intersection(*(set(v) for v in groups.values()))
report["paired"]={"n":len(common), **{k:percentile([g[c] for c in common]) for k,g in groups.items()}}
print(json.dumps(report,indent=2))
