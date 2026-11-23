"""Replay hybrid target ranging on frozen classifier candidates, then score events."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from go2_rescue.config import Config
from go2_rescue.hybrid_ball import HybridBallRange, usable_ball
from go2_rescue.perception import Perception
from go2_rescue.recording import replay
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.tracking import Tracker
from go2_rescue.types import BallObservation, Pose


def indexed(path):
    with path.open() as stream:
        return {round(row.get("t",row.get("timestamp")),3):row for row in map(json.loads,stream)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--start",type=float,default=0.)
    p.add_argument("--end",type=float,default=float("inf"))
    p.add_argument("--level-fixture",action="store_true")
    p.add_argument("--allow-unconfirmed-flat", action="store_true",
                   help="Diagnostic only: allow plane estimates to report before multi-view")
    p.add_argument("--compensate-rotation", action="store_true")
    p.add_argument("--known-size", action="store_true")
    p.add_argument("--require-outline", action="store_true")
    p.add_argument("--quota",type=int,default=0)
    a=p.parse_args()
    cv2.setNumThreads(1)
    r=a.run
    cfg=Config.load(r/"config.json")
    cfg.target_quota=a.quota
    vision=Perception(cfg)
    hybrid=HybridBallRange(vision,allow_unconfirmed_flat=a.allow_unconfirmed_flat,
                           require_outline_for_flat=a.require_outline,use_known_size=a.known_size,
                           compensate_rotation=a.compensate_rotation)
    support=StanceSurface()
    tracks={mode:Tracker(cfg,static_targets=True) for mode in ("legacy_pixels","hybrid")}
    traj=indexed(r/"trajectory.jsonl")
    raw=indexed(r/"target_tracks.jsonl")
    loc=indexed(r/"lidar_diagnostics.jsonl")
    stats={k:{"solutions":0,"events":[],"sources":{}} for k in tracks}
    frames=0
    provisional=0
    for f in replay(r/"recording"):
        t=round(f.timestamp,3)
        if t>a.end:
            break
        if t<a.start-1.:
            continue
        plane=support.update(f)
        if t<a.start or f.image is None or t not in raw or t not in loc:
            continue
        pose=Pose(**loc[t]["pose"])
        if pose.quality<cfg.min_pose_quality:
            continue
        frames+=1
        proposals=[BallObservation(tuple(d["center_px"]),d["radius_px"],d["confidence"],0.)
                   for d in raw[t]["detections"]]
        base_z=np.nan
        d=traj.get(t,{}).get("diagnostics",{})
        if a.level_fixture and plane is not None:
            base_z=-plane[2]
        elif "base_to_support_m" in d.get("support",{}) and "ground_height" in d:
            base_z=d["ground_height"]+cfg.base_height
        legacy=[vision.locate_ball(replace(x),replace(f,points=np.empty((0,3)),point_times=None),pose)
                for x in proposals]
        hybrid.update(proposals,vision.undistort(f.image),f,pose,base_z,plane,.85)
        provisional+=len(hybrid.provisional)
        for k,measurements in (("legacy_pixels",legacy),("hybrid",proposals)):
            eligible=[d for d in measurements if usable_ball(d,.85)]
            stats[k]["solutions"]+=len(eligible)
            for d in eligible:
                counts=stats[k]["sources"]
                counts[d.range_source]=counts.get(d.range_source,0)+1
            for e in tracks[k].update(eligible,pose,f.timestamp):
                obs=next(t for t in tracks[k].tracks if t.id==e.payload["id"]).observation
                stats[k]["events"].append({"t":t,"xy":e.payload["position"],"source":obs.range_source})
    truth=json.loads((r/"evaluation_truth.json").read_text())
    anchor=np.asarray(truth[0]["base_world_xy"])
    ref={round(x["t"],3):x["football_world_xy"] for x in truth}
    scene=json.loads((r/"scene_provenance.json").read_text())
    extras=scene.get("test_fixture_extra_balls_world_xyz",[])
    for scores in stats.values():
        identified=set()
        duplicate=unmatched=0
        for e in scores["events"]:
            primary=ref.get(e["t"])
            targets=[*([primary] if primary is not None else []),*[b[:2] for b in extras]]
            if not targets:
                unmatched+=1
                continue
            ds=np.linalg.norm(np.asarray(targets)-anchor-e["xy"],axis=1)
            i=int(ds.argmin())
            e["nearest_target"]=i
            e["error_m"]=float(ds[i])
            if ds[i]>.5:
                unmatched+=1
            elif i in identified:
                duplicate+=1
            else:
                identified.add(i)
        scores.update(unique_matched=len(identified),unmatched=unmatched,duplicate=duplicate)
    result={"frames":frames, "provisional_flat_candidates":provisional, **stats}
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

