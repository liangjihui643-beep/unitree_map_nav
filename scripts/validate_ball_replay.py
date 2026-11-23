"""Validate RGB target detection and map tracking on a held-out physical survey."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
from go2_rescue.ball_classifier import BallProposalClassifier
from go2_rescue.config import Config
from go2_rescue.perception import Perception
from go2_rescue.recording import replay
from go2_rescue.tracking import Tracker
from go2_rescue.types import Pose


def main():
    ap=argparse.ArgumentParser();ap.add_argument("run",type=Path);ap.add_argument("--model",type=Path,required=True)
    args=ap.parse_args();run=args.run
    config=Config.load(run/"config.json")
    model=BallProposalClassifier(args.model);vision=Perception(config);tracker=Tracker(config, static_targets=True)
    pose_rows=[json.loads(line) for line in (run/"lidar_diagnostics.jsonl").read_text().splitlines()]
    poses={round(r["t"],6):Pose(**r["pose"]) for r in pose_rows}
    totals={"visible":0,"hit":0,"unmatched_image_candidates":0,"empty_frames":0,"empty_false_candidates":0}
    results=[]; panels=[]
    for f in replay(run/"recording"):
        if f.image is None:continue
        pose=poses[round(f.timestamp,6)]
        raw=vision.detect_balls(f.image,small_targets=True)
        detections=model.filter(f.image,raw)
        detections=[vision.locate_ball(d,f,pose) for d in detections]
        events=tracker.update(detections,pose,f.timestamp) if pose.quality>=config.min_pose_quality else []
        ids=cv2.imread(str(run/"evaluation_masks"/f"{f.timestamp:.6f}.png"),0)
        ball=ids&2!=0
        visible=ball.sum()>=30
        height,width=ball.shape; yy,xx=np.ogrid[:height,:width]
        matches=[]
        for d in detections:
            mask=(xx-d.center[0])**2+(yy-d.center[1])**2<=(d.radius+2)**2
            overlap=float((mask&ball).sum()/max(1,(mask|ball).sum()))
            matches.append(overlap>=.45)
        hit=bool(visible and any(matches))
        totals["visible"]+=int(visible);totals["hit"]+=int(hit)
        totals["unmatched_image_candidates"]+=sum(not m for m in matches)
        totals["empty_frames"]+=int(not visible)
        totals["empty_false_candidates"]+=sum(not m for m in matches) if not visible else 0
        results.append({"t":f.timestamp,"visible":bool(visible),"hit":hit,
                        "detections":[{"center_px":list(d.center),"radius_px":d.radius,
                                       "confidence":d.confidence,"map_xy":d.position.tolist() if d.position is not None else None} for d in detections],
                        "events":[asdict(e) for e in events]})
        if 25.8<f.timestamp<26.4 or 27.5<f.timestamp<28.2 or 29.8<f.timestamp<30.3:
            image=f.image.copy()
            for d in detections:
                cv2.circle(image,tuple(np.rint(d.center).astype(int)),int(d.radius)+3,(255,200,30),2)
            cv2.putText(image,f"t={f.timestamp:.1f}s target hit={hit}",(5,20),0,.45,(255,255,255),1)
            panels.append(image)
    totals["recall"]=totals["hit"]/max(1,totals["visible"])
    totals["reported_map_xy"]=[p.tolist() for p in tracker.reported_positions]
    report={"totals":totals,"frames":results,"scope":"held-out RGB and saved estimated poses; geometry+learned proposal filtering"}
    (run/"ball_learned_replay.json").write_text(json.dumps(report,indent=2))
    if panels:
        while len(panels)%2:panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(run/"far-ball-learned.png"),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
    print(json.dumps(totals,indent=2))


if __name__=="__main__":main()
