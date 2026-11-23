import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from go2_rescue.recording import replay


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run",type=Path); args=ap.parse_args(); run=args.run
    config=json.loads((run/"config.json").read_text())
    truth=json.loads((run/"evaluation_truth.json").read_text())
    detection={round(row["t"],6):row for row in [json.loads(s) for s in (run/"target_tracks.jsonl").read_text().splitlines()]}
    c=np.asarray(config["body_from_camera"]); k=np.asarray(config["camera_matrix"])
    rows=[]
    for frame in replay(run/"recording"):
        if frame.image is None: continue
        actual=min(truth,key=lambda x:abs(x["t"]-frame.timestamp))
        if abs(actual["t"]-frame.timestamp)>1e-6: raise ValueError("No exact truth")
        point=actual["football_world_xy"]
        if point is None: continue
        point=np.r_[point,.11]
        rw=Rotation.from_quat(np.roll(frame.quaternion,-1)).as_matrix()
        cam=rw@c[:3,:3]
        origin=np.asarray(actual["base_world_xyz"])+rw@c[:3,3]
        p=(point-origin)@cam
        if p[2]<=0: continue
        center=(k@p)[:2]/p[2]; radius=k[0,0]*.11/p[2]
        mask=cv2.imread(str(run/"evaluation_masks"/f"{frame.timestamp:.6f}.png"),0)
        ball=mask&2!=0
        fraction=float(ball.sum()/(np.pi*radius**2))
        full=bool(fraction>=.75 and center[0]-radius>=0 and center[0]+radius<320
                  and center[1]-radius>=0 and center[1]+radius<240)
        det=detection[round(frame.timestamp,6)]["detections"]
        found=any(d["confidence"]>=.86 and np.linalg.norm(np.asarray(d["center_px"])-center)<=max(5,.4*radius)
                  and .5<=d["radius_px"]/radius<=1.7 for d in det)
        rows.append(dict(t=frame.timestamp,visible_fraction=fraction,full=full,found=found,visible_pixels=int(ball.sum())))
    full=[r for r in rows if r["full"]]; report={"full_frames":len(full),"full_hits":sum(r["found"] for r in full),
        "all_visible_frames":sum(r["visible_pixels"]>=30 for r in rows),"frames":rows}
    (run/"ball_visibility_breakdown.json").write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!="frames"})
    print([r for r in rows if r["full"] and not r["found"]])


if __name__=="__main__": main()
