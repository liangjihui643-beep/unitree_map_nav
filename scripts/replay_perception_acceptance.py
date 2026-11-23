"""Run the production perception entry on a complete archived physical run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from go2_rescue.config import Config
from go2_rescue.controller import Autonomy
from go2_rescue.geometry import gravity_rotation
from go2_rescue.perception_demo import PerceptionDemo, write_png
from go2_rescue.perception_showcase import Showcase, text, BLUE, GOLD
from go2_rescue.recording import replay


def main():
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    config=Config.load(args.source/"config.json")
    autonomy=Autonomy(config);autonomy.localization.use_local_submap=True
    demo=PerceptionDemo(autonomy,root/"models/seg_low_obstacle_perception.onnx",out)
    marker=Showcase.__new__(Showcase)
    video=cv2.VideoWriter(str(out/"showcase.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),5.,(1280,720))
    external=cv2.VideoCapture(str(args.source/"showcase.mp4"))
    totals=dict(rgb_frames=0,ball_visible_frames=0,ball_hit_frames=0,rock_tp=0,rock_fp=0,rock_fn=0,
                rock_on_wall_pixels=0,wall_visible_pixels=0)
    confirmations=[];firstproof=None;next_video=0;last=0.
    (out/"evaluation_masks").mkdir()
    try:
        for f in replay(args.source/"recording"):
            last=f.timestamp
            azimuth=np.rint((np.arctan2(f.points[:,1],f.points[:,0])+np.pi)*72/(2*np.pi)).astype(int)%72
            metadata={"azimuth_index":azimuth,"up_body":gravity_rotation(f.quaternion).T@np.array([0.,0.,1.])}
            demo.observe(f,metadata)
            if f.image is None:continue
            m=cv2.imread(str(args.source/"evaluation_masks"/f"{f.timestamp:.6f}.png"),0)
            ball=m&2!=0;rock=m&1!=0;wall=m&4!=0
            p=demo.front_rubble
            detected=any(Showcase.overlap(d.center,d.radius,ball)>=.45
                         for d in demo.latest_detections if d.confidence>=.86)
            visible=ball.sum()>=30
            totals["rgb_frames"]+=1;totals["ball_visible_frames"]+=int(visible);totals["ball_hit_frames"]+=int(visible and detected)
            totals["rock_tp"]+=int((p&rock).sum());totals["rock_fp"]+=int((p&~rock).sum());totals["rock_fn"]+=int((~p&rock).sum())
            totals["rock_on_wall_pixels"]+=int((p&wall).sum());totals["wall_visible_pixels"]+=int(wall.sum())
            write_png(out/"evaluation_masks"/f"{f.timestamp:.6f}.png",(m&7)+8*p.astype(np.uint8))
            for item in demo.latest_confirmations:
                iou=Showcase.overlap(item["center_px"],item["radius_px"],ball)
                confirmations.append(dict(item,true_ball_mask_iou=iou,visible_ball_pixels=int(ball.sum())))
                if firstproof is None:
                    demo.snapshot(f.timestamp)
                    firstproof=marker.make_proof(f.image,demo,item,iou)
                    write_png(out/"ball-evidence.png",firstproof)
            demo.snapshot(f.timestamp)
            board=np.full((720,1280,3),(30,27,24),np.uint8)
            external.set(cv2.CAP_PROP_POS_MSEC,f.timestamp*1000)
            ok,original=external.read()
            if ok:board[107:327,24:420]=original[107:327,24:420]
            text(board,"GO2 / SENSOR REPLAY OF PHYSICAL RUN",24,34,.83)
            text(board,f"t={f.timestamp:.1f}s / recorded RGB, IMU, joints and legal LiDAR",24,65,.54)
            text(board,"1  Recorded physical motion",24,94)
            text(board,"2  Production perception on RGB",24,375)
            marker.picture(board,demo.last_front,24,390,396,265)
            text(board,"3  Estimated map and fused objects",466,95,.72)
            marker.draw_map(board,demo,(467,109,782,490))
            text(board,"Orange: visual rubble",469,630,.55,GOLD)
            text(board,"Green: supported low relief",755,630,.55,(40,210,75))
            text(board,f"LiDAR {len(f.points)}/72 / known support is encoded separately",469,662,.53)
            text(board,f"Confirmed balls: {len(demo.tracker.reported_positions)}",469,693,.57,BLUE)
            write_png(out/"showcase_live.png",board)
            wanted=round(f.timestamp*5)
            for _ in range(max(1,wanted-next_video)):video.write(board);next_video+=1
        if firstproof is not None:
            for _ in range(18):video.write(firstproof)
    finally:
        video.release();external.release()
        result=demo.finish(last)
    totals["ball_confirmations"]=confirmations
    totals["ball_visible_recall"]=totals["ball_hit_frames"]/max(1,totals["ball_visible_frames"])
    totals["rock_precision"]=totals["rock_tp"]/max(1,totals["rock_tp"]+totals["rock_fp"])
    totals["rock_recall"]=totals["rock_tp"]/max(1,totals["rock_tp"]+totals["rock_fn"])
    result["showcase"]=totals
    summary=json.loads((args.source/"summary.json").read_text())
    summary.update(perception_demo=result,evidence_kind="production perception replay; physical results retained from source",
                   replay_source=str(args.source))
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    (out/"showcase_acceptance.json").write_text(json.dumps(totals,indent=2))
    for name in ("config.json","evaluation_truth.json","scene_provenance.json"):
        shutil.copyfile(args.source/name,out/name)
    shutil.copytree(args.source/"recording",out/"recording")
    (out/"replay_manifest.json").write_text(json.dumps({
        "source":str(args.source),"physical_summary_sha256":hashlib.sha256((args.source/"summary.json").read_bytes()).hexdigest(),
        "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"src/go2_rescue").glob("*.py")},
        "model_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"models/seg_low_obstacle_perception.onnx",root/"models/ball_proposal.onnx")},
    },indent=2))
    print(json.dumps(totals,indent=2))


if __name__=="__main__":main()
