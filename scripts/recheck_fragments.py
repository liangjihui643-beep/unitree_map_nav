"""Apply the final region visibility rule to saved masks and challenge RGB."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from go2_rescue.low_obstacle import LowObstacleDetector, rubble_region_observable
from go2_rescue.config import Config


def filter_mask(pred):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8))
    kept = [i for i in range(1,n) if stats[i,4] >= 25 and rubble_region_observable(*stats[i,:4],pred.shape[1])]
    return np.isin(labels,kept)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    run=args.run
    if (run/"challenge.json").exists():
        c=Config()
        root=Path(__file__).resolve().parents[1]
        d=LowObstacleDetector(root/"models/seg_low_obstacle_perception.onnx",c.camera_matrix,c.body_from_camera,c.base_height)
        report=json.loads((run/"challenge.json").read_text())
        for row in report["views"]:
            image=cv2.imread(str(run/(row["name"]+"-raw.png")))
            truth=cv2.imread(str(run/(row["name"]+"-truth.png")),0)
            p=d.segment(image)
            pred=(p[1]>=.70)&(p.argmax(0)==1)
            structure=((p.argmax(0)==2)&(p[2]>=.45)).astype(np.uint8)
            pred[cv2.dilate(structure,np.ones((3,3),np.uint8))!=0]=False
            pred=filter_mask(pred)
            rock=truth&1!=0
            wall=truth&2!=0
            row.update(tp=int((pred&rock).sum()),fp=int((pred&~rock).sum()),fn=int((~pred&rock).sum()),
                       predicted_rock_pixels=int(pred.sum()),wall_fp=int((pred&wall).sum()))
            row["precision"]=row["tp"]/max(1,row["tp"]+row["fp"])
            row["recall"]=row["tp"]/max(1,row["tp"]+row["fn"])
            image[pred]=(.5*image[pred]+.5*np.array([20,180,255])).astype(np.uint8)
            cv2.imwrite(str(run/(row["name"]+"-final.png")),image)
        report["source"]="same archived RGB and independent true object masks, final border-fragment rule"
        (run/"challenge_final.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    else:
        totals=dict(tp=0,fp=0,fn=0,wall_fp=0,wall_pixels=0)
        for f in sorted((run/"evaluation_masks").glob("*.png")):
            m=cv2.imread(str(f),0)
            pred=filter_mask(m&8!=0)
            rock=m&1!=0
            wall=m&4!=0
            totals["tp"]+=int((pred&rock).sum())
            totals["fp"]+=int((pred&~rock).sum())
            totals["fn"]+=int((~pred&rock).sum())
            totals["wall_fp"]+=int((pred&wall).sum())
            totals["wall_pixels"]+=int(wall.sum())
        totals["precision"]=totals["tp"]/max(1,totals["tp"]+totals["fp"])
        totals["recall"]=totals["tp"]/max(1,totals["tp"]+totals["fn"])
        totals["evaluation_scope"]="saved masks with thin side fragments rejected"
        (run/"rubble_fragment_recheck.json").write_text(json.dumps(totals,indent=2))
        print(json.dumps(totals,indent=2))


if __name__=="__main__":
    main()
