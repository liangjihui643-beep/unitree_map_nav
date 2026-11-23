import argparse
import json
import cv2
import numpy as np
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);args=p.parse_args()
    for script in ("evaluate_perception_demo.py","evaluate_terrain.py","render_height_evidence.py","scene_coverage.py"):
        if script == "render_height_evidence.py" and not json.loads((args.run/"terrain_heights.json").read_text()):
            image=np.full((240,640,3),30,np.uint8)
            cv2.putText(image,"No low-relief laser samples in this run",(15,120),0,.65,(235,235,235),1)
            cv2.imwrite(str(args.run/"terrain-evidence.png"),image)
            continue
        subprocess.run([sys.executable,str(Path("scripts")/script),str(args.run)],check=True)


if __name__=="__main__":main()
