"""Render measured low relief from saved RGB, kinematics and laser samples."""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from go2_rescue.recording import replay
from go2_rescue.mapping import spatial_frame_support
from go2_rescue.geometry import rot2


def label(image,text,x,y,scale=.6,color=(235,240,245)):
    cv2.putText(image,text,(x,y),0,scale,color,1,cv2.LINE_AA)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run",type=Path); args=ap.parse_args(); run=args.run
    c=json.loads((run/"config.json").read_text())
    height=json.loads((run/"terrain_heights.json").read_text())
    if not height: raise ValueError("No measured relief observations")
    first=next(r for r in height if any(.035<=h<=.30 for h in r["height_m"]))
    frame=next(f for f in replay(run/"recording") if abs(f.timestamp-first["t"])<1e-6)
    t=np.asarray(c["body_from_camera"]); k=np.asarray(c["camera_matrix"])
    body=np.asarray(first["points_body"])
    cam=(body-t[:3,3])@t[:3,:3]
    uv=cam@k.T
    uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
    front=cv2.resize(frame.image,(640,480))
    for (u,v),h in zip(uv,first["height_m"]):
        if .035<=h<=.30:
            cv2.drawMarker(front,(2*u,2*v),(40,230,90),cv2.MARKER_CROSS,23,2)
            label(front,f"{h:.3f} m",2*u+12,2*v-12,.65,(40,230,90))
    with np.load(run/"map.npz") as g:
        seen=g["seen"]; odds=g["logodds"]; origin=g["origin"]; res=float(g["resolution"])
    visual=Counter(); low_frames=[]
    for line in (run/"rubble_observations.jsonl").read_text().splitlines():
        row=json.loads(line)
        vis=set(); low=set()
        for r in row["regions"]:
            for field,cells in [("map_xy",vis),("lidar_map_xy",low)]:
                points=np.asarray(r.get(field,[]))
                if len(points):
                    cells.update(map(tuple,np.floor((points-origin)/res).astype(int)))
        visual.update(vis); low_frames.append(low)
    low=spatial_frame_support(low_frames)
    raster=np.full((*seen.shape,3),70,np.uint8); raster[seen]=(210,210,210); raster[odds>.5]=(25,25,25)
    for cell,n in visual.items():
        if n>=3 and all(0<=i<len(seen) for i in cell): raster[cell]=(0,175,255)
    for cell,n in low.items():
        if n>=2 and all(0<=i<len(seen) for i in cell): raster[cell]=(40,210,75)
    occupied=np.argwhere(seen); lo=np.maximum(0,occupied.min(0)-4); hi=np.minimum(seen.shape,np.max(occupied,0)+5)
    crop=cv2.flip(raster[lo[0]:hi[0],lo[1]:hi[1]],0)
    s=min(500/crop.shape[1],440/crop.shape[0]); crop=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
    board=np.full((650,1200,3),(30,27,24),np.uint8)
    label(board,"LOW RELIEF  /  CAMERA + LASER + JOINT / IMU",24,37,.85)
    board[80:560,20:660]=front
    board[80:80+crop.shape[0],685:685+crop.shape[1]]=crop
    label(board,f"Actual RGB + laser cross, t={frame.timestamp:.2f}s",24,70,.65)
    label(board,"Green map cells: repeated nearby laser support",685,70,.47,(40,210,75))
    label(board,"Height = laser hit relative to stance-foot support level",24,600,.72)
    label(board,"Foot level from measured joint / IMU kinematics; local horizontal support assumption",24,630,.59)
    cv2.imwrite(str(run/"terrain-evidence.png"),board)
    cv2.imwrite(str(run/"terrain-overlay.png"),raster[::-1])
    report={"measured_cells":len(low),"supported_cells":sum(n>=2 for n in low.values()),
            "support_radius_m":res*2**.5,"rule":"distinct observation frames near real measured cells only"}
    (run/"terrain_render.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__": main()
