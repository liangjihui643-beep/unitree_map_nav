"""Score estimated relief at exact sample XY using separate reference rays."""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

p=Path(sys.argv[1])
if p.is_dir():
    points={f.stem:json.loads(f.read_text()) for f in p.glob("*-points.json")}
    a=[0.,0.]
    p = p/"height_score.json"
else:
    r=Path(sys.argv[2])
    points=json.loads(p.read_text())
    truth={round(x['t'], 3): x['base_world_xyz'] for x in json.loads((r/'evaluation_truth.json').read_text())}
    with (r/'trajectory.jsonl').open(encoding='utf-8') as f:
        row=next(x for line in f if round((x := json.loads(line))['timestamp'],3) in truth)
    a=np.asarray(truth[round(row['timestamp'],3)][:2]) - [row['pose']['x'], row['pose']['y']]
model=mujoco.MjModel.from_binary_path('scene/mujoco/competition-course.mjb')
data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
for i in range(model.ngeom):
    geom=model.geom(i)
    body=model.body(int(model.geom_bodyid[i])).name
    ground=('ground' in (geom.name or '').lower()
            or model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE or body.startswith('ramp'))
    model.geom_group[i]=1 if body.startswith('pile') else 2 if ground else 5
rock_mask=np.array([0,1,0,0,0,0],np.uint8)
ground_mask=np.array([0,0,1,0,0,0],np.uint8)
gid=np.array([-1],np.int32)
down=np.array([0.,0.,-1.])
rows={}
details={}
for mode, values in points.items():
    if mode=='legacy':
        continue
    n=0
    cells={}
    sample_rows=[]
    for x,y,rock_z,floor_z,relief in values:
        if relief is None:
            continue
        n+=1
        xy=np.array([x,y])+a[:2]
        origin=np.r_[xy,3.]
        top=mujoco.mj_ray(model,data,origin,down,rock_mask,True,-1,gid)
        ground=mujoco.mj_ray(model,data,origin,down,ground_mask,True,-1,gid)
        if 0<=top<ground:
            ix=tuple(np.floor(xy/.10).astype(int))
            cells.setdefault(ix,[]).append(abs(relief-(ground-top)))
            sample_rows.append({'world_xy':xy.tolist(), 'cell':list(map(int, ix)), 'relief_m':relief,
                                'truth_relief_m':ground-top, 'visual_top_m':rock_z,
                                'visual_ground_m':floor_z, 'truth_top_m':3.-top,
                                'truth_ground_m':3.-ground})
    e=np.array([np.median(v) for v in cells.values()])
    rows[mode]={'samples_with_height':n,'valid_truth_cells':len(cells),
                'p50_m':float(np.median(e)) if len(e) else None,
                'p90_m':float(np.quantile(e,.9)) if len(e) else None}
    details[mode]=sample_rows
p.with_name('height_score.json').write_text(json.dumps(rows,indent=2))
p.with_name('height_details.json').write_text(json.dumps(details,indent=2))
print(json.dumps(rows,indent=2))
