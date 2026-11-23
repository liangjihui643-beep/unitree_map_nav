"""Replay one fixed slope window. Truth camera poses are offline diagnosis only."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.geometry import gravity_rotation
from go2_rescue.low_obstacle import LowObstacleDetector, rubble_region_observable
from go2_rescue.recording import replay
from go2_rescue.rubble_visual_height import RubbleVisualHeight
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.types import Pose

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--start', type=float, default=90.)
p.add_argument('--end', type=float, default=125.)
p.add_argument("--interpolate-stone", action="store_true")
p.add_argument("--wider-baseline", action="store_true")
p.add_argument("--independent-ground", action="store_true")
p.add_argument("--contact-ground", action="store_true")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
rows = lambda path: {round(x['timestamp' if 'timestamp' in x else 't'], 3): x
                     for line in path.open(encoding='utf-8') if (x := json.loads(line))}
traj, old = (rows(a.run / f'{n}.jsonl') for n in ('trajectory', 'rubble_observations'))
truth = {round(x['t'], 3): np.asarray(x['base_world_xyz'])
         for x in read(a.run / 'evaluation_truth.json')}
t0 = next(stamp for stamp in traj if stamp in truth)
anchor = truth[t0][:2] - [traj[t0]['pose']['x'], traj[t0]['pose']['y']]
cfg = Config.load(a.run / 'config.json')
vision = LowObstacleDetector('models/seg_rubble_scene.onnx', cfg.camera_matrix,
                             cfg.body_from_camera, cfg.base_height)
solvers = {name: RubbleVisualHeight(np.asarray(cfg.camera_matrix), interpolate_stone=a.interpolate_stone, wider_baseline=a.wider_baseline, independent_ground=a.independent_ground, use_contact_ground=a.contact_ground) for name in ('truth_xyz', 'truth_z_only', 'estimated')}
surface = StanceSurface()
t_camera = np.asarray(cfg.body_from_camera)
stats_geometry = {k: {} for k in solvers}
result = {k: [] for k in ('legacy', *solvers)}
n_frame = 0
for frame in replay(a.run / 'recording'):
    t = round(frame.timestamp, 3)
    if t > a.end:
        break
    if t < a.start-1:
        continue
    surface.update(frame)
    if t not in traj or t not in truth:
        continue
    row = traj[t]
    pose = Pose(**row['pose'])
    if pose.quality < cfg.min_pose_quality:
        continue
    level = gravity_rotation(frame.quaternion)
    support = row["diagnostics"].get("support", {})
    estimated_z = cfg.base_height + row["diagnostics"]["ground_height"]
    if frame.support_points is not None and "base_to_support_m" in support and surface.plane is not None:
        solvers["estimated"].observe_contacts(frame.support_points @ level.T, pose, estimated_z, t)
    if t < a.start or frame.image is None:
        continue
    n_frame += 1
    if t in old:
        result['legacy'].extend([x + [None, None, None] for r in old[t]['regions'] for x in r['map_xy']])
    prob = vision.segment(frame.image)
    label = prob.argmax(axis=0)
    rubble = ((label == 1) & (prob[1] >= .7)).astype(np.uint8)
    rubble[cv2.dilate(((label == 2) & (prob[2] >= .45)).astype(np.uint8),
                     np.ones((3, 3), np.uint8)) != 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rubble)
    mask = np.zeros_like(rubble)
    for i, (x, y, width, height, area) in enumerate(stats[1:n], 1):
        if area >= 25 and rubble_region_observable(x, y, width, height, mask.shape[1]):
            mask[labels == i] = 1
    level = gravity_rotation(frame.quaternion)
    yaw = Rotation.from_quat(np.roll(frame.quaternion, -1)).as_euler('xyz')[2]
    wx, wy, wz = truth[t]
    ground = row['diagnostics']['ground_height']
    support = row['diagnostics'].get('support', {})
    for key, solver in solvers.items():
        if key != 'truth_xyz' and 'base_to_support_m' not in support:
            continue
        p0 = pose if key != 'truth_xyz' else Pose(wx-anchor[0], wy-anchor[1], yaw)
        z = cfg.base_height + ground if key == 'estimated' else wz
        points = solver.observe(frame.image, mask, level@t_camera[:3,:3], level@t_camera[:3,3],
                                p0, ground, surface.plane, t, ground_mask=(label==0)&(prob[0]>=.65), base_z=z)
        for n, v in solver.diagnostics.items():
            stats_geometry[key][n] = stats_geometry[key].get(n, 0) + v
        result[key].extend([[float(v) if np.isfinite(v) else None for v in pt] for pt in points])

model = mujoco.MjModel.from_binary_path('scene/mujoco/competition-course.mjb')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
mask = np.zeros((100, 120), dtype=np.uint8)
for g in range(model.ngeom):
    name = model.body(int(model.geom_bodyid[g])).name
    if not name.startswith('pile') or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    m = model.geom_dataid[g]
    start, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
    xyz = model.mesh_vert[start:start+n] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
    cv2.fillConvexPoly(mask, cv2.convexHull(np.floor(xyz[:,:2]/.10).astype(np.int32)), 1)
summary = {'window_s':[a.start,a.end], 'camera_frames':n_frame, 'geometry':stats_geometry, 'truth_piles_projected':int(mask.sum())}
panels = []
for key, points in result.items():
    cells = {tuple(np.floor((np.array(x[:2])+anchor[:2])/.10).astype(int)) for x in points}
    measured = {tuple(np.floor((np.array(x[:2])+anchor[:2])/.10).astype(int))
                for x in points if x[4] is not None}
    inside = {c for c in cells if 0 <= c[0] < 120 and 0 <= c[1] < 100 and mask[c[1],c[0]]}
    summary[key] = {'observations':len(points), 'cells':len(cells), 'inside_pile_footprint':len(inside),
                    'cells_with_height':len(measured), 'height_inside_footprint':len(inside & measured)}
    canvas = np.full((100,120,3), 40, np.uint8)
    canvas[mask!=0] = (95,95,95)
    for x,y in cells:
        if 0 <= x < 120 and 0 <= y < 100:
            canvas[y,x] = (0,160,250) if (x,y) in inside else (40,40,230)
            if (x,y) in measured:
                canvas[y,x] = (180,80,205)
    panel=cv2.resize(cv2.flip(canvas,0),None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel,key,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    panels.append(panel)
(a.out/'slope_metrics.json').write_text(json.dumps(summary,indent=2))
(a.out/'slope_points.json').write_text(json.dumps(result))
cv2.imwrite(str(a.out/'slope_comparison.png'),np.hstack(panels))
print(json.dumps(summary,indent=2))
