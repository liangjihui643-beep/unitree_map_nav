"""Score the published map and the simultaneous sparse RGB stream, offline."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.recording import replay


def rows(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            yield json.loads(line)


def summary(values):
    if not len(values):
        return {'n': 0, 'p50': None, 'p90': None}
    return {'n': len(values), 'p50': float(np.median(values)), 'p90': float(np.quantile(values,.9))}


class Reference:
    def __init__(self):
        self.model = mujoco.MjModel.from_binary_path('scene/mujoco/competition-course.mjb')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.groups = {key: np.eye(6,dtype=np.uint8)[i] for i, key in enumerate(('unused','rubble','floor'))}
        self.geom_id = np.array([-1],np.int32)
        self.down = np.array([0.,0.,-1.])
        for g in range(self.model.ngeom):
            geom = self.model.geom(g)
            body = self.model.body(self.model.geom_bodyid[g]).name
            is_floor = ('ground' in (geom.name or '').lower() or body.startswith('ramp')
                        or self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
            self.model.geom_group[g] = 1 if body.startswith('pile') else 2 if is_floor else 5
        base = self.model.body('base_link').id
        self.model.geom_group[self.model.body_rootid[self.model.geom_bodyid] == base] = 4

    def ray(self, xy, kind):
        return mujoco.mj_ray(self.model,self.data,np.r_[xy,3.],self.down,
                             self.groups[kind],True,-1,self.geom_id)

    def cell(self, xy, resolution):
        # Mesh edges can cross a cell without crossing its centre.
        offsets = np.linspace(-.35,.35,3)*resolution
        relief = []
        for i in offsets:
            for j in offsets:
                point = xy + [i,j]
                top = self.ray(point,'rubble')
                ground = self.ray(point,'floor')
                if 0 <= top < ground:
                    relief.append(ground-top)
        return len(relief), float(np.quantile(relief,.8)) if relief else None

    def pile_samples(self, origin, resolution):
        cells = set()
        for g in np.flatnonzero(self.model.geom_group == 1):
            center, extent = self.model.geom_aabb[g].reshape(2,3)
            rot = self.data.geom_xmat[g].reshape(3,3)
            center = rot @ center + self.data.geom_xpos[g]
            extent = abs(rot) @ extent
            lower = np.floor((center[:2]-extent[:2]-origin)/resolution).astype(int)
            upper = np.floor((center[:2]+extent[:2]-origin)/resolution).astype(int)
            cells.update((x,y) for x in range(lower[0],upper[0]+1) for y in range(lower[1],upper[1]+1))
        top = {}
        for cell in cells:
            center = origin+(np.asarray(cell)+.5)*resolution
            offsets = np.linspace(-.35,.35,3)*resolution
            values = []
            for i in offsets:
                for j in offsets:
                    xy = center+[i,j]
                    hit = self.ray(xy,'rubble')
                    if 0 <= hit < self.ray(xy,'floor'):
                        values.append([*xy,3.-hit])
            if values:
                top[cell] = max(values,key=lambda item:item[2])
        return top


def visible_truth(reference, run, truth, poses, world_origin, resolution):
    cfg = Config.load(run/'config.json')
    mount = np.asarray(cfg.body_from_camera)
    k = np.asarray(cfg.camera_matrix)
    width,height = cfg.camera_size
    samples = reference.pile_samples(world_origin,resolution)
    indices = list(samples)
    world = np.asarray(list(samples.values()))
    views = [[] for _ in world]
    groups = np.array([0,1,1,0,0,1],np.uint8)
    for frame in replay(run/'recording'):
        t = round(frame.timestamp,3)
        pose = poses.get(t)
        if (frame.image is None or t not in truth or pose is None
                or pose['pose']['quality'] < cfg.min_pose_quality):
            continue
        body_rotation = Rotation.from_quat(np.roll(frame.quaternion,-1)).as_matrix()
        center = np.asarray(truth[t])+body_rotation@mount[:3,3]
        rays = world-center
        local = rays @ (body_rotation@mount[:3,:3])
        positive = np.flatnonzero((local[:,2]>.3)&(local[:,2]<4.))
        uvw = local[positive] @ k.T
        uv = uvw[:,:2]/uvw[:,2,None]
        use = positive[np.all((uv>4)&(uv<[width-4,height-4]),axis=1)]
        if not len(use):
            continue
        distance = np.linalg.norm(rays[use],axis=1)
        unit = rays[use]/distance[:,None]
        hits = np.full(len(use),-1.,float)
        geoms = np.full(len(use),-1,np.int32)
        mujoco.mj_multiRay(reference.model,reference.data,center,unit.ravel(),groups,True,-1,
                           geoms,hits,None,len(use),5.)
        for i in np.flatnonzero((hits>=0)&(abs(hits-distance)<.045)):
            views[use[i]].append(unit[i])
    visible = {cell for cell, r in zip(indices,views) if len(r)}
    parallax = {cell for cell,r in zip(indices,views) if len(r)>1
                and np.min(np.asarray(r)@np.asarray(r).T) < np.cos(np.deg2rad(3.))}
    return visible, parallax


def draw_comparison(output, visible, parallax, cells):
    labels = {tuple(x['ij']):x for x in cells}
    points = np.asarray(list(visible | set(labels)))
    lo, hi = points.min(axis=0)-2, points.max(axis=0)+3
    shape = tuple((hi-lo)[::-1])
    reference = np.full((*shape,3), 30, np.uint8)
    published = reference.copy()
    for cell in visible:
        x,y = cell-lo
        reference[y,x]=(153,153,153) if cell in parallax else (80,80,80)
        published[y,x]=(80,80,80)
    for cell,item in labels.items():
        x,y=cell-lo
        color=(205,85,185) if item['estimated_relief_m'] is not None else (20,168,250)
        published[y,x]=color
    panels=[]
    for plot,name in ((reference,'Reference: visible rubble'),(published,'Published RGB terrain')):
        pad=cv2.resize(cv2.flip(plot,0),None,fx=12,fy=12,interpolation=cv2.INTER_NEAREST)
        pad=cv2.copyMakeBorder(pad,36,34,12,12,cv2.BORDER_CONSTANT,value=(30,30,30))
        cv2.putText(pad,name,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(235,235,235),1,cv2.LINE_AA)
        panels.append(pad)
    cv2.putText(panels[0],'Light: view change >= 3 deg',(12,panels[0].shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1)
    cv2.putText(panels[1],'Amber: contour  Purple: relief',(12,panels[1].shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
    cv2.imwrite(str(output),np.hstack(panels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    read = lambda name: json.loads((a.run / name).read_text(encoding='utf-8'))
    map_layer = read('terrain_layer.json')
    traj = list(rows(a.run/'trajectory.jsonl'))
    truth = {round(x['t'],3):x['base_world_xyz'] for x in read('evaluation_truth.json')}
    first = next(x for x in traj if round(x['timestamp'],3) in truth)
    origin = np.asarray(map_layer['origin_xy_m'])
    res = map_layer['resolution_m']
    world_offset = np.asarray(truth[round(first['timestamp'],3)][:2]) - [first['pose']['x'],first['pose']['y']]
    reference = Reference()
    sparse = set()
    timings = []
    for row in rows(a.run/'visual_pseudocloud.jsonl'):
        sparse.update(tuple(ij) for ij in np.floor((np.asarray(row['points'],float).reshape(-1,5)[:,:2]-origin)/res).astype(int))
        timings.append(row['processing_ms'])
    stats = {'sparse_rgb_same_run': {'cells':0,'truth_overlap':0},
             'published_dense_added': {'cells':0,'truth_overlap':0},
             'published_all': {'cells':0,'truth_overlap':0}}
    errors = []
    sample_table = []
    for item in map_layer['cells']:
        xy = np.asarray(item['xy_m']) + world_offset
        ij = tuple(item['ij'])
        ray_hits, height = reference.cell(xy,res)
        for key, included in (('sparse_rgb_same_run',ij in sparse), ('published_all',True),
                              ('published_dense_added',ij not in sparse and item.get('rgb_dense_view_bins',0)>=2)):
            if included:
                stats[key]['cells'] += 1
                stats[key]['truth_overlap'] += int(ray_hits >= 1)
        visual_height = item.get('height_estimated_m')
        if visual_height is not None and height is not None:
            errors.append(abs(visual_height-height))
        sample_table.append({'ij':list(ij), 'world_xy':xy.tolist(), 'sparse':ij in sparse, 'ray_hits':ray_hits,
                             'estimated_relief_m':visual_height, 'reference_relief_p80_m':height,
                             'measured_relief_min_max_m':item['height_observed_min_max_m']})
    result = {'source':str(a.run),'footprint':stats,'visual_height_error_m':summary(errors),
              'visual_height_cells':sum(x['estimated_relief_m'] is not None for x in sample_table),
              'rgb_reconstruction_processing_ms':summary(timings),
              'height_reference':'p80 of the top minus floor at 3x3 rays in each 10 cm cell',
              'truth_footprint_rule':'any ray through rubble above floor, 3x3 per cell'}
    a.output.mkdir(parents=True,exist_ok=True)
    visible, parallax = visible_truth(reference, a.run, truth,
                                      {round(x['timestamp'],3):x for x in traj},origin+world_offset,res)
    published={tuple(item['ij']) for item in sample_table}
    result['visible_reference']={'visible_cells':len(visible),'covered_cells':len(visible & published),
                                  'parallax_3deg_cells':len(parallax),'covered_parallax_cells':len(parallax & published)}
    if visible or published:
        draw_comparison(a.output/'live_terrain_compare.png', visible, parallax, sample_table)
    (a.output/'live_terrain_score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (a.output/'live_terrain_cells.json').write_text(json.dumps(sample_table,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
