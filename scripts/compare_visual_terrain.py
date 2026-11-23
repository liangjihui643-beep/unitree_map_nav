"""Compare causal visual terrain estimates using one recorded camera stream."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from go2_rescue.config import Config
from go2_rescue.controller import Autonomy
from go2_rescue.perception_demo import PerceptionDemo
from go2_rescue.recording import replay
from go2_rescue.types import Pose
from score_live_terrain import Reference, draw_comparison, rows, summary, visible_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--modes', nargs='+', choices=('pair', 'joint', 'pair-truth', 'joint-truth'),
                        default=['pair', 'joint', 'pair-truth', 'joint-truth'])
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float, default=float('inf'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    config = Config.load(args.run/'config.json')
    truth = {round(p['t'],3): p['base_world_xyz'] for p in
             json.loads((args.run/'evaluation_truth.json').read_text(encoding='utf-8'))}
    records = {round(p['timestamp'],3):p for p in rows(args.run/'trajectory.jsonl')}
    first = next(t for t in records if t in truth)
    frame_shift = np.asarray(truth[first][:2]) - [records[first]['pose']['x'],records[first]['pose']['y']]
    solvers = {}
    for mode in args.modes:
        folder = args.output/mode
        folder.mkdir()
        autonomy = Autonomy(config)
        demo = PerceptionDemo(autonomy,'models/seg_rubble_scene.onnx',folder)
        demo.slope_perception, demo.hybrid_ball = True, True
        demo.visual_height.dense = mode.startswith('pair')
        demo.visual_height.multiview = mode.startswith('joint')
        solvers[mode] = demo

    for frame in replay(args.run/'recording'):
        t = round(frame.timestamp, 3)
        if t > args.end:
            break
        if t < args.start or t not in records:
            continue
        row = records[t]
        pose = Pose(**row['pose'])
        for mode,demo in solvers.items():
            demo.stance_surface.update(frame)
            if mode.endswith('truth'):
                if t not in truth:
                    continue
                yaw = Rotation.from_quat(np.roll(frame.quaternion,-1)).as_euler('xyz')[2]
                # Only this explicitly marked diagnostic supplies scene truth.
                pose_input = Pose(*(np.asarray(truth[t][:2])-frame_shift), yaw, pose.quality, pose.sigma)
                base_z = truth[t][2]
            else:
                pose_input = pose
                base_z = config.base_height + row['diagnostics']['ground_height']
            support = row['diagnostics'].get('support', {})
            demo.grid.support_plane = demo.stance_surface.plane if 'base_to_support_m' in support else None
            demo.grid.ground_height = base_z-config.base_height
            if demo.visual_height.multiview:
                demo.observe_terrain_contacts(frame, pose_input)
            if frame.image is None:
                continue
            demo.autonomy.navigation_support.observe(frame)
            demo.loc.pose = pose_input
            demo.observe_image(frame,pose_input)

    reference = Reference()
    grid = next(iter(solvers.values())).grid
    origin, resolution = grid.origin, config.resolution
    full_window = args.start == 0 and args.end == float('inf')
    if full_window:
        visible, parallax = visible_truth(reference,args.run,truth,records,origin+frame_shift,resolution)
    else:
        visible, parallax = set(), set()
    result = {}
    errors_by_mode = {}
    for mode,demo in solvers.items():
        demo.snapshot(t)
        demo.video.release()
        errors = {}
        table = []
        for item in demo.terrain_layer['cells']:
            ij = tuple(item['ij'])
            ray_hits, height = reference.cell(np.asarray(item['xy_m'])+frame_shift,resolution)
            relief = item.get('height_estimated_m')
            if height is not None and relief is not None:
                errors[ij] = abs(relief-height)
            table.append({'ij':ij, 'estimated_relief_m':relief, 'reference_relief_p80_m':height,
                          'reference_ray_hits':ray_hits})
        errors_by_mode[mode] = errors
        observations = list(rows(demo.output/'visual_pseudocloud.jsonl'))
        cells = {x['ij'] for x in table}
        result[mode] = {
            'map_cells':len(cells),
            'on_rubble':sum(x['reference_ray_hits'] > 0 for x in table),
            'visible_cells':len(visible) if full_window else None,
            'visible_covered':len(cells & visible) if full_window else None,
            'parallax_covered':len(cells & parallax) if full_window else None,
            'has_height':sum(x['estimated_relief_m'] is not None for x in table),
            'height_error_m':summary(list(errors.values())),
            'processing_ms':summary([x['processing_ms'] for x in observations]),
            'dense_points':sum(len(x['dense_points']) for x in observations),
        }
        (demo.output/'scored_cells.json').write_text(json.dumps(table,indent=2),encoding='utf-8')
        if visible or cells:
            draw_comparison(demo.output/'comparison.png',visible,parallax,table)
    for mode,errors in errors_by_mode.items():
        if 'pair' in errors_by_mode:
            common = errors.keys() & errors_by_mode['pair'].keys()
            result[mode]['height_error_same_cells_m'] = summary([errors[c] for c in common])
            result[mode]['paired_baseline_error_m'] = summary([errors_by_mode['pair'][c] for c in common])
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
