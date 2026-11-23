"""Check complete route connectivity and exclusion of known rock footprints."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,label
from go2_rescue.config import Config
from go2_rescue.known_map_navigation import KnownMapNavigator
from go2_rescue.planning import astar

root=Path(__file__).resolve().parents[1]
c=Config(navigation_prior='configs/public-navigation-prior.npz',course_route='configs/perception-coverage-tour.json',
         footprint_half_length=.35,footprint_half_width=.23,safety_margin=.05)
nav=KnownMapNavigator(c);grid=nav.grid;free=grid.traversable();start=grid.cell([0.,0.])
assert free[start], 'Declared start must be free in the explicit prior'
components,_=label(free)
cells=np.argwhere(free & (components==components[start]));xy=grid.origin+(cells+.5)*grid.c.resolution
routes=[];unreachable=[]
cost=1.+.15/np.maximum(.05,distance_transform_edt(free)*grid.c.resolution)
for index,w in enumerate(nav.points):
 target=np.asarray(w['xy']);order=np.argsort(np.linalg.norm(xy-target,axis=1));path=[]
 for i in order[:150]:
  if np.linalg.norm(xy[i]-target)>1.:break
  path=astar(free,start,tuple(cells[i]),cost)
  if path:break
 if not path:
  unreachable.append({'index':index,'requested_xy':target.tolist(),'nearest_reachable_xy':xy[order[0]].tolist(),
                      'distance_m':float(np.linalg.norm(xy[order[0]]-target))})
  continue
 assert not grid.rock[tuple(np.asarray(path).T)].any()
 routes.append({'index':index,'stage':w['stage'],'requested_xy':target.tolist(),
                'planned_xy':grid.world(path[-1]).tolist(), 'path_xy':[grid.world(p).tolist() for p in path]})
 start=path[-1]
(root/'configs/public-survey-paths.json').write_text(json.dumps({'routes':routes,'unreachable':unreachable}))
print(f'Connected: {len(routes)}/{len(nav.points)} survey viewpoints, no rock crossing. Unreachable: {[p["index"] for p in unreachable]}. Full-body motion needs simulation.')
