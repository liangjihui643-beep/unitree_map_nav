import json
from pathlib import Path
import sys
import numpy as np
p=Path(sys.argv[1])
truth=json.loads((p/'evaluation_truth.json').read_text())
frame={round(r['t'],3):r for r in truth}
anchor=np.asarray(truth[0]['base_world_xy'])
xyz=np.asarray([r['base_world_xyz'] for r in truth])
rows=[json.loads(l) for l in (p/'target_tracks.jsonl').read_text().splitlines()]
events=[]
for row in rows:
    for e in row['new_confirmations']:
        ref=np.asarray(frame[round(e['t'],3)]['football_world_xy'])
        events.append({'stamp': e['t'], 'source': e['source'],'position_xy':e['map_xy'],
                       'error_m':float(np.linalg.norm(anchor+e['map_xy']-ref))})
r={'sim_translation_xy_m':np.ptp(xyz[:,:2],axis=0).tolist(),'frames':len(rows),'events':events}
(p/'online_event_score.json').write_text(json.dumps(r,indent=2))
print(r)
