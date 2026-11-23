"""Constructed MuJoCo 72x9 return A/B, with a low wall under the beam."""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from go2_rescue.config import Config
from go2_rescue.geometry import bresenham
from go2_rescue.lidar_obs import RangeOnlyEvidence, encode_lidar_obs
from go2_rescue.mapping import GridMap
from go2_rescue.types import Pose

# The old ground-label failure is tested at x=2. A high first-return
# ray stops on the back wall; all lower rays are masked by range/height or miss
# the low wall when descending from the plateau. Test that physical setup
# separately from the map's logical "unknown must stay unknown" guarantee.
def run(z, wall_half_height, fov):
    xml = f"""<mujoco><worldbody>
       <geom type='box' pos='2 0 {wall_half_height}' size='.12 .30 {wall_half_height}'/>
       <geom type='box' pos='3.2 0 .45' size='.10 .40 .45'/>
    </worldbody></mujoco>"""
    m=mujoco.MjModel.from_xml_string(xml)
    d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    grid=GridMap(Config())
    lidar=RangeOnlyEvidence(grid,0.,fov)
    emit=np.array([0.,0.,z+.1])
    result=[]
    for theta in lidar.elevations:
        vec=np.array([np.cos(theta), 0., np.sin(theta)])
        geom=np.zeros(1,dtype=np.int32)
        r=mujoco.mj_ray(m,d,emit,vec,None,1,-1,geom)
        if .05<=r<=12 and .1+vec[2]*r>=-.22:
            result.append((r,vec))
    if not result:
        raise RuntimeError('No sensor return in fixture')
    dist, vec=min(result,key=lambda e:e[0])
    rng=np.zeros(72,dtype=bool); rng[36]=True
    floor=np.zeros_like(grid.floor)
    lidar.update(grid,encode_lidar_obs(rng,[dist]),np.array([1,0,0,0]),Pose(),.05,12.,
                 base_z=z,ground=floor)
    hit=emit+dist*vec
    grid.ground_height = z-grid.c.base_height
    grid.floor[grid.cell(hit[:2])] = hit[2]-.10
    grid.update(np.array([hit-[0.,0.,z]]), Pose())
    wall_col=grid.cell((2. if fov>1 else 1.5,0.))
    return dict(return_range_m=float(dist), return_xyz_m=hit.tolist(),
                legacy_wall_cell_cleared=bool(grid.logodds[wall_col] < -.1),
                common_projected_ray=bool(lidar.ray_prefix[wall_col]),
                near_ground_clear=bool(lidar.ground_band_clear[wall_col]),
                cells_with_near_ground_evidence=int(lidar.ground_band_clear.sum()))

rows={'wall_from_plateau':run(.62,.12,96.),
      'clear_level_near_floor':run(-.035,.01,1.)}
assert rows['wall_from_plateau']['legacy_wall_cell_cleared']
assert not rows['wall_from_plateau']['near_ground_clear']
assert rows['clear_level_near_floor']['near_ground_clear']
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
