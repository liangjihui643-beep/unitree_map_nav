"""Independent post-run scene coverage/metric map acceptance; no planner import."""
import argparse,json
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.ndimage import binary_dilation,binary_erosion,label
from scipy.spatial import cKDTree

def quantile(values):
 a=np.asarray(values);return dict(zip(['p50_m','p90_m','max_m'],map(float,np.quantile(a,[.5,.9,1])))) if len(a) else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path)
 ap.add_argument('--structural-mask',type=Path);ap.add_argument('--report-dir',type=Path)
 ap.add_argument('--structure-layer',type=Path,help='Independent structure mask; reports baseline/structure/union with one reference')
 ap.add_argument('--audit-map-confidence',action='store_true')
 ap.add_argument('--terrain-layer',type=Path)
 a=ap.parse_args();r=a.run;report=a.report_dir or r
 if (a.structural_mask or a.terrain_layer) and not a.report_dir:
  ap.error('supplemental layer requires a separate report directory')
 report.mkdir(parents=True, exist_ok=True)
 summary=json.loads((r/'summary.json').read_text());truth=json.loads((r/'evaluation_truth.json').read_text());trace=[json.loads(v) for v in (r/'lidar_diagnostics.jsonl').read_text().splitlines()]
 assert len(truth)==len(trace) and np.allclose([v['t'] for v in truth],[v['t'] for v in trace])
 root=Path(__file__).resolve().parents[1];model=mujoco.MjModel.from_binary_path(str(root/'scene/mujoco/competition-course.mjb'));state=mujoco.MjData(model);mujoco.mj_forward(model,state)
 robot=model.body('base_link').id;robot_geoms=np.flatnonzero(model.body_rootid[model.geom_bodyid]==robot);model.geom_group[robot_geoms]=5
 resolution=.10;size=(120,100);floor=np.zeros(size,bool);height=np.full(size,np.nan);occupied=np.zeros(size,bool)
 geom_ids=np.full(size,-1,dtype=int)
 gid=np.array([-1],np.int32);group=np.array([1,1,1,1,1,0],np.uint8)
 for i,j in np.ndindex(size):
  z=mujoco.mj_ray(model,state,np.array([(i+.5)*resolution,(j+.5)*resolution,3.]),np.array([0.,0.,-1.]),group,True,-1,gid)
  if z<0:continue
  height[i,j]=3.-z;g=int(gid[0]);name=model.geom(g).name or '';body=model.body(int(model.geom_bodyid[g])).name or ''
  geom_ids[i,j]=g
  floor[i,j]=bool(('ground' in name.lower() or model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE or body.startswith('ramp')) and height[i,j]<1.)
  occupied[i,j]=not floor[i,j]
 # Reference is approximate circular-centre reachability of source floor,
 # not a command input and not a proof of every body orientation being safe.
 x,y=np.ogrid[-3:4,-3:4];disk=(x*x+y*y)*resolution**2<=.25**2
 centre=floor & ~binary_dilation(occupied,structure=disk)
 components,_=label(centre);start=np.floor(np.asarray(summary['diagnostics_only']['initial_world_position'][:2])/resolution).astype(int)
 start_id=int(components[tuple(start)]);reachable=centre&(components==start_id) if start_id else np.zeros(size,bool)
 if not start_id:raise ValueError('Reference start disconnected after inflation; cannot score coverage')
 # Include the visible floor around a valid reachable robot centre, not just
 # the centreline. No disconnected enclosures are silently treated as visited.
 accessible=floor & binary_dilation(reachable,structure=np.ones((5,5),bool))
 cells=np.argwhere(accessible);xy=(cells+.5)*resolution
 planned=summary.get('planned_public_map_survey',False)
 anchor=np.asarray([.8,.8] if planned else truth[0]['base_world_xy']);yaw=0. if planned else float(summary['initial_yaw']);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
 with np.load(r/'map.npz') as m:
  origin=m['origin'];res=float(m['resolution']);seen=m['seen'];odds=m['logodds'];occ=odds>.5
  extra_structure=np.zeros_like(occ)
  if a.structural_mask:
   with np.load(a.structural_mask) as supplemental:
    if supplemental['mask'].shape!=occ.shape or not np.allclose(supplemental['origin'],origin) or not np.isclose(supplemental['resolution'],res):
     raise ValueError('Structural mask geometry does not match the recorded map')
    extra_structure=supplemental['mask'].astype(bool)&~occ
    occ=occ|extra_structure
  ij=np.floor(((xy-anchor)@rot-origin)/res).astype(int);valid=(ij>=0).all(1)&(ij<np.asarray(seen.shape)).all(1)
  observed=np.zeros(len(cells),bool);observed[valid]=seen[tuple(ij[valid].T)]
  missing=accessible.copy();missing[tuple(cells[observed].T)]=False
  predicted=(origin+(np.argwhere(occ)+.5)*res)@rot.T+anchor
  pred_seen=(origin+(np.argwhere(seen)+.5)*res)@rot.T+anchor
 boundary=occupied & binary_dilation(accessible,iterations=2)
 ref=(np.argwhere(boundary)+.5)*resolution
 in_arena=((predicted>=[0,0])&(predicted<[12,10])).all(1);predicted=predicted[in_arena]
 distance=cKDTree(ref).query(predicted)[0] if len(ref) and len(predicted) else []
 complete_structure=(np.argwhere(occupied)+.5)*resolution
 complete_distance=cKDTree(complete_structure).query(predicted)[0] if len(complete_structure) and len(predicted) else []
 reverse=cKDTree(predicted).query(ref)[0] if len(ref) and len(predicted) else []
 within_boundary=np.asarray(reverse)<=.20+1e-8
 fused_boundary=within_boundary.copy()
 missing_groups={}
 semantic_path=a.terrain_layer or r/'terrain_layer.json'
 semantic=json.loads(semantic_path.read_text()) if semantic_path.exists() else {}
 if semantic.get("schema")=="go2-terrain-layer/1" and len(fused_boundary):
  terrain=np.asarray([cell["xy_m"] for cell in semantic["cells"]],dtype=float).reshape(-1,2)
  reference_is_rubble=np.array([model.body(int(model.geom_bodyid[g])).name.startswith("pile")
                                for g in geom_ids[boundary]])
  if len(terrain) and reference_is_rubble.any():
   semantic_distance=cKDTree(terrain@rot.T+anchor).query(ref[reference_is_rubble])[0]
   fused_boundary[reference_is_rubble]|=semantic_distance<=.20+1e-8
 if len(reverse):
  for geom,within in zip(geom_ids[boundary],within_boundary):
   body=model.body(int(model.geom_bodyid[geom])).name
   name=model.geom(int(geom)).name or body
   group="rubble" if body.startswith("pile") else (
    "tunnel" if name.startswith("tunnel") else
    "outer_boundary" if "boundary" in name else
    "course_walls" if any(k in name for k in ("wall","partition")) else "other_structure")
   counts=missing_groups.setdefault(group,{"reference_cells":0,"within_20cm":0})
   counts["reference_cells"]+=1
   counts["within_20cm"]+=int(within)
 true_xy=np.asarray([v['base_world_xy'] for v in truth]);estimated=np.asarray([[v['pose']['x'],v['pose']['y']] for v in trace])@rot.T+anchor
 error=np.linalg.norm(estimated-true_xy,axis=1)
 events=summary['evaluation'];positions=np.asarray(summary['perception_demo']['confirmed_targets']).reshape(-1,2)
 target=truth[-1].get('football_world_xy');target_errors=np.linalg.norm(positions@rot.T+anchor-np.asarray(target),axis=1).tolist() if target is not None else []
 result={'schema':'go2-scene-coverage-acceptance/1','scene':summary['scene'],'original_start':summary['run_scope']=='original_start','autonomous':summary.get('course_policy')=='ordered','reference_scope':'source top-down first-hit ground/ramp, circular .25m centre clearance, start connected, expanded .2m; approximate floor mask; pipe roof excluded; no reference fed to navigation','reference_accessible_cells':int(accessible.sum()),'reference_resolution_m':resolution,'observed_accessible_cells':int(observed.sum()),'coverage_fraction':float(observed.mean()),'missing_area_m2':float(missing.sum()*resolution**2),'trajectory_error':quantile(error),'occupied_to_reference_boundary':quantile(distance),'reference_boundary_within_20cm':float(within_boundary.mean()) if len(within_boundary) else 0.,'target_marker_errors_m':target_errors,'target_extra_markers':max(0,len(positions)-1),'fell':summary['fell'],'obstacle_contact_episodes':events['obstacle_contact_episodes'],'water_entry_episodes':events['water_entry_episodes'],'out_of_bounds_episodes':events['out_of_bounds_episodes'],'stop_reason':summary['stop_reason'],'whole_scene_pass':False}
 result['reference_boundary_within_20cm']=float(within_boundary.mean()) if len(within_boundary) else 0.
 result['boundary_within_20cm_check']={'overall':int(within_boundary.sum()),'sum_by_structure':int(sum(v['within_20cm'] for v in missing_groups.values()))}
 result['gates']={'coverage_95pct':result['coverage_fraction']>=.95,'trajectory_p90_20cm':bool(result['trajectory_error'] and result['trajectory_error']['p90_m']<=.20),'boundary_p90_15cm':bool(result['occupied_to_reference_boundary'] and result['occupied_to_reference_boundary']['p90_m']<=.15+1e-8),'boundary_recall_90pct':result['reference_boundary_within_20cm']>=.9,'one_target_within_25cm':len(target_errors)==1 and max(target_errors)<=.25,'safe':not summary['fell'] and not any(events[k] for k in ['obstacle_contact_episodes','water_entry_episodes','out_of_bounds_episodes'])}
 result['gates']['autonomous_without_truth_navigation']=not summary.get('known_route_diagnostic',False)
 result['boundary_by_structure']=missing_groups
 result['fused_reference_boundary_within_20cm']=float(fused_boundary.mean()) if len(fused_boundary) else 0.
 result['occupied_to_any_source_structure']=quantile(complete_distance)
 if a.structure_layer:
  with np.load(a.structure_layer) as supplemental:
   extra_structure=supplemental['mask'].astype(bool)
   if extra_structure.shape!=occ.shape or not np.allclose(supplemental['origin'],origin) or not np.isclose(supplemental['resolution'],res):
    raise ValueError('Structure layer geometry does not match the recorded map')
  def structure_metrics(mask):
   samples=(origin+(np.argwhere(mask)+.5)*res)@rot.T+anchor
   samples=samples[((samples>=[0,0])&(samples<[12,10])).all(1)]
   distances=cKDTree(ref).query(samples)[0] if len(ref) and len(samples) else []
   structures=cKDTree(complete_structure).query(samples)[0] if len(complete_structure) and len(samples) else []
   reverse=cKDTree(samples).query(ref)[0] if len(samples) and len(ref) else []
   groups={}
   if len(reverse):
    for geom,within in zip(geom_ids[boundary],reverse<=.20+1e-8):
     body=model.body(int(model.geom_bodyid[geom])).name
     name=model.geom(int(geom)).name or body
     group="rubble" if body.startswith("pile") else (
      "tunnel" if name.startswith("tunnel") else
      "outer_boundary" if "boundary" in name else
      "course_walls" if any(k in name for k in ("wall","partition")) else "other_structure")
     counts=groups.setdefault(group,{"reference_cells":0,"within_20cm":0})
     counts["reference_cells"]+=1;counts["within_20cm"]+=int(within)
   return {'cells':int(len(samples)),'reference_boundary_within_20cm':float(np.mean(np.asarray(reverse)<=.20+1e-8)) if len(reverse) else 0.,
           'occupied_to_reference_boundary':quantile(distances),
           'occupied_to_any_source_structure':quantile(structures),
           'more_than_20cm_from_any_structure':int((np.asarray(structures)>.20+1e-8).sum()) if len(structures) else 0,
           'boundary_by_structure':groups,
           'boundary_within_20cm_check':{'overall':int((np.asarray(reverse)<=.20+1e-8).sum()) if len(reverse) else 0,
                                          'sum_by_structure':int(sum(v['within_20cm'] for v in groups.values()))}}
  result['structure_layer']={'path':str(a.structure_layer),
                             'structure_only':structure_metrics(extra_structure),
                             'union':structure_metrics(occ|extra_structure)}
  result['whole_scene_pass']=False
 if a.audit_map_confidence:
  from scipy.ndimage import convolve
  neighbours=convolve((odds>.5).astype(np.int8),np.ones((3,3),np.int8),mode='constant')
  curve={}
  for key,mask in [('base',odds>.5),('local_support_3',(odds>.5)&(neighbours>=3)),
                   ('odds_1_5',odds>1.5),('odds_2_5',odds>2.5),('odds_3_5',odds>3.5)]:
   samples=(origin+(np.argwhere(mask)+.5)*res)@rot.T+anchor
   samples=samples[((samples>=[0,0])&(samples<[12,10])).all(1)]
   if len(samples):
    curve[key]={'count':len(samples),'boundary_p90':float(np.quantile(cKDTree(ref).query(samples)[0],.90)),
                'recall_20cm':float((cKDTree(samples).query(ref)[0]<=.20+1e-8).mean())}
  result['map_confidence_audit']=curve
 result['navigation_scope']='declared public-map prior survey' if summary.get('planned_public_map_survey',False) else 'online observed-floor navigation'
 if summary.get('planned_public_map_survey',False):
  result['reference_scope']=result['reference_scope'].replace('no reference fed to navigation','navigation uses a separately declared prior from the same public geometry; measured map scored separately')
  route=summary.get('survey_route') or {}
  result['gates']['survey_route_complete']=bool(summary.get('mission_complete') and route.get('route_complete') and route.get('all_requested_viewpoints_reached'))
  result['gates']['original_start_full_survey']=result['original_start'] and summary.get('survey_segment') is None
 result['autonomous']=result['autonomous'] and not summary.get('known_route_diagnostic',False)
 result['whole_scene_pass']=all(result['gates'].values())
 if a.structural_mask:
  result['experimental_structural_mask']=str(a.structural_mask)
  result['whole_scene_pass']=False
 (report/'coverage_acceptance.json').write_text(json.dumps(result,indent=2))
 base=np.full((*size,3),70,np.uint8);base[occupied]=(25,25,25);base[floor]=(175,175,175);base[accessible]=(215,215,215);base[missing]=(55,75,220);base[tuple(cells[observed].T)]=(95,190,110)
 mapped=np.full_like(base,70)
 for points,color in [(pred_seen,(215,215,215)),(predicted,(20,20,20))]:
  cs=np.floor(points/resolution).astype(int);inside=(cs>=0).all(1)&(cs<np.array(size)).all(1);mapped[tuple(cs[inside].T)]=color
 canvas=np.full((640,1460,3),25,np.uint8)
 for im,x,labeltext in [(base,20,'Reference floor: green seen / red missing'),(mapped,750,'Estimated map and autonomous trajectory')]:
  im=cv2.resize(cv2.flip(im.transpose(1,0,2),0),(690,575),interpolation=cv2.INTER_NEAREST);canvas[45:620,x:x+690]=im;cv2.putText(canvas,labeltext,(x,27),0,.63,(240,240,240),1)
 def pix(points,offset):
  q=np.asarray(points).reshape(-1,2);return np.column_stack((offset+q[:,0]/12*690,45+(10-q[:,1])/10*575)).astype(np.int32)
 cv2.polylines(canvas,[pix(true_xy,20)],False,(255,160,25),2);cv2.polylines(canvas,[pix(estimated,750)],False,(255,160,25),2)
 for point in positions:cv2.drawMarker(canvas,tuple(pix(point@rot.T+anchor,750)[0]),(255,180,0),cv2.MARKER_STAR,16,2)
 cv2.imwrite(str(report/'coverage_acceptance.png'),canvas);print(json.dumps(result,indent=2))
 if semantic.get("schema")=="go2-terrain-layer/1":
  fused=mapped.copy()
  extra_xy=(origin+(np.argwhere(extra_structure)+.5)*res)@rot.T+anchor
  extra_indices=np.floor(extra_xy/resolution).astype(int)
  valid_extra=((extra_indices>=0)&(extra_indices<np.asarray(size))).all(1)
  fused[tuple(extra_indices[valid_extra].T)]=(128,106,100)
  for item in semantic["cells"]:
   xy=np.asarray(item["xy_m"])@rot.T+anchor
   index=np.floor(xy/resolution).astype(int)
   if (index>=0).all() and (index<np.asarray(size)).all():
    if item["height_observed_min_max_m"] is not None:
     fused[tuple(index)]=(45,174,80)
    elif "height_estimated_m" in item:
     fused[tuple(index)]=(185,95,182)
    else:
     fused[tuple(index)]=(102,183,223) if item.get("inferred_footprint") else (18,151,239)
  fused=cv2.resize(fused.transpose(1,0,2)[::-1],(690,575),interpolation=cv2.INTER_NEAREST)
  has_visual_height=any("height_estimated_m" in item for item in semantic["cells"])
  image=np.full((752 if has_visual_height else 726,730,3),25,np.uint8);image[45:620,20:710]=fused
  cv2.polylines(image,[pix(estimated,20)],False,(245,150,55),2)
  for point in positions:
   cv2.drawMarker(image,tuple(pix(point@rot.T+anchor,20)[0]),(255,196,40),cv2.MARKER_STAR,18,2)
  cv2.putText(image,'Sensor map | estimated trajectory', (20,28),0,.66,(235,235,235),1)
  for caption,x,y,color in [
   ('Occupied / LiDAR',20,646,(150,150,150)),('RGB rubble',260,646,(18,151,239)),
   ('Measured relief',465,646,(45,174,80)),('Robot path',20,674,(245,150,55)),
   ('Confirmed ball',260,674,(255,196,40)),('Dark gray: unknown',465,674,(190,190,190)),
   ('Pale orange: inferred footprint, height unknown',20,702,(102,183,223)),
   ('3D structure returns',465,702,(181,159,153))]:
   cv2.putText(image,caption,(x,y),0,.51,color,1,cv2.LINE_AA)
  if has_visual_height:
   cv2.putText(image,'Purple: experimental visual height',(20,731),0,.51,(185,95,182),1,cv2.LINE_AA)
  cv2.imwrite(str(report/'fused_map.png'),image)
if __name__=='__main__':main()
