import numpy as np
from go2_rescue.config import Config
from go2_rescue.geometry import rot2
from go2_rescue.mapping import GridMap
from go2_rescue.navigation_support import NavigationSupport, connected_ramp_returns
from go2_rescue.types import Frame,Pose
class Seg:
 def __init__(self,ground):self.ground=ground
 def segment(self,image):
  p=np.zeros((3,240,320),np.float32);p[0 if self.ground else 2]=1.;return p
def observation(support=True):
 return Frame(1.,np.empty((0,3)),image=np.zeros((240,320,3),np.uint8),support_points=np.array([[.23,.14,-.33],[.23,-.14,-.33],[-.23,.14,-.33],[-.23,-.14,-.33]]) if support else None)
def grid():
 c=Config(base_height=.33,navigation_support=True);g=GridMap(c);return c,g

def test_image_never_clears_unobserved_space():
 c,g=grid();n=NavigationSupport(c,Seg(True));n.update(observation(),Pose(),g)
 assert not np.isfinite(g.floor[g.cell([1.,0.])]);assert n.status['visual_cells']==0

def test_missing_support_cannot_project_rgb_floor():
 c,g=grid();g.seen[:]=True;g.logodds[:]=-4;n=NavigationSupport(c,Seg(True));n.update(observation(False),Pose(),g)
 assert not np.isfinite(g.floor).any()

def test_visual_floor_preserves_measured_height_and_occupied_cells():
 c,g=grid();g.seen[:]=True;g.logodds[:]=-4
 measured=g.cell([.8,.05]);g.floor[measured]=.12;g.floor_source[measured]=3
 obstacle=g.cell([.8,-.15]);g.logodds[obstacle]=4
 n=NavigationSupport(c,Seg(True));n.update(observation(),Pose(),g)
 assert g.floor[measured]==np.float32(.12);assert g.floor_source[measured]==3
 assert not np.isfinite(g.floor[obstacle]);assert n.status['visual_cells']>0

def test_structure_image_does_not_become_forward_floor():
 c,g=grid();g.seen[:]=True;g.logodds[:]=-4;n=NavigationSupport(c,Seg(False));n.update(observation(),Pose(),g)
 assert not np.isfinite(g.floor[g.cell([1.,0.])]);assert n.status['visual_cells']==0


def test_predicted_floor_never_drives_vertical_scan_alignment():
 c,g=grid();points=np.array([[1.+i*.1,.2,-.20] for i in range(6)])
 from go2_rescue.types import Pose
 for point in points:
  cell=g.cell(point);g.floor[cell]=.07;g.floor_source[cell]=4
 g.support_plane=np.array([0.,0.,-.33])
 for _ in range(30):g.update(points,Pose())
 assert g.ground_height==0.

def test_navigation_odometry_does_not_reuse_controller_scan_feedback():
 from types import SimpleNamespace
 from go2_rescue.mpc_locomotion import ConvexMPC
 control=object.__new__(ConvexMPC)
 control.go2=SimpleNamespace(R_world_to_body=np.eye(3),R_z=np.eye(3),velocity_world=np.array([.9,0.,0.]),
   kinematic_velocity_world=np.array([.2,0.,0.]),mjdata=SimpleNamespace(qvel=np.zeros(6)))
 control.raw_odometry=True
 assert control.measured_velocity().vx==.2
 control.raw_odometry=False
 assert control.measured_velocity().vx==.9


def test_map_rebuild_restores_support_in_corrected_frame():
 from go2_rescue.localization import Keyframe
 c,g=grid();n=NavigationSupport(c,Seg(False));f=observation();pose=Pose(quality=1.)
 n.support_elevation=.2;g.ground_height=.2;n.prepare(f,pose,g);g.update(f.points,pose);n.update(f,pose,g)
 key=Keyframe(pose,f.points.copy(),np.empty((0,3)),f.timestamp)
 n.capture_keyframe(key,f,g)
 key.pose=Pose(1.,0.,0.,quality=1.)
 g.clear();n.rebuild(g,[key])
 assert abs(g.ground_height-.2)<1e-6
 assert abs(g.floor[g.cell([1.,0.]) ]-.2)<.001
 assert np.isfinite(g.floor[g.cell([1.4,0.])])
 assert not np.isfinite(g.floor[g.cell([-.4,0.])])


def test_cached_rgb_is_reprojected_to_lidar_timestamp():
 from go2_rescue.types import Velocity
 class PatchSeg:
  def segment(self,image):
   p=np.zeros((3,240,320),np.float32);p[2]=1.
   # Static floor point [1.5,.1,-.33] projects near (143.5,187.6).
   p[:,184:192,140:148]=0.;p[0,184:192,140:148]=1.
   return p
 c,g=grid();n=NavigationSupport(c,PatchSeg());first=observation();n.observe(first)
 current=Frame(1.5,np.array([[1.4,.1,-.33]]),velocity=Velocity(.2),support_points=first.support_points)
 n.observe(current)
 assert n.classify_ground(current.points,current).tolist()==[True]
 expired=Frame(1.7,current.points,velocity=Velocity(),support_points=first.support_points)
 n.observe(expired)
 assert n.classify_ground(expired.points,expired).tolist()==[False]


def test_colour_keepout_with_contact_plane_does_not_require_filtered_ground_echo():
 from go2_rescue.hazards import WaterDetector
 c,g=grid();c.water_detection=True;g.support_plane=np.array([0.,0.,-.33])
 f=observation();f.image[165:190,135:175]=[255,190,0]
 detector=WaterDetector(c);polygons=detector.detect(f.image,f,Pose(),g)
 assert len(polygons)==1
 assert detector.last_sources==['rgb_stance_keepout_candidate']
 g.support_plane=None
 assert detector.detect(f.image,f,Pose(),g)==[]


def test_rgb_labelled_measured_floor_needs_no_fake_bridge_to_stance():
 c,g=grid();c.max_slope_deg=14.;g.support_plane=np.array([0.,0.,-.33])
 g.support_contacts=np.array([[-.2,0.,0.],[.2,0.,0.]])
 points=np.array([[2.,-.2,-.2],[2.,0.,-.2],[2.,.2,-.2]])
 g.visual_ground_mask=np.ones(3,bool);g.update(points,Pose())
 cell=g.cell([2.,0.]);assert abs(g.floor[cell]-.13)<.001
 assert g.floor_source[cell]==3 and g.logodds[cell]<0
 assert not np.isfinite(g.floor[g.cell([1.,0.])])


def test_refined_grid_keeps_the_observed_old_cell_footprint():
 from go2_rescue.localization import Keyframe
 c,g=grid();n=NavigationSupport(c,Seg(False));f=observation();pose=Pose(quality=1.)
 n.prepare(f,pose,g);g.update(f.points,pose);n.update(f,pose,g)
 key=Keyframe(pose,f.points.copy(),np.empty((0,3)),f.timestamp);n.capture_keyframe(key,f,g)
 c.resolution=.05;fine=GridMap(c);n.rebuild(fine,[key])
 for x in [.225,.275]:
  for y in [.225,.275]:assert np.isfinite(fine.floor[fine.cell([x,y])])

def test_connected_ramp_extends_floor_without_erasing_step_or_wall():
    x, y = np.meshgrid([.15, .35, .55, .75], [-.15, .15])
    z = -.33 + .34 * (x - .15)
    ramp = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    obstacles = np.array([[.55, .15, .08], [.55, -.15, .22], [1.5, .15, .05]])
    points = np.vstack([ramp, obstacles])
    labels = connected_ramp_returns(points, np.ones(len(points), bool),
                                    np.array([0., 0., -.33]), .1, .45)
    assert labels[:len(ramp)].all()
    assert not labels[len(ramp):].any()
