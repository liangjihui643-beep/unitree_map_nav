"""Short-range RGB floor evidence anchored to measured joint/IMU support.

No occupancy-ray implies floor: each added cell must independently project
inside an eroded high-confidence ground image, agree with free lidar visibility,
and remain within 1.25 m of the measured support plane. Measured terrain wins.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import gravity_rotation, rot2
from .stance_surface import StanceSurface
from .terrain import ground_surface
from .visual_odometry import MetricVisualOdometry


def connected_ramp_returns(points, visual_ground, stance_plane, resolution, max_gap):
    """Extend ground labels from stance onto a connected measured ramp."""
    candidates = visual_ground & (np.linalg.norm(points[:, :2], axis=1) < 2.5)
    ramp = np.zeros(len(points), dtype=bool)
    if np.count_nonzero(candidates) < 4:
        return ramp
    measured = points[candidates]
    samples = ground_surface(measured, stance_plane[2], resolution, min(max_gap, .30), 20.)
    if len(samples) < 4 or np.linalg.matrix_rank(samples[:, :2] - samples[:, :2].mean(axis=0)) < 2:
        return ramp
    # Match measured lower-envelope cells, without labelling elevated returns in those cells.
    keys = np.floor(samples[:, :2] / resolution).astype(int)
    cells = {tuple(key): sample[2] for key, sample in zip(keys, samples)}
    for i in np.flatnonzero(candidates):
        key = tuple(np.floor(points[i, :2] / resolution).astype(int))
        if key in cells and abs(points[i, 2] - cells[key]) <= .025:
            ramp[i] = True
    return ramp


class NavigationSupport:
    def __init__(self, config, segmenter):
        self.c, self.segmenter = config, segmenter
        self.surface = StanceSurface()
        self.vo=MetricVisualOdometry(config)
        self.visual_origin=None
        self.visual_delta=None
        self.odom_position=np.zeros(3)
        self.odom_stamp=None
        self.camera_reference=None
        self.previous = None
        self.support_elevation=0.
        self.status = {}
        self.probability = None
        self.probability_stamp = -1.

    def observe(self,frame):
        if getattr(self,"observed_stamp",None)!=frame.timestamp:
            self.observed_stamp=frame.timestamp
            rotation=Rotation.from_quat(np.roll(frame.quaternion,-1)).as_matrix()
            old_plane=self.surface.plane
            plane=self.surface.update(frame)
            if self.odom_stamp is not None:
                dt=frame.timestamp-self.odom_stamp
                velocity=np.array([frame.velocity.vx,frame.velocity.vy,0.])
                if self.c.kinematic_velocity_frame=="gravity_body":
                    delta=rotation@gravity_rotation(frame.quaternion).T@velocity*dt
                    if plane is not None:
                        delta[2]=float(plane[:2]@velocity[:2])*dt
                        if old_plane is not None:delta[2]-=float(plane[2]-old_plane[2])
                    self.odom_position+=delta
                else:
                    self.odom_position+=rotation@velocity*dt
            self.odom_stamp=frame.timestamp
            if frame.image is not None and frame.image_age<=self.c.max_sensor_age:
                self.probability=self.segmenter.segment(frame.image)
                self.probability_stamp=frame.timestamp
                self.camera_reference=(rotation.copy(),self.odom_position.copy())
                if self.c.metric_visual_odometry:
                    self.visual_delta=self.vo.update(frame,self.surface.plane,self.probability)
        return self.surface.plane

    def visual_prior(self,frame):
        if self.probability_stamp!=frame.timestamp or self.visual_delta is None or self.visual_origin is None:
            return None
        stamp,pose=self.visual_origin
        if abs(stamp-self.visual_delta['from_stamp'])>1e-5:
            return None
        return pose.xy()+rot2(pose.yaw)@self.visual_delta['delta_gravity'][:2]

    def accept_pose(self,frame,pose):
        if self.probability_stamp==frame.timestamp:
            self.visual_origin=(frame.timestamp,type(pose)(**vars(pose)))

    def prepare(self, frame, pose, grid):
        self.prepared_stamp=frame.timestamp
        plane = self.observe(frame)
        grid.support_plane=None
        grid.support_contacts=np.empty((0,3))
        self.status = dict(self.surface.status, visual_cells=0, visual_odometry=self.vo.status)
        if plane is None:
            return
        rotation = rot2(pose.yaw)
        if self.previous is not None:
            delta = pose.xy()-self.previous
            if np.linalg.norm(delta) < .25:
                self.support_elevation += float((rotation @ plane[:2]) @ delta)
        self.previous = pose.xy().copy()
        # Separate support elevation from changing base-to-contact height.
        grid.ground_height=self.support_elevation-float(plane[2])-self.c.base_height
        self.status.update(support_elevation_m=self.support_elevation,base_to_support_m=-float(plane[2]))
        self.current_height=grid.ground_height
        grid.support_plane=plane.copy()
        soles=frame.support_points @ gravity_rotation(frame.quaternion).T
        grid.support_contacts=np.column_stack([soles[:,:2]@rotation.T+pose.xy(),
            soles[:,2]+self.c.base_height+grid.ground_height])

    def classify_ground(self,points,frame,*,points_frame="body",allow_ramp=False):
        """Classify aligned returns; explicitly declare body or gravity frame."""
        if points_frame not in {"body", "gravity"}:
            raise ValueError("points_frame must be body or gravity")
        points=np.asarray(points)
        level=gravity_rotation(frame.quaternion)
        body_points=points if points_frame=="body" else points@level
        result=np.zeros(len(points),dtype=bool)
        if self.probability is None or self.camera_reference is None or frame.timestamp-self.probability_stamp>.6:
            return result
        transform=np.asarray(self.c.body_from_camera)
        # StanceSurface and projected terrain use gravity-aligned points
        # (column convention R @ p, therefore row arrays multiply R.T).
        gravity_points=body_points @ level.T
        plane=self.surface.plane
        if plane is None:
            return result
        plane_residual=gravity_points[:,2]-(gravity_points[:,:2]@plane[:2]+plane[2])
        current=Rotation.from_quat(np.roll(frame.quaternion,-1)).as_matrix()
        previous,position=self.camera_reference
        # Lidar points are expressed in the current body frame. Move them into
        # the cached image's body frame through world orientation and the
        # independent joint-odometry delta; do not rotate by gravity first
        # (that double-applies roll/pitch and shifts ground pixels on ramps).
        body=(body_points@current.T+self.odom_position-position)@previous
        camera=(body-transform[:3,3]) @ transform[:3,:3]
        uvw=camera @ np.asarray(self.c.camera_matrix).T
        uv=np.rint(uvw[:,:2]/np.maximum(uvw[:,2:],1e-6)).astype(int)
        h,w=self.probability.shape[1:]
        valid=(camera[:,2]>.1)&(uv[:,0]>=2)&(uv[:,0]<w-2)&(uv[:,1]>=2)&(uv[:,1]<h-2)
        visual_ground = np.zeros(len(points), dtype=bool)
        for i in np.flatnonzero(valid):
            u,v=uv[i]
            patch=self.probability[:,v-2:v+3,u-2:u+3]
            visual_ground[i] = np.all(patch.argmax(0) == 0) and patch[0].mean() > .85
        result = visual_ground & (abs(plane_residual) < .06)
        if allow_ramp and np.any(visual_ground & ~result):
            result |= connected_ramp_returns(gravity_points, visual_ground, plane,
                                             self.c.resolution, self.c.ground_max_gap)
        return result

    def update(self, frame, pose, grid):
        self.last_added=[]
        if getattr(self,"prepared_stamp",None)!=frame.timestamp:
            self.prepare(frame,pose,grid)
        plane=self.surface.plane
        if plane is None:
            return
        rotation=rot2(pose.yaw)
        # Contact-enclosed standing envelope. The old circular 23 cm evidence
        # was smaller than the known 55 cm bounded stance-continuity turning envelope,
        # preventing any safe turn even while already standing on the floor.
        center=np.array(grid.cell(pose.xy()));radius=int(np.ceil(.62/self.c.resolution))
        di,dj=np.mgrid[-radius:radius+1,-radius:radius+1]
        support_cells=center+np.column_stack([di.ravel(),dj.ravel()])
        inside=(support_cells>=0).all(1)&(support_cells<grid.n).all(1)
        support_cells=support_cells[inside]
        support_xy=(grid.origin+(support_cells+.5)*self.c.resolution-pose.xy())@rotation
        enclosed=np.linalg.norm(support_xy,axis=1)<=.55
        for cell,xy in zip(support_cells[enclosed],support_xy[enclosed]):
            cell=tuple(cell)
            if grid.logodds[cell]>.5 or grid.floor_source[cell]==3:
                continue
            grid.floor[cell]=float(xy@plane[:2]+plane[2]+self.c.base_height+grid.ground_height)
            grid.floor_source[cell]=5
            self.last_added.append((grid.world(cell),float(grid.floor[cell]),5))
            grid.seen[cell]=True
            grid.logodds[cell]=min(-1.,grid.logodds[cell])
        grid._free_cache=grid._surface_safe=None
        # Support directly under the body; no extrapolation before RGB arrives.
        if frame.image is None or frame.image_age > self.c.max_sensor_age:
            return
        prob = self.probability
        ground = ((prob.argmax(0) == 0) & (prob[0] > .85)).astype(np.uint8)
        ground = cv2.erode(ground, np.ones((5, 5), np.uint8))
        lo=np.maximum(0,np.array(grid.cell(pose.xy()-1.6)))
        hi=np.minimum(grid.n-1,np.array(grid.cell(pose.xy()+1.6)))
        i,j=np.mgrid[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
        cells=np.column_stack([i.ravel(),j.ravel()])
        xy = grid.origin+(cells+.5)*self.c.resolution
        local = (xy-pose.xy()) @ rotation
        valid = (local[:,0] > -.25) & (local[:,0] < 1.25) & (abs(local[:,1]) < .85)
        cells, local = cells[valid], local[valid]
        # Fill between sparse azimuth rays only when the image confirms floor
        # and a nearby measured ray puts its occluder farther away.
        # Direct laser visibility is the strongest support evidence, but a
        # downward sparse lidar can leave the next ramp foothold unhit. Keep
        # that evidence separate so high-confidence RGB ground can bridge only
        # a short, connected interval on the current stance plane.
        ray_visible=grid.seen[tuple(cells.T)] & (grid.logodds[tuple(cells.T)] < -.5)
        if len(frame.points):
            rays=frame.points @ gravity_rotation(frame.quaternion).T
            angles=np.arctan2(rays[:,1],rays[:,0]);ranges=np.linalg.norm(rays[:,:2],axis=1)
            angle=np.arctan2(local[:,1],local[:,0]);distance=np.linalg.norm(local,axis=1)
            delta=np.abs((angle[:,None]-angles[None]+np.pi)%(2*np.pi)-np.pi)
            near=delta<=np.deg2rad(7.6)
            occluder=np.min(np.where(near,ranges[None],np.inf),axis=1)
            ray_visible |= near.any(1) & (distance < occluder-.05)
        z = local @ plane[:2] + plane[2]
        gravity = np.column_stack([local, z])
        body = gravity @ gravity_rotation(frame.quaternion)
        transform = np.asarray(self.c.body_from_camera)
        camera = (body-transform[:3,3]) @ transform[:3,:3]
        k = np.asarray(self.c.camera_matrix)
        uvw = camera @ k.T
        uv = np.rint(uvw[:,:2]/np.maximum(uvw[:,2:],1e-6)).astype(int)
        h,w = ground.shape
        valid = (camera[:,2] > .1) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
        selected = np.flatnonzero(valid)
        selected = selected[ground[uv[selected,1],uv[selected,0]] > 0]
        # A visually supported cell may extend the measured plane only when it
        # is close to existing floor, inside the short forward foothold window,
        # and the current location is an explicitly configured slope region.
        # This is a local sensor-fusion bridge, not a map-wide floor guess.
        connected_floor=cv2.dilate(np.isfinite(grid.floor).astype(np.uint8),np.ones((5,5),np.uint8))>0
        slope_region=self.c.slope_limit(pose.xy())>self.c.max_slope_deg+1e-6
        accepted = 0
        visual_only = 0
        visual_only_selected = selected[~ray_visible[selected]]
        visual_only_near = (local[visual_only_selected,0] >= .04) & (local[visual_only_selected,0] <= .85) & (abs(local[visual_only_selected,1]) <= .45)
        visual_only_connected = (connected_floor[tuple(cells[visual_only_selected].T)]
                                 if len(visual_only_selected) else np.empty(0,dtype=bool))
        visual_only_confident = (prob[0,uv[visual_only_selected,1],uv[visual_only_selected,0]] >= .92
                                 if len(visual_only_selected) else np.empty(0,dtype=bool))
        visual_only_clear = ((grid.logodds[tuple(cells[visual_only_selected].T)] <= .5)
                             & (grid.hazard_votes[tuple(cells[visual_only_selected].T)] < 2)
                             if len(visual_only_selected) else np.empty(0,dtype=bool))
        for i in selected:
            cell = tuple(cells[i])
            if not ray_visible[i]:
                confidence=float(prob[0,uv[i,1],uv[i,0]])
                if (not slope_region or confidence < .92 or local[i,0] < .04 or local[i,0] > .85
                        or abs(local[i,1]) > .45 or not connected_floor[cell]):
                    continue
            # Lidar elevations and contact evidence cannot be overwritten by
            # projected RGB. Reobserve projected cells as the support changes.
            if grid.floor_source[cell] == 3 or grid.logodds[cell] > .5 or grid.hazard_votes[cell] >= 2:
                continue
            height = float(z[i]+self.c.base_height+grid.ground_height)
            grid.floor[cell] = height
            grid.floor_source[cell] = 4
            self.last_added.append((grid.world(cell),height,4))
            grid.seen[cell] = True
            grid.logodds[cell] = max(-4.,grid.logodds[cell]-.4)
            accepted += 1
            visual_only += int(not ray_visible[i])
        grid._free_cache = grid._surface_safe = None
        self.status.update(visual_cells=accepted, visual_only_cells=visual_only,
                           visual_ground_pixels=int(ground.sum()),
                           visual_projection_candidates=int(len(selected)),
                           visual_only_candidates=int(len(visual_only_selected)),
                           visual_only_near=int(visual_only_near.sum()),
                           visual_only_connected=int(visual_only_connected.sum()),
                           visual_only_confident=int(visual_only_confident.sum()),
                           visual_only_clear=int(visual_only_clear.sum()),
                           max_extrapolation_m=1.25)



    def capture_keyframe(self,key,frame,grid):
        if key.timestamp is None or abs(key.timestamp-frame.timestamp)>1e-5:
            return
        key.support_height=grid.ground_height
        key.support_plane=None if self.surface.plane is None else self.surface.plane.copy()
        key.support_soles=np.empty((0,3)) if frame.support_points is None else frame.support_points @ gravity_rotation(frame.quaternion).T
        key.visual_ground_mask = self.classify_ground(
            key.points, frame, points_frame="gravity", allow_ramp=self.c.map_connected_ramp_ground)
        key.visual_resolution=float(grid.c.resolution)
        key.visual_floor=[]
        added=getattr(self,"last_added",[]) if getattr(self,"prepared_stamp",None)==frame.timestamp else []
        for xy,z,source in added:
            local=(xy-key.pose.xy())@rot2(key.pose.yaw)
            key.visual_floor.append([*local,z-grid.ground_height-self.c.base_height,source])

    def rebuild(self,grid,keys,corrections=()):
        height=getattr(self,"current_height",grid.ground_height)
        if corrections and self.previous is not None:
            old,new=min(corrections,key=lambda pair:np.linalg.norm(pair[0].xy()-self.previous))
            self.previous=new.xy()+rot2(new.yaw-old.yaw)@(self.previous-old.xy())
        for key in keys:
            grid.ground_height=getattr(key,"support_height",0.)
            grid.support_plane=getattr(key,"support_plane",None)
            soles=getattr(key,"support_soles",np.empty((0,3)))
            grid.support_contacts=np.column_stack([soles[:,:2]@rot2(key.pose.yaw).T+key.pose.xy(),
                soles[:,2]+self.c.base_height+grid.ground_height])
            grid.visual_ground_mask=getattr(key,"visual_ground_mask",None)
            grid.update(key.points, key.pose, getattr(key, "quaternion", None), getattr(key, "timestamp", None))
            ratio=max(1,int(np.ceil(getattr(key,"visual_resolution",grid.c.resolution)/grid.c.resolution)))
            old_res=getattr(key,"visual_resolution",grid.c.resolution)
            offsets=(np.arange(ratio)+.5)*old_res/ratio-old_res/2
            for x,y,z,source in getattr(key,"visual_floor",[]):
                for dx in offsets:
                    for dy in offsets:
                        xy=rot2(key.pose.yaw)@np.array([x+dx,y+dy])+key.pose.xy();cell=grid.cell(xy)
                        if not grid.inside(cell) or grid.logodds[cell]>.5 or grid.floor_source[cell]==3:
                            continue
                        plane=getattr(key,"support_plane",None)
                        gradient=0. if plane is None else float(plane[:2]@np.array([dx,dy]))
                        grid.floor[cell]=z+gradient+grid.ground_height+self.c.base_height
                        grid.floor_source[cell]=int(source);grid.seen[cell]=True;grid.logodds[cell]=min(-1.,grid.logodds[cell])
        grid.ground_height=height
        grid.support_plane=None;grid.support_contacts=np.empty((0,3));grid.visual_ground_mask=None
        grid._free_cache=grid._surface_safe=None
