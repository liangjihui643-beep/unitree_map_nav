"""Navigation on an explicit offline prior; never writes prior into mapping.

Online inputs are estimated pose, measured obstacles and RGB candidates.
There are no simulator imports, root-state queries or scene queries here.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from .geometry import rot2, transform_xy, wrap
from .mapping import GridMap
from .planning import astar, DWA, Explorer
from .types import Pose, Step, Velocity


class PriorNavigationGrid(GridMap):
    def __init__(self, config, path):
        with np.load(path) as p:
            self.base_safe=p['floor'].copy();self.rock=p['rock'].copy()
            self.origin=p['origin'].copy();resolution=float(p['resolution'])
        self.c=replace(config,resolution=resolution,map_size=resolution*len(self.base_safe))
        self.n=len(self.base_safe)
        self.seen=self.base_safe.copy() # Prior-only grid; never saved as measured map.
        self.visual_votes=np.zeros_like(self.rock,dtype=np.uint8)
        self.target_keepout=np.zeros_like(self.rock)
        self.last_visual_view=None
        self._has_hazards=False
        self.hazard_votes=np.zeros_like(self.visual_votes)
        self._free_cache=self._surface_safe=None
        i,j=np.mgrid[0:self.n,0:self.n]
        world=self.origin+(np.stack([i,j],axis=-1)+.5)*resolution+[.8,.8]
        # Declared straight east bypass, public-map coordinates. Keep actual
        # body dimensions; use the already supported 2cm portal buffer only
        # here. Full oriented-body checks still apply to turns and every move.
        self.portal=(world[:,:,0]>11.35)&(world[:,:,0]<12.)&(world[:,:,1]>4.7)&(world[:,:,1]<9.)

    def traversable(self):
        if self._free_cache is None:
            # Known rock footprints also get clearance for uncertain visual
            # boundaries. No terrain height is invented in the measured map.
            rough=binary_dilation(self.rock,iterations=2)|(self.visual_votes>=2)
            self._surface_safe=self.base_safe & ~rough & ~self.target_keepout & (self.hazard_votes<2)
            clearance=distance_transform_edt(self._surface_safe)*self.c.resolution
            margin=np.where(self.portal,min(.02,self.c.safety_margin),self.c.safety_margin)
            self._free_cache=self._surface_safe & (clearance>=self.c.footprint_half_width+margin)
        return self._free_cache


class KnownMapNavigator:
    def __init__(self,config):
        root=Path(__file__).resolve().parents[2]
        self.grid=PriorNavigationGrid(config,root/config.navigation_prior)
        self.planner=DWA(self.grid.c);self.explorer=Explorer()
        self.passage_planner=DWA(replace(self.grid.c,safety_margin=min(.02,self.grid.c.safety_margin),max_vx=.15,max_vy=.05))
        self.slope_planner=DWA(replace(self.grid.c,max_vx=.12,max_reverse_vx=.08,max_vy=.035,max_wz=.18))
        self.points=[dict(w,stage=st['name']) for st in json.loads((root/config.course_route).read_text())['stages'] for w in st['waypoints']]
        self.index=0;self.path=np.empty((0,2));self.goal=None
        self.last_plan=-np.inf;self.last_progress=0.;self.progress_xy=None
        self.status={};self.recovery=np.empty((0,3));self.recovery_index=0
        self.adjusted=[]
        self.unreachable=[]
        self.progress_yaw=None
        self.recovery_margin=.02
        self.last_trusted_pose = None
        self.last_trusted_stamp = -np.inf
        self.scan_turn_anchor = None
        self.observation_started = None
        self.aligned_index = None

    def reacquire(self, frame, heading, points, support_plane):
        """Rotate at the last trusted position when XY cannot be updated."""
        if heading is None or self.last_trusted_pose is None or support_plane is None:
            return None
        if np.linalg.norm(support_plane[:2]) > np.tan(np.deg2rad(5.)):
            return None
        trusted = self.last_trusted_pose
        if self.scan_turn_anchor is None or np.linalg.norm(trusted.xy()-self.scan_turn_anchor.xy()) > .20:
            if frame.timestamp-self.last_trusted_stamp > .35:
                return None
            self.scan_turn_anchor = trusted
            self.scan_turn_started = frame.timestamp
        if frame.timestamp-self.scan_turn_started > 9.5:
            return None
        swept_radius = np.hypot(self.grid.c.footprint_half_length, self.grid.c.footprint_half_width)+.08
        residual = points[:, 2]-(points[:, :2]@support_plane[:2]+support_plane[2])
        if np.any((residual > .08) & (residual < self.grid.c.body_clearance)
                  & (np.linalg.norm(points[:, :2], axis=1) < swept_radius)):
            return None
        delta = wrap(heading-self.scan_turn_anchor.yaw)
        if delta <= -.65 or abs(delta) > .8:
            return None
        pose = Pose(trusted.x, trusted.y, heading)
        return self.planner.safe_turn(pose,-.22,self.grid)

    def alignment_motion(self,pose,frame,events,dt):
        while self.recovery_index<len(self.recovery):
            target=self.recovery[self.recovery_index]
            if np.linalg.norm(target[:2]-pose.xy())<.035 and abs(wrap(target[2]-pose.yaw))<.06:
                self.recovery_index+=1
            else:break
        if self.recovery_index>=len(self.recovery):
            self.recovery=np.empty((0,3));return None
        target=self.recovery[self.recovery_index]
        local=rot2(pose.yaw).T@(target[:2]-pose.xy())
        velocity=np.clip(.8*local,-.05,.05)
        turn=float(np.clip(wrap(target[2]-pose.yaw),-.12,.12))
        half=np.array([self.grid.c.footprint_half_length,self.grid.c.footprint_half_width])+self.recovery_margin
        x,y=np.meshgrid(np.linspace(-half[0],half[0],11),np.linspace(-half[1],half[1],9));offsets=np.c_[x.ravel(),y.ravel()]
        times=np.linspace(0.,.3,5)
        future=pose.xy()+times[:,None]*(rot2(pose.yaw)@velocity)
        safe=self.grid.footprints_safe(future,pose.yaw+times*turn,offsets).all()
        if not safe:
            self.recovery=np.empty((0,3));return Step(Velocity(),pose,'survey_alignment',events,'alignment_replan_needed')
        return Step(Velocity(*velocity,turn),pose,'survey_alignment',events,'full_body_checked_alignment')

    def sync_hazards(self, measured_grid):
        cells=np.argwhere(self.grid.base_safe)
        xy=self.grid.origin+(cells+.5)*self.grid.c.resolution
        source=np.floor((xy-measured_grid.origin)/measured_grid.c.resolution).astype(int)
        inside=(source>=0).all(1)&(source<measured_grid.n).all(1)
        hazard=np.zeros_like(self.grid.hazard_votes)
        hazard[tuple(cells[inside].T)]=measured_grid.hazard_votes[tuple(source[inside].T)]
        if not np.array_equal(hazard,self.grid.hazard_votes):
            self.grid.hazard_votes=hazard
            self.grid._has_hazards=bool((hazard>=2).any())
            self.grid._free_cache=self.grid._surface_safe=None

    def observe_visual_rubble(self,xy,pose):
        if self.grid.last_visual_view is not None and np.linalg.norm(pose.xy()-self.grid.last_visual_view)<.20:
            return
        cells=np.array([self.grid.cell(p) for p in xy],int).reshape(-1,2)
        if not len(cells):return
        inside=(cells>=0).all(1)&(cells<self.grid.n).all(1);cells=cells[inside]
        if not len(cells):return
        mask=np.zeros_like(self.grid.rock);mask[tuple(cells.T)]=True
        mask=binary_dilation(mask,iterations=3)
        self.grid.visual_votes=np.minimum(10,self.grid.visual_votes+mask)
        self.grid.last_visual_view=pose.xy().copy()
        self.grid._free_cache=self.grid._surface_safe=None

    def replace_visual_rubble(self, centers, resolution):
        """Apply the latest complete perception footprint, including withdrawals."""
        grid=self.grid
        mask=np.zeros_like(grid.rock)
        half=resolution/2.
        first_center=grid.origin+.5*grid.c.resolution
        lower=np.ceil((centers-half-first_center)/grid.c.resolution).astype(int)
        upper=np.ceil((centers+half-first_center)/grid.c.resolution).astype(int)
        for start,stop in zip(np.maximum(lower,0),np.minimum(upper,grid.n)):
            if (start<stop).all():
                mask[start[0]:stop[0],start[1]:stop[1]]=True
        if np.array_equal(mask,grid.visual_votes>=2):
            return
        grid.visual_votes[:]=mask.astype(np.uint8)*2
        grid._free_cache=grid._surface_safe=None

    def avoid_confirmed_targets(self, positions):
        keepout = np.zeros_like(self.grid.target_keepout)
        radius = self.grid.c.ball_radius + .12
        extent = int(np.ceil(radius/self.grid.c.resolution))
        for xy in positions:
            i,j = self.grid.cell(xy)
            for a in range(i-extent,i+extent+1):
                for b in range(j-extent,j+extent+1):
                    cell = a,b
                    if self.grid.inside(cell) and np.linalg.norm(self.grid.world(cell)-xy) <= radius:
                        keepout[cell] = True
        if not np.array_equal(self.grid.target_keepout,keepout):
            self.grid.target_keepout = keepout
            self.grid._free_cache = self.grid._surface_safe = None

    def step(self,frame,pose,points,events,dt,plane):
        self.last_trusted_pose = Pose(**vars(pose))
        self.last_trusted_stamp = frame.timestamp
        narrow = self.index < len(self.points) and self.points[self.index].get("narrow_passage")
        progress_distance = .06 if narrow else .15
        if self.progress_xy is None or np.linalg.norm(pose.xy()-self.progress_xy)>progress_distance or abs(wrap(pose.yaw-(self.progress_yaw if self.progress_yaw is not None else pose.yaw)))>.25:
            self.progress_xy=pose.xy().copy();self.progress_yaw=pose.yaw;self.last_progress=frame.timestamp
        if self.index>=len(self.points):
            self.status.update(route_complete=True,all_requested_viewpoints_reached=not self.unreachable,
                               unreachable_viewpoints=self.unreachable)
            return Step(Velocity(),pose,'finished',events,'known_map_survey_route_complete')
        w=self.points[self.index];target=np.asarray(w['xy'],float)
        self.status.update(stage=w['stage'],index=self.index,count=len(self.points))
        free=self.grid.traversable();start=self.grid.cell(pose.xy())
        if not self.grid.inside(start):
            return Step(Velocity(),pose,'blocked',events,'outside_declared_navigation_prior')
        # An inspection viewpoint must be outside the forbidden rock footprint.
        # Snap only this transit survey point, recording the displacement.
        if self.goal is not None and not free[self.grid.cell(self.goal)]:
            self.goal = None
            self.path = np.empty((0, 2))
            self.recovery = np.empty((0, 3))
            self.recovery_index = 0
            self.aligned_index = None
            self.last_plan = -np.inf
        alignment=self.alignment_motion(pose,frame,events,dt)
        if alignment is not None:return alignment
        if self.goal is None:
            components,_=label(free)
            if not free[start]:
                return Step(Velocity(),pose,'blocked',events,'estimated_body_outside_prior_clearance')
            candidates=np.argwhere(free & (components==components[start]))
            xy=self.grid.origin+(candidates+.5)*self.grid.c.resolution
            order=np.argsort(np.linalg.norm(xy-target,axis=1))
            self.goal=None
            for i in order[:150]:
                if np.linalg.norm(xy[i]-target)>1.:break
                p=astar(free,start,tuple(candidates[i]))
                if p:
                    self.goal=xy[i];self.path=np.array([self.grid.world(c) for c in p]);break
            if self.goal is None:
                self.unreachable.append({'index':self.index,'requested':target.tolist(),
                    'reason':'no connected safe viewpoint within 1m; not visited'})
                self.index+=1
                return Step(Velocity(),pose,'survey_route',events,'unsafe_viewpoint_recorded_unvisited')
            self.adjusted.append({'index':self.index,'requested':target.tolist(),'planned':self.goal.tolist()})
            self.last_plan = -np.inf
        tolerance=.18 if w['stage'] in {'L1_ascent','L1_descent'} else max(.25,float(w.get('tolerance',.30)))
        require_heading=('yaw_tolerance' in w or w.get('narrow_passage')) and 'yaw' in w
        if require_heading and np.linalg.norm(self.goal-pose.xy())<1.:
            error=wrap(w['yaw']-pose.yaw)
            tolerance_yaw = w.get('yaw_tolerance',.08)
            if narrow and self.aligned_index == self.index:
                tolerance_yaw += .04
            if abs(error)>tolerance_yaw:
                self.aligned_index = None
                self.observation_started = None
                turn=self.passage_planner.safe_turn(pose,float(np.clip(error,-.25,.25)),self.grid)
                if abs(turn.wz)>.01:
                    return Step(turn,pose,'survey_alignment',events,'align_before_declared_passage')
                self.recovery=self.explorer.margin_escape(self.grid,pose)
                self.recovery_margin=0.
                if not len(self.recovery):
                    old=self.grid.c
                    try:
                        self.grid.c=self.passage_planner.c
                        self.recovery=self.explorer.maneuver(self.grid,pose,goal=self.goal,goal_yaw=w['yaw'],position_tolerance=.15)
                    finally:self.grid.c=old
                    self.recovery_margin=.02
                self.recovery_index=1
                alignment=self.alignment_motion(pose,frame,events,dt)
                if alignment is not None:return alignment
                return Step(Velocity(),pose,'blocked',events,'no_safe_passage_alignment')
            self.aligned_index = self.index
        if np.linalg.norm(self.goal-pose.xy())<tolerance:
            duration = float(w.get("survey_observation_s", 0.))
            if duration:
                if self.observation_started is None:
                    self.observation_started = frame.timestamp
                if frame.timestamp-self.observation_started < duration:
                    return Step(Velocity(),pose,'survey_route',events,'survey_observation_hold')
            self.observation_started = None
            self.index+=1;self.goal=None;self.path=np.empty((0,2));self.last_progress=frame.timestamp
            return Step(Velocity(),pose,'survey_route',events,'survey_viewpoint_reached')
        self.observation_started = None
        if frame.timestamp-self.last_plan>1.:
            path=astar(free,start,self.grid.cell(self.goal),1.+.15/np.maximum(.05,distance_transform_edt(free)*self.grid.c.resolution))
            self.path=np.array([self.grid.world(c) for c in path]).reshape(-1,2);self.last_plan=frame.timestamp
        if len(self.path):
            nearest=int(np.argmin(np.linalg.norm(self.path-pose.xy(),axis=1)))
            self.path=self.path[nearest:]
        obstacles=np.empty((0,3))
        if plane is not None:
            residual=points[:,2]-(points[:,:2]@plane[:2]+plane[2])
            obstacles=points[(residual>.08)&(residual<self.grid.c.body_clearance)&(np.linalg.norm(points[:,:2],axis=1)<1.5)]
        planner=self.passage_planner if self.grid.portal[start] or w.get('narrow_passage') else self.planner
        # Stage names can advance while the body is still descending. Keep
        # measured terrain limits active through that transition.
        if plane is not None and np.linalg.norm(plane[:2])>np.tan(np.deg2rad(5.)):
            planner=self.slope_planner
        command=planner.command(pose,frame.velocity,self.path,obstacles,self.grid,dt,
                                     reverse_limit=.12,forward_limit=.12 if w['stage']=='L1_descent' else .28,
                                     face_path=not bool(w.get('reverse')),
                                     desired_yaw=w.get('yaw') if w.get('reverse') or require_heading else None)
        if command==Velocity() and len(self.path)>1:
            aim=self.path[min(5,len(self.path)-1)]-pose.xy()
            error=wrap(np.arctan2(aim[1],aim[0])-pose.yaw)
            command=planner.safe_turn(pose,float(np.clip(error,-.25,.25)),self.grid)
        if frame.timestamp-self.last_progress>12.:
            return Step(Velocity(),pose,'blocked',events,'survey_no_progress_needs_route_review')
        self.status={'stage':w['stage'],'index':self.index,'count':len(self.points),'goal':self.goal.tolist(),
                     'source':'offline_public_map_prior + estimated pose','uses_simulator_pose':False,'route_complete':False,
                     'visual_avoid_cells':int((self.grid.visual_votes>=2).sum())}
        return Step(command,pose,'survey_route',events,'known_map_route_following')
