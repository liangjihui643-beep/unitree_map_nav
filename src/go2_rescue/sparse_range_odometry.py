"""Shadow odometry using only compressed returns with bounded XY spread."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from .geometry import gravity_rotation, rot2, wrap
from .lidar_obs import decode_lidar_obs
from .localization import Localization
from .types import Pose


class SparseRangeOdometry:
    """Align independent scan windows; IMU fixes yaw, velocity bridges gaps."""

    def __init__(self, config):
        self.c = config
        self.last = self.yaw_zero = None
        self.xy = np.zeros(2)
        self.history = deque()
        self.last_correction = -np.inf
        self.diagnostics = {}
        az = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        fov = config.lidar_vertical_fov_deg
        middle = config.lidar_elevation_center_deg
        el = np.deg2rad(np.linspace(middle-fov/2,middle+fov/2,9))
        self.rays = np.stack((np.cos(az)[:,None]*np.cos(el),
                              np.sin(az)[:,None]*np.cos(el),
                              np.broadcast_to(np.sin(el),(72,9))),axis=-1)

    def endpoints(self, packet, quaternion):
        ids, distances = decode_lidar_obs(packet,self.c.min_range,self.c.max_range)
        level = gravity_rotation(quaternion)
        points = np.array([0.,0.,.10]) @ level.T + (self.rays[ids] @ level.T)*distances[:,None,None]
        result = []
        for hypotheses in points:
            xy = hypotheses[hypotheses[:,2]>=self.c.lidar_height_floor-.025,:2]
            if len(xy) and np.linalg.norm(np.ptp(xy,axis=0)) <= .10:
                result.append((xy.min(axis=0)+xy.max(axis=0))/2)
        return np.asarray(result).reshape(-1,2)

    def update(self, packet, stamp, quaternion, velocity):
        heading = Localization._imu_yaw(quaternion)
        if self.last is None:
            self.yaw_zero = heading
        elif stamp <= self.last:
            raise ValueError("range timestamps must increase")
        yaw = wrap(heading-self.yaw_zero)
        dt = 0. if self.last is None else stamp-self.last
        self.last = stamp
        rotation = rot2(yaw)
        if dt<=.5:
            self.xy += rotation @ (velocity.array()[:2]*dt)
        else:
            self.history.clear()
        endpoints = self.endpoints(packet,quaternion)
        cloud = endpoints @ rotation.T + self.xy
        self.history.append((stamp,cloud))
        while self.history and stamp-self.history[0][0]>3.:
            self.history.popleft()
        self.diagnostics = {"points":len(cloud), "accepted":False}
        if stamp-self.last_correction >= .65:
            old = [x for t,x in self.history if .85<=stamp-t<=2.4 and len(x)]
            new = [x for t,x in self.history if stamp-t<=.55 and len(x)]
            if len(old)>=6 and len(new)>=3:
                delta = self._align(np.vstack(new),np.vstack(old))
                if delta is not None:
                    correction=.5*delta
                    self.xy += correction
                    self.history=deque((t,x+correction if stamp-t<=.55 else x) for t,x in self.history)
                    self.last_correction=stamp
                    self.diagnostics.update(accepted=True, delta_m=correction.tolist())
        return Pose(*self.xy,yaw)

    def _align(self, source, target):
        # Downsample repeated returns before counting geometric constraints.
        def distinct(points):
            _, indices=np.unique(np.floor(points/.05),axis=0,return_index=True)
            return points[indices]
        source,target=distinct(source),distinct(target)
        if min(len(source),len(target))<12:
            return None
        tree=cKDTree(target)
        distance, ids=tree.query(source)
        valid=distance<.14
        if valid.sum()<10 or valid.mean()<.5:
            return None
        # Reject a single wall when longitudinal motion is underconstrained.
        if np.linalg.eigvalsh(np.cov(source[valid].T))[0]<.035:
            return None
        residual=target[ids[valid]]-source[valid]
        delta=np.median(residual,axis=0)
        if not .004<=np.linalg.norm(delta)<=.10:
            return None
        check,_=tree.query(source+delta)
        rms0=np.mean(np.minimum(distance,.15)**2)
        rms1=np.mean(np.minimum(check,.15)**2)
        if rms1>=.85*rms0:
            return None
        return delta
