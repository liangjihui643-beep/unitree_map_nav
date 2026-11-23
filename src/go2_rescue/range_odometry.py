"""No-XYZ odometry baseline using azimuth/range projection and IMU yaw."""
from collections import deque
import numpy as np

from .geometry import gravity_rotation, transform_xy, rot2, wrap
from .lidar_obs import decode_lidar_obs
from .localization import Localization, register
from .types import Pose


class RangeOdometry:
    """Experimental scan matcher. Projected ranges are not occupied endpoints."""

    def __init__(self, min_range=.05, max_range=12.):
        self.min_range, self.max_range = min_range, max_range
        self.yaw_zero = self.last_stamp = None
        self.pose = Pose()
        self.reference = deque(maxlen=5)

    def update(self, obs, stamp, quaternion, velocity):
        azimuth, distance = decode_lidar_obs(obs, self.min_range, self.max_range)
        angle = -np.pi + azimuth * (2*np.pi/72)
        scan = np.column_stack((distance*np.cos(angle), distance*np.sin(angle),
                                np.zeros(len(distance)))) @ gravity_rotation(quaternion).T
        yaw = Localization._imu_yaw(quaternion)
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("range odometry timestamps must increase")
        dt = 0. if self.last_stamp is None else stamp-self.last_stamp
        self.last_stamp = stamp
        if not len(self.reference):
            if len(scan)<16:
                return Pose()
            self.yaw_zero = yaw
            self.pose = Pose(quality=.65,sigma=.10)
            self.reference.append(transform_xy(scan,self.pose))
            return self.pose
        yaw = wrap(yaw-self.yaw_zero)
        guess = Pose(*(self.pose.xy()+rot2(yaw) @ (velocity.array()[:2]*min(dt,.2))),
                     yaw,quality=0.,sigma=min(2.,self.pose.sigma+.05))
        if not .0<dt<=.5 or len(scan)<16:
            self.pose=guess
            return guess
        reference=np.vstack(self.reference)
        candidate,overlap,rmse=register(scan,reference,guess,limit=.45)
        reliable = (overlap>=.55 and rmse<=.11 and abs(wrap(candidate.yaw-yaw))<.10
                    and np.linalg.norm(candidate.xy()-guess.xy()) <= .14)
        if not reliable:
            self.pose=guess
            return guess
        candidate.quality=float(overlap*np.exp(-rmse/.25))
        candidate.sigma=float(max(.05,rmse))
        self.pose=candidate
        self.reference.append(transform_xy(scan,candidate))
        return candidate
