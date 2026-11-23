"""Local support plane from joint-reconstructed stance soles and IMU."""
from collections import deque
import numpy as np
from .geometry import gravity_rotation, rot2


class StanceSurface:
    def __init__(self, max_age=.8):
        self.max_age=max_age
        self.pairs=deque()
        self.plane=None
        self.status={"source":"unavailable","slope_deg":None}

    def update(self, frame):
        now=frame.timestamp
        while self.pairs and now-self.pairs[0][0]>self.max_age:
            self.pairs.popleft()
        self.plane=None
        if frame.support_points is None or len(frame.support_points)<2:
            self.status={"source":"unavailable","reason":"too_few_stance_soles","slope_deg":None}
            return None
        soles=frame.support_points @ gravity_rotation(frame.quaternion).T
        height=-float(np.median(soles[:,2]))
        if not .20<=height<=.55 or np.ptp(soles[:,2])>.28:
            self.status={"source":"unavailable","reason":"inconsistent_stance","slope_deg":None}
            return None
        w,x,y,z=frame.quaternion
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        rotation=rot2(yaw)
        # Centering each simultaneous contact pair removes unknown base height.
        # IMU heading aligns pair directions acquired in different gait phases.
        xy=(soles[:,:2]-soles[:,:2].mean(0)) @ rotation.T
        dz=soles[:,2]-soles[:,2].mean()
        self.pairs.append((now,xy,dz))
        design=np.concatenate([row[1] for row in self.pairs])
        vertical=np.concatenate([row[2] for row in self.pairs])
        spectrum=np.linalg.eigvalsh(design.T @ design)
        if spectrum[0]<.002 or spectrum[0]<.035*spectrum[1]:
            self.plane=np.array([0.,0.,-height])
            self.status={"source":"joint_imu_stance_horizontal","slope_deg":0.,
                         "reason":"pair_directions_incomplete","support_span_s":0.}
            return self.plane
        gradient=np.linalg.lstsq(design,vertical,rcond=None)[0]
        for _ in range(3):
            residual=design @ gradient-vertical
            weights=np.minimum(1.,.012/np.maximum(abs(residual),1e-8))
            gradient=np.linalg.solve(design.T @ (weights[:,None]*design)+np.eye(2)*1e-7,
                                     design.T @ (weights*vertical))
        residual=design @ gradient-vertical
        p80=float(np.quantile(abs(residual),.8))
        slope=float(np.rad2deg(np.arctan(np.linalg.norm(gradient))))
        if slope>35 or p80>.025:
            self.status={"source":"unavailable","reason":"stance_plane_inconsistent",
                         "slope_deg":slope,"residual_p80_m":p80}
            return None
        local=rotation.T @ gradient
        intercept=float(np.median(soles[:,2]-soles[:,:2] @ local))
        self.plane=np.r_[local,intercept]
        self.status={"source":"joint_imu_stance_plane","slope_deg":slope,
                     "residual_p80_m":p80,"support_span_s":now-self.pairs[0][0]}
        return self.plane
