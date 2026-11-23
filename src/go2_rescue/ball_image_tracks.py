"""Time-local 2-D circle association with delayed metric confirmation."""
from collections import Counter, deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import rot2


class BallImageTracks:
    def __init__(self, k, *, pixel_gate=2.5, known_ball_radius=None, compensate_rotation=False):
        self.k = np.asarray(k)
        self.pixel_gate = pixel_gate
        self.known_ball_radius = known_ball_radius
        self.compensate_rotation = compensate_rotation
        self.clear()

    def clear(self):
        self.tracks = []
        self.failures = Counter()

    def update(self, detections, pose, base_z, rotation, origin, stamp, confidence):
        if not np.isfinite(base_z):
            return
        world = np.eye(3)
        world[:2, :2] = rot2(pose.yaw)
        rot = world @ rotation
        camera = np.r_[pose.xy(), base_z] + world @ origin
        self.tracks = [track for track in self.tracks if 0 < stamp-track[-1][0] <= .7]
        items = [d for d in detections if d.confidence >= confidence]
        samples = []
        for d in items:
            uv = np.asarray(d.center)
            ray = rot @ np.linalg.solve(self.k, [*uv, 1.])
            samples.append((stamp, uv, float(d.radius), camera.copy(), ray/np.linalg.norm(ray), rot))
        assigned = set()
        if self.tracks and samples:
            cost = np.full((len(self.tracks),len(samples)), 1e6)
            for a, track in enumerate(self.tracks):
                then, last, radius, *_ = track[-1]
                pred = last
                if len(track)>1:
                    previous = track[-2]
                    pred = last + np.clip((last-previous[1])*(stamp-then)/(then-previous[0]), -12., 12.)
                if self.compensate_rotation:
                    # Predict in the current camera. A heading change is not
                    # target motion and should not break a 2-D association.
                    direction = track[-1][4]
                    if len(track) > 1:
                        change = (direction-track[-2][4])*(stamp-then)/(then-track[-2][0])
                        if np.linalg.norm(change) <= .055:
                            direction = direction+change
                    pixel = self.k @ (direction @ rot)
                    if pixel[2] <= .05:
                        continue
                    pred = pixel[:2] / pixel[2]
                for b, sample in enumerate(samples):
                    dist = np.linalg.norm(sample[1]-pred)
                    logscale = abs(np.log(max(sample[2],1.)/max(radius,1.)))
                    if dist <= max(8., .8*radius) and logscale <= .35:
                        cost[a,b] = dist + 10.*logscale
            for a,b in zip(*linear_sum_assignment(cost)):
                if cost[a,b]>=1e6:
                    continue
                assigned.add(b)
                self.tracks[a].append(samples[b])
                self._solve(self.tracks[a], items[b], pose)
                if items[b].position is None and self.known_ball_radius is not None:
                    self._solve_apparent_size(self.tracks[a], items[b], pose)
        for b, sample in enumerate(samples):
            if b not in assigned:
                self.tracks.append(deque([sample],maxlen=30))

    def _solve(self, trajectory, detection, pose):
        rows = [x for x in trajectory if trajectory[-1][0]-x[0] <= 4.]
        if len(rows) < 4:
            self.failures['short_track'] += 1
            return
        origins = np.array([x[3] for x in rows]); rays = np.array([x[4] for x in rows])
        projectors = np.eye(3) - rays[:,:,None]*rays[:,None,:]
        rhs = np.einsum('ijk,ik->ij',projectors,origins)
        def fit(indices):
            return np.linalg.lstsq(projectors[indices].reshape(-1,3),rhs[indices].ravel(),rcond=None)[0]
        def errors(point):
            camera = np.array([(point-x[3]) @ x[5] for x in rows])
            pixels = camera @ self.k.T
            err = np.linalg.norm(pixels[:,:2]/np.maximum(pixels[:,2:],.001)-np.array([x[1] for x in rows]),axis=1)
            return np.where(camera[:,2]>.25,err,np.inf)
        best = None
        eligible_pairs = 0
        for i in range(len(rows)-2):
            if np.linalg.norm(origins[i,:2]-origins[-1,:2]) < .16 or rays[i]@rays[-1]>np.cos(np.deg2rad(4)):
                continue
            eligible_pairs += 1
            point = fit([i,len(rows)-1])
            inliers = errors(point) <= self.pixel_gate
            if inliers.sum()>=4 and (best is None or inliers.sum()>best.sum()):
                best = inliers
        if best is None:
            self.failures["no_consensus" if eligible_pairs else "baseline_or_parallax"] += 1
            return
        point = fit(best)
        inliers = errors(point) <= self.pixel_gate
        if (not inliers[-1] or inliers.sum()<4 or inliers.sum()<.65*len(rows)
                or np.linalg.norm(np.ptp(origins[inliers,:2],axis=0))<.16
                or np.linalg.norm(point[:2]-origins[-1,:2])>4.):
            self.failures['unstable_fit'] += 1
            return
        detection.position = point[:2]
        detection.sigma = float(np.hypot(.11,pose.sigma))
        detection.range_source = 'visual_image_track'

    def _solve_apparent_size(self, trajectory, detection, pose):
        # A complete sphere of known size provides a depth prior without
        # assuming the ground under it. Check temporal consistency in XYZ
        # before emitting any observation based on that prior.
        rows = [row for row in trajectory if trajectory[-1][0]-row[0] <= 2.]
        if len(rows) < 4 or rows[-1][0]-rows[0][0] < .35:
            return
        points = []
        for _, uv, radius, origin, ray, rot in rows:
            if radius < 12.:
                points.append(None)
                continue
            edge = uv + radius * np.array([[1,0],[-1,0],[0,1],[0,-1]])
            directions = np.linalg.solve(self.k, np.column_stack((edge,np.ones(4))).T).T
            directions /= np.linalg.norm(directions,axis=1)[:,None]
            sines = np.linalg.norm(np.cross(directions,ray @ rot),axis=1)
            alpha = np.median(sines)
            if not .025 < alpha < .40:
                points.append(None)
                continue
            length = self.known_ball_radius / alpha
            points.append(origin + ray*length if .3 <= length <= 2.8 else None)
        last = points[-1]
        if last is None:
            return
        usable = np.asarray([p for p in points if p is not None])
        similar = np.linalg.norm(usable-last,axis=1) <= .13
        if similar.sum() < 4 or similar.sum() < .75*len(rows):
            return
        detection.position = np.median(usable[similar],axis=0)[:2]
        detection.sigma = float(np.hypot(.12,pose.sigma))
        detection.range_source = 'visual_sphere_scale_track'


