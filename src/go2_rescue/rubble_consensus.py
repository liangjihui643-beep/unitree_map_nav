"""Causal agreement between optical-flow and plane-sweep cloud observations."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree


class RubbleConsensus:
    def __init__(self):
        self.history = deque(maxlen=12)

    def update(self, xyz, dense, stamp, camera):
        """Return a mask of supported dense points; keep other points as proposals."""
        result = np.zeros(len(xyz),dtype=bool)
        sparse = xyz[~dense,:3]
        points = xyz[dense,:3]
        if not len(sparse) or not len(points):
            return result
        agreement = cKDTree(sparse).query(points,distance_upper_bound=.06)[0] != np.inf
        candidates = points[agreement]
        if not len(candidates):
            return result
        independent = np.zeros(len(candidates),dtype=int)
        right = candidates-camera
        for t,origin,tree in self.history:
            if not (0 < stamp-t <= 5 and np.linalg.norm(origin-camera) >= .12):
                continue
            left = candidates-origin
            angle = np.sum(left*right,axis=1)/(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1))
            close = tree.query(candidates,distance_upper_bound=.05)[0] != np.inf
            independent += (angle < np.cos(np.deg2rad(2.5))) & close
        if not self.history or np.linalg.norm(self.history[-1][1]-camera) >= .08:
            self.history.append((stamp,camera.copy(),cKDTree(candidates)))
        index = np.flatnonzero(dense)[agreement]
        result[index] = independent >= 1
        return result
