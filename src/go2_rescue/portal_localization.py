"""Point-to-surface recovery for geometry that is ambiguous in an XY wall map."""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot2
from .types import Pose


class SurfaceReference:
    def __init__(self, points, normals):
        self.points = points
        self.normals = normals
        self.tree = cKDTree(points)

    def fit(self, scan, guess, body_height):
        if len(scan) < 12:
            return None
        source = np.column_stack((scan[:, :2]@rot2(guess.yaw).T+guess.xy(), scan[:, 2]+body_height))
        shift = np.zeros(3)
        for _ in range(8):
            current = source+shift
            distance, index = self.tree.query(current)
            normal = self.normals[index]
            residual = np.einsum("ij,ij->i", current-self.points[index],normal)
            keep = (distance < .15) & (abs(residual) < .09)
            if keep.sum() < 12:
                return None
            a, error = normal[keep], residual[keep]
            weight = np.minimum(1., .025/np.maximum(abs(error),1e-8))
            gram = a.T@(a*weight[:,None])
            delta = -np.linalg.solve(gram+np.diag([3.,3.,15.]),a.T@(weight*error)+shift*[3.,3.,15.])
            shift += delta
            if np.linalg.norm(delta) < .0004:
                break
        # Measure XY observability after marginalizing the uncertain height.
        xy_information = gram[:2,:2]-np.outer(gram[:2,2],gram[2,:2])/(gram[2,2]+1e-8)
        rmse = float(np.sqrt(np.mean(error**2)))
        if (np.linalg.eigvalsh(xy_information)[0] < .7 or rmse > .045
                or np.linalg.norm(shift[:2]) > .09 or abs(shift[2]) > .06):
            return None
        return Pose(*(guess.xy()+shift[:2]),guess.yaw), int(keep.sum()), rmse
