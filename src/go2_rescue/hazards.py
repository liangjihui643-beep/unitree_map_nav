"""RGB/lidar ground hazards. Colour cue is configurable, not a water oracle."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import gravity_rotation, transform_xy
from scipy.spatial import cKDTree


class WaterDetector:
    """Detect the competition fixture's cyan puddles with geometric checks.

    A blue pixel alone is insufficient: multiple lidar hits must project into
    the region and form a near-ground, low-slope plane. No depth/geom IDs are
    read. Real deployments must calibrate the colour cue for their environment.
    """
    def __init__(self, config):
        self.c = config
        self.k = np.asarray(config.camera_matrix, float)
        self.camera_from_body = np.linalg.inv(config.body_from_camera)
        self.body_from_camera = np.asarray(config.body_from_camera, float)

    def colour_support_candidate(self,labels,region,frame,pose,grid):
        """Conservative keep-out cue; not a confirmed water/material label."""
        plane=getattr(grid,"support_plane",None)
        if not self.c.navigation_support or plane is None:
            return None
        pixels=np.argwhere(labels==region)[::5]
        if len(pixels)<3:return None
        rays=np.column_stack([(pixels[:,1]-self.k[0,2])/self.k[0,0],
                              (pixels[:,0]-self.k[1,2])/self.k[1,1],np.ones(len(pixels))])
        level=gravity_rotation(frame.quaternion)
        origin=level@self.body_from_camera[:3,3]
        direction=rays@self.body_from_camera[:3,:3].T@level.T
        denominator=direction[:,2]-direction[:,:2]@plane[:2]
        scale=np.divide(plane[2]+origin[:2]@plane[:2]-origin[2],denominator,
            out=np.full(len(rays),np.inf),where=denominator<-.05)
        valid=(scale>.2)&(scale<2.5)
        if valid.sum()<3:return None
        points=origin+scale[valid,None]*direction[valid]
        xy=transform_xy(points,pose)
        # Position/plane uncertainty adds clearance, never clears free space.
        expanded=np.vstack([xy+offset for offset in [[.10,0],[-.10,0],[0,.10],[0,-.10]]])
        polygon=cv2.convexHull(expanded.astype(np.float32))[:,0]
        return polygon if .004<cv2.contourArea(polygon)<1.5 else None

    def detect(self, image, frame, pose, grid):
        if not self.c.water_detection:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.c.water_hsv_lower), np.array(self.c.water_hsv_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        camera = frame.points @ self.camera_from_body[:3, :3].T + self.camera_from_body[:3, 3]
        uvw = camera @ self.k.T
        pixel = np.rint(uvw[:, :2] / np.maximum(uvw[:, 2:], 1e-6)).astype(int)
        valid = (camera[:, 2] > .1) & (pixel[:, 0] >= 0) & (pixel[:, 0] < image.shape[1]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < image.shape[0])
        ids = np.flatnonzero(valid)
        level = gravity_rotation(frame.quaternion)
        points = frame.points @ level.T
        world_xy = transform_xy(points, pose)
        ray_origin = level @ self.body_from_camera[:3, 3]
        polygons = []
        self.last_sources=[]
        for region in range(1, count):
            if stats[region, cv2.CC_STAT_AREA] < 15:
                continue
            hits = ids[labels[pixel[ids, 1], pixel[ids, 0]] == region]
            if len(hits) < 2:
                candidate=self.colour_support_candidate(labels,region,frame,pose,grid)
                if candidate is not None:
                    polygons.append(candidate);self.last_sources.append("rgb_stance_keepout_candidate")
                continue
            # Reject coloured walls/signs: hits must agree with measured floor.
            floor = np.array([grid.floor[grid.cell(p)] if grid.inside(grid.cell(p)) else np.nan for p in world_xy[hits]])
            height = points[hits, 2] + self.c.base_height + grid.ground_height
            hits = hits[np.isfinite(floor) & (abs(height - floor) < .06)]
            if len(hits) < 2:
                continue
            # Nine lidar elevations often leave only two returns on a small
            # puddle. Fit neighbouring confirmed floor as well, rather than
            # inventing depth from the image or fitting a rank-two plane.
            nearby = np.linalg.norm(points[:, :2] - points[hits, :2].mean(axis=0), axis=1) < .45
            neighbours = np.flatnonzero(nearby)
            floor = np.array([grid.floor[grid.cell(p)] if grid.inside(grid.cell(p)) else np.nan for p in world_xy[neighbours]])
            height = points[neighbours, 2] + self.c.base_height + grid.ground_height
            neighbours = neighbours[np.isfinite(floor) & (abs(height-floor) < .05)]
            if len(neighbours) < 6:
                continue
            design = np.column_stack([points[neighbours, :2], np.ones(len(neighbours))])
            plane, _, rank, _ = np.linalg.lstsq(design, points[neighbours, 2], rcond=None)
            if rank < 3 or np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(self.c.max_slope_deg + 4)) or np.quantile(abs(design @ plane - points[neighbours, 2]), .9) > .035:
                continue
            contours, _ = cv2.findContours((labels == region).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)[:, 0]
            rays = np.linalg.solve(self.k, np.column_stack([contour, np.ones(len(contour))]).T).T
            rays = rays @ (level @ self.body_from_camera[:3, :3]).T
            normal = np.r_[-plane[:2], 1.]
            denominator = rays @ normal
            scale = (plane[2] - ray_origin @ normal) / np.where(abs(denominator) > 1e-6, denominator, np.nan)
            xyz = ray_origin + scale[:, None] * rays
            good = np.isfinite(xyz).all(axis=1) & (scale > 0) & (np.linalg.norm(xyz[:, :2], axis=1) < min(4., self.c.max_range))
            if good.sum() < 3:
                continue
            projected = transform_xy(xyz[good], pose)
            measured = world_xy[hits]
            # A local plane is not valid metres beyond its supporting returns.
            # Clip the image projection to measured water neighbourhoods; far
            # portions become confirmed as new scans arrive, rather than a
            # near-horizon extrapolation painting the whole passage blue.
            distance, _ = cKDTree(measured).query(projected)
            projected = projected[distance < .40]
            support = np.vstack([measured + d for d in [[.03, 0.], [-.03, 0.], [0., .03], [0., -.03]]])
            polygon = cv2.convexHull(np.vstack([projected, support]).astype(np.float32))[:, 0]
            area = cv2.contourArea(polygon)
            if .004 < area < 1.5:
                polygons.append(polygon)
                self.last_sources.append("lidar_ground_colour")
        return polygons
