"""Single-frame RGB rubble proposals with optional range-only metric anchors."""
import cv2
import numpy as np

from .geometry import transform_xy
from .lidar_obs import N_AZIMUTH, decode_lidar_obs, encode_lidar_obs
from .rubble_depth_runtime import RGBRubbleDepth, SIZE


RANGE_EDGES_M = np.array([1.25, 2.40, 4.0])
LIDAR_ORIGIN = np.array([0., 0., .10])


def recorded_lidar_obs(points, config):
    """Recover only azimuth and range from older recordings of the selected returns."""
    offsets = points - LIDAR_ORIGIN
    step = 2 * np.pi / N_AZIMUTH
    ids = np.rint((np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi) / step).astype(int) % N_AZIMUTH
    ranges = np.clip(np.linalg.norm(offsets, axis=1), config.min_range, config.max_range)
    valid = np.zeros(N_AZIMUTH, dtype=bool)
    valid[ids] = True
    return encode_lidar_obs(valid, ranges[np.argsort(ids)], config.max_range)


class RubbleRangeAnchors:
    def __init__(self, config, intrinsics):
        self.config = config
        self.k = intrinsics
        center = config.lidar_elevation_center_deg
        half = config.lidar_vertical_fov_deg / 2
        self.elevation = np.deg2rad(np.linspace(center-half, center+half, 9))

    def associate(self, obs, level, rotation, origin, labels):
        ids, ranges = decode_lidar_obs(obs, self.config.min_range, self.config.max_range)
        az = (ids * (2*np.pi/N_AZIMUTH) - np.pi)[:, None]
        az, el = np.broadcast_arrays(az, self.elevation)
        directions = np.stack((np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)), axis=-1)
        endpoints = LIDAR_ORIGIN @ level.T + (directions @ level.T) * ranges[:, None, None]
        optical = (endpoints-origin) @ rotation
        depth = optical[..., 2]
        projected = optical @ self.k.T
        uv = np.divide(projected[..., :2], depth[..., None], out=np.zeros_like(projected[..., :2]),
                       where=depth[..., None] > .1)
        pixels = np.rint(uv).astype(int)
        h, w = labels.shape
        valid = ((depth > .35) & (depth <= 4.) & (endpoints[..., 2] >= self.config.lidar_height_floor-.025)
                 & ((pixels >= 0) & (pixels < [w,h])).all(axis=2))
        label = np.zeros_like(depth, dtype=int)
        u, v = pixels[valid].T
        label[valid] = labels[v, u]
        anchors = []
        for i, beam in enumerate(ids):
            hits = label[i] > 0
            components = np.unique(label[i, hits])
            if len(components) != 1:
                continue
            anchor_depth = depth[i, hits]
            if np.ptp(anchor_depth) > .23:
                continue
            anchors.append((beam, components[0], uv[i, hits], anchor_depth, int(valid[i].sum())))
        corroborated = []
        for beam, component, pixels, depth, visible in anchors:
            nearby = any(other != beam and abs(int(other)-int(beam)) <= 2 and tag == component
                         and abs(np.median(z)-np.median(depth)) <= .25
                         for other, tag, _, z, _ in anchors)
            if nearby or (len(depth) >= 2 and len(depth) * 2 >= visible):
                corroborated.append((component, pixels, depth))
        return corroborated

    @staticmethod
    def depth_at(samples, components, anchors):
        measured = np.full(len(samples), np.nan)
        for component in set(components):
            matched = [(uv, depth) for label, uv, depth in anchors if label == component]
            if component == 0 or not matched:
                continue
            pixels = np.vstack([item[0] for item in matched])
            depth = np.concatenate([item[1] for item in matched])
            indices = np.flatnonzero(components == component)
            gaps = np.linalg.norm(samples[indices, None, :]-pixels[None, :, :], axis=-1)
            for idx, distance in zip(indices, gaps):
                close = distance <= 12.
                if close.any() and np.ptp(depth[close]) <= .24:
                    measured[idx] = np.average(depth[close], weights=1/(2.+distance[close]))
        return measured


class SingleViewRubble:
    def __init__(self, model, intrinsics, sensor_size):
        expected = [[207.8461, 0., 160.], [0., 207.8461, 120.], [0., 0., 1.]]
        if tuple(sensor_size) != (320, 240) or not np.allclose(intrinsics, expected, atol=.05, rtol=0):
            raise ValueError("Single-view rubble model expects the Go2 320x240 camera calibration")
        self.depth = RGBRubbleDepth(model, intrinsics, sensor_size, threshold=.35)
        self.completion = None
        if self.depth.size != (64, 64):
            raise ValueError("Single-view rubble model requires 64x64 RGB")

    def estimate(self, image, rotation, origin, plane):
        camera_height = origin[2] - plane[2]
        depth, _, _ = self.depth.infer(image, rotation[2], camera_height)
        rays = self.depth.rays @ rotation.T
        points = origin + rays * depth[..., None]
        distance = np.linalg.norm((points-origin)[..., :2], axis=-1)
        classes = np.digitize(distance, RANGE_EDGES_M).astype(np.uint8)
        usable = np.isfinite(depth) & (depth >= .35) & (depth <= 4.) & (distance <= RANGE_EDGES_M[-1])
        classes[~usable] = 3
        return points, classes, usable

    def label(self, mask, estimate, pose, rotation, origin, plane, *, ranging=None, context=None):
        """Use one common mask for the range-layer proposal and ground baseline."""
        points, layers, usable = estimate
        small_mask = cv2.resize(mask.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST_EXACT)
        selected = np.zeros(small_mask.shape, dtype=bool)
        regions = []
        for layer in range(3):
            foreground = (small_mask != 0) & (layers == layer) & usable
            _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
            for idx in np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 10):
                if idx == 0:
                    continue
                component = labels == idx
                selected |= component
                xyz = points[component]
                xy = transform_xy(xyz[:, :2], pose)
                ranges = np.linalg.norm((xyz-origin)[:, :2], axis=1)
                center = np.median(xy, axis=0)
                lo, hi = np.quantile(ranges, [.1, .9])
                regions.append({
                    "range_band": ("near", "middle", "far")[layer],
                    "map_xy_center_m": center.tolist(),
                    "horizontal_range_estimate_m": float(np.median(ranges)),
                    "horizontal_range_hint_m": [float(max(.35, lo-.40)), float(hi+.65)],
                    "surface_z_relative_to_body_m": float(np.median(xyz[:, 2])),
                    "height_above_local_ground_m": None,
                    "mask_samples": len(xyz),
                })
        # Report a bounded number of observations without filling intervening cells.
        sample = selected[1::3, 1::3]
        y, x = np.nonzero(sample)
        y, x = y*3+1, x*3+1
        local = points[y, x, :2]
        new_xy = transform_xy(local, pose)
        sample_pixels = np.column_stack((x, y)).tolist()

        # Replicate the legacy bottom-of-mask ground intersection for each component.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(small_mask)
        old = []
        for idx in range(1, len(stats)):
            x0,y0,w,h,area = stats[idx]
            if area < 7 or x0 == 0 or x0+w >= SIZE[0] or y0+h >= SIZE[1]:
                continue
            x = np.arange(x0, x0+w, 2)
            bottom = (labels[::-1, x] == idx).argmax(axis=0)
            has_pixel = (labels[:, x] == idx).any(axis=0)
            y = SIZE[1]-1-bottom
            rays = self.depth.rays[y,x] @ rotation.T
            divisor = rays[:,2] - rays[:,:2] @ plane[:2]
            scale = np.divide(plane[2]+origin[:2]@plane[:2]-origin[2], divisor,
                              out=np.zeros(len(x)), where=divisor < -.03)
            valid = has_pixel & (scale > .25) & (scale < 4.)
            xy = transform_xy(origin[:2]+rays[valid,:2]*scale[valid,None], pose)
            for u,v,point in zip(x[valid],y[valid],xy):
                new = transform_xy(points[v,u,:2][None],pose)[0] if usable[v,u] else None
                old.append({"uv": [int(u),int(v)], "legacy_xy_m": point.tolist(),
                            "range_layer_xy_m": new.tolist() if new is not None else None})
        comparison = {
            "candidate_xy_m": new_xy.tolist(), "sample_uv": sample_pixels,
            "regions": regions, "baseline_samples": old,
            "mask_fraction_with_depth": float(selected.sum()/max(1, int(small_mask.sum()))),
            "status": "single_frame_candidate",
        }
        if ranging is None:
            return local, comparison
        associator, obs, level = ranging
        _, full_labels = cv2.connectedComponents(mask.astype(np.uint8))
        anchors = associator.associate(obs, level, rotation, origin, full_labels)
        y, x = np.nonzero(small_mask[1::3, 1::3])
        anchor_pixels = np.column_stack((x*3+1, y*3+1))
        matched_uv = np.vstack((anchor_pixels, np.asarray([item["uv"] for item in old], int).reshape(-1,2)))
        u, v = matched_uv.T
        # Use the model's OUTPUT grid (160x120), not its RGB input grid (64x64).
        screen = (matched_uv+.5) * np.array([mask.shape[1]/SIZE[0], mask.shape[0]/SIZE[1]]) - .5
        uv_rounded = np.rint(screen).astype(int)
        components = full_labels[uv_rounded[:, 1], uv_rounded[:, 0]]
        measured = associator.depth_at(screen, components, anchors)
        measured_xy = origin[:2] + (self.depth.rays[v,u] @ rotation.T)[:, :2] * measured[:, None]
        dense = measured_xy[:len(anchor_pixels)]
        valid = np.isfinite(dense[:, 0])
        for item, xy in zip(old, measured_xy[len(anchor_pixels):]):
            item["anchored_xy_m"] = transform_xy(xy[None], pose)[0].tolist() if np.isfinite(xy[0]) else None
        comparison.update(
            anchored_xy_m=transform_xy(dense[valid], pose).tolist(),
            anchored_uv=anchor_pixels[valid].tolist(),
            corroborated_beams=len(anchors),
            unresolved_regions=max(0, int(full_labels.max())-len({a[0] for a in anchors})),
        )
        if (self.completion is not None and context is not None
                and (len(anchor_pixels) or self.completion.revision is not None)):
            image, stamp, base_z, floor_mask = context
            rays = self.depth.rays @ rotation.T
            prior = (points-origin) @ rotation[:, 2]
            xy, relief, source, completed = self.completion.complete(
                small_mask.astype(bool),image,prior,rays,rotation,origin,plane,pose,
                stamp,base_z,floor_mask,anchor_pixels,measured[:len(anchor_pixels)])
            comparison.update(completed)
            return xy, comparison
        return dense[valid], comparison
