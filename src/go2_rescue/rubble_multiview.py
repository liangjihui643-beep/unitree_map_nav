"""Joint inverse-depth estimation for a local group of calibrated RGB keyframes."""

import cv2
import numpy as np


def select_sources(view, history):
    candidates = []
    for source in history:
        distance = np.linalg.norm(source.center - view.center)
        same_direction = np.trace(view.from_level.T @ source.from_level) > 2.71
        if 0 < view.stamp - source.stamp <= 7 and .15 <= distance <= .85 and same_direction:
            candidates.append((distance, source))
    selected = []
    for _, source in sorted(candidates, key=lambda entry: -entry[0]):
        if all(np.linalg.norm(source.center - prev.center) >= .10 for prev in selected):
            selected.append(source)
            if len(selected) == 4:
                break
    return selected


def _scaled_view(view, size):
    intensity = cv2.resize(view.gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    labels = cv2.resize(view.labels, size, interpolation=cv2.INTER_NEAREST_EXACT)
    return intensity, labels


def _mean(image):
    return cv2.boxFilter(image, -1, (5, 5))


def _view_costs(source, rotation, view, camera, inverse_camera, size, target, target_mean,
                target_std, labels, samples, rays_world, inverse_depths):
    intensity, source_labels = _scaled_view(source, size)
    source_rotation = np.linalg.solve(camera, _rescale(size, source.gray.shape) @ source.camera[:, :3])
    delta = view.center - source.center
    rotation_h = camera @ source_rotation @ rotation.T @ inverse_camera
    translation_h = np.outer(camera @ source_rotation @ delta, inverse_camera[2])
    costs = np.full((len(inverse_depths), len(samples[0])), np.inf, np.float32)
    ray_norm = np.linalg.norm(rays_world, axis=1)
    ray_dot = rays_world @ delta
    delta_squared = delta @ delta
    ray_squared = ray_norm**2
    sample_labels = labels[samples]
    forward = rays_world @ source_rotation[2]
    forward_delta = delta @ source_rotation[2]
    # Each score needs only the 5x5 neighborhood of its sample. Keep the
    # source uncropped so the projected source pixels stay valid.
    y, x = samples
    top, left = max(0, int(y.min())-2), max(0, int(x.min())-2)
    bottom, right = min(size[1], int(y.max())+3), min(size[0], int(x.max())+3)
    crop_size = (right-left, bottom-top)
    crop_to_image = np.array([[1.,0.,left],[0.,1.,top],[0.,0.,1.]])
    offsets = (y-top, x-left)
    target = target[top:bottom,left:right]
    labels = labels[top:bottom,left:right]
    for i, inv_depth in enumerate(inverse_depths):
        # Use translation-induced parallax per ray, accounting for forward motion.
        dot = ray_squared + inv_depth * ray_dot
        norm = np.sqrt(ray_squared + 2 * inv_depth * ray_dot + inv_depth**2 * delta_squared)
        angle_ok = dot < np.cos(np.deg2rad(1.5)) * ray_norm * norm
        angle_ok &= forward + inv_depth * forward_delta > .1 * inv_depth
        if not angle_ok.any():
            continue
        homography = (rotation_h + translation_h * inv_depth) @ crop_to_image
        warped_labels = cv2.warpPerspective(source_labels, homography, crop_size,
                                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        semantic = (_mean((warped_labels == labels).astype(np.float32))[offsets] >= .76)
        valid = angle_ok & semantic & (warped_labels[offsets] == sample_labels)
        if not valid.any():
            continue
        warped = cv2.warpPerspective(intensity, homography, crop_size,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        source_mean = _mean(warped)[offsets]
        std = np.sqrt(np.maximum(_mean(warped**2)[offsets] - source_mean**2, 0))
        covariance = _mean(target * warped)[offsets] - target_mean * source_mean
        correlation = covariance / np.maximum(target_std * std, 1e-3)
        valid &= std >= 2.
        costs[i, valid] = np.clip(1. - correlation[valid], 0, 2)
    return costs


def _rescale(size, original_shape):
    height, width = original_shape
    sx, sy = size[0] / width, size[1] / height
    # Resizing preserves the centre of each pixel area.
    return np.array([[sx, 0., (sx-1)*.5], [0., sy, (sy-1)*.5], [0., 0., 1.]])


def sweep_multiview(intrinsics, sources, view, *, inverse_depths=None, target_pixels=None):
    """Estimate XYZ from one shared depth hypothesis per pixel and at least two sources."""
    empty = np.empty((0, 3)), np.empty(0, dtype=np.uint8)
    if len(sources) < 2:
        return empty
    height, width = view.gray.shape
    size = (width // 2, height // 2)
    scale = _rescale(size, view.gray.shape)
    camera = scale @ intrinsics
    inverse_camera = np.linalg.inv(camera)
    rotation = np.linalg.solve(camera, scale @ view.camera[:, :3])
    target, labels = _scaled_view(view, size)
    mean = _mean(target)
    std = np.sqrt(np.maximum(_mean(target**2) - mean**2, 0))
    sample_grid = (slice(2, None, 3), slice(2, None, 3))
    search = labels > 0
    if target_pixels is not None:
        needed = np.zeros(labels.shape,np.uint8)
        x, y = target_pixels.T
        needed[y,x] = 1
        search &= cv2.dilate(needed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) != 0
    y, x = np.where(search[sample_grid] & (std[sample_grid] >= 2.))
    samples = (y * 3 + 2, x * 3 + 2)
    if not len(y):
        return empty
    rays = np.column_stack((samples[1], samples[0], np.ones(len(y)))) @ inverse_camera.T
    rays_world = rays @ rotation
    if inverse_depths is None:
        inverse_depths = np.linspace(1/4., 1/.4, 60)
    view_costs = np.stack([
        _view_costs(source, rotation, view, camera, inverse_camera, size, target, mean[samples],
                    std[samples], labels, samples, rays_world, inverse_depths)
        for source in sources
    ])
    # The two best agreeing sources tolerate partial occlusion of the others.
    # The second-best match must itself have good photometric support.
    best_sources = np.partition(view_costs, 1, axis=0)[:2]
    costs = (.35 * best_sources[0] + .65 * best_sources[1])
    costs[best_sources[1] > .27] = np.inf
    column = np.arange(len(y))
    best = costs.argmin(axis=0)
    error = costs[best, column]
    separated = abs(np.arange(len(inverse_depths))[:, None] - best) >= 3
    second = np.where(separated, costs, np.inf).min(axis=0)
    good = (best > 0) & (best + 1 < len(inverse_depths)) & (error < .2) & (second > error + .045)
    best, column = best[good], column[good]
    left, center, right = (costs[best + step, column] for step in (-1, 0, 1))
    finite = np.isfinite(left) & np.isfinite(right)
    offset = np.zeros(len(best))
    offset[finite] = .5 * (left[finite] - right[finite]) / np.maximum(
        left[finite] - 2*center[finite] + right[finite], .001)
    inverse_depth = inverse_depths[best] + np.clip(offset, -.5, .5) * (inverse_depths[1] - inverse_depths[0])
    points = rays_world[good] / inverse_depth[:, None] + view.center
    return points, labels[samples][good]
