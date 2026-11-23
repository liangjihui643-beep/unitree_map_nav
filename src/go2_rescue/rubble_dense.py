"""Metric visual stereo from calibrated RGB and a moving camera."""
import cv2
import numpy as np


def sweep_surface(k, old, view):
    """Sample texture-supported depths, keeping the semantic label in both views."""
    scale = .5
    height, width = view.gray.shape
    size = (width // 2, height // 2)
    k = np.diag([scale, scale, 1.]) @ k
    k_inv = np.linalg.inv(k)
    r_new = np.linalg.solve(k, scale_camera(view.camera)[:, :3])
    r_old = np.linalg.solve(k, scale_camera(old.camera)[:, :3])
    r = r_old @ r_new.T
    t = r_old @ (view.center - old.center)
    # Avoid matching past a large viewpoint/occlusion change.
    if np.linalg.norm(t) < .18 or np.trace(r) < 2.72:
        return np.empty((0, 3)), np.empty(0, dtype=np.uint8)
    source = cv2.resize(old.gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    target = cv2.resize(view.gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    source_labels = cv2.resize(old.labels, size, interpolation=cv2.INTER_NEAREST)
    target_labels = cv2.resize(view.labels, size, interpolation=cv2.INTER_NEAREST)
    # Local contrast is more stable across views than absolute brightness.
    source -= cv2.blur(source, (9, 9))
    target -= cv2.blur(target, (9, 9))
    texture = cv2.blur(target ** 2, (5, 5)) > 12.
    rows, cols = np.indices(target.shape)
    sample = texture & (target_labels > 0) & (rows % 5 == 2) & (cols % 5 == 2)
    y, x = np.where(sample)
    if not len(x):
        return np.empty((0, 3)), np.empty(0, dtype=np.uint8)
    labels = target_labels[y, x]
    pixels = np.column_stack((x, y, np.ones(len(x))))
    unit_rays = pixels @ k_inv.T
    rotation_h = k @ r @ k_inv
    translation_h = np.outer(k @ t, k_inv[2])
    depths = 1. / np.linspace(1./4., 1./.40, 48)
    costs = np.full((len(depths), len(x)), np.inf, dtype=np.float32)
    for i, distance in enumerate(depths):
        h = rotation_h + translation_h / distance
        # H maps target pixels to source pixels.
        warped = cv2.warpPerspective(source, h, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        warped_labels = cv2.warpPerspective(source_labels, h, size,
                                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        agreeing = warped_labels == target_labels
        mask_agreement = cv2.boxFilter(agreeing.astype(np.float32), -1, (5, 5))[y, x] >= .84
        error = cv2.boxFilter(np.abs(target - warped), -1, (5, 5))[y, x]
        costs[i, mask_agreement] = error[mask_agreement]
    columns = np.arange(len(x))
    best = costs.argmin(axis=0)
    error = costs[best, columns]
    separated = np.abs(np.arange(len(depths))[:, None]-best[None, :]) >= 3
    second = np.where(separated, costs, np.inf).min(axis=0)
    # Reject textureless, ambiguous and end-of-sweep matches.
    good = ((best > 0) & (best + 1 < len(depths)) & (error < 5.5)
            & (second > error * 1.28 + .8) & np.isfinite(second))
    best, x, y, unit_rays, labels, columns = (v[good] for v in (best, x, y, unit_rays, labels, columns))
    a, b, c = (costs[best+i, columns] for i in (-1, 0, 1))
    valid_fit = np.isfinite(a) & np.isfinite(c)
    offset = np.zeros(len(best))
    offset[valid_fit] = np.clip(.5*(a[valid_fit]-c[valid_fit]) /
                                np.maximum(a[valid_fit]-2*b[valid_fit]+c[valid_fit], .001), -.5, .5)
    inverse_depth = 1/depths[best] + offset * (1/depths[1] - 1/depths[0])
    points = (unit_rays/inverse_depth[:, None]) @ r_new + view.center
    ray_old, ray_new = points-old.center, points-view.center
    cosine = np.sum(ray_old*ray_new, axis=1) / (np.linalg.norm(ray_old, axis=1)*np.linalg.norm(ray_new, axis=1))
    good = cosine < np.cos(np.deg2rad(np.where(labels == 2, 1.5, 3.)))
    return points[good], labels[good]


def scale_camera(camera):
    scaled = camera.copy()
    scaled[:2] *= .5
    return scaled
