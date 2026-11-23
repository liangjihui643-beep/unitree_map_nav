"""Agent-visible LiDAR encoding and conservative evidence from range only."""
import numpy as np

from .geometry import bresenham, gravity_rotation, transform_xy

N_AZIMUTH = 72


def encode_lidar_obs(valid, ranges, maximum=12.0):
    valid = np.asarray(valid, dtype=bool)
    ranges = np.asarray(ranges, dtype=float)
    if (valid.shape != (N_AZIMUTH,) or ranges.shape != (valid.sum(),)
            or not np.isfinite(ranges).all() or np.any((ranges < 0) | (ranges > maximum))):
        raise ValueError("invalid LiDAR mask/ranges")
    obs = np.ones(2 * N_AZIMUTH, dtype=np.float32)
    obs[:N_AZIMUTH][valid] = ranges / maximum
    obs[N_AZIMUTH:] = valid
    return obs


def decode_lidar_obs(obs, minimum=.05, maximum=12.0):
    obs = np.asarray(obs, dtype=float)
    if obs.shape != (2 * N_AZIMUTH,) or not np.isfinite(obs).all():
        raise ValueError("expected one finite 144-value LiDAR observation")
    norm, mask = np.split(obs, 2)
    if (np.any((mask != 0) & (mask != 1)) or np.any(norm[mask == 0] != 1)
            or np.any(norm[mask == 1] < (minimum-1e-7)/maximum)
            or np.any(norm[mask == 1] > 1)):
        raise ValueError("invalid LiDAR distance/mask")
    ids = np.flatnonzero(mask)
    return ids, norm[ids] * maximum


class RangeOnlyEvidence:
    """Planar envelopes over possible hit elevations; no ground-free assertion."""

    def __init__(self, grid, elevation_center, vertical_fov, *, height_floor_m=None,
                 origin_body=None):
        self.ray_prefix = np.zeros_like(grid.seen)
        self.ground_band_clear = np.zeros_like(grid.seen)
        self.possible_hit = np.zeros_like(grid.seen)
        self.unambiguous_hit = np.zeros_like(grid.seen)
        self.elevations = np.deg2rad(np.linspace(elevation_center-vertical_fov/2,
                                                    elevation_center+vertical_fov/2, 9))
        self.height_floor_m = height_floor_m
        self.origin_body = (np.array([0., 0., .10]) if origin_body is None
                            else np.asarray(origin_body, float).reshape(3))

    def update(self, grid, obs, quaternion, pose, minimum, maximum, *, base_z=None, ground=None):
        ids, ranges = decode_lidar_obs(obs, minimum, maximum)
        az = -np.pi + ids * (2*np.pi / N_AZIMUTH)
        ce, se = np.cos(self.elevations), np.sin(self.elevations)
        directions = np.stack((np.cos(az)[:, None]*ce, np.sin(az)[:, None]*ce,
                              np.broadcast_to(se, (len(ids), len(se)))), axis=-1)
        level = gravity_rotation(quaternion)
        origin = self.origin_body @ level.T
        start = grid.cell(transform_xy(origin[None], pose)[0])
        ends = origin + (directions @ level.T)*ranges[:, None, None]
        ends = transform_xy(ends.reshape(-1, 3), pose).reshape(len(ids), len(se), 2)
        vertical = (directions @ level.T)[:, :, 2]*ranges[:, None]
        start_xy = transform_xy(origin[None], pose)[0]
        for alternatives, dz in zip(ends, vertical):
            if self.height_floor_m is not None:
                # Account for small attitude/range errors at the clipping
                # boundary rather than treating quantized equality as a miss.
                feasible = origin[2] + dz >= self.height_floor_m - .025
                alternatives, dz = alternatives[feasible], dz[feasible]
            if not len(alternatives):
                continue
            ends_in_cells = [grid.cell(end) for end in alternatives]
            if len(set(ends_in_cells)) == 1 and grid.inside(ends_in_cells[0]):
                # All viable elevations land in this same planar cell.
                # This is an endpoint, not a vote about ground obstacle height.
                self.unambiguous_hit[ends_in_cells[0]] = True
            common = None
            for end in ends_in_cells:
                if grid.inside(end):
                    self.possible_hit[end] = True
                prefix = set(bresenham(start, end)) - {end}
                common = prefix if common is None else common & prefix
            for cell in common:
                if not grid.inside(cell):
                    continue
                self.ray_prefix[cell] = True
                if ground is None or base_z is None or not np.isfinite(base_z):
                    continue
                z = ground[cell]
                if not np.isfinite(z):
                    continue
                lower = grid.origin + np.asarray(cell)*grid.c.resolution
                crossing = cell_crossing_fractions(start_xy, alternatives-start_xy,
                                                  lower, lower+grid.c.resolution)
                if crossing is None:
                    continue
                entry, exit = crossing
                relative = base_z + origin[2] - z + np.stack((entry*dz, exit*dz))
                # Every feasible beam must actually intersect the cell and
                # stay within the near-ground band for that crossing.
                if (relative >= .03).all() and (relative <= grid.c.min_obstacle_height).all():
                    self.ground_band_clear[cell] = True


def cell_crossing_fractions(origin, displacements, low, high):
    """Intersect candidate finite segments with an axis-aligned raster cell."""
    parallel = np.abs(displacements) < 1e-10
    if np.any(parallel & ((origin < low) | (origin >= high))):
        return None
    denom = np.where(parallel, 1., displacements)
    first, second = (low-origin)/denom, (high-origin)/denom
    entry = np.maximum(0., np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1))
    exit = np.minimum(1., np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1))
    return (entry, exit) if np.all(exit-entry > 1e-8) else None
