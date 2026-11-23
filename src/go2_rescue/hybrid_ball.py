"""One measurement per target proposal, with multi-view rescue for raised balls."""
from dataclasses import replace

import numpy as np

from .ball_image_tracks import BallImageTracks
from .ball_multiview import fuse_target_position
from .ball_outline import refine_ball_outline
from .geometry import gravity_rotation, rot2


def usable_ball(observation, minimum_confidence):
    return (observation.position is not None and np.isfinite(observation.position).all()
            and observation.confidence >= minimum_confidence and observation.sigma <= .20)


class HybridBallRange:
    def __init__(self, vision, *, allow_unconfirmed_flat=False, require_outline_for_flat=False, use_known_size=False, compensate_rotation=False):
        self.vision = vision
        self.allow_unconfirmed_flat = allow_unconfirmed_flat
        self.require_outline_for_flat = require_outline_for_flat
        self.outline = BallImageTracks(vision.k, known_ball_radius=vision.c.ball_radius if use_known_size else None,
                                       compensate_rotation=compensate_rotation)
        self.provisional = []

    def clear(self):
        self.outline.clear()
        self.provisional = []

    def flat_range_consistent(self, observation, pose, level):
        """Check size independently from the candidate's plane-derived distance."""
        vision = self.vision
        xy = rot2(pose.yaw).T @ (observation.position - pose.xy())
        center = np.r_[xy, vision.c.ball_radius - vision.c.base_height]
        optical = (center - level @ vision.t[:3, 3]) @ (level @ vision.t[:3, :3])
        expected = np.mean(np.diag(vision.k)[:2]) * vision.c.ball_radius
        ratio = observation.radius * optical[2] / expected
        return .78 <= ratio <= 1.22

    def update(self, detections, image, frame, pose, base_z, stance, confidence):
        """Keep a flat-floor hypothesis separate from independent triangulation.

        A slope or missing support disables the uncorroborated flat shortcut.
        The shortcut, when enabled, remains a local hypothesis: stance alone
        does not measure ground height under a distant ball.
        """
        self.provisional = []
        level = gravity_rotation(frame.quaternion)
        mount = self.vision.t
        pairs = []
        for detection in detections:
            if detection.confidence < confidence:
                continue
            refined = replace(detection, position=None, range_source="unknown")
            if refine_ball_outline(image, refined):
                pairs.append((detection, refined))
        self.outline.update([d for _, d in pairs], pose, base_z, level @ mount[:3, :3],
                            level @ mount[:3, 3], frame.timestamp, confidence)
        outlined = {id(original): d for original, d in pairs}
        stereo = {id(original): d for original, d in pairs if usable_ball(d, confidence)}
        # This estimate uses only the pixels + IMU, never hidden LiDAR XYZ.
        pixels_only = replace(frame, points=np.empty((0, 3)), point_times=None)
        flat = (stance is not None and np.isfinite(stance).all()
                and np.linalg.norm(stance[:2]) <= np.tan(np.deg2rad(7.))
                and abs(-stance[2]-self.vision.c.base_height) <= .07)
        choices = []
        for original in detections:
            original.position = None
            original.range_source = "unknown"
            if original.confidence < confidence:
                continue
            multi = stereo.get(id(original))
            probe = outlined.get(id(original)) if self.require_outline_for_flat else original
            single = self.vision.locate_ball(replace(probe), pixels_only, pose) if flat and probe is not None else None
            if single is not None and (not usable_ball(single, confidence)
                                       or not self.flat_range_consistent(single, pose, level)):
                single = None
            if multi is not None:
                if single is not None and multi.range_source == 'visual_image_track':
                    fuse_target_position(multi, single)
                choices.append((True, multi, original))
            elif single is not None:
                self.provisional.append(single)
                if self.allow_unconfirmed_flat:
                    choices.append((False, single, original))
        # Never send two overlapping, spatially agreeing hypotheses to Tracker
        # in a single frame. Do not use a map-only radius to merge nearby balls.
        kept = []
        for _, proposal, original in sorted(choices, key=lambda x: (not x[0], x[1].sigma, -x[1].confidence)):
            if any(np.linalg.norm(np.asarray(proposal.center)-prior.center)
                   <= .75*max(proposal.radius, prior.radius)
                   and np.linalg.norm(proposal.position-prior.position) <= .30 for prior in kept):
                continue
            kept.append(proposal)
            original.position = proposal.position
            original.center, original.radius = proposal.center, proposal.radius
            original.sigma, original.range_source = proposal.sigma, proposal.range_source

