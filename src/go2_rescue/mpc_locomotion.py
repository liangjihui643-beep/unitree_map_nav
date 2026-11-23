"""Pinned open-source Go2 convex MPC with MuJoCo-native model adaptation.

Upstream solver, gait and swing-leg torque controller are reused under MIT.
Only joints + IMU are read from the plant. Absolute simulated root pose/twist
are NOT used by this controller or the navigation stack.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rot2
from .types import Velocity
from .terrain import ground_surface, GroundPatch
from .step_terrain import CurbObservation, raised_swing

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]


def upstream():
    root = Path(__file__).resolve().parents[2] / "third_party/go2_convex_mpc/src"
    if not root.is_dir():
        raise RuntimeError("Pinned third_party/go2_convex_mpc source is required")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from convex_mpc.centroidal_mpc import CentroidalMPC
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.gait import Gait
    from convex_mpc.leg_controller import LegController
    return CentroidalMPC, ComTraj, Gait, LegController


class KinematicModel:
    """Official robot-only dynamics driven by measured joints/IMU.

    Contact-scheduled leg kinematics estimate local base velocity and height.
    Slip and rough terrain can bias this estimate; lidar corrects upper odometry.
    The controller's local origin is independent of the scene/world origin.
    """
    def __init__(self, robot_xml: Path, plant_model, plant_data):
        self.model = mujoco.MjModel.from_xml_path(str(robot_xml))
        self.mjdata = mujoco.MjData(self.model)
        self.plant_model, self.plant_data = plant_model, plant_data
        self.source_q = [plant_model.jnt_qposadr[plant_model.joint(n).id] for n in JOINTS]
        self.source_v = [plant_model.jnt_dofadr[plant_model.joint(n).id] for n in JOINTS]
        self.q_addresses = [self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS]
        self.v_addresses = [self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINTS]
        self.base = self.model.body("base_link").id
        self.feet = {leg: self.model.body(f"{leg}_foot").id for leg in LEGS}
        self.position = np.array([0., 0., .30])
        self.velocity_world = np.zeros(3)
        self.kinematic_velocity_world = np.zeros(3)
        self.forward_hint = None
        self.forward_hint_time = -np.inf
        self.last_time = None
        self.jacobians = {}
        self.previous_jacobians = {}
        self.dt = .006
        self.height_above_support = .30
        self.x_pos_des_world = self.y_pos_des_world = 0.
        self.x_vel_des_world = self.y_vel_des_world = self.yaw_rate_des_world = 0.
        self.current_config = SimpleNamespace()
        self.data = SimpleNamespace(Ig=SimpleNamespace())
        self.update(np.ones(4, dtype=int))
        self.hip_offsets = {leg: self.R_world_to_body @ (self.mjdata.xpos[self.model.body(f"{leg}_thigh").id] - self.position)
                            for leg in LEGS}
        self.current_config.get_dq = lambda: self.mjdata.qvel.copy()

    def update(self, stance):
        source, data, model = self.plant_data, self.mjdata, self.model
        now = float(source.time)
        dt = 0. if self.last_time is None else now - self.last_time
        self.dt = max(.001, dt)
        self.last_time = now
        quaternion = np.asarray(source.sensor("imu_quat").data).copy()
        omega_body = np.asarray(source.sensor("imu_gyro").data).copy()
        rotation = Rotation.from_quat(np.roll(quaternion, -1))
        self.R_body_to_world = rotation.as_matrix()
        self.R_world_to_body = self.R_body_to_world.T
        self.rpy = rotation.as_euler("xyz")
        self.R_z = np.eye(3)
        self.R_z[:2, :2] = rot2(self.rpy[2])
        data.qpos[:3], data.qpos[3:7] = self.position, quaternion
        data.qpos[self.q_addresses] = source.qpos[self.source_q]
        data.qvel[:] = 0.
        data.qvel[self.v_addresses] = source.qvel[self.source_v]
        data.qvel[3:6] = omega_body
        mujoco.mj_forward(model, data)
        estimates, heights = [], []
        for index, leg in enumerate(LEGS):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac, None, self.feet[leg])
            if stance[index]:
                r = data.xpos[self.feet[leg]] - self.position
                estimates.append(-(jac[:, self.v_addresses] @ data.qvel[self.v_addresses]
                                   + np.cross(self.R_body_to_world @ omega_body, r)))
                heights.append(.022 - r[2])
        if estimates:
            measured = np.median(estimates, axis=0)
            # Independent observation; never feed scan-corrected controller
            # velocity back into the same scan matcher's odometry prior.
            self.kinematic_velocity_world = .7*self.kinematic_velocity_world+.3*np.clip(measured,-2.,2.)
            if self.forward_hint is not None and now-self.forward_hint_time < .25:
                forward = self.R_z[:,0]
                measured += .8*(self.forward_hint-measured@forward)*forward
            self.velocity_world = .7 * self.velocity_world + .3 * np.clip(measured, -2., 2.)
            self.position += self.velocity_world * max(0., dt)
            self.height_above_support = float(np.median(heights))
        data.qpos[:3] = self.position
        data.qvel[:3] = self.velocity_world
        mujoco.mj_forward(model, data)
        self.current_config.base_pos = self.position.copy()
        self.current_config.base_vel = self.velocity_world.copy()
        self.pos_com_world = data.subtree_com[self.base].copy()
        jcom = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jcom, self.base)
        self.vel_com_world = jcom @ data.qvel
        mass = float(model.body_subtreemass[self.base])
        inertia = np.zeros((3, 3))
        for body in range(1, model.nbody):
            r = data.xipos[body] - self.pos_com_world
            orient = data.ximat[body].reshape(3, 3)
            inertia += orient @ np.diag(model.body_inertia[body]) @ orient.T
            inertia += model.body_mass[body] * ((r @ r) * np.eye(3) - np.outer(r, r))
        self.data.Ig.mass, self.data.Ig.inertia = mass, inertia
        self.previous_jacobians = self.jacobians
        self.jacobians = {}
        for leg in LEGS:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac, None, self.feet[leg])
            self.jacobians[leg] = jac
        self.mass_matrix = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, self.mass_matrix)

    def get_hip_offset(self, leg):
        return self.hip_offsets[leg].copy()

    def get_single_foot_state_in_world(self, leg):
        return self.mjdata.xpos[self.feet[leg]].copy(), self.jacobians[leg] @ self.mjdata.qvel

    def get_foot_lever_world(self):
        return [self.mjdata.xpos[self.feet[leg]] - self.pos_com_world for leg in LEGS]

    def compute_com_x_vec(self):
        return np.r_[self.pos_com_world, self.rpy, self.vel_com_world,
                     self.R_body_to_world @ self.mjdata.qvel[3:6]].reshape(12, 1)

    def compute_3x3_foot_Jacobian_world(self, leg):
        index = LEGS.index(leg)
        return self.jacobians[leg][:, self.v_addresses[index * 3:index * 3 + 3]]

    def compute_full_foot_Jacobian_world(self, leg):
        return self.jacobians[leg]

    def compute_Jdot_dq_world(self, leg):
        previous = self.previous_jacobians.get(leg, self.jacobians[leg])
        return (self.jacobians[leg] - previous) @ self.mjdata.qvel / self.dt

    def compute_dynamcis_terms(self):
        return self.mjdata.qfrc_bias.copy(), np.zeros_like(self.mass_matrix), self.mass_matrix


class TerrainGait:
    """Reuse upstream swing timing/XY placement with observed terrain height."""
    def __init__(self, gait, controller):
        self.gait, self.controller = gait, controller

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def compute_swing_traj_and_touchdown(self, go2, leg):
        _, touchdown = self.gait.compute_swing_traj_and_touchdown(go2, leg)
        curb = self.controller.curb
        if curb is not None and curb.ready and not curb.entry_complete:
            along = (touchdown[:2]-curb.origin)@curb.direction
            lead = -.12 if leg.startswith("F") else -.085
            if lead < along-curb.edge < .08:
                touchdown[:2] += (curb.edge+.08-along)*curb.direction
            elif along < curb.edge:
                touchdown[:2] -= max(0., along-(curb.edge-.035))*curb.direction
        touchdown[2] = self.controller.ground_z(touchdown[:2]) + .022
        foot, _ = go2.get_single_foot_state_in_world(leg)
        if curb is not None and curb.ready and not curb.entry_complete:
            return raised_swing(foot, touchdown, self.gait.swing_time), touchdown
        return self.gait.make_swing_trajectory(foot, touchdown, self.gait.swing_time, h_sw=.065), touchdown


class ConvexMPC:
    def __init__(self, model, data, resource_root: Path, height=.30):
        MPC, Trajectory, Gait, LegController = upstream()
        from convex_mpc import centroidal_mpc, gait as gait_module
        centroidal_mpc.MU = .4  # Match the official robot's contact friction.
        gait_module.HEIGHT_SWING = .065
        self.go2 = KinematicModel(resource_root / "robot/go2/physics/go2.xml", model, data)
        self.terrain_plane = np.zeros(3)  # z = ax + by + c, controller-local frame.
        self.terrain_updates = 0
        self.terrain_slope_limit_deg = 15.
        self.terrain_patch = None
        self.curb = None
        self.gait = TerrainGait(Gait(2.5, .65), self)
        self.standing = TerrainGait(Gait(2.5, 1.), self)
        self.slope_gait = TerrainGait(Gait(1.5, .75), self)
        self.step_gait = TerrainGait(Gait(1., .60), self)
        self.height = height
        self.traj = Trajectory.__new__(Trajectory)
        self.traj.N = 16
        self.traj.pos_des_world = self.go2.pos_com_world.copy()
        self.leg = LegController()
        self.command = Velocity()
        self.forces = np.zeros((4, 3))
        self.last_solve = -np.inf
        self.solves = 0
        self.solve_times = []
        self.active_gait = self.standing
        self.last_motion_pose = None
        self._reference(0.)
        owner = self
        class SurfaceMPC(MPC):
            def _precompute_friction_matrix(self, traj):
                # Reserve all three coefficients per face, including zero
                # entries, so a changing support normal preserves QP sparsity.
                rows = np.repeat(np.arange(16*self.N), 3)
                columns = []
                for k in range(self.N):
                    for leg in range(4):
                        columns.extend(np.tile(12*self.N+12*k+3*leg+np.arange(3), 4))
                self.surface_rows, self.surface_columns = rows, columns
                return self.surface_friction()

            def surface_friction(self):
                import scipy.sparse as sp
                cones = []
                for leg in LEGS:
                    foot = owner.go2.get_single_foot_state_in_world(leg)[0]
                    gradient = owner.terrain_plane[:2].copy() if owner.terrain_slope_limit_deg > 15. else np.zeros(2)
                    if owner.terrain_patch is not None:
                        offsets = np.array([[.015,0.],[-.015,0.],[0.,.015],[0.,-.015]])
                        z = owner.terrain_patch.height(foot[:2]+offsets)
                        if np.isfinite(z).all():
                            gradient = (z[[0,2]]-z[[1,3]])/.03
                    normal = np.r_[-gradient, 1.];normal /= np.linalg.norm(normal)
                    tx = np.array([1.,0.,gradient[0]]);tx /= np.linalg.norm(tx)
                    ty = np.cross(normal,tx)
                    cones.append(np.array([tx-.4*normal,-tx-.4*normal,ty-.4*normal,-ty-.4*normal]))
                values = np.tile(np.asarray(cones).ravel(), self.N)
                matrix = sp.csc_matrix((values,(self.surface_rows,self.surface_columns)), shape=(16*self.N,self.nvars))
                return self._scipy_to_casadi(matrix)

            def _update_sparse_matrix(self, traj):
                self.A_ineq_static = self.surface_friction()
                return super()._update_sparse_matrix(traj)

        self.surface_solver_type = SurfaceMPC
        self.surface_solver = None
        self.flat_solver = MPC(self.go2, self.traj)
        self.solver = self.flat_solver
        self.actuator_columns = []
        for actuator in range(model.nu):
            joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator, 0]))
            self.actuator_columns.append(JOINTS.index(joint))
        self.limits = .9 * np.minimum(abs(model.actuator_ctrlrange[:, 0]), abs(model.actuator_ctrlrange[:, 1]))

    def ground_z(self, xy):
        if self.curb is not None and self.curb.ready and not self.curb.entry_complete:
            return self.curb.height(xy)
        if self.terrain_patch is not None:
            height = self.terrain_patch.height(xy)[0]
            if np.isfinite(height):
                return float(height)
        return float(np.asarray(xy) @ self.terrain_plane[:2] + self.terrain_plane[2])

    def observe_forward_motion(self, pose, timestamp):
        """Lidar displacement bounds longitudinal stance-slip on the local ramp."""
        old = self.last_motion_pose
        self.last_motion_pose = (float(timestamp), pose.xy().copy())
        if self.terrain_slope_limit_deg <= 15.:
            self.go2.forward_hint = None
            return
        if old is None or pose.quality < .65:
            return
        dt = timestamp-old[0]
        delta = pose.xy()-old[1]
        if not .04 <= dt <= .25 or np.linalg.norm(delta) > .10:
            return
        speed = float(delta@rot2(pose.yaw)[:,0]/dt)
        previous = speed if self.go2.forward_hint is None else self.go2.forward_hint
        self.go2.forward_hint = float(np.clip(.75*previous+.25*speed,-.3,.3))
        self.go2.forward_hint_time = timestamp

    def enable_curb(self):
        """Arm a measured curb capability. Does not enable navigation motion."""
        if self.curb is None:
            self.curb = CurbObservation(self.go2.position[:2], self.go2.R_z[:2, 0],
                                       self.ground_z(self.go2.position[:2]))

    def observe_support(self, frame, visual_ground=None):
        """Fill a missing near-body lidar plane using measured stance/IMU."""
        from .stance_surface import StanceSurface
        if not hasattr(self, "stance_surface"):
            self.stance_surface = StanceSurface()
        plane = self.stance_surface.update(frame)
        if plane is None or self.curb is not None:
            return
        from .geometry import rot2
        yaw = self.stance_surface_yaw = np.arctan2(self.go2.R_body_to_world[1,0], self.go2.R_body_to_world[0,0])
        gradient = rot2(yaw) @ plane[:2]
        # Reconstructed controller state, never simulator root state.
        height = float(self.go2.position[2]+plane[2])
        if np.linalg.norm(gradient) > np.tan(np.deg2rad(self.terrain_slope_limit_deg)):
            return
        slope = .75*self.terrain_plane[:2]+.25*gradient
        previous = self.ground_z(self.go2.position[:2])
        level = .75*previous+.25*height
        self.terrain_plane = np.r_[slope, level-slope@self.go2.position[:2]]
        self.terrain_updates += 1
        if visual_ground is not None and len(visual_ground):
            self.visual_ground_hits=np.asarray(visual_ground) @ self.go2.R_body_to_world.T+self.go2.position
            self.visual_ground_stamp=frame.timestamp
        if frame.timestamp-getattr(self,"visual_ground_stamp",-np.inf)<=1.:
            contacts=frame.support_points @ self.go2.R_body_to_world.T + self.go2.position
            hits=self.visual_ground_hits
            near=(np.linalg.norm(hits[:,:2]-self.go2.position[:2],axis=1)<1.5)
            measured=np.vstack([contacts,hits[near]])
            samples=ground_surface(measured,level,.06,.6,self.terrain_slope_limit_deg)
            if len(samples)>=4:
                self.terrain_patch=GroundPatch(samples,.6,self.terrain_slope_limit_deg)

    def observe_terrain(self, body_points):
        """Estimate a near-body support plane from lidar only, without plant truth."""
        g = self.go2
        body_points = np.asarray(body_points)
        points = body_points @ g.R_body_to_world.T + g.position
        if self.curb is not None:
            if not self.curb.entry_complete:
                self.curb.observe(points, g.position[:2])
            if self.curb.ready and not self.curb.entry_complete:
                distance = (g.position[:2]-self.curb.origin)@self.curb.direction-self.curb.edge
                if distance > .60:
                    self.curb.entry_complete = True
                    self.terrain_plane = np.array([0., 0., self.curb.upper])
                self.terrain_updates += 1
                return
        # Fit the lane beneath the feet, not the flat ground beside a ramp or
        # both side walls. A broad circular patch mixes different surfaces.
        near = (body_points[:, 0] > -.6) & (body_points[:, 0] < .8) & (abs(body_points[:, 1]) < .28)
        reference = g.position[2] - g.height_above_support
        slope_limit = 30. if self.curb is not None and self.curb.entry_complete else self.terrain_slope_limit_deg
        samples = ground_surface(points[near], reference, .06, .45, slope_limit)
        if self.terrain_slope_limit_deg <= 15.:
            self.terrain_patch = None
        if len(samples) < 12:
            return
        if self.terrain_slope_limit_deg > 15. and self.curb is None:
            # At a slope transition one plane cannot describe both front and
            # hind supports. Feet query the observed piecewise surface; COM
            # height/pitch follow its values beneath the measured feet.
            patch = GroundPatch(samples, .45, slope_limit)
            feet = np.array([g.get_single_foot_state_in_world(leg)[0] for leg in LEGS])
            support = patch.height(feet[:, :2])
            self.terrain_patch = patch
            if np.isfinite(support).sum() >= 3:
                valid = np.isfinite(support)
                design = np.column_stack([feet[valid,:2]-g.position[:2], np.ones(valid.sum())])
                fit, _, rank, _ = np.linalg.lstsq(design, support[valid], rcond=None)
                if rank == 3 and np.linalg.norm(fit[:2]) <= np.tan(np.deg2rad(slope_limit)):
                    old = float(g.position[:2]@self.terrain_plane[:2]+self.terrain_plane[2])
                    slope = .8*self.terrain_plane[:2]+.2*fit[:2]
                    height = .8*old+.2*fit[2]
                    self.terrain_plane = np.r_[slope,height-slope@g.position[:2]]
                    self.terrain_updates += 1
                    return
        xy = samples[:, :2] - g.position[:2]
        design = np.column_stack([xy, np.ones(len(samples))])
        fit, _, rank, _ = np.linalg.lstsq(design, samples[:, 2], rcond=None)
        if rank < 3 or np.linalg.norm(fit[:2]) > np.tan(np.deg2rad(slope_limit)) or np.quantile(abs(design @ fit - samples[:, 2]), .9) > .035:
            return
        old_height = self.ground_z(g.position[:2])
        slope = .8 * self.terrain_plane[:2] + .2 * fit[:2]
        height = .8 * old_height + .2 * fit[2]
        self.terrain_plane = np.r_[slope, height - slope @ g.position[:2]]
        self.terrain_updates += 1

    def _reference(self, time):
        g, tr, cmd = self.go2, self.traj, self.command
        dt = min(.05, self.active_gait.gait_period / tr.N)
        times = (np.arange(tr.N) + 1) * dt
        initial = g.compute_com_x_vec().ravel()
        velocity = g.R_z @ np.array([cmd.vx, cmd.vy, 0.])
        velocity[2] = self.terrain_plane[:2] @ velocity[:2]
        # Keep an integrated position reference close to the estimated body.
        tr.pos_des_world[:2] = np.clip(tr.pos_des_world[:2], initial[:2] - .08, initial[:2] + .08)
        tr.pos_des_world[:2] += velocity[:2] * .02
        surface, gradient = self.ground_z(initial[:2]), self.terrain_plane[:2]
        if self.terrain_patch is not None:
            surface = float(initial[:2]@gradient+self.terrain_plane[2])
        if self.curb is not None and self.curb.ready and not self.curb.entry_complete:
            surface, gradient = self.curb.body_surface(initial[:2])
        tr.pos_des_world[2] = surface + self.height
        g.x_pos_des_world, g.y_pos_des_world = tr.pos_des_world[:2]
        g.x_vel_des_world, g.y_vel_des_world = velocity[:2]
        g.yaw_rate_des_world = cmd.wz
        tr.initial_x_vec = initial.reshape(12, 1)
        tr.m, tr.I_com_world = g.data.Ig.mass, g.data.Ig.inertia
        tr.pos_traj_world = tr.pos_des_world[:, None] + velocity[:, None] * times
        tr.vel_traj_world = np.repeat(velocity[:, None], tr.N, axis=1)
        tr.rpy_traj_world = np.zeros((3, tr.N))
        body_gradient = g.R_z[:2, :2].T @ gradient
        tr.rpy_traj_world[0] = np.arctan(body_gradient[1])
        tr.rpy_traj_world[1] = -np.arctan(body_gradient[0])
        tr.rpy_traj_world[2] = initial[5] + cmd.wz * times
        tr.omega_traj_world = np.zeros((3, tr.N))
        tr.omega_traj_world[2] = cmd.wz
        tr.contact_table = self.active_gait.compute_contact_table(time, dt, tr.N)
        for leg, arm in zip(LEGS, g.get_foot_lever_world()):
            setattr(tr, f"r_{leg.lower()}_foot_world", np.repeat(arm[:, None], tr.N, axis=1))
        # Use the upstream discrete single-rigid-body dynamics, gravity and
        # friction-constrained QP; no surrogate force controller is substituted.
        tr._discreteDynamics(dt)

    def update(self, time: float, command: Velocity) -> np.ndarray:
        previous = self.active_gait
        # Always establish support before starting the periodic gait. Keep
        # trotting in place once started so stopping does not strand swing feet.
        step_active = self.curb is not None and self.curb.ready and not self.curb.entry_complete
        self.active_gait = (self.standing if time < .6 else self.step_gait if step_active
                            else self.slope_gait if self.terrain_slope_limit_deg > 15. else self.gait)
        if previous is not self.active_gait:
            self.leg.last_mask[:] = 2
            self.last_solve = -np.inf
        if self.terrain_slope_limit_deg > 15.:
            if self.surface_solver is None:
                self.surface_solver = self.surface_solver_type(self.go2, self.traj)
            desired_solver = self.surface_solver
        else:
            desired_solver = self.flat_solver
        if self.solver is not desired_solver:
            self.solver = desired_solver
            self.last_solve = -np.inf
        self.go2.update(self.active_gait.compute_current_mask(time).ravel())
        self.command = Velocity() if time < .6 else command
        if step_active:
            self.command = Velocity(float(np.clip(self.command.vx, -.3, .45)),
                                    float(np.clip(self.command.vy, -.035, .035)),
                                    float(np.clip(self.command.wz, -.12, .12)))
        if time - self.last_solve >= .019:
            self._reference(time)
            solution = self.solver.solve_QP(self.go2, self.traj)
            if not self.solver.solver.stats().get("success", False):
                raise RuntimeError("Open-source MPC QP did not converge")
            forces = np.asarray(solution["x"]).ravel()[12 * self.traj.N:]
            self.forces = forces[:12].reshape(4, 3)
            if not np.isfinite(self.forces).all():
                raise RuntimeError("MPC produced non-finite forces")
            self.last_solve = time
            self.solves += 1
            self.solve_times.append(self.solver.solve_time / 1000.)
        canonical = np.concatenate([self.leg.compute_leg_torque(leg, self.go2, self.active_gait,
                                                               self.forces[i], time).tau for i, leg in enumerate(LEGS)])
        return np.clip(canonical[self.actuator_columns], -self.limits, self.limits)

    def measured_support_points(self):
        """Reconstructed stance soles from sensed joints and IMU, in body FLU."""
        g = self.go2
        stance = self.active_gait.compute_current_mask(float(g.plant_data.time))[:, 0]
        points = []
        for leg, standing in zip(LEGS, stance):
            if standing:
                sole = g.get_single_foot_state_in_world(leg)[0] - g.position
                sole[2] -= .022
                points.append(g.R_world_to_body @ sole)
        return np.asarray(points, dtype=float).reshape(-1, 3)

    def measured_velocity(self) -> Velocity:
        measured = self.go2.kinematic_velocity_world if getattr(self,"raw_odometry",False) else self.go2.velocity_world
        basis=self.go2.R_z.T if getattr(self,"raw_odometry",False) else self.go2.R_world_to_body
        body = basis @ measured
        return Velocity(float(body[0]), float(body[1]), float(self.go2.mjdata.qvel[5]))
