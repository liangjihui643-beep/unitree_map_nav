#!/usr/bin/env python3
"""官方训练包 MJB 场景的 MuJoCo 环境封装。

- 物理权威：official/robotics-bench-go2-competition-training-resources/
  scene/mujoco/competition-course.mjb（Go2 + 完整赛道，Unitree 官方物理）
- 控制：unitree_rl_gym 同款 PD（kp/kd 可配，unitree 训练值 20/0.5）
- 传感：MJB 自带 imu_quat/imu_gyro/imu_acc + 关节位置/速度/力矩；
  激光雷达契约(72x9)由 mj_ray 合成（训练辅助，评测由平台按合同提供）
- 接口与 mujoco_env/go2_arena_env 保持鸭子类型一致，stack/gait 可直接复用
"""
import math
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(__file__).resolve().parent.parent / 'robotics-bench-go2-competition-training-resources'
MJB = PKG / 'scene/mujoco/competition-course.mjb'

# 赛道范围（由 MJB 几何测得：地面 12x10，起点角(0,0)）
ARENA_X0, ARENA_Y0, ARENA_W, ARENA_H = 0.0, 0.0, 12.0, 10.0

# unitree_rl_gym go2 默认关节角（policy action=0 时的目标位姿）
GO2_DEFAULT_ANGLES = {
    'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
}


class OfficialGo2Env:
    def __init__(self, kp=50.0, kd=3.5, render_camera=False, spawn_xy=(0.6, 0.6)):
        # spawn_xy: 官方 qpos0 在 (0,0) 墙角、与两边围墙重叠；本地训练/演示
        # 用场内偏移点出发。评测时以平台重置逻辑为准。
        self.spawn_xy = spawn_xy
        self.model = mujoco.MjModel.from_binary_path(str(MJB))
        self.data = mujoco.MjData(self.model)
        self.kp, self.kd = kp, kd
        self.dt = self.model.opt.timestep               # 0.002
        self.decimation = 10                             # 50Hz 控制
        self.ctrl_dt = self.dt * self.decimation
        self.render_camera = render_camera

        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            for i in range(self.model.nu)]
        self._base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self._base_qposadr = self.model.jnt_qposadr[self.model.body_jntadr[self._base]]
        jids = [self.model.actuator_trnid[i, 0] for i in range(self.model.nu)]
        self._act_qpos = np.array([self.model.jnt_qposadr[j] for j in jids])
        self._act_dof = np.array([self.model.jnt_dofadr[j] for j in jids])
        # 站立关节角（unitree_rl_gym go2 默认，按执行器顺序）
        self._stand = np.array([GO2_DEFAULT_ANGLES[n + '_joint'] for n in self.joint_names])
        # 机器人自身几何（base_link 子树，body 1..17）改到组4，
        # 激光 mask 只测组2/3（场地），从而排除自击
        robot_bodies = set(range(1, 18))
        for i in range(self.model.ngeom):
            if int(self.model.geom_bodyid[i]) in robot_bodies:
                self.model.geom_group[i] = 4
        self.renderer = None
        self._step_count = 0
        self._lidar_cache = None

    # ------------------------------------------------------------ 基础
    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        b = self._base_qposadr
        self.data.qpos[b:b + 3] = (self.spawn_xy[0], self.spawn_xy[1], 0.445)
        self.data.qpos[self._act_qpos] = self._stand
        self.data.qvel[:] = 0.0
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):   # 站稳
            self.step(self._stand)
        return self._obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        target = np.clip(action,
                         self.model.jnt_range[self.model.actuator_trnid[:, 0], 0],
                         self.model.jnt_range[self.model.actuator_trnid[:, 0], 1])
        for _ in range(self.decimation):
            q = self.data.qpos[self._act_qpos]
            dq = self.data.qvel[self._act_dof]
            tau = self.kp * (target - q) - self.kd * dq
            self.data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0],
                                        self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
            if self._fallen():
                break
        self._step_count += 1
        self._lidar_cache = None
        return self._obs(), 0.0, self._fallen() or self._out(), False, self._info()

    def _base_state(self):
        pos = self.data.xpos[self._base].copy()
        quat = self.data.xquat[self._base].copy()
        rot = self.data.xmat[self._base].reshape(3, 3)
        return pos, quat, rot.T @ self.data.cvel[self._base][3:6]

    def _fallen(self):
        pos, quat, _ = self._base_state()
        up = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return pos[2] < 0.14 or up < 0.2

    def _out(self):
        pos, _, _ = self._base_state()
        return not (-0.6 <= pos[0] <= 12.6 and -0.6 <= pos[1] <= 10.6)

    def recover(self):
        pos, quat, _ = self._base_state()
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        b = self._base_qposadr
        self.data.qpos[b:b + 3] = (pos[0], pos[1], 0.42)
        self.data.qpos[b + 3:b + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qpos[self._act_qpos] = self._stand
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    # ------------------------------------------------------------ 传感
    def sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        n = self.model.sensor_dim[sid]
        return self.data.sensordata[adr:adr + n].copy()

    def lidar(self, shape=(9, 72), max_range=8.0, min_range=0.2,
              vangles=(-15.0, 15.0), height=0.33):
        """契约规格 72x9 360° 激光的合成版（训练辅助）。

        注意：官方 MJB 物理不含激光传感器，评测时由平台按合同提供；
        这里用 mj_ray 以同等规格合成，保证本地训练/验证的接口一致。
        """
        key = (shape, max_range)
        if self._lidar_cache is not None and self._lidar_cache[0] == key:
            return self._lidar_cache[1]
        pos, _, _ = self._base_state()
        origin = pos.copy()
        origin[2] = max(0.05, pos[2] - 0.35 + height)
        nv, nh = shape
        h = np.deg2rad(np.arange(nh) * (360.0 / nh))
        v = np.deg2rad(np.linspace(vangles[0], vangles[1], nv))
        ranges = np.full(shape, np.nan, dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        # 官方 MJB：场地与机器人碰撞体均在组2/3（无组0），bodyexclude 基座子树排除自身
        group_mask = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
        for iv in range(nv):
            cv, sv = math.cos(v[iv]), math.sin(v[iv])
            for i in range(nh):
                d = np.array([cv * math.cos(h[i]), cv * math.sin(h[i]), sv])
                dist = mujoco.mj_ray(self.model, self.data, origin, d,
                                     group_mask, True, -1, geomid)
                if min_range <= dist <= max_range:
                    ranges[iv, i] = dist
        self._lidar_cache = (key, ranges)
        return ranges

    def _obs(self):
        pos, quat, gyro_b = self._base_state()
        return {
            'base_quat': quat.astype(np.float32),
            'base_pos': pos.astype(np.float32),
            'gyro': gyro_b.astype(np.float32),
            'joint_pos': self.data.qpos[self._act_qpos].astype(np.float32),
            'joint_vel': self.data.qvel[self._act_dof].astype(np.float32),
            'imu_acc': self.sensor('imu_acc').astype(np.float32),
            'lidar': self.lidar() if (self._step_count % 5) == 0 else None,
            'rgb': None,
        }

    def _info(self):
        pos, _, _ = self._base_state()
        return {'time': self.data.time, 'step': self._step_count, 'base_pos': pos}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
