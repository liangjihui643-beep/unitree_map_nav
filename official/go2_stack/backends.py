#!/usr/bin/env python3
"""底层行走后端接口（可替换）:

- CrawlBackend: 自建四拍爬行步态（stack/gait.py，今天就能跑的保底）
- UnitreeRLBackend: 加载 unitree_rl_gym 训练的 Go2 策略(torch.jit 或 ONNX)，
  观测构造与官方 deploy_mujoco 完全一致（47 维），PD 用训练同款的 kp=20/kd=0.5

上层（minimal_course.py）只面对统一接口：
    backend.set_cmd(vx, vy, wz)   # 机身速度指令 m/s, rad/s
    action = backend.action(obs)  # 12 维关节位置目标，50Hz
"""
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / 'mujoco_env'))
sys.path.insert(0, str(REPO / 'mujoco_env/stack'))

from gait import TrotGait      # noqa: E402


class CrawlBackend:
    """四拍爬行（无腾空相，跌倒率低），足端轨迹 + MuJoCo 数值 IK。

    该步态在 unitree_mujoco 站立位形（thigh 0.609/calf -1.218）下调参，
    比 unitree_rl_gym 的 action=0 蹲姿（0.8/-1.5）更高、更适合低刚度 PD。
    """

    name = 'crawl'
    # 按执行器顺序 FR,FL,RR,RL x (hip,thigh,calf)
    MUJOCO_STAND = np.array([
        -0.00571868, 0.608813, -1.21763,
        0.00571868, 0.608813, -1.21763,
        -0.00571868, 0.608813, -1.21763,
        0.00571868, 0.608813, -1.21763])

    def __init__(self, env, freq=2.0):
        self.env = env
        env._stand = self.MUJOCO_STAND.copy()
        env.reset()
        self.gait = TrotGait(env, freq=freq)
        self.cmd = (0.0, 0.0)
        self.stand_ticks = int(1.0 / env.ctrl_dt)

    def set_cmd(self, vx, vy, wz):
        self.cmd = (vx, wz)      # 爬行暂不支持横向 vy

    def action(self, obs):
        if self.stand_ticks > 0:
            self.stand_ticks -= 1
            return self.env._stand
        pos, quat, gyro = self.env._base_state()
        a = self.gait.step(self.cmd, quat, gyro, self.env.ctrl_dt)
        return a

    def on_reset(self):
        self.stand_ticks = int(0.8 / self.env.ctrl_dt)
        self.gait._prev = None


class UnitreeRLBackend:
    """unitree_rl_gym 训练策略的部署后端（观测合同与官方 deploy_mujoco 一致）。

    用法：先用 third_party/unitree_rl_gym 训练 Go2 并导出策略：
        python third_party/unitree_rl_gym/scripts/sim2sim.py --load ...  # 或训练+导出
    然后把导出的 .pt(torch.jit) 或 .onnx 路径传给 policy_path。
    """

    name = 'unitree_rl'

    def __init__(self, env, policy_path, num_actions=12, num_obs=47,
                 action_scale=0.25, ang_vel_scale=0.25,
                 dof_pos_scale=1.0, dof_vel_scale=0.05,
                 cmd_scale=(2.0, 2.0, 0.25), use_onnxruntime=False):
        self.env = env
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.action_scale = action_scale
        self.ang_vel_scale = ang_vel_scale
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale
        self.cmd_scale = np.asarray(cmd_scale, dtype=np.float32)
        self.default_angles = env._stand
        self.cmd = np.zeros(3, dtype=np.float32)
        self._action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self._counter = 0
        self.policy_path = policy_path
        self._use_ort = use_onnxruntime or str(policy_path).endswith('.onnx')
        if self._use_ort:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(policy_path),
                                                providers=['CPUExecutionProvider'])
        else:
            import torch
            self.policy = torch.jit.load(str(policy_path))

    def set_cmd(self, vx, vy, wz):
        self.cmd = np.array([vx, vy, wz], dtype=np.float32)

    @staticmethod
    def _gravity_orientation(quat):
        qw, qx, qy, qz = quat
        return np.array([
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz)], dtype=np.float32)

    def action(self, obs):
        self._counter += 1
        # 观测构造（与 unitree_rl_gym deploy/deploy_mujoco 逐行对应）
        pos, quat, gyro_b = self.env._base_state()
        qj = (obs['joint_pos'].astype(np.float32) - self.default_angles) * self.dof_pos_scale
        dqj = obs['joint_vel'].astype(np.float32) * self.dof_vel_scale
        g = self._gravity_orientation(quat)
        omega = gyro_b.astype(np.float32) * self.ang_vel_scale
        period = 0.8
        phase = (self._counter * self.env.ctrl_dt) % period / period
        o = self.obs
        o[:3] = omega
        o[3:6] = g
        o[6:9] = self.cmd * self.cmd_scale
        o[9:9 + 12] = qj
        o[21:33] = dqj
        o[33:45] = self._action
        o[45:47] = (math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase))
        if self._use_ort:
            out = self.session.run(None, {'obs': self.obs[None, :]})[0]
            act = np.asarray(out, dtype=np.float32).reshape(-1)
        else:
            import torch
            with torch.no_grad():
                act = self.policy(torch.from_numpy(self.obs).unsqueeze(0)
                                  ).numpy().squeeze().astype(np.float32)
        self._action = act
        return act * self.action_scale + self.default_angles

    def on_reset(self):
        self._action = np.zeros(self.num_actions, dtype=np.float32)
        self._counter = 0
