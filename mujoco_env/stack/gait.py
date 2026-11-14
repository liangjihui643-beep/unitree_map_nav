#!/usr/bin/env python3
"""Go2 腿部运动学与传统控制步态（PPT 允许"传统控制"路线）。

- 解析逆解（IK）：髋外展(绕x) + 大腿/小腿(绕y) 三连杆，
  与官方 go2.xml 的关节轴向约定一致，数值上经 FK 验证。
- 对角小跑步态：机身系足端椭圆轨迹，支撑相足端随速度反移。
- 姿态稳定：俯仰->大腿角、横滚->外展角 比例-微分修正。

这是把 Gazebo 侧"上层规划 -> cmd_vel -> 控制器"架构迁移到 MuJoCo 的
控制底座：上层(local_nav)输出 (vx, wz)，这里输出 12 维关节位置目标。
"""
import math

import mujoco
import numpy as np

# 与官方 go2.xml 一致
HIP_X = 0.1934          # 前后髋关节 x
HIP_Y = 0.0465          # 左右髋关节 y（左正右负）
THIGH_OFFSET = 0.0955   # 大腿相对髋外展节的侧向偏置
L2 = 0.213              # 大腿长
L3 = 0.213              # 小腿长

LEGS = ('FR', 'FL', 'RR', 'RL')


def leg_side(leg):
    return 1.0 if leg in ('FL', 'RL') else -1.0   # 左 +1 右 -1


def leg_hip_xy(leg):
    return HIP_X if leg in ('FR', 'FL') else -HIP_X, leg_side(leg) * HIP_Y


def fk(leg, q_hip, q_thigh, q_knee):
    """正向运动学：返回足端在机身系的位置 (x, y, z)。"""
    s = leg_side(leg)
    hx, hy = leg_hip_xy(leg)
    ch, sh = math.cos(q_hip), math.sin(q_hip)
    ct, st = math.cos(q_thigh), math.sin(q_thigh)
    ck, sk = math.cos(q_thigh + q_knee), math.sin(q_thigh + q_knee)
    x = hx - L2 * st - L3 * sk
    y = hy + s * THIGH_OFFSET * ch
    z = s * THIGH_OFFSET * sh - L2 * ct - L3 * ck
    return np.array([x, y, z])


def ik(leg, foot):
    """逆运动学：足端机身系坐标 -> (q_hip, q_thigh, q_knee)。"""
    s = leg_side(leg)
    hx, hy = leg_hip_xy(leg)
    vx, vy, vz = np.asarray(foot, dtype=np.float64) - np.array([hx, hy, 0.0])
    # 1) 髋外展：vy = s*l1*cos(th) 恒成立（平面链无 y 分量），
    #    sh 取正分支（θ 尽量小）。可达域 |vy| <= l1，超出则钳到边界。
    a = s * THIGH_OFFSET
    ch = float(np.clip(vy / a, -1.0, 1.0))
    sh = math.sqrt(max(0.0, 1.0 - ch * ch))
    q_hip = math.atan2(sh, ch)
    # 2) 平面二连杆（去掉外展偏置的 z 贡献后）
    u_x = vx
    u_z = vz - a * sh
    # 本机器人膝角恒为负（go2.xml range [-2.72,-0.84]），取肘在后的唯一解
    D = math.hypot(u_x, u_z)
    D = min(max(D, abs(L2 - L3) + 1e-4), L2 + L3 - 1e-4)
    cos_a = (L2 * L2 + D * D - L3 * L3) / (2 * L2 * D)
    cos_a = min(1.0, max(-1.0, cos_a))
    alpha = math.acos(cos_a)
    phi = math.atan2(u_z, u_x)
    q_thigh = -phi + alpha - math.pi / 2
    q_knee = -2.0 * alpha
    return np.array([q_hip, q_thigh, q_knee])


def ik_numeric(env, leg, foot_base, iters=15, damp=1e-3):
    """数值 IK：以 MuJoCo 自身雅可比做阻尼最小二乘，规避 XML 几何约定陷阱。

    foot_base: 期望足端位置（机身系）。返回该腿 [hip, thigh, calf] 关节角。
    """
    m, d = env.model, env.data
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, leg + '_hip_joint'),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, leg + '_thigh_joint'),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, leg + '_calf_joint')]
    adrs = [int(m.jnt_qposadr[j]) for j in jids]
    dofs = [int(m.jnt_dofadr[j]) for j in jids]
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + '_foot')
    base = env._base
    Rw = d.xmat[base].reshape(3, 3)
    target_w = d.xpos[base] + Rw @ np.asarray(foot_base, dtype=np.float64)
    saved = d.qpos.copy()
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target_w - d.xpos[fid]
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jac(m, d, jacp, jacr, fid, d.xpos[fid])
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + damp * damp * np.eye(3), err)
        for j, ad, delta in zip(jids, adrs, dq):
            d.qpos[ad] += float(delta)
            d.qpos[ad] = float(np.clip(d.qpos[ad], m.jnt_range[j][0], m.jnt_range[j][1]))
    solved = np.array([float(d.qpos[a]) for a in adrs])
    d.qpos[:] = saved   # IK 只解不算：还原状态，解作为 PD 目标
    return solved


class TrotGait:
    """四拍爬行步态（无腾空相，任意时刻 >=3 足支撑，静态稳定）。

    cmd: (vx 前进速度 m/s, wz 转向角速度 rad/s)
    输出: 12 维关节位置目标，顺序与 joint_names 一致。
    """

    SWING_FRAC = 0.20        # 每腿摆动相占周期比例：四拍步态必须 < 0.25，
                             # 否则相邻腿摆动窗口重叠（同侧两腿同时离地=侧翻）
    PHASE_OFFS = {'FR': 0.0, 'RR': 0.25, 'FL': 0.5, 'RL': 0.75}

    def __init__(self, env, freq=1.6, swing_h=0.012, k_pitch=0.25, k_roll=0.2):
        self.env = env
        self.names = env.joint_names
        self.freq = freq
        self.swing_h = swing_h
        self.phase = 0.0
        # 用 MuJoCo 自身 FK 取站立足端（机身系）
        m, d = env.model, env.data
        base = env._base
        Rw0 = d.xmat[base].reshape(3, 3).T
        self.stand_feet = {}
        for leg in LEGS:
            fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + '_foot')
            self.stand_feet[leg] = Rw0 @ (d.xpos[fid] - d.xpos[base])
            # 名义足端略内收，远离外展工作区边缘
            self.stand_feet[leg][1] -= leg_side(leg) * 0.008
        self.k_pitch, self.k_dpitch = k_pitch, 0.06
        self.k_roll, self.k_droll = k_roll, 0.05
        self.slew = 0.5          # 关节目标变化率上限 rad/s（平滑启停）
        self._prev = None

    def step(self, cmd, quat, gyro, dt):
        vx, wz = cmd
        self.phase = (self.phase + self.freq * dt) % 1.0
        w, x, y, z = quat
        pitch = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x))
        roll = math.atan2(2 * (w * x - y * z), 1 - 2 * (x * x + y * y))
        targets = {}
        for leg in LEGS:
            foot0 = self.stand_feet[leg]
            phi = (self.phase + self.PHASE_OFFS[leg]) % 1.0
            # 支撑相地面速度（机身速度 + 转向带来的足端线速度）
            gs_x = -vx + wz * foot0[1]
            gs_y = -wz * foot0[0]
            stance_len = abs(gs_x) * (1 - self.SWING_FRAC) / self.freq
            if phi < self.SWING_FRAC:
                t = phi / self.SWING_FRAC
                foot = foot0.copy()
                foot[0] = foot0[0] - stance_len / 2 + stance_len * t
                foot[2] = foot0[2] + self.swing_h * math.sin(np.pi * t)
                foot[1] += gs_y * t * 0.02
            else:
                t = (phi - self.SWING_FRAC) / (1 - self.SWING_FRAC)
                foot = foot0.copy()
                foot[0] = foot0[0] + stance_len / 2 - stance_len * t
                foot[1] = foot0[1] + gs_y * (t - 0.5) * 0.05
            # 可达性保护：足端横向不得超出外展工作区
            hx, hy = leg_hip_xy(leg)
            s_ = leg_side(leg)
            max_dy = THIGH_OFFSET - 0.004
            foot[1] = hy + float(np.clip(foot[1] - hy, -max_dy, max_dy)) * s_                 if s_ > 0 else hy + float(np.clip(foot[1] - hy, -max_dy, max_dy))
            q = ik(leg, foot)
            # 姿态稳定：俯仰->大腿、横滚->外展（左右反向），全部腿施加
            q[1] += -self.k_pitch * pitch - self.k_dpitch * gyro[1]
            q[0] += -leg_side(leg) * (self.k_roll * roll + self.k_droll * gyro[0])
            targets[leg + '_hip'] = q[0]
            targets[leg + '_thigh'] = q[1]
            targets[leg + '_calf'] = q[2]
        action = np.array([targets[n] for n in self.names])
        if self._prev is not None:
            max_d = self.slew * dt * 60      # 限幅窗口放大，避免过度钝化
            action = self._prev + np.clip(action - self._prev, -max_d, max_d)
        self._prev = action.copy()
        return action
