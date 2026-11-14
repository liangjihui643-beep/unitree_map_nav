#!/usr/bin/env python3
"""Go2 废墟场地 MuJoCo 训练环境（对应比赛 PPT 的仿真契约）。

传感器契约（与赛题 PPT《实机本体》一致）：
- 12 关节：髋外展 / 屈伸 / 膝，编码器 50 Hz（关节位置+速度）
- RGB 相机：前视 320x240 @ 10 Hz（uint8 HWC；无深度相机）
- 激光雷达：72 x 9 线、水平 360° @ 10 Hz（量程 0.2~8 m，nan=无回波）
- IMU：陀螺仪（基座角速度，机体系）50 Hz；无 GPS、无场地真实坐标

动作空间：12 维关节位置目标（rad），50 Hz PD 控制（kp=50, kd=3.5），
底层 500 Hz 物理。Go2 模型为官方 unitree_mujoco 的力矩电机 + 关节型，
PD 在本环境内实现。

随机化（PPT：不要依赖固定坐标）：出生点、碎石位置、足球生命体位置
按回合种子随机；评测可能换布局，策略必须靠传感器自己认路。

示例：
    env = Go2ArenaEnv(seed=0)
    obs, info = env.reset()
    for _ in range(500):
        action = policy(obs)              # 如 onnxruntime 加载的 ONNX 策略
        obs, reward, terminated, truncated, info = env.step(action)
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import mujoco

from generate_arena_xml import ARENA_YAML, VICTIM_ZONES, build_arena, merge_robot

PKG = Path(__file__).resolve().parent
ROBOT_XML = PKG / 'robot/go2/go2.xml'
SENSORS_HZ = 50
CAMERA_HZ = 10
LIDAR_HZ = 10
LIDAR_SHAPE = (9, 72)          # 9 线 x 72 点，水平 360°
LIDAR_RANGE = (0.2, 8.0)
# 激光安装：机身前上方 0.33 m（对应 PPT 激光与相机的高度习惯），
# 垂直 9 线 -15°~+15°（PPT 未给垂直角，此处为建模假设）
LIDAR_HEIGHT = 0.33
LIDAR_VANGLES = np.linspace(-15.0, 15.0, LIDAR_SHAPE[0])
PD_KP, PD_KD = 50.0, 3.5
# 站立默认关节角（来自 unitree_mujoco 官方 stand 示例，rad）
STAND_THIGH, STAND_CALF = 0.608813, -1.21763


def stand_pose(joint_names):
    q = np.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        low = name.lower()
        if 'thigh' in low:
            q[i] = STAND_THIGH
        elif 'calf' in low or 'knee' in low:
            q[i] = STAND_CALF
        elif 'hip' in low:
            q[i] = math.copysign(0.00571868, 1.0 if low.startswith(('fl', 'rl')) else -1.0)
    return q


class Go2ArenaEnv:
    def __init__(self, seed=None, randomize=True, robot_xml=None, arena_yaml=None,
                 render_camera=True, kp=PD_KP, kd=PD_KD):
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.kp, self.kd = kp, kd
        self.render_camera = render_camera
        self._arena_yaml = Path(arena_yaml) if arena_yaml else ARENA_YAML
        self._robot_xml = Path(robot_xml) if robot_xml else ROBOT_XML
        self._cfg = yaml.safe_load(self._arena_yaml.read_text())
        self._load_model()

    # ------------------------------------------------------------------ 构建
    def _build_xml(self):
        root, _ = build_arena(self._cfg, self.rng if self.randomize else None,
                              with_victim=True)
        if self._robot_xml.exists():
            merge_robot(root, self._robot_xml)
            base = root.find("./worldbody/body[@name='base_link']")
            if base is None:
                base = root.find(".//body[@name='base']")
            if base is not None and base.find("camera[@name='front_rgb']") is None:
                cam = ET.SubElement(base, 'camera')
                cam.set('name', 'front_rgb')
                cam.set('pos', '0.30 0 0.14')
                cam.set('xyaxes', '0 1 0  0 0 -1')   # 前视（+X 机身系），y 向下
                cam.set('fovy', '90')
        # meshdir 写成绝对路径，保证 from_xml_string 也能找到网格
        compiler = root.find('compiler')
        compiler.set('meshdir', str(self._robot_xml.parent / 'assets'))
        compiler.set('autolimits', 'true')
        return ET.tostring(root, encoding='unicode')

    def _load_model(self):
        self._xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.decimation = 10                          # 50 Hz 控制
        self.ctrl_dt = self.dt * self.decimation
        self._cam_every = max(1, round(SENSORS_HZ / CAMERA_HZ))
        self._lidar_every = max(1, round(SENSORS_HZ / LIDAR_HZ))

        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            for i in range(self.model.nu)]
        self._base = -1
        for name in ('base_link', 'base', 'trunk'):
            self._base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if self._base >= 0:
                break
        assert self._base >= 0, '找不到基座 body（base_link/base/trunk）'
        self._base_qposadr = self.model.jnt_qposadr[self.model.body_jntadr[self._base]]
        # 关键：动作/关节名按"执行器顺序"，而 qpos/qvel 按"模型关节顺序"，
        # 两者在 go2.xml 中不一致（如 FL 在 qpos 中最先）。必须用执行器->关节
        # 的映射取地址，否则左右腿交叉配线：站立时因近似对称而不易察觉，
        # 任何不对称指令都会让关节失控。
        self._act_joints_cache = np.array(
            [self.model.actuator_trnid[i, 0] for i in range(self.model.nu)])
        self._act_qpos = np.array(
            [self.model.jnt_qposadr[j] for j in self._act_joints_cache])
        self._act_dof = np.array(
            [self.model.jnt_dofadr[j] for j in self._act_joints_cache])
        self._stand = stand_pose(self.joint_names)

        self.renderer = None
        if self.render_camera:
            self.model.vis.global_.offwidth = 320
            self.model.vis.global_.offheight = 240
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._step_count = 0

    # ------------------------------------------------------------------ 接口
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.randomize:
            self._load_model()          # 布局随回合重掷
        else:
            mujoco.mj_resetData(self.model, self.data)
        self._place_robot()
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def _place_robot(self):
        sx, sy, _, yaw = self._cfg['spawn']
        if self.randomize:
            sx += self.rng.uniform(-0.15, 0.15)
            sy += self.rng.uniform(-0.15, 0.15)
            yaw += self.rng.uniform(-0.2, 0.2)
        b = self._base_qposadr
        self.data.qpos[b:b + 3] = (sx, sy, 0.445)
        # MuJoCo 四元数顺序为 wxyz
        self.data.qpos[b + 3:b + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.data.qpos[self._act_qpos] = self._stand
        self.data.qvel[:] = 0

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape[0] == self.model.nu, '动作必须是 %d 维关节位置目标' % self.model.nu
        target = np.clip(action, self.model.jnt_range[self._act_joints_cache, 0],
                         self.model.jnt_range[self._act_joints_cache, 1])
        for _ in range(self.decimation):
            q = self.data.qpos[self._act_qpos]
            dq = self.data.qvel[self._act_dof]
            torque = self.kp * (target - q) - self.kd * dq
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0],
                                        self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
            if self._fallen():
                break
        self._step_count += 1
        obs = self._obs()
        terminated = self._fallen() or self._out_of_bounds()
        truncated = self._step_count >= self._max_steps()
        return obs, self._reward(), terminated, truncated, self._info()

    def _max_steps(self):
        return int(60.0 / self.ctrl_dt)

    # ------------------------------------------------------------------ 传感
    def _base_state(self):
        pos = self.data.xpos[self._base].copy()
        quat = self.data.xquat[self._base].copy()
        gyro_w = self.data.cvel[self._base][3:6]
        rot = self.data.xmat[self._base].reshape(3, 3)
        return pos, quat, rot.T @ gyro_w

    def _lidar(self):
        pos, _, _ = self._base_state()
        origin = pos.copy()
        origin[2] = max(0.05, pos[2] - 0.35 + LIDAR_HEIGHT)
        h = np.deg2rad(np.arange(LIDAR_SHAPE[1]) * (360.0 / LIDAR_SHAPE[1]))
        v = np.deg2rad(LIDAR_VANGLES)
        ranges = np.full(LIDAR_SHAPE, np.nan, dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # 组0=场地；机器人视觉/碰撞在2/3组，天然排除
        for iv in range(LIDAR_SHAPE[0]):
            cv, sv = math.cos(v[iv]), math.sin(v[iv])
            for i in range(LIDAR_SHAPE[1]):
                d = np.array([cv * math.cos(h[i]), cv * math.sin(h[i]), sv])
                dist = mujoco.mj_ray(self.model, self.data, origin, d,
                                     group_mask, True, -1, geomid)
                if LIDAR_RANGE[0] <= dist <= LIDAR_RANGE[1]:
                    ranges[iv, i] = dist
        return ranges

    def _rgb(self):
        if self.renderer is None or (self._step_count % self._cam_every) != 0:
            return None
        self.renderer.update_scene(self.data, camera='front_rgb')
        return self.renderer.render().copy()

    def _obs(self):
        pos, quat, gyro_b = self._base_state()
        return {
            'base_quat': quat.astype(np.float32),          # wxyz
            'base_pos': pos.astype(np.float32),            # 训练辅助；PPT 说明无 GPS，评测勿依赖
            'gyro': gyro_b.astype(np.float32),             # 机体系角速度 rad/s @50Hz
            'joint_pos': self.data.qpos[self._act_qpos].astype(np.float32),
            'joint_vel': self.data.qvel[self._act_dof].astype(np.float32),
            'lidar': self._lidar() if (self._step_count % self._lidar_every) == 0 else None,
            'rgb': self._rgb(),
        }

    # ------------------------------------------------------------------ 任务
    def _fallen(self):
        pos, quat, _ = self._base_state()
        up = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return pos[2] < 0.16 or up < 0.2

    def _out_of_bounds(self):
        a = self._cfg['arena']
        x0, y0 = a['minimum']
        pos, _, _ = self._base_state()
        return not (x0 - 0.5 <= pos[0] <= x0 + a['size'][0] + 0.5
                    and y0 - 0.5 <= pos[1] <= y0 + a['size'][1] + 0.5)

    def _reward(self):
        """占位奖励：存活 + 朝终点。正式评分以赛方为准。"""
        pos, quat, _ = self._base_state()
        goal = np.asarray(self._cfg['finish']['center'], dtype=np.float64)
        forward = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return float(0.1 * forward - 0.005 * np.linalg.norm(pos[:2] - goal))

    def _info(self):
        pos, _, _ = self._base_state()
        goal = np.asarray(self._cfg['finish']['center'], dtype=np.float64)
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'victim_ball')
        return {
            'time': self.data.time,
            'step': self._step_count,
            'dist_to_finish': float(np.linalg.norm(pos[:2] - goal)),
            'victim_pos': self.data.xpos[ball].copy() if ball >= 0 else None,
            'in_puddle': self._in_puddle(pos),
            'in_finish': bool(np.all(np.abs(pos[:2] - goal)
                                     < np.asarray(self._cfg['finish']['size']) / 2)),
        }

    def _in_puddle(self, pos):
        ux, uy = self._cfg['slope']['uphill_start']
        run = self._cfg['slope']['height'] / math.tan(math.radians(self._cfg['slope']['angle_deg']))
        return (abs(pos[0] - ux) < self._cfg['slope']['width'] / 2 - 0.05
                and abs(pos[1] - (uy + run * 0.55)) < 0.30)

    def report_victim(self, xy):
        """策略上报生命体位置（探测信号），返回到真值的误差（米）。"""
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'victim_ball')
        if ball < 0:
            return None
        return float(np.linalg.norm(np.asarray(xy, dtype=np.float64)
                                    - self.data.xpos[ball][:2]))

    def reset_soft(self, seed=None):
        """不换布局的复位：只重摆机器人和足球，不重载模型。
        供可视化 viewer 使用——launch_passive 持有的模型/数据对象不能更换。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._place_robot()
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'victim_ball')
        if ball >= 0:
            adr = self.model.jnt_qposadr[self.model.body_jntadr[ball]]
            bx, by = VICTIM_ZONES[self.rng.integers(0, len(VICTIM_ZONES))]
            self.data.qpos[adr:adr + 3] = (bx, by, 0.115)
            self.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def recover(self):
        """跌倒后在原地重新站起（演示脚手架：非比赛合法动作，评测时不可用）。
        保留当前 xy 与朝向，清空速度，恢复站立关节角。"""
        pos, quat, _ = self._base_state()
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        b = self._base_qposadr
        self.data.qpos[b:b + 3] = (pos[0], pos[1], 0.40)
        self.data.qpos[b + 3:b + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qpos[self._act_qpos] = self._stand
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
