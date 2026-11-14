# 官方训练包集成（Go2 MuJoCo 最小完赛）

基于比赛官方训练资源包 `training-package-v8` 的最小完赛集成。
物理权威 = 官方 MJB；底层行走可插拔（当前：自建四拍爬行保底；
宇树开源 RL 策略训练就位后即可切换）。

## 目录

```
official/
├── robotics-bench-go2-competition-training-resources/   # 官方包（45MB，不入库）
│   ├── scene/mujoco/competition-course.mjb              # 物理权威：Go2+完整赛道
│   ├── robot/go2/physics/go2.xml                        # Go2 物理模型(12关节/12电机)
│   └── life-detection/football/                         # 足球生命体模型
├── go2_stack/                     # 本仓库代码
│   ├── official_env.py            # MJB 环境封装（PD 控制/IMU/编码器/合成激光）
│   ├── backends.py                # 行走后端：CrawlBackend / UnitreeRLBackend
│   ├── minimal_course.py          # 最小完赛：建图+A*+局部导航+终点+结束信号
│   └── map_result.png / end_signal.json / minimal_course_map.png  (运行产物)
└── third_party/                   # 宇树开源训练框架（不入库，见下）
    ├── unitree_rl_gym/            # Isaac Gym 训练 + Sim2Sim(deploy_mujoco)
    └── unitree_rl_mjlab/          # MuJoCo 原生训练（本机训练用）
```

## 环境准备

```bash
# Python 3.12 + 官方评测同版本栈
conda create -n mujoco312 python=3.12 -y && conda activate mujoco312
pip install -r mujoco_env/requirements.txt          # mujoco==3.17.0 onnx==1.23.2 onnxruntime==1.31.1

# 官方训练包：从赛题组获取 training-package-v8.zip 解压到 official/
# 宇树开源框架：
git clone https://github.com/unitreerobotics/unitree_rl_mjlab third_party/unitree_rl_mjlab
pip install -e third_party/unitree_rl_mjlab          # torch/rsl_rl 等训练依赖
```

## 最小完赛（当前即可运行）

```bash
cd official/go2_stack
PYTHONPATH=.:../../mujoco_env:../../mujoco_env/stack python minimal_course.py
PYTHONPATH=. ... minimal_course.py --viewer     # 带可视化窗口
```

流程：合成激光(契约 72×9)建占据栅格 → A* 全局规划 → 纯追踪+VFH 局部导航 →
行走后端执行 → 进入终点区(x 9.3~11.9, y 7.0~9.9) → 写出 end_signal.json 结束信号。

当前底层为四拍爬行（保守、无腾空相），在碎石/栏杆处会跌倒并原地恢复
（`recover()` 是演示脚手架，评测不可用）。跌倒率高的根治方案 = 宇树 RL 策略。

## 宇树开源 RL 底层（训练后切换）

```bash
# 1) 训练 Go2 速度跟踪策略（GPU，数小时；4096 环境需要大显存，本机 3060 用 1024）
python third_party/unitree_rl_mjlab/scripts/train.py Unitree-Go2-Flat \
    --env.scene.num-envs=1024 --headless
# 2) 导出/播放验证
python third_party/unitree_rl_mjlab/scripts/play.py Unitree-Go2-Flat --load <logs下model_*.pt>
# 3) 部署到官方赛道（观测合同与 unitree deploy 一致，47 维）
PYTHONPATH=. ... minimal_course.py --backend unitree_rl --policy <policy.pt> --kp 20 --kd 0.5
```

`UnitreeRLBackend` 的观测构造与宇树官方 `deploy_mujoco` 逐行对应：
omega(3) + 重力投影(3) + cmd(3) + 关节相对位置(12) + 关节速度(12) +
上一动作(12) + 相位 sin/cos(2) = 47 维；动作×0.25 + 默认角 = 关节目标。

## 官方包关键事实（分析结论）

- MJB 一处加载全部物理：Go2(12 电机, FR/FL/RR/RL 序) + 赛道 828 几何体
- 传感器：12 关节位置/速度/力矩 + imu_quat/imu_gyro/imu_acc + frame_pos/frame_vel
- **无相机**：RGB 是独立冻结合同，本地需要时自行加渲染（评测由平台提供）
- 官方几何全在组 2/3（机器人自身被本封装改到组 4 以排除激光自击）
- 官方 qpos0 出生点 (0,0) 与围墙重叠，本地用场内偏移 (0.6,0.6) 出发
- 足球：r=0.11, 0.43kg, 黑白格纹（model.xml 可取，需自行摆入场景）

## 与 Gazebo / 自建 MuJoCo 分支的关系

| 分支 | 角色 | 底层 |
| --- | --- | --- |
| `main` / `feat/competition-arena` | Gazebo+ROS 系统集成验证（Go1） | 宇树 Gazebo 控制器 |
| `feat/mujoco-env` | 自建 MuJoCo 环境（几何与 Gazebo 同源，已跑通 explore） | 四拍爬行+数值IK |
| `feat/official-env`（本分支） | **官方评测对齐：最小完赛** | 可插拔，训练后切宇树 RL |
