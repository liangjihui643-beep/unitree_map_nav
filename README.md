# Unitree Noetic 地图与导航工作区

上游仓库：<https://github.com/macc-n/ros_unitree>。

本目录用于复现上游的 **Go1** 仿真，并新增了比赛场地示意模型；尚未完成Go2适配或足球检测。

比赛场景启动：`./run_competition.sh mapping`。包含上下坡、S弯、死胡同、五处碎石和三种管道，见 [比赛场地说明](src/unitree_competition/README.md)。原房间演示继续使用 `./run_demo.sh`。

已在本机完成编译和运行验证：站立、约 1.38 米的受控行走、Gmapping 建图并保存地图、AMCL 定位和 move_base/DWA 到点导航。最终二维避障配置在图形界面模式下的目标点误差约 0.056 米；这是当前简单房间场景的单次验证结果，不代表比赛场景性能。

上游版本：`0c7b48ade880fd9096d196123b3f9c98ea1131b3`。环境和额外依赖版本记录在 `config/environment.json`，上游修改记录在 `patches/ros_unitree-noetic.patch`。外层本地Git仓库的场景开发分支是 `feat/competition-arena`；上游目录保持独立克隆，通过固定版本和补丁复现。

## 目录

- `src/ros_unitree/`：克隆的上游代码。
- `src/unitree_local_demo/`：本地房间场景、组合 launch 和控制器启动脚本。
- `src/unitree_competition/`：参数化比赛场地、坡面模型、生成器及平面图。
- `.venv/`：Python 3.8 虚拟环境，使用系统 ROS Python 包。
- `.runtime/root/`：新增 ROS 控制器和 LCM 等依赖的本地安装前缀。
- `build/`、`devel/`：Catkin 编译输出。
- `logs/`：编译、运行和验证记录。

系统已有的 Ubuntu 20.04、ROS Noetic、Gazebo 11 和 C++ 编译工具继续复用。Python 环境、额外依赖、源码、编译产物和项目 ROS 日志均保存在本目录；这不是包含完整操作系统的容器。

## 编译

```bash
cd /home/l/code/unitree_map_nav
./scripts/build.sh
```

## 建图

```bash
cd /home/l/code/unitree_map_nav
./run_demo.sh mapping
```

控制器自动通过原有键盘接口进入站立和 MoveBase 模式。另一个终端可发送低速运动指令：

```bash
source /home/l/code/unitree_map_nav/activate.sh
rostopic pub -r 10 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1}, angular: {z: 0.0}}'
```

结束速度命令后，要明确发送零速度停车：

```bash
rostopic pub -1 /cmd_vel geometry_msgs/Twist '{}'
```

保存地图：

```bash
rosrun map_server map_saver -f /home/l/code/unitree_map_nav/src/unitree_local_demo/maps/demo_room
```

在启动仿真的终端按 `Ctrl+C` 关闭该次仿真。

如果演示是通过本目录的后台会话脚本启动的，可先查看或停止：

```bash
source /home/l/code/unitree_map_nav/activate.sh
python /home/l/code/unitree_map_nav/scripts/session.py status
python /home/l/code/unitree_map_nav/scripts/session.py stop
```

同一个工作区不要同时启动建图和导航两个仿真。

## 已知地图导航

先结束建图仿真，再启动：

```bash
cd /home/l/code/unitree_map_nav
./run_demo.sh navigation
```

在 RViz 使用 `2D Pose Estimate` 设置初始位置，再通过 `2D Nav Goal` 设置导航目标。房间每次重置时，机器人从约 `(0, 0)`、朝向 x 正方向开始。

若不需要图形界面：

```bash
./run_demo.sh mapping gui:=false rviz:=false
```

ROS master 使用 `11331`，Gazebo master 使用 `11351`，避免与其他工作区默认端口冲突。所有操作终端都应先 `source activate.sh`。

## 运行验证

建图模式启动后，在另一个终端执行：

```bash
source /home/l/code/unitree_map_nav/activate.sh
python /home/l/code/unitree_map_nav/scripts/verify_demo.py mapping
```

它检查激光、IMU、里程计、机器人站立、受控行走及 Gmapping 地图，并保存地图和 JSON 记录。导航模式启动后执行：

```bash
python /home/l/code/unitree_map_nav/scripts/verify_demo.py navigation
```

它通过 AMCL 和 move_base 发送目标点，并检查动作返回值及实际位移。Gazebo 真值仅用于验证测量，不输入建图和导航算法。

验证结果位于 `logs/verification_mapping.json` 和 `logs/verification_navigation.json`。是否验证通过应以实际生成的结果为准。

## 本地适配

- 场景 launch 允许指定本目录中的 world 文件，楼梯模型去除上游机器的绝对路径。
- 导航控制器的初始速度明确置零。
- 新增本地组合 launch 和 PTY 启动器，依次执行上游键盘的站立与 MoveBase 模式切换，不替换运动算法。
- 本地二进制依赖通过 `activate.sh` 加入插件和动态库搜索路径。
- RViz 的路径话题改为实际使用的 GlobalPlanner 和 DWA 话题。
- 本地房间导航使用二维激光障碍层；原始深度相机的三维体素层与腿部里程计的高度原点不一致，会产生传感器越界告警，因此本地演示关闭该层。深度相机模型和数据发布保留，但本次未验证三维相机避障。

上游在没有实时调度权限时会输出 `setProcessScheduler failed`，本机验证是在普通调度权限下完成的，没有修改系统实时调度配置。
