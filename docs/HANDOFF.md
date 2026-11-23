# 感知接续记录（2026-11-18）

先读 AGENTS.md、CURRENT_STATE.md 和 感知交付报告.md。用户要求自主推进感知，达到可展示、误差不大即交付，后续精细优化另交别人。领域是 LiDAR 定位/建图、补偿低矮地表的视觉与激光融合、足球代理检测测距和地图标记。保留现有 MPC、原场景资源与 V1 管道出口安全停机边界。

最新中文页面 output/Go2-移动感知验收.html。结果目录 perception-a-final-support、perception-b-final-support 是最终代码对两份真实物理记录的重放。源物理记录为 perception-a-final-live 和 perception-b-survey-final。最终页面和报告有明确范围，整场自主覆盖与可通行性仍未通过。

已部署两个 ONNX。足球小模型以圆候选为输入，14,065 个裁剪样本训练 10 轮，B 环视未参与训练。支撑脚坐标进入可选 Frame.support_points 并保存到记录，旧记录兼容空字段。StanceSurface 对每个同时刻支撑脚做中心化差分，用 IMU 对齐不同时刻方向，在 0.8 s 内拟合局部坡度；方向不够时注明水平假设，缺数据时失效。PerceptionDemo 使用该参考测低矮高度并跨帧落图。已确认球通过图像和三维位置排除进入碎石层。

关键验证是 16 个低矮样本全部物理复核为石块、目标地图位置误差 3/4 cm、两份记录各一个目标。坡道接近有 46 个可评估斜面帧，坡度误差 p90 1.44°，这段没有完成原坡道。5 项新算法定向检查通过，之前 28 项相关检查通过。Git 工作树元数据在 Windows 入口不可正常查询，以 manifest 源码/模型 SHA 与实际文件为准，未提交或推送。

下步定位全场路线阻塞的输入层。当前正式激光每水平方位最近有效回波、h>=-0.22m，通常看不到平地，空气自由格缺少落脚证据。把近场关节支持、视觉地面候选、局部坡度、激光实测、未知与置信范围接到现有导航/地形接口，再跑真实连续路线。直接入口是本轮验证路径，ROS2 桥和新增支撑脚字段仍需新集成。全场覆盖图是只读真值参照，禁止把参照位置输回定位或导航。

Windows 双击 Go2-Perception-Demo.cmd。较长流程加 --survey --variant b。WSL 使用 .venv/bin/python，训练使用已有 /usr/local/bin/python3.12。仅复算采用 scripts/replay_perception_acceptance.py，再运行 scripts/finish_fused_acceptance.py。所有失败实验保留。旧 V1 交接已存到 HANDOFF_V1_2026-11-16.md。
