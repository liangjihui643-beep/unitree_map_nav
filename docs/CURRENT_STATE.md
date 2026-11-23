# Current State
2026-11-23 非 ONNX 感知接口已接入：MuJoCo 感知演示原子输出 perception_snapshot.json，含可通行地图、碎石层、目标和测距；公开任务输入适配按帧龄解码和去重，144 维仅输出距离证据接口。见 [Go2 感知接口交接](Go2感知接口交接-2026-11-23.md)。

2026-11-23 Ubuntu 24.04 环境已切换为默认 WSL distro：原生 Python 3.12.3；`/opt/gl-libs` 注入 20.04 Mesa 21 EGL/GLX 栈后，24.04 离屏渲染走 D3D12（NVIDIA RTX 4060），不再退回 llvmpipe。相关测试 66 passed；全量 181 passed、1 failed；水洼/稀疏角落测试已拆分修正，剩余坡道项按运控交接处理。两个 `Go2-*.cmd` 已指向 24.04 并设置 `LD_LIBRARY_PATH`/`LIBGL_DRIVERS_PATH`/`MUJOCO_GL=egl`。代码逻辑未改。详见[Ubuntu24环境迁移](Ubuntu24环境迁移-2026-11-23.md)。

2026-11-23 隧道入口：唯一 GridMap 加入测得连续坡面与历史票重分类，分开机身碰撞和坡面落脚膨胀。新版任务同 36 姿态短演示由不通变连通，可通行 98→106，建图/导航回归 39 项通过。未验证 MPC 实走或 144 维独立建图。见 [隧道入口建图修复与小演示](隧道入口建图修复与小演示-2026-11-23.md)。

2026-11-23 按[新版 Go2 接口计划](新版Go2接口与感知建图实施计划-2026-11-23.md)完成第一步：新增 `public_task.py`、合同六步冒烟与传感器参数化。`public-course-agent` 6 个文件摘要全部匹配（scene `5dea69a8…`、sensors `8db3e32b…`），obs `[6,189]`、rgb `[6,3,64,64]`、state `[6,32]`、sensor_meta `[6,8]`、LiDAR `[6,72]`；第 0/5 步新采样、第 1–4 步复用，batch 1/2 与 state 全零通过；`go2-agent-v1` ONNX 未提供，未做模型推理。默认演示地图未改。详见[合同 profile 与冒烟](新版Go2接口合同profile与冒烟-2026-11-23.md)。

2026-11-23 最终保留单一 `GridMap`：移除端点缺失当地地面时回退到脚底高度，改用 2.5 m 内最近实测/插值/足迹地面；无当地地面端点存 `pending_endpoints` 并在每帧重分类；新增 `structure_votes`/`overhead_votes` 层与保存字段；`ContractMap` 退为实验记录，不作为运行时地图。同 154 帧对照（基线取此前已验证落盘值）：占据 1037→1030、坡面假占据 26→17、平地 23→27、真障碍 357→362/893、自由 10479→8040、错误自由 848→849、待定 314。统一记账后坡面/真墙改善，平地误占略升；`tunnel3_east` 局部 `traversable()` 不连通；唯一一次短段（36 帧，同一 GridMap）沿途可通行格 31→98，但终点 (11.40,6.00) 虽为自由、地面/净空/危险正常，仍被膨胀/坡度层挡住，**短段未通过，暂不能定版**。详见[GridMap证据分类合入](GridMap证据分类合入-2026-11-23.md)。
2026-11-22 激光建图进入两轮交付。执行方实现自由更新保护与统一评分，复核方只做代码审查和关键验证，最终只跑一次完整路线。见 [激光建图两轮实现与验收计划](激光建图两轮实现与验收计划-2026-11-22.md)。

2026-11-22 两轮完成并回退：整场挡墙开启跑完 476.6 仿真秒 / 1565.5 墙钟，62/62、零跌倒/接触/进水/越界，覆盖 99.71%、边界召回 91.61%（808/882）、隧道 89/102、普通墙 460/460，但到边界 P90 0.30→0.50 m、占据格 1334→1708，属关键退步；按计划将 `map_wall_truncate` 默认改回 False，保留为实验开关与诊断，交付基线默认地图 + 独立结构层（隧道 80/102→95/102），坡顶越墙标残余限制。评估器 20 cm 容差与分项核对保留（A 777/882、B 724/882）。详见[两轮执行记录](激光建图两轮执行-2026-11-22.md)。

2026-11-22 结构层已作为独立输出接好，未改默认占据地图/规划。`scripts/structure_layer.py` 输出 mask；`evaluate_scene_coverage.py --structure-layer` 用同一真值/分类/阈值并列原图、结构层单独、并集。记录 A（`rubble-coverage-speed-full-01`）：隧道 80/102→95/102（单独）→97/102（并集），普通墙 460/460，边界召回 85.83%→86.51%，到边界 P90 0.30→0.77 m，>20 cm 格 32→61；记录 B（`coverage-delivery-02`）：隧道 79/102→93/102→98/102，墙 460/460，P90 0.316→0.447 m。结构层单独不替代原图（总边界召回 70.98%/69.16%）；并集不进规划，等下一里程碑复核隧道顶封路/坡顶越墙。详见[结构层与统一复算](激光结构层独立输出与统一口径复算-2026-11-22.md)。
2026-11-22 用户确认区域式碎石加速版按本次整场实跑结果验收。后续激光建图改造另行实施，见 [激光建图复核与实施计划](激光建图复核与实施计划-2026-11-22.md)。

2026-11-22 区域式碎石加速版从原起点实跑 62/62 巡查点，476.6 仿真秒，未跌倒、无障碍接触、进水或越界。在线图层发布 444 格，真值内 339 格，超过 20 cm 的轮廓外格 7。完整录像和发布地图见 [整场实跑](../output/rubble-coverage-speed-full-01/index.html)。

2026-11-22 区域式碎石图层缩小多视角计算范围，并用重复视觉测距建立初始小块。483 帧重放 P90 118→93 ms，右下石堆第一次路过发布 8 个正确格，达到 50 格仍在 424.5 仿真秒。最终全场正确 345→349，距碎石超过 20 cm 为 0。独立短段走完两巡查点。见 [覆盖时间线与复现](../experiments/rubble_coverage_speed/README.md)。

2026-11-22 碎石新增可选区域关联与地图贡献替换模式。483 帧同输入重放全场正确格 332→345，圈内可见格 124→128/183，超过 20 cm 的误标 2→0，P90 118 ms。独立短段完成两巡查点，22 次修订、14 次发布格撤回。覆盖与整场时延目标仍待达成。见 [实现、结果与复现](../experiments/rubble_region_mapping/README.md)。

2026-11-22 碎石定位加入同一分割石面内相隔双距离锚点补全。483 帧同输入重放，圈内石堆 113→126/199，整场轮廓内 309→332 格，轮廓外 77→87 格，超 20 cm 由 1→2 格。现场 0.7 仿真秒首次发布，13.5 仿真秒走完碎石区两巡查点。见 [对比、评分与复现](../experiments/rubble_surface_anchors/README.md)。

2026-11-22 碎石规划图层保留绿色核心，并在线补入邻接、有观测支持的候选边缘。483 帧同输入重放中轮廓内 284→309 格、轮廓外 66→77 格、距碎石超过 20 cm 维持 1 格。定向现场 0.7 仿真秒首次发布、完成两处巡查点，未见距碎石超过 20 cm 的标记。见 [三图对照、评分与复现](../experiments/rubble_footprint_fusion/README.md)。

2026-11-22 碎石补全入口改为二维规划图层，规划器替换最新范围，停止源分支的高度拟合与输出。483 帧同输入重放中轮廓内 281→284 格、轮廓外 68→66 格、距石堆超过 20 cm 的格 19→1，定位落图 P90 约 73→46 ms。现场 0.7 仿真秒首次发布，完成两巡查点。见 [演示、评分与复现](../experiments/rubble_footprint/README.md)。

2026-11-22 碎石补全分支加入多视角石面和地面校验及可撤销地图。483 帧整场重放，带激光或独立视角支持的轮廓内格 220→281，轮廓外格 71→73，实证撤销 1 个轮廓外格。新现场单独完成碎石区两个巡查点。见 [对照、评分与复现](../experiments/rubble_revision/README.md)。

2026-11-21 深度补全、历史投影与跨高程单目候选接入 `--rubble-completion-model`。整场重放在真值轮廓内 133→264 格，轮廓外 26→138 格；含实测距离或重复支持的子集为 220/291 格。新现场完成碎石区 2/2 巡查点，2 个栅格输出重复支持的高度。见 [演示和评分](../experiments/rubble_completion/demo/index.html)。

2026-11-21 `--rubble-single-view` 增加激光距离锚定。整场旧录制重放的同像素 P90 29.2→18.0 cm，真值轮廓内 133/159 格，仍有 15 个可见坡下帧缺距离支持。新定向仿真完成 2/2 个碎石巡查点，官网比赛验收和高度仍缺证据。见 [演示与评分](../experiments/rubble_range_fusion/demo/index.html)。

2026-11-21 单帧碎石深度分层候选已接入 `--rubble-single-view`。从原起点跑完 62/62 个巡查点，484.2 仿真秒，零跌倒、接触、进水和越界。分档命中 16730/21494，同一真实碎石射线位置 p90 旧 39.6 cm / 新 72.9 cm，落差子集旧 32.4 cm / 新 87.1 cm。单帧观测定位仍保留为候选。见 [新旧对照与完整视频](../experiments/rubble_single_frame/demo/index.html)。
2026-11-21 新增可选碎石精度层 `--rubble-precision-model`。近距 MPC 新实跑的在线一致性子层水平误差 p90 9.0 cm（517 点），点级相对高度 p90 3.6 cm（294 点），10 格可评分地图高度 p90 6.2 cm。原图覆盖 72/95 格，一致性子层覆盖 15/95 格，另附 5 个 RGB 高度候选。常规距离水平与全图估高仍未通过厘米级验收。见 [碎石定位与高度精度实验](../experiments/rubble_precision/README.md) 和 `output/rubble-precision-online-01/score`。

2026-11-21 新增可选参数 `--multiview-rubble`。两段常规巡查同录像可见轮廓覆盖 67/160 → 92/160、84/178 → 94/178。近距离 MPC 实跑的 44 个可评分高度误差 p90 为 8.9 cm，常规段 6 格 p90 为 15.5 cm。高度验收线为 8 cm。详见 `docs/碎石多关键帧与精度复核-2026-11-21.md`。

2026-11-20 更新：在默认关闭的 `--slope-perception --hybrid-ball` 实验模式，图像轨迹补偿相机转向，未验证的固定平面位置仅记录为候选。原整场录像普通球 3/3，10/20/31 cm 垫高 MPC 无配额各一次在线上报，最大 5.82 cm；只通过目标本地验收，见 `docs/目标转向补偿与垫高验证-2026-11-20.md`。雷达证据层依本地高度规则得到 80 个可确定平面端点格，不输出为障碍；144 维独立定位建图和碎石估高仍未验收，见 `docs/雷达端点歧义复核-2026-11-20.md`。

2026-11-19 碎石感知分项：在线图层新增有界多视角轮廓推断及实验 RGB 双视角估高，控制和规划未改。三球整场记录复算的轮廓命中 421/892→717/892；新代码短采在线生成 31 格视觉估高。整场复算高度误差 p90 为 20.8–27.7 cm，精度与完整度仍待验收。见 `docs/碎石在线轮廓与视觉估高复核-2026-11-19.md`。

2026-11-19 原起点连续整场感知验收通过。coverage-delivery-03 走完 62/62、79.39 m、490.30 仿真秒，零障碍接触、未跌倒、无进水越界。同一次实录经过可复现后处理，最终融合边界召回 91.38%，激光占据位置 p90 0.10 m，普通墙 460/460，球测距偏 0.0498 m、标记偏 0.0572 m。最终交付 `output/coverage-delivery-03/delivery/perception_delivery_report.html`；原在线输出独立保留。见 `docs/MuJoCo整场感知验收与演示-2026-11-19.md`。

持续验收标准与实验记录见 `docs/持续感知验收计划-2026-11-19.md`。碎石短段 33–36 已通过，实时终点墙误标格 37→0。后半段 37–42 已走过，43 二维结构匹配失败；补了隧道三维表面参考，失效记录上 251/251 可定位，等待 north-segment-03 在线验证。整场尚未通过。

2026-11-19 球区局部实跑见 `docs/碎石段定位探针与终点球实跑-2026-11-19.md`。结束区离起点单独启动，走 0.42 m、可见 24 帧命中 16、确认 1 点，标记误差 0.027 m，地形图层 JSON 已实时落盘。墙上标记被分割为碎石，形成 37 个视觉候选格，待处理。碎石段回放短窗仅将卡点后有效定位从 32/297 提到 43/297，改动默认关闭。静态探针在卡点转 −0.50 rad 可由质量 0 恢复到 .661，待闭环验证。

05 的低矮层接口及折返复核见 `docs/低矮地形图层与05折返复核-2026-11-19.md`。同记录复算得到 146 个地形格，其中 124 个同时在激光底图呈自由，14 格带实测高度；重建 272 个在线视觉避让格。新展示运行导出 `terrain_layer.json`，正式联调待验证。

2026-11-19 感知专项复核见 `docs/05巡查-激光建图与路线停机复核-2026-11-19.md`。05 观测格覆盖与实际结构测绘分开看，路线在碎石段主要因定位约束不足停止。射线按高度清除的离线对照保留在 E 盘，误占代价过大，正式地图没有启用。已修复路线沿用被新避障证据封锁的目标格，并将巡查完成写入整场验收门禁，两项仍需集成验证。

## 2026-11-18 新主流程：公开地图预规划巡查

用户确认先绕开碎石，采用预规划路径，不再以逐格实测地面支持为前进前提。新默认入口 `Go2-Full-Coverage.cmd` / `scripts/run_full_coverage.py` 启用 `--planned-survey`：公开地图先验单独用于导航与激光匹配定位，实际测绘图层独立累计，在线不读取仿真位姿。旧流程可用 `--legacy-exploration`。详见 `docs/预规划巡查实施与验证-2026-11-18.md`。

离线路线最终62/62点连通且避开已知石堆；不是动态通过证明。02已完成上下坡，轨迹p90=.03085m、0碰撞/未跌倒，但只覆盖65.35%，S弯停住；03下降末端摔倒，随后修正实时坡面限速。04行走279.9s至碎石入口第30点停住，随后补回入口方向门禁与已有全机身姿态规划。

2026-11-19最终结果：05已结束，308.3s仿真/1841.0s墙钟，43.98m，巡查索引34/62。通过原入口卡点，但末段定位匹配退化，持续localization_unreliable，因无进展停止；路线和整场验收未通过。观测格覆盖98.15%不等于路线完成；轨迹p90=.01929m、边界p90=.14142m、边界20cm召回76.53%。0碰撞/未跌倒/无进水越界，视觉碎石146格、激光低矮14格、测高9条；球未进入视野，无目标标记。本轮后停止追加代码修改及长跑，先向用户交付实际地图与视频。后续优先重放已有末段定位输入，避免反复从起点重跑。历史局部感知精度不可当作整场精度。

## 2026-11-18 感知融合与支撑面参考

目标是激光地图、低矮地表与足球检测测距在同一图层可复现展示。最新说明为 `docs/感知交付报告.md`，中文总览为 `output/Go2-移动感知验收.html`。双击 `Go2-Perception-Results.cmd` 查看，`Go2-Perception-Demo.cmd` 运行。

最终感知代码通过正式 `PerceptionDemo.observe` 重放 `perception-a-final-live`（11 s 实际物理采集）与 `perception-b-survey-final`（32 s 实际物理采集）。最终结果目录为 `perception-a-final-support`、`perception-b-final-support`。A/B 路径 3.59/5.54 m，定位误差 p90 0.05/0.11 m，碎石 P/R 85.9/70.8% 与 85.0/72.6%，16 个低矮激光高度样本全部经物理射线复核属于碎石，足球各一个确认点，误差约 3/4 cm。低矮层使用关节/IMU 支撑参考与正式激光，记录邻域不同帧支持。图像和地图双重排除已确认足球，处理偶发漏检串入碎石。

新增 `StanceSurface` 以不同步态相位的支撑脚中心化差分与 IMU 拟合局部坡度。原 MPC 的坡道接近 8.6 s 有 46 个可评估斜面帧，参考角度 2.7–8.1°，坡度误差 p90 1.44°。新算法数学定向检查 5/5。此前相关回归 28 项通过。真实机器人、全身处于完整 18° 坡面时的目标测高、可通行性、全场自主覆盖和最终模型经过 ROS2 桥的集成仍待验证。

两个部署模型是 `models/seg_low_obstacle_perception.onnx` 与 `models/ball_proposal.onnx`。当前主地面几何为 12×10 m，B 的已观测格占该矩形约 31.5%，包括墙体/遮挡区分母，不能当作整场完成率。

下一项是把现有运动/导航需要的地面支持和感知端来源分层接好，再验证完整连续路线。V1 管道出口安全停机边界继续有效。

## 历史运行记录

2026-11-16, Asia/Shanghai. Original source workspace had no Git metadata; publishing checkout preserves the existing GitHub history. No real-hardware execution. Read AGENTS.md first.

## Documentation consolidation and next-route requirement

2026-11-16: user requested one concise root README combining runtime instructions, actual progress, limitations and responsibilities. README_V1.zh-CN.md is merged and removed; original official README/SHA256SUMS are byte-for-byte archived in source-metadata/training-package-original/. Root checksum list changes only the original README path. Physical/visual assets and algorithm code are unchanged. Local Git checkout was merged to main b245201 via PR1; the user has now authorized publishing the consolidated README and supplied route illustration. Verify the publication commit from Git.

The supplied image is copied unchanged to docs/images/ordered-route-plan.png and captioned as an early original-layout plan, not a completed run.

User reports organizers permit bypassing the pipe. This changes the proposed next route, not existing code: current ordered route still requires pipe2 and stops under its drop guard. No descent implementation requested. Python3.10.22 remains the validated baseline;3.12 and autonomous coverage exploration are unvalidated. See README.md for concise scope and team responsibilities.

## Accepted V1 scope and latest verified result

2026-11-16: the user explicitly ends V1 at the current safe stop inside circular pipe2. Do not implement the east22cm descent in this delivery. Subsequent pipe exit, far-finish ball/report and official finish remain outside this accepted milestone; do not relabel the runtime as a completed competition mission.

Latest physical evidence: output/entry-ramp-original-start-07, original start(.8,.8), local entry-ramp variant,260sim seconds/311.594wall seconds. Autonomy passed ascent, descent, S_curve, dead_end and rubble; entered pipe at197.8s, reached the explicit zero-speed exit guard at203.0s, and remained stopped inside. Final one-second net speed0.000395m/s; continuous2ms audit:0obstacle contacts, no fall, no water, no bounds violations. Runtime ended at its260s time limit, mission_complete=false/full_course_pass=false. All recording flags off. Small portable evidence is in docs/validation/entry-ramp-original-start-07/; large local output logs are excluded from Git.

Local entry scene selects configs/local-entry-ramp-route.json: flat approach alignment, bounded26degree classification region(actual ramp18.7degrees), .22m/s pipe cap, full oriented-body observed corridor planning,20cm waypoint tolerance and short safe near-goal fallback. At pipe entry a sensor-only terrain submap removes accumulated height-reference error while retaining the global map/water evidence. DWA uses accepted scan motion in the corridor. Ground/roof separation preserves absolute headroom; special-region MPC uses measured per-foot surfaces,1.5Hz/75% stance and longitudinal scan feedback. Original2.5Hz MPC remains elsewhere; global14degree/8cm limits and all measured obstacle/support/water checks remain. No truth navigation or root state overwrite. Official files and original-route capability guards remain unchanged.

Final full suite:117passed in47.03s; replacement checkout repeated117passed in70.38s. Compile, pip consistency and original SHA checks passed; GUI4s auto-close/camera smoke passed. No hardware motion or active simulation. See docs/ENTRY_RAMP_NAVIGATION.md.

Repository delivery: user requested replacing the existing GitHub unitree_map_nav contents through a history-preserving merge. Source workspace remains intact; staging checkout is /home/l/下载/unitree_map_nav on replace/go2-rescue-v1-20261116, based on main d7a7d1fdd4bed9f4fee8b00dc5a56b69af3250f6. Virtualenv, output, recordings and caches are not deliverables.

## Prior user-authorized local map + front camera snapshot

User now explicitly requested moving the east pipe2 ramp to the west entry. This supersedes the prior static-map restriction ONLY for the labelled local variant. Source SHA-covered assets unchanged. task_scene(pipe_ramp=entry) recompiles one180degree world transform around tube2 centre; full other-geometry/expected-transform vertex audit, provenance geometry_changes list. --task-fixture defaultsentry; --pipe-ramp official retains previous static layout, no-fixture still originalMJB. AGENTS/DECISIONS updated. Never count modified geometry as official full_course_pass.

West ramp spans aboutx7.95→8.60 at y6.22, rises0→.22m; original east ramp gone, east exit now22cm drop. Original slope/dimensions unchanged. Pipe autonomous capability guard remains; not a validated whole-pipe route. docs/LOCAL_MAP.md.

--viewer now shows real10Hz navigation camera inset bydefault, up to480x360 display, BGR→RGB, resized with viewport; --hide-camera disables. No additional camera render/video/raw/physics/snapshot/archive enabled. Handle.set_images uses existing frame. Actual screen visually inspected.8s pipe-approach scripted stage entry-ramp-camera-01:75camera updates,433syncs,8.008wall,no fall/recordings; original-start autonomous entry-ramp-autonomy-01:12sim/17.935wall under concurrent tests,115camera updates,562syncs,no fall/recording. Neither is pipe/full-course proof.

Frozen-view Ctrl+C preview first exited139; fixed teardown by clearing overlays and waiting for pinned3.17 Handle's Simulate weakref to expire after close(exit request), preserving model/image lifetimes through render-thread destruction. entry-ramp-close-01 Ctrl+C and autonomy-01 automatic close both exit0.107tests43.50s plus resized-inset test passed; SHA/pip/compile passed. No active simulation. GUI previewXWD/PNG in/tmp was manual QA only, automatic image output remains off.

## Latest dead-end response and pipe geometry audit

2026-11-15. Explicit reverse ordered segments now use separate max_reverse_vx(sim default.25m/s, hardware.08); --max-reverse-speed overrides independently. Ordinary ordered nudges/SE2 recovery keep.08, margin escape.05, marked portal reverse.15. Retain original velocity sample endpoints and all body/support/water/braking checks. Development trial02 changed endpoints and fell on ascent25.518s; final restored the original sampling and constrained speed to intended reverse segments.

Frontal evidence collected during approach,3consecutive scans; inspection region.25m(previous.15), stationary window averaging plus pose stability,1s hold retained. Root cause of slow hold:millimetre jitter across arrival boundary continually reset its timer. checkpoint_near() supplies shared2cm hold hysteresis to Course and Autonomy; large motion still resets. Diagnostics log measured twist and front confirmation.

output/deadend-speed-final:original start130sim/132.888wall, passed ascent/descent/S/dead_end/both return portals, continued to rubble waypoint6.0diagnostic obstacle/water/bounds/fall, all recordingFalse, no video/images/physics/scene binary. User baseline run-20261115-222501:dead_end69.9→141.1s(71.2s); final69.9→108s(38.1s), first retreat7.8→1.9s, starts reversing83.4→75.2s. Front confirmed72.9s. NOT full-course/hardware validation.105tests43.44s, SHA/pip/compile passed; no active simulation. docs/DEAD_END.md.

Original pipe geometry read-only audit:official MJB equals training-package-v8.zip entry bySHA256. Outside floor0, inside flat insert.2200000025m, actual22cm curb belongs to supplied official model. docs/PIPE2_MODEL_AUDIT.json. User baseline reachedpipewp0 and stopped intentionally at8cm capability guard; autonomous22cm traversal still unintegrated. Do not lower source geometry or imply this reverse patch solves the pipe.

Earlier output directories(including pipe2-stage-10 and smooth-junction-final) are absent from the current working tree; their descriptions below are historical records, not presently inspectable physical evidence. Use current user baseline and deadend-speed-final for latest directly available measurements.

## Latest performance and S-junction fix

2026-11-15 late session: video/raw sensor capture/physics archives/map-image snapshots now explicit opt-ins. Default run retains only small config/provenance/route/runtime/trajectory/summary files; no runtime-scene.mjb export unless --save-physics or --snapshots. Frame history removed; rolling timings bounded. Viewer sync decoupled from10Hz video, wall-clock cap60Hz. Camera remains10Hz for navigation. Simulation scopes BLAS to1thread/restores on exit; batched ground medians preserve original results. threadpoolctl3.6.0 added to simulation extras/lock.

Current physics/MPC/navigation/OpenCV are CPU; actual GL_RENDERER on this RTX3060 machine is NVIDIA GeForce RTX3060 Laptop GPU. smooth-viewer-02 actual GUI smoke:8sim/8.010wall,434syncs(~54.2/s), no recordings/fall. Not autonomous or a60fps guarantee.

User fast-20261115-215744/220055 runs stalled at S_curve waypoint4: safe-turn rejected and alignment region too small. Route prior moves that alignment point25cm earlier to the bay; 1.5s turn sweep no longer overshoots requested heading. Marked portal margin restored to min(safety_margin,.02), full body/foot-water checks remain. Official static map unchanged. smooth-junction-01 original-start120sim/106.591wall passed S and dead-end enter/exit; smooth-junction-02 intermediate version passed both return-portal points and visited rubble, then deliberately interrupted151.8sim for final GUI verification,0diagnostic incidents. Final GUI smooth-junction-final PASSED the requested junction segment:155sim/167.889wall,5067syncs(mean30.18/s), completed ascent/descent/S/dead_end and both return portals(145.3/148.4s), reached rubble waypoint3.0diagnostic obstacle/water/bounds/fall; stopped by155s time limit, NOT whole-course success. All recording flagsFalse, no media/physics/scene binary;1.24MB small diagnostics. No active simulation. Full suite99passed42.06s; pip/compile/SHA checks passed. docs/PERFORMANCE.md.

## Latest speed change

User requested about0.5m/s. Simulation defaults now max_vx=.5 and passage_max_vx=.5; --max-speed overrides both, explicit config-file speeds are honored. Hardware defaults .3/.08 and calibration gating unchanged. DWA lookahead includes braking distance(up to1.2m), retaining body/ground/hazard/braking constraints. Original2.5Hz MPC gait and torque limits remain; experimental fast gait/force bounds were removed.

speed-05-flat-05 and speed-05-u-exit-02:8sim-second scripted controller stages with continuous2ms evaluation, actual cruise means .596/.582m/s, no contacts/fall/water/bounds and stopped net speeds .036/.012m/s. speed-check.json files pass. These are not autonomous missions. speed-05-autonomy-01:30s original-start Autonomy, commands .48-.50, ascent stage complete26.2s, actual ascent/platform contact, no incidents; stopped at time limit in L1_descent, not a full pass. First straight-east tests01/02/03 approached an official curved wall and fell; preserve raw failures, do not blame the gait from that unsuitable strip.

Final suite96passed36.38s. docs/SPEED.md has commands, limits and evidence. The earlier speed-only change preserved a passage margin expression min(safety_margin,.2); the latest junction fix supersedes it with .02. Curb entry clip remains[-.3,.45], NOT the .045 used in the historical clean tube test. This faster curb clip has not been validated and ordinary --max-speed does not integrate or bypass curb capability. No active simulation.

## Historical full-course requirement (superseded by accepted V1 scope above)

The user explicitly requires CIRCULAR TUNNEL2. Required route: original start before ramp(0.8,0.8,yaw0) → L1 ascent/platform/descent → S bend → dead-end inspection/reverse exit → rubble region avoiding piles → tunnel2 west-to-east including22cm curb/descending exit → far-finish ball/report/stop. Official static geometry must remain unchanged. Real factory SDK and MuJoCo pinned go2-convex-mpc remain separate backends. No ROS2/RL/training added; unknown hardware calibrated=false.

configs/official-course-route.json now includes the required tunnel2 approach/traversal/exit and declares22cm step capability, measured elevation/distance evidence. This is an explicit manually authored fixed-public-map prior, not unknown-layout generalization. The read-only evaluator specifically audits tube2 floor contact and west-to-east traversal; tunnel3 cannot pass. It also audits body/leg/roof contacts. Foot spheres FL/FR/RL/RR are attached to calf bodies, not the fixed *_foot frames.

## Pipe2 work

src/go2_rescue/step_terrain.py and ConvexMPC.enable_curb() implement opt-in sensor-only curb detection, lifted-then-translated swings, different front/rear placement timing and body-height/pitch references. Final candidate uses1Hz diagonal gait; initial slow single-leg crawl failed. The historical clean candidate capped entry vx0.045m/s(current workspace clip0.45 is unvalidated); after entry returns to standard gait and lidar support-plane adaptation(up to30degrees only in this opt-in mode). There is no plant-root pose/velocity input, runtime pose overwrite or root force.

scripts/test_pipe2_controller.py is an explicitly labelled PARTIAL-STAGE controller diagnostic: reset once at(8.1,6.22,.30), scripted forward intent plus measured heading/joint-odometry feedback and estimated-distance stopping; NOT Autonomy, original start, random layouts or hardware. It produces actual physics/video. Truth is rendering/read-only evaluation/stop checks only.

- pipe2-stage-01/02/04/05 fell;03 entered but did not exit before95s. Preserve failures.
-06 exited at55s;07 additionally stopped at61.4s. These lacked continuous collision scoring; their10Hz offline contact audit is insufficient for a clean-pass claim.
-08 exited and stopped,61.402sim seconds, no fall/water/bounds, but continuous2ms contact checking caught one short leg contact at22.832s(tunnel2_segment_16):stage_pass=false.09 still had that contact despite earlier hind placement. Raw results stay unchanged.
-10 PASSED the isolated stage:62.102sim seconds,107.464wall seconds under concurrent test load, entered/traversed/exited and stopped; continuous2ms audit0obstacle contacts/0water/0bounds/no fall. stage_pass=true, full_course_pass=false. Hind lead-.085(previous09-.055,08-.025). This is one specified reset/phase/noise case, not multi-pose robustness or autonomous traversal. Video/physics/source snapshots/provenance archived for10. See docs/PIPE2.md.

IMPORTANT: autonomous step traversal is NOT INTEGRATED. Default8cm navigation capability remains conservative. At pipe stage Autonomy returns blocked/ordered_pipe2_requires_validated_step_capability. It does not silently route through3, clear the curb map, or enlarge global terrain limits. Do not tell the user the default whole-course command now passes. Remaining necessary work is robust entry/exit over pose/phase variations, a collision-checked3D navigation/locomotion interface, then a fresh original-start full run.

## Historical full-route evidence

output/ordered-course-13 traversed the OLD TUNNEL3 route from original start in475.102sim/981.808wall seconds: seven old stages, then finish;0old-evaluator obstacle/water/bounds/fall. Raw strict full_course_pass=false because reports had0.06897m and0.96256m errors. This is NOT evidence for required tunnel2 and must not be rewritten.

Tracker caps reports at a positive declared target_quota, unknown count0 unlimited. output/ordered-course-13-report-replay/report.json replays226finish frames with archived estimated poses, reproduces both original reports, corrected single report error0.06897m; state/return mismatches0. It is component replay, not a new physical/full Autonomy run. This does not solve unknown-count visual false positives or guarantee first-report correctness.

ordered-course-14 was interrupted on user confirmation of mandatory2, during L1_descent; raw logs saved. It is not a completed post-report-patch full run.

## Existing navigation and limitations

Point-to-line ICP with optional stable start-relative IMU heading; overhead planes excluded ONLY from planar matching, retained in full3D mapping. Coverage/inlier-aware quality avoids undercounting intentional ceiling removal. Unknown hardware does not enable stable-yaw assumptions.

Ramp/S/dead-end grid0.10m; at rubble rebuild0.05m from sensor keyframes/estimated poses. Historical U-exit narrow mode used0.02m margin and0.08m/s; current workspace uses the restored2cm portal margin described above and configurable forward cap(defaultsim0.5), preserving physical body/water checks. Normal extra margin0.05m resumes afterward. Pose-region SE(2) recovery, bounded buffer-only escape, transit gates and strict checkpoint headings remain.

Ramp recovery is slow and estimated ground consistency is imperfect. Historic full-run navP95.1853s, so end-to-end10Hz is not established. No80%success, formal scoring, world-coordinate protocol, hardware calibration or native motion validation.

## Checks and entry points

Latest full suite:117passed47.03s after entry-ramp integration;105passed43.44s after the dead-end response change;96passed36.38s was the speed change;90passed68.63s was the earlier tube-stage milestone. No active simulation. Official SHA256SUMS, pip check and compileall passed this turn. Scene SHA256:2dd875d238040d9a8616869cb2e30d00abf131b48b89b011af6dc10bd264089c.

README.md, docs/PIPE2.md, docs/COMPETITION_V1.md, docs/ARCHITECTURE.md, docs/VALIDATION.md. Use .venv/bin/python from project root and new output directories. Never label a custom-reset stage as full-course success.
