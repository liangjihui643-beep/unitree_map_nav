#!/usr/bin/env python3
"""最小完赛：官方 MJB 赛道上，用行走后端 + 轻量导航走到终点区并发结束信号。

对应赛题一句话："自己走、别撞、到终点、发结束"。探索/重规划/全覆盖/
相机识别按赛前约定暂不实现——本脚本用"合成激光 + 占据栅格 + A* + 纯
追踪(带避障钳制)"的最小导航（代码全部来自已就位的 stack/）。

    cd official/go2_stack
    PYTHONPATH=.:../../mujoco_env:../../mujoco_env/stack python minimal_course.py
    PYTHONPATH=. ... minimal_course.py --backend crawl --viewer
    PYTHONPATH=. ... minimal_course.py --backend unitree_rl --policy /path/policy.pt
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'mujoco_env'))
sys.path.insert(0, str(HERE.parents[1] / 'mujoco_env/stack'))

from official_env import OfficialGo2Env      # noqa: E402
from backends import CrawlBackend, UnitreeRLBackend   # noqa: E402
from mapping import OccupancyGrid            # noqa: E402
from planning import astar                   # noqa: E402
from local_nav import LocalNav, _wrap        # noqa: E402


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# 终点区（由 MJB finish_* 隔断墙几何推得：x>9.1 的东北室，入口在西墙 8.4~8.9 缺口）
FINISH_CENTER = (10.6, 8.7)
FINISH_BOX = (9.3, 7.0, 11.9, 9.9)     # x0, y0, x1, y1


def in_finish(pos):
    return FINISH_BOX[0] <= pos[0] <= FINISH_BOX[2] and FINISH_BOX[1] <= pos[1] <= FINISH_BOX[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='crawl', choices=['crawl', 'unitree_rl'])
    parser.add_argument('--policy', default=None, help='unitree_rl 后端策略(.pt/.onnx)')
    parser.add_argument('--kp', type=float, default=None)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--time', type=float, default=600.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    kp = args.kp if args.kp else (20.0 if args.backend == 'unitree_rl' else 50.0)
    kd = 0.5 if args.backend == 'unitree_rl' else 3.5
    env = OfficialGo2Env(kp=kp, kd=kd)
    obs, info = env.reset()
    if args.backend == 'crawl':
        backend = CrawlBackend(env)
    else:
        backend = UnitreeRLBackend(env, args.policy)
    grid = OccupancyGrid(-0.5, -0.5, 13.0, 11.0, res=0.1)
    nav = LocalNav(max_vx=0.15, max_wz=0.4, lookahead=0.4)
    print('=' * 60)
    print('最小完赛演示 | 后端=%s kp=%.0f kd=%.1f | 官方 MJB 赛道' % (backend.name, kp, kd))
    print('=' * 60)

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [6.0, 5.0, 0.3]
        viewer.cam.distance = 15.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -50.0

    rng = np.random.default_rng(args.seed)
    self_print_t = type('T', (), {})()
    path, path_t, replan_t = None, 0.0, 0.0
    stuck_t, unstick_until = -9.0, 0.0
    falls = 0
    finished = False
    sig_path = HERE / 'end_signal.json'
    t0 = time.time()
    try:
        while env.data.time < args.time:
            pos, quat, _ = env._base_state()
            px, py, yaw = pos[0], pos[1], yaw_of(quat)
            # 建图（10Hz，契约激光水平环）
            if obs['lidar'] is not None:
                grid.update(px, py, yaw, [float(v) for v in obs['lidar'][4]])
            # 终点检测 -> 结束信号
            if in_finish(pos):
                finished = True
                sig = {'task': 'rescue', 'signal': 'END', 'time': env.data.time,
                       'position': [float(px), float(py)], 'falls': falls}
                sig_path.write_text(json.dumps(sig, indent=2))
                print('[t=%.1fs] >>> 到达终点区，发出结束信号: %s' % (env.data.time, sig))
                break
            # 规划/重规划（每 4s 或路径走完）
            if path is None or env.data.time - replan_t > 4.0 or \
                    (path and math.hypot(path[-1][0]-px, path[-1][1]-py) > 2.0):
                path = astar(grid, (px, py), FINISH_CENTER, inflation=0.22)
                replan_t = env.data.time
                if path is None and env.data.time - getattr(self_print_t, 'v', -9) > 2.0:
                    self_print_t.v = env.data.time
                    print('[t=%.1fs] 终点暂不可达（建图 %.0f%%），尝试脱困' % (
                        env.data.time, grid.known_fraction() * 100))
            # 局部导航 -> 速度指令
            ring = [float(v) for v in (obs['lidar'][4] if obs['lidar'] is not None
                                       else [float('nan')] * 72)]
            if path is None:
                stuck_t = env.data.time
                if env.data.time - unstick_until > 6.0:
                    unstick_until = env.data.time + 2.0   # 脱困：旋转+后撤换视野
            if env.data.time < unstick_until:
                vx, wz = -0.06, 0.5
            elif path:
                vx, wz, target = nav.cmd(path, px, py, yaw, ring)
            else:
                goals = grid.frontier_goals(px, py)
                if goals:
                    gp = astar(grid, (px, py), goals[0], inflation=0.22)
                    if gp:
                        path = gp
                    vx, wz = 0.12, 0.0
                else:
                    vx, wz = 0.12, 0.0     # 盲目前探（换视野）
            backend.set_cmd(vx, 0.0, wz)
            action = backend.action(obs)
            obs, r, done, trunc, info = env.step(action)
            if done:
                falls += 1
                print('[t=%.1fs] 跌倒 #%d，原地恢复' % (env.data.time, falls))
                env.recover()
                backend.on_reset()
                obs, info = env._obs(), env._info()
                path = None
                if falls > 80:
                    print('跌倒次数过多，终止')
                    break
            if viewer:
                viewer.sync()
                time.sleep(0.004)
    finally:
        if viewer:
            viewer.close()
        from PIL import Image
        img = grid.to_image()
        fx, fy = grid.to_cell(*FINISH_CENTER)
        if 0 <= fx < grid.w and 0 <= fy < grid.h:
            img[grid.h - 1 - fy, fx] = (0, 200, 0)
        Image.fromarray(img).save(HERE / 'minimal_course_map.png')
        env.close()
    print('=' * 60)
    print('结果: %s | 用时 %.0fs | 跌倒 %d 次 | 地图已知 %.0f%% | 加速比 %.0f%%' % (
        '完赛' if finished else '未完成', env.data.time, falls,
        grid.known_fraction() * 100,
        100 * env.data.time / max(time.time() - t0, 1e-6)))
    print('地图: %s（绿=终点）' % (HERE / 'minimal_course_map.png'))
    return 0 if finished else 1


if __name__ == '__main__':
    raise SystemExit(main())
