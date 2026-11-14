#!/usr/bin/env python3
"""全流程演示：建图 -> 探索 -> 避障 -> 识别报位 -> 终点 -> 结束信号。

对应 Gazebo 侧 ./run_competition.sh explore 的 MuJoCo 移植版，
底层行走为传统控制四拍步态（PPT 允许路线）。

    PYTHONPATH=.:stack python stack/run_stack.py            # 无头跑全程
    PYTHONPATH=.:stack python stack/run_stack.py --viewer   # 带可视化窗口
    PYTHONPATH=.:stack python stack/run_stack.py --seed 3 --time 180
"""
import argparse
import time
from pathlib import Path

import numpy as np

import gait  # noqa: F401  (确保 sys.path 下有 stack/)
from go2_arena_env import Go2ArenaEnv
from rescue_task import RescueTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time', type=float, default=240.0)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--no-randomize', action='store_true')
    args = parser.parse_args()

    env = Go2ArenaEnv(seed=args.seed, randomize=not args.no_randomize,
                      kp=100, kd=5, render_camera=True)
    obs, info = env.reset()
    task = RescueTask(env, time_limit=args.time)
    print('=' * 60)
    print('MuJoCo 救援全流程: 建图/探索/避障/识别/报位/终点/结束信号')
    print('机器人 Go2 | 地图分辨率 0.1m | 控制 50Hz | 传感契约按 PPT')
    print('=' * 60)

    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [4.0, 4.0, 0.3]
        viewer.cam.distance = 11.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -40.0

    map_path = Path(__file__).parent / 'map_result.png'
    t0 = time.time()
    done_flag = False
    try:
        while env.data.time < args.time:
            action = task.action(obs)
            obs, reward, done, trunc, info = env.step(action)
            if done:
                task.on_fall()
                obs, info = env._obs(), env._info()
                if task.falls > 60:      # 防死循环
                    print('跌倒次数过多，终止')
                    break
            if task.state == 'END' and env.data.time > task._end_hold + 2.0:
                done_flag = True
                break
            if task.state == 'END' and not hasattr(task, '_end_hold'):
                task._end_hold = env.data.time
            if viewer:
                viewer.sync()
                time.sleep(0.008)
    finally:
        if viewer:
            viewer.close()
        from PIL import Image
        img = task.grid.to_image()
        # 标注终点与上报点
        fx, fy = task.grid.to_cell(*task.finish)
        if 0 <= fx < task.grid.w and 0 <= fy < task.grid.h:
            img[task.grid.h - 1 - fy, fx] = (0, 200, 0)
        for (ex, ey), _ in task.reports:
            cx, cy = task.grid.to_cell(ex, ey)
            if 0 <= cx < task.grid.w and 0 <= cy < task.grid.h:
                img[task.grid.h - 1 - cy, cx] = (255, 0, 0)
        Image.fromarray(img).save(map_path)
        env.close()
    ok = task.summary()
    print('地图已保存: %s（绿=终点 红=上报的生命体位置）' % map_path)
    print('仿真 %.1fs 实际耗时 %.1fs（MuJoCo 无渲染加速比约 %.0f%%）' % (
        env.data.time, time.time() - t0, 100 * env.data.time / max(time.time() - t0, 1e-6)))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
