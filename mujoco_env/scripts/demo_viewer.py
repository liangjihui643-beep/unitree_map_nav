#!/usr/bin/env python3
"""MuJoCo 可视化演示：打开交互窗口，展示当前环境的完整运行。

流程：随机化场地 → Go2 起身站立 → 对角小跑步态尝试前进 →
跌倒检测 / 越界检测 → 自动复位。暂无训练好的策略，步态为手调
正弦 PD（演示用），正式策略请走 RL 训练 + ONNX 导出链路。

用法：
    PYTHONPATH=. python scripts/demo_viewer.py [--no-randomize] [--duration 30]
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from go2_arena_env import Go2ArenaEnv


def trot_gait(phase, stand, joint_names):
    """手调对角小跑：thigh/calf 正弦调制。纯演示，非训练策略。"""
    q = stand.copy()
    freq = 2.0                                   # Hz
    s = np.sin(2 * np.pi * freq * phase)
    c = np.cos(2 * np.pi * freq * phase)
    for i, name in enumerate(joint_names):
        low = name.lower()
        leg_sign = 1.0 if low.startswith(('fr', 'rl')) else -1.0   # 对角同相
        if 'thigh' in low:
            q[i] += 0.22 * s * leg_sign + 0.05
        elif 'calf' in low:
            q[i] += -0.18 * max(0.0, c) * leg_sign - 0.05 * max(0.0, -s)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--duration', type=float, default=40.0)
    parser.add_argument('--trot', action='store_true',
                        help='实验性：尝试手调小跑（无训练策略，大概率很快摔倒）')
    parser.add_argument('--cycle', type=float, default=8.0,
                        help='站立展示多少秒后换一回合新布局')
    args = parser.parse_args()

    env = Go2ArenaEnv(seed=None, randomize=not args.no_randomize)
    obs, info = env.reset()
    stand = env._stand.copy()

    print('=' * 56)
    print('MuJoCo 可视化演示（当前无训练策略，步态为手调 PD 演示）')
    print('关节顺序:', env.joint_names)
    print('=' * 56)

    stand_steps = int(2.0 / env.ctrl_dt)
    cycle_steps = int(args.cycle / env.ctrl_dt)
    phase = 0.0
    state = 'trot' if args.trot else 'showcase'
    falls = 0
    resets = 0
    report_step = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [3.0, 4.0, 0.3]
        viewer.cam.distance = 9.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -35.0
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < args.duration:
            if state == 'showcase':
                action = stand
                if env._step_count >= cycle_steps:
                    print('[t=%.1fs] 本回合结束，随机化换一回合（看碎石/足球换位）'
                          % env.data.time)
                    env.reset()
                    phase = 0.0
            elif state == 'stand':
                action = stand
                if env._step_count >= stand_steps:
                    state = 'trot'
                    print('[t=%.1fs] 起身完成，开始小跑前进（手调演示步态）' % env.data.time)
            else:
                phase += env.ctrl_dt
                action = trot_gait(phase, stand, env.joint_names)

            obs, reward, done, trunc, info = env.step(action)

            report_step += 1
            if report_step % 100 == 0:
                lidar_txt = '无' if obs['lidar'] is None else '%d/648' % np.isfinite(obs['lidar']).sum()
                rgb_txt = '无新帧' if obs['rgb'] is None else '%s' % (obs['rgb'].shape,)
                print('t=%5.1fs pos=(%.2f, %.2f, %.2f) lidar=%s rgb=%s 距终点 %.1fm%s' % (
                    env.data.time, obs['base_pos'][0], obs['base_pos'][1], obs['base_pos'][2],
                    lidar_txt, rgb_txt, info['dist_to_finish'],
                    ' [进球区!]' if info['in_finish'] else ''))

            if done:
                falls += 1
                print('[t=%.1fs] 倒地/越界，复位重开（第 %d 次）' % (env.data.time, falls))
                env.reset()
                state = 'stand' if args.trot else 'showcase'
                phase = 0.0
                resets += 1
            viewer.sync()
            time.sleep(max(0.0, env.ctrl_dt * 0.5 - (time.time() % env.ctrl_dt)))

    env.close()
    print('演示结束：复位 %d 次。提示：正式参赛需训练策略（RL/模仿学习）并导出 ONNX。' % resets)


if __name__ == '__main__':
    main()
