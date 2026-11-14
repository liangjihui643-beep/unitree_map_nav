#!/usr/bin/env python3
"""最小冒烟测试：环境加载、复位、步进、传感器形状、渲染一帧。"""
import time

import numpy as np

from go2_arena_env import Go2ArenaEnv


def main():
    t0 = time.time()
    env = Go2ArenaEnv(seed=0, randomize=True)
    print('模型加载 %.1fs（nu=%d, nq=%d, nv=%d）' % (time.time() - t0,
          env.model.nu, env.model.nq, env.model.nv))
    print('关节顺序:', env.joint_names)

    obs, info = env.reset()
    checks = {
        'base_quat': (obs['base_quat'], (4,)),
        'gyro': (obs['gyro'], (3,)),
        'joint_pos': (obs['joint_pos'], (12,)),
        'joint_vel': (obs['joint_vel'], (12,)),
        'lidar': (obs['lidar'], (9, 72)),
    }
    for name, (value, shape) in checks.items():
        assert value is not None and tuple(value.shape) == shape, (name, None if value is None else value.shape)
    print('lidar 有效回波: %d/648, 最近 %.2f m' % (
        np.isfinite(obs['lidar']).sum(), np.nanmin(obs['lidar'])))
    print('reset info:', {k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in info.items()})

    # 站立 2 s（PD 跟踪默认站立角），再前进试探 2 s
    t0 = time.time()
    stand = env._stand.copy()
    n_hold = int(2.0 / env.ctrl_dt)
    fallen = False
    for i in range(n_hold):
        obs, r, term, trunc, info = env.step(stand)
        fallen = fallen or term
    print('站立 %.1fs: fallen=%s, base_z=%.3f, 用时 %.2fs' % (
        n_hold * env.ctrl_dt, fallen, obs['base_pos'][2], time.time() - t0))

    if env.renderer is not None:
        frame = env._rgb()
        if frame is None:
            env.step(stand)
            frame = env._rgb()
        if frame is not None:
            print('rgb 帧: shape=%s dtype=%s 均值=%.1f' % (
                frame.shape, frame.dtype, frame.mean()))
        else:
            print('rgb 帧: 无（本拍非相机节拍）')

    env2 = Go2ArenaEnv(seed=None, randomize=False, render_camera=False)
    obs2, _ = env2.reset()
    assert obs2['rgb'] is None and obs2['lidar'] is not None
    print('无渲染模式 OK')
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
