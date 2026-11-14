#!/usr/bin/env python3
"""救援任务编排器：探索 -> 识别/报位 -> 终点 -> 结束信号。

把 Gazebo 侧的整套自主流程（explore_lite 前沿探索 + GlobalPlanner
全局规划 + DWA 局部避障 + 识别报位）移植到 MuJoCo 环境，底层行走
用 stack/gait.py 的传统控制四拍步态（PPT 允许路线）。

状态机：EXPLORE（前沿目标探索建图） -> REPORT（发现生命体上报位置）
       -> EXPLORE -> FINISH（前往终点区） -> END（发出结束信号）
跌倒处理：recover() 原地站起（演示脚手架，评测不可用），继续当前任务。
"""
import math
import time

import numpy as np

from gait import TrotGait
from local_nav import LocalNav
from mapping import OccupancyGrid
from perception import detect_ball, estimate_position
from planning import astar


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class RescueTask:
    def __init__(self, env, time_limit=240.0, verbose=True):
        self.env = env
        self.verbose = verbose
        cfg = env._cfg
        a = cfg['arena']
        x0, y0 = a['minimum']
        self.grid = OccupancyGrid(x0 - 1, y0 - 1, a['size'][0] + 2, a['size'][1] + 2)
        self.nav = LocalNav()
        self.gait = TrotGait(env, freq=2.0)
        self.finish = tuple(cfg['finish']['center'])
        self.time_limit = time_limit
        self.state = 'EXPLORE'
        self.path = None
        self.goal = None
        self.cmd = (0.0, 0.0)
        self.falls = 0
        self.reports = []            # [(估计xy, 误差m)]
        self._slow = 0
        self._stand_ticks = int(1.0 / env.ctrl_dt)

    # ------------------------------------------------------------- 主控
    def action(self, obs):
        """每个控制周期(50Hz)调用，返回关节目标。"""
        env = self.env
        pos, quat, gyro = env._base_state()
        yaw = yaw_of(quat)
        self._slow += 1
        if self._slow % 5 == 0 and obs is not None:
            self._think(obs, pos, yaw)
        if self._stand_ticks > 0:
            self._stand_ticks -= 1
            return env._stand
        return self.gait.step(self.cmd, quat, gyro, env.ctrl_dt)

    def _think(self, obs, pos, yaw):
        """10Hz 慢回路：建图、检测、状态机、重规划。"""
        env = self.env
        px, py = pos[0], pos[1]
        if obs['lidar'] is not None:
            ring = obs['lidar'][4]           # 0° 水平环
            self.grid.update(px, py, yaw, [float(r) for r in ring])
        # 识别（任务2）
        det = detect_ball(obs['rgb'])
        if det and self.state != 'END':
            bearing, area = det
            est = estimate_position(px, py, yaw, bearing,
                                    [float(r) for r in (obs['lidar'][4] if obs['lidar'] is not None else [])])
            if est and area > 120:
                self._report(est)
        # 状态机
        if self.state == 'EXPLORE':
            if self.env.data.time > self.time_limit * 0.7:
                self._set_state('FINISH')
            elif self.path is None or self._near_goal(px, py):
                self._plan_explore(px, py)
        elif self.state == 'PROBE':
            # 封闭口袋：直线向前试探（如爬坡），换取新视野
            self.cmd = (0.15, 0.0)
            if self.env.data.time > self._probe_end:
                self._set_state('EXPLORE')
        elif self.state == 'FINISH':
            if self._dist(px, py, *self.finish) < 0.9:
                self._set_state('END')
            elif self.path is None or self._near_goal(px, py) \
                    or (self.goal and self._dist(px, py, *self.goal) > 1.5):
                self._plan(px, py, self.finish)
        elif self.state == 'END':
            self.cmd = (0.0, 0.0)
            return
        # 局部导航
        if self.path:
            ring = [float(r) for r in (obs['lidar'][4] if obs['lidar'] is not None else [float('nan')] * 72)]
            vx, wz, target = self.nav.cmd(self.path, px, py, yaw, ring)
            self.cmd = (vx, wz)
        else:
            self.cmd = (0.0, 0.0)

    # ------------------------------------------------------------- 状态
    @staticmethod
    def _dist(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def _near_goal(self, px, py):
        return self.goal is not None and self._dist(px, py, *self.goal) < 0.5

    def _set_state(self, s):
        if self.verbose:
            print('[t=%.1fs] 状态切换: %s -> %s' % (self.env.data.time, self.state, s))
        self.state = s
        self.path = None
        self.goal = None

    def _report(self, est):
        err = self.env.report_victim(est)
        if err is None:
            return
        self.reports.append((est, err))
        print('[t=%.1fs] 发现生命体! 估计位置 (%.2f, %.2f) 误差 %.2f m（第 %d 次上报）'
              % (self.env.data.time, est[0], est[1], err, len(self.reports)))

    def _plan_explore(self, px, py):
        goals = self.grid.frontier_goals(px, py)
        if not goals:
            # 起点被坡道/围墙围成封闭口袋时前沿为零，向前试探解锁视野
            self._probe_end = self.env.data.time + 1.6
            self._probe_count = getattr(self, '_probe_count', 0) + 1
            if self._probe_count <= 12:
                print('[t=%.1fs] 无可达前沿（封闭区域），向前试探 #%d'
                      % (self.env.data.time, self._probe_count))
                self._set_state('PROBE')
            else:
                self._set_state('FINISH')
            return
        for g in goals[:5]:
            path = astar(self.grid, (px, py), g)
            if path:
                self.path = path
                self.goal = g
                if self.verbose:
                    print('[t=%.1fs] 探索目标 (%.1f, %.1f)，路点 %d 个，地图已知 %.0f%%'
                          % (self.env.data.time, g[0], g[1], len(path),
                             self.grid.known_fraction() * 100))
                return
        self._set_state('FINISH')

    def _plan(self, px, py, goal):
        path = astar(self.grid, (px, py), goal)
        if path is None:
            # 终点不可达（可能未探明）：清掉目标附近障碍再试一次
            print('[t=%.1fs] 规划失败，原地转圈观察' % self.env.data.time)
            self.cmd = (0.0, 0.4)
            return
        self.path = path
        self.goal = goal

    def on_fall(self):
        self.falls += 1
        print('[t=%.1fs] 跌倒 #%d，原地恢复' % (self.env.data.time, self.falls))
        self.env.recover()
        self.path = None          # 重规划
        self._stand_ticks = int(0.8 / self.env.ctrl_dt)

    def summary(self):
        fin = self.state == 'END'
        print('=' * 56)
        print('任务%s' % ('完成：已到终点并发出结束信号' if fin else '未完成'))
        print('用时 %.1fs | 跌倒 %d 次 | 地图已知 %.0f%% | 上报生命体 %d 次'
              % (self.env.data.time, self.falls, self.grid.known_fraction() * 100,
                 len(self.reports)))
        for i, (est, err) in enumerate(self.reports, 1):
            print('  上报 #%d: (%.2f, %.2f) 误差 %.2f m' % (i, est[0], est[1], err))
        print('=' * 56)
        return fin
