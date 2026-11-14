#!/usr/bin/env python3
"""从 arena.yaml 生成 MuJoCo (MJCF) 比赛场地。

与 Gazebo 版 generate_world.py 共用同一份几何配置，保证两个仿真里的
场地尺寸一致。PPT 规则强调"不要依赖固定坐标"：碎石中心、足球生命体
位置支持按回合随机化（--seed）。

用法：
    python generate_arena_xml.py --output arena.xml [--seed N] [--no-victim]
    python generate_arena_xml.py --with-robot robot/go2/go2.xml --output scene.xml
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
ARENA_YAML = ROOT / 'src/unitree_competition/config/arena.yaml'

WALL_RGBA = '0.49 0.51 0.53 1'
TERRAIN_RGBA = '0.68 0.53 0.32 1'
RAMP_RGBA = '0.80 0.66 0.42 1'
SBEND_RGBA = '0.26 0.55 0.61 1'
DEADEND_RGBA = '0.68 0.32 0.28 1'
PIPE_RGBA = '0.55 0.58 0.65 1'
CULVERT_RGBA = '0.28 0.56 0.48 1'
ROCK_RGBA = '0.45 0.42 0.38 1'
START_RGBA = '0.15 0.45 0.85 1'
FINISH_RGBA = '0.15 0.65 0.35 1'
PUDDLE_RGBA = '0.20 0.45 0.75 0.55'
BALL_RGBA = '0.95 0.45 0.05 1'  # 安全橙：训练期识别占位色（评测以官方为准）
# 生命体候选出现区（训练环境自定义，评测时官方随机）
VICTIM_ZONES = [(7.2, 2.2), (6.4, 1.3), (8.6, 3.4), (7.4, 5.6), (9.9, 2.6)]


def el(parent, tag, **attrs):
    node = ET.Element(tag) if parent is None else ET.SubElement(parent, tag)
    for key, value in attrs.items():
        node.set(key, str(value))
    return node


def box(parent, name, center, size, rgba, euler=None, visual_only=False, **extra):
    geom = el(parent, 'geom', name=name, type='box', size='%.4f %.4f %.4f' % size,
              pos='%.4f %.4f %.4f' % center, rgba=rgba, **extra)
    if euler is not None:
        geom.set('euler', '%.6f %.6f %.6f' % euler)
    if visual_only:
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
        geom.set('group', '2')
    return geom


def wall(parent, name, a, b, height, thickness, bottom=0.0, rgba=WALL_RGBA):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    return box(parent, name,
               ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, bottom + height / 2),
               (length / 2, thickness / 2, height / 2), rgba, euler=(0, 0, yaw))


def build_arena(cfg, rng, with_victim=True):
    """返回 arena-only 的 MJCF <mujoco> 根元素。"""
    a = cfg['arena']
    x0, y0 = a['minimum']
    sx, sy = a['size']
    t, wh = a['wall_thickness'], a['wall_height']

    muj = el(None, 'mujoco', model='rescue_arena')
    # cone/impratio 与官方 unitree_mujoco 的 Go2 配置一致（elliptic/100），
    # 否则足底接触行为改变，机器人容易滑倒
    opt = el(muj, 'option', timestep='0.002', gravity='0 0 -9.81',
             cone='elliptic', impratio='100')
    el(opt, 'flag', frictionloss='enable', energy='disable')
    compiler = el(muj, 'compiler', angle='radian', autolimits='true')
    compiler.set('assetdir', str(PKG / 'robot'))


    visual = el(muj, 'visual')
    el(visual, 'headlight', ambient='0.55 0.55 0.55', diffuse='0.85 0.85 0.85',
       specular='0.2 0.2 0.2')
    el(visual, 'rgba', haze='0.15 0.25 0.35 1')
    el(visual, 'global', offwidth='320', offheight='240')

    world = el(muj, 'worldbody')
    el(world, 'light', name='top', pos='2 4 8', dir='-0.2 -0.4 -1',
       directional='true', castshadow='true', diffuse='0.9 0.9 0.9')

    # 地面与围墙（碰撞体）
    box(world, 'floor', (x0 + sx / 2, y0 + sy / 2, -0.05), (sx / 2, sy / 2, 0.05),
        '0.72 0.72 0.75 1')
    wall(world, 'wall_south', (x0 - t / 2, y0 - t / 2), (x0 + sx + t / 2, y0 - t / 2), wh, t)
    wall(world, 'wall_north', (x0 - t / 2, y0 + sy + t / 2), (x0 + sx + t / 2, y0 + sy + t / 2), wh, t)
    wall(world, 'wall_west', (x0 - t / 2, y0), (x0 - t / 2, y0 + sy), wh, t)
    wall(world, 'wall_east', (x0 + sx + t / 2, y0), (x0 + sx + t / 2, y0 + sy), wh, t)

    # 起终点标记（仅视觉）
    for name, key, rgba in (('start', 'start', START_RGBA), ('finish', 'finish', FINISH_RGBA)):
        pad = cfg[key]
        cx, cy = pad['center']
        w, h = pad['size']
        box(world, name + '_pad', (cx, cy, 0.002), (w / 2, h / 2, 0.002), rgba,
            visual_only=True)

    # 坡道系统：上坡(+Y) -> 平台 -> 下坡(+X)。
    # 旋转盒建模：盒长轴沿局部 y（上坡）/ x（下坡），绕对应水平轴俯仰。
    s = cfg['slope']
    run = s['height'] / math.tan(math.radians(s['angle_deg']))
    ux, uy = s['uphill_start']
    dx, dy = s['downhill_start']
    width, high, turn_y = s['width'], s['height'], s['turn_y']
    ang = math.radians(s['angle_deg'])
    embed = high / 2 - 0.25 * math.cos(ang) + 0.02   # 使坡面中点高度=high/2，低端吃进地面
    box(world, 'uphill_ramp', (ux, uy + run / 2, embed),
        (width / 2, run / 2 + 0.10, 0.25), RAMP_RGBA, euler=(ang, 0, 0))
    deck_len = turn_y + width / 2 - (uy + run)
    box(world, 'slope_deck', (ux, uy + run + deck_len / 2, high / 2),
        (width / 2, deck_len / 2, high / 2), TERRAIN_RGBA)
    box(world, 'cross_deck', ((ux + dx) / 2, turn_y, high / 2),
        ((dx - ux) / 2, width / 2, high / 2), TERRAIN_RGBA)
    box(world, 'downhill_ramp', (dx + run / 2, dy, embed),
        (run / 2 + 0.10, width / 2, 0.25), RAMP_RGBA, euler=(0, ang, 0))
    # 坡面浅水洼（仅视觉，环境按 xy 判定"进水洼"）：贴在上坡面中前段
    pud_y = uy + run * 0.55
    pud_z = (run * 0.55) * math.tan(ang) + 0.006
    box(world, 'puddle', (ux, pud_y, pud_z), (width / 2 - 0.05, 0.30, 0.004),
        PUDDLE_RGBA, euler=(ang, 0, 0), visual_only=True)

    # 坡道护栏 + 起点东侧封挡（出发必须上坡）
    wall(world, 'rail_inner', (ux + width / 2 + t / 2, uy),
         (ux + width / 2 + t / 2, turn_y - width / 2 - t), 1.0, t)
    wall(world, 'rail_outer', (ux - width / 2 - t / 2, uy),
         (ux - width / 2 - t / 2, turn_y + width / 2 + t), 1.0, t)
    entry_x = cfg['s_bend']['entry'][0]
    wall(world, 'rail_north', (ux - width / 2, turn_y + width / 2 + t / 2),
         (entry_x, turn_y + width / 2 + t / 2), 1.0, t)
    wall(world, 'rail_south', (ux + width / 2, turn_y - width / 2 - t / 2),
         (entry_x, turn_y - width / 2 - t / 2), 1.0, t)
    wall(world, 'start_partition', (ux + width / 2 + t / 2, y0),
         (ux + width / 2 + t / 2, uy), wh, t)

    # S 弯（圆弧分段墙）
    sb = cfg['s_bend']
    ex, ey = sb['entry']
    radius, clear = sb['radius'], sb['clear_width']
    steps = 40
    for arc, center, begin, end in ((0, (ex, ey - radius), math.pi / 2, -math.pi / 2),
                                    (1, (ex, ey - 3 * radius), math.pi / 2, 3 * math.pi / 2)):
        for side in (-1, 1):
            rr = radius + side * (clear / 2 + t / 2)
            for j in range(steps):
                aa = begin + (end - begin) * j / steps
                bb = begin + (end - begin) * (j + 1) / steps
                p = (center[0] + rr * math.cos(aa), center[1] + rr * math.sin(aa))
                q = (center[0] + rr * math.cos(bb), center[1] + rr * math.sin(bb))
                wall(world, 'sbend_%d_%d_%02d' % (arc, side, j), p, q, sb['wall_height'], t,
                     rgba=SBEND_RGBA)

    # 死胡同（开口 +Y）
    d = cfg['dead_end']
    dex, dey = d['entrance']
    bottom = dey - d['length']
    wall(world, 'dead_left', (dex - d['clear_width'] / 2 - t / 2, bottom),
         (dex - d['clear_width'] / 2 - t / 2, dey), wh, t, rgba=DEADEND_RGBA)
    wall(world, 'dead_right', (dex + d['clear_width'] / 2 + t / 2, bottom),
         (dex + d['clear_width'] / 2 + t / 2, dey), wh, t, rgba=DEADEND_RGBA)
    wall(world, 'dead_end', (dex - d['clear_width'] / 2 - t, bottom - t / 2),
         (dex + d['clear_width'] / 2 + t, bottom - t / 2), wh, t, rgba=DEADEND_RGBA)

    # 碎石（随机化：中心点在配置值附近抖动）
    r = cfg['rubble']
    rock_rng = np.random.default_rng(r['seed'])
    for patch, (cx, cy) in enumerate(r['centers'], 1):
        if rng is not None:
            cx += rng.uniform(-0.4, 0.4)
            cy += rng.uniform(-0.4, 0.4)
        for j in range(r['rocks_per_patch']):
            angle = rock_rng.uniform(0, 2 * math.pi)
            dist = r['patch_radius'] * math.sqrt(rock_rng.random())
            x, y = cx + dist * math.cos(angle), cy + dist * math.sin(angle)
            h = rock_rng.uniform(*r['height_range'])
            s = rock_rng.uniform(0.06, 0.12)
            box(world, 'rock_%d_%02d' % (patch, j), (x, y, h / 2), (s, s * 0.85, h / 2),
                ROCK_RGBA, euler=(0, 0, rock_rng.uniform(0, math.pi)))

    # 管道：1 窄管(应放弃)、2 圆管、3 方涵
    tn = cfg['tunnels']
    p = tn['narrow']
    cx, cy = p['center']
    for side in (-1, 1):
        box(world, 'narrow_side_%d' % side,
            (cx, cy + side * (p['width'] + 0.1) / 2, p['height'] / 2),
            (p['length'] / 2, 0.05, p['height'] / 2), PIPE_RGBA)
    box(world, 'narrow_roof', (cx, cy, p['height'] + 0.05),
        (p['length'] / 2, p['width'] / 2 + 0.05, 0.05), PIPE_RGBA)

    p = tn['round']
    cx, cy = p['center']
    rr, seg = p['inner_diameter'] / 2, 16
    for j in range(seg):
        theta = 2 * math.pi * j / seg
        box(world, 'pipe_%02d' % j,
            (cx, cy + (rr + 0.02) * math.cos(theta), rr + (rr + 0.02) * math.sin(theta)),
            (p['length'] / 2, rr * math.pi / seg + 0.005, 0.02), PIPE_RGBA,
            euler=(theta - math.pi / 2, 0, 0))

    p = tn['rectangular']
    cx, cy = p['center']
    for side in (-1, 1):
        box(world, 'culvert_side_%d' % side, (cx, cy + side * (p['width'] + 0.1) / 2, p['height'] / 2),
            (p['length'] / 2, 0.05, p['height'] / 2), CULVERT_RGBA)
    box(world, 'culvert_roof', (cx, cy, p['height'] + 0.05),
        (p['length'] / 2, p['width'] / 2 + 0.05, 0.05), CULVERT_RGBA)

    # 终点通道隔墙
    wall(world, 'finish_partition', (8.7, 6.0), (8.7, 9.25), wh, t)

    # 足球生命体（随机出现在可达区域，带物理可滚动）
    if with_victim:
        bx, by = VICTIM_ZONES[rng.integers(0, len(VICTIM_ZONES))] if rng is not None else VICTIM_ZONES[0]
        ball = el(world, 'body', name='victim_ball', pos='%.3f %.3f 0.115' % (bx, by))
        el(ball, 'freejoint', name='ball_joint')
        el(ball, 'geom', name='ball_geom', type='sphere', size='0.11',
           rgba=BALL_RGBA, mass='0.43', friction='0.6 0.1 0.1')

    return muj, (ux, uy, dx, dy, run, high)


def merge_robot(arena_root, robot_xml_path):
    """把 Go2 机器人 xml 的 asset/default/worldbody/actuator/sensor 合并进场地。"""
    robot = ET.parse(str(robot_xml_path)).getroot()
    for tag in ('asset', 'default', 'worldbody', 'actuator', 'sensor', 'tendon', 'contact', 'equality'):
        src = robot.find(tag)
        if src is None:
            continue
        dst = arena_root.find(tag)
        if dst is None:
            dst = ET.SubElement(arena_root, tag)
        for child in src:
            dst.append(child)
    return arena_root


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, help='输出 MJCF xml 路径')
    parser.add_argument('--seed', type=int, default=None, help='随机化种子（不给则完全按配置坐标）')
    parser.add_argument('--no-victim', action='store_true', help='不放置足球生命体')
    parser.add_argument('--with-robot', default=None, help='合并机器人 xml（如 robot/go2/go2.xml）')
    args = parser.parse_args()

    cfg = yaml.safe_load(ARENA_YAML.read_text())
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    root, slope_info = build_arena(cfg, rng, with_victim=not args.no_victim)
    if args.with_robot:
        merge_robot(root, Path(args.with_robot))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = ET.tostring(root, encoding='unicode')
    try:
        ET.indent(root, space='  ')  # Python 3.9+
        raw = ET.tostring(root, encoding='unicode')
    except AttributeError:
        from xml.dom import minidom
        raw = minidom.parseString(raw).toprettyxml(indent='  ')
    out.write_text("""<!-- 由 mujoco_env/generate_arena_xml.py 自动生成，勿手改。
     几何真源：src/unitree_competition/config/arena.yaml
     seed=%s victim=%s -->\n""" % (args.seed, not args.no_victim)
        + raw)
    print('written: %s (slope run=%.3f m, deck height=%.2f m)' % (out, slope_info[4], slope_info[5]))


if __name__ == '__main__':
    main()
