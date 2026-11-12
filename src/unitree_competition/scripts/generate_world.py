#!/usr/bin/env python3
"""Generate a deterministic, offline Gazebo arena and an annotated plan.

All traversable terrain, walls, rocks and tunnel shells have collision geometry.
Only coloured start/finish/route markings are visual-only.
"""
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import yaml

PKG = Path(__file__).resolve().parent.parent
C = yaml.safe_load((PKG / 'config/arena.yaml').read_text())
SDF = ET.Element('sdf', version='1.6')
WORLD = ET.SubElement(SDF, 'world', name='competition_arena')
FOOTPRINTS = []


def sub(parent, tag, value=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if value is not None:
        node.text = str(value)
    return node


def values(numbers):
    return ' '.join('%.8g' % n for n in numbers)


def model(name):
    m = sub(WORLD, 'model', name=name)
    sub(m, 'static', 'true')
    return sub(m, 'link', name='structure')


def shape(link, name, pose, color, geometry, collide=True):
    if collide:
        col = sub(link, 'collision', name=name + '_collision')
        sub(col, 'pose', values(pose))
        col.append(ET.fromstring(geometry))
        surface = sub(col, 'surface')
        friction = sub(sub(surface, 'friction'), 'ode')
        sub(friction, 'mu', 0.9)
        sub(friction, 'mu2', 0.9)
    visual = sub(link, 'visual', name=name + '_visual')
    sub(visual, 'pose', values(pose))
    visual.append(ET.fromstring(geometry))
    material = sub(visual, 'material')
    sub(material, 'ambient', values(color))
    sub(material, 'diffuse', values(color))


WALL = (0.49, 0.51, 0.53, 1)
TERRAIN = (0.68, 0.53, 0.32, 1)


def box(link, name, center, size, color=WALL, yaw=0, roll=0, collide=True, plan=True):
    shape(link, name, (*center, roll, 0, yaw), color,
          '<geometry><box><size>%s</size></box></geometry>' % values(size), collide)
    if plan:
        FOOTPRINTS.append((center[:2], size[:2], yaw, color))


def wall(link, name, a, b, height, thickness, bottom=0, color=WALL):
    dx, dy = b[0] - a[0], b[1] - a[1]
    box(link, name, ((a[0]+b[0])/2, (a[1]+b[1])/2, bottom+height/2),
        (math.hypot(dx, dy), thickness, height), color, math.atan2(dy, dx))


def wedge(name, origin, length, width, h0, h1, yaw):
    # A closed triangular-prism terrain mesh, continuous at the floor/deck.
    verts = [(0, -width/2, -0.05), (length, -width/2, -0.05),
             (length, width/2, -0.05), (0, width/2, -0.05),
             (0, -width/2, h0), (length, -width/2, h1),
             (length, width/2, h1), (0, width/2, h0)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    # Binary STL is supported by Gazebo's mesh loader (ASCII STL is not portable).
    import struct
    mesh = PKG / 'models/competition_terrain/meshes' / (name + '.stl')
    mesh.parent.mkdir(parents=True, exist_ok=True)
    with mesh.open('wb') as output:
        output.write(b'Competition terrain: metres'.ljust(80, b'\0'))
        output.write(struct.pack('<I', len(faces)))
        for f in faces:
            v = [verts[i] for i in f]
            a = [v[1][i]-v[0][i] for i in range(3)]
            b = [v[2][i]-v[0][i] for i in range(3)]
            n = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
            norm = math.sqrt(sum(x*x for x in n))
            output.write(struct.pack('<12fH', *[x/norm for x in n], *v[0], *v[1], *v[2], 0))
    link = model(name)
    shape(link, 'terrain', (*origin, 0, 0, 0, yaw), TERRAIN,
          '<geometry><mesh><uri>model://competition_terrain/meshes/%s.stl</uri></mesh></geometry>' % name)
    cx = origin[0] + length/2 * math.cos(yaw)
    cy = origin[1] + length/2 * math.sin(yaw)
    FOOTPRINTS.append(((cx, cy), (length, width), yaw, TERRAIN))


for included in ('sun', 'ground_plane'):
    sub(sub(WORLD, 'include'), 'uri', 'model://' + included)
physics = sub(WORLD, 'physics', name='default', type='ode')
sub(physics, 'max_step_size', C['physics']['max_step_size'])
sub(physics, 'real_time_update_rate', C['physics']['real_time_update_rate'])
solver = sub(sub(physics, 'ode'), 'solver')
sub(solver, 'type', 'quick')
sub(solver, 'iters', 50)
scene = sub(WORLD, 'scene')
sub(scene, 'ambient', '0.72 0.72 0.72 1')
sub(scene, 'background', '0.80 0.85 0.90 1')
sub(sub(sub(WORLD, 'gui'), 'camera', name='user_camera'), 'pose', '17 -12 17 0 0.64 2.14')

a = C['arena']
x0, y0 = a['minimum']
sx, sy = a['size']
t, wh = a['wall_thickness'], a['wall_height']
walls = model('arena_boundary')
wall(walls, 'south', (x0-t/2, y0-t/2), (x0+sx+t/2, y0-t/2), wh, t)
wall(walls, 'north', (x0-t/2, y0+sy+t/2), (x0+sx+t/2, y0+sy+t/2), wh, t)
wall(walls, 'west', (x0-t/2, y0), (x0-t/2, y0+sy), wh, t)
wall(walls, 'east', (x0+sx+t/2, y0), (x0+sx+t/2, y0+sy), wh, t)
for name, color in [('start', (0.15, 0.45, 0.85, 1)), ('finish', (0.15, 0.65, 0.35, 1))]:
    pad = C[name]
    box(model(name+'_pad'), 'mark', (*pad['center'], 0.001), (*pad['size'], 0.002), color, collide=False)

s = C['slope']
run = s['height'] / math.tan(math.radians(s['angle_deg']))
ux, uy = s['uphill_start']
dx, dy = s['downhill_start']
width, high, turn_y = s['width'], s['height'], s['turn_y']
wedge('uphill', (ux, uy), run, width, 0, high, math.pi/2)
wedge('downhill', (dx, dy), run, width, high, 0, 0)
deck = model('slope_turn_platform')
box(deck, 'long_deck', (ux, (uy+run+turn_y+width/2)/2, high/2),
    (width, turn_y+width/2-uy-run, high), TERRAIN)
box(deck, 'cross_deck', ((ux+dx)/2, turn_y, high/2), (dx-ux, width, high), TERRAIN)
rails = model('slope_guard_walls')
wall(rails, 'inner', (ux+width/2+t/2, uy), (ux+width/2+t/2, turn_y-width/2-t), 1.0, t)
wall(rails, 'outer', (ux-width/2-t/2, uy), (ux-width/2-t/2, turn_y+width/2+t), 1.0, t)
wall(rails, 'north', (ux-width/2, turn_y+width/2+t/2), (C['s_bend']['entry'][0], turn_y+width/2+t/2), 1.0, t)
wall(rails, 'south', (ux+width/2, turn_y-width/2-t/2), (C['s_bend']['entry'][0], turn_y-width/2-t/2), 1.0, t)
# 封死起点东侧平地：起点区唯一出口是上坡道（inner 导轨向场地边界延伸）。
partition = model('start_entry_partition')
wall(partition, 'east_gate', (ux+width/2+t/2, y0), (ux+width/2+t/2, uy), wh, t)

s = C['s_bend']
entry_x, entry_y = s['entry']
radius, clear = s['radius'], s['clear_width']
assert radius > clear/2+t, 'S bend inner radius must remain positive'
bend = model('s_bend_walls')
route = []
for arc, center, begin, end in [
        (0, (entry_x, entry_y-radius), math.pi/2, -math.pi/2),
        (1, (entry_x, entry_y-3*radius), math.pi/2, 3*math.pi/2)]:
    steps = 40
    for j in range(steps+1):
        theta = begin + (end-begin)*j/steps
        route.append((center[0]+radius*math.cos(theta), center[1]+radius*math.sin(theta)))
    for side in (-1, 1):
        rr = radius + side*(clear/2+t/2)
        for j in range(steps):
            aa = begin + (end-begin)*j/steps
            bb = begin + (end-begin)*(j+1)/steps
            p = (center[0]+rr*math.cos(aa), center[1]+rr*math.sin(aa))
            q = (center[0]+rr*math.cos(bb), center[1]+rr*math.sin(bb))
            # Adjacent boxes touch; 1mm overlap prevents ray/physics cracks.
            vx, vy = q[0]-p[0], q[1]-p[1]
            scale = 0.0005/math.hypot(vx, vy)
            p = (p[0]-vx*scale, p[1]-vy*scale)
            q = (q[0]+vx*scale, q[1]+vy*scale)
            wall(bend, 'arc%d_side%d_%02d' % (arc, side, j), p, q, s['wall_height'], t,
                 color=(0.26, 0.55, 0.61, 1))

d = C['dead_end']
ex, ey = d['entrance']
bottom = ey-d['length']
dead = model('dead_end')
wall(dead, 'left', (ex-d['clear_width']/2-t/2, bottom), (ex-d['clear_width']/2-t/2, ey), wh, t,
     color=(0.68, 0.32, 0.28, 1))
wall(dead, 'right', (ex+d['clear_width']/2+t/2, bottom), (ex+d['clear_width']/2+t/2, ey), wh, t,
     color=(0.68, 0.32, 0.28, 1))
wall(dead, 'closed_end', (ex-d['clear_width']/2-t, bottom-t/2), (ex+d['clear_width']/2+t, bottom-t/2), wh, t,
     color=(0.68, 0.32, 0.28, 1))

r = C['rubble']
rng = random.Random(r['seed'])
for patch, (cx, cy) in enumerate(r['centers'], 1):
    link = model('rubble_patch_%d' % patch)
    for j in range(r['rocks_per_patch']):
        angle = rng.uniform(0, 2*math.pi)
        distance = r['patch_radius'] * math.sqrt(rng.random())
        x, y = cx+distance*math.cos(angle), cy+distance*math.sin(angle)
        height = rng.uniform(*r['height_range'])
        color = rng.uniform(0.32, 0.56)
        box(link, 'rock_%02d' % j, (x, y, height/2),
            (rng.uniform(0.12, 0.25), rng.uniform(0.10, 0.22), height),
            (color, color*0.95, color*0.88, 1), yaw=rng.uniform(0, math.pi))

tunnels = C['tunnels']
for key, along_x, name in [('narrow', False, 'tunnel_1_reject'), ('rectangular', True, 'tunnel_3_culvert')]:
    p = tunnels[key]
    cx, cy = p['center']
    shell = p.get('shell', 0.1)
    w, h, length = p['width'], p['height'], p['length']
    link = model(name)
    color = (0.48, 0.48, 0.57, 1) if key == 'narrow' else (0.28, 0.56, 0.48, 1)
    for side in (-1, 1):
        center = (cx, cy+side*(w+shell)/2, h/2) if along_x else (cx+side*(w+shell)/2, cy, h/2)
        size = (length, shell, h) if along_x else (shell, length, h)
        box(link, 'side_%d' % side, center, size, color)
    size = (length, w+2*shell, shell) if along_x else (w+2*shell, length, shell)
    box(link, 'roof', (cx, cy, h+shell/2), size, color)

p = tunnels['round']
cx, cy = p['center']
rr, shell, segments = p['inner_diameter']/2, p['shell'], p['segments']
pipe = model('tunnel_2_round_pipe')
for j in range(segments):
    theta = 2*math.pi*j/segments
    box(pipe, 'shell_%02d' % j,
        (cx, cy+(rr+shell/2)*math.cos(theta), rr+(rr+shell/2)*math.sin(theta)),
        (p['length'], 2*(rr+shell)*math.tan(math.pi/segments)+0.002, shell),
        (0.55, 0.58, 0.65, 1), roll=theta-math.pi/2, plan=False)
# The ring is hollow; no cylinder collision is placed across its opening.
FOOTPRINTS.append(((cx, cy), (p['length'], 2*(rr+shell)), 0, (0.55, 0.58, 0.65, 0.8)))
partition = model('finish_lane_partition')
wall(partition, 'partition', (8.7, 6.0), (8.7, 9.25), wh, t)

for directory in ('worlds', 'docs', 'maps'):
    (PKG / directory).mkdir(exist_ok=True)
world_path = PKG / 'worlds/competition.world'
ET.ElementTree(SDF).write(str(world_path), encoding='utf-8', xml_declaration=True)
# Pretty print without requiring Python 3.9's ET.indent.
from xml.dom import minidom
world_path.write_text(minidom.parseString(ET.tostring(SDF)).toprettyxml(indent='  '))

import json
manifest = {'arena_size': a['size'], 'spawn': C['spawn'], 'slope_run': run,
            'models': [m.attrib['name'] for m in WORLD.findall('model')],
            'collision_shapes': len(WORLD.findall('.//collision')),
            'rubble_patches': len(r['centers']), 'seed': r['seed']}
(PKG / 'config/generated_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
# Keep spawn settings in sync with arena.yaml without a second source of truth.
launch_path = PKG / 'launch/competition.launch'
launch = ET.parse(str(launch_path))
for arg in launch.getroot().find('include').findall('arg'):
    if arg.attrib['name'] in ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw'):
        index = ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw').index(arg.attrib['name'])
        arg.set('value', '%.12g' % C['spawn'][index])
launch.write(str(launch_path), encoding='utf-8', xml_declaration=True)

# Exportable technical plan; same geometry source as the Gazebo world.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
fig, ax = plt.subplots(figsize=(12, 10))
ax.add_patch(Rectangle((x0, y0), sx, sy, facecolor='#ece9df', zorder=-10))
for center, size, yaw, color in FOOTPRINTS:
    co, si = math.cos(yaw), math.sin(yaw)
    points = [(center[0]+x*co-y*si, center[1]+x*si+y*co)
              for x, y in [(-size[0]/2, -size[1]/2), (size[0]/2, -size[1]/2),
                           (size[0]/2, size[1]/2), (-size[0]/2, size[1]/2)]]
    ax.add_patch(Polygon(points, facecolor=color, edgecolor='none'))
route = ([(0, -0.4), (0, 7.2), (2.4, 7.2), (4.8, 7.2)] + route
         + [(6.0, 3.2), (7.6, 3.2), (8.0, 4.8), (10.5, 4.8), (10.4, 5.9), (10.1, 8.1)])
ax.plot(*zip(*route), '--', color='#e89e22', lw=2.2, label='Reference route L1->L3 (not an autonomous trajectory)')
labels = [('START\nL1: 直行上坡', (0, -0.15)), ('UP 12 deg\n(from start)', (0, 1.9)), ('0.45 m deck', (1.3, 7.2)),
          ('DOWN 12 deg', (3.35, 7.2)), ('S BEND (L2)\n1.2 m clear', (4.8, 6.2)),
          ('DEAD END branch\n1.2 x 2 m', (ex, (ey+bottom)/2)),
          ('OBSTACLE FIELD (L2/L3)\n5 rubble patches', (7.3, 2.1)),
          ('1: REJECT\n0.2 m wide', (7.2, 7.7)), ('2: ROUND (L3)\nID 1.0 m, L 2 m', (7.8, 5.65)),
          ('3: CULVERT (L3)\n1.2 x 0.6 x 1.5 m', (9.9, 3.7)), ('FINISH', (10.1, 8.1))]
for label, position in labels:
    ax.text(*position, label, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=.22', facecolor='white', alpha=.85, edgecolor='#aaaaaa'))
ax.set(xlim=(x0-.35, x0+sx+.35), ylim=(y0-.35, y0+sy+.35),
       xlabel='World X (m)', ylabel='World Y (m)',
       title='Rescue competition arena | 12 x 10 m | schematic reconstruction')
ax.set_aspect('equal')
ax.set_xticks(range(12))
ax.set_yticks(range(10))
ax.grid(alpha=.18)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(str(PKG / 'docs/arena_plan.png'), dpi=150)
fig.savefig(str(PKG / 'docs/arena_plan.svg'))
print(json.dumps(manifest, indent=2))
