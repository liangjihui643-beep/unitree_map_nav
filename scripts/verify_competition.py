#!/usr/bin/env python3
"""Read sensors and exercise the actual Gazebo collision geometry with probes.

Does not command robot motion or claim autonomous course completion.
"""
import json
import math
from pathlib import Path
import time

import rospy
from gazebo_msgs.srv import GetWorldProperties, GetModelState, SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu, LaserScan
import yaml

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / 'src/unitree_competition'


def main():
    rospy.init_node('verify_competition', anonymous=True)
    config = yaml.safe_load((PKG / 'config/arena.yaml').read_text())
    manifest = json.loads((PKG / 'config/generated_manifest.json').read_text())
    report = {'passed': False, 'robot': 'Go1', 'scope': 'scene, sensors, local SLAM, collision probes',
              'autonomous_course_verified': False, 'collision_probes': {}}
    created = []
    get = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    delete = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    try:
        rospy.wait_for_service('/gazebo/get_world_properties', timeout=60)
        world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
        missing = sorted(set(manifest['models']) - set(world.model_names))
        assert not missing, 'Missing arena models: ' + str(missing)
        report['arena_models'] = len(manifest['models'])
        report['rubble_patches'] = len([n for n in world.model_names if n.startswith('rubble_patch_')])
        assert report['rubble_patches'] == 5
        topics = dict(rospy.get_published_topics())
        scan_topic = '/go1_gazebo/scan'
        imu_topic = next(t for t, typ in topics.items() if typ == 'sensor_msgs/Imu')
        odom_topic = next(t for t, typ in topics.items() if typ == 'nav_msgs/Odometry')
        scan = rospy.wait_for_message(scan_topic, LaserScan, timeout=30)
        rospy.wait_for_message(imu_topic, Imu, timeout=30)
        rospy.wait_for_message(odom_topic, Odometry, timeout=30)
        valid = [r for r in scan.ranges if math.isfinite(r) and scan.range_min < r < scan.range_max]
        assert len(valid) > 20
        report['laser_samples'] = len(scan.ranges)
        report['valid_laser_returns'] = len(valid)
        state = get('go1_gazebo', 'world')
        q = state.pose.orientation
        tilt_cosine = 1-2*(q.x*q.x+q.y*q.y)
        report['robot_position'] = {axis: getattr(state.pose.position, axis) for axis in 'xyz'}
        assert .18 < state.pose.position.z < .50 and tilt_cosine > .85, 'Robot not standing upright'
        grid = rospy.wait_for_message('/map', OccupancyGrid, timeout=30)
        report['map_known_cells'] = sum(v >= 0 for v in grid.data)
        report['map_occupied_cells'] = sum(v >= 50 for v in grid.data)
        report['map_scope'] = 'start-area observation, not a complete course map'
        assert report['map_known_cells'] > 100 and report['map_occupied_cells'] > 10
        slope = config['slope']
        run = manifest['slope_run']
        up = slope['uphill_start']
        down = slope['downhill_start']
        pipe = config['tunnels']['round']
        rect = config['tunnels']['rectangular']
        # Small dynamic probes fall onto surfaces. Low restitution, high friction.
        probes = [
            ('uphill', up[0], up[1]+run/2, .65, (.14, .32)),
            ('downhill', down[0]+run/2, down[1], .65, (.14, .32)),
            ('round_inside', *pipe['center'], .22, (.015, .07)),
            ('round_roof', *pipe['center'], 1.35, (1.07, 1.14)),
            ('culvert_inside', *rect['center'], .25, (.015, .07)),
            ('culvert_roof', *rect['center'], 1.0, (.70, .76)),
        ]
        spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        for index, (label, x, y, z, bounds) in enumerate(probes):
            name = 'arena_probe_%d_%d' % (int(time.time()), index)
            xml = '''<sdf version="1.6"><model name="%s"><link name="probe">
              <inertial><mass>0.02</mass><inertia><ixx>0.0000083</ixx><iyy>0.0000083</iyy><izz>0.0000083</izz></inertia></inertial>
              <collision name="collision"><geometry><box><size>0.05 0.05 0.05</size></box></geometry>
                <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
              <visual name="visual"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></visual>
              </link></model></sdf>''' % name
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.w = 1
            response = spawn(name, xml, '', pose, 'world')
            assert response.success, response.status_message
            created.append(name)
            start_sim, deadline = rospy.Time.now().to_sec(), time.monotonic()+25
            while rospy.Time.now().to_sec()-start_sim < 1.2:
                if time.monotonic() > deadline:
                    raise RuntimeError('Simulation clock stalled during contact check')
                time.sleep(.05)
            measured = get(name, 'world')
            height = measured.pose.position.z
            report['collision_probes'][label] = {'height': height, 'expected_range': bounds}
            assert measured.success and bounds[0] < height < bounds[1], '%s collision height %.4f' % (label, height)
            delete(name)
            created.remove(name)
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        for name in created:
            try:
                delete(name)
            except rospy.ServiceException:
                pass
        output = ROOT / 'logs/verification_competition.json'
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
