#!/usr/bin/env python3
"""Integration checks against live Gazebo, ROS messages and move_base actions.

Ground-truth poses are used only for measuring physical motion in these checks;
they are never sent to the SLAM or navigation algorithms.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu, LaserScan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mapping", "navigation"])
    args = parser.parse_args()
    root = Path(os.environ["UNITREE_MAP_NAV_ROOT"])
    rospy.init_node("unitree_demo_verification", anonymous=True)
    report = {"phase": args.phase, "robot": "go1", "uses_ground_truth_for_evaluation_only": True}
    rospy.wait_for_service("/gazebo/get_model_state", timeout=90)
    model = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    scan = rospy.wait_for_message("/go1_gazebo/scan", LaserScan, timeout=90)
    valid = [x for x in scan.ranges if math.isfinite(x) and scan.range_min <= x <= scan.range_max]
    assert len(valid) > 20, "Too few valid laser returns"
    report["laser_valid_returns"] = len(valid)
    rospy.wait_for_message("/trunk_imu", Imu, timeout=30)
    rospy.wait_for_message("/odom", Odometry, timeout=45)
    time.sleep(9)
    start = model("go1_gazebo", "world")
    assert start.success, start.status_message
    report["standing_height_m"] = start.pose.position.z
    assert 0.18 < start.pose.position.z < 0.5, "Robot is not standing"
    velocity = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    if args.phase == "mapping":
        command = Twist()
        command.linear.x = 0.12
        deadline = time.monotonic() + 10
        try:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                velocity.publish(command)
                time.sleep(0.05)
        finally:
            for _ in range(10):
                velocity.publish(Twist())
                time.sleep(0.05)
        end = model("go1_gazebo", "world")
        distance = math.hypot(end.pose.position.x - start.pose.position.x,
                              end.pose.position.y - start.pose.position.y)
        report["walking_displacement_m"] = distance
        report["final_height_m"] = end.pose.position.z
        assert distance > 0.25, "Commanded walking did not move the robot enough"
        assert end.pose.position.z > 0.18, "Robot fell during walking"
        grid = rospy.wait_for_message("/map", OccupancyGrid, timeout=45)
        known = sum(x >= 0 for x in grid.data)
        occupied = sum(x >= 50 for x in grid.data)
        report.update(map_width=grid.info.width, map_height=grid.info.height,
                      map_resolution_m=grid.info.resolution, map_known_cells=known,
                      map_occupied_cells=occupied)
        assert known > 1000 and occupied > 20, "SLAM map does not contain enough geometry"
        maps = root / "src/unitree_local_demo/maps"
        maps.mkdir(exist_ok=True)
        subprocess.run(["rosrun", "map_server", "map_saver", "-f", str(maps / "demo_room")],
                       check=True, timeout=45)
        report["saved_map"] = str(maps / "demo_room.yaml")
    else:
        report["global_planner"] = rospy.get_param("/move_base/base_global_planner")
        report["local_planner"] = rospy.get_param("/move_base/base_local_planner")
        report["local_costmap_plugins"] = rospy.get_param("/move_base/local_costmap/plugins")
        # Start pose is known because this is a reproducible reset of the test world.
        initial_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
        initial = PoseWithCovarianceStamped()
        initial.header.frame_id = "map"
        initial.pose.pose.orientation.w = 1.0
        initial.pose.covariance[0] = 0.03
        initial.pose.covariance[7] = 0.03
        initial.pose.covariance[35] = 0.03
        time.sleep(1)
        initial.header.stamp = rospy.Time.now()
        initial_pub.publish(initial)
        rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=45)
        client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        assert client.wait_for_server(rospy.Duration(60)), "move_base action server unavailable"
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = 0.9
        goal.target_pose.pose.orientation.w = 1.0
        client.send_goal(goal)
        finished = client.wait_for_result(rospy.Duration(120))
        if not finished:
            client.cancel_goal()
        state = client.get_state()
        result = client.get_result()
        report["navigation_action_state"] = state
        report["navigation_result"] = str(result)
        end = model("go1_gazebo", "world")
        report["physical_goal_error_m"] = math.hypot(end.pose.position.x - 0.9, end.pose.position.y)
        report["final_height_m"] = end.pose.position.z
        assert finished and state == GoalStatus.SUCCEEDED, "Navigation goal did not succeed"
        assert report["physical_goal_error_m"] < 0.45, "Physical robot did not reach the goal"
        assert end.pose.position.z > 0.18, "Robot fell during navigation"

    report["passed"] = True
    path = root / "logs" / ("verification_" + args.phase + ".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
