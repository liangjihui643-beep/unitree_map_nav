#!/usr/bin/env python3
"""Run the unchanged keyboard controller in a PTY and enter MoveBase mode."""
import os
import pty
import select
import signal
import subprocess
import sys
import time

import rospkg
import rospy
from sensor_msgs.msg import JointState


def main():
    rospy.init_node("boot_controller")
    # Wait for the physics and joint controller spawner, not an arbitrary startup delay.
    rospy.wait_for_message("/go1_gazebo/joint_states", JointState, timeout=90)
    package = rospkg.RosPack().get_path("unitree_guide")
    root = os.environ["UNITREE_MAP_NAV_ROOT"]
    binary = os.path.join(root, "devel/lib/unitree_guide/junior_ctrl")
    master, slave = pty.openpty()
    process = subprocess.Popen([binary], stdin=slave, stdout=slave, stderr=slave,
                               cwd=package, start_new_session=True)
    os.close(slave)
    started = time.monotonic()
    stage = 0

    def stop():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
        os.close(master)

    rospy.on_shutdown(stop)
    while not rospy.is_shutdown() and process.poll() is None:
        if select.select([master], [], [], 0.1)[0]:
            try:
                data = os.read(master, 16384)
            except OSError:
                break
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        elapsed = time.monotonic() - started
        if stage == 0 and elapsed > 2:
            os.write(master, b"2")
            stage = 1
            rospy.loginfo("Requested FixedStand using the upstream keyboard interface")
        if stage == 1 and elapsed > 6:
            os.write(master, b"5")
            stage = 2
            rospy.loginfo("Requested MoveBase mode; controller now accepts /cmd_vel")
    if not rospy.is_shutdown():
        raise RuntimeError("Upstream controller exited with code %s" % process.poll())


if __name__ == "__main__":
    main()
