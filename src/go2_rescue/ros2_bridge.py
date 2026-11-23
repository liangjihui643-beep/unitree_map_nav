"""ROS 2 bridge process (Python 3.8 / rclpy) + socket clients used by the 3.12 stack.

Topology (Foxy's rclpy only supports Python 3.8, the repo venv is 3.12):

    sim (3.12)  --socket-->  this bridge (3.8, rclpy)  --ROS 2-->  topics
                                 ^                                  |
                                 |                                  v
    agent (3.12) <--socket--  this bridge  <--ROS 2-- /cmd_vel  (or agent socket)

The bridge carries the FULL 3-D point cloud (PointCloud2 XYZ), camera Image plus
CameraInfo intrinsics, IMU and odometry; nothing is flattened to a 2-D scan.

Run:
    source /opt/ros/foxy/setup.bash
    python3 -m go2_rescue.ros2_bridge --sim-port 8765 --agent-port 8766
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import threading
import time

import numpy as np


def _pack_line(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


class _LineConn:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""
        self.lock = threading.Lock()

    def send(self, obj):
        data = _pack_line(obj)
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:
                pass

    def recv(self):
        try:
            chunk = self.sock.recv(1 << 20)
        except OSError:
            return None
        if not chunk:
            return None
        self.buf += chunk
        if b"\n" not in self.buf:
            return []
        lines = self.buf.split(b"\n")
        self.buf = lines[-1]
        return [json.loads(line) for line in lines[:-1] if line.strip()]


class Ros2Bridge:
    def __init__(self, sim_port=8765, agent_port=8766):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import PointCloud2, PointField, Image, Imu, CameraInfo
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Header

        self._rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("go2_ros2_bridge")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.qos = qos
        self.msgs = dict(PointCloud2=PointCloud2, PointField=PointField, Image=Image,
                         Imu=Imu, CameraInfo=CameraInfo, Odometry=Odometry, Twist=Twist,
                         Header=Header)
        self.pub_points = self.node.create_publisher(PointCloud2, "/lidar/points", qos)
        self.pub_image = self.node.create_publisher(Image, "/camera/image_raw", qos)
        self.pub_info = self.node.create_publisher(CameraInfo, "/camera/camera_info", qos)
        self.pub_imu = self.node.create_publisher(Imu, "/imu", qos)
        self.pub_odom = self.node.create_publisher(Odometry, "/odom", qos)
        self.sub_cmd = self.node.create_subscription(Twist, "/cmd_vel", self._on_ros_cmd, qos)
        self.sim_conn = None
        self.agent_conn = None
        self._lock = threading.Lock()
        self._servers = []

    def _on_ros_cmd(self, msg):
        cmd = {"vx": float(msg.linear.x), "vy": float(msg.linear.y), "wz": float(msg.angular.z)}
        with self._lock:
            if self.sim_conn:
                self.sim_conn.send(dict(kind="cmd", **cmd))
            if self.agent_conn:
                self.agent_conn.send(dict(kind="cmd", **cmd))

    # ---------- publishing ----------
    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def publish_obs(self, obs):
        m = self.msgs
        stamp = self._stamp()
        pts = np.asarray(obs.get("points") or [], np.float32).reshape(-1, 3)
        pc = m["PointCloud2"]()
        pc.header = m["Header"](stamp=stamp, frame_id=obs.get("frame_id", "base_link"))
        pc.height = 1
        pc.width = int(pts.shape[0])
        pc.fields = [m["PointField"](name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(("x", "y", "z"))]
        pc.is_bigendian = False
        pc.point_step = 12
        pc.row_step = 12 * pc.width
        pc.is_dense = True
        pc.data = pts.tobytes()
        self.pub_points.publish(pc)

        if obs.get("image_b64"):
            raw = base64.b64decode(obs["image_b64"])
            img = m["Image"]()
            img.header = m["Header"](stamp=stamp, frame_id=obs.get("camera_frame", "front_camera"))
            img.height, img.width = int(obs["image_h"]), int(obs["image_w"])
            img.encoding = "bgr8"
            img.is_bigendian = 0
            img.step = int(obs["image_step"])
            img.data = raw
            self.pub_image.publish(img)
            info = m["CameraInfo"]()
            info.header = img.header
            info.height, info.width = img.height, img.width
            info.distortion_model = "plumb_bob"
            k = obs.get("K") or [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            info.k = [float(v) for v in k]
            info.p = [k[0], 0., k[2], 0., 0., k[4], k[5], 0., 0., 0., 1., 0.]
            self.pub_info.publish(info)

        q = obs.get("quat") or [1., 0., 0., 0.]
        imu = m["Imu"]()
        imu.header = m["Header"](stamp=stamp, frame_id="base_link")
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = [float(v) for v in q]
        imu.angular_velocity.z = float((obs.get("vel") or [0., 0., 0.])[2])
        self.pub_imu.publish(imu)

        odom = m["Odometry"]()
        odom.header = m["Header"](stamp=stamp, frame_id="map")
        odom.child_frame_id = "base_link"
        v = obs.get("vel") or [0., 0., 0.]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = [float(x) for x in v]
        self.pub_odom.publish(odom)

    # ---------- sockets ----------
    def _accept_loop(self, server, kind):
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            c = _LineConn(conn)
            with self._lock:
                if kind == "sim":
                    self.sim_conn = c
                else:
                    self.agent_conn = c
            self.node.get_logger().info(f"{kind} connected")
            threading.Thread(target=self._read_loop, args=(c, kind), daemon=True).start()

    def _read_loop(self, conn, kind):
        while True:
            msgs = conn.recv()
            if msgs is None:
                with self._lock:
                    if kind == "sim" and self.sim_conn is conn:
                        self.sim_conn = None
                    if kind == "agent" and self.agent_conn is conn:
                        self.agent_conn = None
                return
            for msg in msgs:
                k = msg.get("kind")
                if k == "obs" and kind == "sim":
                    self.publish_obs(msg)
                    with self._lock:
                        if self.agent_conn:
                            self.agent_conn.send(msg)
                elif k == "cmd" and kind == "agent":
                    self._on_ros_cmd(type("M", (), {"linear": type("L", (), {"x": msg["vx"], "y": msg["vy"]})(),
                                                    "angular": type("A", (), {"z": msg["wz"]})()})())

    def serve(self):
        self.node.get_logger().info("bridge serving")
        self._rclpy.spin(self.node)

    def close(self):
        try:
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Socket clients for the Python 3.12 side
# ---------------------------------------------------------------------------
class SimSocketClient:
    """Simulator-side client: send observations, receive the latest command."""

    def __init__(self, host="127.0.0.1", port=8765, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.conn = _LineConn(self.sock)
        self._cmd = None
        self._buf = b""

    def publish(self, frame, measured_velocity, t, K, image, points):
        img_bytes = image.tobytes() if image is not None else None
        obs = {"kind": "obs", "t": float(t), "points": np.asarray(points, float).tolist(),
               "quat": np.asarray(frame.quaternion, float).tolist(),
               "vel": [float(measured_velocity.vx), float(measured_velocity.vy), float(measured_velocity.wz)],
               "K": [float(v) for v in np.asarray(K).ravel()],
               "frame_id": "base_link", "camera_frame": "front_camera"}
        if img_bytes is not None:
            obs.update(image_b64=base64.b64encode(img_bytes).decode(),
                       image_h=int(image.shape[0]), image_w=int(image.shape[1]),
                       image_step=int(image.strides[0]))
        self.conn.send(obs)

    def latest_command(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.sock.settimeout(0.1)
                chunk = self.sock.recv(1 << 20)
            except socket.timeout:
                continue
            if not chunk:
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("kind") == "cmd":
                    self._cmd = (msg["vx"], msg["vy"], msg["wz"])
        return self._cmd


class AgentSocketClient:
    """Agent-side client: receive observations, send commands."""

    def __init__(self, host="127.0.0.1", port=8766, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.conn = _LineConn(self.sock)
        self._buf = b""

    def recv_obs(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.sock.settimeout(0.1)
                chunk = self.sock.recv(1 << 20)
            except socket.timeout:
                continue
            if not chunk:
                return None
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("kind") == "obs":
                    return msg
        return None

    def send_cmd(self, vx, vy, wz):
        self.conn.send({"kind": "cmd", "vx": float(vx), "vy": float(vy), "wz": float(wz)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-port", type=int, default=8765)
    ap.add_argument("--agent-port", type=int, default=8766)
    args = ap.parse_args()
    bridge = Ros2Bridge(args.sim_port, args.agent_port)
    servers = []
    for port, kind in ((args.sim_port, "sim"), (args.agent_port, "agent")):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(4)
        servers.append(s)
        threading.Thread(target=bridge._accept_loop, args=(s, kind), daemon=True).start()
    try:
        bridge.serve()
    except KeyboardInterrupt:
        pass
    finally:
        for s in servers:
            s.close()
        bridge.close()


if __name__ == "__main__":
    main()
