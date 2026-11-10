#!/usr/bin/env bash
# launch-prefix：等 SLAM 地图建成足够大的已知区域、且 map->base 变换可用后，
# 再启动真正的节点命令（"$@"）。防止 m-explore 在地图尚空时第一轮规划得到
# 0 个前沿而永久停止（其 stop() 会停掉规划定时器）。
set -u
for _ in $(seq 1 150); do
  if python3 -c "
import sys, rospy, tf
from nav_msgs.msg import OccupancyGrid
try:
    rospy.init_node('map_ready_probe', anonymous=True)
    g = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
    if sum(1 for v in g.data if v >= 0) < 800:
        sys.exit(1)
    l = tf.TransformListener()
    l.waitForTransform('map', 'base', rospy.Time(), rospy.Duration(4))
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo "wait_map_ready: map ready, launching explore" >&2
    exec "$@"
  fi
  sleep 2
done
echo "wait_map_ready: timeout, launching explore anyway" >&2
exec "$@"
