#!/usr/bin/env bash
if [[ -z "${UNITREE_MAP_NAV_ROOT:-}" ]]; then
  UNITREE_MAP_NAV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  export UNITREE_MAP_NAV_ROOT
fi
source /opt/ros/noetic/setup.bash
source "$UNITREE_MAP_NAV_ROOT/.venv/bin/activate"

_unitree_local_root="$UNITREE_MAP_NAV_ROOT/.runtime/root"
_unitree_local_ros="$_unitree_local_root/opt/ros/noetic"
export CMAKE_PREFIX_PATH="$_unitree_local_ros${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ROS_PACKAGE_PATH="$_unitree_local_ros/share${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export LD_LIBRARY_PATH="$_unitree_local_ros/lib:$_unitree_local_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$_unitree_local_root/usr/lib/x86_64-linux-gnu/pkgconfig:$_unitree_local_ros/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CPATH="$_unitree_local_root/usr/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$_unitree_local_root/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="$_unitree_local_root/usr/bin:$_unitree_local_ros/bin:$PATH"
if [[ -f "$UNITREE_MAP_NAV_ROOT/devel/setup.bash" ]]; then
  source "$UNITREE_MAP_NAV_ROOT/devel/setup.bash"
fi
# Catkin's generated setup omits prefixes without a .catkin marker. Reapply
# the unpacked binary prefix so pluginlib can discover its package manifests.
export CMAKE_PREFIX_PATH="$_unitree_local_ros${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ROS_PACKAGE_PATH="$_unitree_local_ros/share${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export LD_LIBRARY_PATH="$_unitree_local_ros/lib:$_unitree_local_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$UNITREE_MAP_NAV_ROOT/.venv/bin:$_unitree_local_root/usr/bin:$_unitree_local_ros/bin:$PATH"

# Keep runtime data and ports separate from other robotics workspaces.
export ROS_HOME="$UNITREE_MAP_NAV_ROOT/.runtime/ros"
export ROS_LOG_DIR="$UNITREE_MAP_NAV_ROOT/logs/ros"
export ROS_MASTER_URI="${UNITREE_ROS_MASTER_URI:-http://127.0.0.1:11331}"
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME
export GAZEBO_MASTER_URI="${UNITREE_GAZEBO_MASTER_URI:-http://127.0.0.1:11351}"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$UNITREE_MAP_NAV_ROOT/models:$UNITREE_MAP_NAV_ROOT/src/ros_unitree/unitree_ros/unitree_gazebo/worlds/building_editor_models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export GAZEBO_PLUGIN_PATH="$UNITREE_MAP_NAV_ROOT/devel/lib:$_unitree_local_ros/lib:/opt/ros/noetic/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
export GAZEBO_RESOURCE_PATH="/usr/share/gazebo-11${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
export GAZEBO_LOG_PATH="$UNITREE_MAP_NAV_ROOT/logs/gazebo"
export DISABLE_ROS1_EOL_WARNINGS=1
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"
unset _unitree_local_root _unitree_local_ros
