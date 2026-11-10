#!/usr/bin/env bash
# 安装编译/运行本工作区所需的额外 ROS 依赖。
# 有 sudo 时直接用 apt 安装；没有 sudo 时下载 deb 包并解包到 .runtime/root 本地前缀。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

packages=(
  liblcm-bin
  liblcm-dev
  liblcm1
  ros-noetic-effort-controllers
  ros-noetic-global-planner
  ros-noetic-joint-trajectory-controller
  ros-noetic-slam-gmapping
)

if sudo -v 2>/dev/null; then
  sudo apt update
  sudo apt install -y "${packages[@]}"
  echo '依赖已通过 apt 安装完成。'
else
  prefix="$project_root/.runtime/root"
  debs_dir="$project_root/.runtime/debs"
  mkdir -p "$prefix" "$debs_dir"
  echo '没有 sudo，改为下载 deb 并解包到 .runtime/root（免安装方式）。'
  (cd "$debs_dir" && apt download "${packages[@]}")
  for deb in "$debs_dir"/*.deb; do
    dpkg -x "$deb" "$prefix"
  done
  echo "解包完成：$prefix"
fi
