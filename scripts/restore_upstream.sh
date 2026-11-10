#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root/src/ros_unitree"
revision=0c7b48ade880fd9096d196123b3f9c98ea1131b3
if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/macc-n/ros_unitree.git "$source_dir"
  git -C "$source_dir" checkout "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
  echo 'Upstream revision differs; inspect it before applying the local patch.' >&2
  exit 1
fi
if git -C "$source_dir" apply --reverse --check "$project_root/patches/ros_unitree-noetic.patch" 2>/dev/null; then
  echo 'Local patch already applied.'
else
  git -C "$source_dir" apply --check "$project_root/patches/ros_unitree-noetic.patch"
  git -C "$source_dir" apply "$project_root/patches/ros_unitree-noetic.patch"
fi
