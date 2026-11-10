#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/activate.sh"
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  if [[ -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  fi
fi
_unitree_demo_mode="${1:-mapping}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$_unitree_demo_mode" != mapping && "$_unitree_demo_mode" != navigation ]]; then
  echo "Usage: $0 [mapping|navigation] [gui:=false] [rviz:=false]" >&2
  exit 2
fi
exec roslaunch unitree_local_demo demo.launch mode:="$_unitree_demo_mode" "$@"
