#!/usr/bin/env bash
set -eo pipefail
competition_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$competition_root/activate.sh"
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi
competition_mode="${1:-mapping}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$competition_mode" != mapping && "$competition_mode" != navigation ]]; then
  echo "Usage: $0 [mapping|navigation] [gui:=false] [rviz:=false] [map_file:=/path/map.yaml]" >&2
  exit 2
fi
# roslaunch checks map_file overrides too; the default is deliberately separate.
if [[ "$competition_mode" == navigation && ! -f "$competition_root/src/unitree_competition/maps/competition.yaml" && " $* " != *" map_file:="* ]]; then
  echo '先在比赛场景建图并保存 competition.yaml，或传入 map_file:=完整路径。参见比赛包 README。' >&2
  exit 2
fi
exec roslaunch unitree_competition competition.launch mode:="$competition_mode" "$@"
