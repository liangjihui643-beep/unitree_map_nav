#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/activate.sh"
cd "$UNITREE_MAP_NAV_ROOT"
catkin_make -DPYTHON_EXECUTABLE="$UNITREE_MAP_NAV_ROOT/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -j2 -l2 "$@" 2>&1 | tee "$UNITREE_MAP_NAV_ROOT/logs/build.log"
