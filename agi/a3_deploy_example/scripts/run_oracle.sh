#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--print-paths" ]]; then
  echo "oracle_script=${SCRIPT_DIR}/oracle_pose_bridge.py"
  echo "oracle_topic=${A3_ORACLE_TOPIC:-/sim/a3/pelvis_pose}"
  echo "oracle_shm=${A3_ORACLE_SHM:-/dev/shm/pp_oracle_pelvis}"
  exit 0
fi

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

exec /usr/bin/python3 "${SCRIPT_DIR}/oracle_pose_bridge.py" \
  --topic "${A3_ORACLE_TOPIC:-/sim/a3/pelvis_pose}" \
  --shm "${A3_ORACLE_SHM:-/dev/shm/pp_oracle_pelvis}" \
  "$@"
