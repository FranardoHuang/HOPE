#!/usr/bin/env bash
# Terminal 4 (mode C only): bridge /sim/a3/pelvis_pose -> /dev/shm for the runner. distrobox hope.
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 ~/workspace/HOPE/agi/a3_deploy_example/scripts/oracle_pose_bridge.py \
  --topic /sim/a3/pelvis_pose --shm /dev/shm/pp_oracle_pelvis "$@"
