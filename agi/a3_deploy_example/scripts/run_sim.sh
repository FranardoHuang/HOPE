#!/usr/bin/env bash
# Terminal 1: start the A3 ping-pong MuJoCo sim (iceoryx + GUI). Run in distrobox hope.
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/aimrt_mujoco_sim_source/aimrt_mujoco_sim/build/install/bin || exit 1
pkill -x iox-roudi 2>/dev/null; sleep 1     # clear any stale iceoryx daemon
exec ./start_a3_pingpong_iceoryx.sh
