#!/usr/bin/env bash
# Terminal 3: snap the sim robot back to the upright 'stand' keyframe. Run in distrobox hope.
source /opt/ros/jazzy/setup.bash
INSTALL=~/workspace/HOPE/agi/aimrt_mujoco_sim_source/aimrt_mujoco_sim/build/install
export AMENT_PREFIX_PATH="$INSTALL:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$INSTALL/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INSTALL/lib/python3.12/site-packages:${PYTHONPATH:-}"
exec ros2 topic pub --once /sim/a3/reset mujoco_sim_msgs/msg/SimReset \
  "{mode: 1, keyframe_id: 0, set_base: false, set_base_twist: false, set_joints: false, zero_all_velocities: true, clear_ctrl: false}"
