#!/usr/bin/env bash
# FULL sim-to-real rehearsal, headless:
#   fake_ball -> REAL hope_planner (sim profile, publishes flats) -> C++ runner (--planner,
#   oracle base from /sim/a3/pelvis_pose via... NO — external_base from the planner's
#   /a3/base_pose_flat, exactly the hardware wiring) -> AGI MuJoCo sim (iceoryx body-drive)
# The headless standing problem is solved by pty key injection: reset (stand keyframe)
# right before 's' (PD_STAND official gains), reset again while held, then 'm' (MOTION).
set +e
GEAR=/home/dongc1/workspace/HOPE/agi/a3_deploy_example
DIST="$GEAR/dist/a3_deploy_x86_64"
WS=/home/dongc1/workspace/HOPE/hope_ws
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$WS/install/local_setup.bash" 2>/dev/null

echo "[v] cleanup"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "hope_planner_node" 2>/dev/null
pkill -9 -f "fake_ball_publisher" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* 2>/dev/null
sleep 1

echo "[v] sim up (iceoryx body-drive + ros2 /sim/a3/pelvis_pose)"
cd "$GEAR"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; MUJOCO_GL=egl A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
for i in $(seq 1 40); do
  grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null && break
  sleep 1
done
grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null || { echo "[v] SIM FAIL"; tail -8 /tmp/pp_sim.log; exit 1; }

echo "[v] REAL hope_planner (base yaml + sim overlay + drag 0.05 verification physics)"
# drag_k 0.05 on BOTH planner and fake_ball: self-consistent physics under which classic
# lobs survive to the plane at trained heights (0.15 arena drag kills every slow lob at
# z~0 -> gate rejects; the arena value matters for the venue, not for loop verification).
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; source $WS/install/local_setup.bash 2>/dev/null; ros2 run hope_planner hope_planner_node --ros-args \
  --params-file $WS/src/hope_planner/config/hope_planner.yaml \
  --params-file $WS/src/hope_planner/config/hope_planner.sim.yaml \
  -p drag_k:=0.05" >/tmp/pp_planner.log 2>&1 &

echo "[v] fake_ball_publisher (probed serve: post-bounce z=0.72 at x=1.0 = the adaptive plane"
echo "    for a robot at x=0.33; tts@plane ~1.26s)"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; source $WS/install/local_setup.bash 2>/dev/null; ros2 run hope_bringup fake_ball_publisher --ros-args \
  -p serve_forehand:='[3.2, -0.24, 0.5, -2.0, 0.10, 3.6]' \
  -p serve_backhand:='[3.2, 0.24, 0.5, -2.0, -0.10, 3.6]' \
  -p drag_k:=0.05 \
  -p pause_s:=4.0" >/tmp/pp_ball.log 2>&1 &

setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic echo --field pose.position.z /sim/a3/pelvis_pose" >/tmp/pp_pelvisz.log 2>&1 &

echo "[v] flat topics alive?"
sleep 4
timeout 5 bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic hz /racket/command_flat 2>&1 | head -2" | tail -1
timeout 5 bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic hz /a3/base_pose_flat 2>&1 | head -2" | tail -1

echo "[v] prewarm ros2 pub discovery (a straggler reset landing in MOTION teleports the robot"
echo "    to the keyframe mid-hold = the documented reset-during-MOTION trap; observed once)"
"$GEAR/scripts/reset_sim.sh" >/tmp/pp_reset_prewarm.log 2>&1   # blocks until published; warms discovery

echo "[v] STATE-DRIVEN conductor: owns the runner pty, resets until VERIFIED standing,"
echo "    then s -> verify -> m; watches 65s of planner-driven MOTION"
# mujoco_sim_msgs (SimReset) overlay for the conductor's persistent reset publisher
SIM_INSTALL=/home/dongc1/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install
source "$SIM_INSTALL/share/mujoco_sim_msgs/local_setup.bash" 2>/dev/null
export AMENT_PREFIX_PATH="$SIM_INSTALL${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$SIM_INSTALL/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$DIST"
python3 $GEAR/scripts/pp_planner_conductor.py

echo "[v] stopping"
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "hope_planner_node" 2>/dev/null
pkill -9 -f "fake_ball_publisher" 2>/dev/null
pkill -9 -f "ros2 topic echo" 2>/dev/null
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* 2>/dev/null

echo "========================= RESULTS ========================="
echo "--- engages + completions (want >=2 engages, tts0 transferred) ---"
grep -aE "\[pp engage\]|swing complete" /tmp/pp_runner.log | head -12
echo "--- falls AFTER MOTION entry (want NONE after the 'm' key) ---"
grep -anE "FALL GUARD|-> MOTION|-> PD_STAND" /tmp/pp_runner.log | head -12
echo "--- planner status distribution ---"
grep -aoE "PLANNER: [a-z_]+\]" /tmp/pp_runner.log | sort | uniq -c
echo "--- planner side: valid plans? ---"
grep -acE "" /tmp/pp_planner.log | xargs -I{} echo "planner log lines: {}"
grep -aiE "error|traceback" /tmp/pp_planner.log | head -3
echo "--- pelvis z after stand (min in the MOTION window) ---"
grep -aoE "^-?[0-9]+\.[0-9]+" /tmp/pp_pelvisz.log | awk 'NR==1{m=$1;mx=$1}{if($1<m)m=$1;if($1>mx)mx=$1;l=$1}END{printf "min=%.3f max=%.3f last=%.3f n=%d\n",m,mx,l,NR}'
echo "==========================================================="
