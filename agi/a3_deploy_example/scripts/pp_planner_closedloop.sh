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
# PP_VIEWER=1 opens the MuJoCo viewer (needs a display) instead of headless EGL —
# same conductor, same PASS criteria, you just get to WATCH it.
GL_ENV="MUJOCO_GL=egl"
[ "${PP_VIEWER:-0}" = "1" ] && GL_ENV=""
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; $GL_ENV A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
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

echo "[v] fake_ball_publisher: 10-placement coverage sweep (5 fh + 5 bh, alternating;"
echo "    solved+sim-verified 2026-07-06 against the trained hitter boxes):"
echo "      fh arrive y -0.65..-0.20 (box [-0.74,-0.14]) z=0.725, tts@plane ~1.26s"
echo "      bh arrive y +0.02..+0.34 (box [-0.17,0.43]; runner fh/bh split needs y>=0)"
echo "         z=1.008 (box [0.93,1.13] — old vz=3.6 serve arrived 20cm BELOW box), ~1.63s"
# Serve design notes (2026-07-06, replaces the single fh/bh pair):
#   * fh vy SIGN FIX finally landed: legacy +0.10 arrived y=-0.13 = fh box EDGE (the -0.10
#     fix was flagged 2026-07-03 but never applied here). Sweep vy values are solved per
#     placement with the exact publisher physics (drag 0.05, rest 0.85/0.85, dt=1/300).
#   * fh base kinematics [3.2,-0.24,0.5,-2.0,vy,3.6]; bh RETUNED to vx=-1.6 vz=5.0 so the
#     post-bounce arc crosses the x=1.0 plane at trained backhand height.
#   * bh y sweep stays >= +0.02: the runner classifies fh/bh by base-rel target y sign
#     (pp_reference_clock.hpp swing_sign_from_target_y) — negative-y backhands would
#     engage as forehands. The bh box's negative-y part is unreachable by construction.
#   * Order fh1,bh1,fh2,bh2,...: the default 3-point conductor samples 3 distinct
#     placements; PP_POINTS=10 sweeps all 10 (serve<->point sync is deliberately loose —
#     serves keep cycling during stand/reset and a point takes whichever arrives next).
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; source $WS/install/local_setup.bash 2>/dev/null; ros2 run hope_bringup fake_ball_publisher --ros-args \
  -p serves:='[3.2,-0.24,0.5,-2.0,-0.372,3.6, 3.2,0.24,0.5,-1.6,-0.160,5.0, 3.2,-0.24,0.5,-2.0,-0.270,3.6, 3.2,0.24,0.5,-1.6,-0.102,5.0, 3.2,-0.24,0.5,-2.0,-0.168,3.6, 3.2,0.24,0.5,-1.6,-0.044,5.0, 3.2,-0.24,0.5,-2.0,-0.066,3.6, 3.2,0.24,0.5,-1.6,0.015,5.0, 3.2,-0.24,0.5,-2.0,0.036,3.6, 3.2,0.24,0.5,-1.6,0.073,5.0]' \
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
echo "--- FALL GUARD events (want NONE — the per-point 'p' aborts are deliberate) ---"
grep -anE "FALL GUARD|-> MOTION|-> PD_STAND" /tmp/pp_runner.log | head -16
echo "--- planner status distribution ---"
grep -aoE "PLANNER: [a-z_]+\]" /tmp/pp_runner.log | sort | uniq -c
echo "--- planner side: valid plans? ---"
grep -acE "" /tmp/pp_planner.log | xargs -I{} echo "planner log lines: {}"
grep -aiE "error|traceback" /tmp/pp_planner.log | head -3
echo "--- pelvis z after stand (min in the MOTION window) ---"
grep -aoE "^-?[0-9]+\.[0-9]+" /tmp/pp_pelvisz.log | awk 'NR==1{m=$1;mx=$1}{if($1<m)m=$1;if($1>mx)mx=$1;l=$1}END{printf "min=%.3f max=%.3f last=%.3f n=%d\n",m,mx,l,NR}'
echo "==========================================================="
