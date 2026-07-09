#!/usr/bin/env bash
# Gate 3 (2026-07-08 redesign, 110-D hitter_pure) — the CONTINUOUS-RALLY deploy rehearsal:
#   fake_ball (10-placement sweep) -> REAL hope_planner (sim profile, publishes the flats)
#   -> C++ runner --planner (external_base from /a3/base_pose_flat = the hardware wiring)
#   -> AGI MuJoCo sim (iceoryx body-drive)
# vs the legacy pp_planner_closedloop.sh (per-point: serve -> ONE return -> operator 'p'
# -> sim reset), this gate NEVER resets after MOTION entry: the robot must return
# PP_SERVES consecutive serves from its station — the real §9.3 demo regime for the
# station-keeping hitter_pure generation (post-swing recovery INTO the next engage,
# station-drift accumulation, fh<->bh alternation from the walked pose).
# Verdicts: per-serve table + /tmp/pp_rally_report.json; per-tick 110-D obs in
# /tmp/pp_obs.csv (analyze: scripts/pp_rally_report.py). PP_VIEWER=1 to watch.
# Knobs: PP_SERVES (12) PP_PAUSE_S (4.0) PP_RESET_Y (0.0) PP_DROPOUT_AT (0 = off)
#        PP_EXTRA_ARGS (extra runner flags, e.g. "--hold-recover 1.2")
set +e
GEAR=/home/dongc1/workspace/HOPE/agi/a3_deploy_example
DIST="$GEAR/dist/a3_deploy_x86_64"
WS=/home/dongc1/workspace/HOPE/hope_ws
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$WS/install/local_setup.bash" 2>/dev/null

echo "[g3r] cleanup"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "hope_planner_node" 2>/dev/null
pkill -9 -f "fake_ball_publisher" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* /tmp/pp_obs.csv /tmp/pp_rally_report.json 2>/dev/null
sleep 1

echo "[g3r] sim up (iceoryx body-drive + ros2 /sim/a3/pelvis_pose)"
cd "$GEAR"
GL_ENV="MUJOCO_GL=egl"
[ "${PP_VIEWER:-0}" = "1" ] && GL_ENV=""
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; $GL_ENV A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
for i in $(seq 1 40); do
  grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null && break
  sleep 1
done
grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null || { echo "[g3r] SIM FAIL"; tail -8 /tmp/pp_sim.log; exit 1; }

echo "[g3r] REAL hope_planner (base yaml + sim overlay + drag 0.05 verification physics)"
echo "      hitter_pure profile: FIXED plane x=1.03 (paper §IV-B; robot spawns 0.33 ->"
echo "      station = 1.03-0.70 = spawn; no adaptive plane-chase feedback) + PER-SIDE aim"
echo "      (fh land_y +0.70 dtf 0.40 / bh land_y -0.30 dtf 0.35 = offline-solved so the"
echo "      demanded racket vels sit INSIDE the trained per-clip boxes, 10/10 sweep serves;"
echo "      a single aim can only satisfy one side's trained cross-court direction)"
# ⚠ hope_planner.sim.yaml sets x_hit_follow_robot:=true (legacy follow). The `-p
#   x_hit_follow_robot:=false` below OVERRIDES it by POSITION — rcl merges --params-file and -p
#   last-wins in command-line order. KEEP the -p AFTER both --params-file lines; if a --params-file
#   that sets true is ever appended after this -p, follow-mode silently re-enables = drift-fall.
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; source $WS/install/local_setup.bash 2>/dev/null; ros2 run hope_planner hope_planner_node --ros-args \
  --params-file $WS/src/hope_planner/config/hope_planner.yaml \
  --params-file $WS/src/hope_planner/config/hope_planner.sim.yaml \
  -p drag_k:=0.05 -p x_hit_follow_robot:=false -p x_hit:=1.03 \
  -p target_land_y_fh:=0.70 -p target_land_y_bh:=-0.30 \
  -p delta_t_flight_fh:=0.40 -p delta_t_flight_bh:=0.35" >/tmp/pp_planner.log 2>&1 &

echo "[g3r] fake_ball_publisher: 10-placement coverage sweep (5 fh + 5 bh, alternating;"
echo "      re-solved 2026-07-08 vs the exact publisher physics for the 110-D hitter_pure"
echo "      bands: fh arrive y -0.60..-0.20 z=0.783 (box y[-0.65,-0.15] z[0.67,0.97],"
echo "      center 0.82 — the old vz=3.6 serves arrived at the 0.725 LOW EDGE);"
echo "      bh arrive y +0.02..+0.34 z=1.031 (box y[-0.05,0.45] z[0.88,1.18], center 1.03)"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; source $WS/install/local_setup.bash 2>/dev/null; ros2 run hope_bringup fake_ball_publisher --ros-args \
  -p serves:='[3.2,-0.24,0.5,-2.0,-0.332,4.05, 3.2,0.24,0.5,-1.6,-0.160,5.0, 3.2,-0.24,0.5,-2.0,-0.240,4.05, 3.2,0.24,0.5,-1.6,-0.102,5.0, 3.2,-0.24,0.5,-2.0,-0.147,4.05, 3.2,0.24,0.5,-1.6,-0.044,5.0, 3.2,-0.24,0.5,-2.0,-0.055,4.05, 3.2,0.24,0.5,-1.6,0.015,5.0, 3.2,-0.24,0.5,-2.0,0.037,4.05, 3.2,0.24,0.5,-1.6,0.073,5.0]' \
  -p drag_k:=0.05 \
  -p pause_s:=${PP_PAUSE_S:-4.0}" >/tmp/pp_ball.log 2>&1 &

setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic echo --field pose.position.z /sim/a3/pelvis_pose" >/tmp/pp_pelvisz.log 2>&1 &

echo "[g3r] flat topics alive?"
sleep 4
timeout 5 bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic hz /racket/command_flat 2>&1 | head -2" | tail -1
timeout 5 bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 topic hz /a3/base_pose_flat 2>&1 | head -2" | tail -1

echo "[g3r] prewarm ros2 pub discovery (straggler-reset-in-MOTION trap)"
"$GEAR/scripts/reset_sim.sh" >/tmp/pp_reset_prewarm.log 2>&1

echo "[g3r] RALLY conductor: stand -> m ONCE, then ${PP_SERVES:-12} serves, no resets"
SIM_INSTALL=/home/dongc1/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install
source "$SIM_INSTALL/share/mujoco_sim_msgs/local_setup.bash" 2>/dev/null
export AMENT_PREFIX_PATH="$SIM_INSTALL${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$SIM_INSTALL/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$DIST"
python3 "$GEAR/scripts/pp_rally_conductor.py"
RC=$?

echo "[g3r] stopping"
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "hope_planner_node" 2>/dev/null
pkill -9 -f "fake_ball_publisher" 2>/dev/null
pkill -9 -f "ros2 topic echo" 2>/dev/null
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* 2>/dev/null

echo "========================= RESULTS (rally) ========================="
echo "--- per-serve verdicts + SUMMARY are above ([rally] lines); JSON: /tmp/pp_rally_report.json ---"
echo "--- engage/complete/recovery event stream ---"
grep -aE "\[pp engage\]|swing complete|recovery done" /tmp/pp_runner.log | head -40
echo "--- FALL GUARD / mode flips (want: exactly one -> MOTION, no FALL GUARD) ---"
grep -anE "FALL GUARD|-> MOTION|-> PD_STAND|-> PASSIVE" /tmp/pp_runner.log | head -12
echo "--- gate rejections (first 8; z_w=0.00 late-in-flight is a dead ball = normal) ---"
grep -aE "\[pp gate\] REJECT" /tmp/pp_runner.log | head -8
echo "--- planner status distribution ---"
grep -aoE "PLANNER: [a-z_]+\]" /tmp/pp_runner.log | sort | uniq -c
echo "--- localization health (want NO stale-mocap warns outside a deliberate DROPOUT) ---"
grep -acE "NO FRESH mocap base sample" /tmp/pp_runner.log | xargs -I{} echo "stale-base warns: {}"
echo "--- pelvis z (min/max over the run) ---"
grep -aoE "^-?[0-9]+\.[0-9]+" /tmp/pp_pelvisz.log | awk 'NR==1{m=$1;mx=$1}{if($1<m)m=$1;if($1>mx)mx=$1;l=$1}END{printf "min=%.3f max=%.3f last=%.3f n=%d\n",m,mx,l,NR}'
echo "--- deep-dive: python3 $GEAR/scripts/pp_rally_report.py /tmp/pp_obs.csv /tmp/pp_rally_report.json ---"
echo "==================================================================="
exit $RC
