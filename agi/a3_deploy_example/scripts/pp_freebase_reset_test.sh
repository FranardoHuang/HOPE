#!/usr/bin/env bash
# Decisive test: free-base sim + reset-to-stand-keyframe + MOTION policy.
# Confirms the IMU-frame bug was just "sim never reset to stand keyframe".
# Run: distrobox enter hope -- bash scripts/pp_freebase_reset_test.sh
# NOTE: no `set -u` — sourcing ROS2 jazzy setup.bash references unbound vars and would fatally exit.
set -o pipefail
GEAR=/home/dongc1/workspace/HOPE/agi/a3_deploy_example
DIST="$GEAR/dist/a3_deploy_x86_64"
set +u; source /opt/ros/jazzy/setup.bash 2>/dev/null || true; set +u

echo "[test] cleanup"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null || true
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null || true
pkill -9 -x iox-roudi 2>/dev/null || true
rm -f /dev/shm/iox1_0_* 2>/dev/null || true
sleep 1

echo "[test] start free-base sim"
cd "$GEAR"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
for i in $(seq 1 30); do grep -qi "will wait for shutdown" /tmp/pp_sim.log 2>/dev/null && break; sleep 2; done
echo "[test] sim up? $(grep -ci 'will wait for shutdown' /tmp/pp_sim.log)"

echo "[test] start runner (warmup 9s PD_STAND, then MOTION); obs-csv"
cd "$DIST"
rm -f obs_reset.csv
RUN_LEVEL="${RUN_LEVEL:-1}"; RUN_GAIN="${RUN_GAIN:-1.0}"; RUN_EXTRA="${RUN_EXTRA:-}"
echo "[test] runner level=$RUN_LEVEL gain=$RUN_GAIN extra='$RUN_EXTRA'"
setsid bash -c "cd '$DIST' && A3_SOURCE_ROBOT_ENV=0 LD_LIBRARY_PATH=/opt/ros/jazzy/lib A3_TRANSPORT=iceoryx timeout 24 ./run_a3_pingpong.sh --start motion --level $RUN_LEVEL --official-stand --auto-leg-hold --gain-scale $RUN_GAIN $RUN_EXTRA --warmup-sec 9 --obs-csv obs_reset.csv" >/tmp/pp_runner.log 2>&1 < /dev/null &

echo "[test] fire reset_sim.sh x3 during warmup (stand keyframe, identity pelvis)"
sleep 3; "$GEAR/scripts/reset_sim.sh" >/tmp/pp_reset.log 2>&1 || echo "reset1 rc=$?"
sleep 2; "$GEAR/scripts/reset_sim.sh" >>/tmp/pp_reset.log 2>&1 || echo "reset2 rc=$?"
sleep 2; "$GEAR/scripts/reset_sim.sh" >>/tmp/pp_reset.log 2>&1 || echo "reset3 rc=$?"

echo "[test] waiting for runner to finish..."
for i in $(seq 1 30); do pgrep -f a3_deploy_onnx_ref_pingpong >/dev/null 2>&1 || break; sleep 2; done

echo "[test] cleanup sim"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null || true
pkill -9 -x iox-roudi 2>/dev/null || true
echo "[test] DONE. reset log:"; tail -3 /tmp/pp_reset.log 2>/dev/null
