#!/usr/bin/env bash
# Gate 2.5 harness: sim + (optional oracle bridge) + the transition-matrix conductor.
# Usage: run_gate25.sh [--oracle]
set +e
GEAR=/home/dongc1/workspace/HOPE/agi/a3_deploy_example
source /opt/ros/jazzy/setup.bash 2>/dev/null

echo "[g25] cleanup"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "oracle_pose_bridge" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* /dev/shm/pp_oracle_pelvis 2>/dev/null
sleep 1

echo "[g25] sim up"
cd "$GEAR"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; MUJOCO_GL=egl A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
for i in $(seq 1 40); do
  grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null && break
  sleep 1
done
grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null || { echo "[g25] SIM FAIL"; exit 1; }

if [[ "${1:-}" == "--oracle" ]]; then
  echo "[g25] oracle bridge up (real base feedback)"
  setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; bash $GEAR/scripts/run_oracle.sh" >/tmp/pp_oracle.log 2>&1 &
  sleep 3
fi

SIM_INSTALL=/home/dongc1/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install
source "$SIM_INSTALL/share/mujoco_sim_msgs/local_setup.bash" 2>/dev/null
export AMENT_PREFIX_PATH="$SIM_INSTALL${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$SIM_INSTALL/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$GEAR/dist/a3_deploy_x86_64"
python3 $GEAR/scripts/pp_gate25.py "${1:-}"

echo "[g25] stopping"
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null
pkill -9 -f "oracle_pose_bridge" 2>/dev/null
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null
pkill -9 -x iox-roudi 2>/dev/null
rm -f /dev/shm/iox1_0_* 2>/dev/null
exit 0
