#!/usr/bin/env bash
# Terminal 2: launch the deploy runner. Usage: ./run_mode.sh A|B|C [warmup_sec] [gain_scale]
# Runs in distrobox hope. It:
#   1) starts the runner holding the robot upright (PD_STAND),
#   2) AUTO-resets the sim to the stand keyframe twice during warmup (no Terminal 3),
#   3) AUTO-switches to MOTION (the policy) after warmup,
#   4) runs until you press Ctrl-C.
set -e
MODE="${1:?usage: run_mode.sh A|B|C [warmup_sec] [gain_scale]}"
WARMUP="${2:-10}"       # seconds of PD_STAND hold before auto-switch to MOTION
GAIN="${3:-1.0}"        # MOTION gain scale; <1.0 tames the sim's explicit-PD instability
SCRIPTS="$HOME/workspace/HOPE/agi/a3_deploy_example/scripts"
source /opt/ros/jazzy/setup.bash
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64 || exit 1
RT=config/a3_runtime_config.pingpong.yaml
case "$MODE" in
  A|a) FLAG="--loc-mode fabricated"; CSV=/tmp/obs_A_fabricated.csv;;
  B|b) FLAG="--perfect-tracking";    CSV=/tmp/obs_B_perfect.csv;;
  C|c) FLAG="--oracle-pelvis";       CSV=/tmp/obs_C_oracle.csv;;
  *) echo "mode must be A, B, or C"; exit 2;;
esac
echo "============================================================"
echo " MODE $MODE : $FLAG  gain_scale=$GAIN  ->  $CSV"
echo " holding upright + auto-reset, then AUTO-MOTION after ${WARMUP}s."
echo " WATCH: gravZ must be ~ -1.0 before it says 'warmup done -> MOTION'."
echo " Ctrl-C to stop."
echo "============================================================"

# Runner in background, stdin detached so it does not grab the tty (no keys needed).
./a3_deploy_onnx_ref_pingpong --runtime-cfg "$RT" \
    --start motion --warmup-sec "$WARMUP" --gain-scale "$GAIN" --official-stand \
    $FLAG --obs-csv "$CSV" < /dev/null &
RUNNER=$!
trap 'kill $RUNNER 2>/dev/null' INT TERM

# Auto-stand: fire the keyframe reset a couple of times during the warmup window.
( sleep 2; "$SCRIPTS/reset_sim.sh" >/dev/null 2>&1 || true
  sleep 3; "$SCRIPTS/reset_sim.sh" >/dev/null 2>&1 || true
  echo "[run_mode] auto-reset sent (x2); robot should read gravZ ~ -1 now" ) &

wait $RUNNER
