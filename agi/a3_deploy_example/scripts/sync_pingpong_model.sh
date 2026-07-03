#!/usr/bin/env bash
# One-command sync of a freshly exported deploy-parity ONNX into the AGI deploy tree.
#
# Usage (repo root or anywhere):
#   bash scripts/sync_pingpong_model.sh <exported/policy.onnx> <asset_name>
# Example:
#   bash scripts/sync_pingpong_model.sh \
#     ~/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-03_13-32-07/exported/policy.onnx \
#     model_7000_hopex
#
# What it does (the PINGPONG_NEW_CHECKPOINT_TUTORIAL re-sync steps 1, 2, 6a):
#   1. stages the ONNX as a durable asset: assets/a3_runtime/models/<asset_name>.onnx
#      (NEVER hand-copy into dist/ — builds wipe dist/)
#   2. rewrites onnx.model_path in src/.../config/a3_runtime_config.pingpong.yaml
#   3. runs the C++<->Python parity gate (scripts/pingpong_parity/run_parity.sh)
#
# What it does NOT do (run these yourself, in order):
#   4. verify the scripted targets still sit inside the model's training boxes:
#      pp_policy.hpp racket_{pos,vel}_w_clip  vs  cfg/task/HOPEPingPongDeployParity.yaml
#      pos/vel_range_per_clip (compile-time constants — edit + rebuild if the boxes moved)
#   5. rebuild BOTH dists (x86 in the `hope` distrobox, rockchip on the HOST with Docker):
#      bash scripts/build_a3_deploy_pkg.sh --arch x86_64 \
#        --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
#   6. training-side deploy-faithful MuJoCo gate + AGI free-base watch (see tutorial §gates)
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> a3_deploy_example root

ONNX="${1:?usage: sync_pingpong_model.sh <exported/policy.onnx> <asset_name>}"
NAME="${2:?usage: sync_pingpong_model.sh <exported/policy.onnx> <asset_name>}"
NAME="${NAME%.onnx}"
CFG=src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
ASSET="assets/a3_runtime/models/$NAME.onnx"

[ -f "$ONNX" ] || { echo "[sync] onnx not found: $ONNX"; exit 1; }

# 1. stage durable asset
cp "$ONNX" "$ASSET"
echo "[sync] staged: $ASSET ($(stat -c%s "$ASSET") bytes)"

# 2. point the runtime config at it
sed -i -E "s|^(  model_path: \.\./\.\./\.\./\.\./assets/a3_runtime/models/).*\.onnx$|\1$NAME.onnx|" "$CFG"
grep -n "model_path" "$CFG" | head -1

# 3. C++ <-> Python parity gate on the new model
echo "[sync] running parity gate..."
MODEL="$ASSET" bash scripts/pingpong_parity/run_parity.sh

echo ""
echo "[sync] DONE. Next steps:"
echo "  - check pp_policy.hpp scripted targets vs the model's training boxes (see header)"
echo "  - rebuild dists:  bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg $CFG"
echo "                    (+ --arch rockchip on the HOST before hardware)"
echo "  - gates: mujoco_eval_onnx.py --deploy-faithful (training side), pp_freebase_watch.sh (AGI sim)"
