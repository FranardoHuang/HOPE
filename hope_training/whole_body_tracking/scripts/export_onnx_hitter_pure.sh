#!/usr/bin/env bash
# Export policy.onnx from a HOPEPingPongHitterPure (110-D hitter_pure) checkpoint, then KILL
# play.py (it enters an infinite sim-step loop after the export; see export_onnx_hitter.sh).
#
# Usage (inside the `grasping` distrobox, from whole_body_tracking/):
#   bash scripts/export_onnx_hitter_pure.sh <run_dir> [model_XXXX.pt]
# With no checkpoint arg, the NEWEST model_*.pt in the run dir is used.
#
# MOTION CLIPS: pinned to the hopex RE-GROUNDED clips; override with FH=/path BH=/path.
# The export MUST bake the same clips as training (verify against the run's wandb debug.log).
#
# 110-specific metadata the C++ runner needs (attach_onnx_metadata bakes them when the env's
# target_mode == hitter_pure): actor_obs_contract=hitter_pure, clip_seg_lengths /
# clip_strike_phases (tts clock), hitter_pure_pos_range_per_clip / _vel_range_per_clip /
# _base_target_range (station derivation + engage gates), ref_reach_offset_xy (fallback).
# Output: <run_dir>/exported/policy.onnx
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: export_onnx_hitter_pure.sh <run_dir> [model_XXXX.pt]}"
if [ -n "${2:-}" ]; then
  CKPT="$RUN/$2"
else
  CKPT="$(ls -t "$RUN"/model_*.pt | head -1)"
fi
[ -f "$CKPT" ] || { echo "[export] checkpoint not found: $CKPT"; exit 1; }

FH="${FH:-$PWD/artifacts/hope_forehand_hopex/motion.npz}"
BH="${BH:-$PWD/artifacts/hope_backhand_hopex/motion.npz}"
[ -f "$FH" ] && [ -f "$BH" ] || { echo "[export] motion clips missing: $FH / $BH"; exit 1; }

ONNX="$RUN/exported/policy.onnx"
LOG="/tmp/export_hitter_pure.log"
rm -f "$ONNX" "$LOG"
echo "[export] checkpoint: $CKPT"
echo "[export] clips: $FH + $BH"
echo "[export] launching play.py (setsid); ONNX -> $ONNX"

setsid bash -c "cd '$PWD' && source setup_train_env.sh >/dev/null 2>&1 && \
  hope_isaac_py scripts/play.py task=HOPEPingPongHitterPure algo=ppo headless=true \
  export_only=true num_envs=2 checkpoint='$CKPT' 'motion_file=[$FH,$BH]'" >"$LOG" 2>&1 &
PGID=$!
echo "[export] pgid=$PGID  (log: $LOG)"

for i in $(seq 1 120); do
  if [ -f "$ONNX" ]; then
    echo "[export] policy.onnx appeared after ~$((i*5))s; waiting 6s for metadata then killing."
    sleep 6
    break
  fi
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[export] play.py process group exited before onnx appeared — see $LOG"; break
  fi
  sleep 5
done

kill -TERM -"$PGID" 2>/dev/null
sleep 3
kill -KILL -"$PGID" 2>/dev/null

if [ -f "$ONNX" ]; then
  echo "[export] SUCCESS: $ONNX"
  ls -la "$RUN/exported/"
  python3 - "$ONNX" <<'PY' 2>/dev/null || echo "[export] (onnx python env missing — check hitter_pure metadata manually)"
import sys, onnx
m = onnx.load(sys.argv[1])
md = {p.key: p.value for p in m.metadata_props}
need = ("actor_obs_contract", "clip_seg_lengths", "clip_strike_phases",
        "hitter_pure_pos_range_per_clip", "hitter_pure_vel_range_per_clip",
        "hitter_pure_base_target_range")
missing = [k for k in need if k not in md]
if md.get("actor_obs_contract") != "hitter_pure":
    print(f"[export] WARNING: actor_obs_contract={md.get('actor_obs_contract')!r} != 'hitter_pure'")
if missing:
    print(f"[export] WARNING: hitter_pure deploy metadata MISSING: {missing} — C++ runner will refuse this model")
else:
    print(f"[export] hitter_pure metadata OK: dim={md.get('actor_obs_total_dim')} "
          f"boxes={md['hitter_pure_pos_range_per_clip']} station={md['hitter_pure_base_target_range']}")
PY
else
  echo "[export] FAILED — no onnx. tail of log:"; tail -30 "$LOG"; exit 1
fi
