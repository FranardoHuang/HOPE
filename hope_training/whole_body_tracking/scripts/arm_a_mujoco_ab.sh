#!/usr/bin/env bash
# ARM-A clean A/B on the DEPLOY path (MuJoCo deploy-faithful, implicit PD, NO
# ee_body_pos training guard). Baseline = deployed model_11400_hopex.onnx vs the
# ARM-A peak model_14600. Two modes:
#   (1) default multi-swing  -> strike composite/pos/vel/falls = "did strike regress?"
#   (2) --df-hold-steps 1000 -> 20 s hold survival = "did the hold get worse?"
# (The hold-MARGIN discriminator is the AGI Gate 2.5, where 11400 falls ~5 s; this
#  training-side hold only confirms the new model doesn't LOSE the 20 s hold.)
set -u
cd ~/workspace/HOPE/hope_training/whole_body_tracking
PY=../.venv-motion/bin/python
FH=artifacts/hope_forehand_hopex/motion.npz
BH=artifacts/hope_backhand_hopex/motion.npz
OLD=~/workspace/HOPE/agi/a3_deploy_example/assets/a3_runtime/models/model_11400_hopex.onnx
NEW=/tmp/model_14600_armA.onnx
OUT=/tmp/arm_a_mujoco; mkdir -p $OUT

run() {  # $1=label $2=onnx $3...=extra flags
  local label=$1 onnx=$2; shift 2
  echo "########## $label ##########"
  $PY scripts/mujoco_eval_onnx.py --onnx "$onnx" --motion-files $FH $BH \
    --pd-mode implicit --deploy-faithful --noise-scales 0.0 "$@" 2>&1 \
    | sed -n '/DEPLOY-FAITHFUL report/,/====/p'
  echo
}

echo "===================== (1) STRIKE (default multi-swing) ====================="
run "OLD_11400 strike"  "$OLD" --steps 1500
run "NEW_14600 strike"  "$NEW" --steps 1500

echo "===================== (2) 20s HOLD (df-hold-steps 1000) ====================="
run "OLD_11400 hold20s" "$OLD" --df-hold-steps 1000 --df-rest-steps 50 --steps 1100
run "NEW_14600 hold20s" "$NEW" --df-hold-steps 1000 --df-rest-steps 50 --steps 1100
echo "MUJOCO_AB_DONE"
