#!/usr/bin/env bash
# ==============================================================================
# HOPE bring-up DRY-RUN smoke test: build + launch planner_imitate -> wbc_runner.
#
# HARDWARE-SAFE BY CONSTRUCTION:
#   * wbc_runner runs mode=dry_run  -> it never creates a joint-command publisher
#     and never publishes joint_msgs/JointCommand. No hardware can move.
#   * hardware_enable stays false; estop default does not matter in dry_run.
#   * planner_imitate is launched with dry_run:=false ONLY so it PUBLISHES
#     /racket/command (a target topic, not a hardware command) so we can echo it
#     and the runner can build a real obs. The planner never drives hardware.
#
# It does NOT build hardware mode, never sets mode=hardware, never enables output.
#
# Usage:
#   ./smoke_test_dry_run.sh [ONNX_PATH]
# Env overrides:
#   ROS_SETUP=/opt/ros/<distro>/setup.bash   (auto-detected if unset)
#   ONNX_PATH=/abs/path/to/exported/policy.onnx
#   RUN_SECONDS=6                            (how long to run each launch)
# ==============================================================================
# NOTE: do NOT use `set -u` — ROS 2 setup.bash references unset vars
# (e.g. AMENT_TRACE_SETUP_FILES) and would abort the script under nounset.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # hope_ws/
cd "$HERE"

ONNX_PATH="${1:-${ONNX_PATH:-$HERE/../hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/exported/policy.onnx}}"
RUN_SECONDS="${RUN_SECONDS:-6}"
CSV=/tmp/wbc_runner.csv
PIDS=()
PASS=0; FAIL=0
ok(){ echo "  [PASS] $*"; PASS=$((PASS+1)); }
no(){ echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
hdr(){ echo; echo "=== $* ==="; }
cleanup(){ for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; pkill -f planner_imitate_node 2>/dev/null; pkill -f wbc_runner_node 2>/dev/null; }
trap cleanup EXIT

# ---- 0. source ROS ----------------------------------------------------------
hdr "0. source ROS 2"
if [ -z "${ROS_SETUP:-}" ]; then
  ROS_SETUP="$(ls -1 /opt/ros/*/setup.bash 2>/dev/null | head -1)"
fi
if [ -z "${ROS_SETUP:-}" ] || [ ! -f "$ROS_SETUP" ]; then
  echo "  cannot find ROS setup.bash; set ROS_SETUP=/opt/ros/<distro>/setup.bash"; exit 2
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"; echo "  sourced $ROS_SETUP (ROS_DISTRO=${ROS_DISTRO:-?})"

# ---- 1. colcon build --------------------------------------------------------
# Use --packages-up-to so hope_msgs (the RacketCommand interface) is (re)built with
# the SAME ROS/python as the rest. Building only the two leaf packages against a
# stale hope_msgs causes a typesupport/python-version skew at launch, e.g.:
#   "Could not import 'rosidl_typesupport_c' for package 'hope_msgs'"
#   "ImportError: libpython3.X.so: cannot open shared object file"
# If you see that, your hope_msgs was built under a different python than the ROS
# you just sourced -> rebuild the whole ws under ONE ROS: colcon build.
hdr "1. colcon build --packages-up-to hope_planner hope_wbc_runner"
if colcon build --packages-up-to hope_planner hope_wbc_runner --symlink-install; then
  ok "colcon build (incl. hope_msgs)"
else
  no "colcon build"; exit 1
fi

# ---- 2. source overlay ------------------------------------------------------
hdr "2. source install/local_setup.bash"
# shellcheck disable=SC1091
source install/local_setup.bash && ok "sourced overlay" || { no "source overlay"; exit 1; }

# ---- 3. pkg list ------------------------------------------------------------
hdr "3. ros2 pkg list | grep hope_wbc_runner"
if ros2 pkg list | grep -q hope_wbc_runner; then ok "hope_wbc_runner registered"; else no "hope_wbc_runner not found"; fi
ros2 pkg executables hope_wbc_runner 2>/dev/null | grep -q wbc_runner_node && ok "wbc_runner_node executable present" || no "wbc_runner_node missing"
ros2 pkg executables hope_planner 2>/dev/null | grep -q planner_imitate_node && ok "planner_imitate_node executable present" || no "planner_imitate_node missing (rebuild hope_planner)"

# ---- 8. tests (run early; no ROS graph needed) ------------------------------
# NOTE: on this distro `colcon test` invokes the unittest runner, which does not
# collect pytest-style function tests (reports "Ran 0 tests"). Direct pytest is
# the reliable path for these pure-core tests, so we use it here.
hdr "8. unit tests (pytest) for the new packages"
for pkg in hope_planner hope_wbc_runner; do
  if PYTHONPATH="src/$pkg" python3 -m pytest "src/$pkg/test" -q >/tmp/pytest_$pkg.log 2>&1; then
    ok "pytest $pkg ($(grep -oE '[0-9]+ passed' /tmp/pytest_$pkg.log | head -1))"
  else
    no "pytest $pkg (see /tmp/pytest_$pkg.log)"
  fi
done

# ---- onnxruntime gate for the runner steps ----------------------------------
hdr "onnxruntime check (required by wbc_runner, even in dry-run)"
if python3 -c "import onnxruntime" 2>/dev/null; then
  ok "onnxruntime importable in ROS python"
  HAVE_ORT=1
else
  echo "  [SKIP] onnxruntime not in the ROS python -> skipping runner launch (steps 5-7)."
  echo "         install it:  python3 -m pip install onnxruntime   (in the ROS python env)"
  HAVE_ORT=0
fi
[ -f "$ONNX_PATH" ] && ok "onnx model present: $ONNX_PATH" || { no "onnx model NOT found: $ONNX_PATH"; HAVE_ORT=0; }

# ---- 4. launch planner_imitate (PUBLISH /racket/command; still hardware-safe)
hdr "4. launch planner_imitate (publishes /racket/command, level 1 slow forehand)"
rm -f "$CSV"
ros2 launch hope_planner planner_imitate.launch.py dry_run:=false level:=1 > /tmp/planner_imitate.log 2>&1 &
PIDS+=($!); sleep 3
grep -q "planner_imitate started" /tmp/planner_imitate.log && ok "planner_imitate launched" || no "planner_imitate did not start (see /tmp/planner_imitate.log)"

# ---- 6. echo /racket/command ------------------------------------------------
# Two gotchas handled here:
#  (a) planner_imitate publishes BEST_EFFORT; `ros2 topic echo` defaults to RELIABLE (incompatible
#      with a best-effort publisher) -> pass --qos-reliability best_effort.
#  (b) a stale ros2 daemon from a different ROS/python serves incompatible endpoint info over xmlrpc
#      ("unknown tag 'rclpy.endpoint_info.TopicEndpointInfo'") -> stop it so it restarts clean.
hdr "6. ros2 topic echo /racket/command (one message, best_effort QoS)"
ros2 daemon stop >/dev/null 2>&1 || true
if timeout 6 ros2 topic echo --once --qos-reliability best_effort /racket/command > /tmp/racket_cmd.txt 2>&1; then
  grep -q "frame_id" /tmp/racket_cmd.txt && ok "/racket/command publishing (frame_id present)" || no "no /racket/command msg"
  echo "  --- sample ---"; sed -n '1,18p' /tmp/racket_cmd.txt | sed 's/^/    /'
else
  # fall back to a publish-rate check (also proves data is flowing) before declaring failure
  if timeout 5 ros2 topic hz /racket/command 2>/dev/null | grep -q "average rate"; then
    ok "/racket/command publishing (rate confirmed; echo --once just missed the window)"
  else
    no "no message on /racket/command within 6s (is planner_imitate running with dry_run:=false?)"
  fi
fi

# ---- 5+7. launch wbc_runner (dry-run) + verify CSV --------------------------
if [ "$HAVE_ORT" = "1" ]; then
  hdr "5. launch wbc_runner (mode=dry_run, NO hardware output)"
  ros2 launch hope_wbc_runner wbc_runner.launch.py mode:=dry_run \
      onnx_path:="$ONNX_PATH" csv_path:="$CSV" > /tmp/wbc_runner.log 2>&1 &
  PIDS+=($!); sleep "$RUN_SECONDS"
  grep -q "wbc_runner started" /tmp/wbc_runner.log && ok "wbc_runner launched (dry_run)" || no "wbc_runner did not start (see /tmp/wbc_runner.log)"
  grep -q "WILL NOT PUBLISH" /tmp/wbc_runner.log && ok "runner confirms it will NOT publish" || echo "  [warn] could not confirm no-publish banner"
  if grep -qiE "PUBLISH] |joint command publisher" /tmp/wbc_runner.log; then no "runner created/used a joint command publisher in dry_run!"; else ok "no joint command publisher in dry_run"; fi

  hdr "7. verify /tmp/wbc_runner.csv (180-D obs / action / target_q logging)"
  if [ -f "$CSV" ]; then
    ok "CSV created"
    head -1 "$CSV" | grep -q "obs_norm" && head -1 "$CSV" | grep -q "target_q_30" \
      && ok "CSV header has obs_norm + target_q_0..30 (31 joint targets)" || no "CSV header missing expected columns"
    # an ACTIVE forehand row (valid=1, swing forehand, non-zero action/obs)
    if awk -F, 'NR>1 && $4=="forehand" && $9=="1" && $10+0>0 && $11+0>0 {found=1} END{exit !found}' "$CSV"; then
      ok "CSV has active forehand rows with non-zero obs_norm + action_norm"
    else
      no "no active forehand rows (planner may not be publishing; check it ran dry_run:=false)"
    fi
    echo "  --- CSV head ---"; head -3 "$CSV" | cut -c1-160 | sed 's/^/    /'
  else
    no "CSV not created at $CSV"
  fi
else
  echo; echo "(steps 5-7 skipped: onnxruntime/model unavailable)"
fi

cleanup
hdr "SMOKE SUMMARY: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "DRY-RUN SMOKE OK (no hardware command was ever published)." || echo "Some checks FAILED — see logs in /tmp/*.log."
exit "$FAIL"
