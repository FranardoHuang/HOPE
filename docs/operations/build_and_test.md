# Build And Test

Status: Draft

## Task Setup

For package-local Python tests, no ROS environment or ignored local asset is required. The planner pytest needs a Python with `rclpy`, `numpy`, and the `hope_planner` deps available (e.g. a sourced ROS 2 Jazzy environment).

For ROS workspace build, use the ROS environment:

```bash
distrobox enter hope
cd ~/workspace/HOPE/hope_ws
```

No `vendor_assets/` payload is required for planner tests or ROS package discovery.

## Planner Unit Tests

Run from the package directory:

```bash
cd hope_ws/src/hope_planner
python3 -m pytest test
```

Current known result:

- 2026-06-22: 20 passed.
- 2026-06-26: 26 passed with `PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest hope_ws/src/hope_planner/test -q`.

From the repo root, set `PYTHONPATH` explicitly:

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest \
  hope_ws/src/hope_planner/test/test_racket_target_planner.py \
  hope_ws/src/hope_planner/test/test_ball_trajectory_predictor.py \
  hope_ws/src/hope_planner/test/test_quaternion_utils.py -q
```

Current known result:

- 2026-06-26: selected planner math tests above, 16 passed.

## Phase-1 BankExam Adapter And Audit Tests

The following tests need neither a running Isaac instance nor MuJoCo. They
cover the read-only terminal-checkpoint inventory, balanced shared schema-v3
paper, evaluator-owned Isaac adapter, saved-run termination contract and the
authoritative virtual-return scorer:

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_audit_runpod_terminal_runs.py \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_adapter.py \
  hope_training/whole_body_tracking/tests/test_termination_contract.py \
  hope_training/whole_body_tracking/tests/test_virtual_return_scorer.py
```

The terminal audit only reads a run tree and prints shell-quoted `judge.sh`
commands. It never executes them. The remaining modules now exercise the
production seams used by `isaac_bank_exam.py` and `mujoco_eval_onnx.py`:
immutable content IDs/order, exact per-clip quota, strict artifact reload,
nominal profile, tuple/mapping observation adaptation, all-attempt ledger,
saved-run termination preflight and Torch/NumPy scorer parity when Torch is
installed. A green dependency-light suite is implementation evidence; it is
not a substitute for a same-paper Isaac/MuJoCo runtime canary.

Verified 2026-07-11 on the local macOS host:

- adapter/audit suite: 66 passed, 1 optional Torch parity skip;
- formal BankExam/motion/racket/schema/V5 CPU suite: 81 passed;
- union of both groups plus the MuJoCo contract tests: 136 passed, 1 optional
  Torch parity skip;
- planner suite: 105 passed, 2 optional skips.

Reproduce the 81-test formal CPU suite with:

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py \
  hope_training/whole_body_tracking/tests/test_racket_geometry_contract.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  hope_training/whole_body_tracking/tests/test_v5_ablation_accelerator.py
```

## ROS Workspace Build

Run inside the intended ROS environment:

```bash
cd hope_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The `rosdep install` step resolves `vrpn_mocap` VRPN/eigen dependencies and must run before `colcon build`.

Current status:

- Target environment is Linux + ROS 2 Jazzy. The build is to be verified inside the ROS environment described in [setup_environments.md](setup_environments.md). The obsolete root `Dockerfile.hope-ros2-jazzy` has been removed.

## A3 Deploy Source Build And Unit Tests

The active tracked tree is `agi/a3_deploy_example`. Restore its vendor-gated
`thirdparty/unitree_sdk2` dependency from
`vendor_assets/agibot/a3_deploy_example_full/` as described in
[setup_local_sync.md](setup_local_sync.md). Then source `setup_a3_env.sh`; it
selects ROS when available and restores ONNX Runtime 1.19.2 if missing. These
commands compile and test source only; they do not authorize a robot command
test.

Portable GCC/Clang Release build:

```bash
AD="$PWD/agi/a3_deploy_example"
B="$AD/build/release_tests"
cd "$AD"
export HAS_ROS2=0
source setup_a3_env.sh
cmake -S "$AD" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_SRCS=ON -DENABLE_TRT_INFERENCE=OFF \
  -DENABLE_A3_ROS_MSGS=OFF -DENABLE_A3_AIMRT_BACKEND=OFF \
  -DGS_PACKAGE_ARCH_NAME=x86_64 -DGS_RUNTIME_OUTPUT_DIR="$B/runtime"
cmake --build "$B" --target run_tests -j8
"$B/runtime/run_tests" --gtest_color=no
```

Safety finite checks rely on IEEE NaN/Inf semantics. Verify Release did not
re-introduce fast-math:

```bash
grep -o -- '-fno-fast-math\|-fno-finite-math-only' \
  "$B/compile_commands.json" | sort -u
! grep -Eq -- '(^|[[:space:]])-ffast-math([[:space:]]|$)|(^|[[:space:]])-ffinite-math-only([[:space:]]|$)' \
  "$B/compile_commands.json"
```

ROS 2 Jazzy Release build:

```bash
AD="$PWD/agi/a3_deploy_example"
B="$AD/build/ros_release_tests"
source /opt/ros/jazzy/setup.bash
cd "$AD"
export HAS_ROS2=1
source setup_a3_env.sh
cmake -S "$AD" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_SRCS=ON -DENABLE_TRT_INFERENCE=OFF \
  -DENABLE_A3_ROS_MSGS=ON -DENABLE_A3_AIMRT_BACKEND=OFF \
  -DGS_PACKAGE_ARCH_NAME=x86_64 -DGS_RUNTIME_OUTPUT_DIR="$B/runtime" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build "$B" --target run_tests a3_deploy_onnx_ref_pingpong \
  a3_policy_runtime_probe -j8
JM="$B/src/a3/a3_deploy_onnx_ref/joint_msgs_build"
LD_LIBRARY_PATH="$JM:${LD_LIBRARY_PATH:-}" \
  "$B/runtime/run_tests" --gtest_color=no
```

The `LD_LIBRARY_PATH` is needed only for direct-CMake tests so RMW can `dlopen`
the generated FastRTPS typesupport. The package builder stages those libraries
beside the deployed wrapper.

Verified 2026-07-10 on Ubuntu 24.04/Jazzy/GCC 13:

- portable Release: 188 passed, 4 skipped;
- ROS Release: 202 passed, 4 skipped; ping-pong runner and runtime probe linked;
- unknown runner CLI arguments exit 2 before backend/model initialization;
- skips were optional external CSV/FK/end-to-end fixtures, not safety tests.
