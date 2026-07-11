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

- adapter/audit suite: 67 passed, 1 optional Torch parity skip;
- formal BankExam/motion/racket/schema/V5 CPU suite: 85 passed;
- union of both groups plus the MuJoCo contract tests: 141 passed, 1 optional
  Torch parity skip;
- planner suite: 105 passed, 2 optional skips.

Verified later on 2026-07-11 against the current Phase-1 candidate implementation:

- Isaac-dependent face-pairing, strict override, schema-3, runtime-order migration, and Stage-1
  wiring regression: **122 passed** on Pod 2 with the Isaac Python environment;
- dependency-supported formal BankExam/adapter/MuJoCo union: **145 passed** with Torch and the
  Hydra test dependencies available.
- after the diagnostic-export/judge contract review, the local dependency-light formal group is
  **90 passed** and the launch-commit Pod runs passed **130** Isaac/Hydra tests and **153** full
  union tests. The subsequent source-first/inexact-judge regression expands the union to **154
  passed** in a detached Pod evaluation worktree while the live training checkout remains fixed.

The scale-out cadence generator and venue A-B-A timing audit are dependency-light:

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_analyze_rally_intervals.py \
  hope_training/whole_body_tracking/tests/test_generate_phase1_scaleout_curve_manifests.py \
  hope_training/whole_body_tracking/tests/test_phase1_checkpoint_curve_worker.py
python3 hope_training/whole_body_tracking/scripts/generate_phase1_scaleout_curve_manifests.py --check
```

Verified locally on 2026-07-11: `7 passed`; deterministic manifest check passed.

Reproduce the 122-test Isaac-dependent group on a prepared RunPod checkout:

```bash
/workspace/hope_isaac_venv/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_face_sign_per_clip.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py \
  hope_training/whole_body_tracking/tests/test_stage1_wiring.py
```

This group proves that `shared_plus_y` and `legacy_signed_vs_A` select one consistent pair for
reward, privileged observations and metrics; unknown selector/boolean values fail loudly; the
selector and legacy-motion opt-in reach the hard contract/export/judge paths; and motion migration
reorders all four body-indexed arrays into the explicit target order.

Reproduce the current 90-test formal CPU group with:

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

Reproduce the complete 154-test union (use a Python environment with `pytest`, `numpy`, `PyYAML`,
`hydra-core`, and Torch installed):

```bash
/workspace/hope_mjeval_venv/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_audit_runpod_terminal_runs.py \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_adapter.py \
  hope_training/whole_body_tracking/tests/test_termination_contract.py \
  hope_training/whole_body_tracking/tests/test_virtual_return_scorer.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py \
  hope_training/whole_body_tracking/tests/test_racket_geometry_contract.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  hope_training/whole_body_tracking/tests/test_v5_ablation_accelerator.py \
  hope_training/whole_body_tracking/tests/test_judge_plant_contract.py
```

Validate the future semantics-correct plant preregistration and offline
contract compiler without importing Isaac or launching a simulator:

```bash
python3 -m pytest -q \
  tests/test_validate_phase1_plant_semantics_prereg.py \
  hope_training/whole_body_tracking/tests/test_plant_contract_v1.py
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

For the 179-D versioned planner wire, also run the dependency-light Python packer tests and the
focused C++ schema/observation tests. These are source gates only; they do not run Gate 3 or a robot:

```bash
REPO="$(git -C "$AD" rev-parse --show-toplevel)"
PYTHONPATH="$REPO/hope_ws/src/hope_planner" python3 -m pytest -q \
  "$REPO/hope_ws/src/hope_planner/test/test_flat_command_wire.py"
env -u PYTHONPATH python3 \
  "$REPO/hope_training/whole_body_tracking/scripts/standalone_onnx_export.py" \
  --contract-import-smoke
python3 -m pytest -q \
  "$REPO/hope_training/whole_body_tracking/tests/test_stage1_normal_envelope.py" \
  "$REPO/hope_training/whole_body_tracking/tests/test_atomic_output.py" \
  "$REPO/hope_training/whole_body_tracking/tests/test_training_contract_schema3.py" \
  "$REPO/hope_training/whole_body_tracking/tests/test_export_obs_norm_contract.py"
python3 -m pytest -q \
  "$AD/tests/test_face179_preflight_failclosed.py"
"$B/runtime/run_tests" \
  --gtest_filter='PpPlannerInput.*:PpFace179Wire.*' --gtest_color=no
```

2026-07-11 isolated evidence at source commit `8d56ea86`: the Release target
`a3_deploy_onnx_ref_pingpong` linked successfully, the focused filter passed 10/10 and the full
native suite passed 195 with 4 optional-asset skips. This configure deliberately used
`ENABLE_A3_ROS_MSGS=OFF` and `ENABLE_A3_AIMRT_BACKEND=OFF`; it is not the ROS/AimRT or MuJoCo
runtime gate. See G06 for the content-addressed path and dependency/binary SHA values.

### Runner model-only preflight

Use the production runner itself to validate an exported ONNX and its metadata before opening an
AimRT backend. The switch is diagnostic-only and fails unless publishing is disabled:

```bash
"$B/runtime/a3_deploy_onnx_ref_pingpong" \
  --runtime-cfg "$AD/src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml" \
  --model-path /absolute/path/to/policy.onnx \
  --planner --no-publish --model-preflight-only
```

An accepted run exits zero and prints parsed `publishable_model_contract=true`,
`training_contract_exact=1`, `backend_not_initialized=true`, `obs_dim`,
`training_contract_sha256`, and `source_checkpoint_sha256`. A 179-D actor additionally exercises
the exact face-command/bank/source-family metadata checks, recomputes the train-normal-envelope
payload SHA, validates its two sign-preserving spherical caps, and exercises the existing
planner-mode guard. Missing any envelope field fails load; this means the pre-envelope SZ model
used by the 2026-07-11 loader proof is expected to fail under the stricter source and must be
re-exported from its exact train bank before this gate is rerun. An accepted 179 preflight also
prints `normal_envelope_payload_sha256`, `normal_envelope_train_bank_sha256` and
`normal_envelope_source_family_sha256`, plus
`normal_envelope_mount_normal_sign_per_clip=1,-1`, so the loader ledger can be matched to the
export ledger. The envelope is raw mount A; schema-2 input is opponent-facing physical B, and the
runner converts only the normal with `[+1,-1]` after clip selection. The fixture
`configs/phase1_face179_real_bank_envelope_expectations_20260712.json` binds the real train-bank/
family SHA, `757/724` row counts and expected raw-A cap statistics for the re-export check; it is
not a behavior fixture.
The output must contain neither `backend cfg` nor `A3AimrtBackend initialised`. Omitting
`--no-publish`/`--dry-run` exits 2 before model or backend initialization. This is a loader and
metadata gate only. Constructing `PpPolicy` deliberately performs one zero-observation ONNX
prewarm inference; it does not start the policy driver, tick a backend, connect ROS/AimRT,
instantiate MuJoCo, or authorize robot commands.

No-publish disables transport; it does not relax the model contract. The explicit legacy escape
is `--allow-legacy-model-diagnostic`, which requires no-publish and is forbidden with
`--model-preflight-only`. Thus neither plain no-publish nor preflight may open a metadata-stripped,
inexact-schema3 or envelope-less 179 model.

The real-model loader regression must be run with an exported ONNX, not just a synthetic header
test. It guards the ONNX Runtime `TypeInfo`/`TensorTypeAndShapeInfo` owner lifetime:

```bash
A3_PP_ONNX_PATH=/absolute/path/to/policy.onnx \
  "$B/runtime/run_tests" \
  --gtest_filter='PpOnnxPolicy.LoadsOnlyPublishableRealModelWithStableInputTypeInfoLifetime' \
  --gtest_color=no
```

Without `A3_PP_ONNX_PATH` this one optional asset test skips. A real formal 179 export must pass it
before the production preflight is credited.

Then run the production-binary negative integration. It makes only temporary copies of the input
ONNX and requires metadata-stripped, missing-envelope, and `training_contract_exact=0` variants to
exit nonzero before any backend marker. It also requires legacy-diagnostic + preflight to fail CLI
validation with rc2:

```bash
python3 "$AD/scripts/verify_face179_preflight_failclosed.py" \
  --runner "$B/runtime/a3_deploy_onnx_ref_pingpong" \
  --runtime-cfg "$AD/config/a3_runtime_config.pingpong.yaml" \
  --model /absolute/path/to/envelope-bearing-policy.onnx
```

For a direct-CMake build, the runtime may also need the build-tree TBB directory (the package
builder stages this dependency automatically):

```bash
export LD_LIBRARY_PATH="$B/gnu_13.3_cxx20_64_release:${LD_LIBRARY_PATH:-}"
```

Historical evidence in the isolated ROS/AimRT-enabled Release archive at source `a82eba6c` on
2026-07-11:

- all three targets (`run_tests`, production ping-pong runner and runtime probe) linked;
- the formal SZ seed2 model-2000 ONNX SHA was `350b51cc...34cc2`;
- real-model loader regression: 1/1 pass;
- full native suite: 205 pass / 9 optional-asset skips / 0 failures (214 total);
- preflight without no-publish: rc2 before model/backend initialization;
- planner + no-publish preflight: rc0, `obs_dim=179`, contract
  `3a3b3d95...b9972`, checkpoint `d920...5e22`, and no backend-init/start line.

That historical run remains loader-lifetime and backend-order evidence, but does not close the
strict publishable-model gate: its no-publish bit also enabled the legacy loader escape and its
model predates the envelope.

The strict rerun is now complete at source `2fa35340` with formal SZ seed3 model-2000 ONNX
`0c428ddf...b7b155`: native 219 tests = 210 pass, 9 optional skips, 0 fail; positive preflight rc0;
the three metadata mutations rc3; legacy+preflight rc2; no backend marker; 824 compile commands
without unsafe math. Exact inputs, binaries and logs are in
`configs/gate3_face179_strict_preflight_evidence_20260712.json`. Proceed to backend first tick and
vendor Gate 3/Gate 3B; do not reinterpret this model-only result as simulator behavior.

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
