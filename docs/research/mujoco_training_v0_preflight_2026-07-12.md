# Native MuJoCo training v0 preflight (2026-07-12)

Status: **design audited; implementation not started; G04/G05/G06 remain `Partial`**.

This note turns the P0 priority decision into a minimum implementation boundary. It is a read-only
source audit, not a simulator result. No trainer, configuration, Pod process, vendor runtime or
hardware was started or changed.

## Decision

A useful 2--3 day v0 is feasible as a **vendor-Gate3 robot-dynamics and strike-state fine-tune**:
start from a 179-D Isaac actor, train balance-preserving target tracking in native CPU MuJoCo, and
then submit checkpoints to the independent vendor Gate3/Gate3B runtime. It is not yet a physical
return trainer.

The tracked vendor `a3_pingpong.xml` contains the floor, A3 and racket site, but no ball, table or
net. `BallPhysics` exists as an analytic, mocap-body driver, but the tracked MJCF has no required
`ball` body and `MujocoSimModule::SimLoop()` does not construct or step it. Therefore v0 may optimize
kinematic strike state, balance and action discipline only. Formal hit/return reward and scoring
remain blocked on a separate scene/runtime with instrumented ball-racket-table-net contact and
landing state.

`docs/NOW.md` tracks `mujoco-ball-wiring@4607410` as a handoff candidate that adds a mocap ball plus
table/net geometry and simulator-loop wiring. At the audited base it is **not an ancestor of main**,
and its vendor build, transport/QoS, GUI and runtime behavior remain unverified. It does not change
the current main-plant finding above. Physical-return v0 may depend on that work only after its own
reviewed merge, vendor build and independent contact/landing acceptance; it is not a shortcut around
those gates.

Do **not** extract a shared implementation from the current evaluator and let trainer and judge
import it. Reuse the frozen meanings, layouts and fixtures, while implementing them independently.
Sharing observation, action, reward or termination code would let the same bug make training and
evaluation green.

## Audited source boundary

| Concern | Existing source that is useful as a specification | Limitation found |
| --- | --- | --- |
| ONNX and 179/31 metadata | `scripts/mujoco_eval_onnx.py::OnnxPolicy` | Inference/evaluation wrapper, not a training policy module. |
| MuJoCo model and PD stepping | `MujocoRobot.__init__`, `MujocoRobot.apply_pd_and_step` | Mutates effective plant to the requested evaluation profile; it is not automatically vendor Gate3. |
| Command and clock | `RacketCommand`, `run_rollout` | Single-environment mutable RNG/state; must be independently batched. |
| Actor observation | `build_obs` | The 179-D layout is reusable as a frozen contract, but not as shared trainer/judge code. |
| Termination diagnostics | `check_terminations` | Reference-relative guards are not a complete PPO terminal/timeout contract. |
| Evaluation | `run_rollout`, `score_virtual_return` | Produces metrics and analytic virtual-return diagnostics. It has no per-term training reward API. |
| Vendor plant loop | `MujocoSimModule::SimLoop`, `BodyDriveJointActuatorSubscriber::ApplyCtrlData` | Native 1 ms loop applies explicit PD before every `mj_step`; no ball driver is wired. |
| Deploy action adapter | `PpPolicy::ComputeCommand` in `pp_policy.hpp` | Adds neck-passive, optional leg/waist hold and smoothing, runtime gain scaling and hard joint-limit clamp. These are part of the effective plant/action contract. |
| Isaac warm start | `utils/ckpt_compat.py::load_actor_tolerant` | Not actor-only: strict mode restores the runner; fallback retains every shape-matching tensor, including matching critic tensors. |
| Isaac hard contract | `scripts/train.py::_build_training_hard_contract` | Explicitly omits reward weights, termination thresholds and optimizer settings; v0 needs a backend-aware experiment contract in addition. |

The evaluator is Python using native MuJoCo bindings. No independent C/C++ evaluator implementing
the PPO per-term reward was found in the inspected path. The correct pre-PPO requirement is
therefore:

1. reset and action-tape parity against a byte-frozen Python evaluator for the **same explicit
   profile**;
2. per-term reward parity against a new, independent reward-replay oracle;
3. final behavior in the separate vendor C++ Gate3/Gate3B runtime.

## Two profiles, not one claimed parity plant

The two paths may load the same source MJCF bytes while running different effective plants. The
contract must hash the resolved values below, not only `a3_pingpong.xml`.

| Field | `isaac_bank_parity_v1` | `vendor_gate3_v1` |
| --- | --- | --- |
| Purpose | Reproduce the current schema-3 Isaac/BankExam execution profile. | D0 training target and final deployment-simulator prerequisite. |
| Timestep/control | Current Phase-1 profile is 5 ms physics, four substeps per 20 ms actor tick. | Source MJCF is 1 ms; explicit PD is recomputed before every 1 ms `mj_step`. |
| Passive terms | Formal zero-friction profile zeros native damping/frictionloss, then installs schema-bound actuator facts. | Preserve resolved vendor MJCF damping, frictionloss, armature and motor ctrl ranges. |
| Integrator/PD | Isaac-style implicit joints put Kd in damping and use `implicitfast`; explicit joints follow schema. | MJCF default integrator plus vendor subscriber's explicit `Kp(qdes-q)+Kd(dqdes-dq)+tau`. |
| q-des/runtime adapter | Evaluator's optional training soft-limit clamp. | Production hard joint limits, neck passive and the exact selected leg/waist hold, gain, clamp and smoothing flags. |
| Allowed claim | Isaac-profile parity canary and held-out BankExam development evidence. | Gate3-targeted training profile; still requires the real vendor runtime to pass. |

`vendor_gate3_v1` is blocked until these runtime choices are frozen: global/leg/ankle gain scales,
official-stand mode, hold/recover behavior, leg clamp and smoothing, waist/legs passive state, neck
override, command rate and all timeout/reset semantics. Changing one creates a new profile.

## Planned v0 file boundary

These files do not exist yet. The split is deliberately dependency-light and keeps the evaluator
outside the trainer import graph.

```text
hope_training/whole_body_tracking/
  source/whole_body_tracking/whole_body_tracking/mujoco_rl/
    contracts.py       # FACE179/31 and complete engine/profile contract; no Isaac import
    plant.py           # VendorA3Plant: name maps, reset and explicit-PD substeps
    commands.py        # batched train-bank command; one deterministic RNG stream per env
    observations.py    # independent build_face179 implementation
    rewards.py         # pure state -> named reward terms; never imports virtual_return_scorer
    env.py             # MujocoRslRlVecEnv, auto-reset and extras["time_outs"]
    warm_start.py      # strict actor-only transfer
  scripts/
    train_mujoco.py
    mujoco_fixed_tape_oracle.py
    mujoco_reward_replay_oracle.py
  tests/
    test_mujoco_vecenv_contract.py
    test_mujoco_vecenv_single_env_parity.py
    test_mujoco_actor_only_warm_start.py
```

Start with one immutable `MjModel` and one `MjData` per environment, stepping environments in a
plain Python loop. Measure `N=32/64` throughput before adding a C++/OpenMP batch loop. MJX/MJWarp is
a later backend with its own parity burden.

The v0 step order is fixed:

1. resolve pre-step reference/command and raw 179-D observation;
2. compute raw 31-D action and store that raw value as `last_action`;
3. decode q-des, apply the selected profile's runtime post-processing, then run all PD substeps;
4. advance the strike clock;
5. calculate post-step named rewards, terminal state and timeout separately;
6. preserve terminal observation/reward before resetting only the done environments.

`extras["time_outs"]` must distinguish a time-limit bootstrap from a physical fall. The initial v0
paper uses deterministic named-stand reset and `init_at_random_ep_len=False`; random episode-age
initialization would invalidate the reset-first canary.

## Single-environment parity canary

Long PPO runs are blocked until all three legs are green with predeclared per-field tolerances and
negative controls. Do not accept an aggregate error that hides one wrong observation block or event.

### 1. Reset-first observation

- reset the named `stand` keyframe;
- clear time, qvel, act, ctrl, qacc, command hidden state and `last_action`;
- install one train-bank target and the exact hold/reference state;
- compare every raw 179 column, normalized 179 column and Torch-versus-ONNX actor mean;
- bind motion bytes, train-bank bytes, normalizer and observation term order.

### 2. Fixed action tape

Use a policy-independent 100-tick/2-second tape: zeros, small deterministic sinusoids/noise and
out-of-range pulses that exercise q-des and torque clamps. Record at every actor tick:

- raw action and raw `last_action`;
- q-des before and after runtime post-processing;
- final and peak substep torque;
- qpos/qvel, pelvis/torso/racket pose and velocity, contacts and termination booleans.

Run the trainer implementation and the frozen evaluator on each named profile separately. Compare
the production C++ first-tick adapter separately for hard clamp, neck override and optional runtime
post-processing. Exact event booleans and fieldwise numerical thresholds must pass; widening a
tolerance after seeing a failure requires a new preregistration and a negative control.

### 3. Independent reward tape

Save a fixed state/action/next-state ledger, including deliberately good, missed, falling and
timeout rows. Compare every trainer reward term and the total against a separate NumPy/reference
oracle that does not import `mujoco_rl`. The current evaluator cannot serve this role because it
does not compute training rewards.

For v0, freeze only strike-state and stability terms such as racket position/velocity/normal,
strike success/progress, imitation, ready/balance, action rate, torque/contact/slip. Do not train on
analytic virtual-return and then select with the same scorer. Reference-relative guards are
diagnostics or penalties; physical fall and timeout own termination.

## Actor-only warm start

Construct the new actor-critic and optimizer first, then load exactly:

- all and only required `actor.*` tensors;
- the action-distribution parameter (`std`, `log_std` or its bound equivalent);
- actor observation-normalizer state, frozen for v0 so the initial policy is identical.

Do not load `critic.*`, privileged-observation normalizer, optimizer, scheduler or iteration. The
loader must fail on missing/extra actor keys, shape/dtype mismatch or non-finite values. A test must
prove the seeded critic is byte-unchanged, optimizer state is empty and iteration is zero after
load. The source checkpoint and its original hard-contract SHA remain recorded, but the MuJoCo
descendant receives a new backend lineage and cannot inherit Isaac exactness.

## Independent evaluation and common-mode controls

- Run the evaluator from a clean detached checkout and separate environment/process; scrub trainer
  paths from `PYTHONPATH` and pin evaluator source, Python/MuJoCo, MJCF plus mesh closure, model,
  motion, bank/schedule and effective-profile SHA.
- The trainer can open only the schema-3 train split. It must not receive the held-out exam path.
- Copy a closed checkpoint read-only into a no-clobber evaluator directory. Training metrics never
  select the winner.
- K20/q10 remains directional screening; only the preregistered K100 checkpoint milestones decide
  the paired frozen-control versus fine-tune paper. Gate3/Gate3B remains the promotion arbiter.
- Treat shared implementation, train/exam leakage, teacher/reference resets, the same analytic
  return scorer, normalizer drift, unbound runtime action post-processing and in-memory MJCF
  mutation as explicit common-mode false-green risks.

## 2--3 day D0 sequence

1. **Day 0:** freeze `vendor_gate3_v1`, implement plant/179 observation/action adapter and pass the
   reset/action/reward canaries. Do not start PPO while any canary is red.
2. **Day 1:** implement sequential CPU `VecEnv`; require deterministic reset, finite rollout, one
   finite PPO update, save/resume/export smoke, actor-only assertions and measured throughput.
3. **Day 2:** run equal-budget frozen-source versus warm fine-tune with at least two seeds and
   frequent checkpoints. Use independent K20 screens and preregistered K100 decisions, then submit
   the retained checkpoint to exact vendor Gate3 D0. One seed is smoke evidence only.

Use a one-shot D0 episode: named stand, random train-bank target and bounded hold, one complete clip
plus fixed follow-through, then terminate/reset. It makes no carry-state or continuous-rally claim.
Physical ball, table/net scene integration and no-reset recovery follow after this baseline.

## Reproducible read-only audit

Run from repository root at audited base `b93b24250bd05c9daed33471946fbd8ff944c63e`:

```bash
git rev-parse HEAD
if git merge-base --is-ancestor 4607410 HEAD; then
  echo "ball_wiring_ancestor_rc=0"
else
  echo "ball_wiring_ancestor_rc=1"  # expected at this audited base
fi

shasum -a 256 \
  agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/module/mujoco_sim_module/mujoco_sim_module.cc \
  agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/module/mujoco_sim_module/subscriber/joint_actuator_subscriber.cc \
  agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_policy.hpp

python3 - <<'PY'
import pathlib
import xml.etree.ElementTree as ET

p = pathlib.Path(
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
root = ET.parse(p).getroot()
names = [node.get("name", "") for node in root.iter()]
print("option", root.find("option").attrib)
print("ball/table/net names", [
    name for name in names if any(token in name.lower() for token in ("ball", "table", "net"))
])
print("right_racket sites", sum(
    node.tag == "site" and node.get("name") == "right_racket" for node in root.iter()
))
PY

rg -n 'BallPhysics|ball_physics' \
  agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/module/mujoco_sim_module/mujoco_sim_module.{h,cc} || true
rg -n 'SimLoop|ApplyCtrlData|mj_step|command\.stiffness|command\.damping' \
  agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/module/mujoco_sim_module/{mujoco_sim_module.cc,subscriber/joint_actuator_subscriber.cc}

rg -n '^class (OnnxPolicy|MujocoRobot|RacketCommand)|^def (build_obs|check_terminations|run_rollout|score_virtual_return)' \
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py
rg -n 'model\.opt\.timestep|dof_damping|dof_frictionloss|implicitfast|soft_joint_limits' \
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py
rg -n 'NECK PASSIVE|clamp_q_to_limits|last_action_ = action|LEG q_des|WAIST PASSIVE' \
  agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_policy.hpp

sed -n '75,123p' \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/ckpt_compat.py
sed -n '157,163p;1861,1892p' hope_training/whole_body_tracking/scripts/train.py
```

Expected audit facts are: MJCF timestep `0.001`, no ball/table/net names, one `right_racket` site,
MJCF SHA `2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97`, no
`BallPhysics` reference in `MujocoSimModule`, explicit PD immediately before `mj_step`, evaluator
plant mutation and metrics/virtual-return code but no training reward interface, and a tolerant
checkpoint loader that keeps every matching tensor rather than enforcing actor-only transfer. At
this base, `ball_wiring_ancestor_rc=1` proves the separately tracked handoff is not in main.
