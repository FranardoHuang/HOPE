# A3 Deployment Actuator / Controller Audit

**Date:** 2026-06-27  **Scope:** audit-only (no retrain, no reward/policy/deploy changes).
**Question:** when our ONNX policy outputs a 31-D action, how does the real Agibot A3 deploy stack
turn it into joint commands — and is that actuator model compatible with the Isaac-trained policy?

**Checkpoint under audit:** `logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume/model_32200.pt`
(unified forehand+backhand, obs 180-D, action 31-D, 50 Hz).
**Inspected:** `agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/` (and the tracked copy under
`agi/code_deployment/...`). Reproduce the gain table with
`python scripts/inspect_a3_deploy_contract.py`.

---

## Headline verdict

| Axis | Result |
|---|---|
| **Actuator model** | ✅ **COMPATIBLE** — deploy is **position-target + PD-in-backend (implicit servo)**, identical in kind to Isaac's `ImplicitActuator`. NOT explicit software PD. |
| **PD gains / defaults / action_scale** | ✅ Identical for 28/29 shared joints (same `a3.py` source). ⚠️ ONE real diff: `waist_pitch` action_scale 0.590 (ours) vs 0.575 (deploy). |
| **Control rate / decode formula** | ✅ Match — 50 Hz, `q_des = action·action_scale + default`, command held 20 ms. |
| **ONNX I/O + obs + action DOF** | ❌ **INCOMPATIBLE** — deploy is hardcoded for the official HITTER policy: input `obs_dict[1,1570]`, output `action[1,29]`. Ours is `obs[180]+time_step` → `action[31]+6 refs`. |

**Bottom line:** the **actuator/backend is reusable and matches training**; what is HITTER-specific is the
**policy front-end** (the 1570-D tokenizer obs builder + 29-DOF action contract). Our 180-D/31-D policy
is **not a drop-in**, but it does **not** need a 29-D retrain — it needs a small new front-end runner
that reuses the existing 31-DOF backend command interface. **Classification: B** (compatible after a
new runner/wrapper; the backend itself is compatible). It is **not C** — the incompatibility is in the
obs/IO front-end, not the actuator.

---

## 1. Deployment I/O contract

`control_policy.hpp` validates the ONNX session at load and **rejects** anything off-contract:
- **Input:** exactly one tensor named `obs_dict`, shape `[1, 1570]` (640-float HITTER tokenizer +
  930-float proprioception with **10-step history**).
- **Output:** exactly one tensor named `action`, shape `[1, 29]`.
- Config: `a3_runtime_config.yaml` → `onnx.model_path = .../model_step_098000_a3.onnx`, `ort_cpu`.

**Our exported policy** (`exported/policy.onnx`): inputs `obs[1,180]` + `time_step[1,1]`; outputs
`actions[1,31]` + 6 reference tensors (BeyondMimic format). → It would **fail** the deploy's input-name,
input-shape, output-count, output-name, and `action_dim==29` checks. The deploy's `a3_obs_builder`
constructs the 1570-D tokenizer obs; it does **not** build our flat 180-D obs. So the deploy is
**hardcoded for the 1570-D / 29-D HITTER policy** at the IO/obs layer.

## 2. Action decoding path

`a3_action_decoder.cpp`:
```cpp
constexpr double kA3RawActionClip = 20.0;
q_des_mujoco[i] = clamp(raw_action_isaaclab[src], -20, +20) * a3_action_scale[i] + a3_default_angles[i];
```
- Action is a **normalized action**, decoded to a **position target** `q_des = action·scale + default`
  — **identical formula to our training** (`use_default_offset=True`).
- **Clip:** raw action clamped to ±20 (our training has `clip_actions=null`, but ±20 is far outside the
  policy's actual output range — `|action|` ~ N(0, std≈0.54) — so the clip is **non-binding**).
- **No filtering / smoothing / rate-limiting / interpolation** between policy steps. The 50 Hz `q_des`
  is held for the 20 ms tick.
- `dq_des = 0`, `tau_ff = 0` — no commanded velocity or feedforward torque.

## 3. Actuator / control mode — **position target + PD in backend (implicit servo)**

The command struct (`RobotCommand` → `joint_msgs::JointCommand`) carries, per joint:
`{ position(q_des), velocity(dq_des=0), effort(tau_ff=0), stiffness(kp), damping(kd) }`.

`expand_to_backend.cpp` fills `out.q_des`, `out.kp = a3_kps`, `out.kd = a3_kds` **every tick**; `tau_ff`
and `dq_des` stay zero. There is **no `τ = kp·(q_des−q) − kd·q̇` anywhere in user space** (grep-confirmed).
The motor driver / firmware closes the PD loop from `(q_des, kp, kd)` at its own (high) rate.

➡️ **This is exactly Isaac's `ImplicitActuator` semantics** (and our MuJoCo `--pd-mode implicit`). So the
sim-to-sim velocity finding maps cleanly: the **implicit-PD** MuJoCo result (racket_vel_err ≈ 0.31 m/s,
vel_pass ≈ 0.88) — **not** the explicit-PD result (0.61 / 0.35) — is the deployment-relevant one. The
earlier explicit-PD velocity gap is a sim artifact, **not** a deployment risk.

(The `mujoco_sim_standalone` uses `<motor>` torque actuators and applies the PD from `(q_des,kp,kd)` in
its subscriber — i.e. it deliberately emulates the firmware servo, same model.)

## 4. PD gain / default / action_scale comparison (train vs deploy)

Full per-joint table: `python scripts/inspect_a3_deploy_contract.py`. Summary:

- **Kp, Kd: identical** for all 29 shared joints (both transcribe the same `a3.py`):
  legs hip_p/r/y=80/120/80, knee=250, ankle=50; waist=85/50/50; arms shoulder=40, yaw/elbow/wrist_r=30,
  wrist_p/y=20; Kd legs 3/4/3/8/2, waist 3/2/2, arms 3/3/2…. Head (deploy ExpandToBackend): kp=40, kd=2.
- **default_joint_pos: identical** (sub-1e-3 diffs are only the ONNX metadata's 3-decimal rounding).
- **action_scale: identical except `waist_pitch`** — **ours 0.590 vs deploy 0.575** (≈2.6%). Cause: our
  config uses waist_pitch effort_limit 118 N·m (→0.25·118/50=0.59); the deploy `a3.py` used 115 (→0.575).
  Impact: if run through the *official* decoder, waist_pitch targets would be ~0.975× of what the policy
  expects. Minor, single-joint. **A custom runner should use OUR `action_scale` (from the ONNX metadata),
  which already matches training.**

| | training (ours, ONNX meta) | deploy (`a3_policy_parameters.hpp`) |
|---|---|---|
| joints | 31 (incl. 2 head) | 29 (head excluded; filled by ExpandToBackend) |
| order | Isaac articulation (interleaved L/R) | 29-DOF MuJoCo policy view (waist, L-arm, R-arm, L-leg, R-leg) |
| Kp / Kd | per ONNX metadata | identical values |
| default | identical | identical |
| action_scale | identical **except waist_pitch (0.590)** | waist_pitch 0.575 |

## 5. Joint mapping comparison

- **Names match** for all 29 shared joints; the 2 head joints (`head_yaw`, `head_pitch`) exist only on the
  training side. The deploy backend command struct is **31-DOF** (`kA3Dof=31`); `ExpandToBackend` scatters
  the 29 policy joints via `kA3PolicyToSdkIdx` and sets the 2 neck slots to `q_des=0, kp=40, kd=2`.
- **Order differs**: our policy emits Isaac articulation order; the deploy expects its 29-DOF MuJoCo view
  and applies `a3_isaaclab_to_mujoco` — but that permutation assumes the *official* `a3.py` IsaacLab order,
  which is **not** our BeyondMimic order. So a custom runner must build its own name→backend-slot map from
  the ONNX `joint_names` (the runner already has the analogous map in `scripts/mujoco_eval_onnx.py`).
- **Head:** our 31-D policy actually outputs head targets; deploy pins head at 0 (kp40/kd2). A custom
  runner can either forward our head outputs (the backend is 31-DOF) or pin head to default — head is not
  task-relevant and is near-default in our policy either way.

## 6. Control frequency & latency

- **Policy inference:** 50 Hz (`policy_driver.policy_hz = 50.0`). Matches our training/eval (sim_dt 0.005 ×
  decimation 4 = 50 Hz).
- **Command publish:** 50 Hz; each `q_des` **held** for the 20 ms tick (no interpolation/smoothing).
- **Backend state sync:** ~100 Hz (default 2× policy). **Motor servo:** firmware PD at a higher rate (not
  exposed by the deploy stack; typical 0.5–1 kHz) — this is where `(q_des,kp,kd)` is closed.
- **Latency path:** robot state (100 Hz) → cached → RT tick (50 Hz) → ONNX → decode → ExpandToBackend →
  publish → firmware PD. No action smoothing/interpolation in the deploy path.

## 7. Dither feasibility (`action = mean + 0.05·learned_std·randn`)

- **Where to inject:** in the new runner, immediately after the ONNX `actions` output and **before**
  `q_des = action·scale + default` (exactly as `scripts/mujoco_eval_onnx.py` does). `learned_std.npy`
  (31-D) is already staged in `exported/`.
- **Hardware safety:** dither injects fresh random offsets into the joint **position targets** every 20 ms.
  In sim this lowered Isaac's deterministic `ee_body_pos` fragility — but that fragility **did not reproduce
  in MuJoCo** (0 falls deterministic). On real hardware, random target jitter is an added risk with no
  demonstrated benefit in the implicit-PD regime.
- **Recommendation:** **deterministic (noise_scale = 0) for the first hardware tests.** Enable dither only
  in sim or tethered/low-power bring-up if a robustness problem actually appears on hardware. Keep
  deterministic as the default deployment path.

## 8. Compatibility verdict

**B — Compatible after a small front-end runner/wrapper change; the actuator backend itself is compatible.**
- ✅ Actuator model (position + firmware PD / implicit servo) matches Isaac `ImplicitActuator`.
- ✅ PD gains, defaults, decode formula, 50 Hz rate match (one 2.6% waist_pitch action_scale exception).
- ❌ The deploy **policy front-end** (1570-D tokenizer obs builder + 29-DOF / single-output ONNX contract)
  is hardcoded for the official HITTER policy and cannot consume our 180-D / 31-D BeyondMimic ONNX.
- ➡️ Not A (not drop-in). Not C (no 29-D retrain needed — the backend is 31-DOF and generic). Not D
  (all key files present and inspected).

## 9. Recommended next step

**Write a new lightweight A3 ONNX runner for our 180-D / 31-D policy that reuses the existing backend
command interface** (`RobotCommand`/`JointCommand` = position + kp + kd → firmware PD). Concretely it:
1. Builds our verified 180-D obs (the logic already exists in `scripts/mujoco_eval_onnx.py`).
2. Runs our `policy.onnx` (obs + time_step → 31-D action).
3. Decodes `q_des = action·action_scale + default` using **our** ONNX-metadata `action_scale`/`default`
   (so waist_pitch stays 0.590), maps Isaac order → backend slots by joint name, forwards all 31 joints
   (or pins head), attaches `a3_kps`/`a3_kds` (+ head 40/2), and publishes at 50 Hz.
4. Deterministic by default; dither gated behind a flag for sim/bring-up only.

This preserves the proven, actuator-compatible backend and only replaces the HITTER-specific front-end —
**no retrain, no reward change, no policy-weight change, no 29-D conversion.**

**Explicit answer:** the deployment path **is actuator-compatible** with the Isaac-trained policy (position
target + implicit/firmware PD, matching gains and rate). What must change before hardware testing is the
**policy front-end**: a new runner to feed the 180-D obs and 31-D action into the existing backend (and to
use our `action_scale`, fixing the lone waist_pitch 0.590-vs-0.575 difference). The official 1570-D/29-D
HITTER harness cannot run this policy as-is.
