# A3 ping-pong policy in MuJoCo: actuator-fidelity note for the AGI sim team

**TL;DR.** The HOPE ping-pong ONNX (`model_15200`, 180-obs / 31-act) runs **stably and
swings cleanly** in standard MuJoCo when the actuator is integrated **implicitly**
(matching the Isaac `ImplicitActuator` it was trained with). In the AGI `a3_pingpong`
deploy/sim path it **diverges within ~0.1 s**. The robot model (`a3_pingpong.xml`) is
**byte-identical** between the two; the only difference is **how the PD is applied and
integrated**. This is an actuator-model fidelity gap in the sim, not a policy or a
deploy-integration bug.

## Evidence (all reproducible)

1. **Deploy obs/command path is correct.** Running our front-end on the real
   `A3AimrtBackend` + `A3PolicyDriver`, the **first MOTION tick is clean**:
   `|action| ≈ 3`, every observation block sane (ref command 3.7, motion_anchor 0,
   projected_gravity 1.0, joint_vel ≈ 0, racket_target 0.47). Joint map / scatter /
   decode all match `MakeA3Layout31()`.
2. **Then it diverges** in the AGI sim: joint velocities reach ~16 rad/s after a single
   20 ms control step; `|action|` runs away to ~25; the robot flails and falls. This
   happens in **all** localization modes (fabricated / perfect-tracking / oracle), so it
   is **not** an observation or localization issue.
3. **Not a gain problem.** Policy gains are modest (kp 20–250, kd 2–8,
   action_scale ≤ 0.69); the AGI sim already recomputes PD every physics step (1 kHz),
   which is unconditionally stable for these gains. Lowering the gain (×0.3) does not fix
   it (the robot just sags and falls from being under-powered).
4. **Same ONNX is stable in standard MuJoCo with implicit PD** (our `mujoco_eval_onnx.py`,
   zero falls, clean forehand/backhand). Toggling **only** the PD mode in that one script
   flips stability — see the controlled demo below.

## Root cause: PD application / integrator

Identical model file; the difference is entirely in the control path:

| | Isaac training / our `--pd-mode implicit` (stable) | AGI `a3_pingpong` sim (diverges) |
|---|---|---|
| Integrator | `implicitfast` (`mjINT_IMPLICITFAST`) | MJCF default = **Euler (explicit)** |
| kd damping | applied as **passive joint damping**, integrated **implicitly** | applied as **explicit motor torque** `−kd·dq` written to `ctrl` |
| MJCF passive damping on the 31 actuated DOFs | zeroed (kd provides damping; avoids double-damping, matches Isaac) | kept (in addition to the explicit kd term) |
| Control torque | `tau = kp·(q_des − q)` only | `tau = effort + kp·(q_des − q) + kd·(vel − dq)` |

The training-time actuator is Isaac's `ImplicitActuator`: kd damping is folded into an
**implicit** velocity update. Reproducing that in MuJoCo requires the `implicitfast`
integrator with kd as passive damping. The AGI sim instead uses the **explicit Euler**
integrator and feeds kd back as an explicit motor torque — an explicit velocity feedback
loop that is not faithful to the trained actuator and goes unstable for this dynamic,
whole-body swing.

(Reference: `BodyDriveJointActuatorSubscriber::ApplyCtrlData` in
`mujoco_sim_module/subscriber/joint_actuator_subscriber.cc` applies the explicit
`effort + stiffness·(pos−q) + damping·(vel−dq)`; the MJCF `<option …>` has no
`integrator`, so MuJoCo defaults to Euler.)

## Reproduce the flip in one script (recommended live demo)

Same MuJoCo, same model, same ONNX — only `--pd-mode` changes:

```bash
# conda env with mujoco + onnxruntime
cd hope_training/whole_body_tracking
RUN=logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch
MJCF=agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml

# STABLE — implicit actuator (matches Isaac training):
python scripts/mujoco_eval_onnx.py --viewer --pd-mode implicit --noise-scales 0.0 \
  --onnx $RUN/exported/policy.onnx --std $RUN/exported/learned_std.npy --mjcf $MJCF

# AGI-like — explicit kd torque + keep MJCF passive damping (Euler):
python scripts/mujoco_eval_onnx.py --viewer --pd-mode explicit --keep-passive --noise-scales 0.0 \
  --onnx $RUN/exported/policy.onnx --std $RUN/exported/learned_std.npy --mjcf $MJCF
```

## Suggested fix on the AGI sim side (any one is sufficient)

1. **Set `integrator="implicitfast"`** in `a3_pingpong.xml` `<option …>`, and apply kd as
   **passive joint damping** (let MuJoCo integrate it) rather than as an explicit motor
   torque — i.e. command torque = `kp·(q_des − q)` only, with kd loaded into `dof_damping`.
   This matches Isaac's `ImplicitActuator` exactly.
2. Or switch the 31 actuators from `<motor>` to MuJoCo **`<position>`** servos (built-in,
   integrated implicitly) and pass `q_des` + per-joint `kp`/`kd`.
3. Keep the per-step recompute (already done) — the missing piece is **implicit
   integration of the velocity/damping term**, not the update rate.

The real robot's body-drive backend already does PD-in-backend (implicit-style), so this
only affects the **simulator's** fidelity; on hardware the policy is expected to behave
like the stable implicit MuJoCo result.
