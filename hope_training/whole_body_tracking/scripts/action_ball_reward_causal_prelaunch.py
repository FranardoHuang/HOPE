"""One-environment Isaac causal audit for the composed ActionBall Reward.

This is deliberately a *pre-launch* diagnostic, not a trainer feature.  It
constructs the same post-Hydra environment configuration as ``train.py``,
creates one real Isaac environment, and calls the active RewardManager
callables on transactional controlled states.  A probe never accepts raw
reward values from a JSON file: both values come from the live callable.

The controlled baseline and worsening may set several nuisance/gating tensors
to the same values.  The only explicitly controlled Reward-input value that
differs between the two calls is the taxonomy's named causal axis.  Every
explicitly controlled input tensor is restored before the next probe.
Callable-owned diagnostic counters may advance; they are not Reward inputs and
the one-env audit is destroyed after the receipt is written.  Unsupported
terms are retained in the receipt and make the all-objective result fail
closed.

Example (all normal ActionBall identity arguments are still required)::

    python3 scripts/action_ball_reward_causal_prelaunch.py \
      task=HOPEPingPongActionBall num_envs=1 device=cuda:0 \
      ...exact ActionBall Hydra overrides... \
      ++reward_causal_audit.output_dir=/workspace/runs/<new>/reward_causal

The script is intentionally import-safe without Isaac Lab so its receipt and
transaction contracts can be host-tested.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable


SCHEMA_VERSION = 1
SUPPORTED_GROUPS = (
    "mjlab_balance_stability",
    "beyondmimic_imitation",
    "hope_hit_landing_task",
    "immutable_safety",
)
TRACKING_X4_TERMS = frozenset(
    ("racket_position", "racket_velocity", "racket_normal")
)
ONE_SHOT_SCOPE = {
    "virtual_landing": "per_strike",
    "death_penalty": "per_termination",
}
RACKET_PROGRESS_RAW_ABS_CLAMP_M = 0.15
_UNIT_RAW_BOUND_TERMS = frozenset(
    (
        "upright_exp",
        "hold_ready",
        "base_decel",
        "post_strike_brake",
        "hold_heading",
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "lower_body_pose_imitation",
        "racket_position",
        "racket_velocity",
        "racket_normal",
        "base_position",
        "racket_strike_success",
        "strike_capture_bonus",
        "virtual_landing",
        "arm_overreach",
        "hit_unstable_support",
        "death_penalty",
    )
)


class CausalAuditError(RuntimeError):
    """The live Reward audit cannot make a causal claim."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise CausalAuditError(f"{label} must be a finite number, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalAuditError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise CausalAuditError(f"{label} must be finite")
    return out


def _tensor_sha256(tensor: Any) -> str:
    """Hash a real tensor without relying on ``repr`` or object addresses."""

    if not hasattr(tensor, "detach"):
        raise CausalAuditError("causal probe source is not a tensor")
    value = tensor.detach().contiguous().cpu()
    payload = {
        "dtype": str(value.dtype),
        "shape": [int(x) for x in value.shape],
        "bytes_sha256": _sha256_bytes(value.numpy().tobytes()),
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _state_sha256(slots: list["TensorSlot"]) -> str:
    rows = [
        {"label": slot.label, "sha256": _tensor_sha256(slot.tensor)}
        for slot in sorted(slots, key=lambda item: item.label)
    ]
    return _sha256_bytes(_canonical_bytes(rows))


@dataclass
class TensorSlot:
    label: str
    tensor: Any


class _TensorTransaction:
    def __init__(self, slots: list[TensorSlot]):
        labels = [slot.label for slot in slots]
        if not slots or len(labels) != len(set(labels)):
            raise CausalAuditError("transaction needs unique non-empty tensor slots")
        self.slots = slots
        self.saved = [slot.tensor.detach().clone() for slot in slots]

    def restore(self) -> None:
        for slot, saved in zip(self.slots, self.saved):
            slot.tensor.copy_(saved)

    def __enter__(self) -> "_TensorTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()


def _env0_scalar(value: Any, *, label: str) -> float:
    if not hasattr(value, "detach") or value.ndim != 1 or int(value.shape[0]) != 1:
        raise CausalAuditError(
            f"{label} must return a live [1] tensor in the one-env audit"
        )
    return _finite(value.detach().cpu().item(), label=label)


@dataclass
class ProbeSetup:
    slots: list[TensorSlot]
    prepare_baseline: Callable[[], None]
    prepare_worsened: Callable[[], None]
    axis_slot_labels: tuple[str, ...]
    notes: str
    verify_production_readback: Callable[[str], None] | None = None
    production_readback_contract: str | None = None


def _copy_row(dst: Any, src: Any) -> None:
    dst[0].copy_(src[0] if getattr(src, "ndim", 0) > 1 else src)


def _fill_row(dst: Any, value: float) -> None:
    dst[0].fill_(float(value))


def _flat_finite_values(tensor: Any, *, label: str) -> list[float]:
    if not hasattr(tensor, "detach"):
        raise CausalAuditError(f"{label} production readback is not a tensor")
    values = [
        _finite(value, label=f"{label}[{index}]")
        for index, value in enumerate(
            tensor.detach().reshape(-1).cpu().tolist()
        )
    ]
    if not values:
        raise CausalAuditError(f"{label} production readback is empty")
    return values


def _assert_close(actual: float, expected: float, *, label: str, atol: float = 1e-5) -> None:
    if abs(float(actual) - float(expected)) > float(atol):
        raise CausalAuditError(
            f"{label} production readback {actual} != expected {expected} "
            f"(atol={atol})"
        )


def _command(env: Any, name: str) -> Any:
    return env.command_manager.get_term(name)


def _scene_asset(env: Any, name: str) -> Any:
    return env.scene[name]


def _term_param(term_cfg: Any, name: str, default: Any = None) -> Any:
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        raise CausalAuditError("live RewardTerm cfg params must be a dict")
    return params.get(name, default)


def _simple_match_setup(
    measured: Any,
    target: Any,
    *,
    label: str,
    delta: float,
) -> ProbeSetup:
    slots = [TensorSlot(label, measured)]

    def baseline() -> None:
        _copy_row(measured, target)

    def worse() -> None:
        _copy_row(measured, target)
        measured[0].reshape(-1)[0].add_(float(delta))

    return ProbeSetup(slots, baseline, worse, (label,), "controlled exact-match then one-axis error")


def _setup_upright(env: Any, term_cfg: Any) -> ProbeSetup:
    del term_cfg
    # projected_gravity_b is a derived property.  Control its authoritative
    # root-link quaternion buffer, not the temporary returned tensor.
    quat = _scene_asset(env, "robot").data.root_link_state_w[:, 3:7]
    slot = TensorSlot("base_uprightness", quat)

    def baseline() -> None:
        quat[0].copy_(quat.new_tensor([1.0, 0.0, 0.0, 0.0]))

    def worse() -> None:
        quat[0].copy_(
            quat.new_tensor(
                [math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0]
            )
        )

    def verify(stage: str) -> None:
        projected = _flat_finite_values(
            _scene_asset(env, "robot").data.projected_gravity_b[0],
            label=f"upright_exp.{stage}.projected_gravity_b",
        )
        if len(projected) != 3:
            raise CausalAuditError(
                "upright_exp production projected_gravity_b must have three components"
            )
        expected_z = -1.0 if stage == "baseline" else -math.cos(math.pi / 6)
        _assert_close(
            projected[2],
            expected_z,
            label=f"upright_exp.{stage}.projected_gravity_b.z",
        )

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (slot.label,),
        "0 tilt -> controlled 30deg-class tilt",
        verify,
        "authoritative root-link quaternion -> robot.data.projected_gravity_b.z",
    )


def _setup_motion(env: Any, term_cfg: Any, term_name: str) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    std = _finite(_term_param(term_cfg, "std"), label=f"{term_name}.std")
    mapping = {
        "motion_global_anchor_pos": ("anchor_position_error", "robot_anchor_pos_w", "anchor_pos_w", std),
        "motion_body_pos": ("body_position_error", "robot_body_pos_w", "body_pos_relative_w", std),
        "motion_body_lin_vel": (
            "body_linear_velocity_error",
            "robot_body_lin_vel_w",
            "body_lin_vel_w",
            std,
        ),
        "motion_body_ang_vel": (
            "body_angular_velocity_error",
            "robot_body_ang_vel_w",
            "body_ang_vel_w",
            std,
        ),
    }
    if term_name in mapping:
        axis, measured_name, target_name, delta = mapping[term_name]
        if term_name.startswith("motion_body_"):
            data_name = {
                "motion_body_pos": "body_pos_w",
                "motion_body_lin_vel": "body_lin_vel_w",
                "motion_body_ang_vel": "body_ang_vel_w",
            }[term_name]
            source = getattr(command.robot.data, data_name)
            command_body_names = list(command.cfg.body_names)
            configured_names = _term_param(term_cfg, "body_names")
            local_ids = [
                index
                for index, body_name in enumerate(command_body_names)
                if configured_names is None or body_name in configured_names
            ]
            global_ids = [int(command.body_indexes[index]) for index in local_ids]
            if not global_ids:
                raise CausalAuditError(f"{term_name}: resolved zero rewarded bodies")
            target = getattr(command, target_name)[:, local_ids]
            slots = [TensorSlot(axis, source)]

            def baseline_body() -> None:
                source[0, global_ids] = target[0]

            def worse_body() -> None:
                baseline_body()
                source[0, global_ids[0]].reshape(-1)[0].add_(float(delta))

            setup = ProbeSetup(
                slots,
                baseline_body,
                worse_body,
                (axis,),
                "controlled exact-match then one rewarded-body one-axis error",
            )
        else:
            setup = _simple_match_setup(
                getattr(command, measured_name),
                getattr(command, target_name),
                label=axis,
                delta=delta,
            )
        if term_name.startswith("motion_body_"):
            # ``command.in_hold`` is a derived property.  Its two authoritative
            # sources must be controlled or a reset that happens to be held can
            # silently gate both paired calls to zero.
            hold_counter = command.hold_counter
            hold_metric = command.metrics["in_hold"]
            setup.slots.extend(
                [
                    TensorSlot(
                        f"{term_name}.fixed_hold_counter", hold_counter
                    ),
                    TensorSlot(
                        f"{term_name}.fixed_hold_metric", hold_metric
                    ),
                ]
            )
            old_baseline = setup.prepare_baseline
            old_worse = setup.prepare_worsened

            def baseline_swing() -> None:
                hold_counter[0] = 0
                hold_metric[0] = 0
                old_baseline()

            def worse_swing() -> None:
                hold_counter[0] = 0
                hold_metric[0] = 0
                old_worse()

            def verify_swing(stage: str) -> None:
                values = _flat_finite_values(
                    command.in_hold[0],
                    label=f"{term_name}.{stage}.in_hold",
                )
                if any(bool(value) for value in values):
                    raise CausalAuditError(
                        f"{term_name}: production in_hold getter remained true"
                    )

            setup.prepare_baseline = baseline_swing
            setup.prepare_worsened = worse_swing
            setup.verify_production_readback = verify_swing
            setup.production_readback_contract = (
                "authoritative hold_counter + metrics[in_hold] -> command.in_hold false"
            )
            setup.notes += (
                "; swing-only authoritative hold_counter/metric zero "
                "identically in both calls"
            )
        return setup
    if term_name in ("motion_global_anchor_ori", "motion_body_ori"):
        axis = (
            "anchor_orientation_error"
            if term_name == "motion_global_anchor_ori"
            else "body_orientation_error"
        )
        if term_name == "motion_global_anchor_ori":
            measured = command.robot_anchor_quat_w
            target = command.anchor_quat_w
            selected_source = measured
            global_ids = None
        else:
            selected_source = command.robot.data.body_quat_w
            command_body_names = list(command.cfg.body_names)
            configured_names = _term_param(term_cfg, "body_names")
            local_ids = [
                index
                for index, body_name in enumerate(command_body_names)
                if configured_names is None or body_name in configured_names
            ]
            global_ids = [int(command.body_indexes[index]) for index in local_ids]
            if not global_ids:
                raise CausalAuditError(f"{term_name}: resolved zero rewarded bodies")
            target = command.body_quat_relative_w[:, local_ids]
        slots = [TensorSlot(axis, selected_source)]

        def baseline() -> None:
            if global_ids is None:
                _copy_row(selected_source, target)
            else:
                selected_source[0, global_ids] = target[0]

        def worse() -> None:
            baseline()
            row = (
                selected_source[0].reshape(-1, 4)
                if global_ids is None
                else selected_source[0, global_ids].reshape(-1, 4)
            )
            v = row[0]
            idx = int(v.detach().abs().argmin().item())
            basis = v.new_zeros(4)
            basis[idx] = 1.0
            orth = basis - torch_dot(v, basis) * v
            if global_ids is None:
                row[0].copy_(orth / orth.norm())
            else:
                selected_source[0, global_ids[0]].copy_(orth / orth.norm())

        setup = ProbeSetup(
            slots,
            baseline,
            worse,
            (axis,),
            "exact quaternion match -> normalized orthogonal quaternion error",
        )
        if term_name == "motion_body_ori":
            hold_counter = command.hold_counter
            hold_metric = command.metrics["in_hold"]
            setup.slots.extend(
                [
                    TensorSlot(
                        f"{term_name}.fixed_hold_counter", hold_counter
                    ),
                    TensorSlot(
                        f"{term_name}.fixed_hold_metric", hold_metric
                    ),
                ]
            )
            old_baseline = setup.prepare_baseline
            old_worse = setup.prepare_worsened

            def baseline_swing() -> None:
                hold_counter[0] = 0
                hold_metric[0] = 0
                old_baseline()

            def worse_swing() -> None:
                hold_counter[0] = 0
                hold_metric[0] = 0
                old_worse()

            def verify_swing(stage: str) -> None:
                values = _flat_finite_values(
                    command.in_hold[0],
                    label=f"{term_name}.{stage}.in_hold",
                )
                if any(bool(value) for value in values):
                    raise CausalAuditError(
                        f"{term_name}: production in_hold getter remained true"
                    )

            setup.prepare_baseline = baseline_swing
            setup.prepare_worsened = worse_swing
            setup.verify_production_readback = verify_swing
            setup.production_readback_contract = (
                "authoritative hold_counter + metrics[in_hold] -> command.in_hold false"
            )
            setup.notes += (
                "; swing-only authoritative hold_counter/metric zero "
                "identically in both calls"
            )
        return setup
    raise CausalAuditError(f"no motion mutation recipe for {term_name}")


def _setup_racket(env: Any, term_cfg: Any, term_name: str) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    if term_name == "racket_position":
        measured = command.racket_pos_w
        target_now = (
            command.racket_target_pos_w
            - command.racket_target_vel_w * command.time_to_strike.unsqueeze(-1)
        )
        gate = getattr(command, "strike_window_pos", None)
        gate = command.strike_window if gate is None else gate
        setup = _simple_match_setup(
            measured,
            target_now,
            label="racket_position_error",
            delta=_finite(_term_param(term_cfg, "std"), label="racket_position.std"),
        )
    elif term_name == "racket_velocity":
        measured = command.racket_lin_vel_w
        gate = getattr(command, "strike_window_wide", None)
        gate = command.strike_window if gate is None else gate
        setup = _simple_match_setup(
            measured,
            command.racket_target_vel_w,
            label="racket_velocity_error",
            delta=_finite(_term_param(term_cfg, "std"), label="racket_velocity.std"),
        )
    elif term_name == "base_position":
        measured = command.base_pos_w[:, :2]
        gate = command.pre_strike
        setup = _simple_match_setup(
            measured,
            command.base_target_pos_w,
            label="base_task_position_error",
            delta=_finite(_term_param(term_cfg, "std"), label="base_position.std"),
        )
    elif term_name == "racket_progress":
        progress = command.racket_progress
        gate = command.pre_strike
        slot = TensorSlot("racket_target_progress", progress)

        def baseline_progress() -> None:
            progress[0] = 0.1

        def worse_progress() -> None:
            progress[0] = 0.0

        setup = ProbeSetup(
            [slot],
            baseline_progress,
            worse_progress,
            (slot.label,),
            "positive controlled progress -> zero progress",
        )
    else:
        raise CausalAuditError(f"no racket mutation recipe for {term_name}")
    gate_slot = TensorSlot(f"{term_name}.fixed_gate", gate)
    setup.slots.append(gate_slot)
    old_baseline = setup.prepare_baseline
    old_worse = setup.prepare_worsened

    def baseline_with_gate() -> None:
        gate[0] = True
        old_baseline()

    def worse_with_gate() -> None:
        gate[0] = True
        old_worse()

    setup.prepare_baseline = baseline_with_gate
    setup.prepare_worsened = worse_with_gate
    setup.notes += "; gate=true identically in both calls"
    if term_name == "racket_velocity" and _term_param(
        term_cfg, "pos_gate_radius"
    ) is not None:
        # The production velocity term can be power-gated by current racket
        # position.  Pin that independent gate at its maximum so a distant
        # reset cannot numerically erase the velocity causal axis.
        position = command.racket_pos_w
        target_position = command.racket_target_pos_w
        setup.slots.append(
            TensorSlot("racket_velocity.fixed_position_gate", position)
        )
        old_baseline_gate = setup.prepare_baseline
        old_worse_gate = setup.prepare_worsened

        def baseline_with_position_gate() -> None:
            _copy_row(position, target_position)
            old_baseline_gate()

        def worse_with_position_gate() -> None:
            _copy_row(position, target_position)
            old_worse_gate()

        setup.prepare_baseline = baseline_with_position_gate
        setup.prepare_worsened = worse_with_position_gate
        setup.notes += "; production position power-gate pinned at exact target"
    return setup


def _setup_racket_normal(env: Any, term_cfg: Any) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    # Use the production helper so face-command A-vs-A and clip-reference
    # signed-face semantics cannot drift in this audit.
    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import _face_pair

    measured, target = _face_pair(command)
    gate = getattr(command, "strike_window_wide", None)
    gate = command.strike_window if gate is None else gate
    slots = [
        TensorSlot("signed_racket_face_error", measured),
        TensorSlot("racket_normal.fixed_gate", gate),
    ]
    position = None
    target_position = None
    if _term_param(term_cfg, "pos_gate_radius") is not None:
        position = command.racket_pos_w
        target_position = command.racket_target_pos_w
        slots.append(
            TensorSlot("racket_normal.fixed_position_gate", position)
        )

    def baseline() -> None:
        gate[0] = True
        if position is not None:
            _copy_row(position, target_position)
        _copy_row(measured, target)

    def worse() -> None:
        gate[0] = True
        if position is not None:
            _copy_row(position, target_position)
        _copy_row(measured, target)
        row = measured[0].reshape(-1, 3)
        v = row[0]
        # Pick the least parallel canonical axis and Gram-Schmidt it.  This is
        # a finite unit normal exactly orthogonal to the demanded face.
        idx = int(v.detach().abs().argmin().item())
        basis = v.new_zeros(3)
        basis[idx] = 1.0
        orth = basis - torch_dot(v, basis) * v
        row[0].copy_(orth / orth.norm())

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("signed_racket_face_error",),
        (
            "exact face match -> orthogonal face; strike gate fixed true"
            + (
                "; production position power-gate pinned at exact target"
                if position is not None
                else ""
            )
        ),
    )


def torch_dot(a: Any, b: Any) -> Any:
    # Kept tiny and local so importing this script on a host does not import
    # torch/Isaac.  Both operands are live torch tensors at runtime.
    return (a * b).sum()


def _setup_virtual_landing(env: Any, term_cfg: Any) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    if _finite(
        _term_param(term_cfg, "settle_delay_s", 0.0),
        label="virtual_landing.settle_delay_s",
    ) != 0.0:
        raise CausalAuditError(
            "transactional virtual_landing probe currently requires settle_delay_s=0; "
            "a delayed prize needs a separately reviewed attempt-clock transaction"
        )
    landing = command.vb_landing_xy
    target = command._vb_target_xy
    gates = [
        ("virtual_landing.fixed_fired", command.vb_fired),
        ("virtual_landing.fixed_landing_valid", command.vb_landing_valid),
        ("virtual_landing.fixed_net_clear", command.vb_net_clear),
        ("virtual_landing.fixed_opponent", command.vb_on_opponent),
    ]
    slots = [TensorSlot("virtual_landing_error", landing)] + [
        TensorSlot(label, tensor) for label, tensor in gates
    ]

    def fixed_gate() -> None:
        for _, tensor in gates:
            tensor[0] = True

    def baseline() -> None:
        fixed_gate()
        _copy_row(landing, target)

    def worse() -> None:
        fixed_gate()
        _copy_row(landing, target)
        sigma = _finite(command.cfg.vb_landing_sigma, label="vb_landing_sigma")
        landing[0, 0].add_(sigma)

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("virtual_landing_error",),
        "all legality/fired gates fixed true; landing error alone grows by one sigma",
    )


def _setup_action_rate(env: Any, term_cfg: Any) -> ProbeSetup:
    del term_cfg
    current = env.action_manager.action
    previous = env.action_manager.prev_action
    slots = [
        TensorSlot("clamped_action_rate", current),
        TensorSlot("action_rate.fixed_previous", previous),
    ]

    def baseline() -> None:
        current[0].zero_()
        previous[0].zero_()

    def worse() -> None:
        current[0].zero_()
        previous[0].zero_()
        current[0, 0] = 1.0

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("clamped_action_rate",),
        "zero first difference -> one-axis unit first difference",
    )


def _setup_action_acc(env: Any, term_cfg: Any) -> ProbeSetup:
    term = env.action_manager.get_term(str(_term_param(term_cfg, "action_name", "joint_pos")))
    current = term.raw_actions
    previous = term.prev_raw_actions
    before_previous = term.prev_prev_raw_actions
    previous_valid = term._prev_raw_actions_valid
    before_previous_valid = term._prev_prev_raw_actions_valid
    slots = [
        TensorSlot("action_acceleration", current),
        TensorSlot("action_acc.fixed_previous", previous),
        TensorSlot("action_acc.fixed_before_previous", before_previous),
        TensorSlot("action_acc.fixed_previous_valid", previous_valid),
        TensorSlot("action_acc.fixed_before_previous_valid", before_previous_valid),
    ]

    def baseline() -> None:
        current[0].zero_()
        previous[0].zero_()
        before_previous[0].zero_()
        previous_valid[0] = True
        before_previous_valid[0] = True

    def worse() -> None:
        baseline()
        current[0, 0] = 1.0

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("action_acceleration",),
        "zero second difference -> one-axis unit second difference; validity fixed true",
    )


def _controlled_limit_values(limits: Any, default: Any, source: Any) -> tuple[Any, Any]:
    lower = limits[0, :, 0] if limits.ndim == 3 else limits[:, 0]
    upper = limits[0, :, 1] if limits.ndim == 3 else limits[:, 1]
    default_row = default[0] if default.ndim == 2 else default
    baseline = default_row
    # Pick the widest stance-safe joint and move just inside the upper 8% band.
    span = upper - lower
    distance = (upper - default_row) / span
    joint = int(distance.detach().argmax().item())
    worse = default_row.detach().clone()
    worse[joint] = upper[joint] - 0.02 * span[joint]
    return baseline, worse


def _setup_qdes_limit(env: Any, term_cfg: Any) -> ProbeSetup:
    action = env.action_manager.get_term(str(_term_param(term_cfg, "action_name", "joint_pos")))
    source = action.processed_actions
    data = action._asset.data
    baseline_value, worse_value = _controlled_limit_values(
        data.soft_joint_pos_limits, data.default_joint_pos, source
    )
    slot = TensorSlot("qdes_joint_soft_limit", source)

    def baseline() -> None:
        _copy_row(source, baseline_value)

    def worse() -> None:
        _copy_row(source, worse_value)

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (slot.label,),
        "default qdes -> one joint inside the upper soft-limit band",
    )


def _setup_actual_limit(env: Any, term_cfg: Any) -> ProbeSetup:
    asset_cfg = _term_param(term_cfg, "asset_cfg")
    data = _scene_asset(env, asset_cfg.name).data
    source = data.joint_pos
    baseline_value, worse_value = _controlled_limit_values(
        data.soft_joint_pos_limits, data.default_joint_pos, source
    )
    slot = TensorSlot("actual_joint_soft_limit", source)

    def baseline() -> None:
        _copy_row(source, baseline_value)

    def worse() -> None:
        _copy_row(source, worse_value)

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (slot.label,),
        "default actual q -> one joint inside the upper soft-limit band",
    )


def _setup_joint_torque(env: Any, term_cfg: Any) -> ProbeSetup:
    asset_cfg = _term_param(term_cfg, "asset_cfg")
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    torque = _scene_asset(env, asset_name).data.applied_torque
    slot = TensorSlot("joint_torque", torque)

    def baseline() -> None:
        torque[0].zero_()

    def worse() -> None:
        torque[0].zero_()
        torque[0, 0] = 1.0

    return ProbeSetup([slot], baseline, worse, (slot.label,), "zero torque -> one-axis unit torque")


def _setup_arm_torque_saturation(env: Any, term_cfg: Any) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
        _require_explicit_torque_saturation_backend,
        _torque_sat_joint_idx,
    )

    _, ids = _torque_sat_joint_idx(env, str(_term_param(term_cfg, "command_name")))
    _require_explicit_torque_saturation_backend(command, ids)
    torque = command.robot.data.computed_torque
    limits = command.robot.data.joint_effort_limits
    if not ids or not hasattr(torque, "detach") or not hasattr(limits, "detach"):
        raise CausalAuditError("arm_torque_saturation live torque/limit sources are unavailable")
    slots = [
        TensorSlot("arm_torque_saturation", torque),
        TensorSlot("arm_torque.fixed_effort_limits", limits),
    ]

    def baseline() -> None:
        torque[0, ids] = 0.0

    def worse() -> None:
        baseline()
        torque[0, ids[0]] = 2.0 * limits[0, ids[0]]

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("arm_torque_saturation",),
        "computed torque within envelope -> one reviewed arm/waist joint at 2x live limit",
    )


def _setup_robot_l2(
    env: Any, term_cfg: Any, *, term_name: str, axis: str, tensor_name: str, component: int | None
) -> ProbeSetup:
    asset_cfg = _term_param(term_cfg, "asset_cfg")
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    tensor = getattr(_scene_asset(env, asset_name).data, tensor_name)
    slot = TensorSlot(axis, tensor)

    def baseline() -> None:
        tensor[0].zero_()

    def worse() -> None:
        tensor[0].zero_()
        flat = tensor[0].reshape(-1)
        flat[0 if component is None else component] = 1.0

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (axis,),
        f"{term_name}: zero live state -> one-axis unit state",
    )


def _setup_root_velocity_l2(
    env: Any, term_cfg: Any, *, term_name: str, axis: str, velocity_slice: slice, component: int
) -> ProbeSetup:
    asset_cfg = _term_param(term_cfg, "asset_cfg")
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    state = _scene_asset(env, asset_name).data.root_state_w
    velocity = state[:, velocity_slice]
    quat = state[:, 3:7]
    production_attr = (
        "root_ang_vel_b" if term_name == "base_ang_vel_xy" else "root_lin_vel_b"
    )
    slots = [
        TensorSlot(axis, velocity),
        TensorSlot(f"{term_name}.fixed_root_quaternion", quat),
    ]

    def baseline() -> None:
        quat[0].copy_(quat.new_tensor([1.0, 0.0, 0.0, 0.0]))
        velocity[0].zero_()

    def worse() -> None:
        baseline()
        velocity[0, component] = 1.0

    def verify(stage: str) -> None:
        readback = _flat_finite_values(
            getattr(_scene_asset(env, asset_name).data, production_attr)[0],
            label=f"{term_name}.{stage}.{production_attr}",
        )
        if len(readback) != 3:
            raise CausalAuditError(
                f"{term_name}: production {production_attr} must have three components"
            )
        expected = 0.0 if stage == "baseline" else 1.0
        _assert_close(
            readback[component],
            expected,
            label=f"{term_name}.{stage}.{production_attr}[{component}]",
        )
        for index, value in enumerate(readback):
            if index != component:
                _assert_close(
                    value,
                    0.0,
                    label=f"{term_name}.{stage}.{production_attr}[{index}]",
                )

    return ProbeSetup(
        slots,
        baseline,
        worse,
        (axis,),
        f"{term_name}: identity root frame fixed; zero world velocity -> one-axis unit velocity",
        verify,
        f"authoritative root_state_w -> robot.data.{production_attr}",
    )


def _setup_undesired_contacts(env: Any, term_cfg: Any) -> ProbeSetup:
    sensor_cfg = _term_param(term_cfg, "sensor_cfg")
    threshold = _finite(_term_param(term_cfg, "threshold"), label="undesired_contacts.threshold")
    history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history
    ids = list(sensor_cfg.body_ids)
    if not ids:
        raise CausalAuditError("undesired_contacts resolved no bodies")
    slot = TensorSlot("undesired_contact", history)

    def baseline() -> None:
        history[0, :, ids, :] = 0.0

    def worse() -> None:
        baseline()
        history[0, 0, ids[0], 2] = 2.0 * threshold

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (slot.label,),
        "all selected contact histories zero -> one selected body above threshold",
    )


def _setup_command_magnitude(
    env: Any, term_cfg: Any, *, term_name: str, axis: str, attr: str
) -> ProbeSetup:
    command = _command(env, str(_term_param(term_cfg, "command_name")))
    tensor = getattr(command, attr)
    slot = TensorSlot(axis, tensor)

    def baseline() -> None:
        tensor[0] = 0.0

    def worse() -> None:
        tensor[0] = 1.0

    return ProbeSetup(
        [slot],
        baseline,
        worse,
        (axis,),
        f"{term_name}: zero cached live magnitude -> unit magnitude",
    )


def _setup_foot_soft_landing(env: Any, term_cfg: Any) -> ProbeSetup:
    sensor_cfg = _term_param(term_cfg, "sensor_cfg")
    threshold = _finite(
        _term_param(term_cfg, "force_threshold_n", 300.0),
        label="foot_soft_landing.force_threshold_n",
    )
    sensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history
    contact_time = getattr(sensor.data, "current_contact_time", None)
    if contact_time is None:
        raise CausalAuditError(
            "foot_soft_landing requires live current_contact_time for a controlled first-contact gate"
        )
    ids = list(sensor_cfg.body_ids)
    if len(ids) != 2:
        raise CausalAuditError("foot_soft_landing requires exactly two resolved feet")
    slots = [
        TensorSlot("foot_landing_impact", history),
        TensorSlot("foot_landing.fixed_first_contact", contact_time),
    ]

    def fixed_first_contact() -> None:
        contact_time[0, ids] = float(env.step_dt) * 0.5

    def baseline() -> None:
        fixed_first_contact()
        history[0, :, ids, :] = 0.0

    def worse() -> None:
        baseline()
        history[0, 0, ids[0], 2] = 2.0 * threshold

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("foot_landing_impact",),
        "first-contact clock fixed identically; vertical impact alone grows 0 -> 2x threshold",
    )


def _setup_death(env: Any, term_cfg: Any) -> ProbeSetup:
    term_names = tuple(_term_param(term_cfg, "term_names"))
    expected = (
        "base_fell_tilt",
        "base_too_low",
        "joint_actual_forbidden",
        "joint_qdes_forbidden",
        "robot_hit_table",
    )
    if term_names != expected:
        raise CausalAuditError(
            "death_penalty does not bind the exact hard-safety termination union"
        )
    manager = env.termination_manager
    active = tuple(getattr(manager, "active_terms", ()))
    if any(name not in active for name in term_names):
        raise CausalAuditError(
            "death_penalty hard-safety termination masks are unavailable"
        )
    masks = [manager.get_term(name) for name in term_names]
    slots = [
        TensorSlot(
            "unsafe_termination"
            if index == 0
            else f"death.fixed_hard_safety_mask.{name}",
            mask,
        )
        for index, (name, mask) in enumerate(zip(term_names, masks))
    ]

    def baseline() -> None:
        for mask in masks:
            mask[0] = False

    def worse() -> None:
        baseline()
        masks[0][0] = True

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("unsafe_termination",),
        "all hard-safety masks false -> isolated base_fell_tilt true; "
        "reference-envelope masks are not inputs",
    )


def _setup_support(env: Any, term_cfg: Any) -> ProbeSetup:
    sensor_cfg = _term_param(term_cfg, "sensor_cfg")
    command = _command(
        env, str(_term_param(term_cfg, "command_name", "racket_target"))
    )
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w
    window = command.strike_window
    ids = list(sensor_cfg.body_ids)
    slots = [
        TensorSlot("strike_support", forces),
        TensorSlot("support.fixed_strike_window", window),
    ]

    def baseline() -> None:
        window[0] = True
        if len(ids) != 2:
            raise CausalAuditError("hit_unstable_support requires exactly two resolved feet")
        forces[0, ids, :] = 0.0
        forces[0, ids, 2] = 100.0

    def worse() -> None:
        baseline()
        forces[0, ids[1], :].zero_()

    return ProbeSetup(
        slots,
        baseline,
        worse,
        ("strike_support",),
        "strike gate fixed true; two-foot support -> one-foot support",
    )


_SETUP_BY_TERM: dict[str, Callable[[Any, Any], ProbeSetup]] = {
    "upright_exp": _setup_upright,
    "racket_position": lambda e, c: _setup_racket(e, c, "racket_position"),
    "racket_velocity": lambda e, c: _setup_racket(e, c, "racket_velocity"),
    "racket_normal": _setup_racket_normal,
    "base_position": lambda e, c: _setup_racket(e, c, "base_position"),
    "racket_progress": lambda e, c: _setup_racket(e, c, "racket_progress"),
    "virtual_landing": _setup_virtual_landing,
    "action_rate_clamped": _setup_action_rate,
    "action_acc_l2": _setup_action_acc,
    "qdes_limit_barrier": _setup_qdes_limit,
    "joint_limit": _setup_actual_limit,
    "joint_torques": _setup_joint_torque,
    "arm_torque_saturation": _setup_arm_torque_saturation,
    "base_ang_vel_xy": lambda e, c: _setup_root_velocity_l2(
        e,
        c,
        term_name="base_ang_vel_xy",
        axis="base_roll_pitch_rate",
        velocity_slice=slice(10, 13),
        component=0,
    ),
    "base_lin_vel_z": lambda e, c: _setup_root_velocity_l2(
        e,
        c,
        term_name="base_lin_vel_z",
        axis="base_vertical_speed",
        velocity_slice=slice(7, 10),
        component=2,
    ),
    "joint_vel": lambda e, c: _setup_robot_l2(
        e,
        c,
        term_name="joint_vel",
        axis="joint_speed",
        tensor_name="joint_vel",
        component=None,
    ),
    "undesired_contacts": _setup_undesired_contacts,
    "foot_slip_sq": lambda e, c: _setup_command_magnitude(
        e,
        c,
        term_name="foot_slip_sq",
        axis="stance_foot_slip",
        attr="foot_slip_sq",
    ),
    "foot_velocity": lambda e, c: _setup_command_magnitude(
        e,
        c,
        term_name="foot_velocity",
        axis="foot_speed",
        attr="foot_vel_sq",
    ),
    "foot_soft_landing": _setup_foot_soft_landing,
    "death_penalty": _setup_death,
    "hit_unstable_support": _setup_support,
}
for _motion_term in (
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
):
    _SETUP_BY_TERM[_motion_term] = (
        lambda e, c, name=_motion_term: _setup_motion(e, c, name)
    )


def _call_live_term(env: Any, term_cfg: Any, *, name: str) -> float:
    func = getattr(term_cfg, "func", None)
    params = getattr(term_cfg, "params", None)
    if not callable(func) or not isinstance(params, dict):
        raise CausalAuditError(f"{name}: live RewardManager term cfg is incomplete")
    return _env0_scalar(func(env, **params), label=f"{name}.raw")


def probe_active_objective(
    env: Any,
    manager: Any,
    taxonomy_row: dict[str, Any],
    *,
    step_dt_s: float,
) -> dict[str, Any]:
    name = str(taxonomy_row["name"])
    setup_factory = _SETUP_BY_TERM.get(name)
    if setup_factory is None:
        return {
            "term_name": name,
            "group": taxonomy_row["group"],
            "causal_axis": taxonomy_row["causal_axis"],
            "accounting_scope": ONE_SHOT_SCOPE.get(name, "per_control_step"),
            "status": "unsupported_fail_closed",
            "reason": "no reviewed transactional mutation recipe",
        }
    term_cfg = manager.get_term_cfg(name)
    setup = setup_factory(env, term_cfg)
    if tuple(setup.axis_slot_labels) != (str(taxonomy_row["causal_axis"]),):
        raise CausalAuditError(
            f"{name}: mutation axis {setup.axis_slot_labels!r} does not equal taxonomy "
            f"{taxonomy_row['causal_axis']!r}"
        )
    weight = _finite(taxonomy_row["weight"], label=f"{name}.weight")
    original_state_sha = _state_sha256(setup.slots)
    with _TensorTransaction(setup.slots):
        setup.prepare_baseline()
        if setup.verify_production_readback is not None:
            setup.verify_production_readback("baseline")
        baseline_state_sha = _state_sha256(setup.slots)
        baseline_context_sha = _state_sha256(
            [
                slot
                for slot in setup.slots
                if slot.label not in setup.axis_slot_labels
            ]
        )
        baseline_raw = _call_live_term(env, term_cfg, name=name)
        setup.prepare_worsened()
        if setup.verify_production_readback is not None:
            setup.verify_production_readback("worsened")
        worsened_state_sha = _state_sha256(setup.slots)
        worsened_context_sha = _state_sha256(
            [
                slot
                for slot in setup.slots
                if slot.label not in setup.axis_slot_labels
            ]
        )
        worsened_raw = _call_live_term(env, term_cfg, name=name)
    if _state_sha256(setup.slots) != original_state_sha:
        raise CausalAuditError(f"{name}: transactional live tensors were not restored")
    if baseline_state_sha == worsened_state_sha:
        raise CausalAuditError(f"{name}: controlled states are byte-identical")
    if baseline_context_sha != worsened_context_sha:
        raise CausalAuditError(
            f"{name}: nuisance/gating context changed between paired calls"
        )
    baseline_weighted = baseline_raw * weight * step_dt_s
    worsened_weighted = worsened_raw * weight * step_dt_s
    delta = worsened_weighted - baseline_weighted
    if not math.isfinite(delta) or delta >= 0.0:
        raise CausalAuditError(
            f"{name}: live worsening is not strictly harmful: "
            f"raw {baseline_raw}->{worsened_raw}, weighted delta={delta}"
        )
    return {
        "term_name": name,
        "callable": taxonomy_row["callable"],
        "group": taxonomy_row["group"],
        "causal_axis": taxonomy_row["causal_axis"],
        "accounting_scope": ONE_SHOT_SCOPE.get(name, "per_control_step"),
        "status": "causal_pass",
        "baseline_raw": baseline_raw,
        "worsened_raw": worsened_raw,
        "baseline_weighted_per_step": baseline_weighted,
        "worsened_weighted_per_step": worsened_weighted,
        "weighted_delta_per_step": delta,
        "baseline_state_sha256": baseline_state_sha,
        "worsened_state_sha256": worsened_state_sha,
        "frozen_context_sha256": baseline_context_sha,
        "causal_axis_slot_labels": list(setup.axis_slot_labels),
        "frozen_context_slot_labels": [
            slot.label
            for slot in setup.slots
            if slot.label not in setup.axis_slot_labels
        ],
        "production_readback_contract": setup.production_readback_contract,
        "notes": setup.notes,
    }


def _callable_raw_abs_bound(
    row: dict[str, Any],
) -> tuple[float | None, str]:
    """Return a proven raw magnitude bound, never a guessed unit bound."""

    name = str(row["name"])
    params = row.get("params")
    params = params if isinstance(params, dict) else {}
    if name in _UNIT_RAW_BOUND_TERMS:
        return 1.0, "callable_contract_unit_interval_or_indicator"
    if name == "racket_progress":
        return (
            RACKET_PROGRESS_RAW_ABS_CLAMP_M,
            "callable_clamp_m",
        )
    if name == "qdes_limit_barrier":
        return 31.0, "sum_of_31_per_joint_unit_caps"
    if name == "joint_limit":
        count = params.get("expected_joint_count")
        if type(count) is int and count > 0:
            return float(count), "sum_of_expected_joint_count_per_joint_unit_caps"
        return None, "expected_joint_count_missing"
    if name == "action_rate_clamped":
        clamp = params.get("value_clamp")
        if type(clamp) in (int, float) and not isinstance(clamp, bool):
            clamp = float(clamp)
            if math.isfinite(clamp) and clamp > 0.0:
                return clamp, "callable_value_clamp"
        return None, "value_clamp_missing_or_invalid"
    if name == "action_acc_l2":
        clamp = params.get("value_clamp")
        if type(clamp) in (int, float) and not isinstance(clamp, bool):
            clamp = float(clamp)
            if math.isfinite(clamp) and clamp > 0.0:
                return clamp, "callable_value_clamp"
        return None, "unbounded_second_difference_without_value_clamp"
    if name == "foot_soft_landing":
        # Two reviewed feet, each excess fraction is clamped to 3.
        return 6.0, "two_feet_times_per_foot_excess_cap_3"
    return None, "unknown_or_unbounded_callable"


def _candidate_recipes(active_terms: list[dict[str, Any]], step_dt_s: float) -> list[dict[str, Any]]:
    candidates = []
    for candidate_id, scale in (("A_composed_baseline", 1.0), ("B_tracking_x4", 4.0)):
        unit_budgets = {
            group: {
                "dense_positive_per_control_step": 0.0,
                "dense_negative_per_control_step": 0.0,
                "one_shot_positive_per_event": 0.0,
                "one_shot_negative_per_event": 0.0,
            }
            for group in SUPPORTED_GROUPS
        }
        bounded_budgets = {
            group: {
                "dense_positive_per_control_step": 0.0,
                "dense_negative_per_control_step": 0.0,
                "one_shot_positive_per_event": 0.0,
                "one_shot_negative_per_event": 0.0,
            }
            for group in SUPPORTED_GROUPS
        }
        unknown_budgets = {
            group: {
                "dense_positive_per_control_step": [],
                "dense_negative_per_control_step": [],
                "one_shot_positive_per_event": [],
                "one_shot_negative_per_event": [],
            }
            for group in SUPPORTED_GROUPS
        }
        terms = []
        for row in active_terms:
            weight = float(row["weight"])
            if candidate_id == "B_tracking_x4" and row["name"] in TRACKING_X4_TERMS:
                weight *= scale
            # Unit raw budgets are explicit dimensional summaries, not an
            # empirical typical-value claim and not an automatic tuning rule.
            unit = weight * step_dt_s
            scope = ONE_SHOT_SCOPE.get(row["name"], "per_control_step")
            if scope == "per_control_step":
                key = (
                    "dense_positive_per_control_step"
                    if unit > 0.0
                    else "dense_negative_per_control_step"
                )
            else:
                key = (
                    "one_shot_positive_per_event"
                    if unit > 0.0
                    else "one_shot_negative_per_event"
                )
            unit_budgets[row["group"]][key] += unit
            callable_raw_abs_bound, bound_basis = _callable_raw_abs_bound(row)
            bounded = (
                None
                if callable_raw_abs_bound is None
                else unit * callable_raw_abs_bound
            )
            if bounded is None:
                unknown_budgets[row["group"]][key].append(row["name"])
            else:
                bounded_budgets[row["group"]][key] += bounded
            terms.append(
                {
                    "name": row["name"],
                    "weight": weight,
                    "unit_raw_weighted_budget": unit,
                    "callable_raw_abs_bound_for_budget": callable_raw_abs_bound,
                    "callable_raw_abs_bound_basis": bound_basis,
                    "callable_bound_status": (
                        "proven"
                        if callable_raw_abs_bound is not None
                        else "unknown_or_unbounded"
                    ),
                    "bounded_weighted_budget": bounded,
                }
            )
        for group in SUPPORTED_GROUPS:
            for key, names in unknown_budgets[group].items():
                if names:
                    bounded_budgets[group][key] = None
        candidates.append(
            {
                "candidate_id": candidate_id,
                "applied_to_training": False,
                "terms": terms,
                "unit_raw_dimensioned_budget_by_group": unit_budgets,
                "callable_bounded_dimensioned_budget_by_group": bounded_budgets,
                "unknown_or_unbounded_terms_by_group": unknown_budgets,
            }
        )
    return candidates


def _racket_progress_accounting(
    active_terms: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    *,
    step_dt_s: float,
) -> dict[str, Any]:
    terms = [row for row in active_terms if row["name"] == "racket_progress"]
    if not terms:
        return {
            "active": False,
            "status": "not_in_effective_recipe",
        }
    if len(terms) != 1:
        raise CausalAuditError("effective recipe contains duplicate racket_progress")
    term = terms[0]
    weight = _finite(term["weight"], label="racket_progress.weight")
    coverage = [
        row for row in coverage_rows if row["term_name"] == "racket_progress"
    ]
    if len(coverage) != 1:
        raise CausalAuditError("racket_progress causal coverage is not exactly one row")
    return {
        "active": True,
        "semantics": "signed_reset_gated_distance_difference_telescoping_by_swing",
        "raw_unit": "meters_of_racket_target_distance_reduction_per_control_step",
        "raw_clamp_m_per_control_step": [
            -RACKET_PROGRESS_RAW_ABS_CLAMP_M,
            RACKET_PROGRESS_RAW_ABS_CLAMP_M,
        ],
        "weight": weight,
        "policy_dt_s": step_dt_s,
        "unit_raw_weighted_per_control_step": weight * step_dt_s,
        "callable_abs_weighted_per_control_step_cap": (
            abs(weight) * step_dt_s * RACKET_PROGRESS_RAW_ABS_CLAMP_M
        ),
        "per_swing_signed_weighted_formula": (
            "weight * policy_dt * sum_eligible(clamp(prev_distance-current_distance,"
            "-0.15,0.15)); reset/resample steps contribute exactly zero"
        ),
        "controlled_prelaunch_probe": {
            key: coverage[0].get(key)
            for key in (
                "status",
                "baseline_raw",
                "worsened_raw",
                "baseline_weighted_per_step",
                "worsened_weighted_per_step",
                "weighted_delta_per_step",
            )
        },
        "empirical_rollout_accounting": {
            "status": "required_from_training_activation_ledger_not_prelaunch_fabricated",
            "eligible_prestrike_control_step_count": None,
            "eligible_swing_count": None,
            "reset_or_resample_zeroed_step_count": None,
            "signed_raw_sum_m": None,
            "positive_raw_sum_m": None,
            "negative_raw_sum_m": None,
            "weighted_signed_sum": None,
            "per_swing_weighted_signed_p5_p50_p95": None,
        },
        "interpretation_guard": (
            "the unit-raw value is not a sustainable per-step income claim; compare the "
            "callable cap and signed per-swing telescoping sum over an explicit eligible denominator"
        ),
    }


def build_action_ball_reward_intervention_contracts() -> dict[str, Any]:
    """Declare executable paired canaries without claiming their outcomes."""

    pending = "REQUIRED_LIVE_CANARY_NOT_EXECUTED"
    rows = [
        {
            "intervention_id": "delayed_hard_death",
            "status": pending,
            "purpose": (
                "prove a policy cannot profit by collecting dense income before "
                "a delayed hard-safety death"
            ),
            "required_terms": ["death_penalty"],
            "paired_branches": {
                "control": "stay_inside_hard_safety_envelope",
                "intervention": (
                    "take_the_matched_hard_violation_at_the_registered_lag"
                ),
            },
            "matched_state_contract": (
                "same serialized simulator state, command/action tape, reset "
                "generation, and RNG state up to the intervention edge"
            ),
            "intervention_lag_control_steps": [0, 12, 40, 78, 100],
            "readout": (
                "discounted_return_intervention_minus_control_from_pair_start"
            ),
            "acceptance": {
                "statistic": "paired_one_sided_95pct_confidence_interval",
                "threshold": "upper_bound_lt_0_at_every_registered_lag",
                "minimum_valid_pairs_per_lag": 64,
            },
            "result": None,
        },
        {
            "intervention_id": "reference_reset",
            "status": pending,
            "purpose": (
                "prove a reference-envelope reset is not a profitable escape "
                "from a matched legal continuation"
            ),
            "required_termination_terms": [
                "anchor_pos",
                "anchor_ori",
                "ee_body_pos",
            ],
            "paired_branches": {
                "control": "continue_from_matched_state_without_reference_reset",
                "intervention": (
                    "cross_exactly_one_registered_reference_envelope_term"
                ),
            },
            "matched_state_contract": (
                "same post-landing state, action identity, command tape, RNG "
                "state, and horizon; only the reference-envelope axis changes"
            ),
            "readout": (
                "discounted_return_reference_reset_minus_legal_continuation"
            ),
            "acceptance": {
                "statistic": "paired_one_sided_95pct_confidence_interval",
                "threshold": "upper_bound_lt_0_for_each_reference_term",
                "minimum_valid_pairs_per_reference_term": 64,
            },
            "result": None,
        },
        {
            "intervention_id": "soft_retreat_vs_cross",
            "status": pending,
            "purpose": (
                "prove the adopted q_des/actual soft barriers prefer retreat "
                "over moving deeper toward a hard crossing"
            ),
            "required_terms": [
                "qdes_limit_barrier",
                "joint_limit",
            ],
            "channels": [
                "qdes_only",
                "actual_only",
                "qdes_and_actual",
            ],
            "paired_branches": {
                "control": "retreat_from_the_soft_limit_band",
                "intervention": (
                    "cross_deeper_through_the_same_soft_limit_band"
                ),
            },
            "matched_state_contract": (
                "same joint, side, normalized margin depth, command/action "
                "context, RNG state, and no unrelated termination edge"
            ),
            "horizon_control_steps": [2, 45, 46, 180, 181],
            "readout": (
                "discounted_return_cross_minus_retreat_before_or_through_horizon"
            ),
            "acceptance": {
                "statistic": "paired_one_sided_95pct_confidence_interval",
                "threshold": (
                    "upper_bound_lt_0_for_every_channel_and_horizon"
                ),
                "minimum_valid_pairs_per_channel_horizon": 64,
            },
            "result": None,
        },
        {
            "intervention_id": "progress_closed_loop",
            "status": pending,
            "purpose": (
                "prove racket_progress cannot mint return on a closed distance loop"
            ),
            "required_terms": ["racket_progress"],
            "paired_branches": {
                "forward_loop": (
                    "leave_and_return_to_the_exact_same_racket_target_distance"
                ),
                "reverse_loop": (
                    "execute_the_time_reversed_distance_path_from_the_same_state"
                ),
            },
            "matched_state_contract": (
                "same swing/reset generation, identical start/end distance, "
                "no reset/resample step, and identical non-progress Reward inputs"
            ),
            "readout": [
                "absolute_undiscounted_racket_progress_return",
                "absolute_discounted_forward_reverse_mean_return",
            ],
            "acceptance": {
                "undiscounted_abs_max": 1.0e-6,
                "discounted_forward_reverse_mean_abs_max": 1.0e-6,
                "minimum_valid_closed_loops": 64,
            },
            "result": None,
        },
    ]
    return {
        "schema_version": 1,
        "status": pending,
        "executor_contract": (
            "restore each paired live Isaac state/RNG snapshot, execute both "
            "branches, retain raw pair rows, then evaluate the registered bound"
        ),
        "rows": rows,
    }


def build_live_causal_report(
    env: Any,
    effective_reward_receipt: dict[str, Any],
    *,
    step_dt_s: float,
) -> dict[str, Any]:
    from whole_body_tracking.utils.effective_reward_recipe import (
        REWARD_TERM_ROLE_OBJECTIVE,
        build_action_ball_reward_group_taxonomy,
        build_effective_reward_receipt,
    )

    if int(getattr(env, "num_envs", -1)) != 1:
        raise CausalAuditError("formal Reward causal audit requires exactly one live env")
    manager = getattr(env, "reward_manager", None)
    if manager is None:
        raise CausalAuditError("live env has no RewardManager")
    recipe_terms = effective_reward_receipt.get("terms")
    expected_live_names = {
        str(row["name"]) for row in recipe_terms
    }
    live_names = {
        str(name)
        for name in manager.active_terms
        if float(manager.get_term_cfg(name).weight) != 0.0
    }
    if live_names != expected_live_names:
        raise CausalAuditError(
            "post-compose recipe and live RewardManager non-zero term sets differ: "
            f"recipe_only={sorted(expected_live_names - live_names)}, "
            f"manager_only={sorted(live_names - expected_live_names)}"
        )
    live_manager_receipt = build_effective_reward_receipt(
        {
            name: manager.get_term_cfg(name)
            for name in sorted(live_names)
        },
        expected_sha256=effective_reward_receipt["sha256"],
    )
    if live_manager_receipt != effective_reward_receipt:
        raise CausalAuditError(
            "live RewardManager callable/weight/params differ from the composed recipe"
        )
    taxonomy = build_action_ball_reward_group_taxonomy(recipe_terms)
    active = [
        row
        for row in taxonomy["active_terms"]
        if row["role"] == REWARD_TERM_ROLE_OBJECTIVE
    ]
    recipe_by_name = {row["name"]: row for row in recipe_terms}
    active_with_params = [
        {**row, "params": dict(recipe_by_name[row["name"]]["params"])}
        for row in active
    ]
    rows = []
    for row in active:
        try:
            rows.append(
                probe_active_objective(
                    env, manager, row, step_dt_s=step_dt_s
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "term_name": row["name"],
                    "group": row["group"],
                    "causal_axis": row["causal_axis"],
                    "accounting_scope": ONE_SHOT_SCOPE.get(
                        row["name"], "per_control_step"
                    ),
                    "status": "probe_failed_closed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    names = [row["term_name"] for row in rows]
    if len(names) != len(set(names)) or set(names) != {
        row["name"] for row in active
    }:
        raise CausalAuditError("causal coverage table does not close over active objectives")
    groups = {}
    for group in SUPPORTED_GROUPS:
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {
            "active_objective_count": len(selected),
            "causal_pass_count": sum(row["status"] == "causal_pass" for row in selected),
            "all_active_objectives_causal": bool(selected)
            and all(row["status"] == "causal_pass" for row in selected),
            "controlled_dense_baseline_positive_per_control_step": sum(
                max(0.0, float(row.get("baseline_weighted_per_step", 0.0)))
                for row in selected
                if row["accounting_scope"] == "per_control_step"
            ),
            "controlled_dense_baseline_negative_per_control_step": sum(
                min(0.0, float(row.get("baseline_weighted_per_step", 0.0)))
                for row in selected
                if row["accounting_scope"] == "per_control_step"
            ),
            "controlled_dense_worsening_delta_per_control_step": sum(
                float(row.get("weighted_delta_per_step", 0.0))
                for row in selected
                if row["accounting_scope"] == "per_control_step"
            ),
            "controlled_one_shot_baseline_positive_per_event": sum(
                max(0.0, float(row.get("baseline_weighted_per_step", 0.0)))
                for row in selected
                if row["accounting_scope"] != "per_control_step"
            ),
            "controlled_one_shot_baseline_negative_per_event": sum(
                min(0.0, float(row.get("baseline_weighted_per_step", 0.0)))
                for row in selected
                if row["accounting_scope"] != "per_control_step"
            ),
            "controlled_one_shot_worsening_delta_per_event": sum(
                float(row.get("weighted_delta_per_step", 0.0))
                for row in selected
                if row["accounting_scope"] != "per_control_step"
            ),
        }
    imitation_step = groups["beyondmimic_imitation"][
        "controlled_dense_baseline_positive_per_control_step"
    ]
    landing_event = groups["hope_hit_landing_task"][
        "controlled_one_shot_baseline_positive_per_event"
    ]
    death_event = abs(
        groups["immutable_safety"][
            "controlled_one_shot_worsening_delta_per_event"
        ]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "claim": "live_isaac_one_env_transactional_reward_causality",
        "natural_nonzero_is_not_causal_evidence": True,
        "effective_reward_recipe_sha256": effective_reward_receipt["sha256"],
        "taxonomy_sha256": taxonomy["sha256"],
        "step_dt_s": step_dt_s,
        "active_objective_count": len(active),
        "causal_pass_count": sum(row["status"] == "causal_pass" for row in rows),
        "all_active_objectives_causal": bool(active)
        and all(row["status"] == "causal_pass" for row in rows),
        "groups": groups,
        "dimensioned_ratio_summary": {
            "controlled_imitation_exact_match_income_per_control_step": imitation_step,
            "controlled_landing_legal_center_income_per_strike": landing_event,
            "controlled_death_cost_per_termination": death_event,
            "landing_equals_imitation_control_steps": (
                None if imitation_step <= 0.0 else landing_event / imitation_step
            ),
            "death_to_perfect_landing_abs_ratio": (
                None if landing_event <= 0.0 else death_event / landing_event
            ),
            "warning": (
                "controlled endpoints prove sign and dimensioned scale only; runtime "
                "p50/p95 typical contributions come from the activation ledger after launch"
            ),
        },
        "coverage": rows,
        "candidate_weight_summaries": _candidate_recipes(
            active_with_params, step_dt_s
        ),
        "racket_progress_accounting": _racket_progress_accounting(
            active, rows, step_dt_s=step_dt_s
        ),
        "intervention_contracts": (
            build_action_ball_reward_intervention_contracts()
        ),
        "candidate_policy": (
            "A is the composed baseline. B multiplies only racket position/velocity/normal "
            "weights by four. Both are reported only; this audit never changes training weights."
        ),
        "callable_diagnostic_side_effect_policy": (
            "explicitly controlled Reward-input tensors are transactionally restored; "
            "callable-owned diagnostics/counters may advance, are not accepted as causal "
            "inputs, and the one-env audit is destroyed after receipt creation"
        ),
    }
    report["sha256"] = _sha256_bytes(_canonical_bytes(report))
    return report


def write_no_clobber_receipt(
    output_dir: pathlib.Path,
    report: dict[str, Any],
    *,
    bindings: dict[str, Any],
) -> pathlib.Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "action_ball_reward_causal_prelaunch_receipt",
        "bindings": bindings,
        "report": report,
    }
    receipt["sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    target = output_dir / "receipt.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(target), flags, 0o444)
    try:
        os.write(fd, _canonical_bytes(receipt) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def _git_binding(repo_root: pathlib.Path, producer_path: pathlib.Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root),
        text=True,
    )
    if dirty:
        raise CausalAuditError(
            "formal Reward causal audit requires a clean checkout including untracked files"
        )
    producer_path = producer_path.resolve()
    try:
        producer_relative = producer_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CausalAuditError("Reward causal producer is outside the bound repository") from exc
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", str(producer_relative)],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
        )
        head_bytes = subprocess.check_output(
            ["git", "show", f"HEAD:{producer_relative.as_posix()}"],
            cwd=str(repo_root),
        )
    except subprocess.CalledProcessError as exc:
        raise CausalAuditError(
            "formal Reward causal producer must be tracked by the bound HEAD"
        ) from exc
    working_sha = _sha256_file(producer_path)
    head_sha = _sha256_bytes(head_bytes)
    if working_sha != head_sha:
        raise CausalAuditError(
            "Reward causal producer bytes differ from the bound HEAD"
        )
    return {
        "commit": head,
        "checkout_clean_including_untracked": True,
        "producer_head_blob_sha256": head_sha,
    }


def _hard_exit_after_audit_failure() -> None:
    """Preserve a failing process status even when Isaac ``app.close`` hard-exits."""

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def _run_isaac(cfg: Any) -> pathlib.Path:
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401
    from train import (
        _apply_task_overrides,
        _assert_physical_validity_guards_present,
        _build_effective_reward_receipt_for_training,
        _physical_validity_guards_required,
        _registry_clip_name,
        _validate_action_ball_motion_sources,
        resolve_motion_sources,
    )

    audit_cfg = cfg.get("reward_causal_audit", None)
    if audit_cfg is None or not str(audit_cfg.get("output_dir", "")).strip():
        raise CausalAuditError(
            "set ++reward_causal_audit.output_dir to a new no-clobber directory"
        )
    if int(cfg.num_envs) != 1:
        raise CausalAuditError("set num_envs=1 for the formal Reward causal audit")
    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=1)
    _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    motion_files, _ = resolve_motion_sources(cfg)
    env_cfg.commands.motion.motion_file = (
        motion_files if len(motion_files) > 1 else motion_files[0]
    )
    _validate_action_ball_motion_sources(env_cfg, motion_files)
    racket_cfg = getattr(env_cfg.commands, "racket_target", None)
    if _physical_validity_guards_required(racket_cfg):
        _assert_physical_validity_guards_present(racket_cfg)
    effective = _build_effective_reward_receipt_for_training(
        env_cfg, cfg, require_expected_sha256=True
    )
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    try:
        runtime = env.unwrapped
        # The first reset realizes SceneEntityCfg IDs and manager buffers.
        env.reset(seed=int(cfg.seed))
        from whole_body_tracking.utils.effective_reward_recipe import (
            build_effective_reward_receipt,
        )

        runtime_effective = build_effective_reward_receipt(
            runtime.cfg, expected_sha256=effective["sha256"]
        )
        if runtime_effective != effective:
            raise CausalAuditError(
                "effective Reward changed between post-Hydra pre-gym compose and live runtime"
            )
        step_dt_s = _finite(runtime.step_dt, label="runtime.step_dt")
        report = build_live_causal_report(
            runtime, effective, step_dt_s=step_dt_s
        )
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        racket_cfg = runtime.command_manager.get_term("racket_target").cfg
        manifest_path = pathlib.Path(
            str(getattr(racket_cfg, "action_ball_manifest_path", ""))
        ).resolve()
        configured_manifest_sha = str(
            getattr(racket_cfg, "action_ball_manifest_sha256", "")
        )
        if not manifest_path.is_file():
            raise CausalAuditError(
                f"live ActionBall manifest does not exist: {manifest_path}"
            )
        actual_manifest_sha = _sha256_file(manifest_path)
        if actual_manifest_sha != configured_manifest_sha:
            raise CausalAuditError(
                "live ActionBall manifest file SHA differs from configured SHA"
            )
        from omegaconf import OmegaConf

        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        producer_path = pathlib.Path(__file__).resolve()
        bindings = {
            **_git_binding(repo_root, producer_path),
            "producer_path": str(producer_path.relative_to(repo_root)),
            "producer_sha256": _sha256_file(producer_path),
            "task_id": task_id,
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "num_envs": 1,
            "motion_files": [
                {"path": str(pathlib.Path(path).resolve()), "sha256": _sha256_file(pathlib.Path(path))}
                for path in motion_files
            ],
            "resolved_hydra_cfg_sha256": _sha256_bytes(
                _canonical_bytes(resolved_cfg)
            ),
            "action_ball_identity": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": actual_manifest_sha,
                "configured_manifest_sha256": configured_manifest_sha,
                "policy_contract_sha256": str(
                    getattr(racket_cfg, "action_ball_policy_contract_sha256", "")
                ),
                "clip_names": [
                    str(value)
                    for value in (
                        getattr(racket_cfg, "clip_names_per_clip", ()) or ()
                    )
                ],
                "target_mode": str(getattr(racket_cfg, "target_mode", "")),
            },
            "effective_reward_receipt": effective,
        }
        target = write_no_clobber_receipt(
            pathlib.Path(str(audit_cfg.output_dir)), report, bindings=bindings
        )
        if not report["all_active_objectives_causal"]:
            raise CausalAuditError(
                f"receipt written FAIL_CLOSED at {target}: not every active objective "
                "has a strict negative live weighted delta"
            )
        return target
    finally:
        env.close()


def main() -> None:
    # Hydra and Isaac are intentionally imported only by the executable path.
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../cfg", config_name="train")
    def _hydra_main(cfg: Any) -> None:
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        sys.argv = sys.argv[:1]
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, device=str(cfg.device))
        app = launcher.app
        try:
            target = _run_isaac(cfg)
            print(f"[reward-causal] PASS {target}", flush=True)
        except Exception:
            import traceback

            print("\n[reward-causal] ERROR during audit:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # ``app.close()`` may itself call os._exit(0).  Never enter the
            # successful close path after a failed audit or orchestration can
            # misclassify a real FAIL as exit 0.
            _hard_exit_after_audit_failure()
        app.close()

    _hydra_main()


if __name__ == "__main__":
    main()
