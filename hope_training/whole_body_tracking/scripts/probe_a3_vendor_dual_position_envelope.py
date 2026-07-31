#!/usr/bin/env python3
"""Run the source-bound A3 waist dual-position-envelope stress probe.

This is a simulator-only mechanical test, never a training or deployment
authorization.  Exactly eight environments share four state tapes:

``waist_roll/waist_pitch x lower/upper x H_ctrl ON/OFF``.

For each side, ``R`` is the distance from the live two-percent control edge to
the unchanged mechanical edge.  The initial state is 0.1 R inside H_ctrl and
its exact 5 ms kinematic projection is 0.6 R outside H_ctrl (therefore 0.4 R
inside H_mech).  The ON/OFF pair has byte-identical q0, qdot and q_des; only
the live PhysX limit row differs.  The existing action diagnostic retains its
20 ms ``capture_proxy`` meaning and is recorded as telemetry around the first
tick.  The verdict instead follows the complete four-tick (20 ms) policy
horizon: every tick records q/qdot/q_des, ON remains strictly inside H_ctrl,
both conditions remain strictly inside H_mech, and OFF enters the open
H_ctrl-to-H_mech band on the first tick.  Neither measurement is called a
constraint impulse.

The probe writes one JSON receipt with O_EXCL outside the source and Isaac Lab
trees.  It requires an exact clean source HEAD both before Kit startup and
immediately before publication.  Live limits are restored to all-environment
H_ctrl in ``finally`` and exact readback is mandatory; a restoration failure
can only publish FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
KIND = "a3_vendor_dual_position_envelope_stress_v2"
CONFIRM = "SIM_ONLY_A3_DUAL_POSITION_ENVELOPE_8ENV_FOUR_TICKS"
EXACT_TASK = "HOPE-PingPong-ActionBall-AgibotA3-v0"
VENDOR_TASK_PROFILE = "HOPEPingPongActionBallA3VendorV1"
EXACT_NUM_ENVS = 8
EXACT_PHYSICS_DT_S = 0.005
POLICY_HORIZON_PHYSICS_TICKS = 4
POLICY_HORIZON_S = EXACT_PHYSICS_DT_S * POLICY_HORIZON_PHYSICS_TICKS
CONTROL_INSET_FRACTION = 0.02
VENDOR_CONTROL_STEP_ACTION_DELAY = (0, 2)
VENDOR_GUARD_BRAKE_MODE = "max_inward_until_nonoutward_v1"
VENDOR_GUARD_MARGIN_FRACTION = 0.06
DIAGNOSTIC_POLICY_CONTRACT_SHA256 = "0" * 64
ACTOR_OBS_CONTRACT = (
    "action_ball_table_pose_twist_heading_task_teacher_start_v2"
)
WBT_RELATIVE = Path("hope_training/whole_body_tracking")
VENDOR_TASK_SOURCE = (
    WBT_RELATIVE / "cfg/task/HOPEPingPongActionBallA3VendorV1.yaml"
)
TRAIN_SOURCE = WBT_RELATIVE / "scripts/train.py"
ACTION_REGISTRY_SOURCE = WBT_RELATIVE / "scripts/a3_vendor_action_registry.py"
Q0_INNER_CAGE_FRACTION = 0.1
KINEMATIC_OUTER_CAGE_FRACTION = 0.6
MECHANICAL_REMAINING_CAGE_FRACTION = 0.4
STRESSED_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
SIDES = ("lower", "upper")
CONDITIONS = ("on", "off")
STAGE_MARKERS = (
    "vendor_profile_bind_begin",
    "vendor_profile_bind_done",
    "gym_make_begin",
    "gym_make_done",
    "reset_begin",
    "reset_done",
    "hctrl_readback_begin",
    "hctrl_readback_done",
    "mixed_limit_readback_begin",
    "mixed_limit_readback_done",
    "sim_step_begin",
    "sim_step_done",
)


class DualEnvelopeProbeError(RuntimeError):
    """The strict dual-envelope probe contract was not satisfied."""


def _emit_stage_marker(stage: str) -> None:
    """Print one flush-safe marker whose vocabulary and order are code-owned."""

    if stage not in STAGE_MARKERS:
        raise DualEnvelopeProbeError(f"unknown live-probe stage marker: {stage!r}")
    print(f"A3_DUAL_POSITION_ENVELOPE_STAGE={stage}", flush=True)


def _node_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    try:
        return getattr(node, key)
    except AttributeError:
        try:
            return node.get(key, default)
        except (AttributeError, TypeError):
            return default


def _load_exact_source_module(path: Path, module_name: str):
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DualEnvelopeProbeError(f"cannot load exact source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if Path(module.__file__).resolve(strict=True) != path:
        raise DualEnvelopeProbeError(f"source module resolved to wrong bytes: {path}")
    return module


def _compose_vendor_task_profile(source_root: Path):
    """Compose the exact Hydra leaf used by long training, without copying YAML."""

    import hydra
    from omegaconf import OmegaConf

    cfg_dir = (source_root / WBT_RELATIVE / "cfg").resolve(strict=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = hydra.compose(
            config_name="train",
            overrides=[f"task={VENDOR_TASK_PROFILE}"],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg.task, False)
    _validate_vendor_profile_binding(cfg.task)
    return cfg.task


def _validate_vendor_profile_binding(task: Any, env_cfg: Any | None = None) -> None:
    """Fail closed if the composed leaf or translated action contract drifts."""

    if str(_node_get(task, "name", "")) != VENDOR_TASK_PROFILE:
        raise DualEnvelopeProbeError(
            "Hydra did not compose the exact VendorV1 task profile"
        )
    if str(_node_get(task, "gym_task", "")) != EXACT_TASK:
        raise DualEnvelopeProbeError("VendorV1 gym task binding drifted")
    actions = _node_get(task, "actions")
    expected = {
        "control_step_action_delay_min": VENDOR_CONTROL_STEP_ACTION_DELAY[0],
        "control_step_action_delay_max": VENDOR_CONTROL_STEP_ACTION_DELAY[1],
        "pre_apply_guard_brake_mode": VENDOR_GUARD_BRAKE_MODE,
        "pre_apply_guard_margin_fraction": VENDOR_GUARD_MARGIN_FRACTION,
        "physx_control_position_limit_inset_fraction": CONTROL_INSET_FRACTION,
    }
    for name, value in expected.items():
        actual = _node_get(actions, name)
        if actual != value:
            raise DualEnvelopeProbeError(
                f"VendorV1 task.actions.{name} drifted: "
                f"expected={value!r} actual={actual!r}"
            )
    if env_cfg is None:
        return
    joint_action_cfg = _node_get(_node_get(env_cfg, "actions"), "joint_pos")
    if joint_action_cfg is None:
        raise DualEnvelopeProbeError("translated VendorV1 actions.joint_pos is absent")
    for name, value in expected.items():
        actual = _node_get(joint_action_cfg, name)
        if actual != value:
            raise DualEnvelopeProbeError(
                f"translated VendorV1 actions.joint_pos.{name} drifted: "
                f"expected={value!r} actual={actual!r}"
            )


def _resolve_vendor_action_binding(
    source_root: Path,
    motion_files: Sequence[Path],
) -> dict[str, str]:
    """Resolve the N=1 diagnostic identity from the code-owned action registry."""

    if len(motion_files) != 1:
        raise DualEnvelopeProbeError(
            "VendorV1 dual-envelope stress requires exactly one N=1 motion"
        )
    registry = _load_exact_source_module(
        source_root / ACTION_REGISTRY_SOURCE,
        "_a3_dual_envelope_action_registry",
    )
    motion = motion_files[0].resolve(strict=True)
    matches = []
    for action_id, config in registry.ACTION_CONFIGS.items():
        expected_motion = (source_root / config.stable_motion.path).resolve(
            strict=True
        )
        if motion == expected_motion:
            matches.append((str(action_id), config))
    if len(matches) != 1:
        raise DualEnvelopeProbeError(
            "motion must match exactly one code-owned A3 vendor N=1 action"
        )
    action_id, config = matches[0]
    stable_pin = registry.stable_pin(config.stable_motion)
    if _sha256_file(motion) != stable_pin["sha256"]:
        raise DualEnvelopeProbeError("code-owned A3 vendor motion bytes drifted")
    manifest_pin = registry.require_materialized_pin(
        config.identity_manifest,
        action_id=action_id,
        layer="identity manifest",
    )
    manifest = (source_root / manifest_pin["path"]).resolve(strict=True)
    if _sha256_file(manifest) != manifest_pin["sha256"]:
        raise DualEnvelopeProbeError("code-owned A3 vendor identity manifest bytes drifted")
    return {
        "action_id": action_id,
        "motion_path": str(motion),
        "manifest_path": str(manifest),
        "manifest_sha256": str(manifest_pin["sha256"]),
    }


def _bind_diagnostic_identity(task: Any, binding: Mapping[str, str]) -> None:
    """Fill launcher-owned N=1 sentinels; VendorV1 scientific values stay untouched."""

    task.actor_obs_contract = ACTOR_OBS_CONTRACT
    task.racket.clip_names = [binding["action_id"]]
    task.racket.action_ball_manifest_path = binding["manifest_path"]
    task.racket.action_ball_manifest_sha256 = binding["manifest_sha256"]
    task.racket.action_ball_policy_contract_sha256 = (
        DIAGNOSTIC_POLICY_CONTRACT_SHA256
    )
    task.racket.action_ball_diagnostic_unauthorized = True
    task.racket.reference_guard_mode = "metrics_only"


def _materialize_vendor_env_cfg(args: argparse.Namespace):
    """Use the same Hydra leaf and translator as train.py before Gym construction."""

    from isaaclab_tasks.utils import parse_env_cfg

    source_root = args.source_root.resolve(strict=True)
    task = _compose_vendor_task_profile(source_root)
    binding = _resolve_vendor_action_binding(source_root, args.motion_file)
    _bind_diagnostic_identity(task, binding)
    env_cfg = parse_env_cfg(
        str(task.gym_task),
        device=str(args.device),
        num_envs=EXACT_NUM_ENVS,
    )
    env_cfg.commands.motion.motion_file = binding["motion_path"]
    train = _load_exact_source_module(
        source_root / TRAIN_SOURCE,
        "_a3_dual_envelope_train_assembler",
    )
    applied = train._apply_task_overrides(
        env_cfg,
        task,
        clip_name=binding["action_id"],
    )
    _validate_vendor_profile_binding(task, env_cfg)
    task_profile_source = (source_root / VENDOR_TASK_SOURCE).resolve(strict=True)
    train_source = (source_root / TRAIN_SOURCE).resolve(strict=True)
    action_registry_source = (source_root / ACTION_REGISTRY_SOURCE).resolve(
        strict=True
    )
    return env_cfg, {
        **binding,
        "task_profile": VENDOR_TASK_PROFILE,
        "task_profile_source": str(task_profile_source),
        "task_profile_source_sha256": _sha256_file(task_profile_source),
        "training_assembler_source": str(train_source),
        "training_assembler_source_sha256": _sha256_file(train_source),
        "action_registry_source": str(action_registry_source),
        "action_registry_source_sha256": _sha256_file(action_registry_source),
        "gym_task": str(task.gym_task),
        "applied_task_override_count": len(applied),
        "control_step_action_delay": list(VENDOR_CONTROL_STEP_ACTION_DELAY),
        "physx_control_position_limit_inset_fraction": CONTROL_INSET_FRACTION,
        "pre_apply_guard_brake_mode": VENDOR_GUARD_BRAKE_MODE,
        "pre_apply_guard_margin_fraction": VENDOR_GUARD_MARGIN_FRACTION,
    }


def _disable_debug_visualization(env_cfg: Any) -> list[str]:
    """Disable only operational debug drawing; no scientific axis changes."""

    targets = (
        (
            "commands.motion.debug_vis",
            _node_get(_node_get(env_cfg, "commands"), "motion"),
        ),
        (
            "scene.contact_forces.debug_vis",
            _node_get(_node_get(env_cfg, "scene"), "contact_forces"),
        ),
    )
    disabled = []
    for label, node in targets:
        if node is None or not hasattr(node, "debug_vis"):
            raise DualEnvelopeProbeError(
                f"VendorV1 debug visualization surface is absent: {label}"
            )
        node.debug_vis = False
        if node.debug_vis is not False:
            raise DualEnvelopeProbeError(f"failed to disable {label}")
        disabled.append(label)
    return disabled


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_clean_exact_checkout(
    root: Path,
    expected_commit: str,
    *,
    script_path: Path | None = None,
) -> str:
    root = root.resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise DualEnvelopeProbeError(f"source root is not exact Git top-level: {top}")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_commit or len(head) != 40:
        raise DualEnvelopeProbeError(
            f"source HEAD mismatch: expected={expected_commit!r} actual={head!r}"
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise DualEnvelopeProbeError("source checkout must be exactly clean")
    if script_path is not None:
        resolved = script_path.resolve(strict=True)
        if not _inside(resolved, root):
            raise DualEnvelopeProbeError("probe script is outside the reviewed source root")
        relative = resolved.relative_to(root).as_posix()
        tracked = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if tracked != resolved.read_bytes():
            raise DualEnvelopeProbeError("probe script bytes differ from exact source HEAD")
    return head


def _validate_output_path(output: Path, forbidden_roots: Sequence[Path]) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        raise DualEnvelopeProbeError("--output must be an absolute JSON path")
    parent = output.parent.resolve(strict=True)
    resolved = parent / output.name
    if resolved.suffix != ".json":
        raise DualEnvelopeProbeError("--output must end in .json")
    for raw_root in forbidden_roots:
        root = raw_root.resolve(strict=True)
        if _inside(resolved, root):
            raise DualEnvelopeProbeError(
                f"output must remain outside protected root: {root}"
            )
    if resolved.exists() or resolved.is_symlink():
        raise DualEnvelopeProbeError(f"no-clobber output already exists: {resolved}")
    return resolved


def _installed_isaaclab_root() -> Path:
    """Resolve the editable/install tree that the live probe will import."""

    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise DualEnvelopeProbeError("the selected Python cannot resolve isaaclab")
    package_file = Path(spec.origin).resolve(strict=True)
    try:
        return Path(_git(package_file.parent, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except (OSError, subprocess.CalledProcessError):
        return package_file.parent


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DualEnvelopeProbeError(f"{label} must be one finite number")
    return float(value)


def _float32_round(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def build_stress_tape(
    joint_names: Sequence[str],
    mechanical_limits: Sequence[Sequence[float]],
    control_limits: Sequence[Sequence[float]],
    *,
    physics_dt_s: float = EXACT_PHYSICS_DT_S,
) -> list[dict[str, Any]]:
    """Build the exact 8-row same-state ON/OFF tape without simulator imports."""

    if type(physics_dt_s) is not float or physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("stress tape requires exact physics_dt_s=0.005")
    names = tuple(str(name) for name in joint_names)
    if len(names) != len(set(names)) or any(names.count(name) != 1 for name in STRESSED_JOINTS):
        raise DualEnvelopeProbeError("joint order must contain each stressed waist exactly once")
    if len(mechanical_limits) != len(names) or len(control_limits) != len(names):
        raise DualEnvelopeProbeError("limit row count must equal joint-name count")

    rows: list[dict[str, Any]] = []
    for joint_name in STRESSED_JOINTS:
        joint_index = names.index(joint_name)
        mechanical = tuple(
            _finite_number(value, f"H_mech[{joint_name}]")
            for value in mechanical_limits[joint_index]
        )
        control = tuple(
            _finite_number(value, f"H_ctrl[{joint_name}]")
            for value in control_limits[joint_index]
        )
        if len(mechanical) != 2 or len(control) != 2:
            raise DualEnvelopeProbeError("every limit row must be [lower, upper]")
        hard_lower, hard_upper = mechanical
        ctrl_lower, ctrl_upper = control
        span = hard_upper - hard_lower
        if not span > 0.0:
            raise DualEnvelopeProbeError(f"{joint_name} has non-positive H_mech span")
        lower_reserve = ctrl_lower - hard_lower
        upper_reserve = hard_upper - ctrl_upper
        expected_reserve = CONTROL_INSET_FRACTION * span
        if not math.isclose(lower_reserve, expected_reserve, rel_tol=0.0, abs_tol=1.0e-7):
            raise DualEnvelopeProbeError(f"{joint_name} lower H_ctrl reserve is not exact 2%")
        if not math.isclose(upper_reserve, expected_reserve, rel_tol=0.0, abs_tol=1.0e-7):
            raise DualEnvelopeProbeError(f"{joint_name} upper H_ctrl reserve is not exact 2%")

        for side in SIDES:
            direction = -1.0 if side == "lower" else 1.0
            mechanical_edge = hard_lower if side == "lower" else hard_upper
            control_edge = ctrl_lower if side == "lower" else ctrl_upper
            reserve = lower_reserve if side == "lower" else upper_reserve
            q0 = control_edge - direction * Q0_INNER_CAGE_FRACTION * reserve
            kinematic_q_5ms = (
                control_edge
                + direction * KINEMATIC_OUTER_CAGE_FRACTION * reserve
            )
            qdot = (kinematic_q_5ms - q0) / physics_dt_s
            if not math.isclose(
                abs(qdot),
                (Q0_INNER_CAGE_FRACTION + KINEMATIC_OUTER_CAGE_FRACTION)
                * reserve
                / physics_dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AssertionError("internal qdot reserve arithmetic drifted")
            mechanical_gap = direction * (mechanical_edge - kinematic_q_5ms)
            if not math.isclose(
                mechanical_gap,
                MECHANICAL_REMAINING_CAGE_FRACTION * reserve,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AssertionError("internal H_mech remaining-reserve arithmetic drifted")
            for condition in CONDITIONS:
                rows.append(
                    {
                        "env_id": len(rows),
                        "joint": joint_name,
                        "joint_index": joint_index,
                        "side": side,
                        "condition": condition,
                        "direction": int(direction),
                        "h_mech_edge_rad": mechanical_edge,
                        "h_ctrl_edge_rad": control_edge,
                        "cage_reserve_rad": reserve,
                        "q0_rad": q0,
                        "qdot0_rad_s": qdot,
                        "qdes_rad": q0,
                        "kinematic_q_5ms_rad": kinematic_q_5ms,
                        "kinematic_crosses_h_ctrl": True,
                        "kinematic_mechanical_gap_rad": mechanical_gap,
                    }
                )
    validate_stress_tape(rows)
    return rows


def validate_stress_tape(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError("stress tape must contain exactly 8 rows")
    expected = [
        (joint, side, condition)
        for joint in STRESSED_JOINTS
        for side in SIDES
        for condition in CONDITIONS
    ]
    observed = []
    for env_id, row in enumerate(rows):
        if row.get("env_id") != env_id:
            raise DualEnvelopeProbeError("stress tape env ids are not exact identity order")
        observed.append((row.get("joint"), row.get("side"), row.get("condition")))
        for key in (
            "h_mech_edge_rad",
            "h_ctrl_edge_rad",
            "cage_reserve_rad",
            "q0_rad",
            "qdot0_rad_s",
            "qdes_rad",
            "kinematic_q_5ms_rad",
            "kinematic_mechanical_gap_rad",
        ):
            _finite_number(row.get(key), f"stress_tape[{env_id}].{key}")
        if row.get("kinematic_crosses_h_ctrl") is not True:
            raise DualEnvelopeProbeError("every tape row must cross H_ctrl kinematically in 5 ms")
        if row["qdes_rad"] != row["q0_rad"]:
            raise DualEnvelopeProbeError("q_des must equal q0 exactly")
    if observed != expected:
        raise DualEnvelopeProbeError("stress tape joint/side/condition order drifted")
    for pair_start in range(0, EXACT_NUM_ENVS, 2):
        on = dict(rows[pair_start])
        off = dict(rows[pair_start + 1])
        for key in ("env_id", "condition"):
            on.pop(key)
            off.pop(key)
        if on != off:
            raise DualEnvelopeProbeError("ON/OFF pair does not share the exact same state tape")


def _side_gaps(row: Mapping[str, Any], q_after: float) -> tuple[float, float]:
    direction = int(row["direction"])
    control_gap = -direction * (q_after - float(row["h_ctrl_edge_rad"]))
    mechanical_gap = -direction * (q_after - float(row["h_mech_edge_rad"]))
    return control_gap, mechanical_gap


def validate_runtime_result(
    tape: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    *,
    physics_dt_s: float,
    live_limits_restored_exact: bool,
) -> dict[str, Any]:
    """Validate the four-tick policy-horizon trajectory and receipt aggregates.

    The old 20 ms ballistic proxy remains verbatim telemetry.  It is not the
    verdict because an ON row may still have a small outward velocity after
    tick one while its actual trajectory remains safely captured.  No
    tolerance weakens either position envelope.
    """

    validate_stress_tape(tape)
    if physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("live physics dt is not exact 0.005 s")
    if live_limits_restored_exact is not True:
        raise DualEnvelopeProbeError("all-environment H_ctrl restoration/readback failed")
    if len(observations) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError("runtime must return exactly 8 observation rows")

    normalized: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], dict[str, int]] = {
        (joint, side): {
            "strict_5ms_kinematic_attempt_count": 0,
            "trajectory_tick_count": 0,
            "on_strict_hctrl_tick_count": 0,
            "on_strict_hmech_tick_count": 0,
            "off_strict_hmech_tick_count": 0,
            "off_first_tick_ctrl_band_entry_count": 0,
            "qdes_equal_q0_exact_tick_count": 0,
            "mechanical_penetration_count": 0,
        }
        for joint in STRESSED_JOINTS
        for side in SIDES
    }
    for env_id, (tape_row, raw) in enumerate(zip(tape, observations, strict=True)):
        if raw.get("env_id") != env_id:
            raise DualEnvelopeProbeError("runtime observation env ids drifted")
        if raw.get("joint") != tape_row["joint"] or raw.get("side") != tape_row["side"]:
            raise DualEnvelopeProbeError("runtime observation identity differs from tape")
        if raw.get("condition") != tape_row["condition"]:
            raise DualEnvelopeProbeError("runtime condition differs from tape")
        q0_live = _finite_number(
            raw.get("q0_live_rad"), f"observation[{env_id}].q0_live"
        )
        if q0_live != _float32_round(float(tape_row["q0_rad"])):
            raise DualEnvelopeProbeError("live q0 differs from the float32 exact stress tape")
        trajectory = raw.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != POLICY_HORIZON_PHYSICS_TICKS:
            raise DualEnvelopeProbeError(
                "runtime row must contain exactly four ordered physics ticks"
            )
        condition = str(tape_row["condition"])
        key = (str(tape_row["joint"]), str(tape_row["side"]))
        pair_counts[key]["strict_5ms_kinematic_attempt_count"] += 1
        normalized_ticks = []
        for offset, sample in enumerate(trajectory):
            tick_index = offset + 1
            if not isinstance(sample, Mapping) or sample.get("tick_index") != tick_index:
                raise DualEnvelopeProbeError("physics tick identity/order drifted")
            expected_elapsed = tick_index * EXACT_PHYSICS_DT_S
            elapsed = _finite_number(
                sample.get("elapsed_s"),
                f"observation[{env_id}].trajectory[{offset}].elapsed_s",
            )
            if elapsed != expected_elapsed:
                raise DualEnvelopeProbeError("physics tick elapsed time drifted")
            q = _finite_number(
                sample.get("q_rad"),
                f"observation[{env_id}].trajectory[{offset}].q",
            )
            qdot = _finite_number(
                sample.get("qdot_rad_s"),
                f"observation[{env_id}].trajectory[{offset}].qdot",
            )
            qdes = _finite_number(
                sample.get("qdes_rad"),
                f"observation[{env_id}].trajectory[{offset}].qdes",
            )
            if qdes != q0_live:
                raise DualEnvelopeProbeError("live q_des differs from exact live q0")
            control_gap, mechanical_gap = _side_gaps(tape_row, q)
            if not mechanical_gap > 0.0:
                raise DualEnvelopeProbeError("stress trajectory touched/crossed H_mech")
            if condition == "on" and not control_gap > 0.0:
                raise DualEnvelopeProbeError(
                    "H_ctrl ON left strict H_ctrl during policy horizon"
                )
            if condition == "off" and tick_index == 1 and not control_gap <= 0.0:
                raise DualEnvelopeProbeError(
                    "H_ctrl OFF did not enter [H_ctrl,H_mech) on first tick"
                )
            local = pair_counts[key]
            local["trajectory_tick_count"] += 1
            local["qdes_equal_q0_exact_tick_count"] += 1
            if condition == "on":
                local["on_strict_hctrl_tick_count"] += 1
                local["on_strict_hmech_tick_count"] += 1
            else:
                local["off_strict_hmech_tick_count"] += 1
                if tick_index == 1:
                    local["off_first_tick_ctrl_band_entry_count"] += 1
            normalized_ticks.append(
                {
                    **dict(sample),
                    "elapsed_s": elapsed,
                    "q_rad": q,
                    "qdot_rad_s": qdot,
                    "qdes_rad": qdes,
                    "signed_ctrl_gap_rad": control_gap,
                    "signed_mechanical_gap_rad": mechanical_gap,
                }
            )
        normalized.append(
            {
                **dict(raw),
                "q0_live_rad": q0_live,
                "trajectory": normalized_ticks,
                "strict_5ms_kinematic_crossing": True,
            }
        )

    phase_rows: dict[str, dict[object, object]] = {}
    for phase in ("pre_step", "post_step"):
        phase_payload = diagnostic.get(phase)
        if not isinstance(phase_payload, Mapping):
            raise DualEnvelopeProbeError(f"verified {phase} H_ctrl diagnostic is absent")
        diag = phase_payload.get("physx_control_position_limits")
        if not isinstance(diag, Mapping) or diag.get("enabled") is not True:
            raise DualEnvelopeProbeError(f"verified {phase} 20 ms H_ctrl diagnostic is absent")
        if diag.get("semantics") != (
            "kinematic H_ctrl proxy; not a PhysX constraint impulse getter"
        ):
            raise DualEnvelopeProbeError(f"{phase} diagnostic proxy semantics drifted")
        if float(diag.get("ballistic_horizon_s", -1.0)) != 0.02:
            raise DualEnvelopeProbeError(
                f"{phase} capture_proxy horizon must remain 20 ms"
            )
        by_joint = diag.get("by_joint")
        if not isinstance(by_joint, list):
            raise DualEnvelopeProbeError(f"{phase} diagnostic by_joint rows are absent")
        phase_rows[phase] = {
            row.get("joint"): row.get("sides")
            for row in by_joint
            if isinstance(row, Mapping)
        }
    aggregate_rows = []
    for joint in STRESSED_JOINTS:
        for side in SIDES:
            values_by_phase: dict[str, Mapping[str, Any]] = {}
            for phase in ("pre_step", "post_step"):
                sides = phase_rows[phase].get(joint)
                if not isinstance(sides, Mapping):
                    raise DualEnvelopeProbeError(
                        f"{phase} diagnostic is missing {joint}"
                    )
                values = sides.get(side)
                if not isinstance(values, Mapping):
                    raise DualEnvelopeProbeError(
                        f"{phase} diagnostic is missing {joint}/{side}"
                    )
                values_by_phase[phase] = values
            pre = values_by_phase["pre_step"]
            post = values_by_phase["post_step"]
            diagnostic_keys = (
                "ballistic_attempt_proxy",
                "capture_proxy",
                "ctrl_penetration_readback",
            )
            pre_tuple = tuple(pre.get(name) for name in diagnostic_keys)
            post_tuple = tuple(post.get(name) for name in diagnostic_keys)
            if any(type(value) is not int or not 0 <= value <= 2 for value in (*pre_tuple, *post_tuple)):
                raise DualEnvelopeProbeError(
                    f"{joint}/{side} 20 ms diagnostic counters are malformed"
                )
            if pre_tuple != (2, 0, 0):
                raise DualEnvelopeProbeError(
                    f"{joint}/{side} expected pre=2/0/0, "
                    f"got pre={pre_tuple[0]}/{pre_tuple[1]}/{pre_tuple[2]}"
                )
            if post_tuple[2] != 1:
                raise DualEnvelopeProbeError(
                    f"{joint}/{side} first-tick diagnostic must observe one OFF penetration"
                )
            local = pair_counts[(joint, side)]
            if local != {
                "strict_5ms_kinematic_attempt_count": 2,
                "trajectory_tick_count": 8,
                "on_strict_hctrl_tick_count": 4,
                "on_strict_hmech_tick_count": 4,
                "off_strict_hmech_tick_count": 4,
                "off_first_tick_ctrl_band_entry_count": 1,
                "qdes_equal_q0_exact_tick_count": 8,
                "mechanical_penetration_count": 0,
            }:
                raise AssertionError("internal per-pair aggregate drifted")
            aggregate_rows.append(
                {
                    "joint": joint,
                    "side": side,
                    **local,
                    "existing_20ms_ballistic_attempt_proxy_count": pre_tuple[0],
                    "post_20ms_ballistic_attempt_proxy_count": post_tuple[0],
                    "existing_20ms_capture_proxy_count": post_tuple[1],
                    "post_ctrl_penetration_readback_count": post_tuple[2],
                }
            )

    return {
        "observations": normalized,
        "aggregate_by_joint_side": aggregate_rows,
        "mechanical_penetration_count": 0,
        "all_rows_finite": True,
        "all_qdes_equal_q0_exact": True,
        "all_live_limits_restored_to_hctrl_exact": True,
        "policy_horizon_physics_ticks": POLICY_HORIZON_PHYSICS_TICKS,
        "policy_horizon_s": POLICY_HORIZON_S,
        "existing_diagnostic_verdict_role": "telemetry_only",
        "existing_diagnostic": dict(diagnostic),
    }


def build_receipt(
    *,
    source_commit: str,
    source_script_sha256: str,
    task: str,
    motion_files: Sequence[Mapping[str, str]],
    tape: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any] | None,
    live_limit_identity: Mapping[str, Any] | None,
    restore: Mapping[str, Any],
    error: str | None,
    failure_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    restore_exact = restore.get("attempted") is True and restore.get("exact_readback") is True
    passed = error is None and runtime is not None and restore_exact
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "PASS" if passed else "FAIL",
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "source_commit": source_commit,
        "producer": {
            "path": "hope_training/whole_body_tracking/scripts/"
            "probe_a3_vendor_dual_position_envelope.py",
            "sha256": source_script_sha256,
        },
        "task": task,
        "motion_files": list(motion_files),
        "contract": {
            "num_envs": EXACT_NUM_ENVS,
            "physics_ticks": POLICY_HORIZON_PHYSICS_TICKS,
            "physics_dt_s": EXACT_PHYSICS_DT_S,
            "policy_horizon_s": POLICY_HORIZON_S,
            "stressed_joints": list(STRESSED_JOINTS),
            "sides": list(SIDES),
            "conditions": list(CONDITIONS),
            "q0_inner_cage_fraction": Q0_INNER_CAGE_FRACTION,
            "kinematic_outer_cage_fraction": KINEMATIC_OUTER_CAGE_FRACTION,
            "kinematic_remaining_mechanical_cage_fraction": (
                MECHANICAL_REMAINING_CAGE_FRACTION
            ),
            "qdot_formula": "direction*(0.1+0.6)*R/0.005",
            "qdes_contract": "qdes=q0 exact",
            "existing_capture_proxy_horizon_s": 0.02,
            "existing_capture_proxy_verdict_role": "telemetry_only",
            "existing_capture_proxy_sampling": "pre_tick_1_and_post_tick_1",
            "strict_kinematic_crossing_horizon_s": 0.005,
            "trajectory_contract": (
                "each tick finite q/qdot/qdes; ON strict H_ctrl; ON/OFF strict "
                "H_mech; OFF first tick in [H_ctrl,H_mech); qdes=q0 exact"
            ),
            "measurement_semantics": (
                "kinematic positions/velocities and H_ctrl proxies; no constraint impulse getter"
            ),
            "disabled_randomization": [
                "push",
                "pd_gain",
                "mass",
                "base_com",
                "physics_material",
                "joint_default_pos",
            ],
        },
        "stress_tape": [dict(row) for row in tape],
        "live_limit_identity": None if live_limit_identity is None else dict(live_limit_identity),
        "runtime": None if runtime is None else dict(runtime),
        "failure_evidence": (
            None if passed or failure_evidence is None else dict(failure_evidence)
        ),
        "restore": dict(restore),
        "error": error,
    }
    return {**content, "content_sha256": _sha256_bytes(_canonical_json_bytes(content))}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(task=EXACT_TASK)
    parser.add_argument("--motion-file", action="append", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args(argv)


def _disable_randomization(env_cfg: Any) -> list[str]:
    events = getattr(env_cfg, "events", None)
    if events is None:
        raise DualEnvelopeProbeError("task exposes no EventCfg")
    names = (
        "push_robot",
        "force_push",
        "force_push_sweep",
        "combined_push",
        "combined_push_sweep",
        "randomize_pd_gains",
        "randomize_link_mass",
        "base_com",
        "physics_material",
        "add_joint_default_pos",
    )
    disabled = []
    for name in names:
        if hasattr(events, name):
            setattr(events, name, None)
            disabled.append(name)
    required = {
        "push_robot",
        "randomize_pd_gains",
        "randomize_link_mass",
        "base_com",
        "physics_material",
        "add_joint_default_pos",
    }
    if not required.issubset(disabled):
        raise DualEnvelopeProbeError(
            f"task is missing required randomization switches: {sorted(required - set(disabled))}"
        )
    return disabled


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run_live(
    args: argparse.Namespace,
    *,
    resource_sink: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Launch Kit, run four physics ticks, and restore H_ctrl without closing Kit."""

    # Isaac imports belong after CLI/source validation so host tests can import this module.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=bool(args.headless),
        device=str(args.device),
        enable_cameras=False,
    )
    simulation_app = launcher.app
    if resource_sink is not None:
        resource_sink["simulation_app"] = simulation_app
    env = None
    action_term = None
    robot = None
    all_hctrl_cpu = None
    restore = {"attempted": False, "exact_readback": False, "error": None}
    tape: list[dict[str, Any]] = []
    live_identity: dict[str, Any] = {}
    if resource_sink is not None:
        resource_sink.update(
            {
                "restore": restore,
                "tape": tape,
                "identity": live_identity,
                "failure_evidence": None,
            }
        )
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        import whole_body_tracking.tasks  # noqa: F401

        _emit_stage_marker("vendor_profile_bind_begin")
        env_cfg, vendor_binding = _materialize_vendor_env_cfg(args)
        _emit_stage_marker("vendor_profile_bind_done")
        if float(env_cfg.sim.dt) != EXACT_PHYSICS_DT_S or int(env_cfg.decimation) != 4:
            raise DualEnvelopeProbeError("task must retain dt=0.005 and decimation=4")
        disabled_debug_visualization = _disable_debug_visualization(env_cfg)
        disabled_events = _disable_randomization(env_cfg)

        _emit_stage_marker("gym_make_begin")
        env = gym.make(vendor_binding["gym_task"], cfg=env_cfg, render_mode=None)
        _emit_stage_marker("gym_make_done")
        base = env.unwrapped
        _emit_stage_marker("reset_begin")
        reset = env.reset()
        _emit_stage_marker("reset_done")
        if not isinstance(reset, tuple) or len(reset) != 2:
            raise DualEnvelopeProbeError("Gym reset did not return (observation, info)")
        if int(base.num_envs) != EXACT_NUM_ENVS:
            raise DualEnvelopeProbeError("live task did not construct exactly 8 envs")
        if float(base.physics_dt) != EXACT_PHYSICS_DT_S:
            raise DualEnvelopeProbeError("live physics_dt is not exact 0.005")
        action_term = base.action_manager.get_term("joint_pos")
        robot = base.scene["robot"]
        _emit_stage_marker("hctrl_readback_begin")
        action_term.verify_physx_control_position_limit_readback()
        _emit_stage_marker("hctrl_readback_done")
        # Reset may legitimately exercise the normal update diagnostic before
        # this four-tick tape starts.  Clear both its published counters and its
        # temporal proxy history so the receipt contains only these 8 envs.
        action_term.consume_actual_joint_forbidden_diagnostic()
        for name in (
            "_physx_control_diagnostic_previous_attempt",
            "_physx_control_diagnostic_current_penetration_dwell",
            "_physx_control_diagnostic_previous_qdot",
            "_physx_control_diagnostic_previous_qdot_valid",
        ):
            tensor = getattr(action_term, name, None)
            if tensor is None:
                raise DualEnvelopeProbeError(
                    f"H_ctrl diagnostic history buffer disappeared: {name}"
                )
            tensor.zero_()
        mechanical = robot.data.joint_pos_limits.detach().cpu()
        all_hctrl_cpu = action_term._physx_control_joint_pos_limits_snapshot.detach().clone()
        names = tuple(str(name) for name in robot.joint_names)
        tape.extend(
            build_stress_tape(
                names,
                mechanical[0].tolist(),
                all_hctrl_cpu[0].tolist(),
            )
        )

        mixed = all_hctrl_cpu.clone()
        for row in tape:
            if row["condition"] == "off":
                mixed[row["env_id"]] = mechanical[row["env_id"]]
        indices = robot._ALL_INDICES.detach().cpu()
        root_view = robot.root_physx_view
        _emit_stage_marker("mixed_limit_readback_begin")
        root_view.set_dof_limits(mixed, indices=indices)
        mixed_readback = root_view.get_dof_limits()
        if not torch.equal(mixed_readback, mixed):
            raise DualEnvelopeProbeError("mixed H_ctrl ON/OFF live-limit readback is not exact")
        _emit_stage_marker("mixed_limit_readback_done")

        q0 = robot.data.default_joint_pos.detach().clone()
        qdot0 = torch.zeros_like(q0)
        if q0.dtype != torch.float32:
            raise DualEnvelopeProbeError("stress probe requires the shipped float32 articulation")
        for row in tape:
            q0[row["env_id"], row["joint_index"]] = row["q0_rad"]
            qdot0[row["env_id"], row["joint_index"]] = row["qdot0_rad_s"]
        if not torch.all(torch.isfinite(q0)) or not torch.all(torch.isfinite(qdot0)):
            raise DualEnvelopeProbeError("constructed q0/qdot tape is non-finite")
        robot.write_joint_state_to_sim(q0, qdot0)
        robot.set_joint_position_target(q0)
        base.scene.write_data_to_sim()

        action_term._record_physx_control_position_limit_diagnostic(
            joint_pos=q0,
            joint_vel=qdot0,
        )
        pre_diagnostic = action_term.consume_actual_joint_forbidden_diagnostic()
        observations = [
            {
                "env_id": row["env_id"],
                "joint": row["joint"],
                "side": row["side"],
                "condition": row["condition"],
                "q0_live_rad": float(
                    q0[row["env_id"], row["joint_index"]].detach().cpu()
                ),
                "trajectory": [],
            }
            for row in tape
        ]
        pending = {
            "observations": observations,
            "diagnostic": {
                "pre_step": pre_diagnostic,
                "post_step": None,
            },
            "physics_dt_s": float(base.physics_dt),
        }
        if resource_sink is not None:
            # This object is intentionally updated in place after every tick so
            # a mid-horizon exception still publishes the completed prefix.
            resource_sink["failure_evidence"] = pending

        _emit_stage_marker("sim_step_begin")
        for tick_index in range(1, POLICY_HORIZON_PHYSICS_TICKS + 1):
            base.sim.step(render=False)
            base.scene.update(EXACT_PHYSICS_DT_S)
            q_tick = robot.data.joint_pos.detach().clone()
            qdot_tick = robot.data.joint_vel.detach().clone()
            qdes_source = getattr(robot.data, "joint_pos_target", None)
            if not torch.is_tensor(qdes_source):
                qdes_source = getattr(robot, "_joint_pos_target", None)
            if not torch.is_tensor(qdes_source) or tuple(qdes_source.shape) != tuple(
                q0.shape
            ):
                raise DualEnvelopeProbeError(
                    "live articulation exposes no exact joint-position target buffer"
                )
            qdes_tick = qdes_source.detach().clone()
            for row, observation in zip(tape, observations, strict=True):
                env_id = row["env_id"]
                joint_index = row["joint_index"]
                observation["trajectory"].append(
                    {
                        "tick_index": tick_index,
                        "elapsed_s": tick_index * EXACT_PHYSICS_DT_S,
                        "q_rad": float(q_tick[env_id, joint_index].detach().cpu()),
                        "qdot_rad_s": float(
                            qdot_tick[env_id, joint_index].detach().cpu()
                        ),
                        "qdes_rad": float(
                            qdes_tick[env_id, joint_index].detach().cpu()
                        ),
                    }
                )
            if tick_index == 1:
                # Preserve the old diagnostic at its old sampling point.  Its
                # 20 ms ballistic proxy is receipt telemetry, not the new
                # four-tick actual-position verdict.
                action_term._record_physx_control_position_limit_diagnostic(
                    joint_pos=q_tick,
                    joint_vel=qdot_tick,
                )
                pending["diagnostic"]["post_step"] = (
                    action_term.consume_actual_joint_forbidden_diagnostic()
                )
        _emit_stage_marker("sim_step_done")
        live_identity.update({
            "run_specific_live_limit_sha256": (
                action_term._physx_control_position_limit_readback_sha256
            ),
            "setter_no_mutation_sha256": (
                action_term._physx_control_setter_no_mutation_sha256
            ),
            "mixed_live_limit_sha256": _sha256_bytes(
                mixed.contiguous().numpy().tobytes()
            ),
            "mixed_readback_exact": True,
            "disabled_event_terms": disabled_events,
            "disabled_debug_visualization": disabled_debug_visualization,
            "vendor_binding": vendor_binding,
        })
        # Validate only after finally restored the live plant.  The raw
        # trajectory has already been preserved tick-by-tick above.
    finally:
        if robot is not None and all_hctrl_cpu is not None:
            restore["attempted"] = True
            try:
                indices = robot._ALL_INDICES.detach().cpu()
                robot.root_physx_view.set_dof_limits(all_hctrl_cpu, indices=indices)
                readback = robot.root_physx_view.get_dof_limits()
                import torch

                restore["exact_readback"] = bool(torch.equal(readback, all_hctrl_cpu))
                if not restore["exact_readback"]:
                    restore["error"] = "restored live-limit readback differs from H_ctrl"
            except Exception as exc:  # noqa: BLE001 - preserve honest FAIL evidence
                restore["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            env.close()

        # Isaac Sim 4.5 closes with ``os._exit(0)`` on this runtime.  Closing
        # here would prevent the caller from validating and publishing the
        # no-clobber receipt, and would make a missing receipt look like rc=0.
        # The caller owns the app only after receipt publication.

    return pending, tape, {"identity": live_identity, "restore": restore}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute is not True or args.confirm != CONFIRM:
        raise SystemExit(f"refusing simulator probe without --execute --confirm {CONFIRM}")
    source_root = args.source_root.resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    source_commit = _verify_clean_exact_checkout(
        source_root,
        args.expected_source_commit,
        script_path=script_path,
    )
    output = _validate_output_path(
        args.output,
        (source_root, _installed_isaaclab_root()),
    )
    motion_files = []
    for raw in args.motion_file:
        path = raw.resolve(strict=True)
        if not path.is_file():
            raise DualEnvelopeProbeError(f"motion input is not a file: {path}")
        motion_files.append({"path": str(path), "sha256": _sha256_file(path)})

    error = None
    runtime = None
    tape: list[dict[str, Any]] = []
    identity = None
    restore: dict[str, Any] = {
        "attempted": False,
        "exact_readback": False,
        "error": "runtime did not reach restoration",
    }
    resources: dict[str, Any] = {}
    failure_evidence: Mapping[str, Any] | None = None
    publication_complete = False
    exit_code = 1
    try:
        try:
            raw_runtime, tape, sidecar = _run_live(args, resource_sink=resources)
            identity = sidecar["identity"]
            restore = sidecar["restore"]
            failure_evidence = raw_runtime
            runtime = validate_runtime_result(
                tape,
                raw_runtime["observations"],
                raw_runtime["diagnostic"],
                physics_dt_s=raw_runtime["physics_dt_s"],
                live_limits_restored_exact=(
                    restore.get("attempted") is True
                    and restore.get("exact_readback") is True
                ),
            )
        except Exception as exc:  # noqa: BLE001 - FAIL receipt is intentional
            error = f"{type(exc).__name__}: {exc}"
            tape = list(resources.get("tape", tape))
            identity = resources.get("identity", identity)
            restore = resources.get("restore", restore)
            failure_evidence = resources.get("failure_evidence", failure_evidence)

        # Re-attest the source immediately before the only publication side effect.
        _verify_clean_exact_checkout(
            source_root,
            source_commit,
            script_path=script_path,
        )
        for motion in motion_files:
            if _sha256_file(Path(motion["path"])) != motion["sha256"]:
                raise DualEnvelopeProbeError("motion input changed during the probe")
        payload = build_receipt(
            source_commit=source_commit,
            source_script_sha256=_sha256_file(script_path),
            task=args.task,
            motion_files=motion_files,
            tape=tape,
            runtime=runtime,
            live_limit_identity=identity,
            restore=restore,
            error=error,
            failure_evidence=failure_evidence,
        )
        _write_json_exclusive(output, payload)
        print(
            "A3_DUAL_POSITION_ENVELOPE_STRESS="
            + json.dumps(
                {
                    "status": payload["status"],
                    "output": str(output),
                    "content_sha256": payload["content_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        exit_code = 0 if payload["status"] == "PASS" else 1
        publication_complete = True
        return exit_code
    except BaseException:  # noqa: BLE001 - Kit close otherwise hides traceback
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app = resources.get("simulation_app")
        if simulation_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            if publication_complete and exit_code == 0:
                # Receipt and console marker are durable before Kit's hard exit.
                simulation_app.close()
            else:
                # Kit close forces rc=0.  Preserve FAIL/unpublished as nonzero.
                os._exit(exit_code if publication_complete else 1)


if __name__ == "__main__":
    raise SystemExit(main())
