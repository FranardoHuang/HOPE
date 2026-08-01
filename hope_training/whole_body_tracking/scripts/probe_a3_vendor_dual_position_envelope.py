#!/usr/bin/env python3
"""Run the source-bound A3 selected-joint dual-position-envelope stress probe.

This is a simulator-only mechanical test, never a training or deployment
authorization.  Exactly sixteen environments share eight state tapes:

``waist_roll/waist_pitch/left_ankle_roll/right_ankle_roll``
``x lower/upper x H_ctrl ON/OFF``.

For each side, ``R`` is the distance from the live two-percent control edge to
the unchanged mechanical edge.  The initial state is 0.1 R inside H_ctrl and
its exact 5 ms kinematic projection is 0.6 R outside H_ctrl (therefore 0.4 R
inside H_mech).  The ON/OFF pair has a byte-identical full-articulation input
tape: initial 31-joint q/qdot/q_des, origin-relative root pose/velocity and
isolated external rigid-object state, plus all 31 q_des inputs at every tick.
Only the target joint's live PhysX limit entry differs.  Tick-output q/qdot and
robot-root state are sealed and compared but are allowed to differ because
that difference is the mechanism under test; isolated external rigid objects
must remain pair-exact.  The existing action diagnostic retains its 20 ms
``capture_proxy`` meaning and is recorded as telemetry around the first tick.
The verdict follows a differential four-tick (20 ms) policy horizon: ON may
penetrate H_ctrl by less than the full control-to-mechanical reserve but must
remain strictly inside H_mech; the same-tape OFF row disables H_ctrl only for
that target joint, enters the control-to-mechanical band on tick one, and must
touch or cross H_mech within the horizon.  Neither measurement is called a
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


SCHEMA_VERSION = 5
KIND = "a3_vendor_dual_position_envelope_differential_stress_v5"
CONFIRM = "SIM_ONLY_A3_DUAL_POSITION_ENVELOPE_16ENV_DIFFERENTIAL_V5"
EXACT_TASK = "HOPE-PingPong-ActionBall-AgibotA3-v0"
VENDOR_TASK_PROFILE = "HOPEPingPongActionBallA3VendorV1"
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
ROBOT_JOINT_COUNT = 31
ROOT_POSITION_DIM = 3
ROOT_QUATERNION_DIM = 4
ROOT_VELOCITY_DIM = 3
FULL_STATE_SCHEMA_VERSION = 1
STRESSED_JOINTS = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
SIDES = ("lower", "upper")
CONDITIONS = ("on", "off")
EXACT_NUM_ENVS = len(STRESSED_JOINTS) * len(SIDES) * len(CONDITIONS)
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


def _validate_uniform_limit_row_bytes(
    row_bytes: Sequence[bytes],
    *,
    label: str,
) -> None:
    """Require every environment to expose one byte-identical limit row."""

    if len(row_bytes) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError(
            f"{label} must contain exactly {EXACT_NUM_ENVS} environment rows"
        )
    if any(type(row) is not bytes or not row for row in row_bytes):
        raise DualEnvelopeProbeError(f"{label} rows must be non-empty exact bytes")
    if any(row != row_bytes[0] for row in row_bytes[1:]):
        raise DualEnvelopeProbeError(
            f"{label} differs across environments; env0 cannot define all tapes"
        )


def _finite_vector(value: object, *, label: str, expected: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise DualEnvelopeProbeError(
            f"{label} must contain exactly {expected} finite numbers"
        )
    return [
        _finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    ]


def _payload_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes({"value": value}))


def _seal_full_state_snapshot(content: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one raw full-system state; validation always recomputes the seal."""

    payload = dict(content)
    payload.pop("content_sha256", None)
    return {
        **payload,
        "content_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }


def _normalize_full_state_snapshot(
    raw: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate exact full-articulation/root/external-state receipt content."""

    if not isinstance(raw, Mapping):
        raise DualEnvelopeProbeError(f"{label} must be one full-state Mapping")
    expected_keys = {
        "schema_version",
        "joint_pos_rad",
        "joint_vel_rad_s",
        "joint_pos_target_rad",
        "robot_root_origin_relative",
        "scene_rigid_objects",
        "content_sha256",
    }
    if set(raw) != expected_keys:
        raise DualEnvelopeProbeError(f"{label} full-state keys drifted")
    if raw.get("schema_version") != FULL_STATE_SCHEMA_VERSION:
        raise DualEnvelopeProbeError(f"{label} full-state schema drifted")
    joint_pos = _finite_vector(
        raw.get("joint_pos_rad"),
        label=f"{label}.joint_pos_rad",
        expected=ROBOT_JOINT_COUNT,
    )
    joint_vel = _finite_vector(
        raw.get("joint_vel_rad_s"),
        label=f"{label}.joint_vel_rad_s",
        expected=ROBOT_JOINT_COUNT,
    )
    joint_target = _finite_vector(
        raw.get("joint_pos_target_rad"),
        label=f"{label}.joint_pos_target_rad",
        expected=ROBOT_JOINT_COUNT,
    )
    root_raw = raw.get("robot_root_origin_relative")
    root_keys = {
        "position_m",
        "quaternion_xyzw",
        "linear_velocity_w_m_s",
        "angular_velocity_w_rad_s",
    }
    if not isinstance(root_raw, Mapping) or set(root_raw) != root_keys:
        raise DualEnvelopeProbeError(f"{label} robot-root keys drifted")
    root = {
        "position_m": _finite_vector(
            root_raw.get("position_m"),
            label=f"{label}.robot_root.position_m",
            expected=ROOT_POSITION_DIM,
        ),
        "quaternion_xyzw": _finite_vector(
            root_raw.get("quaternion_xyzw"),
            label=f"{label}.robot_root.quaternion_xyzw",
            expected=ROOT_QUATERNION_DIM,
        ),
        "linear_velocity_w_m_s": _finite_vector(
            root_raw.get("linear_velocity_w_m_s"),
            label=f"{label}.robot_root.linear_velocity_w_m_s",
            expected=ROOT_VELOCITY_DIM,
        ),
        "angular_velocity_w_rad_s": _finite_vector(
            root_raw.get("angular_velocity_w_rad_s"),
            label=f"{label}.robot_root.angular_velocity_w_rad_s",
            expected=ROOT_VELOCITY_DIM,
        ),
    }
    quat_norm = math.sqrt(sum(value * value for value in root["quaternion_xyzw"]))
    if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise DualEnvelopeProbeError(f"{label} robot-root quaternion is not unit")

    rigid_raw = raw.get("scene_rigid_objects")
    if not isinstance(rigid_raw, list):
        raise DualEnvelopeProbeError(f"{label} scene_rigid_objects must be a list")
    rigid_objects = []
    rigid_keys = {
        "name",
        "position_m",
        "quaternion_xyzw",
        "linear_velocity_w_m_s",
        "angular_velocity_w_rad_s",
    }
    for index, item in enumerate(rigid_raw):
        if not isinstance(item, Mapping) or set(item) != rigid_keys:
            raise DualEnvelopeProbeError(
                f"{label}.scene_rigid_objects[{index}] keys drifted"
            )
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise DualEnvelopeProbeError(
                f"{label}.scene_rigid_objects[{index}].name is invalid"
            )
        normalized_item = {
            "name": name,
            "position_m": _finite_vector(
                item.get("position_m"),
                label=f"{label}.scene_rigid_objects[{index}].position_m",
                expected=ROOT_POSITION_DIM,
            ),
            "quaternion_xyzw": _finite_vector(
                item.get("quaternion_xyzw"),
                label=f"{label}.scene_rigid_objects[{index}].quaternion_xyzw",
                expected=ROOT_QUATERNION_DIM,
            ),
            "linear_velocity_w_m_s": _finite_vector(
                item.get("linear_velocity_w_m_s"),
                label=(
                    f"{label}.scene_rigid_objects[{index}].linear_velocity_w_m_s"
                ),
                expected=ROOT_VELOCITY_DIM,
            ),
            "angular_velocity_w_rad_s": _finite_vector(
                item.get("angular_velocity_w_rad_s"),
                label=(
                    f"{label}.scene_rigid_objects[{index}].angular_velocity_w_rad_s"
                ),
                expected=ROOT_VELOCITY_DIM,
            ),
        }
        rigid_quat_norm = math.sqrt(
            sum(value * value for value in normalized_item["quaternion_xyzw"])
        )
        if not math.isclose(
            rigid_quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5
        ):
            raise DualEnvelopeProbeError(
                f"{label}.scene_rigid_objects[{index}] quaternion is not unit"
            )
        rigid_objects.append(normalized_item)
    rigid_names = [item["name"] for item in rigid_objects]
    if rigid_names != sorted(set(rigid_names)):
        raise DualEnvelopeProbeError(
            f"{label} rigid-object names must be unique canonical order"
        )

    content = {
        "schema_version": FULL_STATE_SCHEMA_VERSION,
        "joint_pos_rad": joint_pos,
        "joint_vel_rad_s": joint_vel,
        "joint_pos_target_rad": joint_target,
        "robot_root_origin_relative": root,
        "scene_rigid_objects": rigid_objects,
    }
    expected_sha256 = _sha256_bytes(_canonical_json_bytes(content))
    if raw.get("content_sha256") != expected_sha256:
        raise DualEnvelopeProbeError(f"{label} full-state digest mismatch")
    return {**content, "content_sha256": expected_sha256}


def _pair_component_proof(
    on_value: object,
    off_value: object,
    *,
    label: str,
    require_exact: bool,
) -> dict[str, Any]:
    """Digest and compare one pair component without trusting producer booleans."""

    on_sha256 = _payload_sha256(on_value)
    off_sha256 = _payload_sha256(off_value)
    exact = on_sha256 == off_sha256
    if require_exact and not exact:
        raise DualEnvelopeProbeError(f"ON/OFF {label} is not exact pair parity")
    return {
        "on_sha256": on_sha256,
        "off_sha256": off_sha256,
        "exact": exact,
        "required_exact": require_exact,
    }


def build_stress_tape(
    joint_names: Sequence[str],
    mechanical_limits: Sequence[Sequence[float]],
    control_limits: Sequence[Sequence[float]],
    *,
    physics_dt_s: float = EXACT_PHYSICS_DT_S,
) -> list[dict[str, Any]]:
    """Build the selected-joint same-state ON/OFF tape without simulator imports."""

    if type(physics_dt_s) is not float or physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("stress tape requires exact physics_dt_s=0.005")
    names = tuple(str(name) for name in joint_names)
    if len(names) != len(set(names)) or any(
        names.count(name) != 1 for name in STRESSED_JOINTS
    ):
        raise DualEnvelopeProbeError(
            "joint order must contain each selected stressed joint exactly once"
        )
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
        raise DualEnvelopeProbeError(
            f"stress tape must contain exactly {EXACT_NUM_ENVS} rows"
        )
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
    verdict because the differentiating evidence is actual ON/OFF motion under
    an exact common input tape.  H_mech remains strict for ON; OFF touching it
    is positive-control evidence only and never a safety authorization.
    """

    validate_stress_tape(tape)
    if physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("live physics dt is not exact 0.005 s")
    if live_limits_restored_exact is not True:
        raise DualEnvelopeProbeError("all-environment H_ctrl restoration/readback failed")
    if len(observations) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError(
            f"runtime must return exactly {EXACT_NUM_ENVS} observation rows"
        )

    normalized: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], dict[str, int]] = {
        (joint, side): {
            "strict_5ms_kinematic_attempt_count": 0,
            "trajectory_tick_count": 0,
            "on_strict_hmech_tick_count": 0,
            "on_ctrl_penetration_tick_count": 0,
            "off_first_tick_ctrl_band_entry_count": 0,
            "off_mech_touch_or_penetration_tick_count": 0,
            "qdes_equal_q0_exact_tick_count": 0,
        }
        for joint in STRESSED_JOINTS
        for side in SIDES
    }
    pair_metrics: dict[tuple[str, str], dict[str, float]] = {
        (joint, side): {
            "max_on_ctrl_penetration_rad": 0.0,
            "min_on_mech_gap_rad": math.inf,
            "max_off_mech_penetration_rad": 0.0,
        }
        for joint in STRESSED_JOINTS
        for side in SIDES
    }
    for env_id, (tape_row, raw) in enumerate(zip(tape, observations)):
        if not isinstance(raw, Mapping):
            raise DualEnvelopeProbeError("runtime observation row must be a Mapping")
        if raw.get("env_id") != env_id:
            raise DualEnvelopeProbeError("runtime observation env ids drifted")
        if raw.get("joint") != tape_row["joint"] or raw.get("side") != tape_row["side"]:
            raise DualEnvelopeProbeError("runtime observation identity differs from tape")
        if raw.get("condition") != tape_row["condition"]:
            raise DualEnvelopeProbeError("runtime condition differs from tape")
        joint_index = int(tape_row["joint_index"])
        if not 0 <= joint_index < ROBOT_JOINT_COUNT:
            raise DualEnvelopeProbeError("stressed joint index is outside 31-joint state")
        initial_state = _normalize_full_state_snapshot(
            raw.get("initial_full_state"),
            label=f"observation[{env_id}].initial_full_state",
        )
        q0_live = _finite_number(
            raw.get("q0_live_rad"), f"observation[{env_id}].q0_live"
        )
        qdot0_live = _finite_number(
            raw.get("qdot0_live_rad_s"), f"observation[{env_id}].qdot0_live"
        )
        if q0_live != _float32_round(float(tape_row["q0_rad"])):
            raise DualEnvelopeProbeError("live q0 differs from the float32 exact stress tape")
        if qdot0_live != _float32_round(float(tape_row["qdot0_rad_s"])):
            raise DualEnvelopeProbeError(
                "live initial qdot differs from the float32 exact stress tape"
            )
        if initial_state["joint_pos_rad"][joint_index] != q0_live:
            raise DualEnvelopeProbeError(
                "initial full joint q differs from stressed-axis q0 readback"
            )
        if initial_state["joint_vel_rad_s"][joint_index] != qdot0_live:
            raise DualEnvelopeProbeError(
                "initial full joint qdot differs from stressed-axis readback"
            )
        if initial_state["joint_pos_target_rad"][joint_index] != q0_live:
            raise DualEnvelopeProbeError(
                "initial full q_des differs from stressed-axis q0 readback"
            )
        _pair_component_proof(
            initial_state["joint_pos_rad"],
            initial_state["joint_pos_target_rad"],
            label=f"observation[{env_id}] initial q versus q_des",
            require_exact=True,
        )
        trajectory = raw.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != POLICY_HORIZON_PHYSICS_TICKS:
            raise DualEnvelopeProbeError(
                "runtime row must contain exactly four ordered physics ticks"
            )
        condition = str(tape_row["condition"])
        key = (str(tape_row["joint"]), str(tape_row["side"]))
        pair_counts[key]["strict_5ms_kinematic_attempt_count"] += 1
        off_mech_touch_or_penetration_count = 0
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
            full_state = _normalize_full_state_snapshot(
                sample.get("full_state"),
                label=(
                    f"observation[{env_id}].trajectory[{offset}].full_state"
                ),
            )
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
            if full_state["joint_pos_rad"][joint_index] != q:
                raise DualEnvelopeProbeError(
                    "tick full joint q differs from stressed-axis q readback"
                )
            if full_state["joint_vel_rad_s"][joint_index] != qdot:
                raise DualEnvelopeProbeError(
                    "tick full joint qdot differs from stressed-axis readback"
                )
            if full_state["joint_pos_target_rad"][joint_index] != qdes:
                raise DualEnvelopeProbeError(
                    "tick full q_des differs from stressed-axis target readback"
                )
            if qdes != q0_live:
                raise DualEnvelopeProbeError("live q_des differs from exact live q0")
            _pair_component_proof(
                initial_state["joint_pos_target_rad"],
                full_state["joint_pos_target_rad"],
                label=f"observation[{env_id}] tick {tick_index} full q_des hold",
                require_exact=True,
            )
            control_gap, mechanical_gap = _side_gaps(tape_row, q)
            local = pair_counts[key]
            metrics = pair_metrics[key]
            local["trajectory_tick_count"] += 1
            local["qdes_equal_q0_exact_tick_count"] += 1
            if condition == "on":
                ctrl_penetration = max(0.0, -control_gap)
                if not ctrl_penetration < float(tape_row["cage_reserve_rad"]):
                    raise DualEnvelopeProbeError(
                        "H_ctrl ON penetration reached the full cage reserve"
                    )
                if not mechanical_gap > 0.0:
                    raise DualEnvelopeProbeError(
                        "H_ctrl ON touched/crossed H_mech during policy horizon"
                    )
                local["on_strict_hmech_tick_count"] += 1
                if ctrl_penetration > 0.0:
                    local["on_ctrl_penetration_tick_count"] += 1
                metrics["max_on_ctrl_penetration_rad"] = max(
                    metrics["max_on_ctrl_penetration_rad"], ctrl_penetration
                )
                metrics["min_on_mech_gap_rad"] = min(
                    metrics["min_on_mech_gap_rad"], mechanical_gap
                )
            else:
                if tick_index == 1:
                    if not control_gap <= 0.0 or not mechanical_gap > 0.0:
                        raise DualEnvelopeProbeError(
                            "H_ctrl OFF tick one is not in [H_ctrl,Hmech)"
                        )
                    local["off_first_tick_ctrl_band_entry_count"] += 1
                if mechanical_gap <= 0.0:
                    off_mech_touch_or_penetration_count += 1
                    local["off_mech_touch_or_penetration_tick_count"] += 1
                metrics["max_off_mech_penetration_rad"] = max(
                    metrics["max_off_mech_penetration_rad"],
                    max(0.0, -mechanical_gap),
                )
            normalized_ticks.append(
                {
                    **dict(sample),
                    "elapsed_s": elapsed,
                    "q_rad": q,
                    "qdot_rad_s": qdot,
                    "qdes_rad": qdes,
                    "full_state": full_state,
                    "signed_ctrl_gap_rad": control_gap,
                    "signed_mechanical_gap_rad": mechanical_gap,
                }
            )
        if condition == "off" and off_mech_touch_or_penetration_count == 0:
            raise DualEnvelopeProbeError(
                "H_ctrl OFF did not touch/cross H_mech within policy horizon"
            )
        normalized.append(
            {
                **dict(raw),
                "q0_live_rad": q0_live,
                "qdot0_live_rad_s": qdot0_live,
                "initial_full_state": initial_state,
                "trajectory": normalized_ticks,
                "strict_5ms_kinematic_crossing": True,
            }
        )

    pair_state_parity = []
    for pair_start in range(0, EXACT_NUM_ENVS, 2):
        on = normalized[pair_start]
        off = normalized[pair_start + 1]
        tape_on = tape[pair_start]
        tape_off = tape[pair_start + 1]
        if on["q0_live_rad"] != off["q0_live_rad"]:
            raise DualEnvelopeProbeError("ON/OFF live q0 is not exact same-tape")
        if (
            tape_on["qdot0_rad_s"] != tape_off["qdot0_rad_s"]
            or on["qdot0_live_rad_s"] != off["qdot0_live_rad_s"]
        ):
            raise DualEnvelopeProbeError("ON/OFF initial qdot is not exact same-tape")
        if [tick["qdes_rad"] for tick in on["trajectory"]] != [
            tick["qdes_rad"] for tick in off["trajectory"]
        ]:
            raise DualEnvelopeProbeError("ON/OFF q_des trajectory is not exact same-tape")
        on_initial = on["initial_full_state"]
        off_initial = off["initial_full_state"]
        initial_input = {
            "joint_pos_rad": _pair_component_proof(
                on_initial["joint_pos_rad"],
                off_initial["joint_pos_rad"],
                label="initial full 31-joint q",
                require_exact=True,
            ),
            "joint_vel_rad_s": _pair_component_proof(
                on_initial["joint_vel_rad_s"],
                off_initial["joint_vel_rad_s"],
                label="initial full 31-joint qdot",
                require_exact=True,
            ),
            "joint_pos_target_rad": _pair_component_proof(
                on_initial["joint_pos_target_rad"],
                off_initial["joint_pos_target_rad"],
                label="initial full 31-joint q_des",
                require_exact=True,
            ),
            "robot_root_origin_relative": _pair_component_proof(
                on_initial["robot_root_origin_relative"],
                off_initial["robot_root_origin_relative"],
                label="initial origin-relative robot root",
                require_exact=True,
            ),
            "scene_rigid_objects": _pair_component_proof(
                on_initial["scene_rigid_objects"],
                off_initial["scene_rigid_objects"],
                label="initial isolated scene rigid objects",
                require_exact=True,
            ),
            "full_state": _pair_component_proof(
                {
                    key: value
                    for key, value in on_initial.items()
                    if key != "content_sha256"
                },
                {
                    key: value
                    for key, value in off_initial.items()
                    if key != "content_sha256"
                },
                label="initial full-system state",
                require_exact=True,
            ),
        }
        tick_rows = []
        for tick_index, (on_tick, off_tick) in enumerate(
            zip(on["trajectory"], off["trajectory"]),
            start=1,
        ):
            on_state = on_tick["full_state"]
            off_state = off_tick["full_state"]
            tick_rows.append(
                {
                    "tick_index": tick_index,
                    "input_joint_pos_target_rad": _pair_component_proof(
                        on_state["joint_pos_target_rad"],
                        off_state["joint_pos_target_rad"],
                        label=f"tick {tick_index} full 31-joint q_des input",
                        require_exact=True,
                    ),
                    "isolated_scene_rigid_objects": _pair_component_proof(
                        on_state["scene_rigid_objects"],
                        off_state["scene_rigid_objects"],
                        label=f"tick {tick_index} isolated scene rigid objects",
                        require_exact=True,
                    ),
                    "output_joint_pos_rad": _pair_component_proof(
                        on_state["joint_pos_rad"],
                        off_state["joint_pos_rad"],
                        label=f"tick {tick_index} output full joint q",
                        require_exact=False,
                    ),
                    "output_joint_vel_rad_s": _pair_component_proof(
                        on_state["joint_vel_rad_s"],
                        off_state["joint_vel_rad_s"],
                        label=f"tick {tick_index} output full joint qdot",
                        require_exact=False,
                    ),
                    "output_robot_root_origin_relative": _pair_component_proof(
                        on_state["robot_root_origin_relative"],
                        off_state["robot_root_origin_relative"],
                        label=f"tick {tick_index} output robot root",
                        require_exact=False,
                    ),
                }
            )
        pair_state_parity.append(
            {
                "joint": tape_on["joint"],
                "side": tape_on["side"],
                "on_env_id": pair_start,
                "off_env_id": pair_start + 1,
                "initial_input": initial_input,
                "ticks": tick_rows,
                "exact_input_tape": True,
                "tick_output_q_qdot_root_may_differ": True,
            }
        )

    diagnostic_phases: dict[str, Mapping[str, Any]] = {}
    for phase in ("pre_step", "post_step"):
        phase_payload = diagnostic.get(phase)
        if not isinstance(phase_payload, Mapping):
            raise DualEnvelopeProbeError(
                f"{phase} diagnostic must be one recordable Mapping"
            )
        diagnostic_phases[phase] = phase_payload
    try:
        _canonical_json_bytes({"diagnostic": dict(diagnostic)})
    except (TypeError, ValueError) as exc:
        raise DualEnvelopeProbeError(
            f"diagnostic telemetry is not canonically recordable: {exc}"
        ) from exc

    def telemetry_counter(phase: str, joint: str, side: str, name: str) -> int | None:
        """Best-effort parsing only; no diagnostic value participates in PASS."""

        diag = diagnostic_phases[phase].get("physx_control_position_limits")
        if not isinstance(diag, Mapping):
            return None
        rows = diag.get("by_joint")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, Mapping) or row.get("joint") != joint:
                continue
            sides = row.get("sides")
            values = sides.get(side) if isinstance(sides, Mapping) else None
            value = values.get(name) if isinstance(values, Mapping) else None
            return value if type(value) is int else None
        return None

    aggregate_rows = []
    for joint in STRESSED_JOINTS:
        for side in SIDES:
            local = pair_counts[(joint, side)]
            expected_exact = {
                "strict_5ms_kinematic_attempt_count": 2,
                "trajectory_tick_count": 8,
                "on_strict_hmech_tick_count": 4,
                "off_first_tick_ctrl_band_entry_count": 1,
                "qdes_equal_q0_exact_tick_count": 8,
            }
            if any(local[name] != value for name, value in expected_exact.items()):
                raise AssertionError("internal per-pair aggregate drifted")
            if local["off_mech_touch_or_penetration_tick_count"] < 1:
                raise AssertionError("OFF positive-control aggregate drifted")
            metrics = pair_metrics[(joint, side)]
            if not math.isfinite(metrics["min_on_mech_gap_rad"]):
                raise AssertionError("ON mechanical-gap aggregate is absent")
            aggregate_rows.append(
                {
                    "joint": joint,
                    "side": side,
                    **local,
                    **metrics,
                    "same_tape_q0_qdot_qdes_exact": True,
                    "same_tape_initial_full_system_and_all_tick_qdes_exact": True,
                    "existing_20ms_ballistic_attempt_proxy_count": telemetry_counter(
                        "pre_step", joint, side, "ballistic_attempt_proxy"
                    ),
                    "post_20ms_ballistic_attempt_proxy_count": telemetry_counter(
                        "post_step", joint, side, "ballistic_attempt_proxy"
                    ),
                    "existing_20ms_capture_proxy_count": telemetry_counter(
                        "post_step", joint, side, "capture_proxy"
                    ),
                    "post_ctrl_penetration_readback_count": telemetry_counter(
                        "post_step", joint, side, "ctrl_penetration_readback"
                    ),
                }
            )

    return {
        "observations": normalized,
        "aggregate_by_joint_side": aggregate_rows,
        "pair_state_parity": pair_state_parity,
        "on_mechanical_touch_or_penetration_count": 0,
        "off_mechanical_touch_or_penetration_count": sum(
            row["off_mech_touch_or_penetration_tick_count"]
            for row in aggregate_rows
        ),
        "max_on_ctrl_penetration_rad": max(
            row["max_on_ctrl_penetration_rad"] for row in aggregate_rows
        ),
        "min_on_mech_gap_rad": min(
            row["min_on_mech_gap_rad"] for row in aggregate_rows
        ),
        "max_off_mech_penetration_rad": max(
            row["max_off_mech_penetration_rad"] for row in aggregate_rows
        ),
        "all_rows_finite": True,
        "all_qdes_equal_q0_exact": True,
        "all_on_off_input_tapes_exact": True,
        "all_initial_full_system_states_pair_exact": True,
        "all_tick_full_joint_qdes_inputs_pair_exact": True,
        "all_tick_isolated_scene_rigid_objects_pair_exact": True,
        "tick_output_q_qdot_root_are_comparisons_not_input_parity": True,
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
            "robot_joint_count": ROBOT_JOINT_COUNT,
            "full_state_schema_version": FULL_STATE_SCHEMA_VERSION,
            "q0_inner_cage_fraction": Q0_INNER_CAGE_FRACTION,
            "kinematic_outer_cage_fraction": KINEMATIC_OUTER_CAGE_FRACTION,
            "kinematic_remaining_mechanical_cage_fraction": (
                MECHANICAL_REMAINING_CAGE_FRACTION
            ),
            "qdot_formula": "direction*(0.1+0.6)*R/0.005",
            "qdes_contract": (
                "initial full 31-joint qdes=q0 exact and all four tick full "
                "31-joint qdes inputs are exact ON/OFF pair parity"
            ),
            "input_tape_contract": (
                "initial full 31-joint q/qdot/qdes, origin-relative raw-PhysX "
                "robot root transform/velocity and isolated scene rigid-object "
                "states are exact ON/OFF pair parity"
            ),
            "tick_output_contract": (
                "full 31-joint q/qdot and origin-relative robot root are sealed "
                "and pair-compared as outputs but need not be equal; isolated "
                "scene rigid objects remain exact pair parity"
            ),
            "existing_capture_proxy_horizon_s": 0.02,
            "existing_capture_proxy_verdict_role": "telemetry_only",
            "existing_capture_proxy_sampling": "pre_tick_1_and_post_tick_1",
            "strict_kinematic_crossing_horizon_s": 0.005,
            "off_hmech_touch_semantics": (
                "same-tape positive control only; never safety authorization"
            ),
            "trajectory_contract": (
                "each tick finite full q/qdot/qdes/root/external state with "
                "component digests; same-tape ON strict H_mech with "
                "H_ctrl penetration below cage reserve; OFF tick1 in "
                "[H_ctrl,H_mech) and touches/crosses H_mech within four ticks; "
                "full 31-joint qdes=q0 exact"
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
            raise DualEnvelopeProbeError(
                f"live task did not construct exactly {EXACT_NUM_ENVS} envs"
            )
        if float(base.physics_dt) != EXACT_PHYSICS_DT_S:
            raise DualEnvelopeProbeError("live physics_dt is not exact 0.005")
        action_term = base.action_manager.get_term("joint_pos")
        robot = base.scene["robot"]
        root_view = robot.root_physx_view
        _emit_stage_marker("hctrl_readback_begin")
        action_term.verify_physx_control_position_limit_readback()
        public_contract_getter = getattr(
            action_term, "physx_control_position_limits_contract", None
        )
        if not callable(public_contract_getter):
            raise DualEnvelopeProbeError(
                "action term exposes no public H_ctrl/H_mech contract"
            )
        position_limit_contract = public_contract_getter()
        if not isinstance(position_limit_contract, Mapping):
            raise DualEnvelopeProbeError(
                "public H_ctrl/H_mech contract is not a Mapping"
            )
        names = tuple(str(name) for name in robot.joint_names)
        if len(names) != ROBOT_JOINT_COUNT or len(set(names)) != ROBOT_JOINT_COUNT:
            raise DualEnvelopeProbeError(
                "live robot must expose exactly 31 unique ordered joints"
            )
        selected_names = tuple(
            str(name)
            for name in position_limit_contract.get("selected_joint_names", ())
        )
        selected_indices = tuple(
            int(index)
            for index in position_limit_contract.get("selected_joint_indices", ())
        )
        expected_selected_indices = tuple(
            names.index(name) for name in STRESSED_JOINTS
        )
        if position_limit_contract.get("enabled") is not True:
            raise DualEnvelopeProbeError("public H_ctrl/H_mech contract is disabled")
        if selected_names != STRESSED_JOINTS:
            raise DualEnvelopeProbeError(
                "public H_ctrl selected-joint order differs from the stress contract"
            )
        if selected_indices != expected_selected_indices:
            raise DualEnvelopeProbeError(
                "public H_ctrl selected indices differ from live robot joint order"
            )
        if position_limit_contract.get("unselected_joint_count") != (
            len(names) - len(STRESSED_JOINTS)
        ):
            raise DualEnvelopeProbeError(
                "public H_ctrl unselected-joint count differs from live robot"
            )
        if tuple(position_limit_contract.get("joint_order", ())) != names:
            raise DualEnvelopeProbeError(
                "public H_ctrl joint order differs from live robot joint order"
            )
        if (
            position_limit_contract.get("inset_fraction_per_side_hard_span")
            != CONTROL_INSET_FRACTION
        ):
            raise DualEnvelopeProbeError("public H_ctrl inset fraction drifted")
        if (
            position_limit_contract.get("mechanical_edge_ledger_uses_h_mech")
            is not True
            or position_limit_contract.get("soft_qdes_envelope_unchanged") is not True
        ):
            raise DualEnvelopeProbeError(
                "public H_ctrl mechanical-ledger/soft-qdes semantics drifted"
            )
        public_readback_sha256 = position_limit_contract.get("readback_sha256")
        if (
            not isinstance(public_readback_sha256, str)
            or len(public_readback_sha256) != 64
        ):
            raise DualEnvelopeProbeError(
                "public H_ctrl readback identity is not one SHA-256"
            )
        setter_no_mutation_sha256 = (
            action_term.physx_control_setter_no_mutation_sha256
        )
        if (
            not isinstance(setter_no_mutation_sha256, str)
            or len(setter_no_mutation_sha256) != 64
        ):
            raise DualEnvelopeProbeError(
                "public H_ctrl setter identity is not one SHA-256"
            )
        mechanical = position_limit_contract.get("mechanical_joint_pos_limits")
        control = position_limit_contract.get("control_joint_pos_limits")
        if not torch.is_tensor(mechanical) or not torch.is_tensor(control):
            raise DualEnvelopeProbeError(
                "public H_ctrl/H_mech contract omitted exact tensors"
            )
        if tuple(mechanical.shape) != tuple(control.shape) or tuple(
            mechanical.shape
        ) != (EXACT_NUM_ENVS, len(names), 2):
            raise DualEnvelopeProbeError(
                "public H_ctrl/H_mech tensor shape differs from live articulation"
            )
        mechanical = mechanical.detach().cpu().clone()
        all_hctrl_cpu = control.detach().cpu().clone()
        _validate_uniform_limit_row_bytes(
            [
                mechanical[env_id].contiguous().numpy().tobytes()
                for env_id in range(EXACT_NUM_ENVS)
            ],
            label="public H_mech",
        )
        _validate_uniform_limit_row_bytes(
            [
                all_hctrl_cpu[env_id].contiguous().numpy().tobytes()
                for env_id in range(EXACT_NUM_ENVS)
            ],
            label="public H_ctrl",
        )
        hmech_data_readback = robot.data.joint_pos_limits.detach().cpu()
        if not torch.equal(hmech_data_readback, mechanical):
            raise DualEnvelopeProbeError(
                "public H_mech tensor differs from independent articulation data"
            )
        hctrl_root_readback = root_view.get_dof_limits()
        if not torch.equal(hctrl_root_readback, all_hctrl_cpu):
            raise DualEnvelopeProbeError(
                "public H_ctrl tensor differs from independent root PhysX readback"
            )
        _emit_stage_marker("hctrl_readback_done")
        # Reset may legitimately exercise the normal update diagnostic before
        # this four-tick tape starts.  Clear both its published counters and its
        # temporal proxy history so the receipt contains only these selected
        # joint ON/OFF environments.
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
                mixed[row["env_id"], row["joint_index"]] = mechanical[
                    row["env_id"], row["joint_index"]
                ]
        indices = robot._ALL_INDICES.detach().cpu()
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

        env_origins = base.scene.env_origins.detach().clone()
        if tuple(env_origins.shape) != (EXACT_NUM_ENVS, ROOT_POSITION_DIM):
            raise DualEnvelopeProbeError("scene env-origin tensor shape drifted")
        raw_rigid_objects = getattr(base.scene, "rigid_objects", None)
        if not isinstance(raw_rigid_objects, Mapping):
            raise DualEnvelopeProbeError(
                "interactive scene exposes no rigid_objects Mapping"
            )
        rigid_object_items = tuple(
            sorted(
                ((str(name), asset) for name, asset in raw_rigid_objects.items()),
                key=lambda item: item[0],
            )
        )
        if len({name for name, _ in rigid_object_items}) != len(rigid_object_items):
            raise DualEnvelopeProbeError("scene rigid-object names are not unique")
        env_ids_device = torch.arange(
            EXACT_NUM_ENVS,
            device=env_origins.device,
            dtype=torch.long,
        )
        isolated_rigid_object_names = []
        for object_index, (object_name, rigid_object) in enumerate(
            rigid_object_items
        ):
            root_state = rigid_object.data.root_state_w.detach().clone()
            if tuple(root_state.shape) != (EXACT_NUM_ENVS, 13):
                raise DualEnvelopeProbeError(
                    f"scene rigid object {object_name!r} root-state shape drifted"
                )
            pose_wxyz = root_state[:, :7].clone()
            velocity = torch.zeros_like(root_state[:, 7:13])
            parking_local = torch.tensor(
                [16.0 + 2.0 * object_index, 0.0, 16.0],
                device=pose_wxyz.device,
                dtype=pose_wxyz.dtype,
            )
            pose_wxyz[:, :3] = env_origins.to(
                device=pose_wxyz.device,
                dtype=pose_wxyz.dtype,
            ) + parking_local
            pose_wxyz[:, 3:7] = 0.0
            pose_wxyz[:, 3] = 1.0
            rigid_object.write_root_pose_to_sim(
                pose_wxyz,
                env_ids=env_ids_device.to(device=pose_wxyz.device),
            )
            rigid_object.write_root_velocity_to_sim(
                velocity,
                env_ids=env_ids_device.to(device=velocity.device),
            )
            isolated_rigid_object_names.append(object_name)

        robot.write_joint_state_to_sim(q0, qdot0)
        robot.set_joint_position_target(q0)
        base.scene.write_data_to_sim()
        # Read PhysX directly without advancing physics.  ArticulationData's
        # zero-dt update does not advance its timestamp and may return the
        # buffers populated by write_joint_state_to_sim(), so it is not an
        # independent live-state attestation.
        q0_live_readback = root_view.get_dof_positions().detach().clone()
        qdot0_live_readback = root_view.get_dof_velocities().detach().clone()
        if tuple(q0_live_readback.shape) != tuple(q0.shape) or tuple(
            qdot0_live_readback.shape
        ) != tuple(qdot0.shape):
            raise DualEnvelopeProbeError("initial simulator readback shape drifted")
        if not torch.all(torch.isfinite(q0_live_readback)) or not torch.all(
            torch.isfinite(qdot0_live_readback)
        ):
            raise DualEnvelopeProbeError("initial simulator readback is non-finite")

        def read_joint_position_target() -> Any:
            qdes_source = getattr(robot.data, "joint_pos_target", None)
            if not torch.is_tensor(qdes_source):
                qdes_source = getattr(robot, "_joint_pos_target", None)
            if not torch.is_tensor(qdes_source) or tuple(qdes_source.shape) != tuple(
                q0.shape
            ):
                raise DualEnvelopeProbeError(
                    "live articulation exposes no exact joint-position target buffer"
                )
            return qdes_source.detach().clone()

        def capture_full_states(
            joint_pos: Any,
            joint_vel: Any,
            joint_target: Any,
        ) -> list[dict[str, Any]]:
            expected_joint_shape = (EXACT_NUM_ENVS, ROBOT_JOINT_COUNT)
            for label, tensor in (
                ("joint_pos", joint_pos),
                ("joint_vel", joint_vel),
                ("joint_target", joint_target),
            ):
                if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_joint_shape:
                    raise DualEnvelopeProbeError(
                        f"full-state {label} tensor shape drifted"
                    )
                if not torch.all(torch.isfinite(tensor)):
                    raise DualEnvelopeProbeError(
                        f"full-state {label} tensor is non-finite"
                    )
            root_transform = root_view.get_root_transforms().detach().clone()
            root_velocity = root_view.get_root_velocities().detach().clone()
            if tuple(root_transform.shape) != (EXACT_NUM_ENVS, 7) or tuple(
                root_velocity.shape
            ) != (EXACT_NUM_ENVS, 6):
                raise DualEnvelopeProbeError("raw PhysX robot-root shape drifted")
            root_origins = env_origins.to(
                device=root_transform.device,
                dtype=root_transform.dtype,
            )
            root_position_local = root_transform[:, :3] - root_origins
            if not torch.all(torch.isfinite(root_transform)) or not torch.all(
                torch.isfinite(root_velocity)
            ):
                raise DualEnvelopeProbeError("raw PhysX robot-root state is non-finite")

            rigid_by_env: list[list[dict[str, Any]]] = [
                [] for _ in range(EXACT_NUM_ENVS)
            ]
            for object_index, (object_name, rigid_object) in enumerate(
                rigid_object_items
            ):
                transform = (
                    rigid_object.root_physx_view.get_transforms().detach().clone()
                )
                velocity = (
                    rigid_object.root_physx_view.get_velocities().detach().clone()
                )
                if tuple(transform.shape) != (EXACT_NUM_ENVS, 7) or tuple(
                    velocity.shape
                ) != (EXACT_NUM_ENVS, 6):
                    raise DualEnvelopeProbeError(
                        f"raw PhysX rigid object {object_name!r} shape drifted"
                    )
                object_origins = env_origins.to(
                    device=transform.device,
                    dtype=transform.dtype,
                )
                position_local = transform[:, :3] - object_origins
                if not torch.all(torch.isfinite(transform)) or not torch.all(
                    torch.isfinite(velocity)
                ):
                    raise DualEnvelopeProbeError(
                        f"raw PhysX rigid object {object_name!r} is non-finite"
                    )
                if not torch.all(position_local[:, 0] >= 15.0) or not torch.all(
                    position_local[:, 2] >= 15.0
                ):
                    raise DualEnvelopeProbeError(
                        f"scene rigid object {object_name!r} was not isolated"
                    )
                for env_id in range(EXACT_NUM_ENVS):
                    rigid_by_env[env_id].append(
                        {
                            "name": object_name,
                            "position_m": position_local[env_id].detach().cpu().tolist(),
                            "quaternion_xyzw": transform[
                                env_id, 3:7
                            ].detach().cpu().tolist(),
                            "linear_velocity_w_m_s": velocity[
                                env_id, :3
                            ].detach().cpu().tolist(),
                            "angular_velocity_w_rad_s": velocity[
                                env_id, 3:6
                            ].detach().cpu().tolist(),
                        }
                    )

            joint_pos_cpu = joint_pos.detach().cpu()
            joint_vel_cpu = joint_vel.detach().cpu()
            joint_target_cpu = joint_target.detach().cpu()
            root_position_cpu = root_position_local.detach().cpu()
            root_transform_cpu = root_transform.detach().cpu()
            root_velocity_cpu = root_velocity.detach().cpu()
            snapshots = []
            for env_id in range(EXACT_NUM_ENVS):
                snapshots.append(
                    _seal_full_state_snapshot(
                        {
                            "schema_version": FULL_STATE_SCHEMA_VERSION,
                            "joint_pos_rad": joint_pos_cpu[env_id].tolist(),
                            "joint_vel_rad_s": joint_vel_cpu[env_id].tolist(),
                            "joint_pos_target_rad": joint_target_cpu[env_id].tolist(),
                            "robot_root_origin_relative": {
                                "position_m": root_position_cpu[env_id].tolist(),
                                "quaternion_xyzw": root_transform_cpu[
                                    env_id, 3:7
                                ].tolist(),
                                "linear_velocity_w_m_s": root_velocity_cpu[
                                    env_id, :3
                                ].tolist(),
                                "angular_velocity_w_rad_s": root_velocity_cpu[
                                    env_id, 3:6
                                ].tolist(),
                            },
                            "scene_rigid_objects": rigid_by_env[env_id],
                        }
                    )
                )
            return snapshots

        qdes_initial = read_joint_position_target()
        initial_full_states = capture_full_states(
            q0_live_readback,
            qdot0_live_readback,
            qdes_initial,
        )

        action_term._record_physx_control_position_limit_diagnostic(
            joint_pos=q0_live_readback,
            joint_vel=qdot0_live_readback,
        )
        pre_diagnostic = action_term.consume_actual_joint_forbidden_diagnostic()
        observations = [
            {
                "env_id": row["env_id"],
                "joint": row["joint"],
                "side": row["side"],
                "condition": row["condition"],
                "q0_live_rad": float(
                    q0_live_readback[
                        row["env_id"], row["joint_index"]
                    ].detach().cpu()
                ),
                "qdot0_live_rad_s": float(
                    qdot0_live_readback[
                        row["env_id"], row["joint_index"]
                    ].detach().cpu()
                ),
                "initial_full_state": initial_full_states[row["env_id"]],
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
            q_tick = root_view.get_dof_positions().detach().clone()
            qdot_tick = root_view.get_dof_velocities().detach().clone()
            qdes_tick = read_joint_position_target()
            tick_full_states = capture_full_states(q_tick, qdot_tick, qdes_tick)
            for row, observation in zip(tape, observations):
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
                        "full_state": tick_full_states[env_id],
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
            "run_specific_live_limit_sha256": public_readback_sha256,
            "setter_no_mutation_sha256": setter_no_mutation_sha256,
            "mixed_live_limit_sha256": _sha256_bytes(
                mixed.contiguous().numpy().tobytes()
            ),
            "mixed_readback_exact": True,
            "public_contract_selected_joint_names": list(selected_names),
            "public_contract_selected_joint_indices": list(selected_indices),
            "public_contract_readback_sha256": public_readback_sha256,
            "public_hmech_matches_articulation_data_exact": True,
            "public_hctrl_matches_root_physx_readback_exact": True,
            "off_condition_disables_only_target_joint_hctrl": True,
            "isolated_scene_rigid_object_names": isolated_rigid_object_names,
            "full_system_pair_state_receipt_schema_version": (
                FULL_STATE_SCHEMA_VERSION
            ),
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
