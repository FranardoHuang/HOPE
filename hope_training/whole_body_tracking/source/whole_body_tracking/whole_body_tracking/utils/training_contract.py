"""Immutable training/export execution-contract helpers.

This module deliberately has no Isaac, Torch, Hydra, or ONNX imports.  It is shared by the
training entry point and both export paths, and its duck-typed runtime extractor is covered by
dependency-light tests.  Schema 3 is the first schema that binds the policy's execution values
(joint/action order, decoder, nominal PD envelope, q-des limits, timing, body/reference order and
the exact actor layout) rather than only task-level configuration.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping


TRAINING_CONTRACT_SCHEMA_VERSION = 3
CHECKPOINT_CONTRACT_SCHEMA_KEY = "training_contract_schema_version"
CHECKPOINT_CONTRACT_SHA_KEY = "training_contract_sha256"
CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "training_contract_lineage_exact"
SCHEMA3_TASK_KEYS = (
    "racket_control_point",
    "racket_control_point_offset_wrist_m",
)

# Isaac Lab 2.1 passes ``ImplicitActuatorCfg.friction`` to PhysX as a dimensionless,
# load-dependent joint-friction coefficient.  It is *not* MuJoCo ``frictionloss`` (a constant
# Coulomb torque in N m).  Keep these strings in the immutable contract so a consumer cannot
# silently copy the same-looking numbers between physics backends and call that exact parity.
JOINT_FRICTION_BACKEND = "physx"
JOINT_FRICTION_SEMANTICS = "load_dependent_spatial_force_coefficient"
JOINT_FRICTION_UNITS = "dimensionless"

RUNTIME_EXECUTION_KEYS = (
    "articulation_joint_names",
    "action_joint_ids",
    "joint_names",
    "default_joint_pos",
    "action_scale",
    "joint_stiffness",
    "joint_damping",
    "joint_effort_limits",
    "joint_actuator_types",
    "joint_armature",
    "joint_friction_coefficients",
    "joint_velocity_limits",
    "joint_friction_backend",
    "joint_friction_semantics",
    "joint_friction_units",
    "qdes_joint_pos_limits",
    "action_use_default_offset",
    "qdes_clamp",
    "physics_step_dt_s",
    "policy_step_dt_s",
    "control_decimation",
    "actor_obs_contract",
    "actor_obs_mode",
    "actor_obs_total_dim",
    "actor_obs_term_names",
    "actor_obs_term_dims",
    "observation_history_lengths",
    "articulation_body_names",
    "body_names",
    "body_indices",
    "anchor_body_name",
    "anchor_body_index",
    "motion_segment_lengths",
    "motion_clip_fps",
    "motion_kinematics_contracts",
    "motion_kinematics_exact",
)


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _flat_floats(value, *, name: str, expected: int | None = None) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise RuntimeError(f"{name} must be a vector or a single nominal row, got {len(raw)} rows")
        raw = raw[0]
    out = [float(item) for item in raw]
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{name} has {len(out)} values, expected {expected}")
    if any(not math.isfinite(item) for item in out):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return out


def _nominal_row(value, *, name: str, expected: int) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return _flat_floats(raw, name=name, expected=expected)


def _joint_ids(value, count: int) -> list[int]:
    if isinstance(value, slice):
        if value != slice(None):
            return list(range(count))[value]
        return list(range(count))
    return [int(item) for item in _tolist(value)]


def _joint_actuator_types(robot, count: int) -> list[str]:
    """Resolve the instantiated actuator integration model for every articulation joint."""
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping) or not actuators:
        raise RuntimeError("robot.actuators is unavailable; actuator integration cannot be proven")
    result: list[str | None] = [None] * count
    for group_name, actuator in actuators.items():
        if not hasattr(actuator, "joint_indices") or not hasattr(actuator, "is_implicit_model"):
            raise RuntimeError(
                f"actuator group {group_name!r} lacks joint_indices/is_implicit_model"
            )
        kind = "implicit" if bool(actuator.is_implicit_model) else "explicit"
        for joint_id in _joint_ids(actuator.joint_indices, count):
            if not (0 <= joint_id < count):
                raise RuntimeError(
                    f"actuator group {group_name!r} has out-of-range joint id {joint_id}"
                )
            if result[joint_id] is not None:
                raise RuntimeError(
                    f"joint {joint_id} belongs to multiple actuator groups"
                )
            result[joint_id] = kind
    missing = [index for index, value in enumerate(result) if value is None]
    if missing:
        raise RuntimeError(f"actuator integration is unresolved for joints {missing}")
    return [str(value) for value in result]


def _policy_layout(env) -> tuple[list[str], list[int], int]:
    manager = env.observation_manager
    names = [str(name) for name in manager.active_terms["policy"]]
    raw_dims = manager.group_obs_term_dim["policy"]
    dims = []
    for dim in raw_dims:
        if isinstance(dim, (tuple, list)):
            if len(dim) != 1:
                raise RuntimeError(f"policy observation term has non-flat dimension {dim!r}")
            dim = dim[0]
        dims.append(int(dim))
    total = manager.group_obs_dim["policy"]
    if isinstance(total, (tuple, list)):
        if len(total) != 1:
            raise RuntimeError(f"policy observation group has non-flat dimension {total!r}")
        total = total[0]
    total = int(total)
    if len(names) != len(dims) or sum(dims) != total:
        raise RuntimeError(
            f"invalid policy layout: names={len(names)} dims={dims} total={total}"
        )
    return names, dims, total


def _observation_history_lengths(env, names: list[str]) -> list[int]:
    group_cfg = env.observation_manager.cfg.policy
    if group_cfg.history_length is not None:
        return [int(group_cfg.history_length)] * len(names)
    cfg_by_name = group_cfg.to_dict()
    out = []
    for name in names:
        raw = cfg_by_name[name]["history_length"]
        value = 0 if raw is None else int(raw)
        out.append(1 if value == 0 else value)
    return out


def runtime_execution_facts(env, actor_contract) -> dict:
    """Extract schema-3 execution facts from the instantiated, startup-initialized environment."""
    robot = env.scene["robot"]
    data = robot.data
    articulation_names = list(
        getattr(data, "joint_names", getattr(robot, "joint_names", ()))
    )
    if not articulation_names or len(set(articulation_names)) != len(articulation_names):
        raise RuntimeError("robot articulation joint names are empty or non-unique")
    n = len(articulation_names)

    action = env.action_manager.get_term("joint_pos")
    ids = _joint_ids(getattr(action, "_joint_ids", slice(None)), n)
    identity = list(range(n))
    if ids != identity:
        raise RuntimeError(
            "schema-3 ONNX requires identity action/articulation order because actions and baked "
            f"reference joints share one joint_names contract; got action_joint_ids={ids}"
        )
    joint_names = [articulation_names[index] for index in ids]
    action_cfg = getattr(action, "cfg", getattr(env.cfg.actions, "joint_pos", None))
    use_default_offset = bool(getattr(action_cfg, "use_default_offset", False))
    if not use_default_offset:
        raise RuntimeError(
            "schema-3 deploy decoder requires JointPositionAction use_default_offset=True"
        )

    if not hasattr(data, "default_joint_pos_nominal"):
        raise RuntimeError(
            "robot.data.default_joint_pos_nominal is missing; the startup nominal-pose capture "
            "must run before writing a schema-3 contract"
        )
    default_q = _flat_floats(
        data.default_joint_pos_nominal, name="default_joint_pos_nominal", expected=n
    )
    action_scale = _nominal_row(action._scale, name="action_scale", expected=n)
    kp = _nominal_row(data.default_joint_stiffness, name="joint_stiffness", expected=n)
    kd = _nominal_row(data.default_joint_damping, name="joint_damping", expected=n)
    effort = _nominal_row(data.joint_effort_limits, name="joint_effort_limits", expected=n)
    actuator_types = _joint_actuator_types(robot, n)
    armature = _nominal_row(
        data.default_joint_armature, name="joint_armature", expected=n
    )
    friction = _nominal_row(
        data.default_joint_friction_coeff,
        name="joint_friction_coefficients",
        expected=n,
    )
    velocity_limits = _nominal_row(
        data.joint_vel_limits, name="joint_velocity_limits", expected=n
    )
    if any(value <= 0.0 for value in action_scale):
        raise RuntimeError("action_scale must be finite and positive")
    if any(value <= 0.0 for value in kp) or any(value <= 0.0 for value in kd):
        raise RuntimeError("nominal joint stiffness/damping must be finite and positive")
    if any(value <= 0.0 for value in effort):
        raise RuntimeError("joint_effort_limits must be finite and positive")
    if any(value < 0.0 for value in armature):
        raise RuntimeError("joint_armature must be finite and non-negative")
    if any(value < 0.0 for value in friction):
        raise RuntimeError("joint_friction_coefficients must be finite and non-negative")
    if any(value <= 0.0 for value in velocity_limits):
        raise RuntimeError("joint_velocity_limits must be finite and positive")

    limits_raw = _tolist(data.soft_joint_pos_limits)
    if (
        limits_raw
        and isinstance(limits_raw[0], (list, tuple))
        and limits_raw[0]
        and isinstance(limits_raw[0][0], (list, tuple))
    ):
        # Isaac stores [num_envs, num_joints, 2]; select the nominal first environment.
        limits_raw = limits_raw[0]
    if len(limits_raw) != n:
        raise RuntimeError(f"soft_joint_pos_limits has {len(limits_raw)} joints, expected {n}")
    limits = []
    for index, pair in enumerate(limits_raw):
        if len(pair) != 2:
            raise RuntimeError(f"soft_joint_pos_limits[{index}] is not [lo, hi]")
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise RuntimeError(f"invalid soft q-des limits for {joint_names[index]}: {(lo, hi)}")
        limits.append([lo, hi])

    physics_dt = float(env.physics_dt)
    policy_dt = float(env.step_dt)
    decimation = int(env.cfg.decimation)
    if not (math.isfinite(physics_dt) and physics_dt > 0.0):
        raise RuntimeError(f"invalid physics dt {physics_dt!r}")
    if not (math.isfinite(policy_dt) and policy_dt > 0.0) or decimation <= 0:
        raise RuntimeError(f"invalid policy dt/decimation {policy_dt!r}/{decimation!r}")
    if not math.isclose(policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"policy dt {policy_dt:.17g} != physics dt {physics_dt:.17g} * {decimation}"
        )

    obs_names, obs_dims, obs_total = _policy_layout(env)
    history = _observation_history_lengths(env, obs_names)
    if actor_contract is not None:
        expected_layout = [(term.name, int(term.dim)) for term in actor_contract.terms]
        if list(zip(obs_names, obs_dims)) != expected_layout or obs_total != int(
            actor_contract.total_dim
        ):
            raise RuntimeError("actor contract object does not match the instantiated policy layout")

    motion = env.command_manager.get_term("motion")
    body_ids = [int(item) for item in _tolist(motion.body_indexes)]
    robot_body_names = list(getattr(robot, "body_names", ()))
    if (
        not robot_body_names
        or any(not str(name) for name in robot_body_names)
        or len(set(robot_body_names)) != len(robot_body_names)
    ):
        raise RuntimeError("robot body_names are unavailable or non-unique")
    if any(index < 0 or index >= len(robot_body_names) for index in body_ids):
        raise RuntimeError(f"resolved motion body indices are out of range: {body_ids}")
    body_names = [robot_body_names[index] for index in body_ids]
    configured_bodies = [str(name) for name in motion.cfg.body_names]
    if body_names != configured_bodies:
        raise RuntimeError(
            f"resolved motion body order {body_names} != configured order {configured_bodies}"
        )
    anchor_name = str(motion.cfg.anchor_body_name)
    if anchor_name not in body_names:
        raise RuntimeError(f"motion anchor {anchor_name!r} is absent from resolved body order")
    anchor_index = body_names.index(anchor_name)
    segment_lengths = [int(value) for value in _tolist(motion.motion.seg_len)]
    if not segment_lengths or any(value <= 0 for value in segment_lengths):
        raise RuntimeError(f"invalid motion segment lengths {segment_lengths}")
    clip_fps = [float(value) for value in motion.motion.per_clip_fps]
    if len(clip_fps) != len(segment_lengths):
        raise RuntimeError(
            "motion fps count does not match segments: "
            f"{len(clip_fps)} vs {len(segment_lengths)}"
        )
    policy_hz = 1.0 / policy_dt
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in clip_fps
    ):
        raise RuntimeError(
            f"motion clip fps {clip_fps} must all equal policy rate {policy_hz:.12g} Hz"
        )
    kinematics_contracts = [dict(item) for item in motion.motion.kinematics_contracts]
    if len(kinematics_contracts) != len(segment_lengths):
        raise RuntimeError(
            "motion kinematics-contract count does not match segments: "
            f"{len(kinematics_contracts)} vs {len(segment_lengths)}"
        )
    kinematics_exact = bool(motion.motion.kinematics_contract_exact)
    if kinematics_exact != all(bool(item.get("exact", False)) for item in kinematics_contracts):
        raise RuntimeError("motion kinematics exact flag disagrees with per-clip contracts")

    qdes_clamp = bool(
        getattr(action, "_clamp_enabled", getattr(env.cfg.actions.joint_pos, "clamp", False))
    )
    # R-a actor leg-reference masking leaves the 62-D command layout untouched, so without this
    # fact a masked run's contract is byte-indistinguishable from an unmasked one — and no
    # evaluator implements the mask. Detect it from the swapped observation term (train.py wires
    # task.actor_leg_ref_mask to mdp.generated_commands_actor_leg_masked). Key present only when
    # True: unmasked contracts and their sha256 stay byte-identical. Detection unwraps partials
    # and prefers the durable marker attribute stamped at the function definition, so a rename
    # or wrapper cannot silently drop the fact (a miss would defeat the evaluator's refusal).
    _cmd_term = getattr(
        getattr(getattr(env.cfg, "observations", None), "policy", None), "command", None
    )
    _cmd_func = getattr(_cmd_term, "func", None)
    while isinstance(_cmd_func, functools.partial):
        _cmd_func = _cmd_func.func
    actor_leg_ref_mask = bool(
        _cmd_func is not None
        and (
            getattr(_cmd_func, "actor_leg_ref_mask", False)
            or getattr(_cmd_func, "__name__", "") == "generated_commands_actor_leg_masked"
        )
    )
    facts = {
        "articulation_joint_names": articulation_names,
        "action_joint_ids": ids,
        "joint_names": joint_names,
        "default_joint_pos": default_q,
        "action_scale": action_scale,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "joint_effort_limits": effort,
        "joint_actuator_types": actuator_types,
        "joint_armature": armature,
        "joint_friction_coefficients": friction,
        "joint_velocity_limits": velocity_limits,
        "joint_friction_backend": JOINT_FRICTION_BACKEND,
        "joint_friction_semantics": JOINT_FRICTION_SEMANTICS,
        "joint_friction_units": JOINT_FRICTION_UNITS,
        "qdes_joint_pos_limits": limits,
        "action_use_default_offset": use_default_offset,
        "qdes_clamp": qdes_clamp,
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": decimation,
        "actor_obs_contract": getattr(actor_contract, "name", None),
        "actor_obs_mode": getattr(actor_contract, "obs_mode", None),
        "actor_obs_total_dim": obs_total,
        "actor_obs_term_names": obs_names,
        "actor_obs_term_dims": obs_dims,
        "observation_history_lengths": history,
        "articulation_body_names": robot_body_names,
        "body_names": body_names,
        "body_indices": body_ids,
        "anchor_body_name": anchor_name,
        "anchor_body_index": anchor_index,
        "motion_segment_lengths": segment_lengths,
        "motion_clip_fps": clip_fps,
        "motion_kinematics_contracts": kinematics_contracts,
        "motion_kinematics_exact": kinematics_exact,
    }
    if actor_leg_ref_mask:
        facts["actor_leg_ref_mask"] = True
    return facts


def validate_schema3_contract_structure(contract: Mapping) -> None:
    """Validate a schema-3 sidecar without promoting it to a formal-exact lineage.

    Schema 3 binds the instantiated execution contract even for deliberately diagnostic runs
    (for example, a causal continuation on an untagged legacy motion).  Those sidecars still need
    complete, internally consistent runtime facts and an adjacent checkpoint hash binding; the
    narrower :func:`validate_schema3_contract` adds the formal schema-2 motion requirement.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    try:
        schema = int(contract.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid training-contract schema version") from exc
    if schema != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported formal training-contract schema {schema}; expected "
            f"{TRAINING_CONTRACT_SCHEMA_VERSION}"
        )
    missing = [key for key in (*RUNTIME_EXECUTION_KEYS, *SCHEMA3_TASK_KEYS) if key not in contract]
    if missing:
        raise ValueError("schema-3 training contract missing execution facts: " + ", ".join(missing))
    for key in ("face_command_enabled", "motion_allow_legacy_link_origin_velocity"):
        if key in contract and not isinstance(contract[key], bool):
            raise ValueError(f"schema-3 {key} must be boolean when present")
    if "face_command_pairing" in contract and contract["face_command_pairing"] not in (
        "shared_plus_y",
        "legacy_signed_vs_A",
    ):
        raise ValueError("schema-3 face_command_pairing is invalid")
    if contract.get("actor_obs_contract") == "deploy_parity_face179":
        if contract.get("face_command_enabled") is not True:
            raise ValueError("formal face179 schema-3 contract requires face_command_enabled=true")
        if contract.get("face_command_pairing") != "shared_plus_y":
            raise ValueError("formal face179 schema-3 contract requires shared_plus_y")
        try:
            raw_face_signs = contract["mount_normal_sign_per_clip"]
            if any(isinstance(value, bool) for value in raw_face_signs):
                raise ValueError
            face_signs = tuple(float(value) for value in raw_face_signs)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
            ) from exc
        if face_signs != (1.0, -1.0):
            raise ValueError(
                "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
            )

    joint_names = contract["joint_names"]
    if not isinstance(joint_names, (list, tuple)) or not joint_names:
        raise ValueError("schema-3 joint_names must be a non-empty array")
    n = len(joint_names)
    if len(set(str(value) for value in joint_names)) != n:
        raise ValueError("schema-3 joint_names must be unique")
    raw_actuator_types = contract["joint_actuator_types"]
    actuator_types = (
        list(raw_actuator_types)
        if isinstance(raw_actuator_types, (list, tuple))
        else []
    )
    if len(actuator_types) != n or any(
        value not in ("implicit", "explicit") for value in actuator_types
    ):
        raise ValueError(
            "schema-3 joint_actuator_types must contain one implicit|explicit value per joint"
        )

    def finite_vector(key: str, *, positive: bool) -> None:
        value = contract[key]
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ValueError(f"schema-3 {key} must contain one value per joint")
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"schema-3 {key} must be numeric") from exc
        if any(
            not math.isfinite(item) or (item <= 0.0 if positive else item < 0.0)
            for item in numbers
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"schema-3 {key} must be finite and {qualifier}")

    finite_vector("joint_stiffness", positive=True)
    finite_vector("joint_damping", positive=True)
    finite_vector("joint_effort_limits", positive=True)
    finite_vector("joint_armature", positive=False)
    finite_vector("joint_friction_coefficients", positive=False)
    finite_vector("joint_velocity_limits", positive=True)
    if contract["joint_friction_backend"] != JOINT_FRICTION_BACKEND:
        raise ValueError("schema-3 joint_friction_backend must be physx")
    if contract["joint_friction_semantics"] != JOINT_FRICTION_SEMANTICS:
        raise ValueError(
            "schema-3 joint_friction_semantics does not describe Isaac/PhysX joint friction"
        )
    if contract["joint_friction_units"] != JOINT_FRICTION_UNITS:
        raise ValueError("schema-3 joint_friction_units must be dimensionless")

    articulation_body_names_raw = contract["articulation_body_names"]
    if not isinstance(articulation_body_names_raw, (list, tuple)):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    articulation_body_names = [str(value) for value in articulation_body_names_raw]
    if (
        not articulation_body_names
        or any(not value for value in articulation_body_names)
        or len(set(articulation_body_names)) != len(articulation_body_names)
    ):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    selected_body_names_raw = contract["body_names"]
    selected_body_indices = contract["body_indices"]
    if (
        not isinstance(selected_body_names_raw, (list, tuple))
        or not isinstance(selected_body_indices, (list, tuple))
        or len(selected_body_names_raw) != len(selected_body_indices)
        or not selected_body_names_raw
    ):
        raise ValueError("schema-3 selected body names/indices are malformed")
    selected_body_names = [str(value) for value in selected_body_names_raw]
    if any(not value for value in selected_body_names) or len(set(selected_body_names)) != len(
        selected_body_names
    ):
        raise ValueError("schema-3 selected body names must be non-empty and unique")
    try:
        parsed_body_indices = []
        for raw_index in selected_body_indices:
            index = int(raw_index)
            if isinstance(raw_index, bool) or float(raw_index) != float(index):
                raise ValueError
            parsed_body_indices.append(index)
        if any(
            index < 0 or index >= len(articulation_body_names)
            for index in parsed_body_indices
        ):
            raise IndexError
        resolved_selected = [
            articulation_body_names[index] for index in parsed_body_indices
        ]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("schema-3 body_indices are outside articulation_body_names") from exc
    if resolved_selected != selected_body_names:
        raise ValueError("schema-3 selected body names do not match articulation body indices")

    segment_lengths = contract["motion_segment_lengths"]
    clip_fps = contract["motion_clip_fps"]
    kinematics = contract["motion_kinematics_contracts"]
    if not isinstance(segment_lengths, (list, tuple)) or not segment_lengths:
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    try:
        parsed_segment_lengths = [int(value) for value in segment_lengths]
    except (TypeError, ValueError) as exc:
        raise ValueError("schema-3 motion_segment_lengths must be positive") from exc
    if any(value <= 0 for value in parsed_segment_lengths):
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    if (
        not isinstance(clip_fps, (list, tuple))
        or len(clip_fps) != len(segment_lengths)
        or not isinstance(kinematics, (list, tuple))
        or len(kinematics) != len(segment_lengths)
    ):
        raise ValueError("schema-3 motion fps/kinematics counts must match segments")
    try:
        policy_hz = 1.0 / float(contract["policy_step_dt_s"])
        parsed_fps = [float(value) for value in clip_fps]
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("schema-3 motion_clip_fps/policy_step_dt_s are invalid") from exc
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in parsed_fps
    ):
        raise ValueError("schema-3 every motion clip fps must equal the policy rate")
    expected_body_order = articulation_body_names
    clip_exact_flags = []
    for index, item in enumerate(kinematics):
        if not isinstance(item, Mapping):
            raise ValueError(f"schema-3 motion kinematics clip {index} must be an object")
        missing_item = [
            key
            for key in (
                "schema_version",
                "body_pos_point",
                "body_lin_vel_point",
                "body_names",
                "exact",
            )
            if key not in item
        ]
        if missing_item:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is missing "
                + ", ".join(missing_item)
            )
        if not isinstance(item["exact"], bool):
            raise ValueError(f"schema-3 motion kinematics clip {index} exact must be boolean")
        clip_exact = item["exact"] is True
        clip_exact_flags.append(clip_exact)
        try:
            item_schema = (
                None if item.get("schema_version") is None else int(item["schema_version"])
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid schema_version"
            )
        if item_schema not in (None, 1, 2):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has unsupported schema_version "
                f"{item_schema!r}"
            )
        pos_point = item.get("body_pos_point")
        vel_point = item.get("body_lin_vel_point")
        if pos_point not in (None, "link_origin") or vel_point not in (
            None,
            "link_origin",
            "center_of_mass",
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid point semantics"
            )
        raw_body_names = item.get("body_names")
        if raw_body_names is not None:
            if not isinstance(raw_body_names, (list, tuple)):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names must be an array or null"
                )
            item_body_names = [str(value) for value in raw_body_names]
            if (
                not item_body_names
                or any(not value for value in item_body_names)
                or len(set(item_body_names)) != len(item_body_names)
                or item_body_names != expected_body_order
            ):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names do not match the "
                    "runtime articulation"
                )
        status = item.get("status")
        if (not clip_exact) and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty"
            )
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty when present"
            )
        if clip_exact and (
            item_schema != 2
            or pos_point != "link_origin"
            or vel_point != "center_of_mass"
            or raw_body_names is None
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} claims exact without an exact "
                "schema-2 body order"
            )
        if not clip_exact and item_schema == 2:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is schema-2 but marked inexact"
            )
    motion_exact = contract["motion_kinematics_exact"]
    if not isinstance(motion_exact, bool) or motion_exact != all(clip_exact_flags):
        raise ValueError(
            "schema-3 motion_kinematics_exact disagrees with the per-clip contracts"
        )


def validate_schema3_contract(contract: Mapping) -> None:
    """Validate the formal-exact subset of the schema-3 execution contract."""

    validate_schema3_contract_structure(contract)
    if contract["motion_kinematics_exact"] is not True:
        raise ValueError("schema-3 formal lineage requires motion_kinematics_exact=true")


def checkpoint_claims_contract(checkpoint: Mapping) -> bool:
    """Return whether checkpoint infos claim any adjacent training-contract binding."""

    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    return isinstance(infos, Mapping) and any(
        key in infos
        for key in (
            CHECKPOINT_CONTRACT_SCHEMA_KEY,
            CHECKPOINT_CONTRACT_SHA_KEY,
            CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY,
        )
    )


def checkpoint_contract_binding(checkpoint: Mapping) -> tuple[int | None, str | None]:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    if not isinstance(infos, Mapping):
        return None, None
    schema_raw = infos.get(CHECKPOINT_CONTRACT_SCHEMA_KEY)
    digest_raw = infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
    try:
        schema = None if schema_raw is None else int(schema_raw)
    except (TypeError, ValueError):
        return None, None
    digest = None if digest_raw is None else str(digest_raw).strip().lower()
    return schema, digest


def checkpoint_contract_lineage_exact(checkpoint: Mapping) -> bool:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    value = infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY) if isinstance(infos, Mapping) else None
    return value in (True, 1, "1")


def require_checkpoint_contract_binding(
    checkpoint: Mapping, *, schema: int, sha256: str, require_lineage_exact: bool = True
) -> None:
    bound_schema, bound_sha = checkpoint_contract_binding(checkpoint)
    expected_sha = str(sha256).strip().lower()
    if bound_schema != schema or bound_sha != expected_sha:
        raise ValueError(
            "checkpoint is not bound to the adjacent training contract: "
            f"checkpoint schema/sha={bound_schema}/{bound_sha}, file={schema}/{expected_sha}"
        )
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ValueError("training-contract SHA256 is malformed")
    if require_lineage_exact and not checkpoint_contract_lineage_exact(checkpoint):
        infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
        lineage_exact = (
            infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
            if isinstance(infos, Mapping)
            else None
        )
        raise ValueError(
            "checkpoint contract binding is not exact-lineage eligible "
            f"({CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY}={lineage_exact!r})"
        )
