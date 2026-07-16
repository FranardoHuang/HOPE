"""Default-off Isaac Lab 2.1 adapter for the lateral-balance perturbation source.

The pinned Isaac Lab ``Articulation.write_data_to_sim`` path submits external forces with
``position_data=None``.  In the pinned PhysX tensor API that means the link transform (link
origin), not the link centre of mass (COM).  This adapter therefore does not use Isaac Lab's
external-wrench command buffers to apply the perturbation.  Before every physics substep it:

* proves the complete built-in force/torque buffers are still zero and unowned;
* reads the current link poses and current local COM offsets;
* computes the current torso COM in WORLD coordinates;
* calls ``ArticulationView.apply_forces_and_torques_at_position`` directly with an explicit
  WORLD position and ``is_global=True``; and
* proves its private full-articulation command stayed byte-for-byte unchanged across the scene
  write.

The direct PhysX setter has no solver-consumed wrench getter.  Receipts therefore prove the exact
command and explicit COM position submitted at the synchronous setter/scene-write boundary, not
the wrench integrated by the solver.  An explicit opt-in trainer wrapper imports this candidate,
but launch remains forbidden until the strict full-scene solver-response and throughput gates pass.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .lateral_perturbation import (
    LateralApplicationLedgerRow,
    LateralPerturbationConfig,
    LateralPerturbationStep,
    LateralPulseScheduler,
    LateralWrenchPreflightReceipt,
    dispatch_lateral_wrench_fail_closed,
    frozen_lateral_training_contract,
)

_ISAACLAB_TAG = "v2.1.0"
_ISAACLAB_COMMIT = "21f7136325136ca3f6ca4e0a8125edffe5c24f7e"
_BACKEND_CONTRACT = {
    "schema_version": 2,
    "isaaclab_tag": _ISAACLAB_TAG,
    "isaaclab_commit": _ISAACLAB_COMMIT,
    "submit_api": "ArticulationView.apply_forces_and_torques_at_position",
    "position_data": "explicit_current_torso_com_world",
    "position_data_none_forbidden": True,
    "is_global": True,
    "built_in_buffer_policy": "full_zero_and_has_external_wrench_false_exclusive_guard",
    "private_command_policy": "full_articulation_world_force_torque_exact_readback",
    "full_articulation_buffer_overwrite": True,
    "solver_execution_readback_available": False,
}
_TRANSFORM_CONTRACT = {
    "schema_version": 2,
    "input": "world_wrench_at_torso_com",
    "backend": "world_wrench_at_explicit_world_position",
    "body_pose": "isaaclab_body_pos_w_and_wxyz_body_quat_w",
    "local_com": "ArticulationData.com_pos_b_device_normalized",
    "refresh": "before_every_physics_substep",
    "algorithm": "com_w_equals_link_origin_w_plus_quat_rotate_wxyz_com_b_v1",
}

_SCENE_ENTITY_CFG_TYPE = (
    "isaaclab.managers.scene_entity_cfg",
    "SceneEntityCfg",
)
_SCENE_ENTITY_CFG_FIELDS = (
    "name",
    "joint_names",
    "joint_ids",
    "fixed_tendon_names",
    "fixed_tendon_ids",
    "body_names",
    "body_ids",
    "object_collection_names",
    "object_collection_ids",
    "preserve_order",
)
_EVENT_TERM_CFG_TYPE = (
    "isaaclab.managers.manager_term_cfg",
    "EventTermCfg",
)
_EVENT_TERM_CFG_FIELDS = (
    "func",
    "params",
    "mode",
    "interval_range_s",
    "is_global_time",
    "min_step_count_between_reset",
)

_TRAINING_METRIC_SCHEMA_VERSION = 2
_TRAINING_COUNTER_METRICS = {
    "opportunity_count": "lateral_perturbation_opportunity_count",
    "eligible_opportunity_count": "lateral_perturbation_eligible_opportunity_count",
    "selected_start_count": "lateral_perturbation_selected_start_count",
    "nonzero_pulse_command_count": "lateral_perturbation_nonzero_pulse_command_count",
    "backend_accepted_pulse_count": "lateral_perturbation_backend_accepted_pulse_count",
    "sampled_normalized_impulse_abs_sum_mps": (
        "lateral_perturbation_sampled_normalized_impulse_abs_sum_mps"
    ),
    "commanded_normalized_impulse_abs_sum_mps": (
        "lateral_perturbation_commanded_normalized_impulse_abs_sum_mps"
    ),
    "backend_accepted_normalized_impulse_abs_sum_mps": (
        "lateral_perturbation_backend_accepted_normalized_impulse_abs_sum_mps"
    ),
}
_TRAINING_ABANDONED_COUNTERS = {
    "abandoned_uncommanded_impulse_abs_sum_mps": (
        "lateral_perturbation_reset_abandoned_uncommanded_impulse_abs_sum_mps",
        "lateral_perturbation_strike_abandoned_uncommanded_impulse_abs_sum_mps",
        "lateral_perturbation_window_abandoned_uncommanded_impulse_abs_sum_mps",
    ),
    "abandoned_not_backend_accepted_impulse_abs_sum_mps": (
        "lateral_perturbation_reset_abandoned_not_backend_accepted_impulse_abs_sum_mps",
        "lateral_perturbation_strike_abandoned_not_backend_accepted_impulse_abs_sum_mps",
        "lateral_perturbation_window_abandoned_not_backend_accepted_impulse_abs_sum_mps",
    ),
}


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_event_parameter_value(value: object, *, path: str) -> dict[str, object]:
    """Encode one EventManager parameter without repr-, coercion- or type-loss.

    Only exact JSON-safe scalar/container building blocks are supported.  Every node carries a
    type tag so ``True`` cannot alias ``1`` and a tuple cannot alias a list.  Opaque configuration
    objects are deliberately rejected instead of leaking process-specific ``repr`` addresses into
    a checkpoint identity.
    """

    value_type_identity = (type(value).__module__, type(value).__qualname__)
    if value_type_identity == _SCENE_ENTITY_CFG_TYPE:
        return _canonical_scene_entity_cfg(value, path=path)
    if callable(value):
        raise RuntimeError(f"{path} contains an unsupported callable parameter value")
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{path} contains a non-finite float")
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) in (list, tuple):
        return {
            "type": "list" if type(value) is list else "tuple",
            "items": [
                _canonical_event_parameter_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise RuntimeError(f"{path} contains a non-string mapping key")
        return {
            "type": "mapping",
            "entries": [
                {
                    "key": key,
                    "value": _canonical_event_parameter_value(
                        value[key], path=f"{path}.{key}"
                    ),
                }
                for key in sorted(value)
            ],
        }
    value_type = type(value)
    raise RuntimeError(
        f"{path} contains unsupported opaque parameter type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _exact_dataclass_field_names(
    value: object, *, expected: tuple[str, ...], path: str
) -> None:
    if not is_dataclass(value) or isinstance(value, type):
        raise RuntimeError(f"{path} is not the pinned dataclass/configclass shape")
    actual = tuple(field.name for field in fields(value))
    if actual != expected:
        raise RuntimeError(
            f"{path} field schema differs from the pinned Isaac Lab v2.1.0 contract"
        )


def _canonical_scene_entity_field(value: object, *, path: str) -> dict[str, object]:
    if type(value) is slice:
        parts = (value.start, value.stop, value.step)
        if not all(part is None or type(part) is int for part in parts):
            raise RuntimeError(f"{path} contains a non-integer slice component")
        return {
            "type": "slice",
            "start": _canonical_event_parameter_value(value.start, path=f"{path}.start"),
            "stop": _canonical_event_parameter_value(value.stop, path=f"{path}.stop"),
            "step": _canonical_event_parameter_value(value.step, path=f"{path}.step"),
        }
    return _canonical_event_parameter_value(value, path=path)


def _canonical_scene_entity_cfg(value: object, *, path: str) -> dict[str, object]:
    """Whitelist the exact pinned SceneEntityCfg, including resolved selector ids."""

    _exact_dataclass_field_names(value, expected=_SCENE_ENTITY_CFG_FIELDS, path=path)
    encoded_fields = {
        name: _canonical_scene_entity_field(
            getattr(value, name), path=f"{path}.{name}"
        )
        for name in _SCENE_ENTITY_CFG_FIELDS
    }
    name_value = getattr(value, "name")
    if type(name_value) is not str or not name_value:
        raise RuntimeError(f"{path}.name must be a non-empty exact string")
    if type(getattr(value, "preserve_order")) is not bool:
        raise RuntimeError(f"{path}.preserve_order must be an exact bool")
    return {
        "type": "pinned_isaaclab_configclass",
        "python_type": {
            "module": _SCENE_ENTITY_CFG_TYPE[0],
            "qualname": _SCENE_ENTITY_CFG_TYPE[1],
        },
        "isaaclab_tag": _ISAACLAB_TAG,
        "isaaclab_commit": _ISAACLAB_COMMIT,
        "fields": encoded_fields,
    }


def _callable_implementation_identity(func: object, *, term_name: str) -> dict[str, object]:
    """Bind a plain module-level Python function to exact source bytes, never an object repr."""

    if not callable(func):
        raise RuntimeError(f"EventManager term {term_name!r} func is not callable")
    if not inspect.isfunction(func):
        func_type = type(func)
        raise RuntimeError(
            f"EventManager term {term_name!r} uses unsupported callable implementation type "
            f"{func_type.__module__}.{func_type.__qualname__}"
        )
    if hasattr(func, "__wrapped__"):
        raise RuntimeError(
            f"EventManager term {term_name!r} uses a decorated callable; wrapper chain is unbound"
        )
    source_callable = func
    kind = "plain_module_python_function"
    module = getattr(source_callable, "__module__", None)
    qualname = getattr(source_callable, "__qualname__", None)
    if type(module) is not str or not module or type(qualname) is not str or not qualname:
        raise RuntimeError(
            f"EventManager term {term_name!r} callable exposes no stable module/qualname"
        )
    if "<locals>" in qualname:
        raise RuntimeError(
            f"EventManager term {term_name!r} callable is not a module-level function"
        )
    source_path_raw = inspect.getsourcefile(source_callable)
    if type(source_path_raw) is not str or not source_path_raw:
        raise RuntimeError(f"EventManager term {term_name!r} callable exposes no source file")
    source_path = Path(source_path_raw)
    if source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError(
            f"EventManager term {term_name!r} callable source is not a regular non-symlink file"
        )
    try:
        source_lines, first_line = inspect.getsourcelines(source_callable)
        module_bytes = source_path.read_bytes()
    except (OSError, IOError, TypeError) as exc:
        raise RuntimeError(
            f"EventManager term {term_name!r} callable source cannot be read exactly"
        ) from exc
    implementation_bytes = "".join(source_lines).encode("utf-8")
    closure = getattr(source_callable, "__closure__", None)
    closure_values = []
    if closure is not None:
        for index, cell in enumerate(closure):
            try:
                cell_value = cell.cell_contents
            except ValueError as exc:
                raise RuntimeError(
                    f"EventManager term {term_name!r} callable has an empty closure cell"
                ) from exc
            closure_values.append(
                _canonical_event_parameter_value(
                    cell_value,
                    path=f"EventManager term {term_name!r} callable closure[{index}]",
                )
            )
    return {
        "kind": kind,
        "module": module,
        "qualname": qualname,
        "source_first_line": int(first_line),
        "implementation_source_sha256": hashlib.sha256(implementation_bytes).hexdigest(),
        "module_file_sha256": hashlib.sha256(module_bytes).hexdigest(),
        "defaults": _canonical_event_parameter_value(
            getattr(source_callable, "__defaults__", None),
            path=f"EventManager term {term_name!r} callable defaults",
        ),
        "keyword_defaults": _canonical_event_parameter_value(
            getattr(source_callable, "__kwdefaults__", None),
            path=f"EventManager term {term_name!r} callable keyword defaults",
        ),
        "closure": closure_values,
    }


def _canonical_event_term_cfg(
    term_cfg: object, *, active_mode: str, term_name: str
) -> dict[str, object]:
    cfg_type_identity = (type(term_cfg).__module__, type(term_cfg).__qualname__)
    if cfg_type_identity != _EVENT_TERM_CFG_TYPE:
        cfg_type = type(term_cfg)
        raise RuntimeError(
            f"EventManager term {term_name!r} uses unsupported cfg type "
            f"{cfg_type.__module__}.{cfg_type.__qualname__}"
        )
    _exact_dataclass_field_names(
        term_cfg,
        expected=_EVENT_TERM_CFG_FIELDS,
        path=f"EventManager term {term_name!r} cfg",
    )
    mode = getattr(term_cfg, "mode")
    if type(mode) is not str or mode != active_mode:
        raise RuntimeError(
            f"EventManager term {term_name!r} cfg mode does not match active mode"
        )
    params = getattr(term_cfg, "params")
    if type(params) is not dict or not all(type(key) is str for key in params):
        raise RuntimeError(
            f"EventManager term {term_name!r} exposes no exact string-key parameter mapping"
        )
    return {
        "python_type": {
            "module": _EVENT_TERM_CFG_TYPE[0],
            "qualname": _EVENT_TERM_CFG_TYPE[1],
        },
        "callable_implementation": _callable_implementation_identity(
            getattr(term_cfg, "func"), term_name=term_name
        ),
        "behavior": {
            "mode": _canonical_event_parameter_value(
                mode, path=f"EventManager term {term_name!r} mode"
            ),
            "interval_range_s": _canonical_event_parameter_value(
                getattr(term_cfg, "interval_range_s"),
                path=f"EventManager term {term_name!r} interval_range_s",
            ),
            "is_global_time": _canonical_event_parameter_value(
                getattr(term_cfg, "is_global_time"),
                path=f"EventManager term {term_name!r} is_global_time",
            ),
            "min_step_count_between_reset": _canonical_event_parameter_value(
                getattr(term_cfg, "min_step_count_between_reset"),
                path=f"EventManager term {term_name!r} min_step_count_between_reset",
            ),
        },
        "parameters": {
            key: _canonical_event_parameter_value(
                params[key], path=f"EventManager term {term_name!r} parameter {key!r}"
            )
            for key in sorted(params)
        },
    }


def isaac_lateral_backend_contract() -> dict[str, object]:
    """Return a copy of the reviewed direct-PhysX application contract."""

    return dict(_BACKEND_CONTRACT)


def isaac_lateral_backend_identity_sha256() -> str:
    return _canonical_sha256(_BACKEND_CONTRACT)


def isaac_lateral_transform_contract() -> dict[str, object]:
    """Return a copy of the explicit torso-COM position contract."""

    return dict(_TRANSFORM_CONTRACT)


def isaac_lateral_transform_identity_sha256() -> str:
    return _canonical_sha256(_TRANSFORM_CONTRACT)


def isaac_lateral_event_term_manifest(event_manager: object) -> list[dict[str, object]]:
    """Bind every active term and its exact supported parameters; reject interval writers."""

    active = getattr(event_manager, "active_terms", None)
    get_term_cfg = getattr(event_manager, "get_term_cfg", None)
    if not isinstance(active, dict) or not callable(get_term_cfg):
        raise RuntimeError("EventManager exposes no auditable active-term contract")
    manifest: list[dict[str, object]] = []
    for mode in sorted(active):
        if mode not in ("startup", "reset"):
            raise RuntimeError(
                f"lateral runtime supports only startup/reset EventManager modes, got {mode!r}"
            )
        names = active[mode]
        if not isinstance(names, list) or not all(type(name) is str and name for name in names):
            raise RuntimeError("EventManager active-term names have an unexpected shape")
        if len(set(names)) != len(names):
            raise RuntimeError(f"EventManager mode {mode!r} contains duplicate term names")
        for name in names:
            term_cfg = get_term_cfg(name)
            manifest.append(
                {
                    "mode": mode,
                    "name": name,
                    "term_cfg": _canonical_event_term_cfg(
                        term_cfg, active_mode=mode, term_name=name
                    ),
                }
            )
    return manifest


def isaac_lateral_event_term_manifest_identity_sha256(
    event_term_manifest: list[dict[str, object]],
) -> str:
    """Return the canonical identity used both by attach and live drift checks."""

    return _canonical_sha256({"schema_version": 2, "terms": event_term_manifest})


def isaac_lateral_training_hard_contract(
    *,
    cell: str,
    cfg: LateralPerturbationConfig,
    event_term_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Bind an enabled trainer cell to its scheduler, adapter and metric identities."""

    manifest_payload = [dict(row) for row in event_term_manifest]
    # ``allow_nan=False`` also defends this public seam from a caller that bypasses the manifest
    # builder.  The round trip gives the returned contract a detached, JSON-only copy.
    try:
        manifest_payload = json.loads(
            json.dumps(
                manifest_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("active EventManager manifest is not canonical JSON") from exc

    return {
        **frozen_lateral_training_contract(cell=cell, cfg=cfg),
        "isaac_backend": isaac_lateral_backend_contract(),
        "isaac_backend_identity_sha256": isaac_lateral_backend_identity_sha256(),
        "world_to_backend_transform": isaac_lateral_transform_contract(),
        "world_to_backend_transform_identity_sha256": (
            isaac_lateral_transform_identity_sha256()
        ),
        "active_event_terms": manifest_payload,
        "active_event_terms_identity_sha256": (
            isaac_lateral_event_term_manifest_identity_sha256(manifest_payload)
        ),
        "interval_event_terms_present": False,
        "metrics_schema_version": _TRAINING_METRIC_SCHEMA_VERSION,
        "integer_metrics": [
            "opportunity_count",
            "eligible_opportunity_count",
            "selected_start_count",
            "nonzero_pulse_command_count",
            "backend_accepted_pulse_count",
            "zero_overwrite_step_count",
        ],
        "mass_metrics_kg": [
            "actual_total_mass_min_kg",
            "actual_total_mass_mean_kg",
            "actual_total_mass_max_kg",
        ],
        "impulse_metrics_mps": [
            "sampled_normalized_impulse_abs_sum_mps",
            "commanded_normalized_impulse_abs_sum_mps",
            "backend_accepted_normalized_impulse_abs_sum_mps",
            "abandoned_uncommanded_impulse_abs_sum_mps",
            "abandoned_not_backend_accepted_impulse_abs_sum_mps",
        ],
        "backend_accepted_metric_semantics": (
            "scheduler commit and synchronous direct-setter/scene-write submission accepted; "
            "never solver-consumed or solver-integrated evidence"
        ),
        "solver_execution_readback_available": False,
    }


def _require_tensor(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


def _host_all(condition: torch.Tensor, message: str) -> None:
    predicate = torch.all(condition)
    if not torch.equal(predicate, torch.ones_like(predicate, dtype=torch.bool)):
        raise RuntimeError(message)


def _quat_rotate_wxyz(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a BODY vector into WORLD using scalar-first unit quaternions."""

    q_vec = quat_wxyz[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


@dataclass(frozen=True)
class IsaacLateralSubstepReceipt:
    """Exact direct-setter evidence for one physics substep."""

    policy_step_token: int
    physics_substep_index: int
    commanded_force_w: torch.Tensor
    commanded_torque_w: torch.Tensor
    submitted_force_w_full: torch.Tensor
    submitted_torque_w_full: torch.Tensor
    submitted_position_w_full: torch.Tensor
    torso_com_position_w: torch.Tensor
    torso_local_com_position_b: torch.Tensor
    direct_physx_call_completed_synchronously: bool
    scene_write_completed_synchronously: bool
    private_command_readback_exact: bool
    built_in_wrench_buffers_zero_exact: bool
    solver_execution_readback_available: bool = False

    def clone(self) -> "IsaacLateralSubstepReceipt":
        return IsaacLateralSubstepReceipt(
            policy_step_token=self.policy_step_token,
            physics_substep_index=self.physics_substep_index,
            commanded_force_w=self.commanded_force_w.clone(),
            commanded_torque_w=self.commanded_torque_w.clone(),
            submitted_force_w_full=self.submitted_force_w_full.clone(),
            submitted_torque_w_full=self.submitted_torque_w_full.clone(),
            submitted_position_w_full=self.submitted_position_w_full.clone(),
            torso_com_position_w=self.torso_com_position_w.clone(),
            torso_local_com_position_b=self.torso_local_com_position_b.clone(),
            direct_physx_call_completed_synchronously=(
                self.direct_physx_call_completed_synchronously
            ),
            scene_write_completed_synchronously=self.scene_write_completed_synchronously,
            private_command_readback_exact=self.private_command_readback_exact,
            built_in_wrench_buffers_zero_exact=self.built_in_wrench_buffers_zero_exact,
            solver_execution_readback_available=self.solver_execution_readback_available,
        )


@dataclass(frozen=True)
class IsaacLateralPolicyStepReceipt:
    """Episode/window/application reconciliation for one policy step."""

    step_token: int
    episode_indices: torch.Tensor
    episode_steps: torch.Tensor
    recovery_hold_eligible: torch.Tensor
    strike_window: torch.Tensor
    safe_window_remaining_steps: torch.Tensor
    scheduler_step: LateralPerturbationStep
    application_ledger: LateralApplicationLedgerRow
    physics_substeps: tuple[IsaacLateralSubstepReceipt, ...]
    reset_after_step: torch.Tensor
    reset_scene_write_observed: bool
    reset_live_wrench_zero_exact: bool
    async_backend_completion_synchronized: bool
    solver_execution_readback_available: bool = False

    def clone(self) -> "IsaacLateralPolicyStepReceipt":
        return IsaacLateralPolicyStepReceipt(
            step_token=self.step_token,
            episode_indices=self.episode_indices.clone(),
            episode_steps=self.episode_steps.clone(),
            recovery_hold_eligible=self.recovery_hold_eligible.clone(),
            strike_window=self.strike_window.clone(),
            safe_window_remaining_steps=self.safe_window_remaining_steps.clone(),
            scheduler_step=self.scheduler_step.clone(),
            application_ledger=self.application_ledger.clone(),
            physics_substeps=tuple(row.clone() for row in self.physics_substeps),
            reset_after_step=self.reset_after_step.clone(),
            reset_scene_write_observed=self.reset_scene_write_observed,
            reset_live_wrench_zero_exact=self.reset_live_wrench_zero_exact,
            async_backend_completion_synchronized=self.async_backend_completion_synchronized,
            solver_execution_readback_available=self.solver_execution_readback_available,
        )


@dataclass(frozen=True)
class _StagedIsaacWrench:
    source_token: object
    step_token: int
    force_w: torch.Tensor
    torque_w: torch.Tensor
    receipt: LateralWrenchPreflightReceipt


class IsaacLab21LateralWrenchAdapter:
    """Fail-closed direct-PhysX adapter for one live Isaac Lab 2.1 articulation.

    Isaac Lab's built-in wrench buffers are treated as a competing-writer detector and are never
    used for this command.  They must retain their original identities, remain zero across every
    body/environment, and keep ``has_external_wrench=False``.  The adapter owns a separate private
    full-articulation WORLD command and submits it at an explicit current torso COM position.
    """

    body_name = "torso_link"
    input_force_frame = "world"
    application_point = "center_of_mass"
    full_batch_overwrite = True
    inactive_zero_overwrite = True
    preflight_side_effect_free = True
    commit_failure_is_terminal = True
    discard_is_noexcept = True
    world_to_backend_transform_identity_sha256 = isaac_lateral_transform_identity_sha256()
    application_backend_identity_sha256 = isaac_lateral_backend_identity_sha256()

    def __init__(
        self,
        robot: object,
        *,
        synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        self._robot = robot
        body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
        if body_names.count(self.body_name) != 1:
            raise RuntimeError(
                f"Isaac lateral adapter requires exactly one {self.body_name!r}, got {body_names}"
            )
        self._body_index = body_names.index(self.body_name)
        self._num_envs = int(getattr(robot, "num_instances", -1))
        self._num_bodies = int(getattr(robot, "num_bodies", -1))
        if self._num_envs <= 0 or self._num_bodies != len(body_names):
            raise RuntimeError("Isaac articulation instance/body count is inconsistent")

        force_buffer = getattr(robot, "_external_force_b", None)
        torque_buffer = getattr(robot, "_external_torque_b", None)
        expected = (self._num_envs, self._num_bodies, 3)
        _require_tensor("robot._external_force_b", force_buffer, shape=expected)
        _require_tensor(
            "robot._external_torque_b",
            torque_buffer,
            shape=expected,
            dtype=force_buffer.dtype,
            device=force_buffer.device,
        )
        if not torch.is_floating_point(force_buffer):
            raise RuntimeError("Isaac external-wrench buffer must use a floating dtype")
        if getattr(robot, "has_external_wrench", None) is not False:
            raise RuntimeError("lateral adapter refuses an existing external-wrench owner")
        _host_all(force_buffer == 0.0, "lateral adapter refuses a non-zero built-in force buffer")
        _host_all(torque_buffer == 0.0, "lateral adapter refuses a non-zero built-in torque buffer")

        root_view = getattr(robot, "root_physx_view", None)
        apply = getattr(root_view, "apply_forces_and_torques_at_position", None)
        if not callable(apply):
            raise RuntimeError("pinned PhysX articulation view exposes no direct wrench setter")
        for name in ("get_masses",):
            if not callable(getattr(root_view, name, None)):
                raise RuntimeError(f"pinned PhysX articulation view exposes no {name}")
        all_indices = getattr(robot, "_ALL_INDICES", None)
        if not isinstance(all_indices, torch.Tensor):
            raise RuntimeError("Isaac articulation exposes no stable _ALL_INDICES tensor")
        if tuple(all_indices.shape) != (self._num_envs,) or all_indices.device != force_buffer.device:
            raise RuntimeError("Isaac articulation _ALL_INDICES shape/device mismatch")
        if all_indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("Isaac articulation _ALL_INDICES must use an integer dtype")
        _host_all(
            all_indices == torch.arange(self._num_envs, device=all_indices.device),
            "Isaac articulation _ALL_INDICES does not enumerate every environment exactly",
        )

        self._root_view = root_view
        self.application_backend_token = root_view
        self._force_buffer_token = force_buffer
        self._torque_buffer_token = torque_buffer
        self._all_indices_token = all_indices
        self._device = force_buffer.device
        self._dtype = force_buffer.dtype
        self._uses_default_synchronize = synchronize is None
        self._synchronize_fn = synchronize or self._default_synchronize
        self._pending: _StagedIsaacWrench | None = None
        self._committed_step_token: int | None = None
        self._private_force_w_full = torch.zeros_like(force_buffer)
        self._private_torque_w_full = torch.zeros_like(torque_buffer)
        self._expected_private_force_w_full = self._private_force_w_full.clone()
        self._expected_private_torque_w_full = self._private_torque_w_full.clone()
        self._last_submitted_position_w_full: torch.Tensor | None = None
        self._direct_submit_attempt_count = 0
        self._dirty_unknown = False
        self._terminal = False
        self._terminal_zero_submit_attempted = False
        self._terminal_zero_submit_succeeded = False
        self._assert_competing_writer_absent("adapter attach")

    @staticmethod
    def _default_synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dirty_unknown(self) -> bool:
        return self._dirty_unknown

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def terminal_zero_submit_succeeded(self) -> bool:
        return self._terminal_zero_submit_succeeded

    @property
    def async_completion_synchronization_trusted(self) -> bool:
        return self._device.type != "cuda" or self._uses_default_synchronize

    def _synchronize(self) -> None:
        self._synchronize_fn(self._device)

    def _mark_terminal(self, *, dirty_unknown: bool = True) -> None:
        if dirty_unknown:
            self._dirty_unknown = True
        self._terminal = True

    def _assert_competing_writer_absent(self, context: str, *, allow_terminal: bool = False) -> None:
        if self._terminal and not allow_terminal:
            raise RuntimeError("Isaac lateral backend is terminal")
        try:
            force = getattr(self._robot, "_external_force_b", None)
            torque = getattr(self._robot, "_external_torque_b", None)
            indices = getattr(self._robot, "_ALL_INDICES", None)
            if force is not self._force_buffer_token or torque is not self._torque_buffer_token:
                raise RuntimeError(f"built-in wrench-buffer identity changed during {context}")
            if indices is not self._all_indices_token:
                raise RuntimeError(f"Isaac _ALL_INDICES identity changed during {context}")
            if getattr(self._robot, "root_physx_view", None) is not self._root_view:
                raise RuntimeError(f"PhysX articulation-view identity changed during {context}")
            if getattr(self._robot, "has_external_wrench", None) is not False:
                raise RuntimeError(f"competing built-in wrench owner activated during {context}")
            self._synchronize()
            _host_all(force == 0.0, f"competing/non-torso force writer detected during {context}")
            _host_all(torque == 0.0, f"competing/non-torso torque writer detected during {context}")
        except BaseException:
            self._mark_terminal()
            raise

    def _assert_private_command_unchanged(self, context: str) -> None:
        try:
            _host_all(
                self._private_force_w_full == self._expected_private_force_w_full,
                f"private force command changed before {context}",
            )
            _host_all(
                self._private_torque_w_full == self._expected_private_torque_w_full,
                f"private torque command changed before {context}",
            )
        except BaseException:
            self._mark_terminal()
            raise

    def _replace_private_command(self, force_w_full: torch.Tensor, torque_w_full: torch.Tensor) -> None:
        """Checked full-buffer replacement; callers must never write a subset in place."""

        self._assert_competing_writer_absent("private command replacement")
        self._assert_private_command_unchanged("private command replacement")
        _require_tensor(
            "force_w_full",
            force_w_full,
            shape=(self._num_envs, self._num_bodies, 3),
            dtype=self._dtype,
            device=self._device,
        )
        _require_tensor(
            "torque_w_full",
            torque_w_full,
            shape=(self._num_envs, self._num_bodies, 3),
            dtype=self._dtype,
            device=self._device,
        )
        try:
            self._private_force_w_full.copy_(force_w_full)
            self._private_torque_w_full.copy_(torque_w_full)
            self._synchronize()
            _host_all(self._private_force_w_full == force_w_full, "private force readback mismatch")
            _host_all(self._private_torque_w_full == torque_w_full, "private torque readback mismatch")
        except BaseException:
            self._mark_terminal()
            raise
        self._expected_private_force_w_full = force_w_full.clone()
        self._expected_private_torque_w_full = torque_w_full.clone()

    def read_actual_total_mass_kg(self) -> torch.Tensor:
        """Read the current post-randomization PhysX body masses without casting."""

        self._assert_competing_writer_absent("mass read")
        masses = self._root_view.get_masses()
        _require_tensor(
            "root_physx_view.get_masses()",
            masses,
            shape=(self._num_envs, self._num_bodies),
            dtype=self._dtype,
            device=self._device,
        )
        _host_all(
            torch.isfinite(masses) & masses.gt(0.0),
            "post-randomization body masses must be finite and positive",
        )
        total = masses.sum(dim=-1)
        _host_all(torch.isfinite(total) & total.gt(0.0), "total mass must be finite and positive")
        return total

    def _current_world_application_positions(
        self, *, allow_terminal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_competing_writer_absent("COM position read", allow_terminal=allow_terminal)
        try:
            data = getattr(self._robot, "data", None)
            body_pos_w = _require_tensor(
                "robot.data.body_pos_w",
                getattr(data, "body_pos_w", None),
                shape=(self._num_envs, self._num_bodies, 3),
                dtype=self._dtype,
                device=self._device,
            )
            body_quat_w = _require_tensor(
                "robot.data.body_quat_w",
                getattr(data, "body_quat_w", None),
                shape=(self._num_envs, self._num_bodies, 4),
                dtype=self._dtype,
                device=self._device,
            )
            com_pos_b = _require_tensor(
                "robot.data.com_pos_b",
                getattr(data, "com_pos_b", None),
                shape=(self._num_envs, self._num_bodies, 3),
                dtype=self._dtype,
                device=self._device,
            )
            _host_all(torch.isfinite(body_pos_w), "link origin positions must be finite")
            _host_all(torch.isfinite(body_quat_w), "link quaternions must be finite")
            _host_all(torch.isfinite(com_pos_b), "device-normalized local COM positions must be finite")
            torso_quat_w = body_quat_w[:, self._body_index]
            norm_sq = torch.sum(torso_quat_w * torso_quat_w, dim=-1)
            _host_all(
                torch.abs(norm_sq - 1.0).le(2.0e-4),
                "torso quaternion must be unit length before COM transform",
            )
            torso_local_com_b = com_pos_b[:, self._body_index]
            torso_com_w = body_pos_w[:, self._body_index] + _quat_rotate_wxyz(
                torso_quat_w, torso_local_com_b
            )
            positions_w = body_pos_w.clone()
            positions_w[:, self._body_index] = torso_com_w
            _host_all(
                torch.isfinite(torso_com_w),
                "derived WORLD torso COM positions must be finite",
            )
            _host_all(
                torch.isfinite(positions_w),
                "full direct-setter WORLD positions must be finite",
            )
            return positions_w, torso_com_w, torso_local_com_b
        except BaseException:
            self._mark_terminal()
            raise

    def preflight_world_wrench_at_body_com(
        self,
        *,
        step_token: int,
        total_mass_kg: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        preflight_token: object,
    ) -> LateralWrenchPreflightReceipt:
        if self._terminal:
            raise RuntimeError("Isaac lateral backend is terminal")
        if self._pending is not None:
            raise RuntimeError("Isaac lateral adapter already has a pending preflight")
        if type(step_token) is not int or step_token < 0:
            raise ValueError("step_token must be a non-negative plain int")
        self._assert_competing_writer_absent("preflight")
        self._assert_private_command_unchanged("preflight")
        _require_tensor(
            "total_mass_kg",
            total_mass_kg,
            shape=(self._num_envs,),
            dtype=self._dtype,
            device=self._device,
        )
        _require_tensor(
            "force_w",
            force_w,
            shape=(self._num_envs, 1, 3),
            dtype=self._dtype,
            device=self._device,
        )
        _require_tensor(
            "torque_w",
            torque_w,
            shape=(self._num_envs, 1, 3),
            dtype=self._dtype,
            device=self._device,
        )
        _host_all(torch.isfinite(force_w), "commanded WORLD force must be finite")
        _host_all(torch.isfinite(torque_w), "commanded WORLD torque must be finite")
        _host_all(force_w[:, :, 0] == 0.0, "WORLD-X force is forbidden")
        _host_all(force_w[:, :, 2] == 0.0, "WORLD-Z force is forbidden")
        _host_all(torque_w == 0.0, "explicit torque is forbidden")
        actual_mass = self.read_actual_total_mass_kg()
        _host_all(
            actual_mass == total_mass_kg,
            "dispatch total mass does not match current post-randomization PhysX mass",
        )
        scheduled_mask = force_w[:, 0, 1].ne(0.0)
        receipt = LateralWrenchPreflightReceipt(
            step_token=step_token,
            body_name=self.body_name,
            input_force_frame=self.input_force_frame,
            application_point=self.application_point,
            full_batch_overwrite=True,
            inactive_zero_overwrite=True,
            zero_torque=True,
            world_to_backend_transform_identity_sha256=(
                self.world_to_backend_transform_identity_sha256
            ),
            application_backend_identity_sha256=self.application_backend_identity_sha256,
            actual_total_mass_kg=actual_mass.clone(),
            commanded_force_w=force_w.clone(),
            commanded_torque_w=torque_w.clone(),
            scheduled_nonzero_force_mask=scheduled_mask.clone(),
            preflight_token=preflight_token,
        )
        self._pending = _StagedIsaacWrench(
            source_token=preflight_token,
            step_token=step_token,
            force_w=force_w.clone(),
            torque_w=torque_w.clone(),
            receipt=receipt,
        )
        return receipt

    def commit_preflighted_world_wrench_at_body_com(self, *, preflight_token: object) -> None:
        staged = self._pending
        if staged is None or staged.source_token is not preflight_token:
            raise RuntimeError("Isaac lateral commit received a stale/foreign preflight token")
        full_force = torch.zeros_like(self._private_force_w_full)
        full_torque = torch.zeros_like(self._private_torque_w_full)
        full_force[:, self._body_index] = staged.force_w[:, 0]
        full_torque[:, self._body_index] = staged.torque_w[:, 0]
        try:
            self._replace_private_command(full_force, full_torque)
        except BaseException:
            self._pending = None
            raise
        self._committed_step_token = staged.step_token
        self._pending = None
        return None

    def discard_preflighted_world_wrench_at_body_com(self, *, preflight_token: object) -> None:
        if self._pending is not None and self._pending.source_token is preflight_token:
            self._pending = None
        return None

    def _direct_submit(
        self,
        force_w_full: torch.Tensor,
        torque_w_full: torch.Tensor,
        positions_w_full: torch.Tensor,
        *,
        context: str,
        allow_terminal: bool = False,
    ) -> None:
        self._assert_competing_writer_absent(context, allow_terminal=allow_terminal)
        if not allow_terminal:
            self._assert_private_command_unchanged(context)
        try:
            # Preserve a trusted explicit position before the setter.  A setter may enqueue and
            # then throw, so any attempted call—not only a successful return—requires a terminal
            # zero overwrite at this exact position.
            self._last_submitted_position_w_full = positions_w_full.clone()
            if not allow_terminal:
                self._direct_submit_attempt_count += 1
            result = self._root_view.apply_forces_and_torques_at_position(
                force_data=force_w_full.reshape(-1, 3),
                torque_data=torque_w_full.reshape(-1, 3),
                position_data=positions_w_full.reshape(-1, 3),
                indices=self._all_indices_token,
                is_global=True,
            )
            if result is not None:
                raise RuntimeError("direct PhysX wrench setter returned a non-None result")
            self._synchronize()
            self._assert_competing_writer_absent(
                context + " post-submit", allow_terminal=allow_terminal
            )
            if not allow_terminal:
                self._assert_private_command_unchanged(context + " post-submit")
        except BaseException:
            self._mark_terminal()
            raise

    def apply_before_sim_substep(
        self, *, policy_step_token: int, physics_substep_index: int
    ) -> IsaacLateralSubstepReceipt:
        """Submit the frozen WORLD command at the current explicit torso COM."""

        if self._terminal:
            raise RuntimeError("Isaac lateral backend is terminal")
        if self._committed_step_token != policy_step_token:
            raise RuntimeError("substep apply has no matching committed policy-step command")
        if type(physics_substep_index) is not int or physics_substep_index < 0:
            raise ValueError("physics_substep_index must be a non-negative plain int")
        positions_w, torso_com_w, torso_local_com_b = self._current_world_application_positions()
        self._direct_submit(
            self._private_force_w_full,
            self._private_torque_w_full,
            positions_w,
            context=f"physics substep {physics_substep_index}",
        )
        return IsaacLateralSubstepReceipt(
            policy_step_token=policy_step_token,
            physics_substep_index=physics_substep_index,
            commanded_force_w=(
                self._private_force_w_full[:, self._body_index : self._body_index + 1].clone()
            ),
            commanded_torque_w=(
                self._private_torque_w_full[:, self._body_index : self._body_index + 1].clone()
            ),
            submitted_force_w_full=self._private_force_w_full.clone(),
            submitted_torque_w_full=self._private_torque_w_full.clone(),
            submitted_position_w_full=positions_w.clone(),
            torso_com_position_w=torso_com_w.clone(),
            torso_local_com_position_b=torso_local_com_b.clone(),
            direct_physx_call_completed_synchronously=(
                self.async_completion_synchronization_trusted
            ),
            scene_write_completed_synchronously=False,
            private_command_readback_exact=True,
            built_in_wrench_buffers_zero_exact=True,
        )

    def confirm_scene_write_completed(
        self, receipt: IsaacLateralSubstepReceipt
    ) -> IsaacLateralSubstepReceipt:
        if receipt.policy_step_token != self._committed_step_token:
            raise RuntimeError("scene-write receipt belongs to another policy step")
        self._assert_competing_writer_absent("scene write readback")
        self._assert_private_command_unchanged("scene write readback")
        return IsaacLateralSubstepReceipt(
            policy_step_token=receipt.policy_step_token,
            physics_substep_index=receipt.physics_substep_index,
            commanded_force_w=receipt.commanded_force_w.clone(),
            commanded_torque_w=receipt.commanded_torque_w.clone(),
            submitted_force_w_full=receipt.submitted_force_w_full.clone(),
            submitted_torque_w_full=receipt.submitted_torque_w_full.clone(),
            submitted_position_w_full=receipt.submitted_position_w_full.clone(),
            torso_com_position_w=receipt.torso_com_position_w.clone(),
            torso_local_com_position_b=receipt.torso_local_com_position_b.clone(),
            direct_physx_call_completed_synchronously=(
                receipt.direct_physx_call_completed_synchronously
            ),
            scene_write_completed_synchronously=(
                self.async_completion_synchronization_trusted
            ),
            private_command_readback_exact=True,
            built_in_wrench_buffers_zero_exact=True,
            solver_execution_readback_available=False,
        )

    def clear_before_reset_scene_write(self) -> None:
        """Submit explicit all-zero live command before the reset-only scene write."""

        zeros = torch.zeros_like(self._private_force_w_full)
        self._replace_private_command(zeros, zeros)
        positions_w, _, _ = self._current_world_application_positions()
        self._direct_submit(zeros, zeros, positions_w, context="reset clear")

    def assert_live_zero_after_reset_scene_write(self) -> None:
        self._assert_competing_writer_absent("reset scene write readback")
        self._assert_private_command_unchanged("reset scene write readback")
        _host_all(self._private_force_w_full == 0.0, "reset left a private force command alive")
        _host_all(self._private_torque_w_full == 0.0, "reset left a private torque command alive")

    def terminal_zero_clear_noexcept(self) -> bool:
        """Best-effort direct zero overwrite; always leaves the adapter terminal.

        A successful explicit end-of-rollout clear is a clean terminal state.  A clear after an
        earlier failure, or a failed clear, remains DIRTY/UNKNOWN and cannot publish success.
        """

        self._mark_terminal(dirty_unknown=False)
        if self._terminal_zero_submit_attempted:
            return self._terminal_zero_submit_succeeded
        self._terminal_zero_submit_attempted = True
        try:
            # Do not erase evidence from another writer.  A competing built-in buffer makes the
            # run terminal and is never overwritten by this probe.
            self._assert_competing_writer_absent("terminal clear", allow_terminal=True)
            if self._direct_submit_attempt_count == 0:
                self._terminal_zero_submit_succeeded = True
                return True
            # Check the prior private bytes before a clear, but never rely on a possibly corrupt
            # private buffer to construct the live zero command.  It is left untouched as
            # evidence; fresh local zeros are sent directly to PhysX.
            try:
                self._assert_private_command_unchanged("terminal clear")
            except BaseException:
                pass
            zeros = torch.zeros_like(self._private_force_w_full)
            try:
                positions_w, _, _ = self._current_world_application_positions(allow_terminal=True)
            except BaseException:
                if self._last_submitted_position_w_full is None:
                    raise
                positions_w = self._last_submitted_position_w_full.clone()
            self._direct_submit(
                zeros,
                zeros,
                positions_w,
                context="terminal zero overwrite",
                allow_terminal=True,
            )
            self._terminal_zero_submit_succeeded = True
        except BaseException:
            self._dirty_unknown = True
            self._terminal_zero_submit_succeeded = False
        return self._terminal_zero_submit_succeeded


class IsaacLateralPerturbationRuntimeHook:
    """Fail-closed wrapper around one ``ManagerBasedRLEnv`` step path.

    Probe callers retain complete per-substep receipts by default.  The trainer path explicitly
    selects ``retain_receipts=False`` and emits bounded scalar counters instead; retaining a full
    4096-environment receipt for every PPO step would otherwise grow memory without bound.
    """

    def __init__(
        self,
        env: object,
        cfg: LateralPerturbationConfig,
        *,
        enabled: bool = False,
        retain_receipts: bool = True,
        synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if type(retain_receipts) is not bool:
            raise ValueError("retain_receipts must be a bool")
        self.enabled = enabled
        self._retain_receipts = retain_receipts
        self._env = env
        self._cfg = cfg
        self._dirty_unknown = False
        self._terminal = False
        self._receipts: list[IsaacLateralPolicyStepReceipt] = []
        self._pending_training_metrics: dict[str, torch.Tensor] | None = None
        self._event_term_manifest: list[dict[str, object]] = []
        self._event_term_manifest_identity_sha256 = ""
        if not enabled:
            self._adapter = None
            self._scheduler = None
            return

        num_envs = int(getattr(env, "num_envs", -1))
        device = torch.device(getattr(env, "device", "cpu"))
        if num_envs <= 0:
            raise RuntimeError("runtime hook requires a vectorized environment")
        self._event_term_manifest = isaac_lateral_event_term_manifest(
            getattr(env, "event_manager", None)
        )
        self._event_term_manifest_identity_sha256 = (
            isaac_lateral_event_term_manifest_identity_sha256(
                self._event_term_manifest
            )
        )
        step_dt = float(getattr(env, "step_dt", float("nan")))
        if not math.isfinite(step_dt) or not math.isclose(
            step_dt, cfg.policy_dt_s, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError("environment step_dt does not match perturbation policy_dt_s")
        robot = env.scene["robot"]
        self._adapter = IsaacLab21LateralWrenchAdapter(robot, synchronize=synchronize)
        if self._adapter.num_envs != num_envs or self._adapter.device != device:
            raise RuntimeError("environment and articulation batch/device disagree")
        self._scheduler = LateralPulseScheduler(
            num_envs,
            cfg,
            device=device,
            require_application_ack=True,
        )
        self._num_envs = num_envs
        self._device = device
        self._episode_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        env_episode_steps = _require_tensor(
            "env.episode_length_buf",
            getattr(env, "episode_length_buf", None),
            shape=(num_envs,),
            dtype=torch.long,
            device=device,
        )
        _host_all(
            env_episode_steps == 0,
            "lateral runtime probe must attach immediately after a full environment reset",
        )
        self._step_token = 0

    @property
    def dirty_unknown(self) -> bool:
        return self._dirty_unknown or bool(
            self._adapter is not None and self._adapter.dirty_unknown
        )

    @property
    def event_term_manifest(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._event_term_manifest]

    @property
    def event_term_manifest_identity_sha256(self) -> str:
        return self._event_term_manifest_identity_sha256

    def _assert_event_term_contract_unchanged(self) -> None:
        live = isaac_lateral_event_term_manifest(
            getattr(self._env, "event_manager", None)
        )
        live_sha256 = isaac_lateral_event_term_manifest_identity_sha256(live)
        if live_sha256 != self._event_term_manifest_identity_sha256:
            raise RuntimeError(
                "active EventManager term config drifted after lateral runtime attach"
            )

    @property
    def terminal_zero_submit_succeeded(self) -> bool:
        return bool(
            self._adapter is not None and self._adapter.terminal_zero_submit_succeeded
        )

    @property
    def terminal(self) -> bool:
        return self._terminal or bool(
            self._adapter is not None and self._adapter.terminal
        )

    def _command_terms(self) -> tuple[object, object]:
        manager = getattr(self._env, "command_manager", None)
        get_term = getattr(manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError("environment command manager exposes no get_term")
        return get_term("motion"), get_term("racket_target")

    def _derive_windows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion, target = self._command_terms()
        if bool(getattr(motion, "event_timing_enabled", False)):
            raise RuntimeError("lateral runtime v1 does not infer safe windows for event-driven T1")
        hold = getattr(motion, "in_hold")
        if callable(hold):
            hold = hold()
        hold = _require_tensor(
            "motion.in_hold",
            hold,
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        strike = _require_tensor(
            "racket_target.strike_window",
            getattr(target, "strike_window", None),
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        pre_strike = _require_tensor(
            "racket_target.pre_strike",
            getattr(target, "pre_strike", None),
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        post_strike = ~pre_strike & ~strike
        eligible = (hold | post_strike) & ~strike
        hold_counter = _require_tensor(
            "motion.hold_counter",
            getattr(motion, "hold_counter", None),
            shape=(self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        hold_remaining = torch.maximum(hold_counter, hold.to(dtype=torch.long))
        time_steps = _require_tensor(
            "motion.time_steps",
            getattr(motion, "time_steps", None),
            shape=(self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        motion_asset = getattr(motion, "motion", None)
        if bool(getattr(motion, "_multiseg", False)):
            clip_id = _require_tensor(
                "motion.clip_id",
                getattr(motion, "clip_id", None),
                shape=(self._num_envs,),
                dtype=torch.long,
                device=self._device,
            )
            seg_start = getattr(motion_asset, "seg_start", None)
            seg_len = getattr(motion_asset, "seg_len", None)
            if not isinstance(seg_start, torch.Tensor) or not isinstance(seg_len, torch.Tensor):
                raise RuntimeError("multi-clip motion exposes no segment bounds")
            post_remaining = seg_start[clip_id] + seg_len[clip_id] - time_steps
        else:
            total = int(getattr(motion_asset, "time_step_total", -1))
            if total <= 0:
                raise RuntimeError("single-clip motion exposes no positive time_step_total")
            post_remaining = total - time_steps
        post_remaining = post_remaining.clamp_min(0).to(dtype=torch.long)
        safe_remaining = torch.where(
            hold,
            hold_remaining,
            torch.where(post_strike, post_remaining, torch.zeros_like(post_remaining)),
        )
        return eligible, strike, safe_remaining

    def _prepare_policy_step(
        self,
    ) -> tuple[
        LateralPerturbationStep,
        LateralApplicationLedgerRow,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self._scheduler is not None and self._adapter is not None
        _host_all(
            getattr(self._env, "episode_length_buf") == self._episode_steps,
            "environment episode clock diverged from lateral runtime ledger",
        )
        eligible, strike, safe = self._derive_windows()
        result = self._scheduler.step(
            step_token=self._step_token,
            episode_indices=self._episode_indices,
            episode_steps=self._episode_steps,
            recovery_hold_eligible=eligible,
            strike_window=strike,
            safe_window_remaining_steps=safe,
        )
        total_mass = self._adapter.read_actual_total_mass_kg()
        application = dispatch_lateral_wrench_fail_closed(
            scheduler=self._scheduler,
            result=result,
            total_mass_kg=total_mass,
            adapter=self._adapter,
        )
        return result, application, eligible, strike, safe

    def _terminal_guard_noexcept(self) -> None:
        self._dirty_unknown = True
        self._terminal = True
        if self._adapter is not None:
            self._adapter.terminal_zero_clear_noexcept()

    def terminate_lateral_wrench_noexcept(self) -> bool:
        """Idempotently terminalize the hook and best-effort overwrite our live command with zero.

        A probe must call this immediately after its last environment step and before cloning,
        hashing, validating or publishing receipts.  The method never raises; ``False`` means the
        caller must preserve evidence and publish no success artifact.
        """

        if not self.enabled:
            self._terminal = True
            return True
        assert self._adapter is not None
        if self.dirty_unknown:
            self._terminal_guard_noexcept()
            return False
        self._terminal = True
        succeeded = self._adapter.terminal_zero_clear_noexcept()
        if not succeeded or self._adapter.dirty_unknown:
            self._dirty_unknown = True
            return False
        return True

    def step(self, action: Any) -> Any:
        """Execute one explicit probe step, or terminate and zero on any validation error."""

        if not self.enabled:
            return self._env.step(action)
        if self.dirty_unknown:
            raise RuntimeError("lateral runtime hook is DIRTY/UNKNOWN")
        if self.terminal:
            raise RuntimeError("lateral runtime hook is terminal")
        assert self._adapter is not None and self._scheduler is not None

        scene = getattr(self._env, "scene", None)
        original_write = getattr(scene, "write_data_to_sim", None)
        had_instance_override = bool(
            hasattr(scene, "__dict__") and "write_data_to_sim" in scene.__dict__
        )
        old_instance_override = (
            scene.__dict__.get("write_data_to_sim") if hasattr(scene, "__dict__") else None
        )
        dispatch_succeeded = False
        scene_hook_installed = False
        try:
            self._assert_event_term_contract_unchanged()
            result, application, eligible, strike, safe = self._prepare_policy_step()
            dispatch_succeeded = True
            if not callable(original_write):
                raise RuntimeError("environment scene exposes no write_data_to_sim")
            decimation = int(getattr(getattr(self._env, "cfg", None), "decimation", -1))
            if decimation <= 0:
                raise RuntimeError("environment cfg exposes no positive decimation")
            substeps: list[IsaacLateralSubstepReceipt] = []
            scene_write_count = 0
            reset_scene_write_observed = False
            reset_zero_exact = False

            def intercepted_scene_write(*args: object, **kwargs: object) -> object:
                nonlocal scene_write_count, reset_scene_write_observed, reset_zero_exact
                if scene_write_count < decimation:
                    row = self._adapter.apply_before_sim_substep(
                        policy_step_token=self._step_token,
                        physics_substep_index=scene_write_count,
                    )
                    write_result = original_write(*args, **kwargs)
                    if write_result is not None:
                        raise RuntimeError("scene.write_data_to_sim returned a non-None result")
                    substeps.append(self._adapter.confirm_scene_write_completed(row))
                else:
                    reset_mask = _require_tensor(
                        "env.reset_buf",
                        getattr(self._env, "reset_buf", None),
                        shape=(self._num_envs,),
                        dtype=torch.bool,
                        device=self._device,
                    )
                    if not bool(torch.any(reset_mask).detach().cpu()):
                        raise RuntimeError("reset-only scene write has an empty reset mask")
                    # The environment reset has already run.  Any non-torso/non-reset writer in
                    # that path is detected before this adapter changes its own private command.
                    self._adapter.clear_before_reset_scene_write()
                    write_result = original_write(*args, **kwargs)
                    if write_result is not None:
                        raise RuntimeError("reset scene.write_data_to_sim returned non-None")
                    self._adapter.assert_live_zero_after_reset_scene_write()
                    reset_scene_write_observed = True
                    reset_zero_exact = True
                scene_write_count += 1
                return None

            setattr(scene, "write_data_to_sim", intercepted_scene_write)
            scene_hook_installed = True
            output = self._env.step(action)
            self._assert_event_term_contract_unchanged()

            if scene_write_count != decimation + int(reset_scene_write_observed):
                raise RuntimeError("pinned ManagerBasedRLEnv exposed an unexpected scene-write count")
            if len(substeps) != decimation:
                raise RuntimeError("pinned ManagerBasedRLEnv skipped a physics scene write")
            if not isinstance(output, tuple) or len(output) < 4:
                raise RuntimeError("environment step returned no terminated/truncated tensors")
            terminated = _require_tensor(
                "terminated",
                output[2],
                shape=(self._num_envs,),
                dtype=torch.bool,
                device=self._device,
            )
            truncated = _require_tensor(
                "truncated",
                output[3],
                shape=(self._num_envs,),
                dtype=torch.bool,
                device=self._device,
            )
            reset = terminated | truncated
            reset_any = bool(torch.any(reset).detach().cpu())
            if reset_any != reset_scene_write_observed:
                raise RuntimeError("reset mask and reset scene-write evidence disagree")
            if reset_any and not reset_zero_exact:
                raise RuntimeError("reset scene write lacks exact zero evidence")
            expected_next_steps = self._episode_steps + 1
            expected_next_steps = torch.where(
                reset, torch.zeros_like(expected_next_steps), expected_next_steps
            )
            _host_all(
                getattr(self._env, "episode_length_buf") == expected_next_steps,
                "post-step environment episode clock does not reconcile",
            )
            if self._retain_receipts:
                self._receipts.append(
                    IsaacLateralPolicyStepReceipt(
                        step_token=self._step_token,
                        episode_indices=self._episode_indices.clone(),
                        episode_steps=self._episode_steps.clone(),
                        recovery_hold_eligible=eligible.clone(),
                        strike_window=strike.clone(),
                        safe_window_remaining_steps=safe.clone(),
                        scheduler_step=result.clone(),
                        application_ledger=application.clone(),
                        physics_substeps=tuple(row.clone() for row in substeps),
                        reset_after_step=reset.clone(),
                        reset_scene_write_observed=reset_scene_write_observed,
                        reset_live_wrench_zero_exact=(not reset_any or reset_zero_exact),
                        async_backend_completion_synchronized=(
                            self._adapter.async_completion_synchronization_trusted
                        ),
                        solver_execution_readback_available=False,
                    )
                )
            else:
                counters = self._scheduler.consume_counters()
                metrics: dict[str, torch.Tensor] = {}
                for public_name, counter_name in _TRAINING_COUNTER_METRICS.items():
                    value = counters[counter_name]
                    if value.ndim != 0:
                        raise RuntimeError(
                            f"lateral training counter {counter_name!r} is not scalar"
                        )
                    metrics[public_name] = value.detach().clone()
                for public_name, counter_names in _TRAINING_ABANDONED_COUNTERS.items():
                    metrics[public_name] = sum(
                        (counters[name] for name in counter_names),
                        torch.zeros((), dtype=torch.float64, device=self._device),
                    ).detach().clone()
                nonzero_force_count = application.backend_accepted_force_env_count
                metrics["zero_overwrite_step_count"] = nonzero_force_count.eq(0).to(
                    dtype=torch.long
                )
                mass = application.actual_total_mass_kg
                metrics["actual_total_mass_min_kg"] = mass.amin().detach().clone()
                metrics["actual_total_mass_mean_kg"] = mass.mean().detach().clone()
                metrics["actual_total_mass_max_kg"] = mass.amax().detach().clone()
                self._pending_training_metrics = metrics
            self._episode_steps.copy_(expected_next_steps)
            self._episode_indices.add_(reset.to(dtype=torch.long))
            self._step_token += 1
            return output
        except BaseException:
            # Once any previous/current command may have reached the direct PhysX setter, no
            # validation or environment error may escape without a terminal zero attempt.
            if dispatch_succeeded or self._step_token > 0 or self._adapter.dirty_unknown:
                self._terminal_guard_noexcept()
            raise
        finally:
            if scene_hook_installed:
                try:
                    if had_instance_override:
                        setattr(scene, "write_data_to_sim", old_instance_override)
                    else:
                        delattr(scene, "write_data_to_sim")
                except BaseException as exc:
                    if dispatch_succeeded:
                        self._terminal_guard_noexcept()
                    raise RuntimeError(
                        "failed to restore scene.write_data_to_sim after lateral probe step"
                    ) from exc

    def receipts(self) -> tuple[IsaacLateralPolicyStepReceipt, ...]:
        return tuple(row.clone() for row in self._receipts)

    def consume_counters(self) -> dict[str, torch.Tensor]:
        if self._scheduler is None:
            return {}
        return self._scheduler.consume_counters()

    def consume_training_metrics(self) -> dict[str, torch.Tensor]:
        """Return one bounded scalar metric row for a non-retaining trainer step."""

        if self._retain_receipts:
            raise RuntimeError(
                "consume_training_metrics requires retain_receipts=False"
            )
        if self._pending_training_metrics is None:
            return {}
        metrics = {
            name: value.detach().clone()
            for name, value in self._pending_training_metrics.items()
        }
        self._pending_training_metrics = None
        return metrics


class IsaacLateralPerturbationTrainingRuntime:
    """Trainer-facing adapter that injects one scalar activation row into ``extras['log']``.

    The enclosing training entrypoint owns the actual Gym wrapper.  This dependency-light object
    owns the fail-closed hook, validates the five-element Gym step tuple, refuses metric-key
    collisions and terminally zeroes the private wrench before closing the underlying environment.
    """

    _METRIC_PREFIX = "Metrics/lateral_perturbation_"

    def __init__(
        self,
        env: object,
        cfg: LateralPerturbationConfig,
        *,
        cell: str,
        synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        runtime_env = getattr(env, "unwrapped", env)
        self._env = env
        self._hook = IsaacLateralPerturbationRuntimeHook(
            runtime_env,
            cfg,
            enabled=True,
            retain_receipts=False,
            synchronize=synchronize,
        )
        self.hard_contract = isaac_lateral_training_hard_contract(
            cell=cell,
            cfg=cfg,
            event_term_manifest=self._hook.event_term_manifest,
        )
        self._closed = False

    @property
    def hook(self) -> IsaacLateralPerturbationRuntimeHook:
        return self._hook

    def step(self, action: Any) -> Any:
        try:
            output = self._hook.step(action)
            if not isinstance(output, tuple) or len(output) < 5:
                raise RuntimeError(
                    "lateral trainer requires a five-element Gym step tuple with extras"
                )
            extras_source = output[4]
            if not isinstance(extras_source, dict):
                raise RuntimeError("lateral trainer Gym extras must be a dict")
            log_source = extras_source.get("log", {})
            if not isinstance(log_source, dict):
                raise RuntimeError("lateral trainer extras['log'] must be a dict")
            # Never write our metric row into the ManagerBasedRLEnv-owned extras object.  Some
            # runtime versions reuse it; a shallow copy keeps prior-step keys from becoming a
            # false same-step collision and prevents the trainer adapter from owning env state.
            extras = dict(extras_source)
            log = dict(log_source)
            extras["log"] = log
            metrics = self._hook.consume_training_metrics()
            if not metrics:
                raise RuntimeError("lateral trainer produced no activation/application metrics")
            for name, value in metrics.items():
                key = self._METRIC_PREFIX + name
                if key in log:
                    raise RuntimeError(f"lateral trainer metric collision for {key!r}")
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise RuntimeError(f"lateral trainer metric {name!r} is not a scalar tensor")
                log[key] = value
            output_parts = list(output)
            output_parts[4] = extras
            return tuple(output_parts)
        except BaseException:
            self._hook.terminate_lateral_wrench_noexcept()
            raise

    def reset(self, *args: object, **kwargs: object) -> Any:
        """Reject a reset that would bypass the hook's episode/zero-write ledger."""

        del args, kwargs
        self._hook.terminate_lateral_wrench_noexcept()
        raise RuntimeError(
            "lateral trainer forbids out-of-band reset; ManagerBasedRLEnv must auto-reset "
            "inside the intercepted step path"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        zero_ok = self._hook.terminate_lateral_wrench_noexcept()
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        if not zero_ok or self._hook.dirty_unknown:
            raise RuntimeError(
                "lateral trainer could not prove terminal zero overwrite; run is invalid"
            )


__all__ = [
    "IsaacLab21LateralWrenchAdapter",
    "IsaacLateralPerturbationRuntimeHook",
    "IsaacLateralPerturbationTrainingRuntime",
    "IsaacLateralPolicyStepReceipt",
    "IsaacLateralSubstepReceipt",
    "isaac_lateral_backend_contract",
    "isaac_lateral_backend_identity_sha256",
    "isaac_lateral_event_term_manifest",
    "isaac_lateral_event_term_manifest_identity_sha256",
    "isaac_lateral_transform_contract",
    "isaac_lateral_transform_identity_sha256",
    "isaac_lateral_training_hard_contract",
]
