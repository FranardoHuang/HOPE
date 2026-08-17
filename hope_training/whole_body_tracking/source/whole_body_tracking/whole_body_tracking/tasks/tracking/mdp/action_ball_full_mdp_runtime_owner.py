"""Production all-owner ActionBall runtime boundary.

The live reveal lane is the Device-R05/ActionEpoch row transaction owned by
the bound leaves.  The retired portable-R05 compact executor is deliberately
absent: until policy-step orchestration is construction-bound to the live row
lane, both public reveal entry points fail before inspecting caller payloads.
No manifest requirement or private compatibility body stands in for runtime
integration.

The R10 join-snapshot provider is complete here.  It is privately constructed
by the runtime owner, retains owner-issued per-world reset/current roots, and
issues immutable snapshots only for an exact complete-step R10 boundary.  A
physical checkpoint adapter validates its exact class/API schema and opaque
snapshot identity; it must not reverse-pin this file's byte SHA because this
owner already pins the physical leaf (which would create an impossible SHA
cycle).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import struct
import threading
from typing import ClassVar, NoReturn, Optional, Tuple
import weakref

import torch


RUNTIME_OWNER_SCHEMA_VERSION = 1
RUNTIME_OWNER_KIND = "action_ball_full_mdp_runtime_owner_v1"
RUNTIME_DEPENDENCY_INVENTORY_KIND = (
    "action_ball_full_mdp_runtime_dependency_inventory_v1"
)
CHECKPOINT_JOIN_SNAPSHOT_KIND = (
    "action_ball_full_mdp_checkpoint_join_snapshot_v1"
)
CHECKPOINT_JOIN_STATE_KIND = (
    "action_ball_full_mdp_checkpoint_join_state_v1"
)
AUDIT_FRONTIER_CLAIM_KIND = (
    "action_ball_full_mdp_audit_frontier_claim_v1"
)
AUDIT_FRONTIER_RING_CAPACITY = 64

RUNTIME_INTEGRATED = False
POST_PHYSICS_INTEGRATED = False
SELECTED_RESET_INTEGRATED = False
PPO_DRAIN_BINDINGS_INTEGRATED = False
R10_SHARED_JOIN_PROVIDER_IMPLEMENTED = True
R10_SHARED_JOIN_PROVIDER_INTEGRATED = False
R11_AUDIT_CONSUMER_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

CHILD_COMPLETION_ORDER = (
    "motion",
    "racket",
    "r06_flight",
    "physical_ball",
)
GLOBAL_POISON_ORDER = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
    "r05",
)
SELECTED_RESET_AUTHORITY_API_METHODS = (
    "project_r05_true_reset",
    "require_owned_r05_true_reset_commit",
    "require_owned_r05_true_reset_abort",
)
SELECTED_RESET_COMMIT_PROOF_ORDER = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
)

# Exact mirror of ``action_ball_full_mdp_rewards.ORDERED_CONSUMERS``.  Keeping
# the names here makes the production top owner independently reject a
# reordered or shortened manager call graph.  The focused integration test
# cross-checks this tuple against the Reward module, so neither side can drift
# silently.
FULL_MDP_REWARD_ORDERED_CONSUMERS = (
    "r03:racket_position",
    "r03:racket_velocity",
    "r03:racket_normal",
    "r03:racket_position_coarse",
    "r03:racket_velocity_coarse",
    "r03:racket_normal_coarse",
    "r03:racket_position_precision",
    "r03:racket_velocity_precision",
    "r03:racket_normal_precision",
    "r03:paddle_center_proximity",
    "physical:physical_selected_contact",
    "r06:common_on_table_outcome",
    "r06:post_contact_placement_guidance",
    "r07:common_recovery_reward_v1",
)
FULL_MDP_REWARD_OWNER_ORDER = ("r03", "physical", "r06", "r07")
FULL_MDP_REWARD_OWNER_CONSUMERS = (
    ("r03", tuple(value.split(":", 1)[1] for value in FULL_MDP_REWARD_ORDERED_CONSUMERS[:10])),
    ("physical", ("physical_selected_contact",)),
    (
        "r06",
        ("common_on_table_outcome", "post_contact_placement_guidance"),
    ),
    ("r07", ("common_recovery_reward_v1",)),
)

REWARD_CHILD_PUBLISH_METHOD = "publish_full_mdp_pre_reward"
REWARD_CHILD_REQUIRE_PUBLISH_METHOD = "require_owned_full_mdp_pre_reward"
REWARD_CHILD_REQUIRE_PAYMENT_METHOD = "require_owned_full_mdp_reward_payment"
REWARD_CHILD_CLOSE_METHOD = "close_full_mdp_reward_cycle"
REWARD_CHILD_REQUIRE_CLOSE_METHOD = "require_owned_full_mdp_reward_close"

_SOURCE_ROOT = Path(__file__).resolve().parents[4]


class ActionBallFullMdpRuntimeOwnerError(RuntimeError):
    """The exact all-owner runtime contract was not satisfied."""


class ActionBallFullMdpRuntimeDependencyError(
    ActionBallFullMdpRuntimeOwnerError
):
    """One source/class/API pin required by the production DAG is absent."""


class ActionBallFullMdpRuntimePoisonedError(
    ActionBallFullMdpRuntimeOwnerError
):
    """A partial global transition permanently invalidated this owner."""


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionBallFullMdpRuntimeOwnerError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ActionBallFullMdpRuntimeOwnerError(
            "runtime authority is not canonical ASCII JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _plain_portable_dataclass(value: object, *, label: str) -> object:
    """Project a leaf-issued immutable audit row without accepting mappings."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is tuple:
        return [
            _plain_portable_dataclass(item, label=label) for item in value
        ]
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or parameters.frozen is not True:
            raise ActionBallFullMdpRuntimeOwnerError(
                f"{label} must be an immutable portable dataclass"
            )
        return {
            field.name: _plain_portable_dataclass(
                getattr(value, field.name),
                label=label,
            )
            for field in fields(value)
        }
    raise ActionBallFullMdpRuntimeOwnerError(
        f"{label} contains a non-portable value"
    )


def _portable_receipt_sha256(value: object, *, label: str) -> str:
    return _canonical_sha256(
        {
            "kind": "action_ball_full_mdp_leaf_audit_receipt_v1",
            "module": type(value).__module__,
            "class": type(value).__name__,
            "content": _plain_portable_dataclass(value, label=label),
        }
    )


def _plain_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ActionBallFullMdpRuntimeOwnerError(
            f"{label} must be a non-negative exact int"
        )
    return value


def _plain_positive_int(value: object, *, label: str) -> int:
    result = _plain_nonnegative_int(value, label=label)
    if result == 0:
        raise ActionBallFullMdpRuntimeOwnerError(
            f"{label} must be a positive exact int"
        )
    return result


def _same_device(left: object, right: object) -> bool:
    """Compare the exact canonical device spelling without importing torch."""

    try:
        left_name = str(left)
        right_name = str(right)
    except BaseException:
        return False
    return (
        type(left_name) is str
        and type(right_name) is str
        and bool(left_name)
        and left_name == right_name
    )


def _method_surface_sha256(
    source_bytes: bytes,
    *,
    class_name: str,
    method_names: Tuple[str, ...],
    field_names: Tuple[str, ...],
) -> str:
    """Hash exact source segments, independent of the Python AST version."""

    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"{class_name} source is not valid UTF-8 Python"
        ) from exc
    classes = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    if len(classes) != 1:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"expected exactly one {class_name} class"
        )
    by_name = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    fields_by_name = {
        node.target.id: node
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    missing = tuple(name for name in method_names if name not in by_name)
    missing_fields = tuple(
        name for name in field_names if name not in fields_by_name
    )
    if missing or missing_fields:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"{class_name} production API is missing methods={missing!r} "
            f"fields={missing_fields!r}"
        )
    # Canonical transport representation: the exact UTF-8 code-point segment
    # returned from the original bytes.  Newlines and indentation are retained
    # verbatim; no unparse, AST dump, or interpreter-version field participates.
    source_text = source_bytes.decode("utf-8")

    def source_segment(node: ast.AST) -> str:
        value = ast.get_source_segment(source_text, node, padded=False)
        if type(value) is not str:
            raise ActionBallFullMdpRuntimeDependencyError(
                f"{class_name} API source segment is unavailable"
            )
        return value

    payload = {
        "fields": tuple(
            (
                name,
                source_segment(fields_by_name[name]),
            )
            for name in field_names
        ),
        "methods": tuple(
            (
                name,
                source_segment(by_name[name]),
            )
            for name in method_names
        ),
    }
    return _canonical_sha256(payload)


def _class_ast_sha256(source_bytes: bytes, *, class_name: str) -> str:
    """Hash one exact class source segment across supported Python versions."""

    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"{class_name} source is not valid UTF-8 Python"
        ) from exc
    classes = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    if len(classes) != 1:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"expected exactly one {class_name} class"
        )
    source_text = source_bytes.decode("utf-8")
    segment = ast.get_source_segment(source_text, classes[0], padded=False)
    if type(segment) is not str:
        raise ActionBallFullMdpRuntimeDependencyError(
            f"{class_name} source segment is unavailable"
        )
    return _canonical_sha256(segment)


def _selected_reset_authority_api_sha256(source_bytes: bytes) -> str:
    """Portable hash of only the Device-R05 construction authority seam."""

    return _method_surface_sha256(
        source_bytes,
        class_name="ActionBallFullMdpRuntimeOwner",
        method_names=SELECTED_RESET_AUTHORITY_API_METHODS,
        field_names=(),
    )


@dataclass(frozen=True)
class ActionBallFullMdpRuntimeDependencySpec:
    role: str
    module_name: str
    class_name: str
    relative_source_path: str
    expected_source_sha256: Optional[str]
    expected_api_sha256: Optional[str]
    required_methods: Tuple[str, ...]
    required_fields: Tuple[str, ...] = ()

    def source_path(self) -> Path:
        return _SOURCE_ROOT / self.relative_source_path


@dataclass(frozen=True)
class ActionBallFullMdpRuntimeDependencyObservation:
    role: str
    module_name: str
    class_name: str
    relative_source_path: str
    expected_source_sha256: Optional[str]
    observed_source_sha256: str
    expected_api_sha256: Optional[str]
    observed_api_sha256: str
    frozen: bool
    blocker: Optional[str]


@dataclass(frozen=True)
class ActionBallFullMdpRuntimeDependencyInventory:
    schema_version: int
    kind: str
    rows: Tuple[ActionBallFullMdpRuntimeDependencyObservation, ...]
    child_completion_order: Tuple[str, ...]
    global_poison_order: Tuple[str, ...]
    runtime_integrated: bool
    post_physics_integrated: bool
    selected_reset_integrated: bool
    ppo_drain_bindings_integrated: bool
    r10_shared_join_provider_integrated: bool
    r11_audit_consumer_integrated: bool
    launch_authorized: bool
    diagnostic_unauthorized: bool
    content_sha256: str

    @property
    def blockers(self) -> Tuple[str, ...]:
        values = [row.blocker for row in self.rows if row.blocker is not None]
        for integrated, label in (
            (self.runtime_integrated, "runtime hot reveal is not integrated"),
            (
                self.post_physics_integrated,
                "post-physics producer callpoint is not integrated",
            ),
            (
                self.selected_reset_integrated,
                "selected-reset owner state machine awaits Device-R05 "
                "device-fault aggregation and real N=2 CPU/CUDA interop",
            ),
            (
                self.ppo_drain_bindings_integrated,
                "PPO drain exact leaf bindings are not integrated",
            ),
            (
                self.r10_shared_join_provider_integrated,
                "R10 envelope lacks the drain-audit third claim and "
                "owner-issued post-seal finalize; seven leaf-issued live "
                "mutation versions are not joined to the last-ACK drain "
                "highwaters",
            ),
            (
                self.r11_audit_consumer_integrated,
                "R11 exact audit consumer is not construction-bound",
            ),
        ):
            if not integrated:
                values.append(label)
        return tuple(values)

    @property
    def frozen(self) -> bool:
        return not self.blockers


_DEPENDENCY_SPECS = (
    ActionBallFullMdpRuntimeDependencySpec(
        role="r05_reveal_owner",
        module_name="action_ball_continuous_runtime_transaction",
        class_name="ContinuousRuntimeTransactionOwner",
        relative_source_path="action_ball_continuous_runtime_transaction.py",
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "poison_global_reveal_epoch",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="device_r05_owner",
        module_name="action_ball_continuous_runtime_transaction_device",
        class_name="DeviceR05Owner",
        relative_source_path=(
            "action_ball_continuous_runtime_transaction_device.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "project_owned_genesis_for_child",
            "require_owned_genesis_projection",
            "project_full_mdp_env_reset_binding",
            "require_owned_full_mdp_env_reset_binding",
            "bind_true_reset_authority",
            "prepare_many",
            "preview",
            "abort_prepared",
            "abort_preview",
            "stage_terminal",
            "arm_terminal",
            "commit_terminal",
            "record_child_completion",
            "poison_from_external_failure",
            "require_healthy",
            "prepare_true_reset_many",
            "require_owned_prepared_true_reset",
            "abort_true_reset_many",
            "commit_true_reset_many",
            "require_owned_true_reset_receipt",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="motion_child",
        module_name="whole_body_tracking.tasks.tracking.mdp.commands",
        class_name="MotionCommand",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/commands.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "bind_action_ball_continuous_motion_device_r05_reveal",
            "poison_global_reveal_epoch",
            "bind_action_ball_continuous_motion_selected_reset",
            "prepare_action_ball_continuous_motion_selected_reset",
            "arm_prevalidated_action_ball_continuous_motion_selected_reset",
            "abort_prevalidated_action_ball_continuous_motion_selected_reset",
            "commit_prevalidated_action_ball_continuous_motion_selected_reset",
            "require_owned_selected_reset_commit",
            "complete_action_ball_continuous_motion_selected_reset_after_r05",
            "require_owned_selected_reset_completion",
            "consume_owned_selected_reset_completion",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="racket_child",
        module_name="whole_body_tracking.tasks.tracking.mdp.hope_commands",
        class_name="RacketTargetCommand",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "bind_action_ball_full_mdp_racket_staging",
            "poison_global_reveal_epoch",
            "bind_action_ball_continuous_racket_selected_reset",
            "stage_action_ball_continuous_racket_selected_reset",
            "finalize_action_ball_continuous_racket_selected_reset",
            "abort_prevalidated_action_ball_continuous_racket_selected_reset",
            "commit_prevalidated_action_ball_continuous_racket_selected_reset",
            "require_owned_selected_reset_commit",
            "complete_action_ball_continuous_racket_selected_reset_after_r05",
            "require_owned_selected_reset_completion",
            "consume_owned_selected_reset_completion",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r06_child",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_landing_outcome_device"
        ),
        class_name="ActionBallLandingOutcomeDeviceCoordinator",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_landing_outcome_device.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "bind_physical_park_token_authority",
            "poison_global_reveal_epoch",
            "bind_device_r05_reset_owner",
            "prepare_selected_reset",
            "selected_reset_mask_capability",
            "require_owned_selected_reset_mask_capability",
            "arm_prevalidated_selected_reset",
            "commit_prevalidated_selected_reset",
            "require_owned_selected_reset_commit",
            "complete_selected_reset_after_r05",
            "abort_selected_reset",
            "poison_selected_reset",
            "require_owned_selected_reset_completion",
            "consume_owned_selected_reset_completion",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="physical_ball_child",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_physical_flight_device"
        ),
        class_name="ActionBallPhysicalFlightDeviceOwner",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_physical_flight_device.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "bind_r06_owner",
            "poison_global_reveal_epoch",
            "capture_post_physics_facts",
            "build_post_physics_publication",
            "publish_post_physics_to_r06",
            "retire_post_physics_to_r06",
            "bind_device_r05_reset_owner",
            "stage_selected_true_reset",
            "finalize_selected_true_reset",
            "abort_selected_true_reset",
            "poison_selected_reset",
            "prearm_selected_true_reset",
            "commit_prevalidated_selected_true_reset",
            "require_committed_selected_reset_park_token",
            "acknowledge_r06_selected_reset_commit",
            "complete_selected_true_reset_after_r05",
            "require_owned_selected_reset_completion",
            "consume_owned_selected_reset_completion",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r03_child",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_strike_fact_device"
        ),
        class_name="ActionBallStrikeFactDeviceCoordinator",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_strike_fact_device.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "prepare_pre_optimizer_ppo_boundary_device_pack",
            "abort_pre_optimizer_ppo_boundary_device_pack",
            "acknowledge_pre_optimizer_ppo_boundary",
            "poison_pre_optimizer_ppo_boundary",
            "require_owned_pre_optimizer_ppo_boundary_receipt",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r07_child",
        module_name="action_ball_continuous_recovery_device",
        class_name="ContinuousRecoveryDeviceCoordinator",
        relative_source_path="action_ball_continuous_recovery_device.py",
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "prepare_pre_optimizer_ppo_boundary_device_pack",
            "abort_pre_optimizer_ppo_boundary_device_pack",
            "acknowledge_pre_optimizer_ppo_boundary",
            "poison_pre_optimizer_ppo_boundary",
            "require_owned_pre_optimizer_ppo_boundary_receipt",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="ppo_drain_owner",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        ),
        class_name="ActionBallFullMdpPpoDrainOwner",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_full_mdp_ppo_drain.py"
        ),
        expected_source_sha256=(
            "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
        ),
        expected_api_sha256=(
            "b1bb7e370f44ee06cf655d3e8e0c009c734444e70379f0060899ba1ac54ce9fe"
        ),
        required_methods=(
            "prepare_pre_optimizer_ppo_boundary",
            "transfer_decode_pre_optimizer_ppo_boundary",
            "mark_optimizer_returned",
            "acknowledge_post_update",
            "poison_optimizer_failure",
            "require_exact_leaf_bindings",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="ppo_drain_checkpoint_contract",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        ),
        class_name="ActionBallFullMdpPpoDrainOwner",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_full_mdp_ppo_drain.py"
        ),
        expected_source_sha256=(
            "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
        ),
        expected_api_sha256=(
            "0d1e1949222e174f5f6501f52717e344a5cc5038706152d5b71b2bd3576ff125"
        ),
        required_methods=(
            "snapshot_for_checkpoint_boundary",
            "require_owned_checkpoint_snapshot",
            "restore_checkpoint",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="ppo_drain_runner_frontier_contract",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        ),
        class_name="ActionBallFullMdpPpoDrainOwner",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_full_mdp_ppo_drain.py"
        ),
        expected_source_sha256=(
            "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
        ),
        expected_api_sha256=(
            "c55464b27c4321e0ccbed43b1ed6be913bc30778056d0d89a72a6a0cd49a7e11"
        ),
        required_methods=("require_owned_runner_frontier_projection",),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="ppo_drain_leaf_ack_contract",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        ),
        class_name="LeafDevicePackAuthority",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_full_mdp_ppo_drain.py"
        ),
        expected_source_sha256=(
            "674d4d1ab6c7f1ac7f8b6a0c32e25003d2c5ee784921dedb5e43cb29fc35122e"
        ),
        expected_api_sha256=(
            "f759474e1576a151b37939d128b0ae2c58b02f4cf90007353b41fadad03d902d"
        ),
        required_methods=("require_owned_ack",),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="physical_checkpoint_adapter",
        module_name=(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_physical_flight_device"
        ),
        class_name="PhysicalFlightCheckpointAdapter",
        relative_source_path=(
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_physical_flight_device.py"
        ),
        expected_source_sha256=None,
        expected_api_sha256=None,
        required_methods=(
            "__init__",
            "mutation_version",
            "live_digest",
            "freeze",
            "export_sealed",
            "prepare_restore",
            "commit_restore",
            "rollback_restore",
            "poison_restore",
            "checkpoint_join_claims",
            "checkpoint_boundary_authority_sha256",
            "checkpoint_config_authority_sha256",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r10_checkpoint_contract",
        module_name="action_ball_full_mdp_checkpoint",
        class_name="CheckpointBoundary",
        relative_source_path="action_ball_full_mdp_checkpoint.py",
        expected_source_sha256=(
            "e1d4f27616c57888404a293695eb9dfd09a77e559e9ce4945f7dbfec0d1363b2"
        ),
        expected_api_sha256=(
            "64df39c5a25bfdf6b47de70fb2ef7a758f9ac91c8230169367325e2b7955da56"
        ),
        required_methods=(),
        required_fields=(
            "boundary_id_sha256",
            "update_index",
            "ppo_phase",
            "environment_step_phase",
            "rollout_storage_empty",
            "actor_frontier_sealed",
            "critic_frontier_sealed",
            "recurrent_frontier",
            "gae_in_flight",
            "optimizer_in_flight",
            "reset_in_flight",
            "worlds",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r10_checkpoint_finalizer_contract",
        module_name="action_ball_full_mdp_checkpoint",
        class_name="CheckpointPublicationFinalizationAuthority",
        relative_source_path="action_ball_full_mdp_checkpoint.py",
        expected_source_sha256=(
            "e1d4f27616c57888404a293695eb9dfd09a77e559e9ce4945f7dbfec0d1363b2"
        ),
        expected_api_sha256=(
            "2384305a1584159b1193de240599650e7a5634f9cfa7d39a2e2fea6268de384c"
        ),
        required_methods=(
            "__init__",
            "construction_bundle",
            "require_owned_construction_bundle",
            "finalize_publication",
            "validate",
            "require_finalization_receipt",
            "require_owned_finalization_receipt",
        ),
    ),
    ActionBallFullMdpRuntimeDependencySpec(
        role="r10_checkpoint_finalizer_construction_bundle",
        module_name="action_ball_full_mdp_checkpoint",
        class_name="CheckpointPublicationFinalizerConstructionBundle",
        relative_source_path="action_ball_full_mdp_checkpoint.py",
        expected_source_sha256=(
            "e1d4f27616c57888404a293695eb9dfd09a77e559e9ce4945f7dbfec0d1363b2"
        ),
        expected_api_sha256=(
            "5c94c90af265c8dd6c625164f44438490b5a81f6aba6f28a239cf29c85dfc95d"
        ),
        required_methods=(
            "__new__",
            "__setattr__",
            "__copy__",
            "__deepcopy__",
            "__reduce__",
            "schema_version",
            "kind",
            "registry_sha256",
            "callback_bindings",
            "live_mutation_bindings",
            "diagnostic_allow_missing_live_highwaters",
            "production_live_highwater_join",
            "distinct_identity_count",
            "content_sha256",
        ),
    ),
)


def action_ball_full_mdp_runtime_dependency_inventory(
) -> ActionBallFullMdpRuntimeDependencyInventory:
    rows = []
    for spec in _DEPENDENCY_SPECS:
        path = spec.source_path()
        if not path.is_file():
            raise ActionBallFullMdpRuntimeDependencyError(
                f"dependency source is missing: {spec.relative_source_path}"
            )
        source = path.read_bytes()
        observed_source = hashlib.sha256(source).hexdigest()
        api_error = None
        try:
            observed_api = _method_surface_sha256(
                source,
                class_name=spec.class_name,
                method_names=spec.required_methods,
                field_names=spec.required_fields,
            )
        except ActionBallFullMdpRuntimeDependencyError as exc:
            observed_api = "0" * 64
            api_error = str(exc)
        blocker = None
        if api_error is not None:
            blocker = f"{spec.role}: {api_error}"
        elif spec.expected_source_sha256 is None:
            blocker = f"{spec.role}: final source SHA is not frozen"
        elif observed_source != spec.expected_source_sha256:
            blocker = f"{spec.role}: source SHA differs"
        elif spec.expected_api_sha256 is None:
            blocker = f"{spec.role}: final class API SHA is not frozen"
        elif observed_api != spec.expected_api_sha256:
            blocker = f"{spec.role}: class API SHA differs"
        rows.append(
            ActionBallFullMdpRuntimeDependencyObservation(
                role=spec.role,
                module_name=spec.module_name,
                class_name=spec.class_name,
                relative_source_path=spec.relative_source_path,
                expected_source_sha256=spec.expected_source_sha256,
                observed_source_sha256=observed_source,
                expected_api_sha256=spec.expected_api_sha256,
                observed_api_sha256=observed_api,
                frozen=blocker is None,
                blocker=blocker,
            )
        )
    content = {
        "schema_version": RUNTIME_OWNER_SCHEMA_VERSION,
        "kind": RUNTIME_DEPENDENCY_INVENTORY_KIND,
        "rows": [
            {
                "role": row.role,
                "module_name": row.module_name,
                "class_name": row.class_name,
                "relative_source_path": row.relative_source_path,
                "expected_source_sha256": row.expected_source_sha256,
                "observed_source_sha256": row.observed_source_sha256,
                "expected_api_sha256": row.expected_api_sha256,
                "observed_api_sha256": row.observed_api_sha256,
                "frozen": row.frozen,
                "blocker": row.blocker,
            }
            for row in rows
        ],
        "child_completion_order": list(CHILD_COMPLETION_ORDER),
        "global_poison_order": list(GLOBAL_POISON_ORDER),
        "runtime_integrated": RUNTIME_INTEGRATED,
        "post_physics_integrated": POST_PHYSICS_INTEGRATED,
        "selected_reset_integrated": SELECTED_RESET_INTEGRATED,
        "ppo_drain_bindings_integrated": PPO_DRAIN_BINDINGS_INTEGRATED,
        "r10_shared_join_provider_integrated": (
            R10_SHARED_JOIN_PROVIDER_INTEGRATED
        ),
        "r11_audit_consumer_integrated": R11_AUDIT_CONSUMER_INTEGRATED,
        "launch_authorized": LAUNCH_AUTHORIZED,
        "diagnostic_unauthorized": DIAGNOSTIC_UNAUTHORIZED,
        "provider_api_schema_sha256": PROVIDER_API_SCHEMA_SHA256,
    }
    physical_path = next(
        spec.source_path()
        for spec in _DEPENDENCY_SPECS
        if spec.role == "physical_checkpoint_adapter"
    )
    try:
        physical_tree = ast.parse(physical_path.read_text(encoding="utf-8"))
        physical_provider_pin = next(
            ast.literal_eval(node.value)
            for node in physical_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                == "CHECKPOINT_JOIN_PROVIDER_API_SCHEMA_SHA256"
                for target in node.targets
            )
        )
    except (OSError, SyntaxError, ValueError, StopIteration):
        physical_provider_pin = None
    if physical_provider_pin != PROVIDER_API_SCHEMA_SHA256:
        blocker = (
            "physical_checkpoint_adapter: shared join provider API pin differs"
        )
        content["physical_checkpoint_provider_api_blocker"] = blocker
        physical_index = next(
            index
            for index, row in enumerate(rows)
            if row.role == "physical_checkpoint_adapter"
        )
        physical_row = rows[physical_index]
        if physical_row.blocker is None:
            physical_row = ActionBallFullMdpRuntimeDependencyObservation(
                role=physical_row.role,
                module_name=physical_row.module_name,
                class_name=physical_row.class_name,
                relative_source_path=physical_row.relative_source_path,
                expected_source_sha256=physical_row.expected_source_sha256,
                observed_source_sha256=physical_row.observed_source_sha256,
                expected_api_sha256=physical_row.expected_api_sha256,
                observed_api_sha256=physical_row.observed_api_sha256,
                frozen=False,
                blocker=blocker,
            )
            rows[physical_index] = physical_row
            content["rows"][physical_index]["frozen"] = False
            content["rows"][physical_index]["blocker"] = blocker
    return ActionBallFullMdpRuntimeDependencyInventory(
        schema_version=RUNTIME_OWNER_SCHEMA_VERSION,
        kind=RUNTIME_DEPENDENCY_INVENTORY_KIND,
        rows=tuple(rows),
        child_completion_order=CHILD_COMPLETION_ORDER,
        global_poison_order=GLOBAL_POISON_ORDER,
        runtime_integrated=RUNTIME_INTEGRATED,
        post_physics_integrated=POST_PHYSICS_INTEGRATED,
        selected_reset_integrated=SELECTED_RESET_INTEGRATED,
        ppo_drain_bindings_integrated=PPO_DRAIN_BINDINGS_INTEGRATED,
        r10_shared_join_provider_integrated=(
            R10_SHARED_JOIN_PROVIDER_INTEGRATED
        ),
        r11_audit_consumer_integrated=R11_AUDIT_CONSUMER_INTEGRATED,
        launch_authorized=LAUNCH_AUTHORIZED,
        diagnostic_unauthorized=DIAGNOSTIC_UNAUTHORIZED,
        content_sha256=_canonical_sha256(content),
    )


def _require_frozen_dependency_inventory(
) -> ActionBallFullMdpRuntimeDependencyInventory:
    inventory = action_ball_full_mdp_runtime_dependency_inventory()
    if not inventory.frozen:
        raise ActionBallFullMdpRuntimeDependencyError(
            "runtime dependency DAG is not frozen: "
            + "; ".join(inventory.blockers)
        )
    return inventory


def _load_frozen_dependency_modules(
    inventory: ActionBallFullMdpRuntimeDependencyInventory,
) -> dict[str, object]:
    """Import only the exact source/class DAG already frozen by inventory."""

    if (
        type(inventory) is not ActionBallFullMdpRuntimeDependencyInventory
        or not inventory.frozen
    ):
        raise ActionBallFullMdpRuntimeDependencyError(
            "dependency modules require one frozen inventory"
        )
    observed_by_role = {row.role: row for row in inventory.rows}
    modules: dict[str, object] = {}
    for spec in _DEPENDENCY_SPECS:
        row = observed_by_role.get(spec.role)
        if row is None or not row.frozen:
            raise ActionBallFullMdpRuntimeDependencyError(
                f"dependency role is not frozen: {spec.role}"
            )
        module = modules.get(spec.module_name)
        if module is None:
            try:
                module = importlib.import_module(spec.module_name)
            except BaseException as exc:
                raise ActionBallFullMdpRuntimeDependencyError(
                    f"dependency import failed: {spec.role}"
                ) from exc
            modules[spec.module_name] = module
        source_file = getattr(module, "__file__", None)
        dependency_class = getattr(module, spec.class_name, None)
        expected_path = spec.source_path().resolve()
        if (
            type(source_file) is not str
            or Path(source_file).resolve() != expected_path
            or not isinstance(dependency_class, type)
            or dependency_class.__module__ != module.__name__
            or hashlib.sha256(expected_path.read_bytes()).hexdigest()
            != row.expected_source_sha256
        ):
            raise ActionBallFullMdpRuntimeDependencyError(
                f"loaded dependency identity differs: {spec.role}"
            )
    return {
        spec.role: modules[spec.module_name]
        for spec in _DEPENDENCY_SPECS
    }


# ---- R10 shared join snapshot provider ---------------------------------


@dataclass(frozen=True)
class ActionBallFullMdpWorldResetJoinRow:
    world_id: int
    reset_generation: int
    reset_identity_sha256: str

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.world_id, label="world_id")
        _plain_positive_int(self.reset_generation, label="reset_generation")
        _require_sha256(
            self.reset_identity_sha256,
            label="reset_identity_sha256",
        )


@dataclass(frozen=True)
class ActionBallFullMdpTaskBallR06JoinRow:
    world_id: int
    reset_generation: int
    task_ball_r06_current_sha256: str

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.world_id, label="world_id")
        _plain_positive_int(self.reset_generation, label="reset_generation")
        _require_sha256(
            self.task_ball_r06_current_sha256,
            label="task_ball_r06_current_sha256",
        )


@dataclass(frozen=True)
class _ActionBallFullMdpCheckpointJoinState:
    schema_version: int
    kind: str
    sequence: int
    world_reset_rows: Tuple[ActionBallFullMdpWorldResetJoinRow, ...]
    task_ball_r06_rows: Tuple[ActionBallFullMdpTaskBallR06JoinRow, ...]
    per_world_reset_identity_sha256: str
    task_ball_r06_current_sha256: str
    canonical_sha256: str
    _runtime_owner_identity: object


@dataclass(frozen=True)
class _CheckpointJoinSnapshotPayload:
    provider_identity: object
    runtime_owner_identity: object
    boundary: object
    boundary_sha256: str
    state: _ActionBallFullMdpCheckpointJoinState
    drain_snapshot: object
    drain_frontier_sha256: str
    r10_audit_claim: object
    canonical_sha256: str


class ActionBallFullMdpCheckpointJoinSnapshot:
    """Opaque immutable provider-issued R10 join snapshot."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("checkpoint join snapshots are provider-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("checkpoint join snapshots are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("checkpoint join snapshots cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("checkpoint join snapshots cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("checkpoint join snapshots cannot be serialized")

    @staticmethod
    def _payload(
        value: "ActionBallFullMdpCheckpointJoinSnapshot",
    ) -> _CheckpointJoinSnapshotPayload:
        payload = _lookup_join_snapshot(value)
        if payload is None:
            raise ActionBallFullMdpRuntimeOwnerError(
                "checkpoint join snapshot is not provider-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return RUNTIME_OWNER_SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return CHECKPOINT_JOIN_SNAPSHOT_KIND

    @property
    def boundary_sha256(self) -> str:
        return self._payload(self).boundary_sha256

    @property
    def per_world_reset_identity(self) -> str:
        return self._payload(self).state.per_world_reset_identity_sha256

    @property
    def task_ball_r06_current(self) -> str:
        return self._payload(self).state.task_ball_r06_current_sha256

    @property
    def ppo_drain_frontier(self) -> str:
        return self._payload(self).drain_frontier_sha256

    @property
    def runtime_join_state_sha256(self) -> str:
        return self._payload(self).state.canonical_sha256

    @property
    def canonical_sha256(self) -> str:
        return self._payload(self).canonical_sha256


def _make_join_snapshot_registry():
    rows: weakref.WeakKeyDictionary[
        ActionBallFullMdpCheckpointJoinSnapshot,
        _CheckpointJoinSnapshotPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _CheckpointJoinSnapshotPayload,
    ) -> ActionBallFullMdpCheckpointJoinSnapshot:
        value = object.__new__(ActionBallFullMdpCheckpointJoinSnapshot)
        with lock:
            rows[value] = payload
        return value

    def lookup(
        value: ActionBallFullMdpCheckpointJoinSnapshot,
    ) -> Optional[_CheckpointJoinSnapshotPayload]:
        with lock:
            return rows.get(value)

    return mint, lookup


_mint_join_snapshot, _lookup_join_snapshot = _make_join_snapshot_registry()
del _make_join_snapshot_registry


@dataclass(frozen=True)
class _AuditFrontierRow:
    drain_sequence: int
    update_index: int
    completed_environment_steps: int
    global_receipt: object
    r03_receipt: object
    r07_receipt: object
    r03_receipt_sha256: str
    r07_receipt_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class _AuditFrontierClaimPayload:
    owner_identity: object
    consumer_kind: str
    consumer_identity: object
    row: Optional[_AuditFrontierRow]
    drain_snapshot: object
    drain_frontier_sha256: str
    canonical_sha256: str


class ActionBallFullMdpAuditFrontierClaim:
    """Opaque top-issued lease over one immutable R03/R07 audit row."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("audit frontier claims are runtime-owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("audit frontier claims are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("audit frontier claims cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("audit frontier claims cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("audit frontier claims cannot be serialized")

    @staticmethod
    def _payload(
        value: "ActionBallFullMdpAuditFrontierClaim",
    ) -> _AuditFrontierClaimPayload:
        payload = _lookup_audit_frontier_claim(value)
        if payload is None:
            raise ActionBallFullMdpRuntimeOwnerError(
                "audit frontier claim is not runtime-owner-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return RUNTIME_OWNER_SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return AUDIT_FRONTIER_CLAIM_KIND

    @property
    def consumer_kind(self) -> str:
        return self._payload(self).consumer_kind

    @property
    def drain_sequence(self) -> int:
        row = self._payload(self).row
        return 0 if row is None else row.drain_sequence

    @property
    def update_index(self) -> int:
        row = self._payload(self).row
        return -1 if row is None else row.update_index

    @property
    def completed_environment_steps(self) -> int:
        row = self._payload(self).row
        return -1 if row is None else row.completed_environment_steps

    @property
    def r03_receipt_sha256(self) -> Optional[str]:
        row = self._payload(self).row
        return None if row is None else row.r03_receipt_sha256

    @property
    def r07_receipt_sha256(self) -> Optional[str]:
        row = self._payload(self).row
        return None if row is None else row.r07_receipt_sha256

    @property
    def drain_frontier_sha256(self) -> str:
        return self._payload(self).drain_frontier_sha256

    @property
    def canonical_sha256(self) -> str:
        return self._payload(self).canonical_sha256


def _make_audit_frontier_claim_registry():
    rows: weakref.WeakKeyDictionary[
        ActionBallFullMdpAuditFrontierClaim,
        _AuditFrontierClaimPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _AuditFrontierClaimPayload,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        value = object.__new__(ActionBallFullMdpAuditFrontierClaim)
        with lock:
            rows[value] = payload
        return value

    def lookup(
        value: ActionBallFullMdpAuditFrontierClaim,
    ) -> Optional[_AuditFrontierClaimPayload]:
        with lock:
            return rows.get(value)

    return mint, lookup


_mint_audit_frontier_claim, _lookup_audit_frontier_claim = (
    _make_audit_frontier_claim_registry()
)
del _make_audit_frontier_claim_registry

_PROVIDER_CONSTRUCTION_TOKEN = object()
_PROVIDER_STATE_TOKEN = object()

_PROVIDER_API_SURFACE = (
    "snapshot_for_checkpoint_boundary(boundary)",
    "require_owned_snapshot(boundary,snapshot,expected_root)",
    "prepare_checkpoint_audit_claim(boundary,snapshot,owner_id)",
    "checkpoint_join_claims(boundary,snapshot,owner_id,audit_claim)",
    "provider_identity",
    "runtime_owner_identity",
    "current_runtime_join_state_sha256",
)
_PROVIDER_SCHEMA_METHODS = (
    "__init__",
    "provider_identity",
    "runtime_owner_identity",
    "current_runtime_join_state_sha256",
    "_publish_runtime_join_state",
    "_validate_state",
    "snapshot_for_checkpoint_boundary",
    "prepare_checkpoint_audit_claim",
    "require_owned_snapshot",
    "checkpoint_join_claims",
)
_SNAPSHOT_SCHEMA_METHODS = (
    "__new__",
    "__setattr__",
    "__copy__",
    "__deepcopy__",
    "__reduce__",
    "_payload",
    "schema_version",
    "kind",
    "boundary_sha256",
    "per_world_reset_identity",
    "task_ball_r06_current",
    "ppo_drain_frontier",
    "runtime_join_state_sha256",
    "canonical_sha256",
)


class ActionBallFullMdpCheckpointJoinSnapshotProvider:
    """Owner-retained shared join authority used by the real R10 adapters."""

    API_SCHEMA_SHA256: ClassVar[str]

    def __init__(
        self,
        *,
        num_envs: int,
        runtime_owner_identity: object,
        runtime_owner: object,
        ppo_drain_owner: object,
        checkpoint_module: object,
        _token: object,
    ) -> None:
        if _token is not _PROVIDER_CONSTRUCTION_TOKEN:
            raise ActionBallFullMdpRuntimeOwnerError(
                "checkpoint join provider is runtime-owner constructed only"
            )
        self._num_envs = _plain_positive_int(num_envs, label="num_envs")
        self._runtime_owner_identity = runtime_owner_identity
        self._runtime_owner = runtime_owner
        self._ppo_drain_owner = ppo_drain_owner
        self._checkpoint_module = checkpoint_module
        self._provider_identity = object()
        self._state: Optional[_ActionBallFullMdpCheckpointJoinState] = None
        self._last_snapshot: Optional[
            ActionBallFullMdpCheckpointJoinSnapshot
        ] = None
        self._active_audit_claim: Optional[
            ActionBallFullMdpAuditFrontierClaim
        ] = None
        self._lock = threading.RLock()

    @property
    def provider_identity(self) -> object:
        return self._provider_identity

    @property
    def runtime_owner_identity(self) -> object:
        return self._runtime_owner_identity

    @property
    def current_runtime_join_state_sha256(self) -> Optional[str]:
        with self._lock:
            return None if self._state is None else self._state.canonical_sha256

    def _publish_runtime_join_state(
        self,
        state: _ActionBallFullMdpCheckpointJoinState,
        *,
        runtime_owner_identity: object,
        _token: object,
    ) -> None:
        """Install only a state minted inside the owning runtime transaction."""

        with self._lock:
            if (
                _token is not _PROVIDER_STATE_TOKEN
                or runtime_owner_identity is not self._runtime_owner_identity
                or type(state) is not _ActionBallFullMdpCheckpointJoinState
                or state._runtime_owner_identity
                is not self._runtime_owner_identity
                or state.schema_version != RUNTIME_OWNER_SCHEMA_VERSION
                or state.kind != CHECKPOINT_JOIN_STATE_KIND
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint join state is foreign or caller-authored"
                )
            self._validate_state(state)
            previous = self._state
            if previous is not None:
                if state.sequence != previous.sequence + 1:
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "checkpoint join state sequence did not advance exactly once"
                    )
                for old_row, new_row in zip(
                    previous.world_reset_rows,
                    state.world_reset_rows,
                    strict=True,
                ):
                    generation_delta = (
                        new_row.reset_generation - old_row.reset_generation
                    )
                    if generation_delta not in (0, 1):
                        raise ActionBallFullMdpRuntimeOwnerError(
                            "world reset generation did not advance monotonically"
                        )
                    if (
                        generation_delta == 0
                        and new_row.reset_identity_sha256
                        != old_row.reset_identity_sha256
                    ):
                        raise ActionBallFullMdpRuntimeOwnerError(
                            "world reset identity changed within one generation"
                        )
                    if (
                        generation_delta == 1
                        and new_row.reset_identity_sha256
                        == old_row.reset_identity_sha256
                    ):
                        raise ActionBallFullMdpRuntimeOwnerError(
                            "world reset identity was reused across generations"
                        )
            self._state = state
            self._last_snapshot = None

    def _validate_state(
        self, state: _ActionBallFullMdpCheckpointJoinState
    ) -> None:
        if (
            type(state.schema_version) is not int
            or state.schema_version != RUNTIME_OWNER_SCHEMA_VERSION
            or state.kind != CHECKPOINT_JOIN_STATE_KIND
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "checkpoint join state schema differs"
            )
        _plain_positive_int(state.sequence, label="join_state.sequence")
        reset_rows = state.world_reset_rows
        current_rows = state.task_ball_r06_rows
        expected_ids = tuple(range(self._num_envs))
        if (
            type(reset_rows) is not tuple
            or type(current_rows) is not tuple
            or any(
                type(row) is not ActionBallFullMdpWorldResetJoinRow
                for row in reset_rows
            )
            or any(
                type(row) is not ActionBallFullMdpTaskBallR06JoinRow
                for row in current_rows
            )
            or tuple(row.world_id for row in reset_rows) != expected_ids
            or tuple(row.world_id for row in current_rows) != expected_ids
            or tuple(row.reset_generation for row in reset_rows)
            != tuple(row.reset_generation for row in current_rows)
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "checkpoint join state world/reset surface differs"
            )
        reset_root = _canonical_sha256(
            {
                "kind": "action_ball_per_world_reset_identity_join_v1",
                "rows": [
                    {
                        "world_id": row.world_id,
                        "reset_generation": row.reset_generation,
                        "reset_identity_sha256": row.reset_identity_sha256,
                    }
                    for row in reset_rows
                ],
            }
        )
        current_root = _canonical_sha256(
            {
                "kind": "action_ball_task_ball_r06_current_join_v1",
                "rows": [
                    {
                        "world_id": row.world_id,
                        "reset_generation": row.reset_generation,
                        "task_ball_r06_current_sha256": (
                            row.task_ball_r06_current_sha256
                        ),
                    }
                    for row in current_rows
                ],
            }
        )
        canonical = _canonical_sha256(
            {
                "schema_version": state.schema_version,
                "kind": state.kind,
                "sequence": state.sequence,
                "per_world_reset_identity_sha256": reset_root,
                "task_ball_r06_current_sha256": current_root,
            }
        )
        if (
            state.per_world_reset_identity_sha256 != reset_root
            or state.task_ball_r06_current_sha256 != current_root
            or state.canonical_sha256 != canonical
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "checkpoint join state roots differ from owner rows"
            )

    def snapshot_for_checkpoint_boundary(
        self, boundary: object
    ) -> ActionBallFullMdpCheckpointJoinSnapshot:
        """Mint one snapshot only at an exact complete-step R10 frontier."""

        with self._lock:
            checkpoint = self._checkpoint_module
            if (
                type(boundary) is not checkpoint.CheckpointBoundary
                or self._state is None
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint join snapshot requires exact boundary and live state"
                )
            checkpoint.validate_checkpoint_boundary(boundary)
            state = self._state
            if (
                len(boundary.worlds) != self._num_envs
                or tuple(world.world_id for world in boundary.worlds)
                != tuple(range(self._num_envs))
                or tuple(world.reset_generation for world in boundary.worlds)
                != tuple(row.reset_generation for row in state.world_reset_rows)
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint boundary differs from retained reset identities"
                )
            boundary_root = checkpoint.boundary_sha256(boundary)
            drain_snapshot = self._ppo_drain_owner.snapshot_for_checkpoint_boundary(
                boundary
            )
            owned_drain = self._ppo_drain_owner.require_owned_checkpoint_snapshot(
                boundary,
                drain_snapshot,
            )
            if owned_drain is not drain_snapshot:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "PPO drain checkpoint snapshot identity differs"
                )
            drain_root = _require_sha256(
                getattr(owned_drain, "checkpoint_frontier_sha256", None),
                label="PPO drain checkpoint frontier",
            )
            audit_claim = self._runtime_owner._claim_r10_audit_frontier(
                owned_drain
            )
            if self._last_snapshot is not None:
                retained = _lookup_join_snapshot(self._last_snapshot)
                if (
                    retained is not None
                    and retained.provider_identity is self._provider_identity
                    and retained.runtime_owner_identity
                    is self._runtime_owner_identity
                    and retained.boundary is boundary
                    and retained.boundary_sha256 == boundary_root
                    and retained.state is state
                    and retained.drain_snapshot is owned_drain
                    and retained.drain_frontier_sha256 == drain_root
                    and retained.r10_audit_claim is audit_claim
                ):
                    self._active_audit_claim = audit_claim
                    return self._last_snapshot
            root = _canonical_sha256(
                {
                    "schema_version": RUNTIME_OWNER_SCHEMA_VERSION,
                    "kind": CHECKPOINT_JOIN_SNAPSHOT_KIND,
                    "boundary_sha256": boundary_root,
                    "runtime_join_state_sha256": state.canonical_sha256,
                    "per_world_reset_identity": (
                        state.per_world_reset_identity_sha256
                    ),
                    "task_ball_r06_current": (
                        state.task_ball_r06_current_sha256
                    ),
                    "ppo_drain_frontier": drain_root,
                }
            )
            payload = _CheckpointJoinSnapshotPayload(
                provider_identity=self._provider_identity,
                runtime_owner_identity=self._runtime_owner_identity,
                boundary=boundary,
                boundary_sha256=boundary_root,
                state=state,
                drain_snapshot=owned_drain,
                drain_frontier_sha256=drain_root,
                r10_audit_claim=audit_claim,
                canonical_sha256=root,
            )
            snapshot = _mint_join_snapshot(payload)
            self._last_snapshot = snapshot
            self._active_audit_claim = audit_claim
            return snapshot

    def prepare_checkpoint_audit_claim(
        self,
        boundary: object,
        snapshot: object,
        owner_id: str,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        """Return the exact unacknowledged R10 audit claim in this snapshot."""

        with self._lock:
            if owner_id != "env.ball_physical":
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint audit claim owner differs"
                )
            if type(snapshot) is not ActionBallFullMdpCheckpointJoinSnapshot:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint audit claim requires the exact snapshot type"
                )
            payload = _lookup_join_snapshot(snapshot)
            if payload is None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint audit claim snapshot is not provider-issued"
                )
            self.require_owned_snapshot(
                boundary,
                snapshot,
                payload.canonical_sha256,
            )
            claim = payload.r10_audit_claim
            if (
                claim is not self._active_audit_claim
                or self._runtime_owner._require_owned_audit_frontier_claim(
                    claim,
                    "r10",
                )
                is not claim
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint audit claim is stale or foreign"
                )
            return claim

    def require_owned_snapshot(
        self,
        boundary: object,
        snapshot: object,
        expected_root: str,
    ) -> ActionBallFullMdpCheckpointJoinSnapshot:
        """Return only this provider's current exact boundary snapshot."""

        with self._lock:
            expected = _require_sha256(expected_root, label="expected_root")
            payload = (
                _lookup_join_snapshot(snapshot)
                if type(snapshot) is ActionBallFullMdpCheckpointJoinSnapshot
                else None
            )
            checkpoint = self._checkpoint_module
            if (
                payload is None
                or snapshot is not self._last_snapshot
                or payload.provider_identity is not self._provider_identity
                or payload.runtime_owner_identity
                is not self._runtime_owner_identity
                or payload.boundary is not boundary
                or type(boundary) is not checkpoint.CheckpointBoundary
                or payload.boundary_sha256
                != checkpoint.boundary_sha256(boundary)
                or payload.canonical_sha256 != expected
                or self._state is not payload.state
                or self._ppo_drain_owner.require_owned_checkpoint_snapshot(
                    boundary,
                    payload.drain_snapshot,
                )
                is not payload.drain_snapshot
                or payload.drain_frontier_sha256
                != payload.drain_snapshot.checkpoint_frontier_sha256
                or self._runtime_owner._require_owned_audit_frontier_claim(
                    payload.r10_audit_claim,
                    "r10",
                )
                is not payload.r10_audit_claim
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint join snapshot is stale, foreign, or root-mismatched"
                )
            return snapshot

    def checkpoint_join_claims(
        self,
        boundary: object,
        snapshot: object,
        owner_id: str,
        audit_claim: object,
    ) -> Tuple[object, ...]:
        """Return three roots; the audit claim remains unacknowledged."""

        # The R10 wrapper owns the longer checkpoint freeze.  This local lock
        # additionally makes current-snapshot validation and both claim reads
        # one indivisible provider operation.
        with self._lock:
            if owner_id != "env.ball_physical":
                raise ActionBallFullMdpRuntimeOwnerError(
                    "shared join provider does not own this R10 owner surface"
                )
            if type(snapshot) is not ActionBallFullMdpCheckpointJoinSnapshot:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint join claim requires the exact snapshot type"
                )
            payload = _lookup_join_snapshot(snapshot)
            if payload is None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint join claim snapshot is not provider-issued"
                )
            owned = self.require_owned_snapshot(
                boundary,
                snapshot,
                expected_root=payload.canonical_sha256,
            )
            prepared_audit = self.prepare_checkpoint_audit_claim(
                boundary,
                snapshot,
                owner_id,
            )
            if prepared_audit is not audit_claim:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "checkpoint audit claim identity differs"
                )
            checkpoint = self._checkpoint_module
            rows = [
                checkpoint.OwnerJoinClaim(
                    join_id="per_world_reset_identity",
                    value_sha256=owned.per_world_reset_identity,
                )
            ]
            rows.append(
                checkpoint.OwnerJoinClaim(
                    join_id="task_ball_r06_current",
                    value_sha256=owned.task_ball_r06_current,
                )
            )
            rows.append(
                checkpoint.OwnerJoinClaim(
                    join_id="ppo_drain_frontier",
                    value_sha256=owned.ppo_drain_frontier,
                )
            )
            return tuple(rows)


def _provider_api_schema_sha256(source_bytes: bytes) -> str:
    """Bind the exported provider schema to its actual complete class AST."""

    provider_api_sha256 = _method_surface_sha256(
        source_bytes,
        class_name="ActionBallFullMdpCheckpointJoinSnapshotProvider",
        method_names=_PROVIDER_SCHEMA_METHODS,
        field_names=("API_SCHEMA_SHA256",),
    )
    snapshot_api_sha256 = _method_surface_sha256(
        source_bytes,
        class_name="ActionBallFullMdpCheckpointJoinSnapshot",
        method_names=_SNAPSHOT_SCHEMA_METHODS,
        field_names=(),
    )
    provider_class_ast_sha256 = _class_ast_sha256(
        source_bytes,
        class_name="ActionBallFullMdpCheckpointJoinSnapshotProvider",
    )
    snapshot_class_ast_sha256 = _class_ast_sha256(
        source_bytes,
        class_name="ActionBallFullMdpCheckpointJoinSnapshot",
    )
    return _canonical_sha256(
        {
            "schema_version": RUNTIME_OWNER_SCHEMA_VERSION,
            "kind": "action_ball_full_mdp_checkpoint_join_provider_api_v2",
            "class": "ActionBallFullMdpCheckpointJoinSnapshotProvider",
            "class_api_sha256": provider_api_sha256,
            "class_ast_sha256": provider_class_ast_sha256,
            "snapshot_class": "ActionBallFullMdpCheckpointJoinSnapshot",
            "snapshot_class_api_sha256": snapshot_api_sha256,
            "snapshot_class_ast_sha256": snapshot_class_ast_sha256,
            "surface": list(_PROVIDER_API_SURFACE),
            "join_ids": [
                "per_world_reset_identity",
                "task_ball_r06_current",
                "ppo_drain_frontier",
            ],
            "source_pin_direction": "runtime_owner_to_physical_only",
        }
    )


PROVIDER_API_SCHEMA_SHA256 = _provider_api_schema_sha256(
    Path(__file__).resolve().read_bytes()
)
ActionBallFullMdpCheckpointJoinSnapshotProvider.API_SCHEMA_SHA256 = (
    PROVIDER_API_SCHEMA_SHA256
)


# ---- Global terminal receipt -------------------------------------------


@dataclass(frozen=True)
class _SelectedTrueResetReceiptPayload:
    owner_identity: object
    event: object
    prepared_true_reset: object
    device_r05_receipt: object
    child_completions: Tuple[object, ...]
    sequence: int


class ActionBallFullMdpSelectedTrueResetReceipt:
    """Opaque safe-settlement receipt consumed exactly once by the env.

    The capability attests only that the fixed four-child/R05-last settlement
    and completion-consumption order finished.  It deliberately exposes no
    pass/healthy flag and no device facts; device fault truth remains in the
    child owners and the sole global drain/reveal authorities.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("selected-reset receipts are runtime-owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("selected-reset receipts are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("selected-reset receipts cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("selected-reset receipts cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("selected-reset receipts cannot be serialized")


def _make_selected_true_reset_receipt_registry():
    rows: weakref.WeakKeyDictionary[
        ActionBallFullMdpSelectedTrueResetReceipt,
        _SelectedTrueResetReceiptPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _SelectedTrueResetReceiptPayload,
    ) -> ActionBallFullMdpSelectedTrueResetReceipt:
        value = object.__new__(ActionBallFullMdpSelectedTrueResetReceipt)
        with lock:
            rows[value] = payload
        return value

    def lookup(
        value: ActionBallFullMdpSelectedTrueResetReceipt,
    ) -> Optional[_SelectedTrueResetReceiptPayload]:
        with lock:
            return rows.get(value)

    return mint, lookup


(
    _mint_selected_true_reset_receipt,
    _lookup_selected_true_reset_receipt,
) = _make_selected_true_reset_receipt_registry()
del _make_selected_true_reset_receipt_registry


@dataclass(frozen=True)
class ActionBallFullMdpRewardOwnerBinding:
    """Construction-only projection of the four causal Reward owners."""

    r03: object
    physical: object
    r06: object
    r07: object
    num_envs: int
    device: object


class ActionBallFullMdpPreRewardPublication:
    """Opaque top-owned identity for one pre-Reward transition."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("full-MDP pre-Reward publications are top-owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("full-MDP pre-Reward publications are immutable")


@dataclass(frozen=True)
class ActionBallFullMdpPreRewardView:
    """Clone-only facts authenticated against one exact top publication."""

    terminated: torch.Tensor
    time_out: torch.Tensor
    r03_publication: object
    r07_publication: object
    physical_reward_cycle: object
    r06_reward_cycle: object


class ActionBallFullMdpRewardCloseReceipt:
    """Opaque proof that all four real owner epochs closed in order."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("full-MDP Reward close receipts are top-owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("full-MDP Reward close receipts are immutable")


@dataclass(frozen=True)
class _PreRewardPublicationPayload:
    owner_identity: object
    runtime_lease: object
    control_step: int
    r03_publication: object
    r07_publication: object
    terminated: torch.Tensor
    time_out: torch.Tensor
    physical_reward_cycle: object = None
    r06_reward_cycle: object = None


@dataclass(frozen=True)
class _RewardClosePayload:
    owner_identity: object
    pre_reward_publication: ActionBallFullMdpPreRewardPublication
    control_step: int
    owner_close_receipts: Tuple[Tuple[str, object], ...]


_PRE_REWARD_PUBLICATION_REGISTRY: weakref.WeakKeyDictionary[
    ActionBallFullMdpPreRewardPublication, _PreRewardPublicationPayload
] = weakref.WeakKeyDictionary()
_REWARD_CLOSE_REGISTRY: weakref.WeakKeyDictionary[
    ActionBallFullMdpRewardCloseReceipt, _RewardClosePayload
] = weakref.WeakKeyDictionary()


def _mint_pre_reward_publication(
    payload: _PreRewardPublicationPayload,
) -> ActionBallFullMdpPreRewardPublication:
    value = object.__new__(ActionBallFullMdpPreRewardPublication)
    _PRE_REWARD_PUBLICATION_REGISTRY[value] = payload
    return value


def _mint_reward_close_receipt(
    payload: _RewardClosePayload,
) -> ActionBallFullMdpRewardCloseReceipt:
    value = object.__new__(ActionBallFullMdpRewardCloseReceipt)
    _REWARD_CLOSE_REGISTRY[value] = payload
    return value


class ActionBallFullMdpRuntimeOwner:
    """Sole top coordinator for the exact owner graph.

    Production construction is deliberately two-gated: the dependency
    inventory is frozen before any supplied object is inspected, then every
    object is checked against the class loaded from that exact source graph.
    The environment transports only an identity lease.  It never supplies a
    reveal request, post-physics fact, or reset closure.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("use ActionBallFullMdpRuntimeOwner.create")

    @classmethod
    def create(
        cls,
        *,
        num_envs: int,
        device: object,
        r05_owner: object,
        device_r05_owner: object,
        motion_owner: object,
        racket_owner: object,
        r06_owner: object,
        physical_owner: object,
        r03_owner: object,
        r07_owner: object,
        ppo_drain_owner: object,
        env: object,
        env_lease: object,
    ) -> "ActionBallFullMdpRuntimeOwner":
        """Construct only from an already-frozen exact dependency graph."""

        # This line must remain first.  It prevents a shape-compatible object,
        # fixture receipt, or caller-authored authority from becoming a
        # temporary success path while the dependency DAG is still changing.
        inventory = _require_frozen_dependency_inventory()
        return cls._create_with_inventory(
            inventory=inventory,
            num_envs=num_envs,
            device=device,
            r05_owner=r05_owner,
            device_r05_owner=device_r05_owner,
            motion_owner=motion_owner,
            racket_owner=racket_owner,
            r06_owner=r06_owner,
            physical_owner=physical_owner,
            r03_owner=r03_owner,
            r07_owner=r07_owner,
            ppo_drain_owner=ppo_drain_owner,
            env=env,
            env_lease=env_lease,
        )

    @classmethod
    def create_from_env(
        cls,
        env: object,
        lease: object,
    ) -> "ActionBallFullMdpRuntimeOwner":
        """Discover the exact owners through one lease-bound environment."""

        # Keep the same no-inspection-before-freeze property as ``create``.
        inventory = _require_frozen_dependency_inventory()
        if env is None or lease is None:
            raise ActionBallFullMdpRuntimeDependencyError(
                "runtime factory requires an exact env and non-null lease"
            )
        if getattr(env, "action_ball_full_mdp_runtime_lease", None) is not lease:
            raise ActionBallFullMdpRuntimeDependencyError(
                "runtime factory lease does not belong to this environment"
            )

        getter_names = (
            "action_ball_full_mdp_num_envs",
            "action_ball_full_mdp_device",
            "action_ball_full_mdp_r05_owner",
            "action_ball_full_mdp_device_r05_owner",
            "action_ball_full_mdp_motion_owner",
            "action_ball_full_mdp_racket_owner",
            "action_ball_full_mdp_r06_owner",
            "action_ball_full_mdp_physical_owner",
            "action_ball_full_mdp_r03_owner",
            "action_ball_full_mdp_r07_owner",
            "action_ball_full_mdp_ppo_drain_owner",
        )
        getters = tuple(getattr(env, name, None) for name in getter_names)
        if any(not callable(value) for value in getters):
            raise ActionBallFullMdpRuntimeDependencyError(
                "environment is missing an exact full-MDP owner getter"
            )
        values = tuple(getter(lease) for getter in getters)
        return cls._create_with_inventory(
            inventory=inventory,
            num_envs=values[0],
            device=values[1],
            r05_owner=values[2],
            device_r05_owner=values[3],
            motion_owner=values[4],
            racket_owner=values[5],
            r06_owner=values[6],
            physical_owner=values[7],
            r03_owner=values[8],
            r07_owner=values[9],
            ppo_drain_owner=values[10],
            env=env,
            env_lease=lease,
        )

    @classmethod
    def _create_with_inventory(
        cls,
        *,
        inventory: ActionBallFullMdpRuntimeDependencyInventory,
        num_envs: int,
        device: object,
        r05_owner: object,
        device_r05_owner: object,
        motion_owner: object,
        racket_owner: object,
        r06_owner: object,
        physical_owner: object,
        r03_owner: object,
        r07_owner: object,
        ppo_drain_owner: object,
        env: object,
        env_lease: object,
    ) -> "ActionBallFullMdpRuntimeOwner":
        modules = _load_frozen_dependency_modules(inventory)
        count = _plain_positive_int(num_envs, label="num_envs")
        role_objects = {
            "r05_reveal_owner": r05_owner,
            "device_r05_owner": device_r05_owner,
            "motion_child": motion_owner,
            "racket_child": racket_owner,
            "r06_child": r06_owner,
            "physical_ball_child": physical_owner,
            "r03_child": r03_owner,
            "r07_child": r07_owner,
            "ppo_drain_owner": ppo_drain_owner,
        }
        specs = {spec.role: spec for spec in _DEPENDENCY_SPECS}
        for role, value in role_objects.items():
            expected = getattr(modules[role], specs[role].class_name)
            if type(value) is not expected:
                raise ActionBallFullMdpRuntimeDependencyError(
                    f"runtime dependency has the wrong exact type: {role}"
                )
        if env is None or env_lease is None:
            raise ActionBallFullMdpRuntimeDependencyError(
                "runtime owner requires an exact environment lease"
            )
        for role in (
            "motion_child",
            "racket_child",
            "r06_child",
            "physical_ball_child",
            "r03_child",
            "r07_child",
        ):
            owner = role_objects[role]
            if getattr(owner, "num_envs", None) != count:
                raise ActionBallFullMdpRuntimeDependencyError(
                    f"runtime dependency num_envs differs: {role}"
                )
            if not _same_device(getattr(owner, "device", None), device):
                raise ActionBallFullMdpRuntimeDependencyError(
                    f"runtime dependency device differs: {role}"
                )
        if (
            ppo_drain_owner.num_envs != count
            or not _same_device(ppo_drain_owner.device, device)
        ):
            raise ActionBallFullMdpRuntimeDependencyError(
                "PPO drain owner shape/device differs"
            )
        # The drain is preconstructed because it also owns R03/R07.  Join its
        # seven private leaf identities exactly once before reading any drain
        # runtime property; type and device equality alone cannot exclude a
        # correctly-shaped coordinator bound to foreign causal writers.
        ppo_drain_owner.require_exact_leaf_bindings(
            {
                "r05_runtime": device_r05_owner,
                "motion": motion_owner,
                "racket": racket_owner,
                "physical_ball": physical_owner,
                "r06_landing_outcome": r06_owner,
                "r03_strike_fact": r03_owner,
                "r07_recovery": r07_owner,
            }
        )

        owner = object.__new__(cls)
        owner._identity = object()
        owner._inventory = inventory
        owner._num_envs = count
        owner._device = device
        owner._env = env
        owner._env_lease = env_lease
        owner._r05 = r05_owner
        owner._device_r05 = device_r05_owner
        owner._motion = motion_owner
        owner._racket = racket_owner
        owner._r06 = r06_owner
        owner._physical = physical_owner
        owner._r03 = r03_owner
        owner._r07 = r07_owner
        owner._ppo_drain = ppo_drain_owner
        owner._active_optimizer_receipt = None
        owner._active_optimizer_update_index = None
        owner._audit_frontier_ring = []
        owner._r10_audit_highwater = 0
        owner._r11_audit_highwater = 0
        owner._active_r10_audit_claim = None
        owner._active_r11_audit_claim = None
        owner._r10_checkpoint_publication_validator = None
        owner._r10_checkpoint_publication_consumer = None
        owner._r11_audit_consumer = None
        owner._r11_audit_consumer_validator = None
        owner._audit_consumer_binding_open = True
        owner._poisoned = False
        owner._poison_reason = None
        owner._poison_failures = ()
        owner._selected_reset_event = None
        owner._selected_reset_prepared = None
        owner._selected_reset_child_commits = None
        owner._selected_reset_child_commits_started = False
        owner._selected_reset_projection = None
        owner._selected_reset_r05_receipt = None
        owner._selected_reset_completions = None
        owner._selected_reset_env_binding = None
        owner._selected_reset_env_binding_view = None
        owner._selected_reset_receipt = None
        owner._selected_reset_receipt_consumed = False
        owner._selected_reset_sequence = 0
        owner._reward_owner_binding_open = True
        owner._reward_owner_binding = None
        owner._full_mdp_reward_graph = None
        owner._active_pre_reward_publication = None
        owner._active_pre_reward_payload = None
        owner._reward_poisoned = False
        owner._reward_poison_reason = None
        owner._last_final_postphysics_control_step = None
        owner._lock = threading.RLock()

        # All four children must consume Device-R05's independent genesis
        # while its construction window is still open.  In particular, the
        # physical reset binder must precede construction of its checkpoint
        # adapter: the adapter closes the physical construction seam.  Never
        # read Device-R05 ``num_envs`` or a state-view property here: those
        # public views intentionally close the same construction window.
        motion_owner.bind_action_ball_continuous_motion_device_r05_reveal(
            device_r05_owner
        )
        racket_owner.bind_action_ball_full_mdp_racket_staging(
            device_r05_owner
        )
        physical_owner.bind_device_r05_reset_owner(
            device_r05_owner,
            prepared_reset_validator=(
                device_r05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                device_r05_owner.require_owned_true_reset_receipt
            ),
        )
        r06_owner.bind_device_r05_reset_owner(
            device_r05_owner,
            prepared_reset_validator=(
                device_r05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                device_r05_owner.require_owned_true_reset_receipt
            ),
        )
        physical_owner.bind_r06_owner(r06_owner)
        env_binding = device_r05_owner.project_full_mdp_env_reset_binding()
        env_binding_view = (
            device_r05_owner.require_owned_full_mdp_env_reset_binding(
                env_binding
            )
        )
        owner._selected_reset_env_binding = env_binding
        owner._selected_reset_env_binding_view = env_binding_view
        env_reset_binder = getattr(
            env,
            "bind_action_ball_full_mdp_selected_reset_authority",
            None,
        )
        if not callable(env_reset_binder):
            raise ActionBallFullMdpRuntimeDependencyError(
                "environment selected-reset authority binder is absent"
            )
        env_reset_binder(
            env_lease,
            expected_top=owner,
            result_validator=(
                owner.require_owned_selected_true_reset_receipt
            ),
            live_reset_ledger_identity=(
                env_binding_view.live_reset_ledger_identity
            ),
            world_reset_identity=env_binding_view.world_reset_identity,
        )
        authority_api_sha256 = _selected_reset_authority_api_sha256(
            Path(__file__).resolve().read_bytes()
        )
        motion_owner.bind_action_ball_continuous_motion_selected_reset(
            device_r05_owner,
            prepared_reset_validator=(
                device_r05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                device_r05_owner.require_owned_true_reset_receipt
            ),
            authority_source_sha256=authority_api_sha256,
            diagnostic=False,
        )
        racket_owner.bind_action_ball_continuous_racket_selected_reset(
            device_r05_owner,
            prepared_reset_validator=(
                device_r05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                device_r05_owner.require_owned_true_reset_receipt
            ),
            authority_source_sha256=authority_api_sha256,
            diagnostic=False,
        )

        checkpoint_module = modules["r10_checkpoint_contract"]
        provider = ActionBallFullMdpCheckpointJoinSnapshotProvider(
            num_envs=count,
            runtime_owner_identity=owner._identity,
            runtime_owner=owner,
            ppo_drain_owner=ppo_drain_owner,
            checkpoint_module=checkpoint_module,
            _token=_PROVIDER_CONSTRUCTION_TOKEN,
        )
        adapter_module = modules["physical_checkpoint_adapter"]
        adapter = adapter_module.PhysicalFlightCheckpointAdapter(
            physical_owner=physical_owner,
            shared_join_snapshot_provider=provider,
        )
        owner._checkpoint_join_snapshot_provider = provider
        owner._r10_checkpoint_adapter = adapter
        # Close Device-R05 only after every child/env genesis join and the
        # physical checkpoint adapter are construction-bound.  No Device-R05
        # public state view is read anywhere above this final close.
        device_r05_owner.bind_true_reset_authority(owner)
        owner._audit_consumer_binding_open = False
        return owner

    @property
    def full_mdp_runtime_dependency_dag_sha256(self) -> str:
        return self._inventory.content_sha256

    @property
    def full_mdp_runtime_env(self) -> object:
        return self._env

    @property
    def full_mdp_runtime_lease(self) -> object:
        return self._env_lease

    @property
    def launch_authorized(self) -> bool:
        return LAUNCH_AUTHORIZED

    @property
    def diagnostic_unauthorized(self) -> bool:
        return DIAGNOSTIC_UNAUTHORIZED

    # Backward-compatible aliases for the already-shipped env validator.
    @property
    def full_mdp_post_physics_dependency_dag_sha256(self) -> str:
        return self.full_mdp_runtime_dependency_dag_sha256

    @property
    def full_mdp_post_physics_env(self) -> object:
        return self.full_mdp_runtime_env

    @property
    def full_mdp_post_physics_lease(self) -> object:
        return self.full_mdp_runtime_lease

    @property
    def checkpoint_join_snapshot_provider(
        self,
    ) -> ActionBallFullMdpCheckpointJoinSnapshotProvider:
        return self._checkpoint_join_snapshot_provider

    @property
    def action_ball_r10_checkpoint_adapter(self) -> object:
        return self._r10_checkpoint_adapter

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def poison_reason(self) -> Optional[str]:
        return self._poison_reason

    @property
    def poison_failures(self) -> Tuple[Tuple[str, str], ...]:
        return self._poison_failures

    def require_healthy(self) -> None:
        hot_failure = None
        try:
            self._device_r05.require_healthy()
        except BaseException as exc:
            hot_failure = exc
        if (
            self._poisoned
            or bool(getattr(self._ppo_drain, "poisoned", False))
            or hot_failure is not None
        ):
            raise ActionBallFullMdpRuntimePoisonedError(
                self._poison_reason
                or getattr(self._ppo_drain, "poison_reason", None)
                or (
                    None
                    if hot_failure is None
                    else self._failure_reason("Device-R05 is unhealthy", hot_failure)
                )
                or "full-MDP runtime owner requires cold replacement"
            )

        if bool(getattr(self, "_reward_poisoned", False)):
            raise ActionBallFullMdpRuntimePoisonedError(
                self._reward_poison_reason
                or "full-MDP Reward transaction requires cold replacement"
            )

    def _require_no_selected_reset_debt(self, *, operation: str) -> None:
        if (
            self._selected_reset_event is not None
            or self._selected_reset_prepared is not None
            or self._selected_reset_receipt is not None
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                f"{operation} cannot cross an unsettled selected reset"
            )

        if getattr(self, "_active_pre_reward_publication", None) is not None:
            raise ActionBallFullMdpRuntimeOwnerError(
                f"{operation} cannot cross an unsettled Reward transaction"
            )

    @staticmethod
    def _failure_reason(prefix: str, exc: BaseException) -> str:
        detail = str(exc).strip()
        return prefix if not detail else f"{prefix}: {detail}"

    def _poison_global(self, reason: str) -> None:
        clean = reason if type(reason) is str and reason.strip() else "global runtime failure"
        if self._poison_reason is None:
            self._poison_reason = clean
        self._poisoned = True
        failures = list(self._poison_failures)
        for name, child in (
            ("motion", self._motion),
            ("racket", self._racket),
            ("physical_ball", self._physical),
            ("r06_flight", self._r06),
            ("r05", self._r05),
        ):
            try:
                child.poison_global_reveal_epoch(clean)
            except BaseException as exc:
                failures.append((name, type(exc).__name__))
        # Device-R05 is the production hot/reset owner.  It must fail-stop
        # independently from the portable audit R05 after any global partial
        # mutation.  The leaf accepts a stable integer reason code so this
        # path never makes a caller-authored string into an authority fact.
        try:
            self._device_r05.poison_from_external_failure(10)
        except BaseException as exc:
            failures.append(("device_r05", type(exc).__name__))
        self._poison_failures = tuple(failures)

    def _poison_reward(self, reason: str) -> None:
        clean = (
            reason
            if type(reason) is str and reason.strip()
            else "full-MDP Reward transaction failed"
        )
        first_poison = not self._reward_poisoned
        if self._reward_poison_reason is None:
            self._reward_poison_reason = clean
        self._reward_poisoned = True
        if first_poison:
            failures = list(self._poison_failures)
            for name, child, method_name in (
                ("r03", self._r03, "_poison_pre_optimizer"),
                ("physical_ball", self._physical, "_poison_physical_reward"),
                ("r06_flight", self._r06, "_poison_full_mdp_reward_protocol"),
                ("r07_recovery", self._r07, "_poison_full_mdp_reward"),
            ):
                try:
                    poison = getattr(child, method_name)
                    poison(clean)
                except BaseException as exc:
                    failures.append((name + "_reward", type(exc).__name__))
            self._poison_failures = tuple(failures)
        self._poison_global(clean)


    def before_policy_step(self, control_step: int, action: object) -> None:
        """HOLD until the live row transaction owns policy-step orchestration.

        The retired portable-R05 executor has no fallback entry.  Reject before
        asking Motion for a request or mutating any Device-R05/ActionEpoch leaf.
        """

        self.require_healthy()
        self._require_no_selected_reset_debt(operation="policy step")
        del control_step, action
        raise ActionBallFullMdpRuntimeDependencyError(
            "runtime reveal HOLD: children do not consume Device-R05 hot tokens"
        )

    def execute_reveal(self, request: object) -> None:
        """Diagnostic/private entry; production must use ``before_policy_step``."""

        del request
        raise ActionBallFullMdpRuntimeDependencyError(
            "direct reveal execution is a tombstone; use before_policy_step"
        )


    def publish_post_physics_substep(self, stamp: object) -> None:
        """Capture physical facts at their producer, then publish and retire."""

        with self._lock:
            self.require_healthy()
            self._require_no_selected_reset_debt(operation="postphysics")
            capture = getattr(
                self._physical,
                "capture_post_physics_facts",
                None,
            )
            if not callable(capture):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "physical post-physics fact producer is not connected"
                )
            try:
                facts = capture(stamp)
                publication = self._physical.build_post_physics_publication(
                    facts=facts
                )
                result = self._physical.publish_post_physics_to_r06(
                    publication
                )
                self._physical.retire_post_physics_to_r06(result)
                stamp_control_step = getattr(stamp, "control_step", None)
                stamp_substep = getattr(stamp, "physics_substep", None)
                stamp_decimation = getattr(
                    stamp, "physics_substeps_per_control", None
                )
                if (
                    type(stamp_control_step) is int
                    and type(stamp_substep) is int
                    and type(stamp_decimation) is int
                    and stamp_control_step >= 0
                    and stamp_decimation > 0
                    and stamp_substep == stamp_decimation - 1
                ):
                    self._last_final_postphysics_control_step = stamp_control_step
            except BaseException as exc:
                self._poison_global(
                    self._failure_reason(
                        "post-physics publication failed after physics step",
                        exc,
                    )
                )
                raise

    def bind_full_mdp_reward_owners(
        self,
        *,
        runtime_lease: object,
        ordered_consumers: object,
        reward_graph: object,
    ) -> ActionBallFullMdpRewardOwnerBinding:
        """Bind the exact four causal Reward owners once during construction."""

        with self._lock:
            self.require_healthy()
            if runtime_lease is not self._env_lease:
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward owner binding lease differs from the runtime owner"
                )
            if type(ordered_consumers) is not tuple or ordered_consumers != (
                FULL_MDP_REWARD_ORDERED_CONSUMERS
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward manager consumer order differs from the exact fourteen"
                )
            if not getattr(self, "_reward_owner_binding_open", True):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward owner binding is construction-only and already closed"
                )
            reward_graph_type = type(reward_graph)
            try:
                reward_graph_source = inspect.getsourcefile(reward_graph_type)
            except (TypeError, OSError):
                reward_graph_source = None
            if (
                reward_graph_type.__name__ != "FreshFullMdpRewardGraph"
                or reward_graph_type.__module__
                != "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_rewards"
                or reward_graph_source is None
                or Path(reward_graph_source).resolve()
                != Path(__file__).with_name(
                    "action_ball_full_mdp_rewards.py"
                ).resolve()
                or getattr(reward_graph, "_runtime_owner", None) is not self
                or getattr(reward_graph, "_runtime_lease", None) is not runtime_lease
                or getattr(reward_graph, "diagnostic_unauthorized", None) is not False
                or getattr(reward_graph, "_production_constructed", None) is not True
                or getattr(reward_graph, "active_cycle", None) is not None
                or not callable(getattr(reward_graph, "close_after_reward", None))
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward owner binding requires the exact fresh graph before first use"
                )
            owners = {
                "r03": self._r03,
                "physical": self._physical,
                "r06": self._r06,
                "r07": self._r07,
            }
            graph_owners = getattr(reward_graph, "_owners", None)
            if (
                graph_owners is None
                or any(
                    getattr(graph_owners, owner_name, None) is not owner
                    for owner_name, owner in owners.items()
                )
                or getattr(graph_owners, "num_envs", None) != self._num_envs
                or not _same_device(
                    getattr(graph_owners, "device", None), self._device
                )
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward graph causal owner identity/shape/device differs"
                )
            for owner_name, consumers in FULL_MDP_REWARD_OWNER_CONSUMERS:
                owner = owners[owner_name]
                method_names = (
                    REWARD_CHILD_REQUIRE_PAYMENT_METHOD,
                    REWARD_CHILD_CLOSE_METHOD,
                    REWARD_CHILD_REQUIRE_CLOSE_METHOD,
                )
                if owner_name in ("r03", "r07"):
                    method_names = (
                        REWARD_CHILD_PUBLISH_METHOD,
                        REWARD_CHILD_REQUIRE_PUBLISH_METHOD,
                        *method_names,
                    )
                for method_name in method_names:
                    if not callable(getattr(owner, method_name, None)):
                        raise ActionBallFullMdpRuntimeDependencyError(
                            f"{owner_name} Reward owner lacks exact method "
                            f"{method_name}"
                        )
                declared = tuple(getattr(owner, "full_mdp_reward_consumers", ()))
                if declared != consumers:
                    raise ActionBallFullMdpRuntimeDependencyError(
                        f"{owner_name} Reward consumer ABI differs"
                    )
                if (
                    getattr(owner, "num_envs", None) != self._num_envs
                    or not _same_device(getattr(owner, "device", None), self._device)
                ):
                    raise ActionBallFullMdpRuntimeDependencyError(
                        f"{owner_name} Reward owner shape/device differs"
                    )
            binding = ActionBallFullMdpRewardOwnerBinding(
                r03=self._r03,
                physical=self._physical,
                r06=self._r06,
                r07=self._r07,
                num_envs=self._num_envs,
                device=self._device,
            )
            physical_bind = getattr(
                self._physical,
                "_bind_full_mdp_reward_graph_from_top",
                None,
            )
            r06_bind = getattr(
                self._r06,
                "_bind_full_mdp_reward_graph_from_top",
                None,
            )
            if (
                not callable(physical_bind)
                or not callable(r06_bind)
                or not callable(
                    getattr(self._physical, "open_full_mdp_reward_cycle", None)
                )
                or not callable(
                    getattr(self._r06, "open_full_mdp_reward_cycle", None)
                )
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Physical/R06 Reward owner lacks exact bind/open methods"
                )
            try:
                physical_bind(
                    runtime_owner=self,
                    ordered_consumers=("physical_selected_contact",),
                )
                r06_bind(
                    runtime_owner=self,
                    ordered_consumers=(
                        "common_on_table_outcome",
                        "post_contact_placement_guidance",
                    ),
                )
            except BaseException as exc:
                self._poison_reward(
                    self._failure_reason(
                        "Reward construction bind failed",
                        exc,
                    )
                )
                raise
            self._reward_owner_binding = binding
            self._full_mdp_reward_graph = reward_graph
            self._reward_owner_binding_open = False
            return binding

    @staticmethod
    def _clone_reward_bool(value: object, *, num_envs: int, device: object, label: str) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (num_envs,)
            or value.dtype != torch.bool
            or not _same_device(value.device, device)
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                f"{label} must be device-local bool [num_envs]"
            )
        return value.detach().clone()

    def publish_full_mdp_pre_reward(
        self,
        *,
        runtime_lease: object,
        control_step: int,
    ) -> ActionBallFullMdpPreRewardPublication:
        """Publish R03 then R07 after the final physical postphysics substep."""

        with self._lock:
            self.require_healthy()
            if runtime_lease is not self._env_lease:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "pre-Reward runtime lease differs"
                )
            step = _plain_nonnegative_int(control_step, label="control_step")
            if self._reward_owner_binding is None:
                raise ActionBallFullMdpRuntimeDependencyError(
                    "Reward owners are not construction-bound"
                )
            if self._active_pre_reward_publication is not None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "previous Reward transaction is still open"
                )
            if self._last_final_postphysics_control_step != step:
                raise ActionBallFullMdpRuntimeDependencyError(
                    "pre-Reward publication did not follow the final "
                    "postphysics substep of this transition"
                )
            try:
                r03_publication = self._r03.publish_full_mdp_pre_reward(
                    control_step=step,
                    runtime_owner=self,
                )
                r03_owned = self._r03.require_owned_full_mdp_pre_reward(
                    r03_publication,
                    control_step=step,
                    runtime_owner=self,
                )
                r07_publication = self._r07.publish_full_mdp_pre_reward(
                    control_step=step,
                    runtime_owner=self,
                )
                r07_owned = self._r07.require_owned_full_mdp_pre_reward(
                    r07_publication,
                    control_step=step,
                    runtime_owner=self,
                )
                terminated = self._clone_reward_bool(
                    getattr(r03_owned, "terminated", None),
                    num_envs=self._num_envs,
                    device=self._device,
                    label="R03 terminated",
                )
                r07_terminated = self._clone_reward_bool(
                    getattr(r07_owned, "terminated", None),
                    num_envs=self._num_envs,
                    device=self._device,
                    label="R07 terminated",
                )
                terminated.logical_or_(r07_terminated)
                time_out = self._clone_reward_bool(
                    getattr(r03_owned, "time_out", None),
                    num_envs=self._num_envs,
                    device=self._device,
                    label="R03 time_out",
                )
                r07_time_out = self._clone_reward_bool(
                    getattr(r07_owned, "time_out", None),
                    num_envs=self._num_envs,
                    device=self._device,
                    label="R07 time_out",
                )
                time_out.logical_or_(r07_time_out)
                payload = _PreRewardPublicationPayload(
                    owner_identity=self._identity,
                    runtime_lease=runtime_lease,
                    control_step=step,
                    r03_publication=r03_publication,
                    r07_publication=r07_publication,
                    terminated=terminated,
                    time_out=time_out,
                )
                publication = _mint_pre_reward_publication(payload)
                self._active_pre_reward_publication = publication
                self._active_pre_reward_payload = payload
                physical_cycle = self._physical.open_full_mdp_reward_cycle(
                    publication,
                    control_step=step,
                    runtime_owner=self,
                )
                r06_cycle = self._r06.open_full_mdp_reward_cycle(
                    publication,
                    control_step=step,
                    runtime_owner=self,
                )
                payload = replace(
                    payload,
                    physical_reward_cycle=physical_cycle,
                    r06_reward_cycle=r06_cycle,
                )
                _PRE_REWARD_PUBLICATION_REGISTRY[publication] = payload
                self._active_pre_reward_payload = payload
                self._last_final_postphysics_control_step = None
                return publication
            except BaseException as exc:
                self._poison_reward(
                    self._failure_reason("pre-Reward publication failed", exc)
                )
                raise

    def require_owned_full_mdp_pre_reward(
        self,
        publication: object,
        *,
        runtime_lease: object,
        control_step: int,
    ) -> ActionBallFullMdpPreRewardView:
        """Authenticate one exact active publication and clone its device facts."""

        with self._lock:
            self.require_healthy()
            try:
                payload = _PRE_REWARD_PUBLICATION_REGISTRY.get(publication)
            except TypeError:
                payload = None
            if (
                type(publication) is not ActionBallFullMdpPreRewardPublication
                or publication is not self._active_pre_reward_publication
                or payload is None
                or payload is not self._active_pre_reward_payload
                or payload.owner_identity is not self._identity
                or runtime_lease is not payload.runtime_lease
                or _plain_nonnegative_int(control_step, label="control_step")
                != payload.control_step
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "pre-Reward publication is foreign, stale, or wrong-step"
                )
            return ActionBallFullMdpPreRewardView(
                terminated=payload.terminated.detach().clone(),
                time_out=payload.time_out.detach().clone(),
                r03_publication=payload.r03_publication,
                r07_publication=payload.r07_publication,
                physical_reward_cycle=payload.physical_reward_cycle,
                r06_reward_cycle=payload.r06_reward_cycle,
            )

    def after_reward_close(self, control_step: int) -> None:
        """Close the one construction-bound graph immediately after RewardManager."""

        with self._lock:
            self.require_healthy()
            step = _plain_nonnegative_int(control_step, label="control_step")
            graph = self._full_mdp_reward_graph
            payload = self._active_pre_reward_payload
            if (
                graph is None
                or self._reward_owner_binding is None
                or self._active_pre_reward_publication is None
                or payload is None
                or payload.owner_identity is not self._identity
                or payload.runtime_lease is not self._env_lease
                or payload.control_step != step
                or getattr(graph, "active_cycle", None) is None
            ):
                self._poison_reward(
                    "after-Reward close lacks the exact active control-step graph"
                )
                raise ActionBallFullMdpRuntimeOwnerError(
                    "after-Reward close lacks the exact active control-step graph"
                )
            try:
                graph.close_after_reward()
            except BaseException as exc:
                self._poison_reward(
                    self._failure_reason("after-Reward graph close failed", exc)
                )
                raise
            if (
                self._active_pre_reward_publication is not None
                or getattr(graph, "active_cycle", None) is not None
            ):
                self._poison_reward(
                    "after-Reward graph close left transaction debt"
                )
                raise ActionBallFullMdpRuntimeOwnerError(
                    "after-Reward graph close left transaction debt"
                )
            return None

    def close_full_mdp_reward_cycle(
        self,
        *,
        runtime_lease: object,
        pre_reward_publication: object,
        ordered_owner_payment_results: object,
        ordered_consumers: object,
    ) -> ActionBallFullMdpRewardCloseReceipt:
        """Validate fourteen owner verdicts, then close four real owner epochs."""

        with self._lock:
            self.require_healthy()
            try:
                payload = _PRE_REWARD_PUBLICATION_REGISTRY.get(
                    pre_reward_publication
                )
            except TypeError:
                payload = None
            if (
                runtime_lease is not self._env_lease
                or pre_reward_publication is not self._active_pre_reward_publication
                or payload is None
                or payload is not self._active_pre_reward_payload
                or payload.owner_identity is not self._identity
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "Reward close publication/lease differs"
                )
            if (
                type(ordered_consumers) is not tuple
                or ordered_consumers != FULL_MDP_REWARD_ORDERED_CONSUMERS
                or type(ordered_owner_payment_results) is not tuple
                or len(ordered_owner_payment_results) != len(ordered_consumers)
                or any(value is None for value in ordered_owner_payment_results)
                or len({id(value) for value in ordered_owner_payment_results})
                != len(ordered_owner_payment_results)
            ):
                self._poison_reward(
                    "Reward owner verdicts are missing, duplicated, or reordered"
                )
                raise ActionBallFullMdpRuntimeOwnerError(
                    "Reward owner verdicts are missing, duplicated, or reordered"
                )
            owners = {
                "r03": self._r03,
                "physical": self._physical,
                "r06": self._r06,
                "r07": self._r07,
            }
            try:
                for consumer, verdict in zip(
                    ordered_consumers, ordered_owner_payment_results
                ):
                    owner_name, leaf_consumer = consumer.split(":", 1)
                    validated = owners[owner_name].require_owned_full_mdp_reward_payment(
                        verdict,
                        consumer=leaf_consumer,
                        control_step=payload.control_step,
                        runtime_owner=self,
                    )
                    if validated is not verdict:
                        raise ActionBallFullMdpRuntimeOwnerError(
                            f"{consumer} owner verdict identity differs"
                        )
                close_receipts = []
                for owner_name in FULL_MDP_REWARD_OWNER_ORDER:
                    owner = owners[owner_name]
                    consumers = dict(FULL_MDP_REWARD_OWNER_CONSUMERS)[owner_name]
                    verdicts = tuple(
                        ordered_owner_payment_results[
                            FULL_MDP_REWARD_ORDERED_CONSUMERS.index(
                                f"{owner_name}:{consumer}"
                            )
                        ]
                        for consumer in consumers
                    )
                    close_receipt = owner.close_full_mdp_reward_cycle(
                        control_step=payload.control_step,
                        pre_reward_publication=(
                            payload.r03_publication
                            if owner_name == "r03"
                            else payload.r07_publication
                            if owner_name == "r07"
                            else pre_reward_publication
                        ),
                        ordered_consumers=consumers,
                        ordered_payment_verdicts=verdicts,
                        runtime_owner=self,
                    )
                    owned_close = owner.require_owned_full_mdp_reward_close(
                        close_receipt,
                        control_step=payload.control_step,
                        runtime_owner=self,
                    )
                    if owned_close is not close_receipt:
                        raise ActionBallFullMdpRuntimeOwnerError(
                            f"{owner_name} Reward close identity differs"
                        )
                    close_receipts.append((owner_name, close_receipt))
                result = _mint_reward_close_receipt(
                    _RewardClosePayload(
                        owner_identity=self._identity,
                        pre_reward_publication=pre_reward_publication,
                        control_step=payload.control_step,
                        owner_close_receipts=tuple(close_receipts),
                    )
                )
            except BaseException as exc:
                # Payment/close is irreversible.  A partial close can never be
                # retried into success or be crossed by reset/optimizer.
                self._poison_reward(
                    self._failure_reason("Reward close failed after payment", exc)
                )
                raise
            self._active_pre_reward_publication = None
            self._active_pre_reward_payload = None
            return result

    def project_r05_true_reset(
        self,
        receipt: object,
        *,
        device: object,
        num_envs: int,
        live_reset_ledger_identity: object,
        live_reset_generation: object,
    ) -> object:
        """Project one env-issued event onto Device-R05 without a host read."""

        with self._lock:
            view = self._selected_reset_env_binding_view
            if (
                receipt is not self._selected_reset_event
                or self._selected_reset_prepared is not None
                or view is None
                or getattr(view, "device_r05_owner", None)
                is not self._device_r05
                or getattr(view, "live_reset_ledger_identity", None)
                is not live_reset_ledger_identity
                or type(num_envs) is not int
                or num_envs != self._num_envs
                or not _same_device(device, self._device)
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset event or live-ledger authority differs"
                )
            projector = getattr(
                self._env,
                "project_action_ball_full_mdp_selected_reset_event",
                None,
            )
            if not callable(projector):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "environment selected-reset event projector is not bound"
                )
            projection = projector(
                receipt,
                expected_top=self,
                device=device,
                num_envs=num_envs,
                live_reset_ledger_identity=live_reset_ledger_identity,
                live_reset_generation=live_reset_generation,
            )
            env_module = importlib.import_module(type(self._env).__module__)
            expected_type = getattr(
                env_module,
                "FullMdpSelectedResetProjection",
                None,
            )
            index = getattr(projection, "selected_env_index", None)
            mask = getattr(projection, "selected_mask", None)
            generation_before = getattr(
                projection,
                "generation_before",
                None,
            )
            generation_after = getattr(
                projection,
                "generation_after",
                None,
            )
            generation_overflow_fault = getattr(
                projection,
                "generation_overflow_fault",
                None,
            )
            if (
                expected_type is None
                or type(projection) is not expected_type
                or getattr(projection, "reset_event_identity", None) is None
                or type(index) is not type(live_reset_generation)
                or getattr(index, "ndim", None) != 1
                or getattr(index, "shape", (0,))[0] < 1
                or getattr(index, "device", None)
                != getattr(live_reset_generation, "device", None)
                or getattr(index, "dtype", None) is None
                or type(mask) is not type(live_reset_generation)
                or getattr(mask, "shape", None) != (self._num_envs,)
                or getattr(mask, "device", None)
                != getattr(live_reset_generation, "device", None)
                or type(generation_before) is not type(live_reset_generation)
                or type(generation_after) is not type(live_reset_generation)
                or type(generation_overflow_fault)
                is not type(live_reset_generation)
                or getattr(generation_before, "shape", None)
                != (self._num_envs,)
                or getattr(generation_after, "shape", None)
                != (self._num_envs,)
                or getattr(generation_overflow_fault, "shape", None)
                != (self._num_envs,)
                or getattr(generation_before, "dtype", None)
                != getattr(live_reset_generation, "dtype", None)
                or getattr(generation_after, "dtype", None)
                != getattr(live_reset_generation, "dtype", None)
                or getattr(generation_before, "device", None)
                != getattr(live_reset_generation, "device", None)
                or getattr(generation_after, "device", None)
                != getattr(live_reset_generation, "device", None)
                or getattr(generation_overflow_fault, "device", None)
                != getattr(live_reset_generation, "device", None)
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "environment selected-reset projection ABI differs"
                )
            # Selected reset is rare and has an irreversible four-owner park.
            # Perform one named packed D2H preflight before returning any
            # Device-R05 event projection.  A delayed CUDA assertion cannot
            # authorize the same batch, and comparing four caller/device
            # mirrors separately would add hidden synchronizations.
            torch = importlib.import_module("torch")
            if (
                type(index) is not torch.Tensor
                or type(mask) is not torch.Tensor
                or type(generation_before) is not torch.Tensor
                or type(generation_after) is not torch.Tensor
                or type(generation_overflow_fault) is not torch.Tensor
                or index.dtype != torch.int64
                or mask.dtype != torch.bool
                or generation_before.dtype != torch.int64
                or generation_after.dtype != torch.int64
                or generation_overflow_fault.dtype != torch.bool
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset projection is not exact Torch device state"
                )
            reconstructed_mask = torch.zeros_like(mask)
            reconstructed_mask.index_fill_(0, index, True)
            expected_overflow = torch.logical_and(
                mask,
                generation_before == torch.iinfo(torch.int64).max,
            )
            expected_after = generation_before + torch.logical_and(
                mask, ~expected_overflow
            ).to(torch.int64)
            order_fault = (
                torch.any(index[1:] <= index[:-1]).to(torch.int64)
                if index.shape[0] > 1
                else torch.zeros((), dtype=torch.int64, device=index.device)
            )
            packed_preflight = torch.stack(
                (
                    torch.sum(
                        torch.logical_xor(reconstructed_mask, mask),
                        dtype=torch.int64,
                    ),
                    order_fault,
                    torch.sum(
                        generation_before != live_reset_generation,
                        dtype=torch.int64,
                    ),
                    torch.sum(
                        generation_after != expected_after,
                        dtype=torch.int64,
                    ),
                    torch.sum(
                        generation_overflow_fault != expected_overflow,
                        dtype=torch.int64,
                    ),
                    torch.sum(expected_overflow, dtype=torch.int64),
                )
            )
            preflight_host = packed_preflight.to(
                device="cpu", non_blocking=False
            ).untyped_storage()
            preflight_bytes = bytes(preflight_host)[: packed_preflight.numel() * 8]
            if any(
                value != 0
                for value in struct.unpack(
                    "=" + "q" * packed_preflight.numel(),
                    preflight_bytes,
                )
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset packed preflight rejected selection, generation, or int64 continuation"
                )
            device_module = importlib.import_module(
                type(self._device_r05).__module__
            )
            result = device_module.DeviceTrueResetEventProjection(
                reset_event_identity=projection.reset_event_identity,
                selected_env_index=index,
                selected_mask=mask,
            )
            self._selected_reset_projection = projection
            return result

    def _poison_selected_reset(self, reason: str) -> None:
        """Fail-stop every selected-reset mutation owner without rollback."""

        clean = (
            reason
            if type(reason) is str and reason.strip()
            else "selected-reset transaction failure"
        )
        if self._poison_reason is None:
            self._poison_reason = clean
        self._poisoned = True
        failures = list(self._poison_failures)
        for name, child, method_name in (
            ("motion", self._motion, "poison_global_reveal_epoch"),
            ("racket", self._racket, "poison_global_reveal_epoch"),
            ("physical_ball", self._physical, "poison_selected_reset"),
            ("r06_flight", self._r06, "poison_selected_reset"),
        ):
            try:
                poison = getattr(child, method_name)
                poison(clean)
            except BaseException as exc:
                failures.append((name, type(exc).__name__))
        try:
            self._device_r05.poison_from_external_failure(11)
        except BaseException as exc:
            failures.append(("device_r05", type(exc).__name__))
        self._poison_failures = tuple(failures)

    def _abort_selected_reset_precommit(
        self,
        *,
        prepared: object,
        motion_value: object,
        racket_value: object,
        r06_prepared: object,
        physical_value: object,
    ) -> None:
        """Abort every private image in reverse order before R06 arm."""

        failures = []
        for name, value, abort in (
            (
                "physical_ball",
                physical_value,
                self._physical.abort_selected_true_reset,
            ),
            (
                "r06_flight",
                r06_prepared,
                self._r06.abort_selected_reset,
            ),
            (
                "racket",
                racket_value,
                self._racket.abort_prevalidated_action_ball_continuous_racket_selected_reset,
            ),
            (
                "motion",
                motion_value,
                self._motion.abort_prevalidated_action_ball_continuous_motion_selected_reset,
            ),
        ):
            if value is None:
                continue
            try:
                abort(value)
            except BaseException as exc:
                failures.append((name, type(exc).__name__))
        if prepared is not None:
            try:
                self._device_r05.abort_true_reset_many(prepared)
            except BaseException as exc:
                failures.append(("device_r05", type(exc).__name__))
        if failures:
            self._poison_failures = tuple(
                (*self._poison_failures, *failures)
            )
            self._poison_selected_reset(
                "selected-reset precommit abort was incomplete"
            )
            return
        self._selected_reset_event = None
        self._selected_reset_projection = None
        self._selected_reset_prepared = None
        self._selected_reset_child_commits = None
        self._selected_reset_child_commits_started = False
        self._selected_reset_r05_receipt = None
        self._selected_reset_completions = None

    def require_owned_r05_true_reset_commit(
        self,
        prepared: object,
        *,
        owner_view: object,
    ) -> object:
        """Settle Device-R05's exact writer view with four child commits.

        ``owner_view`` is minted inside ``DeviceR05Owner.commit_true_reset_many``
        from the retained prepare record.  It is accepted only at that callback
        boundary; the top never manufactures a substitute view or interprets
        its device fault tensor as a host-side success bit.
        """

        with self._lock:
            commits = self._selected_reset_child_commits
            projection = self._selected_reset_projection
            device_module = importlib.import_module(
                type(self._device_r05).__module__
            )
            owner_view_type = getattr(
                device_module,
                "DeviceR05TrueResetCommitInput",
                None,
            )
            selected_mask = getattr(owner_view, "selected_mask", None)
            generation_before = getattr(
                owner_view,
                "generation_before",
                None,
            )
            generation_after = getattr(
                owner_view,
                "generation_after",
                None,
            )
            generation_overflow_fault = getattr(
                owner_view,
                "generation_overflow_fault",
                None,
            )
            if (
                prepared is not self._selected_reset_prepared
                or type(commits) is not tuple
                or len(commits) != len(SELECTED_RESET_COMMIT_PROOF_ORDER)
                or projection is None
                or not self._selected_reset_child_commits_started
                or owner_view_type is None
                or type(owner_view) is not owner_view_type
                or getattr(owner_view, "prepared_true_reset", None)
                is not prepared
                or getattr(owner_view, "reset_event_identity", None)
                is not projection.reset_event_identity
                or type(selected_mask)
                is not type(projection.selected_mask)
                or getattr(selected_mask, "shape", None)
                != (self._num_envs,)
                or getattr(selected_mask, "device", None)
                != getattr(projection.selected_mask, "device", None)
                or getattr(selected_mask, "dtype", None)
                != getattr(projection.selected_mask, "dtype", None)
                or type(generation_before)
                is not type(projection.generation_before)
                or type(generation_after)
                is not type(projection.generation_after)
                or type(generation_overflow_fault)
                is not type(projection.selected_mask)
                or any(
                    getattr(value, "shape", None) != (self._num_envs,)
                    for value in (
                        generation_before,
                        generation_after,
                        generation_overflow_fault,
                    )
                )
                or any(
                    getattr(value, "device", None)
                    != getattr(projection.selected_mask, "device", None)
                    for value in (
                        generation_before,
                        generation_after,
                        generation_overflow_fault,
                    )
                )
                or getattr(generation_before, "dtype", None)
                != getattr(projection.generation_before, "dtype", None)
                or getattr(generation_after, "dtype", None)
                != getattr(projection.generation_after, "dtype", None)
                or getattr(generation_overflow_fault, "dtype", None)
                != getattr(projection.selected_mask, "dtype", None)
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset writer view or child proof differs"
                )
            validators = (
                self._motion.require_owned_selected_reset_commit,
                self._racket.require_owned_selected_reset_commit,
                self._physical.require_owned_selected_reset_commit,
                self._r06.require_owned_selected_reset_commit,
            )
            owned = tuple(
                validator(
                    commit,
                    expected_prepared_true_reset=prepared,
                )
                if name != "physical_ball"
                else validator(commit)
                for name, validator, commit in zip(
                    SELECTED_RESET_COMMIT_PROOF_ORDER,
                    validators,
                    commits,
                )
            )
            if any(value is not commit for value, commit in zip(owned, commits)):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset child commit validator changed identity"
                )
            return device_module.DeviceTrueResetCommitProjection(
                prepared_true_reset=prepared,
                reset_event_identity=projection.reset_event_identity,
                child_kinds=SELECTED_RESET_COMMIT_PROOF_ORDER,
                child_commit_identities=commits,
            )

    def require_owned_r05_true_reset_abort(
        self,
        prepared: object,
    ) -> object:
        """Prove the active reset has not crossed the irreversible cutoff."""

        with self._lock:
            projection = self._selected_reset_projection
            if (
                prepared is not self._selected_reset_prepared
                or projection is None
                or self._selected_reset_child_commits_started
                or self._selected_reset_child_commits is not None
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset abort proof is unavailable"
                )
            device_module = importlib.import_module(
                type(self._device_r05).__module__
            )
            return device_module.DeviceTrueResetAbortProjection(
                prepared_true_reset=prepared,
                reset_event_identity=projection.reset_event_identity,
                child_commits_started=False,
            )

    def require_owned_r05_true_reset_child_completion(
        self,
        receipt: object,
        *,
        child_kind: str,
        child_receipt: object,
    ) -> object:
        """Join one actual leaf completion to the exact R05-last receipt."""

        with self._lock:
            if (
                receipt is not self._selected_reset_r05_receipt
                or child_kind not in CHILD_COMPLETION_ORDER
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset child completion is not globally retained"
                )
            child_index = CHILD_COMPLETION_ORDER.index(child_kind)
            if (
                self._selected_reset_completions is None
                or self._selected_reset_completions[child_index]
                is not child_receipt
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset child completion identity differs"
                )
            device_module = importlib.import_module(
                type(self._device_r05).__module__
            )
            return device_module.DeviceTrueResetChildCompletionProjection(
                true_reset_receipt=receipt,
                child_kind=child_kind,
                child_receipt=child_receipt,
            )

    def selected_true_reset(
        self,
        event: object,
    ) -> object:
        """Run the fixed four-child reset transaction with Device-R05 last."""

        with self._lock:
            self.require_healthy()
            self._require_no_selected_reset_debt(operation="selected reset")
            if (
                event is None
                or self._selected_reset_event is not None
                or self._selected_reset_receipt is not None
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "one selected-reset event or env receipt is already active"
                )
            self._selected_reset_event = event
            self._selected_reset_projection = None
            self._selected_reset_prepared = None
            self._selected_reset_child_commits = None
            self._selected_reset_child_commits_started = False
            prepared = None
            motion_value = None
            racket_value = None
            r06_prepared = None
            physical_value = None
            irreversible = False
            try:
                prepared = self._device_r05.prepare_true_reset_many(event)
                self._selected_reset_prepared = prepared

                motion_stage = (
                    self._motion
                    .prepare_action_ball_continuous_motion_selected_reset(
                        prepared
                    )
                )
                motion_value = motion_stage
                motion_prevalidated = (
                    self._motion
                    .arm_prevalidated_action_ball_continuous_motion_selected_reset(
                        motion_stage
                    )
                )
                motion_value = motion_prevalidated

                racket_stage = (
                    self._racket
                    .stage_action_ball_continuous_racket_selected_reset(
                        prepared
                    )
                )
                racket_value = racket_stage
                racket_prevalidated = (
                    self._racket
                    .finalize_action_ball_continuous_racket_selected_reset(
                        racket_stage
                    )
                )
                racket_value = racket_prevalidated

                r06_prepared = self._r06.prepare_selected_reset(prepared)
                physical_stage = self._physical.stage_selected_true_reset(
                    r06_prepared
                )
                physical_value = physical_stage
                physical_finalized = (
                    self._physical.finalize_selected_true_reset(
                        physical_stage
                    )
                )
                physical_value = physical_finalized

                # R06 cannot abort once armed.  Set the irreversible boundary
                # before the arm call so an arm exception is poison-only.
                irreversible = True
                r06_armed = self._r06.arm_prevalidated_selected_reset(
                    r06_prepared,
                    physical_finalized,
                )
                physical_armed = self._physical.prearm_selected_true_reset(
                    physical_finalized,
                    r06_armed,
                )

                self._selected_reset_child_commits_started = True
                physical_commit = (
                    self._physical.commit_prevalidated_selected_true_reset(
                        physical_armed
                    )
                )
                if (
                    self._physical.require_owned_selected_reset_commit(
                        physical_commit
                    )
                    is not physical_commit
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "physical selected-reset commit identity differs"
                    )
                r06_commit = self._r06.commit_prevalidated_selected_reset(
                    r06_armed,
                    physical_commit,
                )
                if (
                    self._r06.require_owned_selected_reset_commit(
                        r06_commit,
                        expected_prepared_true_reset=prepared,
                    )
                    is not r06_commit
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "R06 selected-reset commit identity differs"
                    )
                self._physical.acknowledge_r06_selected_reset_commit(
                    physical_commit,
                    r06_commit,
                )
                motion_commit = (
                    self._motion
                    .commit_prevalidated_action_ball_continuous_motion_selected_reset(
                        motion_prevalidated
                    )
                )
                racket_commit = (
                    self._racket
                    .commit_prevalidated_action_ball_continuous_racket_selected_reset(
                        racket_prevalidated
                    )
                )
                # Prevalidate all four opaque child identities before asking
                # Device-R05 to cross its last-write boundary.  Device-R05
                # then calls ``require_owned_r05_true_reset_commit`` with its
                # own clone-only writer view; the top never forges that view.
                prevalidated_commits = (
                    self._motion.require_owned_selected_reset_commit(
                        motion_commit,
                        expected_prepared_true_reset=prepared,
                    ),
                    self._racket.require_owned_selected_reset_commit(
                        racket_commit,
                        expected_prepared_true_reset=prepared,
                    ),
                    physical_commit,
                    r06_commit,
                )
                commits = (
                    motion_commit,
                    racket_commit,
                    physical_commit,
                    r06_commit,
                )
                if any(
                    owned is not commit
                    for owned, commit in zip(prevalidated_commits, commits)
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "selected-reset child precommit identity differs"
                    )
                self._selected_reset_child_commits = commits
                # Device-R05 invokes the top authority callback internally
                # and publishes its mutation only after that proof returns.
                r05_receipt = self._device_r05.commit_true_reset_many(prepared)

                motion_completion = (
                    self._motion
                    .complete_action_ball_continuous_motion_selected_reset_after_r05(
                        motion_commit,
                        r05_receipt,
                    )
                )
                racket_completion = (
                    self._racket
                    .complete_action_ball_continuous_racket_selected_reset_after_r05(
                        racket_commit,
                        r05_receipt,
                    )
                )
                r06_completion = self._r06.complete_selected_reset_after_r05(
                    r06_commit,
                    r05_receipt,
                )
                physical_completion = (
                    self._physical.complete_selected_true_reset_after_r05(
                        physical_commit,
                        r06_commit,
                        r05_receipt,
                    )
                )
                completions = (
                    motion_completion,
                    racket_completion,
                    r06_completion,
                    physical_completion,
                )
                self._selected_reset_r05_receipt = r05_receipt
                self._selected_reset_completions = completions
                validated = (
                    self._motion.require_owned_selected_reset_completion(
                        motion_completion,
                        expected_prepared_true_reset=prepared,
                    ),
                    self._racket.require_owned_selected_reset_completion(
                        racket_completion,
                        expected_prepared_true_reset=prepared,
                    ),
                    self._r06.require_owned_selected_reset_completion(
                        r06_completion
                    ),
                    self._physical.require_owned_selected_reset_completion(
                        physical_completion
                    ),
                )
                if any(
                    owned is not completion
                    for owned, completion in zip(validated, completions)
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "selected-reset completion identity differs"
                    )
                self._selected_reset_sequence += 1
                receipt = _mint_selected_true_reset_receipt(
                    _SelectedTrueResetReceiptPayload(
                        owner_identity=self._identity,
                        event=event,
                        prepared_true_reset=prepared,
                        device_r05_receipt=r05_receipt,
                        child_completions=completions,
                        sequence=self._selected_reset_sequence,
                    )
                )
                self._selected_reset_receipt = receipt
                self._selected_reset_receipt_consumed = False
                # Device-R05 must not self-sign the four completion columns.
                # Feed each retained owner-issued completion back through this
                # construction-bound top authority before any leaf consumes
                # its capability or a drain/checkpoint can begin.
                for child_kind, completion in zip(
                    CHILD_COMPLETION_ORDER, completions
                ):
                    self._device_r05.record_true_reset_child_completion(
                        r05_receipt,
                        child_kind=child_kind,
                        child_receipt=completion,
                    )
                consumed = (
                    self._motion.consume_owned_selected_reset_completion(
                        motion_completion,
                        expected_prepared_true_reset=prepared,
                    ),
                    self._racket.consume_owned_selected_reset_completion(
                        racket_completion,
                        expected_prepared_true_reset=prepared,
                    ),
                    self._r06.consume_owned_selected_reset_completion(
                        r06_completion
                    ),
                    self._physical.consume_owned_selected_reset_completion(
                        physical_completion
                    ),
                )
                if any(
                    owned is not completion
                    for owned, completion in zip(consumed, completions)
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "selected-reset completion consume identity differs"
                    )
                return receipt
            except BaseException as exc:
                if irreversible:
                    self._poison_selected_reset(
                        self._failure_reason(
                            "selected reset failed after R06 irreversible arm",
                            exc,
                        )
                    )
                else:
                    self._abort_selected_reset_precommit(
                        prepared=prepared,
                        motion_value=motion_value,
                        racket_value=racket_value,
                        r06_prepared=r06_prepared,
                        physical_value=physical_value,
                    )
                raise

    def require_owned_selected_true_reset_receipt(
        self,
        receipt: object,
        expected_event: object,
    ) -> object:
        """Consume the exact safe-settlement receipt for the env event once."""

        with self._lock:
            payload = (
                _lookup_selected_true_reset_receipt(receipt)
                if type(receipt) is ActionBallFullMdpSelectedTrueResetReceipt
                else None
            )
            if (
                payload is None
                or receipt is not self._selected_reset_receipt
                or payload.owner_identity is not self._identity
                or payload.event is not expected_event
                or expected_event is not self._selected_reset_event
                or payload.prepared_true_reset
                is not self._selected_reset_prepared
                or self._selected_reset_receipt_consumed
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "selected-reset receipt is stale, foreign, or replayed"
                )
            self._selected_reset_receipt_consumed = True
            self._selected_reset_event = None
            self._selected_reset_projection = None
            self._selected_reset_prepared = None
            self._selected_reset_child_commits = None
            self._selected_reset_child_commits_started = False
            self._selected_reset_r05_receipt = None
            self._selected_reset_completions = None
            self._selected_reset_receipt = None
            return receipt

    def bind_r10_checkpoint_publication_authority(
        self,
        consumer: object,
        validator: object,
    ) -> None:
        """Construction-bind the global, post-publication R10 ACK authority."""

        with self._lock:
            if (
                not self._audit_consumer_binding_open
                or consumer is None
                or not callable(validator)
                or getattr(validator, "__self__", None) is not consumer
                or self._r10_checkpoint_publication_consumer is not None
                or self._r10_checkpoint_publication_validator is not None
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "R10 checkpoint publication authority is not construction-bound"
                )
            self._r10_checkpoint_publication_consumer = consumer
            self._r10_checkpoint_publication_validator = validator

    def bind_r11_audit_consumer(
        self,
        consumer: object,
        validator: object,
    ) -> None:
        """Construction-bind one exact monitor consumer and its row validator."""

        with self._lock:
            if (
                not self._audit_consumer_binding_open
                or consumer is None
                or not callable(validator)
                or getattr(validator, "__self__", None) is not consumer
                or self._r11_audit_consumer is not None
                or self._r11_audit_consumer_validator is not None
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "R11 audit consumer is not construction-bound"
                )
            self._r11_audit_consumer = consumer
            self._r11_audit_consumer_validator = validator

    def _audit_claim_root(
        self,
        consumer_kind: str,
        row: Optional[_AuditFrontierRow],
        drain_frontier_sha256: str,
    ) -> str:
        return _canonical_sha256(
            {
                "schema_version": RUNTIME_OWNER_SCHEMA_VERSION,
                "kind": AUDIT_FRONTIER_CLAIM_KIND,
                "consumer_kind": consumer_kind,
                "drain_sequence": 0 if row is None else row.drain_sequence,
                "update_index": -1 if row is None else row.update_index,
                "completed_environment_steps": (
                    -1 if row is None else row.completed_environment_steps
                ),
                "audit_row_sha256": (
                    None if row is None else row.canonical_sha256
                ),
                "r03_receipt_sha256": (
                    None if row is None else row.r03_receipt_sha256
                ),
                "r07_receipt_sha256": (
                    None if row is None else row.r07_receipt_sha256
                ),
                "ppo_drain_frontier_sha256": drain_frontier_sha256,
            }
        )

    def _mint_audit_frontier_claim(
        self,
        consumer_kind: str,
        consumer_identity: object,
        row: Optional[_AuditFrontierRow],
        drain_snapshot: object,
        drain_frontier_sha256: object,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        drain_root = _require_sha256(
            drain_frontier_sha256,
            label="PPO drain checkpoint frontier",
        )
        if (
            drain_snapshot is not None
            and getattr(drain_snapshot, "checkpoint_frontier_sha256", None)
            != drain_root
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "PPO drain checkpoint frontier differs from its snapshot"
            )
        return _mint_audit_frontier_claim(
            _AuditFrontierClaimPayload(
                owner_identity=self._identity,
                consumer_kind=consumer_kind,
                consumer_identity=consumer_identity,
                row=row,
                drain_snapshot=drain_snapshot,
                drain_frontier_sha256=drain_root,
                canonical_sha256=self._audit_claim_root(
                    consumer_kind,
                    row,
                    drain_root,
                ),
            )
        )

    def _claim_r10_audit_frontier(
        self,
        drain_snapshot: object,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        """Join an owner-issued drain snapshot to the latest retained audit row."""

        with self._lock:
            self.require_healthy()
            sequence = _plain_nonnegative_int(
                getattr(drain_snapshot, "drain_sequence", None),
                label="PPO drain checkpoint drain_sequence",
            )
            next_update = _plain_nonnegative_int(
                getattr(drain_snapshot, "next_update_index", None),
                label="PPO drain checkpoint next_update_index",
            )
            completed = getattr(
                drain_snapshot,
                "last_completed_environment_steps",
                None,
            )
            highwaters = getattr(
                drain_snapshot,
                "mutation_version_highwaters",
                None,
            )
            if type(highwaters) is not tuple:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "PPO drain checkpoint mutation highwaters differ"
                )
            row = None
            if sequence == 0:
                if (
                    completed != -1
                    or any(value is not None for _name, value in highwaters)
                    or self._audit_frontier_ring
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "empty PPO drain frontier disagrees with audit ring"
                    )
            else:
                if (
                    type(completed) is not int
                    or completed < 0
                    or not self._audit_frontier_ring
                    or any(value is None for _name, value in highwaters)
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "nonempty PPO drain frontier lacks an exact audit row"
                    )
                row = self._audit_frontier_ring[-1]
                if (
                    row.drain_sequence != sequence
                    or row.update_index + 1 != next_update
                    or row.completed_environment_steps != completed
                ):
                    raise ActionBallFullMdpRuntimeOwnerError(
                        "PPO drain checkpoint chronology differs from audit ring"
                    )
            active = self._active_r10_audit_claim
            active_payload = (
                _lookup_audit_frontier_claim(active)
                if type(active) is ActionBallFullMdpAuditFrontierClaim
                else None
            )
            if (
                active_payload is not None
                and active_payload.owner_identity is self._identity
                and active_payload.consumer_kind == "r10"
                and active_payload.row is row
                and active_payload.drain_snapshot is drain_snapshot
            ):
                return active
            consumer_identity = self._checkpoint_join_snapshot_provider
            claim = self._mint_audit_frontier_claim(
                "r10",
                consumer_identity,
                row,
                drain_snapshot,
                getattr(drain_snapshot, "checkpoint_frontier_sha256", None),
            )
            self._active_r10_audit_claim = claim
            return claim

    def _require_owned_audit_frontier_claim(
        self,
        claim: object,
        consumer_kind: str,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        payload = (
            _lookup_audit_frontier_claim(claim)
            if type(claim) is ActionBallFullMdpAuditFrontierClaim
            else None
        )
        expected_active = (
            self._active_r10_audit_claim
            if consumer_kind == "r10"
            else self._active_r11_audit_claim
        )
        expected_consumer = (
            self._checkpoint_join_snapshot_provider
            if consumer_kind == "r10"
            else self._r11_audit_consumer
        )
        if (
            consumer_kind not in ("r10", "r11")
            or payload is None
            or claim is not expected_active
            or payload.owner_identity is not self._identity
            or payload.consumer_kind != consumer_kind
            or payload.consumer_identity is not expected_consumer
            or payload.canonical_sha256
            != self._audit_claim_root(
                consumer_kind,
                payload.row,
                payload.drain_frontier_sha256,
            )
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "audit frontier claim is stale, foreign, or content-mismatched"
            )
        return claim

    def finalize_r10_audit_frontier(
        self,
        publication: object,
        claim: object,
    ) -> None:
        """Advance R10 only after an exact global checkpoint publication."""

        with self._lock:
            owned = self._require_owned_audit_frontier_claim(claim, "r10")
            validator = self._r10_checkpoint_publication_validator
            try:
                validated = (
                    None
                    if not callable(validator)
                    else validator(publication, owned)
                )
            except BaseException as exc:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R10 global checkpoint publication validation failed"
                ) from exc
            if (
                self._r10_checkpoint_publication_consumer is None
                or not callable(validator)
                or validated is not publication
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "R10 global checkpoint publication validator is not bound"
                )
            payload = _lookup_audit_frontier_claim(owned)
            if payload is None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R10 audit claim registry entry disappeared"
                )
            if payload.row is not None:
                self._r10_audit_highwater = max(
                    self._r10_audit_highwater,
                    payload.row.drain_sequence,
                )
            self._active_r10_audit_claim = None
            self._garbage_collect_audit_frontier()

    def claim_r11_audit_frontier(
        self,
        consumer: object,
    ) -> ActionBallFullMdpAuditFrontierClaim:
        """Lease the monitor's earliest unacknowledged retained audit row."""

        with self._lock:
            self.require_healthy()
            if (
                consumer is not self._r11_audit_consumer
                or self._r11_audit_consumer_validator is None
            ):
                raise ActionBallFullMdpRuntimeDependencyError(
                    "R11 exact audit consumer is not bound"
                )
            if self._active_r11_audit_claim is not None:
                return self._require_owned_audit_frontier_claim(
                    self._active_r11_audit_claim,
                    "r11",
                )
            row = next(
                (
                    value
                    for value in self._audit_frontier_ring
                    if value.drain_sequence > self._r11_audit_highwater
                ),
                None,
            )
            if row is None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R11 audit frontier has no unacknowledged row"
                )
            claim = self._mint_audit_frontier_claim(
                "r11",
                consumer,
                row,
                None,
                row.canonical_sha256,
            )
            self._active_r11_audit_claim = claim
            return claim

    def acknowledge_r11_audit_frontier(
        self,
        consumer: object,
        claim: object,
    ) -> None:
        """Validate the exact leaf receipts inside the bound monitor callback."""

        with self._lock:
            if consumer is not self._r11_audit_consumer:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R11 audit consumer identity differs"
                )
            owned = self._require_owned_audit_frontier_claim(claim, "r11")
            payload = _lookup_audit_frontier_claim(owned)
            validator = self._r11_audit_consumer_validator
            try:
                validated = (
                    None
                    if not callable(validator) or payload is None or payload.row is None
                    else validator(
                        owned,
                        payload.row.r03_receipt,
                        payload.row.r07_receipt,
                    )
                )
            except BaseException as exc:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R11 monitor audit validation failed"
                ) from exc
            if (
                payload is None
                or payload.row is None
                or not callable(validator)
                or validated is not owned
            ):
                raise ActionBallFullMdpRuntimeOwnerError(
                    "R11 monitor did not acknowledge the exact audit row"
                )
            self._r11_audit_highwater = payload.row.drain_sequence
            self._active_r11_audit_claim = None
            self._garbage_collect_audit_frontier()

    def _garbage_collect_audit_frontier(self) -> None:
        retained_floor = min(
            self._r10_audit_highwater,
            self._r11_audit_highwater,
        )
        self._audit_frontier_ring[:] = [
            row
            for row in self._audit_frontier_ring
            if row.drain_sequence > retained_floor
        ]

    def prepare_pre_optimizer_ppo_boundary(
        self,
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        with self._lock:
            self.require_healthy()
            self._require_no_selected_reset_debt(operation="PPO drain")
            update = _plain_nonnegative_int(
                update_index,
                label="update_index",
            )
            completed = _plain_nonnegative_int(
                completed_environment_steps,
                label="completed_environment_steps",
            )
            if self._active_optimizer_receipt is not None:
                raise ActionBallFullMdpRuntimeOwnerError(
                    "one optimizer drain receipt is already active"
                )
            retained_floor = min(
                self._r10_audit_highwater,
                self._r11_audit_highwater,
            )
            unconsumed = sum(
                1
                for row in self._audit_frontier_ring
                if row.drain_sequence > retained_floor
            )
            if unconsumed >= AUDIT_FRONTIER_RING_CAPACITY:
                raise ActionBallFullMdpRuntimeDependencyError(
                    "R03/R07 top audit frontier ring would overwrite "
                    "unconsumed R10/R11 evidence"
                )
            try:
                prepared = self._ppo_drain.prepare_pre_optimizer_ppo_boundary(
                    update_index=update,
                    completed_environment_steps=completed,
                )
                receipt = self._ppo_drain.transfer_decode_pre_optimizer_ppo_boundary(
                    prepared
                )
            except BaseException as exc:
                self._poison_global(
                    self._failure_reason("global PPO drain failed", exc)
                )
                raise
            self._active_optimizer_receipt = receipt
            self._active_optimizer_update_index = update
            return receipt

    def _require_active_optimizer_receipt(
        self,
        receipt: object,
        update_index: int,
    ) -> object:
        update = _plain_nonnegative_int(update_index, label="update_index")
        if (
            receipt is not self._active_optimizer_receipt
            or update != self._active_optimizer_update_index
        ):
            raise ActionBallFullMdpRuntimeOwnerError(
                "optimizer drain receipt identity/update differs"
            )
        return receipt

    def mark_optimizer_returned(
        self,
        receipt: object,
        *,
        update_index: int,
    ) -> None:
        with self._lock:
            self.require_healthy()
            owned = self._require_active_optimizer_receipt(
                receipt,
                update_index,
            )
            try:
                self._ppo_drain.mark_optimizer_returned(owned)
            except BaseException as exc:
                self._poison_global(
                    self._failure_reason(
                        "optimizer-return acknowledgement failed",
                        exc,
                    )
                )
                raise

    def acknowledge_post_update(
        self,
        receipt: object,
        *,
        update_index: int,
    ) -> None:
        with self._lock:
            self.require_healthy()
            owned = self._require_active_optimizer_receipt(
                receipt,
                update_index,
            )
            try:
                self._ppo_drain.acknowledge_post_update(owned)
                r03_receipt = (
                    self._r03.require_owned_pre_optimizer_ppo_boundary_receipt(
                        owned
                    )
                )
                r07_receipt = (
                    self._r07.require_owned_pre_optimizer_ppo_boundary_receipt(
                        owned
                    )
                )
            except BaseException as exc:
                self._poison_global(
                    self._failure_reason(
                        "post-update drain acknowledgement failed",
                        exc,
                    )
                )
                raise
            sequence = _plain_positive_int(
                getattr(owned, "drain_sequence", None),
                label="global drain_sequence",
            )
            update = _plain_nonnegative_int(
                getattr(owned, "update_index", None),
                label="global drain update_index",
            )
            completed = _plain_nonnegative_int(
                getattr(owned, "completed_environment_steps", None),
                label="global drain completed_environment_steps",
            )
            r03_root = _portable_receipt_sha256(
                r03_receipt,
                label="R03 portable audit receipt",
            )
            r07_root = _portable_receipt_sha256(
                r07_receipt,
                label="R07 portable audit receipt",
            )
            row_root = _canonical_sha256(
                {
                    "kind": "action_ball_full_mdp_audit_frontier_row_v1",
                    "drain_sequence": sequence,
                    "update_index": update,
                    "completed_environment_steps": completed,
                    "r03_receipt_sha256": r03_root,
                    "r07_receipt_sha256": r07_root,
                }
            )
            self._audit_frontier_ring.append(
                _AuditFrontierRow(
                    drain_sequence=sequence,
                    update_index=update,
                    completed_environment_steps=completed,
                    global_receipt=owned,
                    r03_receipt=r03_receipt,
                    r07_receipt=r07_receipt,
                    r03_receipt_sha256=r03_root,
                    r07_receipt_sha256=r07_root,
                    canonical_sha256=row_root,
                )
            )
            self._garbage_collect_audit_frontier()
            self._active_optimizer_receipt = None
            self._active_optimizer_update_index = None

    def poison_optimizer_boundary(
        self,
        receipt_or_none: object,
        *,
        update_index: int,
        reason: str,
    ) -> None:
        with self._lock:
            update = _plain_nonnegative_int(update_index, label="update_index")
            clean = reason if type(reason) is str and reason.strip() else (
                "optimizer failed after full-MDP drain"
            )
            receipt = self._active_optimizer_receipt
            if receipt is not None and (
                receipt_or_none is not receipt
                or update != self._active_optimizer_update_index
            ):
                clean = "foreign optimizer poison receipt/update"
            self._poison_global(clean)
            if receipt is not None:
                try:
                    self._ppo_drain.poison_optimizer_failure(
                        receipt,
                        reason=clean,
                    )
                except BaseException as exc:
                    self._poison_failures = tuple(
                        (
                            *self._poison_failures,
                            ("ppo_drain", type(exc).__name__),
                        )
                    )


__all__ = [
    "RUNTIME_OWNER_SCHEMA_VERSION",
    "RUNTIME_OWNER_KIND",
    "RUNTIME_DEPENDENCY_INVENTORY_KIND",
    "CHECKPOINT_JOIN_SNAPSHOT_KIND",
    "CHECKPOINT_JOIN_STATE_KIND",
    "FULL_MDP_REWARD_ORDERED_CONSUMERS",
    "FULL_MDP_REWARD_OWNER_ORDER",
    "RUNTIME_INTEGRATED",
    "POST_PHYSICS_INTEGRATED",
    "SELECTED_RESET_INTEGRATED",
    "PPO_DRAIN_BINDINGS_INTEGRATED",
    "R10_SHARED_JOIN_PROVIDER_IMPLEMENTED",
    "R10_SHARED_JOIN_PROVIDER_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "DIAGNOSTIC_UNAUTHORIZED",
    "CHILD_COMPLETION_ORDER",
    "GLOBAL_POISON_ORDER",
    "PROVIDER_API_SCHEMA_SHA256",
    "ActionBallFullMdpRuntimeOwner",
    "ActionBallFullMdpRewardOwnerBinding",
    "ActionBallFullMdpPreRewardPublication",
    "ActionBallFullMdpPreRewardView",
    "ActionBallFullMdpRewardCloseReceipt",
    "ActionBallFullMdpRuntimeOwnerError",
    "ActionBallFullMdpRuntimeDependencyError",
    "ActionBallFullMdpRuntimePoisonedError",
    "ActionBallFullMdpRuntimeDependencySpec",
    "ActionBallFullMdpRuntimeDependencyObservation",
    "ActionBallFullMdpRuntimeDependencyInventory",
    "ActionBallFullMdpWorldResetJoinRow",
    "ActionBallFullMdpTaskBallR06JoinRow",
    "ActionBallFullMdpCheckpointJoinSnapshot",
    "ActionBallFullMdpCheckpointJoinSnapshotProvider",
    "ActionBallFullMdpSelectedTrueResetReceipt",
    "action_ball_full_mdp_runtime_dependency_inventory",
]
