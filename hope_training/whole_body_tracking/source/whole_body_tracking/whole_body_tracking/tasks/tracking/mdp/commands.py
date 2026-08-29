from __future__ import annotations

import hashlib
import io
import json
import math
import numpy as np
import os
import stat
import struct
import sys
import torch
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.event_timing import (
    EVENT_TIMING_MODE_DISABLED,
    EVENT_TIMING_MODE_POST_STRIKE_T1,
    EVENT_TIMING_MODES,
    EventTimingScheduler,
    load_event_schedule,
)
from whole_body_tracking.tasks.tracking.mdp.post_swing_teacher import (
    CAPTURE_CLAIM_KIND,
    CAPTURE_CLAIM_NAME,
    CAPTURE_CONTRACT,
    CAPTURE_RESULT_KIND,
    CAPTURE_RESULT_NAME,
    CAPTURE_STATE_NAME,
    PostSwingTeacherError,
    _canonical_json_bytes,
    _publish_bytes_no_clobber,
    load_post_swing_teacher_states,
    sha256_file,
)
from whole_body_tracking.tasks.tracking.mdp.planner_revision import (
    InitialTtsMixture,
    PLANNER_TASK_REVISION_SCHEMA_VERSION,
    PhaseGovernorProfile,
)
try:
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_full_mdp_portable_catalog as _FULL_MDP_PORTABLE_CATALOG,
    )
except ImportError:  # Dependency-light spec-loaded command tests.
    import importlib.util as _importlib_util

    _portable_name = "_action_ball_full_mdp_portable_catalog_for_commands"
    _FULL_MDP_PORTABLE_CATALOG = sys.modules.get(_portable_name)
    if _FULL_MDP_PORTABLE_CATALOG is None:
        _portable_spec = _importlib_util.spec_from_file_location(
            _portable_name,
            Path(__file__).resolve().with_name(
                "action_ball_full_mdp_portable_catalog.py"
            ),
        )
        if _portable_spec is None or _portable_spec.loader is None:
            raise ImportError("cannot load the portable FullMDP catalog")
        _FULL_MDP_PORTABLE_CATALOG = _importlib_util.module_from_spec(
            _portable_spec
        )
        sys.modules[_portable_name] = _FULL_MDP_PORTABLE_CATALOG
        _portable_spec.loader.exec_module(_FULL_MDP_PORTABLE_CATALOG)

ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT = (
    _FULL_MDP_PORTABLE_CATALOG.ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT
)
ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND = (
    _FULL_MDP_PORTABLE_CATALOG.ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND
)
ActionBallFullMdpDiagnosticCatalogTable = (
    _FULL_MDP_PORTABLE_CATALOG.ActionBallFullMdpDiagnosticCatalogTable
)
load_action_ball_full_mdp_diagnostic_catalog_table = (
    _FULL_MDP_PORTABLE_CATALOG.load_action_ball_full_mdp_diagnostic_catalog_table
)
_ACTION_BALL_FULL_MDP_FRESH_REFERENCE_DUE_COUNT = (
    _FULL_MDP_PORTABLE_CATALOG.FRESH_REFERENCE_DUE_COUNT
)
_ACTION_BALL_FULL_MDP_FRESH_EPISODE_HORIZON_TICKS = (
    _FULL_MDP_PORTABLE_CATALOG.FRESH_EPISODE_HORIZON_TICKS
)
_ACTION_BALL_FULL_MDP_FRESH_SCHEDULE_EXHAUSTED_TIME_S = (
    _FULL_MDP_PORTABLE_CATALOG.
    FRESH_SCHEDULE_EXHAUSTED_TIME_TO_NEXT_OPPORTUNITY_S
)

try:
    import action_ball_full_mdp_row_identity as _ACTION_BALL_ROW_IDENTITY
except ImportError:  # pragma: no cover - installed package import
    from whole_body_tracking import (
        action_ball_full_mdp_row_identity as _ACTION_BALL_ROW_IDENTITY,
    )


def _stand_start_yaw_samples(yaw_range, count: int, device):
    """Return stand-start yaw samples, or ``None`` for the byte-identical [0, 0] default.

    A degenerate non-zero range is a deterministic curriculum point, not an off switch.
    Avoiding an RNG draw there also makes fixed-yaw evaluation exactly reproducible.
    """
    yaw_lo, yaw_hi = (float(yaw_range[0]), float(yaw_range[1]))
    if yaw_lo == 0.0 and yaw_hi == 0.0:
        return None
    if yaw_lo == yaw_hi:
        return torch.full((count,), yaw_lo, device=device)
    return sample_uniform(yaw_lo, yaw_hi, (count,), device)


def _motion_anchor_relative_body_transform(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    *,
    expected_body_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-anchor body poses with one rigid transform per environment.

    The transform is independent of the tracked-body index. Keep that value
    at ``[N, *]`` until the IsaacLab quaternion kernels require matching
    ``[N, B, *]`` shapes, then use zero-stride views rather than materializing
    four repeated anchor tensors. Inputs are never mutated.
    """

    if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
        raise ValueError(
            "motion anchor tensor shape differs from the configured body layout"
        )
    batch_size = body_pos_w.shape[0]
    body_count = body_pos_w.shape[1]
    if (
        body_quat_w.ndim != 3
        or tuple(body_quat_w.shape) != (batch_size, body_count, 4)
        or tuple(anchor_pos_w.shape) != (batch_size, 3)
        or tuple(anchor_quat_w.shape) != (batch_size, 4)
        or tuple(robot_anchor_pos_w.shape) != (batch_size, 3)
        or tuple(robot_anchor_quat_w.shape) != (batch_size, 4)
        or (
            expected_body_count is not None
            and body_count != expected_body_count
        )
    ):
        raise ValueError(
            "motion anchor tensor shape differs from the configured body layout"
        )

    delta_pos_w = robot_anchor_pos_w.clone()
    delta_pos_w[..., 2] = anchor_pos_w[..., 2]
    delta_ori_w = yaw_quat(
        quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w))
    )
    delta_pos_w_by_body = delta_pos_w[:, None, :].expand(
        -1, body_count, -1
    )
    delta_ori_w_by_body = delta_ori_w[:, None, :].expand(
        -1, body_count, -1
    )
    anchor_pos_w_by_body = anchor_pos_w[:, None, :].expand(
        -1, body_count, -1
    )
    return (
        quat_mul(delta_ori_w_by_body, body_quat_w),
        delta_pos_w_by_body
        + quat_apply(
            delta_ori_w_by_body,
            body_pos_w - anchor_pos_w_by_body,
        ),
    )


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_CANONICAL_REGISTRY_RUNTIME_MODULE = None
# 起点扰动斜坡的法则住在 utils/training_contract.py。这个文件在依赖极轻的测试里
# 是被 spec_from_file_location 单独加载的("whole_body_tracking 不是包"),所以
# 这里不能写普通 import,必须和 canonical registry 一样按仓库路径加载真字节。
_TRAINING_CONTRACT_RUNTIME_MODULE = None
_ACTION_BALL_RUNTIME_MODULE = None


ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE = 0
ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX = 1
ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED = 2

# ActionEpoch owns the packed fault namespace and validates these values again
# when the exact Motion owner is construction-bound.  Motion keeps only the
# four producer-side constants needed to avoid a hot import in each tick.
_ACTION_EPOCH_ROW_FAULT_MOTION_CADENCE_OVERDUE = 1 << 23
_ACTION_EPOCH_ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW = 1 << 24
_ACTION_EPOCH_ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT = 1 << 25
_ACTION_EPOCH_ROW_FAULT_MOTION_TASK_TIMING_CONTRACT = 1 << 26
_ACTION_EPOCH_MOTION_ROW_FAULT_BINDINGS = (
    (
        "ROW_FAULT_MOTION_CADENCE_OVERDUE",
        _ACTION_EPOCH_ROW_FAULT_MOTION_CADENCE_OVERDUE,
    ),
    (
        "ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW",
        _ACTION_EPOCH_ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW,
    ),
    (
        "ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT",
        _ACTION_EPOCH_ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
    ),
    (
        "ROW_FAULT_MOTION_TASK_TIMING_CONTRACT",
        _ACTION_EPOCH_ROW_FAULT_MOTION_TASK_TIMING_CONTRACT,
    ),
)
_ACTION_EPOCH_MOTION_ROW_FAULT_BITS = frozenset(
    value for _name, value in _ACTION_EPOCH_MOTION_ROW_FAULT_BINDINGS
)


@dataclass(frozen=True)
class ActionBallFullMdpCompletedActionFrame0Reference:
    """Device-only Motion frame-0 recovery reference for R07.

    Genesis IDLE uses Motion's code-owned upcoming schedule while carrying a
    neutral shot key.  Only the epoch-owned completed lifecycle carries an
    exact full shot key and selects that completed action.  This is a
    clone-only Motion projection, not an admission token.  Callers cannot
    supply an action, verdict, suffix receipt, or graph-cycle handle.
    """

    motion_owner: object
    epoch_owner: object
    epoch_version: int
    cadence_tick: torch.Tensor
    shot_key: _ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey
    reference_kind: torch.Tensor
    reference_action_slot: torch.Tensor
    reference_action_uid: torch.Tensor
    root_position_m: torch.Tensor
    root_orientation_wxyz: torch.Tensor
    joint_position_rad: torch.Tensor
    body_position_m: torch.Tensor
    body_orientation_wxyz: torch.Tensor
    station_anchor_xy_m: torch.Tensor
    validity: torch.Tensor
    producer_fault_bits: torch.Tensor


def require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
    motion_cfg: object,
    racket_cfg: object,
    *,
    table: ActionBallFullMdpDiagnosticCatalogTable | None = None,
) -> ActionBallFullMdpDiagnosticCatalogTable:
    """Require the exact cfg fields consumed by Motion and Racket.

    This is a structural equality check, not an authority.  Callers cannot
    supply a path or digest to the table loader; the optional ``table`` only
    lets the code-owned constructor and live command validate the same frozen
    value without a third filesystem pass.
    """

    if table is None:
        table = load_action_ball_full_mdp_diagnostic_catalog_table()
    if type(table) is not ActionBallFullMdpDiagnosticCatalogTable:
        raise ValueError("full-MDP diagnostic catalog table type differs")
    if (
        getattr(motion_cfg, "action_ball_full_mdp_diagnostic_catalog", None)
        != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND
        or tuple(getattr(motion_cfg, "motion_file", ()) or ())
        != table.motion_files
        or tuple(getattr(motion_cfg, "clip_family_per_clip", ()) or ())
        != table.clip_family_per_clip
        or tuple(getattr(racket_cfg, "clip_names_per_clip", ()) or ())
        != table.action_order
        or tuple(getattr(racket_cfg, "strike_phase_per_clip", ()) or ())
        != table.strike_phase_per_clip
        or tuple(
            getattr(racket_cfg, "mount_normal_sign_per_clip", ()) or ()
        )
        != table.mount_normal_sign_per_clip
        or str(getattr(racket_cfg, "motion_teacher_racket_source", ""))
        != "measured_channel"
    ):
        raise ValueError(
            "fresh full-MDP Motion/Racket cfg differs from the code-owned "
            "active N=1 diagnostic catalog"
        )
    return table


_ACTION_BALL_CONTINUOUS_MOTION_PROFILE_KIND = (
    "whole_body_tracking.action_ball_continuous_motion_projection_v1"
)
_ACTION_BALL_CONTINUOUS_MOTION_CLOCK_KIND = "episode_tick_v1"
_ACTION_BALL_CONTINUOUS_READY_REFERENCE_KIND = (
    "completed_action_frame0_zero_velocity_v1"
)
_ACTION_BALL_R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0 = 1
_ACTION_BALL_R07_REFERENCE_COMPLETED_ACTION_FRAME0 = 2
ACTION_BALL_CONTINUOUS_MOTION_PHASES = (
    "pre_reveal_hidden",
    "active_opportunity",
    "post_deadline_suffix",
    "recovery_hidden",
    "ready_hold",
    "recovery_unavailable",
    "infrastructure_invalid",
)
_ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE = {
    name: index for index, name in enumerate(ACTION_BALL_CONTINUOUS_MOTION_PHASES)
}
ACTION_BALL_CONTINUOUS_CANONICAL_PHASES = (
    "prepare_visible",
    "swing",
    "follow_through",
    "recover_hidden",
    "ready_hold",
)
_ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE = {
    name: index
    for index, name in enumerate(ACTION_BALL_CONTINUOUS_CANONICAL_PHASES)
}
ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE = 0
ACTION_BALL_CONTINUOUS_CANONICAL_SWING = 1
ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH = 2
ACTION_BALL_CONTINUOUS_CANONICAL_RECOVER_HIDDEN = 3
ACTION_BALL_CONTINUOUS_CANONICAL_READY_HOLD = 4
_ACTION_BALL_CONTINUOUS_R05_SOURCE_SHA256 = (
    "82d71c987e51cf4b5940744b124b36f9055d7a655af5e2279e2a6841cb1077dc"
)
_ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256 = (
    "a5762b2e4838a3bdc58c2a30822467d27e4fb1006a37fcc3faf3948f7c2c24fe"
)
_ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256 = (
    "cfc212a4ef2fd2078df99114c28f55df93b0605e0a126049b24b07fc636b16aa"
)
_ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256 = (
    "4e715720b741991905d7c6cf8aa5ddf6c5a1e617773b6132aa33368468736cdd"
)
_ACTION_BALL_CONTINUOUS_TERMINAL_BOUNDARY_AUTHORITY_DOMAIN = (
    "action_ball_full_mdp_reveal_boundary"
)
_ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME = (
    "motion_selected_preflight_rejected"
)
_ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT = 1
# The fresh question chronology is a different owner domain from the cadence
# close deadline.  ``12`` is the frozen ordinary construction reason for a
# candidate whose physical final segment cannot contain one complete Motion
# tick; it is telemetry, not an infrastructure fault.
_ACTION_BALL_MOTION_QUESTION_NO_COMPLETE_HORIZON = 12
_ACTION_BALL_MOTION_QUESTION_FAULT_NONFINITE = 1 << 48
_ACTION_BALL_MOTION_QUESTION_FAULT_UNATTRIBUTABLE = 1 << 49
_ACTION_BALL_MOTION_QUESTION_FAULT_TICK_OVERFLOW = 1 << 50
# This bit is intentionally outside Motion's declared fault schema.  It is
# used only to make an unattributable owner-integrity mismatch poison the
# packed boundary instead of laundering that mismatch into a valid CENSOR.
_ACTION_BALL_CONTINUOUS_MOTION_UNATTRIBUTABLE_BIT = 2
# Exact AST-source-segment surface hash of the construction-bound top reset
# authority.  This is deliberately not a full-file hash: the top and all four
# children can pin the three authority methods without a cyclic source pin.
# A diagnostic fixture may omit the hash only when it explicitly binds with
# ``diagnostic=True``; production must present this exact surface identity.
_ACTION_BALL_CONTINUOUS_MOTION_SELECTED_RESET_AUTHORITY_API_SHA256 = (
    "c54c56ecc5fce051dfadd3e2bb6d90d68acedd3c7095c88508c704ac052da6da"
)
_ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_KIND = (
    "action_ball_continuous_motion_fresh_checkpoint_v1"
)
_ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_SCHEMA_VERSION = 7
_ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS = (
    ("sequence_active", "_action_ball_continuous_sequence_active", False),
    ("control_tick", "_action_ball_continuous_episode_step", False),
    ("scheduled_ordinal", "_action_ball_continuous_scheduled_ordinal", False),
    ("reveal_tick", "_action_ball_continuous_current_reveal_step", False),
    ("deadline_tick", "_action_ball_continuous_current_deadline_step", False),
    ("next_reveal_tick", "_action_ball_continuous_next_reveal_step", False),
    ("last_closed_ordinal", "_action_ball_continuous_last_closed_ordinal", False),
    ("opportunities_consumed", "_action_ball_continuous_opportunities_consumed", True),
    ("policy_opportunities_created", "_action_ball_continuous_policy_opportunities_created", True),
    ("infrastructure_censors_consumed", "_action_ball_continuous_infrastructure_censors_consumed", True),
    ("current_policy_opportunity", "_action_ball_continuous_current_policy_opportunity", False),
    ("motion_active", "_action_ball_continuous_motion_active", False),
    ("suffix_complete", "_action_ball_continuous_suffix_complete", False),
    ("ready_reference_active", "_action_ball_continuous_ready_reference_active", False),
    ("ready_at_reveal", "_action_ball_continuous_ready_at_reveal", False),
    ("reveal_due", "_action_ball_continuous_reveal_due", False),
    ("deadline_due", "_action_ball_continuous_deadline_due", False),
    ("recovery_unavailable", "_action_ball_continuous_recovery_unavailable", False),
    ("task_commit_pending", "_action_ball_continuous_task_commit_pending", False),
    ("task_commit_missed", "_action_ball_continuous_task_commit_missed", False),
    ("task_committed", "_action_ball_continuous_task_committed", False),
    ("motion_release_pending", "_action_ball_continuous_motion_release_pending", False),
    ("motion_release_missed", "_action_ball_continuous_motion_release_missed", False),
    ("legacy_phase", "_action_ball_continuous_phase", True),
    ("canonical_phase", "_action_ball_continuous_canonical_phase", True),
    ("canonical_phase_start_tick", "_action_ball_continuous_canonical_phase_start_tick", False),
    ("task_identity", "_action_ball_continuous_canonical_task_identity", False),
    ("cadence_identity", "_action_ball_continuous_canonical_cadence_identity", False),
    ("action_uid", "_action_ball_continuous_canonical_action_uid", False),
    ("shot_index", "_action_ball_continuous_canonical_shot_index", False),
    ("outcome_identity", "_action_ball_continuous_canonical_outcome_identity", False),
    ("task_receipt_sha256", "_action_ball_continuous_canonical_task_receipt_sha256", True),
    ("cadence_receipt_sha256", "_action_ball_continuous_canonical_cadence_receipt_sha256", True),
    ("candidate_identity", "_action_ball_continuous_canonical_candidate_identity", False),
    ("contact_tick", "_action_ball_continuous_canonical_contact_tick", False),
    ("launch_tick", "_action_ball_continuous_canonical_launch_tick", False),
    ("chosen_horizon_tick", "_action_ball_continuous_canonical_chosen_horizon_tick", False),
    ("task_close_tick", "_action_ball_continuous_canonical_task_close_tick", False),
    ("task_valid", "_action_ball_continuous_canonical_task_valid", False),
    ("timing_active", "_action_ball_task_timing_active", False),
    ("playback_started", "_action_ball_continuous_canonical_playback_started", False),
    ("pending_elapsed_s", "_action_ball_task_pending_elapsed_s", True),
    ("task_age_s", "_action_ball_task_age_s", True),
    ("time_to_contact_s", "_action_ball_time_to_contact_s", True),
    ("teacher_rate", "_action_ball_teacher_rate", True),
    ("scaled_t_hit_s", "_action_ball_scaled_t_hit_s", True),
    ("scaled_t_cycle_s", "_action_ball_scaled_t_cycle_s", True),
    ("pre_swing_wait_s", "_action_ball_pre_swing_wait_s", True),
    ("reset_generation", "_action_ball_reset_generation", True),
    ("swing_generation", "_action_ball_swing_generation", True),
    ("action_slot", "clip_id", True),
    ("teacher_time_step", "time_steps", True),
    ("teacher_time_step_f", "time_steps_f", True),
    ("teacher_speed_scale", "speed_scale", True),
    ("teacher_hold_counter", "hold_counter", True),
    (
        "reset_ready_body_pos_w",
        "_action_ball_safe_ready_body_pos_w",
        False,
    ),
    (
        "reset_ready_body_quat_w",
        "_action_ball_safe_ready_body_quat_w",
        False,
    ),
    (
        "reset_ready_reference_pending",
        "_action_ball_safe_ready_reference_pending",
        False,
    ),
    ("body_pos_relative_w", "body_pos_relative_w", False),
    ("body_quat_relative_w", "body_quat_relative_w", False),
    ("reset_pending", "_action_ball_continuous_motion_reset_pending", False),
)


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionProjection:
    """Isolated current-tick snapshot published by the Motion command.

    Every tensor is a detached clone.  Consumers may therefore retain or even
    accidentally mutate their snapshot without aliasing Motion's live owner
    buffers.  This type deliberately has no task, target, inbound-ball or
    future-question field.
    """

    common_step: int
    episode_tick: torch.Tensor
    reveal_due: torch.Tensor
    closed_mask: torch.Tensor
    close_reason: torch.Tensor
    deadline_due: torch.Tensor
    scheduled_ordinal: torch.Tensor
    reveal_tick: torch.Tensor
    deadline_tick: torch.Tensor
    next_reveal_tick: torch.Tensor
    ready_at_reveal: torch.Tensor
    motion_active: torch.Tensor
    ready_reference_active: torch.Tensor
    suffix_complete: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor


class ActionBallContinuousMotionObservationToken:
    """Opaque registry key for one already-published Motion observation."""

    __slots__ = ()


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionObservationView:
    """One publication-frozen Motion chronology with consumer isolation.

    The five-state lifecycle is independently written by Motion.  It is never
    derived from the legacy seven-state diagnostic phase, and none of the
    tensors aliases live owner storage.  The owner retains one private
    publication instance and returns a fresh clone-only instance to each
    consumer so an in-place tensor write cannot cross the trust boundary.
    """

    motion_owner: object
    publication_identity: object
    common_step: int
    control_tick: torch.Tensor
    phase: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_uid: torch.Tensor
    task_identity: torch.Tensor
    task_valid: torch.Tensor
    time_to_contact_remaining_s: torch.Tensor
    time_to_teacher_start_remaining_s: torch.Tensor
    time_to_next_reveal_s: torch.Tensor


class ActionBallMotionQuestionChronologyReceipt:
    """Opaque Motion-owned exact contact/launch chronology capability."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Motion question chronology receipts are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Motion question chronology receipts are immutable")

    def __copy__(self):
        raise TypeError("Motion question chronology receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Motion question chronology receipts cannot be copied")

    def __reduce__(self):
        raise TypeError("Motion question chronology receipts cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Motion question chronology receipts cannot be serialized")


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallMotionQuestionChronologyView:
    """Clone-only exact question chronology produced by Motion.

    ``contact_tick`` is derived from the immutable task receipt and the real
    control tick.  ``deadline_tick`` is deliberately absent: cadence close is
    another fact with another purpose and must never be substituted for exact
    contact.  Candidate-keyed physical horizons come from the independent
    Physical numerical owner and remain bound to its opaque receipt.
    """

    motion_owner: object
    chronology_identity: object
    selected_env_index: torch.Tensor
    current_tick: torch.Tensor
    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    earliest_launch_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_s: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    task_identity: torch.Tensor
    cadence_identity: torch.Tensor
    task_receipt_sha256: torch.Tensor
    motion_task_f32: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault: torch.Tensor


@dataclass(frozen=True, eq=False, repr=False)
class _ActionBallMotionQuestionChronologyRecord:
    chronology_identity: object
    physical_horizon_owner: object
    physical_horizon_receipt: object
    selected_env_index: torch.Tensor
    current_tick: torch.Tensor
    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    earliest_launch_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_s: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    task_identity: torch.Tensor
    cadence_identity: torch.Tensor
    task_receipt_sha256: torch.Tensor
    motion_task_f32: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault: torch.Tensor


class ActionBallMotionQuestionProductionHold(RuntimeError):
    """The owner boundary is real, but the complete production ABI is not."""


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousTaskCommitToken:
    """Single-use Motion token returned only after complete staged validation."""

    _owner_nonce: object
    serial: int
    common_step: int
    env_ids: tuple[int, ...]
    scheduled_ordinals: tuple[int, ...]
    episode_ticks: tuple[int, ...]
    reveal_ticks: tuple[int, ...]
    deadline_ticks: tuple[int, ...]
    next_reveal_ticks: tuple[int, ...]
    reset_generations: tuple[int, ...]
    swing_generations: tuple[int, ...]
    task_refs: tuple[object, ...]
    _timing_rows: tuple[tuple[float, ...], ...]
    _active_task_refs_before: tuple[object, ...]
    _committed_task_refs_before: tuple[object, ...]


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionStage:
    """Private Motion child staged against one exact unarmed R05 preview.

    All fields are host authorities or detached construction material.  The
    token is not a policy opportunity, does not expose a live tensor alias,
    and cannot publish Motion state.  ``finalize`` is the only phase allowed
    to allocate device after-images.
    """

    _owner_nonce: object
    serial: int
    owner_mutation_version: int
    common_step: int
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    all_owner_install_root_sha256: str
    prepared_batch_sha256: str
    env_ids: tuple[int, ...]
    scheduled_ordinals: tuple[int, ...]
    episode_ticks: tuple[int, ...]
    reveal_ticks: tuple[int, ...]
    deadline_ticks: tuple[int, ...]
    next_reveal_ticks: tuple[int, ...]
    reset_generations: tuple[int, ...]
    swing_generations: tuple[int, ...]
    action_slots: tuple[int, ...]
    ready_at_reveal: tuple[bool, ...]
    runtime_task_refs: tuple[object, ...]
    runtime_task_receipts: tuple[object, ...]
    runtime_task_receipt_sha256s: tuple[str, ...]
    timing_after_image_sha256: str
    motion_child_token_root_sha256: str
    _timing_rows: tuple[tuple[float, ...], ...]
    _timing_f32_le: bytes
    _prearm_payload_json: bytes
    _reveal_final_public_token: object
    _reveal_final_private_token: object


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionPrearmedInstall:
    """Opaque retained token for one fully materialized Motion leaf."""

    _owner_nonce: object
    serial: int
    owner_mutation_version: int
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    selected_env_ids: tuple[int, ...]
    canonical_sha256: str


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionArmedInstall:
    """Opaque single-use identity emitted after the exact global row check."""

    _owner_nonce: object
    serial: int


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionCensoredInstall:
    """Opaque terminal identity for one owner-issued global CENSOR."""

    _owner_nonce: object
    serial: int


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionChildTerminalToken:
    """Opaque proof that Motion copied its prevalidated child after-image."""

    _owner_nonce: object
    serial: int
    decision: str


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionSelectedResetStage:
    """Opaque, owner-minted reset stage with no live Motion writes."""

    _owner_nonce: object
    serial: int
    owner_mutation_version: int
    stage_sha256: str


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionSelectedResetPrevalidated:
    """Opaque reset handle after every fallible after-image check."""

    _owner_nonce: object
    serial: int
    stage_sha256: str


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionSelectedResetChildTerminalToken:
    """Opaque proof that Motion copied its prevalidated reset after-image."""

    _owner_nonce: object
    serial: int
    stage_sha256: str


@dataclass(frozen=True, eq=False, repr=False)
class ActionBallContinuousMotionSelectedResetCompletionToken:
    """Opaque single-use Motion ACK minted only after exact Device-R05-last."""

    _owner_nonce: object
    serial: int
    stage_sha256: str


@dataclass
class _ActionBallContinuousMotionGlobalDrainLease:
    """Private pre-transfer lease for the sole global PPO boundary."""

    pack: object
    authority: object
    update_index: int
    completed_environment_steps: int
    owner_mutation_version: int
    terminal_resolution_total: int
    expected_values: tuple[int, int, int, int]
    source_tensor_receipts: tuple[tuple[torch.Tensor, int], ...]
    stage: str = "prepared"


@dataclass(frozen=True)
class ActionBallContinuousMotionCommitReceipt:
    """Typed ACCEPT/CENSOR chronology emitted by the Motion child."""

    schema_version: int
    kind: str
    decision: str
    reveal_final_preview_sha256: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_sha256: str
    motion_child_token_root_sha256: str
    prepared_r05_terminal_claim_sha256: str
    expected_r05_terminal_kind: str
    expected_r05_terminal_sha256: str
    timing_after_image_sha256: str
    selected_env_ids: tuple[int, ...]
    owner_mutation_version_before: int
    owner_mutation_version_after: int
    installed_count: int
    censored_count: int
    policy_opportunity_created: bool
    runtime_integrated: bool
    launch_authorized: bool

    @property
    def canonical_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "decision": self.decision,
            "reveal_final_preview_sha256": (
                self.reveal_final_preview_sha256
            ),
            "global_boundary_receipt_sha256": (
                self.global_boundary_receipt_sha256
            ),
            "global_boundary_packet_sha256": (
                self.global_boundary_packet_sha256
            ),
            "motion_child_token_root_sha256": (
                self.motion_child_token_root_sha256
            ),
            "prepared_r05_terminal_claim_sha256": (
                self.prepared_r05_terminal_claim_sha256
            ),
            "expected_r05_terminal_kind": self.expected_r05_terminal_kind,
            "expected_r05_terminal_sha256": (
                self.expected_r05_terminal_sha256
            ),
            "timing_after_image_sha256": self.timing_after_image_sha256,
            "selected_env_ids": list(self.selected_env_ids),
            "owner_mutation_version_before": (
                self.owner_mutation_version_before
            ),
            "owner_mutation_version_after": (
                self.owner_mutation_version_after
            ),
            "installed_count": self.installed_count,
            "censored_count": self.censored_count,
            "policy_opportunity_created": self.policy_opportunity_created,
            "runtime_integrated": self.runtime_integrated,
            "launch_authorized": self.launch_authorized,
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class ActionBallContinuousMotionCensorReceipt:
    """Typed zero-policy-opportunity Motion chronology for global CENSOR."""

    schema_version: int
    kind: str
    decision: str
    reveal_final_preview_sha256: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_sha256: str
    motion_child_token_root_sha256: str
    prepared_r05_terminal_claim_sha256: str
    expected_r05_terminal_kind: str
    expected_r05_terminal_sha256: str
    selected_env_ids: tuple[int, ...]
    owner_mutation_version_before: int
    owner_mutation_version_after: int
    censored_count: int
    policy_opportunity_created: bool
    runtime_integrated: bool
    launch_authorized: bool

    @property
    def canonical_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "decision": self.decision,
            "reveal_final_preview_sha256": (
                self.reveal_final_preview_sha256
            ),
            "global_boundary_receipt_sha256": (
                self.global_boundary_receipt_sha256
            ),
            "global_boundary_packet_sha256": (
                self.global_boundary_packet_sha256
            ),
            "motion_child_token_root_sha256": (
                self.motion_child_token_root_sha256
            ),
            "prepared_r05_terminal_claim_sha256": (
                self.prepared_r05_terminal_claim_sha256
            ),
            "expected_r05_terminal_kind": self.expected_r05_terminal_kind,
            "expected_r05_terminal_sha256": (
                self.expected_r05_terminal_sha256
            ),
            "selected_env_ids": list(self.selected_env_ids),
            "owner_mutation_version_before": (
                self.owner_mutation_version_before
            ),
            "owner_mutation_version_after": (
                self.owner_mutation_version_after
            ),
            "censored_count": self.censored_count,
            "policy_opportunity_created": self.policy_opportunity_created,
            "runtime_integrated": self.runtime_integrated,
            "launch_authorized": self.launch_authorized,
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _parse_action_ball_continuous_motion_profile(value):
    """Validate one Motion-side projection of the external C01/C02 contract.

    This is deliberately not a fourth schedule authority.  Its self-hash only
    detects mutable-config drift; a fresh runtime cannot reset or advance until
    the independently retained C01 continuous-contract and C02 recovery-
    contract identities bind this exact timing projection.  It contains no
    target, ball, solver or Reward fields and cannot authorize those owners.
    ``None`` remains the literal legacy path.
    """

    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError(
            "action_ball_continuous_motion_cadence must be one exact mapping"
        )
    expected = {
        "schema_version",
        "kind",
        "clock_kind",
        "continuous_contract_authority_sha256",
        "recovery_contract_authority_sha256",
        "ready_reference_kind",
        "canonical_sha256",
    }
    if set(value) != expected:
        raise ValueError(
            "action_ball_continuous_motion_cadence keys differ: "
            f"missing={sorted(expected - set(value))!r}, "
            f"unknown={sorted(set(value) - expected)!r}"
        )
    payload = {key: value[key] for key in expected - {"canonical_sha256"}}
    if value["schema_version"] != 1:
        raise ValueError(
            "action_ball_continuous_motion_cadence schema_version differs"
        )
    if value["kind"] != _ACTION_BALL_CONTINUOUS_MOTION_PROFILE_KIND:
        raise ValueError("action_ball continuous Motion profile kind differs")
    if value["clock_kind"] != _ACTION_BALL_CONTINUOUS_MOTION_CLOCK_KIND:
        raise ValueError("action_ball continuous Motion clock kind differs")
    if (
        value["ready_reference_kind"]
        != _ACTION_BALL_CONTINUOUS_READY_REFERENCE_KIND
    ):
        raise ValueError(
            "action_ball continuous Motion ready-reference kind differs"
        )
    for name in (
        "continuous_contract_authority_sha256",
        "recovery_contract_authority_sha256",
        "canonical_sha256",
    ):
        digest = value[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"{name} must be one lowercase SHA-256")
    actual_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    if value["canonical_sha256"] != actual_sha256:
        raise ValueError(
            "action_ball continuous Motion profile canonical SHA-256 differs"
        )
    # Retain a private copy so a mutable Hydra/config mapping cannot rewrite
    # the parent-authority pins after construction.  Timing itself arrives only
    # through the external binding below and is absent from this profile.
    return MappingProxyType(dict(value))


# ``CommandTerm.compute`` performs a device-side ``time_left <= 0`` followed by
# ``nonzero`` on every policy tick.  These command terms use the exact
# 1e9-second pair as the repository's declared "time resampling disabled"
# sentinel: resets and explicit manager resamples own command replacement.
# Keep the optimization local to that exact, construction-bound contract.  If
# a mutable config drifts after construction, callers fall back to IsaacLab's
# complete implementation on the very next tick.
_TIME_RESAMPLING_DISABLED_RANGE_S = (1.0e9, 1.0e9)


def _has_exact_disabled_time_resampling_contract(cfg) -> bool:
    raw = getattr(cfg, "resampling_time_range", None)
    if isinstance(raw, (str, bytes)):
        return False
    try:
        values = tuple(raw)
    except TypeError:
        return False
    if len(values) != 2:
        return False
    if any(type(value) not in (int, float) for value in values):
        return False
    return tuple(float(value) for value in values) == (
        _TIME_RESAMPLING_DISABLED_RANGE_S
    )


def _bind_disabled_time_resampling_fast_path(command) -> None:
    """Bind the construction-time half of the immutable fast-path contract."""

    command._disabled_time_resampling_fast_path_bound = (
        _has_exact_disabled_time_resampling_contract(command.cfg)
    )
    # This is a one-way eligibility latch.  A shorter runtime range can create
    # a shorter ``time_left`` through the base resampler; changing the config
    # back to the sentinel must never make that state look sentinel-origin
    # again.  A newly constructed non-sentinel command is likewise ineligible
    # for its entire lifetime.
    command._disabled_time_resampling_fast_path_poisoned = (
        not command._disabled_time_resampling_fast_path_bound
    )
    # ``CommandTerm.__init__`` creates an all-zero timer.  The first real
    # manager reset is what writes the sentinel-origin timer; before that, the
    # base compute must retain authority so an accidental early tick performs
    # the same immediate resample as IsaacLab.
    command._disabled_time_resampling_fast_path_armed = False
    command._disabled_time_resampling_timer_receipt = None


def _poison_disabled_time_resampling_fast_path_on_drift(command) -> None:
    """Permanently revoke the fast path after any observed contract drift."""

    if (
        getattr(command, "_disabled_time_resampling_fast_path_bound", False)
        is True
        and not _has_exact_disabled_time_resampling_contract(command.cfg)
    ):
        command._disabled_time_resampling_fast_path_poisoned = True
        command._disabled_time_resampling_fast_path_armed = False
        command._disabled_time_resampling_timer_receipt = None


def _tensor_identity_version_receipt(tensor):
    """Return host-only tensor identity/version proof, or ``None``."""

    if not torch.is_tensor(tensor):
        return None
    try:
        version = tensor._version
    except RuntimeError:
        return None
    if type(version) is not int:
        return None
    # Retain the object itself rather than only ``id(tensor)``.  That closes
    # Python object-id reuse after an off-lane replacement without reading any
    # tensor value or synchronizing the device.
    return (tensor, version)


def _tensor_matches_identity_version_receipt(tensor, receipt) -> bool:
    """Check a host-only receipt without invoking tensor value equality."""

    if not isinstance(receipt, tuple) or len(receipt) != 2:
        return False
    expected_tensor, expected_version = receipt
    if expected_tensor is not tensor:
        return False
    current = _tensor_identity_version_receipt(tensor)
    return current is not None and current[1] == expected_version


def _revoke_disabled_time_resampling_fast_path_on_timer_drift(
    command,
) -> None:
    """Disarm when ``time_left`` was replaced or modified off-lane."""

    if getattr(
        command,
        "_disabled_time_resampling_fast_path_armed",
        False,
    ) is not True:
        return
    if not _tensor_matches_identity_version_receipt(
        getattr(command, "time_left", None),
        getattr(
            command, "_disabled_time_resampling_timer_receipt", None
        ),
    ):
        command._disabled_time_resampling_fast_path_armed = False
        command._disabled_time_resampling_timer_receipt = None


def _resample_scope_covers_all_envs(command, env_ids) -> bool:
    """Recognize the exact ordered all-environment reset scope.

    Full CUDA tensors are copied only at this resample/reset boundary.  The
    per-tick hot path performs no device reduction or readback.
    """

    if isinstance(env_ids, slice):
        start, stop, step = env_ids.indices(command.num_envs)
        return start == 0 and stop == command.num_envs and step == 1
    if torch.is_tensor(env_ids):
        if (
            env_ids.ndim != 1
            or env_ids.numel() != command.num_envs
            # The manager's authoritative reset IDs are int64.  Reject every
            # other dtype fail-closed, including legacy uint8 masks.
            or env_ids.dtype != torch.long
        ):
            return False
        rows = env_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    else:
        try:
            rows = list(env_ids)
        except TypeError:
            return False
        if (
            len(rows) != command.num_envs
            or any(type(value) is not int for value in rows)
        ):
            return False
    if rows != list(range(command.num_envs)):
        return False
    return True


def _arm_disabled_time_resampling_fast_path_after_resample(
    command, env_ids
) -> None:
    """Arm only after a successful full reset wrote sentinel-origin timers."""

    if (
        getattr(command, "_disabled_time_resampling_fast_path_bound", False)
        is True
        and getattr(
            command,
            "_disabled_time_resampling_fast_path_poisoned",
            True,
        )
        is False
        and _has_exact_disabled_time_resampling_contract(command.cfg)
    ):
        already_armed = getattr(
            command,
            "_disabled_time_resampling_fast_path_armed",
            False,
        ) is True
        if already_armed or _resample_scope_covers_all_envs(
            command, env_ids
        ):
            receipt = _tensor_identity_version_receipt(
                getattr(command, "time_left", None)
            )
            if receipt is not None:
                command._disabled_time_resampling_fast_path_armed = True
                command._disabled_time_resampling_timer_receipt = receipt


def _disabled_time_resampling_fast_path_active(command) -> bool:
    """Return true only while both bound and live config contracts agree."""

    _revoke_disabled_time_resampling_fast_path_on_timer_drift(command)
    if (
        getattr(command, "_disabled_time_resampling_fast_path_bound", False)
        is not True
        or getattr(
            command,
            "_disabled_time_resampling_fast_path_poisoned",
            True,
        )
        is not False
        or getattr(
            command,
            "_disabled_time_resampling_fast_path_armed",
            False,
        )
        is not True
    ):
        return False
    # Read the mutable config once.  A false result is committed to the
    # one-way latch in the same branch, so a later sentinel value cannot revive
    # eligibility after an observed drift.
    if not _has_exact_disabled_time_resampling_contract(command.cfg):
        command._disabled_time_resampling_fast_path_poisoned = True
        command._disabled_time_resampling_fast_path_armed = False
        command._disabled_time_resampling_timer_receipt = None
        return False
    return True


def _compute_without_disabled_time_resampling_scan(command, dt: float) -> bool:
    """Run the exact base ordering without the impossible expiry scan.

    The administrative ``time_left`` clock still receives the same subtraction
    as IsaacLab.  Only ``(time_left <= 0).nonzero()`` and its resample branch are
    omitted under the exact disabled sentinel, so metrics, command updates and
    any code which snapshots ``time_left`` retain their previous values.
    """

    if not _disabled_time_resampling_fast_path_active(command):
        return False
    command._update_metrics()
    command.time_left -= dt
    receipt = _tensor_identity_version_receipt(command.time_left)
    command._disabled_time_resampling_timer_receipt = receipt
    if receipt is None:
        command._disabled_time_resampling_fast_path_armed = False
    command._update_command()
    return True


def _split_ready_nonloop_wrap_scan_is_impossible(motion) -> bool:
    """Prove that ``Motion.just_resampled`` has no writer on this tick path.

    This is deliberately stronger than merely checking the diagnostic flag.
    The non-looping N=1 ActionBall binding, disabled event scheduler, singleton
    clip, completion latch and exact boolean mask must all still be present.
    Any runtime/config mutation drops back to the historical scan.
    """

    if (
        getattr(motion, "action_ball_diagnostic_split_ready_teacher", None)
        is not True
        or getattr(
            motion,
            "action_ball_single_stroke_timeout_enabled",
            None,
        )
        is not True
        or getattr(motion, "_action_ball_birth_broker", None) is None
        or getattr(motion, "_multiseg", None) is not False
        or getattr(motion, "_event_scheduler", None) is not None
        or type(getattr(getattr(motion, "motion", None), "num_segments", None))
        is not int
        or motion.motion.num_segments != 1
    ):
        return False
    wrapped = getattr(motion, "just_resampled", None)
    completion = getattr(
        motion, "_action_ball_single_stroke_complete", None
    )
    token = getattr(
        getattr(motion, "_env", None),
        "common_step_counter",
        None,
    )
    receipt = getattr(motion, "_split_ready_empty_wrap_receipt", None)
    return (
        torch.is_tensor(wrapped)
        and wrapped.dtype == torch.bool
        and tuple(wrapped.shape) == (motion.num_envs,)
        and torch.is_tensor(completion)
        and completion.dtype == torch.bool
        and tuple(completion.shape) == (motion.num_envs,)
        and type(token) is int
        and isinstance(receipt, tuple)
        and len(receipt) == 2
        and receipt[0] == token
        and _tensor_matches_identity_version_receipt(
            wrapped, receipt[1]
        )
    )


class MotionLoader:
    """Loads one or more motion clips into a single concatenated time axis.

    Passing several files (HITTER unified policy: forehand + backhand) concatenates them along the time
    dimension and records per-clip ``seg_start`` / ``seg_len`` so the command can step/wrap/strike within
    one clip ("segment") at a time, selected per-env by swing type. A single file behaves exactly as
    before: one segment spanning the whole motion, ``time_step_total`` unchanged.
    """

    _KINEMATICS_SCHEMA = 2
    _KINEMATICS_CORE_KEYS = (
        "kinematics_schema_version", "body_pos_point", "body_lin_vel_point"
    )
    _KINEMATICS_BODY_NAMES_KEY = "body_names"
    _MEASURED_RACKET_SCHEMA = 4
    _MEASURED_RACKET_ARRAY_KEYS = (
        "measured_racket_site_pos_w",
        "measured_racket_normal_w",
        "measured_racket_long_axis_w",
    )
    _MEASURED_RACKET_META_KEYS = (
        "measured_racket_schema_version",
        "measured_racket_position_semantics",
        "measured_racket_normal_semantics",
        "measured_racket_long_axis_semantics",
        "measured_racket_robot_mount_normal_sign",
        "measured_racket_robot_butt_to_blade_axis_local",
        "measured_racket_robot_rigid_visual_mesh_sha256",
        "measured_racket_source_sha256",
        "measured_racket_retarget_admitted",
        "measured_racket_retarget_receipt_sha256",
        "measured_racket_joint_order_contract_id",
        "measured_racket_joint_order_contract_sha256",
    )

    @staticmethod
    def _meta_scalar(data, key: str) -> str:
        raw = np.asarray(data[key]).reshape(-1)
        if raw.size != 1:
            raise ValueError(f"motion metadata {key} must be scalar, got {np.asarray(data[key]).shape}")
        value = raw[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    @staticmethod
    def _meta_body_names(data, key: str) -> tuple[str, ...]:
        raw = np.asarray(data[key])
        if raw.ndim != 1:
            raise ValueError(f"motion metadata {key} must be one-dimensional, got {raw.shape}")
        names = []
        for value in raw.tolist():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            names.append(str(value))
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"motion metadata {key} must contain unique non-empty names")
        return tuple(names)

    @staticmethod
    def _fps_scalar(data, path: str) -> float:
        raw = np.asarray(data["fps"])
        if raw.size != 1:
            raise ValueError(f"{path}: fps must be scalar, got shape {raw.shape}")
        fps = float(raw.reshape(-1)[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path}: fps must be finite and positive, got {fps!r}")
        return fps

    @staticmethod
    def _validate_motion_array_shapes(
        data, path: str, articulation_body_count: int
    ) -> int:
        """Validate the shared time axis and full-articulation body-column shape."""

        expected_tail = {
            "body_pos_w": (articulation_body_count, 3),
            "body_quat_w": (articulation_body_count, 4),
            "body_lin_vel_w": (articulation_body_count, 3),
            "body_ang_vel_w": (articulation_body_count, 3),
        }
        arrays = {key: np.asarray(data[key]) for key in ("joint_pos", "joint_vel", *expected_tail)}
        if arrays["joint_pos"].ndim != 2 or arrays["joint_vel"].shape != arrays["joint_pos"].shape:
            raise ValueError(
                f"{path}: joint_pos/joint_vel must have the same (T,J) shape, got "
                f"{arrays['joint_pos'].shape}/{arrays['joint_vel'].shape}"
            )
        frame_count = int(arrays["joint_pos"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"{path}: motion clip contains no frames")
        for key, tail in expected_tail.items():
            expected = (frame_count, *tail)
            if arrays[key].shape != expected:
                raise ValueError(f"{path}: {key} has shape {arrays[key].shape}, expected {expected}")
        return frame_count

    @classmethod
    def _kinematics_contract(
        cls,
        data,
        path: str,
        articulation_body_names: tuple[str, ...],
        *,
        allow_legacy_link_origin_velocity: bool = False,
    ) -> dict:
        """Validate body point semantics without guessing from a filename.

        Untagged historical Isaac clips remain loadable but exact-ineligible.
        Untagged legacy MuJoCo/retime clips have a decisive content signature:
        body_lin_vel_w == d(body_pos_w)/dt under meaningful angular motion.
        Those are fail-closed because MotionCommand rewards COM velocity.
        """

        files = set(data.files)
        present = [key in files for key in cls._KINEMATICS_CORE_KEYS]
        if any(present) and not all(present):
            raise ValueError(f"{path}: partial/malformed motion kinematics metadata")
        if not any(present) and cls._KINEMATICS_BODY_NAMES_KEY in files:
            raise ValueError(f"{path}: body_names exists without a kinematics schema")
        if all(present):
            schema_raw = np.asarray(data[cls._KINEMATICS_CORE_KEYS[0]]).reshape(-1)
            if schema_raw.size != 1:
                raise ValueError(
                    f"{path}: kinematics_schema_version must be scalar, got "
                    f"{np.asarray(data[cls._KINEMATICS_CORE_KEYS[0]]).shape}"
                )
            schema = int(schema_raw[0])
            pos_point = cls._meta_scalar(data, cls._KINEMATICS_CORE_KEYS[1])
            vel_point = cls._meta_scalar(data, cls._KINEMATICS_CORE_KEYS[2])
            if schema not in (1, cls._KINEMATICS_SCHEMA) or pos_point != "link_origin":
                raise ValueError(
                    f"{path}: unsupported motion kinematics contract "
                    f"schema={schema} pos={pos_point!r} vel={vel_point!r}"
                )
            if vel_point != "center_of_mass":
                raise ValueError(
                    f"{path}: body_lin_vel_point={vel_point!r}, but Isaac MotionCommand compares "
                    "against COM velocity. Run scripts/migrate_motion_kinematics.py with an explicit "
                    "--source-point; link-origin velocity must not enter formal training."
                )
            body_names = None
            if cls._KINEMATICS_BODY_NAMES_KEY in files:
                body_names = cls._meta_body_names(data, cls._KINEMATICS_BODY_NAMES_KEY)
                if body_names != articulation_body_names:
                    raise ValueError(
                        f"{path}: body_names/order does not match the runtime articulation: "
                        f"file={list(body_names)} runtime={list(articulation_body_names)}"
                    )
            if schema == cls._KINEMATICS_SCHEMA and body_names is None:
                raise ValueError(f"{path}: schema-{schema} motion is missing body_names")
            exact = schema == cls._KINEMATICS_SCHEMA and body_names is not None
            return {
                "schema_version": schema,
                "body_pos_point": pos_point,
                "body_lin_vel_point": vel_point,
                "body_names": None if body_names is None else list(body_names),
                "exact": exact,
                "status": "declared_v2" if exact else "legacy_v1_unbound_body_order",
            }

        pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        lin = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        ang = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if pos.shape != lin.shape or ang.shape != lin.shape or len(pos) < 2 or fps <= 0.0:
            raise ValueError(f"{path}: invalid legacy motion arrays for point-semantics audit")
        link_fd = np.gradient(pos, 1.0 / fps, axis=0)
        fd_max = float(np.max(np.abs(lin - link_fd)))
        max_ang = float(np.max(np.linalg.norm(ang, axis=-1)))
        if max_ang > 0.2 and fd_max <= 1.0e-4:
            if not allow_legacy_link_origin_velocity:
                raise ValueError(
                    f"{path}: untagged body_lin_vel_w is numerically d(link-origin position)/dt "
                    f"(max residual {fd_max:.3e} m/s, max |omega| {max_ang:.2f} rad/s), but "
                    "MotionCommand rewards COM velocity. This is the pre-2026-07-10 V5/MuJoCo "
                    "converter signature. Migrate it explicitly with "
                    "scripts/migrate_motion_kinematics.py --source-point link_origin; refusing "
                    "to train on the wrong point."
                )
            return {
                "schema_version": None,
                "body_pos_point": "link_origin",
                "body_lin_vel_point": "link_origin",
                "body_names": None,
                "exact": False,
                "status": "legacy_link_origin_velocity_diagnostic_only",
                "link_fd_max_abs_mps": fd_max,
                "max_ang_radps": max_ang,
            }
        return {
            "schema_version": None, "body_pos_point": None, "body_lin_vel_point": None,
            "body_names": None,
            "exact": False, "status": "legacy_unbound_assumed_com",
            "link_fd_max_abs_mps": fd_max, "max_ang_radps": max_ang,
        }

    @classmethod
    def _measured_racket_contract(
        cls, data, path: str, frame_count: int
    ) -> dict | None:
        """Validate an optional same-clock measured-paddle teacher channel.

        The channel is deliberately separate from robot body FK.  Retargeting may use the measured
        blade trajectory as an optimization target, but reconstructing it later from the optimized
        robot wrist is not an independent teacher and silently discards the original residual.
        Partial or ambiguous channels therefore fail closed; legacy clips with none remain loadable.
        """

        keys = (*cls._MEASURED_RACKET_ARRAY_KEYS, *cls._MEASURED_RACKET_META_KEYS)
        present = [key in set(data.files) for key in keys]
        if not any(present):
            return None
        if not all(present):
            missing = [key for key, exists in zip(keys, present) if not exists]
            raise ValueError(f"{path}: partial measured-racket contract; missing {missing}")
        raw_schema = np.asarray(data["measured_racket_schema_version"]).reshape(-1)
        if raw_schema.size != 1 or int(raw_schema[0]) != cls._MEASURED_RACKET_SCHEMA:
            raise ValueError(
                f"{path}: measured_racket_schema_version must be "
                f"{cls._MEASURED_RACKET_SCHEMA}"
            )
        position_semantics = cls._meta_scalar(data, "measured_racket_position_semantics")
        normal_semantics = cls._meta_scalar(data, "measured_racket_normal_semantics")
        long_axis_semantics = cls._meta_scalar(
            data, "measured_racket_long_axis_semantics"
        )
        if position_semantics != "physical_blade_center":
            raise ValueError(
                f"{path}: measured racket position must mean physical_blade_center, got "
                f"{position_semantics!r}"
            )
        if normal_semantics != "signed_physical_hitting_face":
            raise ValueError(
                f"{path}: measured racket normal must mean signed_physical_hitting_face, got "
                f"{normal_semantics!r}"
            )
        if long_axis_semantics != "measured_paddle_butt_to_blade":
            raise ValueError(
                f"{path}: measured racket long axis must mean "
                "measured_paddle_butt_to_blade, got "
                f"{long_axis_semantics!r}"
            )
        mount_sign_raw = np.asarray(
            data["measured_racket_robot_mount_normal_sign"]
        ).reshape(-1)
        if mount_sign_raw.size != 1 or float(mount_sign_raw[0]) not in (-1.0, 1.0):
            raise ValueError(
                f"{path}: measured_racket_robot_mount_normal_sign must be scalar +1/-1"
            )
        robot_mount_normal_sign = int(float(mount_sign_raw[0]))
        from . import racket_contact_geometry as contact_geometry

        robot_axis_local = np.asarray(
            data["measured_racket_robot_butt_to_blade_axis_local"],
            dtype=np.float64,
        ).reshape(-1)
        expected_axis_local = np.asarray(
            contact_geometry.RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
            dtype=np.float64,
        )
        if robot_axis_local.shape != (3,) or not np.array_equal(
            robot_axis_local, expected_axis_local
        ):
            raise ValueError(
                f"{path}: measured racket robot butt-to-blade axis changed"
            )
        robot_mesh_sha256 = cls._meta_scalar(
            data, "measured_racket_robot_rigid_visual_mesh_sha256"
        )
        if robot_mesh_sha256 != contact_geometry.RACKET_RIGID_VISUAL_MESH_SHA256:
            raise ValueError(
                f"{path}: measured racket rigid-racket visual mesh SHA changed"
            )
        source_sha256 = cls._meta_scalar(data, "measured_racket_source_sha256")
        if len(source_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha256):
            raise ValueError(f"{path}: measured_racket_source_sha256 is not lowercase SHA-256")
        receipt_sha256 = cls._meta_scalar(data, "measured_racket_retarget_receipt_sha256")
        if len(receipt_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in receipt_sha256
        ):
            raise ValueError(
                f"{path}: measured_racket_retarget_receipt_sha256 is not lowercase SHA-256"
            )
        joint_order_contract_id = cls._meta_scalar(
            data, "measured_racket_joint_order_contract_id"
        )
        if joint_order_contract_id != "a3-gmr-dof-pos-to-runtime-articulation-v1":
            raise ValueError(f"{path}: measured racket joint-order contract id changed")
        joint_order_contract_sha256 = cls._meta_scalar(
            data, "measured_racket_joint_order_contract_sha256"
        )
        if len(joint_order_contract_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in joint_order_contract_sha256
        ):
            raise ValueError(
                f"{path}: measured racket joint-order contract SHA is not lowercase SHA-256"
            )
        admitted = np.asarray(data["measured_racket_retarget_admitted"]).reshape(-1)
        if admitted.size != 1 or int(admitted[0]) != 1:
            raise ValueError(
                f"{path}: measured racket teacher requires an admitted canonical-site retarget"
            )
        position = np.asarray(data["measured_racket_site_pos_w"], dtype=np.float64)
        normal = np.asarray(data["measured_racket_normal_w"], dtype=np.float64)
        long_axis = np.asarray(
            data["measured_racket_long_axis_w"], dtype=np.float64
        )
        expected = (frame_count, 3)
        if (
            position.shape != expected
            or normal.shape != expected
            or long_axis.shape != expected
        ):
            raise ValueError(
                f"{path}: measured racket position/normal/long-axis must all be "
                f"{expected}, got {position.shape}/{normal.shape}/{long_axis.shape}"
            )
        if (
            not np.isfinite(position).all()
            or not np.isfinite(normal).all()
            or not np.isfinite(long_axis).all()
        ):
            raise ValueError(f"{path}: measured racket channel contains non-finite values")
        normal_norm = np.linalg.norm(normal, axis=-1)
        long_axis_norm = np.linalg.norm(long_axis, axis=-1)
        if float(np.max(np.abs(normal_norm - 1.0))) > 1.0e-3:
            raise ValueError(
                f"{path}: measured racket normals are not unit length "
                f"(max error {float(np.max(np.abs(normal_norm - 1.0))):.3e})"
            )
        if float(np.max(np.abs(long_axis_norm - 1.0))) > 1.0e-3:
            raise ValueError(
                f"{path}: measured racket long axes are not unit length "
                f"(max error {float(np.max(np.abs(long_axis_norm - 1.0))):.3e})"
            )
        orthogonality = np.abs(np.sum(normal * long_axis, axis=-1))
        if float(np.max(orthogonality)) > 1.0e-3:
            raise ValueError(
                f"{path}: measured racket face/long axes are not orthogonal "
                f"(max abs dot {float(np.max(orthogonality)):.3e})"
            )
        return {
            "schema_version": cls._MEASURED_RACKET_SCHEMA,
            "position_semantics": position_semantics,
            "normal_semantics": normal_semantics,
            "long_axis_semantics": long_axis_semantics,
            "robot_mount_normal_sign": robot_mount_normal_sign,
            "robot_butt_to_blade_axis_local": robot_axis_local.tolist(),
            "robot_rigid_visual_mesh_sha256": robot_mesh_sha256,
            "source_sha256": source_sha256,
            "retarget_receipt_sha256": receipt_sha256,
            "joint_order_contract_id": joint_order_contract_id,
            "joint_order_contract_sha256": joint_order_contract_sha256,
        }

    @staticmethod
    def _raw_first_three_pose_bytes_static(data, frame_count: int) -> bool:
        """Prove an exact float32-static prefix before runtime conversion.

        The split-ready roundoff exception is source-specific.  Comparing only
        the runtime tensors would allow a float64 sub-ULP change to disappear
        during MotionLoader's float32 conversion, while ``torch.equal`` also
        treats ``+0.0`` and ``-0.0`` as equal.  Require native float32 source
        arrays and compare their C-order row bytes without any numeric cast.
        """

        if int(frame_count) < 3:
            return False
        for channel_name in ("joint_pos", "body_pos_w", "body_quat_w"):
            array = data[channel_name]
            if array.dtype != np.dtype(np.float32):
                return False
            row0 = np.ascontiguousarray(array[0]).tobytes(order="C")
            if (
                row0 != np.ascontiguousarray(array[1]).tobytes(order="C")
                or row0 != np.ascontiguousarray(array[2]).tobytes(order="C")
            ):
                return False
        return True

    def __init__(
        self,
        motion_file,
        body_indexes: Sequence[int],
        *,
        motion_payloads: Sequence[bytes] | None = None,
        articulation_body_names: Sequence[str],
        selected_body_names: Sequence[str],
        device: str = "cpu",
        allow_legacy_link_origin_velocity: bool = False,
    ):
        files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        if not files:
            raise ValueError("MotionLoader needs at least one motion file")
        if motion_payloads is None:
            payloads: tuple[bytes | None, ...] = (None,) * len(files)
        else:
            try:
                payloads = tuple(motion_payloads)
            except TypeError as exc:
                raise ValueError(
                    "MotionLoader motion_payloads must be an ordered byte sequence"
                ) from exc
            if len(payloads) != len(files) or any(
                type(payload) is not bytes for payload in payloads
            ):
                raise ValueError(
                    "MotionLoader needs exactly one immutable bytes snapshot per motion file"
                )
        articulation_names = tuple(str(name) for name in articulation_body_names)
        selected_names = tuple(str(name) for name in selected_body_names)
        if (not articulation_names or len(set(articulation_names)) != len(articulation_names)
                or not selected_names or len(set(selected_names)) != len(selected_names)):
            raise ValueError("runtime articulation/selected body names must be non-empty and unique")
        indexes = [int(value) for value in (
            body_indexes.detach().cpu().tolist()
            if hasattr(body_indexes, "detach")
            else list(body_indexes)
        )]
        if len(indexes) != len(selected_names):
            raise ValueError(
                f"selected body indexes/names disagree: {indexes} vs {list(selected_names)}"
            )
        if any(index < 0 or index >= len(articulation_names) for index in indexes):
            raise ValueError(f"selected body index is outside articulation order: {indexes}")
        resolved_selected = tuple(articulation_names[index] for index in indexes)
        if resolved_selected != selected_names:
            raise ValueError(
                f"runtime selected body order mismatch: indexes resolve to {list(resolved_selected)}, "
                f"configured={list(selected_names)}"
            )
        jp, jv, bp, bq, bl, ba = [], [], [], [], [], []
        measured_racket_pos, measured_racket_normal, measured_racket_long_axis = [], [], []
        measured_racket_presence = []
        self.measured_racket_contracts = []
        seg_lens = []
        self.kinematics_contracts = []
        split_ready_raw_prefix_pose_bytes_static = []
        per_clip_fps = []
        for f, payload in zip(files, payloads):
            if payload is None:
                if not os.path.isfile(f):
                    raise FileNotFoundError(f"Invalid motion file path: {f}")
                source = f
            else:
                source = io.BytesIO(payload)
            with np.load(source, allow_pickle=False) as data:
                fps = self._fps_scalar(data, f)
                per_clip_fps.append(fps)
                frame_count = self._validate_motion_array_shapes(
                    data, f, len(articulation_names)
                )
                _kin = self._kinematics_contract(
                    data,
                    f,
                    articulation_names,
                    allow_legacy_link_origin_velocity=allow_legacy_link_origin_velocity,
                )
                self.kinematics_contracts.append(_kin)
                split_ready_raw_prefix_pose_bytes_static.append(
                    self._raw_first_three_pose_bytes_static(data, frame_count)
                )
                _measured = self._measured_racket_contract(data, f, frame_count)
                measured_racket_presence.append(_measured is not None)
                self.measured_racket_contracts.append(_measured)
                if not _kin["exact"]:
                    print(
                        f"[MotionLoader WARN] {f}: legacy motion lacks a schema-2 bound body order; "
                        "allowed for checkpoint compatibility but formal lineage is exact-ineligible. "
                        "Migrate/re-export the clip with kinematics schema 2. "
                        f"audit={_kin}",
                        flush=True,
                    )
                jp.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
                jv.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
                bp.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
                bq.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
                bl.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
                ba.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
                if _measured is not None:
                    measured_racket_pos.append(
                        torch.tensor(
                            data["measured_racket_site_pos_w"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    measured_racket_normal.append(
                        torch.tensor(
                            data["measured_racket_normal_w"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    measured_racket_long_axis.append(
                        torch.tensor(
                            data["measured_racket_long_axis_w"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                seg_lens.append(frame_count)
        self._split_ready_raw_prefix_pose_bytes_static = tuple(
            split_ready_raw_prefix_pose_bytes_static
        )
        if any(measured_racket_presence) and not all(measured_racket_presence):
            missing = [files[index] for index, present in enumerate(measured_racket_presence) if not present]
            raise ValueError(
                "mixed measured-racket availability across one motion bank; missing channels in "
                f"{missing}"
            )
        first_fps = per_clip_fps[0]
        if any(not math.isclose(value, first_fps, rel_tol=0.0, abs_tol=1.0e-12)
               for value in per_clip_fps[1:]):
            raise ValueError(f"motion clips have unequal fps values: {per_clip_fps}")
        self.fps = first_fps
        self.per_clip_fps = tuple(per_clip_fps)
        self.joint_pos = torch.cat(jp, dim=0)
        self.joint_vel = torch.cat(jv, dim=0)
        self._body_pos_w = torch.cat(bp, dim=0)
        self._body_quat_w = torch.cat(bq, dim=0)
        self._body_lin_vel_w = torch.cat(bl, dim=0)
        self._body_ang_vel_w = torch.cat(ba, dim=0)
        self.measured_racket_available = bool(measured_racket_presence and all(measured_racket_presence))
        self.measured_racket_mount_normal_sign_per_clip = (
            tuple(
                int(contract["robot_mount_normal_sign"])
                for contract in self.measured_racket_contracts
            )
            if self.measured_racket_available
            else ()
        )
        self._measured_racket_site_pos_w = (
            torch.cat(measured_racket_pos, dim=0) if self.measured_racket_available else None
        )
        self._measured_racket_normal_w = (
            torch.cat(measured_racket_normal, dim=0) if self.measured_racket_available else None
        )
        self._measured_racket_long_axis_w = (
            torch.cat(measured_racket_long_axis, dim=0)
            if self.measured_racket_available
            else None
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        # Per-clip segment boundaries on the concatenated time axis.
        self.num_segments = len(seg_lens)
        self.seg_len = torch.tensor(seg_lens, dtype=torch.long, device=device)
        self.seg_start = torch.zeros(self.num_segments, dtype=torch.long, device=device)
        if self.num_segments > 1:
            self.seg_start[1:] = torch.cumsum(self.seg_len, dim=0)[:-1]
        self.kinematics_contract_exact = all(item["exact"] for item in self.kinematics_contracts)

    @property
    def split_ready_raw_prefix_pose_bytes_static(self) -> tuple[bool, ...]:
        """Read-only source-byte evidence aligned one-to-one with clips."""

        return self._split_ready_raw_prefix_pose_bytes_static

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


CLIP_FAMILY_FOREHAND = "forehand"
CLIP_FAMILY_BACKHAND = "backhand"
_CLIP_FAMILIES = (CLIP_FAMILY_FOREHAND, CLIP_FAMILY_BACKHAND)
_A3_CANONICAL_READY_JOINT_COUNT = 31
_A3_PHYSX_CONTROL_POSITION_LIMIT_JOINT_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


def resolve_clip_family_is_forehand(clip_family_per_clip, num_segments: int) -> tuple[bool, ...]:
    """Resolve the per-clip swing-family config into "is clip i a forehand?" booleans.

    人话:回答"第 i 个 clip 是正手还是反手"。没配表(None,现役所有在跑臂)就按老规矩推:
    单 clip 当正手、恰好 2 clip = (正手, 反手)——和四处写死的 ``clips == 0`` 判断逐字节同值;
    3 个及以上 clip 没配表直接报错,因为那正是变速正手变体会被悄悄当成反手训错的场景
    (spdmix v2 可行性备忘 2026-07-22 硬绑定一),宁可开机炸也不猜。配了表就整表核:长度要对上
    加载的 clip 数、值只认 "forehand"/"backhand"、正反手至少各一个,错了当场 ValueError。
    """
    nseg = int(num_segments)
    if nseg < 1:
        raise ValueError(f"clip family resolution needs at least one loaded clip, got {nseg}")
    if clip_family_per_clip is None:
        if nseg == 1:
            return (True,)
        if nseg == 2:
            return (True, False)
        raise ValueError(
            f"the loaded motion has {nseg} clips but task.motion.clip_family_per_clip is unset — "
            "the legacy 'clip 0 is the forehand, every other clip is the backhand' rule only ever "
            "matched the exact (forehand, backhand) 2-clip list; with more clips it would silently "
            "mislabel every extra forehand variant as a backhand (swing_sign/obs/target side all "
            "wrong). Declare one family per clip in motion_file order, e.g. "
            '["forehand","forehand","forehand","backhand","backhand","backhand"].'
        )
    families = tuple(str(value) for value in clip_family_per_clip)
    if len(families) != nseg:
        raise ValueError(
            f"clip_family_per_clip has {len(families)} entries but the loaded motion has {nseg} "
            "clip(s) — align it with the motion_file clip order (same order as "
            "strike_phase_per_clip / mount_normal_sign_per_clip)"
        )
    unknown = sorted(set(families) - set(_CLIP_FAMILIES))
    if unknown:
        raise ValueError(
            f"clip_family_per_clip entries must be one of {_CLIP_FAMILIES}, got {unknown}"
        )
    # The both-families rule is about the UNIFIED policy: with two or more clips, swing_sign, the
    # swing-type observation and the target side are all keyed off the family split, so a one-sided
    # table would train one lane and leave the other dead. A SINGLE-clip run has no split to key on
    # — swing_sign is one constant for every env — so the rule has nothing to protect there, and
    # applying it anyway leaves a single-clip arm no way to say which hand it is. It then falls into
    # the ``None`` default, which hardcodes "single clip is a forehand": every backhand-only arm
    # silently reports as a forehand and its per-side metrics read a structural 0.0000 while the
    # aggregate moves. 人话:一条只有反手的臂本来连"我是反手"都说不出口,只能被默认当成正手,
    # 于是逐侧指标恒为 0 —— 正是 07-26 把 45% 回球率读废的那个坑的镜像。
    if nseg >= 2 and (
        CLIP_FAMILY_FOREHAND not in families or CLIP_FAMILY_BACKHAND not in families
    ):
        raise ValueError(
            "clip_family_per_clip must contain at least one forehand and one backhand clip, got "
            f"{families} — the unified policy keys swing_sign, the swing-type observation and the "
            "target side off both families"
        )
    return tuple(value == CLIP_FAMILY_FOREHAND for value in families)


class _BalancedRoundRobinClipSampler:
    """Deterministic, exactly balanced clip allocation without touching global RNG.

    One seeded permutation defines a cyclic clip order. Every prefix of that
    infinite cycle gives each clip either ``floor(k / N)`` or ``ceil(k / N)``
    assignments, so the cumulative count spread is always at most one even
    when callers use different batch sizes.
    """

    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        num_segments: int,
        seed: int,
        clip_order: Sequence[str],
        device,
    ):
        if type(num_segments) is not int or num_segments < 1:
            raise ValueError(
                "balanced clip sampler num_segments must be a positive integer, "
                f"got {num_segments!r}"
            )
        if type(seed) is not int or not (0 <= seed < 2**63):
            raise ValueError(
                "balanced_clip_sampling_seed must be an integer in [0, 2**63)"
            )
        order = tuple(clip_order)
        if len(order) != num_segments or any(
            type(item) is not str for item in order
        ):
            raise ValueError(
                "balanced clip sampler clip_order must contain one path string per segment"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.num_segments = num_segments
        self.seed = seed
        self.clip_order = order
        self.device = torch.device(device)
        self.permutation = torch.randperm(
            num_segments, generator=generator, dtype=torch.long
        ).to(self.device)
        self.cursor = 0

    def sample(self, count: int) -> torch.Tensor:
        if type(count) is not int or count < 0:
            raise ValueError(
                "balanced clip sample count must be a non-negative integer, "
                f"got {count!r}"
            )
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        positions = (
            torch.arange(count, dtype=torch.long, device=self.device) + self.cursor
        ) % self.num_segments
        sampled = self.permutation[positions]
        self.cursor = (self.cursor + count) % self.num_segments
        return sampled

    def state_dict(self) -> dict:
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "num_segments": self.num_segments,
            "seed": self.seed,
            "clip_order": self.clip_order,
            "permutation": tuple(
                int(value) for value in self.permutation.cpu().tolist()
            ),
            "cursor": self.cursor,
        }

    def validate_state_dict(self, state: dict) -> tuple[tuple[int, ...], int]:
        """Validate one saved cursor without changing the live sampler.

        Command checkpoint preflight runs before *any* policy, optimizer, normalizer, command, or
        action state is allowed to move.  Keeping this parser separate from ``load_state_dict``
        lets ``MotionCommand.validate_exact_resume_state_dict`` exercise the exact same schema and
        identity checks without the old mutate-then-rollback validation trick.
        """

        if type(state) is not dict:
            raise ValueError("balanced clip sampler state must be a dictionary")
        expected_keys = {
            "schema_version",
            "num_segments",
            "seed",
            "clip_order",
            "permutation",
            "cursor",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "balanced clip sampler state keys do not match the strict schema"
            )
        if state.get("schema_version") != self._STATE_SCHEMA_VERSION:
            raise ValueError(
                "balanced clip sampler state has an unsupported schema_version"
            )
        if state.get("num_segments") != self.num_segments:
            raise ValueError(
                "balanced clip sampler state num_segments does not match the loaded motion"
            )
        if state.get("seed") != self.seed:
            raise ValueError(
                "balanced clip sampler state seed does not match the configured seed"
            )
        if tuple(state.get("clip_order", ())) != self.clip_order:
            raise ValueError(
                "balanced clip sampler state clip_order does not match the loaded motion order"
            )
        permutation = state.get("permutation")
        if type(permutation) not in (tuple, list):
            raise ValueError(
                "balanced clip sampler state permutation must be an ordered sequence"
            )
        if (
            any(type(value) is not int for value in permutation)
            or sorted(permutation) != list(range(self.num_segments))
        ):
            raise ValueError(
                "balanced clip sampler state permutation is not a bijection of clip ids"
            )
        cursor = state.get("cursor")
        if type(cursor) is not int or not (0 <= cursor < self.num_segments):
            raise ValueError(
                "balanced clip sampler state cursor must be an integer inside the permutation"
            )
        return tuple(permutation), cursor

    def load_state_dict(self, state: dict):
        permutation, cursor = self.validate_state_dict(state)
        # Restore saved bytes instead of regenerating, so exact resume survives
        # a future torch release changing randperm internals.
        self.permutation = torch.tensor(
            permutation, dtype=torch.long, device=self.device
        )
        self.cursor = cursor


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _EXACT_RESUME_STATE_KIND = "whole_body_tracking.MotionCommand"
    _EXACT_RESUME_STATE_SCHEMA_VERSION = 2
    _ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION = 5
    _ACTION_BALL_INT64_MAX = (1 << 63) - 1
    # A measured source whose first three poses are bitwise static can carry a
    # tiny frame-0 angular-velocity residue from float64 quaternion arithmetic
    # before storage as float32.  This diagnostic-only bound applies to
    # body_ang_vel_w alone; joint/linear velocity remain literal-zero gates.
    _SPLIT_READY_TEACHER_START_BODY_ANG_ROUNDOFF_MAX = 1.0e-14

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _bind_disabled_time_resampling_fast_path(self)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        canonical_ready_mode = getattr(self.cfg, "canonical_ready_mode", False)
        if type(canonical_ready_mode) is not bool:
            raise ValueError("canonical_ready_mode must be an exact boolean")
        self.canonical_ready_mode = canonical_ready_mode
        diagnostic_split_ready_teacher = getattr(
            self.cfg,
            "action_ball_diagnostic_split_ready_teacher",
            False,
        )
        if type(diagnostic_split_ready_teacher) is not bool:
            raise ValueError(
                "action_ball_diagnostic_split_ready_teacher must be an exact boolean"
            )
        self.action_ball_diagnostic_split_ready_teacher = (
            diagnostic_split_ready_teacher
        )
        single_stroke_timeout_enabled = getattr(
            self.cfg,
            "action_ball_single_stroke_timeout_enabled",
            False,
        )
        if type(single_stroke_timeout_enabled) is not bool:
            raise ValueError(
                "action_ball_single_stroke_timeout_enabled must be an exact boolean"
            )
        if (
            single_stroke_timeout_enabled
            and not diagnostic_split_ready_teacher
        ):
            raise ValueError(
                "action_ball_single_stroke_timeout_enabled requires "
                "action_ball_diagnostic_split_ready_teacher=true"
            )
        self.action_ball_single_stroke_timeout_enabled = (
            single_stroke_timeout_enabled
        )
        self._canonical_motion_registry = None
        self._canonical_motion_admission = None
        self._canonical_motion_promotion_binding = None
        self._canonical_motion_registry_module = None
        # Freeze Hydra/ListConfig/custom iterable input once.  Admission hashes
        # and MotionLoader must consume the same ordered path identity.
        self._motion_files = self._configured_motion_files(self.cfg.motion_file)
        # Bind the bytes actually admitted at construction, not whatever may happen to occupy
        # the same paths at checkpoint time. Exact resume must never pour an old curriculum or
        # replay ring into different clip content that reused the same filenames.
        self._motion_file_sha256 = tuple(
            sha256_file(path) for path in self._motion_files
        )
        racket_cfg_for_diag = getattr(
            getattr(getattr(env, "cfg", None), "commands", None),
            "racket_target",
            None,
        )
        diagnostic_unauthorized = getattr(
            racket_cfg_for_diag,
            "action_ball_diagnostic_unauthorized",
            False,
        )
        if type(diagnostic_unauthorized) is not bool:
            raise ValueError(
                "action_ball_diagnostic_unauthorized must be an exact boolean"
            )
        self._canonical_diagnostic_unauthorized = diagnostic_unauthorized
        diagnostic_catalog = getattr(
            self.cfg, "action_ball_full_mdp_diagnostic_catalog", None
        )
        self._action_ball_full_mdp_diagnostic_catalog_table = None
        if diagnostic_catalog is not None:
            env_cfg = getattr(env, "cfg", None)
            # A module/name pair can be forged by an unrelated class.  Import
            # the already initialized cfg module at this live constructor
            # boundary and require one of its two exact registered leaf types.
            from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
                hope_env_cfg as _hope_env_cfg,
            )

            exact_env_cfg_types = (
                _hope_env_cfg.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg,
                _hope_env_cfg.HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg,
            )
            env_commands_cfg = getattr(env_cfg, "commands", None)
            env_motion_cfg = getattr(env_commands_cfg, "motion", None)
            racket_cfg = getattr(env_commands_cfg, "racket_target", None)
            if (
                diagnostic_catalog
                != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND
                or type(env_cfg) not in exact_env_cfg_types
                # ManagerBase deliberately deep-copies its cfg before term
                # construction.  Require both that source and this adopted
                # copy below; object identity would reject every real
                # CommandManager construction and is not a trust boundary.
                or type(env_motion_cfg) is not type(self.cfg)
                or str(getattr(racket_cfg, "target_mode", ""))
                != "action_ball_full_mdp"
                or diagnostic_unauthorized is not True
                or canonical_ready_mode is not True
                or diagnostic_split_ready_teacher is not True
                or single_stroke_timeout_enabled is not False
                or getattr(self.cfg, "action_ball_dynamic_ready", None)
                is None
                or getattr(self.cfg, "allow_legacy_link_origin_velocity", None)
                is not False
                or getattr(self.cfg, "balanced_clip_sampling", None) is not True
                or getattr(self.cfg, "stand_start_prob", None) != 0.0
                or tuple(getattr(self.cfg, "stand_start_yaw_range", ()))
                != (0.0, 0.0)
                or tuple(getattr(self.cfg, "hold_steps_range", ())) != (0, 0)
                or getattr(self.cfg, "stand_start_min_hold", None) != 0
                or getattr(self.cfg, "post_swing_start_prob", None) != 0.0
                or getattr(self.cfg, "post_swing_min_hold", None) != 0
                or getattr(self.cfg, "wrap_teleport", None) is not False
                or getattr(self.cfg, "clip_switch_prob", None) != 0.0
                or getattr(self.cfg, "event_timing_mode", None)
                != EVENT_TIMING_MODE_DISABLED
                or tuple(getattr(self.cfg, "speed_scale_range", ()))
                != (1.0, 1.0)
                or getattr(self.cfg, "speed_scale_per_clip", None) is not None
                or getattr(self.cfg, "stagger_initial_clock", None) is not False
                or getattr(self.cfg, "stagger_hold_max_steps", None) != 0
                or getattr(self.cfg, "rsi_skip_settle_frames", None) != 0
                or getattr(self.cfg, "planner_revision_enabled", None) is not False
                or tuple(getattr(self.cfg, "joint_position_range", ()))
                != (0.0, 0.0)
                or any(
                    tuple(getattr(self.cfg, name, {}).get(axis, ()))
                    != (0.0, 0.0)
                    for name in ("pose_range", "velocity_range")
                    for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                )
            ):
                raise ValueError(
                    "full-MDP diagnostic catalog is restricted to the exact fresh "
                    "N=1 diagnostic EnvCfg with one dynamic split-ready binding"
                )
            table = load_action_ball_full_mdp_diagnostic_catalog_table()
            if (
                require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
                    env_motion_cfg,
                    racket_cfg,
                    table=table,
                )
                is not table
                or require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
                    self.cfg,
                    racket_cfg,
                    table=table,
                )
                is not table
                or self._motion_files != table.motion_files
                or self._motion_file_sha256 != table.motion_sha256
            ):
                raise ValueError(
                    "fresh full-MDP Motion/Racket table differs from the code-owned "
                    "active N=1 diagnostic catalog"
                )
            self._action_ball_full_mdp_diagnostic_catalog_table = table
        # 起点扰动斜坡必须在 canonical-ready 复位守卫之前解析:那道守卫要知道
        # "非零的静态种子是不是有一条已声明的 ramp 在背书"。
        self._configure_start_pose_ramp()
        if self.action_ball_diagnostic_split_ready_teacher and (
            not self.canonical_ready_mode or not diagnostic_unauthorized
        ):
            raise ValueError(
                "action_ball_diagnostic_split_ready_teacher requires "
                "canonical_ready_mode=true and "
                "action_ball_diagnostic_unauthorized=true"
            )
        if self._action_ball_full_mdp_diagnostic_catalog_table is not None:
            # The capture clips are not ready-to-ready loops.  This branch
            # owns only immutable adoption of their exact current bytes; D05
            # and the hot Motion epoch remain responsible for first reveal and
            # recovery, and formal/reset authority remains absent.
            self._motion_payloads = self._snapshot_diagnostic_motion_bytes()
        elif self.canonical_ready_mode and diagnostic_unauthorized:
            # Franco 2026-07-28 approved DIAGNOSTIC bypass: skip the registry
            # trust chain only.  The physical canonical-ready clip contract
            # (_validate_canonical_ready_clips) and the reset-curricula guard
            # below stay fully enforced — a bypassed run may not corrupt the
            # ready-entry geometry, it may only skip authorization.  Retain an
            # immutable snapshot even in diagnostic mode so MotionLoader and
            # the later action-ball broker bind the same bytes.  This snapshot
            # proves identity/TOCTOU closure only; it does not mint canonical
            # admission.
            print(
                "[MotionCommand] WARN canonical_ready_mode DIAGNOSTIC "
                "UNAUTHORIZED: registry/certificate admission bypassed; "
                "clip ready-entry contract still enforced",
                flush=True,
            )
            self._validate_canonical_ready_config()
            self._canonical_registry_tables = None
            self._motion_payloads = (
                self._snapshot_diagnostic_motion_bytes()
            )
        elif self.canonical_ready_mode:
            self._validate_canonical_ready_config()
            self._canonical_registry_tables = (
                self._load_and_validate_canonical_registry(env)
            )
            self._motion_payloads = self._snapshot_canonical_motion_bytes()
        else:
            # Default (non-canonical) motion_file channel: pre-branch behavior —
            # MotionLoader reads the raw NPZ paths directly, no code-owned trust
            # set. Admission is scoped to the canonical registry consumer above.
            self._motion_payloads = None
        self.motion = MotionLoader(
            self._motion_files,
            self.body_indexes,
            motion_payloads=self._motion_payloads,
            articulation_body_names=self.robot.body_names,
            selected_body_names=self.cfg.body_names,
            device=self.device,
            allow_legacy_link_origin_velocity=bool(
                self.cfg.allow_legacy_link_origin_velocity
            ),
        )
        if self._action_ball_full_mdp_diagnostic_catalog_table is not None:
            table = self._action_ball_full_mdp_diagnostic_catalog_table
            if (
                int(self.motion.num_segments)
                != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT
                or self.motion.kinematics_contract_exact is not True
                or self.motion.measured_racket_available is not True
                or tuple(
                    float(value)
                    for value in (
                        self.motion.measured_racket_mount_normal_sign_per_clip
                    )
                )
                != table.mount_normal_sign_per_clip
                or type(self._motion_payloads) is not tuple
                or len(self._motion_payloads)
                != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT
                or tuple(
                    hashlib.sha256(value).hexdigest()
                    for value in self._motion_payloads
                )
                != table.motion_sha256
            ):
                raise ValueError(
                    "fresh full-MDP diagnostic MotionLoader did not adopt the exact "
                    "active N=1 schema-2/measured-racket catalog"
                )
        if (
            bool(
                getattr(
                    self,
                    "action_ball_diagnostic_split_ready_teacher",
                    False,
                )
            )
            and int(self.motion.num_segments) != 1
        ):
            raise ValueError(
                "action_ball_diagnostic_split_ready_teacher is restricted to "
                "one measured N=1 stroke"
            )
        if self.canonical_ready_mode:
            if not self._canonical_diagnostic_unauthorized:
                self._validate_canonical_registry_motion_bytes()
            self._validate_canonical_ready_clips()
        self._configure_action_ball_dynamic_ready()
        if (
            bool(
                getattr(
                    self,
                    "action_ball_diagnostic_split_ready_teacher",
                    False,
                )
            )
            and self._action_ball_dynamic_ready_binding_sha256 is None
        ):
            raise ValueError(
                "action_ball_diagnostic_split_ready_teacher requires one "
                "validated action_ball_dynamic_ready binding"
            )
        if (
            self.action_ball_diagnostic_split_ready_teacher
            and self.action_ball_single_stroke_timeout_enabled
        ):
            print(
                "[MotionCommand] WARN measured N=1 split-ready diagnostic: "
                "true reset uses binding physical_ready; the non-looping "
                "teacher remains the immutable motion timeline and terminates "
                "after one complete stroke",
                flush=True,
            )
        elif self.action_ball_diagnostic_split_ready_teacher:
            print(
                "[MotionCommand] WARN measured N=1 split-ready bridge: "
                "single-stroke completion timeout is disabled, so clip-cycle "
                "rows may enter the existing natural-wrap path; this switch "
                "alone does not establish recovery or continuous-task semantics",
                flush=True,
            )
        expected_fps = 1.0 / float(env.step_dt)
        if not math.isfinite(expected_fps) or not math.isclose(
            self.motion.fps, expected_fps, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                "motion fps must equal the policy rate exactly enough for one-frame-per-step "
                f"playback: clips={list(self.motion.per_clip_fps)} policy_hz={expected_fps:.12g}"
            )
        # GROUNDING preflight (2026-07-03): grounding is a world transform of
        # the articulation root, not the tracked torso anchor.  A torso can
        # legitimately yaw tens of degrees at swing frame 0 while the pelvis
        # and every measured world-space racket channel are already +X
        # grounded.  Checking the torso here produced false turn-and-walk
        # warnings for the exact active N=1 measured row.
        configured_body_names = tuple(str(name) for name in self.cfg.body_names)
        robot_body_names = tuple(str(name) for name in self.robot.body_names)
        if (
            not configured_body_names
            or configured_body_names[0] != "pelvis_link"
            or not robot_body_names
            or robot_body_names[0] != configured_body_names[0]
        ):
            raise ValueError(
                "motion grounding preflight requires pelvis_link as the "
                "first selected and articulation body"
            )
        grounding_body_index = 0
        for _c in range(self.motion.num_segments):
            _q0 = self.motion.body_quat_w[
                int(self.motion.seg_start[_c]), grounding_body_index
            ]
            _w, _x, _y, _z = (float(_q0[0]), float(_q0[1]), float(_q0[2]), float(_q0[3]))
            _yaw0 = math.degrees(math.atan2(2.0 * (_w * _z + _x * _y), 1.0 - 2.0 * (_y * _y + _z * _z)))
            if abs(_yaw0) > 10.0:
                print(
                    f"[MotionCommand WARN] clip {_c} frame-0 tracked-root yaw = {_yaw0:+.1f} deg — this clip "
                    "was NOT re-grounded to +X (scripts/reground_hope_frame.py). Target boxes assume "
                    "+X grounding; training on it produces a turn-and-walk policy that needs "
                    "oracle/mocap localization at deploy. Pin registry_name to the re-grounded "
                    "lineage (hopex/v3) or re-ground and re-upload before training.",
                    flush=True,
                )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # R14 retiming: float shadow clock + per-env playback speed. Inactive at the default
        # (1.0, 1.0), keeping the integer-clock path byte-identical; when active, time_steps is
        # derived as round(time_steps_f) (matching the deploy clock's round() in time_step_for).
        _s_rng = tuple(float(x) for x in self.cfg.speed_scale_range)
        if len(_s_rng) != 2 or not (0.0 < _s_rng[0] <= _s_rng[1]):
            raise ValueError(f"speed_scale_range must be (lo, hi) with 0 < lo <= hi, got {self.cfg.speed_scale_range}")
        _s_lo, _s_hi = _s_rng
        self.retiming_active = not (_s_lo == 1.0 and _s_hi == 1.0)
        # FIXED per-clip playback speed (backhand-fix ablation 2026-07-08): e.g. (1.0, 0.8) plays
        # the backhand reference at 0.8x while the forehand stays 1.0x. Deterministic per clip
        # (no per-swing randomness), rides the same R14 float-clock path. Overrides
        # speed_scale_range sampling when set; None (default) = byte-identical legacy behavior.
        self._speed_per_clip = None
        if getattr(self.cfg, "speed_scale_per_clip", None) is not None:
            _spc = tuple(float(x) for x in self.cfg.speed_scale_per_clip)
            if any(s <= 0.0 for s in _spc):
                raise ValueError(f"speed_scale_per_clip must be positive, got {_spc}")
            if len(_spc) != self.motion.num_segments:
                raise ValueError(
                    f"speed_scale_per_clip has {len(_spc)} entries but the motion has "
                    f"{self.motion.num_segments} clip(s)")
            self._speed_per_clip = torch.tensor(_spc, device=self.device)
            self.retiming_active = True
            print(f"[MotionCommand] speed_scale_per_clip ACTIVE: {_spc} "
                  f"(fixed per-clip reference playback; overrides speed_scale_range)", flush=True)
        self.time_steps_f = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.speed_scale = torch.ones(self.num_envs, device=self.device)
        # Same-ball planner revisions (default OFF).  This is deliberately a separate clock from
        # the historical R14 random/fixed retimer: a revised deadline changes the phase rate of the
        # *same* physical ball task, never the clip, bank row, reward truth, or simulator state.
        # The RacketTargetCommand installs an immutable task identity and then submits one atomic
        # target/TTS revision per policy step.  This command consumes the accepted TTS on the next
        # step and advances a monotonic, acceleration-bounded reference phase.
        self.planner_revision_enabled = bool(
            getattr(self.cfg, "planner_revision_enabled", False)
        )
        self._planner_revision_profile: PhaseGovernorProfile | None = None
        self._planner_initial_tts_mixture: InitialTtsMixture | None = None
        if self.planner_revision_enabled:
            raw_profile = getattr(self.cfg, "planner_revision_profile", None)
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete planner_revision_profile mapping"
                )
            self._planner_revision_profile = PhaseGovernorProfile.from_mapping(raw_profile)
            initial_tts = tuple(
                float(value)
                for value in getattr(
                    self.cfg, "planner_revision_initial_tts_range_s", ()
                )
            )
            if (
                len(initial_tts) != 2
                or not math.isfinite(initial_tts[0])
                or not math.isfinite(initial_tts[1])
                or not (
                    self._planner_revision_profile.min_tts_s
                    <= initial_tts[0]
                    < initial_tts[1]
                    <= self._planner_revision_profile.max_tts_s
                )
            ):
                raise ValueError(
                    "planner_revision_initial_tts_range_s must be a non-degenerate ordered "
                    "finite pair inside "
                    "the complete profile TTS envelope"
                )
            raw_mixture = getattr(
                self.cfg, "planner_revision_initial_tts_mixture", None
            )
            if not isinstance(raw_mixture, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete "
                    "planner_revision_initial_tts_mixture mapping"
                )
            self._planner_initial_tts_mixture = InitialTtsMixture.from_mapping(
                raw_mixture
            )
            self._planner_initial_tts_mixture.validate_support(
                lo_s=initial_tts[0], hi_s=initial_tts[1]
            )
            if not math.isclose(
                self._planner_revision_profile.policy_dt_s,
                float(env.step_dt),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "planner revision profile policy_dt_s must equal the runtime policy step: "
                    f"profile={self._planner_revision_profile.policy_dt_s} runtime={env.step_dt}"
                )
            if self._speed_per_clip is not None or _s_rng != (1.0, 1.0):
                raise ValueError(
                    "planner revision phase governor is incompatible with R14 speed_scale_range/"
                    "speed_scale_per_clip; it owns the sole reference clock"
                )
            if str(getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED)) \
                    != EVENT_TIMING_MODE_DISABLED:
                raise ValueError(
                    "planner revision phase governor is incompatible with event_timing_mode; "
                    "one task may have only one deadline owner"
                )
            if (
                tuple(int(value) for value in self.cfg.hold_steps_range) != (0, 0)
                or int(self.cfg.stand_start_min_hold) != 0
                or int(self.cfg.post_swing_min_hold) != 0
            ):
                raise ValueError(
                    "planner revision initial_tts_range_s owns preparation time; legacy random/min "
                    "hold clocks must all be zero"
                )
            # Force velocity scaling through the existing, audited retiming lane.  Unlike R14,
            # speed_scale is recomputed from the *actual* phase delta each step below.
            self.retiming_active = True
            n = self.num_envs
            self._planner_active = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._planner_control_epoch = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_revision = torch.full(
                (n,), -1, dtype=torch.long, device=self.device
            )
            self._planner_start_step = torch.zeros(n, device=self.device)
            self._planner_strike_step = torch.zeros(n, device=self.device)
            self._planner_phase_rate = torch.zeros(n, device=self.device)
            self._planner_slow_only_next = torch.zeros(
                n, dtype=torch.bool, device=self.device
            )
            self._planner_desired_tts = torch.zeros(n, device=self.device)
            self._planner_begin_tts = torch.zeros(n, device=self.device)
            self._planner_truth_tts = torch.zeros(n, device=self.device)
            # 带符号孪生时钟(2026-07-25):truth tts 是任务期限语义,触球后 clamp 钉 0
            # (obs/critic 读它,合同如此)。但 |tts|<=0.12 的击球窗掩码若也读它,窗就从
            # 触球一直开到 clip 收尾——随挥全程 ~50-100 步顶着 ±0.12 s 的设计语义,
            # position/normal 触球后停拍可薅、站稳包/face 税全程计费、模仿被 0.25x 捂嘴。
            # 窗掩码改读这条不截断的时钟:触球后照常转负,窗在 +0.12 s 如约关闭。
            # 非 active(reset 后新任务未装)置大正哨兵 = 窗关闭(fail-closed;
            # 旧行为是残留 0 → 空档期窗误开)。
            self._planner_truth_tts_signed = torch.full(
                (n,), 1.0e6, device=self.device
            )
            # Immutable task-begin envelope baseline.  A latest-value transport may legitimately
            # skip active revisions, so envelope checks may not depend on whichever revision the
            # consumer happened to observe previously.
            self._planner_begin_target_pos = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_vel = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_normal = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_normal[:, 0] = 1.0
            self.metrics["planner_revision_accepted"] = torch.zeros(n, device=self.device)
            self.metrics["planner_revision_rejected"] = torch.zeros(n, device=self.device)
            self.metrics["planner_phase_rate_per_s"] = torch.zeros(n, device=self.device)
            self.metrics["planner_truth_tts_s"] = torch.zeros(n, device=self.device)
        # Unified multi-clip (HITTER forehand+backhand) support. With one clip these are inert and the
        # behaviour below is byte-identical to the single-clip path. clip_id[env] selects which segment
        # (swing type) the env is currently imitating.
        self._multiseg = self.motion.num_segments > 1
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Action-conditioned ball-first is bound later by RacketTargetCommand, after both command
        # terms have been constructed from one admitted manifest.  Keeping every field ``None``
        # until that one-shot bind preserves the legacy reset path exactly, including its random
        # draws.  The broker owns immutable birth receipts; MotionCommand remains the sole owner of
        # articulation reset writes.
        self._action_ball_birth_broker = None
        self._action_ball_runtime_module_bound = None
        self._action_ball_trusted_repo_root = None
        self._action_ball_motion_admission_receipt_sha256 = None
        # Racket owns the solved per-swing task graph.  Motion receives only two public,
        # read-only callables: one frozen receipt accessor and one digest over Racket's complete
        # exact-resume payload.  No sampler/curriculum/pool object is retained here.
        self._action_ball_task_ref_for_env = None
        self._action_ball_task_receipt_resolver = None
        self._action_ball_shared_state_sha256_accessor = None
        self._action_ball_expected_shared_racket_state_sha256 = None
        # The immutable-N1 diagnostic may bind a separate receipt-free task
        # view after the ordinary birth authority is admitted.  ``None`` keeps
        # every online/banded/legacy branch byte-for-behavior unchanged.
        self._action_ball_fixed_view_identity_sha256 = None
        self._action_ball_fixed_view_timing_row = None
        self._action_ball_fixed_view_timing_row_device = None
        self._action_ball_fixed_view_broker_state_accessor = None
        self._action_ball_action_uids = None
        self._action_ball_motion_sha256 = None
        self._action_ball_segment_lengths = None
        self._action_ball_ready_root_z = None
        self._action_ball_ready_root_quat = None
        self._action_ball_reset_generation = None
        self._action_ball_swing_generation = None
        self._action_ball_birth_receipt_sha256 = None
        self._action_ball_seen_birth_receipts = None
        self._action_ball_active_task_refs = None
        self._action_ball_task_timing_active = None
        self._action_ball_public_task_valid = None
        self._action_ball_safe_ready_body_pos_w = None
        self._action_ball_safe_ready_body_quat_w = None
        self._action_ball_safe_ready_reference_pending = None
        self._action_ball_safe_ready_pending_count = None
        self._action_ball_diagnostic_pending_row_count = None
        self._action_ball_task_pending_elapsed_s = None
        self._action_ball_task_age_s = None
        self._action_ball_time_to_contact_s = None
        self._action_ball_teacher_rate = None
        self._action_ball_scaled_t_hit_s = None
        self._action_ball_scaled_t_cycle_s = None
        self._action_ball_pre_swing_wait_s = None
        balanced_clip_sampling = getattr(self.cfg, "balanced_clip_sampling", False)
        if type(balanced_clip_sampling) is not bool:
            raise ValueError("balanced_clip_sampling must be an exact boolean")
        self._balanced_clip_sampler: _BalancedRoundRobinClipSampler | None = None
        if balanced_clip_sampling:
            self._balanced_clip_sampler = _BalancedRoundRobinClipSampler(
                num_segments=int(self.motion.num_segments),
                seed=getattr(self.cfg, "balanced_clip_sampling_seed", 0),
                clip_order=self._motion_files,
                device=self.device,
            )
            print(
                "[MotionCommand] balanced_clip_sampling ACTIVE: "
                f"clips={self.motion.num_segments} "
                f"seed={self._balanced_clip_sampler.seed} "
                "(seeded round-robin clip allocation; exact count spread <= 1)",
                flush=True,
            )
        # 每 clip 的 forehand/backhand 家族表(spdmix v2 硬绑定一)。显式配置在这里整表校验
        # (boot fail-loud:长度==clip 数、值合法、正反手至少各一)并落成张量;None(默认,现役
        # 所有在跑臂)= 不建表、不打印、行为逐字节不变——查表方(clip_family_is_forehand)在第一次
        # 用到时按"单 clip 正手 / 恰 2 clip = (正手, 反手)"懒推导,>2 clip 缺表当场报错。
        self._clip_family_is_forehand_t: torch.Tensor | None = None
        if getattr(self.cfg, "clip_family_per_clip", None) is not None:
            self._clip_family_is_forehand_t = torch.tensor(
                resolve_clip_family_is_forehand(
                    self.cfg.clip_family_per_clip, int(self.motion.num_segments)
                ),
                dtype=torch.bool,
                device=self.device,
            )
            print(
                "[MotionCommand] clip_family_per_clip ACTIVE: "
                f"{tuple(str(value) for value in self.cfg.clip_family_per_clip)} "
                "(per-clip forehand/backhand lookup replaces the clips==0 hardcode)",
                flush=True,
            )
        # Robust per-step "this env just wrapped to a new swing" signal, consumed by the racket-target
        # command to resample its target. Replaces a time_steps<prev heuristic that fails when a clip
        # wrap jumps the index to a HIGHER segment start (forehand->backhand on the concatenated axis).
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Host-owned same-tick proof published only after the split-ready
        # non-loop writer has completed the full Motion update.  Racket never
        # infers emptiness from static mode flags alone.
        self._split_ready_empty_wrap_receipt = None
        # Pre-swing hold state (see cfg.hold_steps_range): while hold_counter > 0 the reference
        # clock is frozen at the swing's first frame ("waiting for the ball"). _update_command
        # decrements it. in_hold is exposed for rewards/metrics.
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # True only while _resample_command is being invoked from an intra-episode clip WRAP
        # (as opposed to a true episode reset) — wraps skip the RSI teleport (cfg.wrap_teleport).
        self._resampling_from_wrap = False
        # --- stagger_initial_clock (metric-sync fix 2026-07-09; default OFF = byte-identical) ------
        # Disease: 4096 envs constructed/resumed at the SAME instant + a low fall rate => they all
        # time out together, swing together, and reset together (episode_length sawtooth 52->485,
        # mass timeouts) — every EMA metric (fall rates, completion, return rates) then reads a
        # synchronized-queue oscillation instead of a steady rate. Cure, one flag, two one-shot
        # biases: (a) each env's FIRST true reset adds U[0, stagger_hold_max_steps] extra hold, so
        # the cohort's swing/strike phases spread within the first episode; (b) the first
        # _update_command after construction adds U[0, max_episode_length) to every env's episode
        # clock, so the FIRST timeouts — and every episode boundary after them — spread instead of
        # firing in one wave. 人话:开了它,4096 个 env 的"到点超时+挥拍节拍"被随机错开,EMA 指标
        # 不再集体振荡;默认关,现役跑法完全不受影响。
        self._stagger_hold_pending: torch.Tensor | None = None
        self._stagger_ep_pending = False
        if bool(getattr(self.cfg, "stagger_initial_clock", False)):
            self._stagger_hold_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._stagger_ep_pending = True
        # T1 continuous timing is deliberately a separate, fail-closed command path.  It reuses
        # native clip playback, exact pre-swing hold and no-wrap carry-state, but none of the
        # random wrap/switch/retiming mechanisms.  The schedule bytes are verified before any
        # event state exists; RacketTargetCommand later binds every immutable row to the loaded
        # train bank and supplies the native strike offset for each clip.
        self._event_timing_mode = str(
            getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED)
        )
        if self._event_timing_mode not in EVENT_TIMING_MODES:
            raise ValueError(
                f"event_timing_mode must be one of {EVENT_TIMING_MODES}, "
                f"got {self._event_timing_mode!r}"
            )
        self._event_schedule = None
        self._event_scheduler: EventTimingScheduler | None = None
        self._event_native_strike_ticks: torch.Tensor | None = None
        if self._event_timing_mode == EVENT_TIMING_MODE_POST_STRIKE_T1:
            schedule_path = str(getattr(self.cfg, "event_timing_schedule", "") or "").strip()
            schedule_sha = str(
                getattr(self.cfg, "event_timing_schedule_sha256", "") or ""
            ).strip()
            if not schedule_path or not schedule_sha:
                raise ValueError(
                    "post_strike_t1 requires event_timing_schedule and its exact byte SHA-256"
                )
            if bool(getattr(self.cfg, "event_timing_repeat", False)):
                raise ValueError(
                    "post_strike_t1 rows may not repeat within an episode; materialize enough "
                    "immutable rows and reset only at the sequence boundary"
                )
            if bool(self.cfg.wrap_teleport):
                raise ValueError("post_strike_t1 requires wrap_teleport=false (carry state)")
            if float(self.cfg.clip_switch_prob) != 0.0:
                raise ValueError("post_strike_t1 requires clip_switch_prob=0")
            if bool(self.cfg.stagger_initial_clock):
                raise ValueError("post_strike_t1 requires stagger_initial_clock=false")
            if self.retiming_active:
                raise ValueError("post_strike_t1 requires native one-frame-per-step playback")
            if int(getattr(self.cfg, "rsi_skip_settle_frames", 0)) != 0:
                raise ValueError(
                    "post_strike_t1 event installs require rsi_skip_settle_frames=0; skipping "
                    "native clip frames would change immutable deadline feasibility"
                )
            self._event_schedule = load_event_schedule(schedule_path, schedule_sha)
            actual_rate = 1.0 / float(env.step_dt)
            if not math.isclose(
                actual_rate,
                float(self._event_schedule.policy_rate_hz),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "event schedule policy rate does not match the instantiated control rate: "
                    f"schedule={self._event_schedule.policy_rate_hz} runtime={actual_rate:.12g}"
                )
            bad_clips = sorted(
                {row.clip_id for row in self._event_schedule.rows}
                - set(range(int(self.motion.num_segments)))
            )
            if bad_clips:
                raise ValueError(
                    f"event schedule references unloaded motion clip ids {bad_clips}"
                )
            self._event_scheduler = EventTimingScheduler(
                self._event_schedule,
                num_envs=self.num_envs,
                device=self.device,
            )
        self._configure_action_ball_continuous_motion_cadence()
        # A8: post-swing initial-state ring buffer (root state stored ORIGIN-RELATIVE in [:3] so a
        # snapshot from env B can seed env A; quats/velocities/joints are origin-invariant).
        # Tensors are allocated lazily at first capture (dof count comes from live robot data).
        self._post_swing_root: torch.Tensor | None = None
        self._post_swing_joint_pos: torch.Tensor | None = None
        self._post_swing_joint_vel: torch.Tensor | None = None
        self._post_swing_count = 0
        self._post_swing_ptr = 0
        self._post_swing_teacher_hard_contract: dict | None = None
        ready = getattr(self.cfg, "post_swing_require_ready_at_init", False)
        fail_fast = getattr(self.cfg, "post_swing_fail_fast_first_reset", False)
        require_readback = getattr(self.cfg, "post_swing_first_reset_require_readback", False)
        if any(type(value) is not bool for value in (ready, fail_fast, require_readback)):
            raise ValueError("post-swing teacher gates require exact booleans")
        self._post_swing_require_ready_at_init = ready
        self._post_swing_fail_fast_first_reset = fail_fast
        self._post_swing_first_reset_require_readback = require_readback
        min_count = getattr(self.cfg, "post_swing_first_reset_min_adopted_count", 1)
        min_fraction = getattr(self.cfg, "post_swing_first_reset_min_adopted_fraction", 0.0)
        tolerance = getattr(self.cfg, "post_swing_first_reset_selection_tolerance", 1.0)
        if type(min_count) is not int or min_count <= 0:
            raise ValueError("post_swing_first_reset_min_adopted_count must be a positive integer")
        if (
            type(min_fraction) is not float
            or not math.isfinite(min_fraction)
            or not 0.0 <= min_fraction <= 1.0
            or type(tolerance) is not float
            or not math.isfinite(tolerance)
            or not 0.0 <= tolerance <= 1.0
        ):
            raise ValueError("post-swing first-reset fractions must be finite JSON-style floats in [0,1]")
        self._post_swing_first_reset_min_adopted_count = min_count
        self._post_swing_first_reset_min_adopted_fraction = min_fraction
        self._post_swing_first_reset_selection_tolerance = tolerance
        if (
            require_readback
            or min_count != 1
            or min_fraction != 0.0
            or tolerance != 1.0
        ) and not fail_fast:
            raise ValueError(
                "post-swing first-reset acceptance thresholds require fail_fast_first_reset=true"
            )
        self._post_swing_first_reset_checked = False
        # The capture producer intentionally lives inside MotionCommand.  There is no reusable
        # writer that accepts caller-supplied arrays and no module-global Python "capability".
        # The only state snapshot is taken from live articulation tensors in the natural-wrap
        # branch below.  Its artifact makes the narrower, auditable claim that exact reviewed
        # source owned an O_EXCL namespace and emitted these bytes; it is not a cryptographic
        # proof that an unmodified Python runtime executed a particular callback.
        self._post_swing_capture_output_dir: Path | None = None
        self._post_swing_capture_target_count = 0
        self._post_swing_capture_motion_clips: list[dict] = []
        self._post_swing_capture_joint_names: list[str] = []
        self._post_swing_capture_producer_source_sha256: str | None = None
        self._post_swing_capture_runtime_hard_contract_sha256: str | None = None
        self._post_swing_capture_claim_sha256: str | None = None
        self._post_swing_capture_claim_fd: int | None = None
        self._post_swing_capture_roots: list[np.ndarray] = []
        self._post_swing_capture_joint_pos: list[np.ndarray] = []
        self._post_swing_capture_joint_vel: list[np.ndarray] = []
        self._post_swing_capture_count = 0
        self._post_swing_capture_complete = False
        # Activation accounting is kept outside ``metrics`` because command metrics are
        # instantaneous per-environment values, while these are event counts accumulated over
        # one PPO update.  MotionOnPolicyRunner consumes and resets them exactly once from its
        # existing per-update logger.  Integer device scalars avoid a host sync on every reset.
        self._post_swing_activation_counters = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in (
                "post_swing_replay_buffer_not_ready_reset_count",
                "post_swing_replay_eligible_reset_count",
                "post_swing_replay_random_not_selected_reset_count",
                "post_swing_replay_selected_reset_count",
                "post_swing_replay_started_reset_count",
            )
        }
        self._load_post_swing_teacher_if_configured()
        self._configure_post_swing_capture_if_requested()
        # Reward-mechanism activation accounting.  These counters live on the motion command so
        # every imitation reward term can record into one per-update ledger without touching the
        # simulator or sampling another random number.  The unit of V1 is one environment sample
        # evaluated by the body-linear-velocity imitation term.  The unit of V2 is one
        # (imitation reward term, environment) sample inside the wide strike window; V2 therefore
        # counts every real scaled reward application rather than inferring activation from an
        # aggregate reward value.
        self._reward_activation_counters = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in (
                "v1_velocity_mimic_eligible_sample_count",
                "v1_held_wrist_excluded_sample_count",
                "v2_strike_window_eligible_imitation_sample_count",
                "v2_quarter_scaled_strike_window_imitation_sample_count",
            )
        }
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["in_hold"] = torch.zeros(self.num_envs, device=self.device)
        if bool(getattr(self, "_start_pose_ramp_enabled", False)):
            # 出生偏移必须自陈:只放一个"斜坡开着"的计数器,没人会去读。
            # 这三条记录的是每个 env 这一条命实际被挪了多远/转了多少,
            # 收据出生 + 这三个数 = 物理出生,任何时候都能对得上。
            self.metrics["start_pose_ramp_progress"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["start_pose_ramp_dx_m"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["start_pose_ramp_dy_m"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["start_pose_ramp_dyaw_rad"] = torch.zeros(
                self.num_envs, device=self.device
            )
        if self._event_scheduler is not None:
            self.metrics["event_timing_armed"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_installed"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_unavailable"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_infeasible"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_deadline_due"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_opportunities_consumed"] = torch.zeros(self.num_envs, device=self.device)
        if self.action_ball_continuous_motion_enabled:
            self.metrics["action_ball_continuous_phase"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["action_ball_continuous_reveal_due"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["action_ball_continuous_deadline_due"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[
                "action_ball_continuous_recovery_unavailable"
            ] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[
                "action_ball_continuous_task_commit_missed"
            ] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[
                "action_ball_continuous_motion_release_missed"
            ] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[
                "action_ball_continuous_opportunities_consumed"
            ] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[
                "action_ball_continuous_policy_opportunities_created"
            ] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[
                "action_ball_continuous_infrastructure_censors_consumed"
            ] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"reference_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)

    def _configure_action_ball_continuous_motion_cadence(self) -> None:
        """Allocate the default-off Motion half of the fresh successor clock."""

        profile = _parse_action_ball_continuous_motion_profile(
            getattr(
                self.cfg,
                "action_ball_continuous_motion_cadence",
                None,
            )
        )
        self._action_ball_continuous_motion_profile = profile
        self._action_ball_continuous_ready_authority = None
        self._action_ball_continuous_parent_authority_binding = None
        self._action_ball_continuous_schedule_projection = None
        if profile is None:
            return
        conflicts = []
        if (
            not self.canonical_ready_mode
            and self._action_ball_full_mdp_diagnostic_catalog_table is None
        ):
            conflicts.append("canonical_ready_mode must be true")
        if self.action_ball_single_stroke_timeout_enabled:
            conflicts.append(
                "action_ball_single_stroke_timeout_enabled must be false"
            )
        if bool(self.cfg.wrap_teleport):
            conflicts.append("wrap_teleport must be false")
        if float(self.cfg.clip_switch_prob) != 0.0:
            conflicts.append("clip_switch_prob must be zero")
        if bool(self.cfg.stagger_initial_clock):
            conflicts.append("stagger_initial_clock must be false")
        if self.retiming_active:
            conflicts.append("reference retiming must be disabled")
        if self.planner_revision_enabled:
            conflicts.append("planner revision clock must be disabled")
        if self._event_timing_mode != EVENT_TIMING_MODE_DISABLED:
            conflicts.append("legacy event_timing_mode must be disabled")
        if int(getattr(self.cfg, "rsi_skip_settle_frames", 0)) != 0:
            conflicts.append("rsi_skip_settle_frames must be zero")
        if conflicts:
            raise ValueError(
                "action_ball_continuous_motion_cadence conflicts: "
                + "; ".join(conflicts)
            )

        n = self.num_envs
        long_options = {"dtype": torch.long, "device": self.device}
        bool_options = {"dtype": torch.bool, "device": self.device}
        self._action_ball_continuous_sequence_active = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_episode_step = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_scheduled_ordinal = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_current_reveal_step = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_current_deadline_step = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_next_reveal_step = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_last_closed_ordinal = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_opportunities_consumed = torch.zeros(
            n, **long_options
        )
        self._action_ball_continuous_policy_opportunities_created = (
            torch.zeros(n, **long_options)
        )
        self._action_ball_continuous_infrastructure_censors_consumed = (
            torch.zeros(n, **long_options)
        )
        self._action_ball_continuous_current_policy_opportunity = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_motion_active = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_suffix_complete = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_ready_reference_active = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_ready_at_reveal = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_reveal_due = torch.zeros(
            n, **bool_options
        )
        # These two row-aligned buffers are one-tick Motion facts.  They do
        # not carry task/outcome/ball identity and do not replace ActionEpoch's
        # lifecycle.  The after-command owner joins this edge to the epoch's
        # own current row before any later D05 prepare.
        self._action_ball_continuous_closed_mask = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_close_reason = torch.full(
            (n,), ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE, **long_options
        )
        self._action_ball_continuous_deadline_due = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_recovery_unavailable = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_task_commit_pending = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_task_commit_missed = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_task_committed = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_motion_release_pending = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_motion_release_missed = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_committed_task_refs = [None] * n
        self._action_ball_continuous_commit_owner_nonce = object()
        self._action_ball_continuous_next_commit_token_serial = 0
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_current_projection = None
        # R08 owns a separate, exact five-state chronology.  These buffers are
        # not initialized from the legacy seven-state phase and are not public
        # tensor aliases.  The observation token is a capability into the
        # owner-private record below; the clone-only view is minted only by
        # the exact validator.
        self._action_ball_continuous_observation_owner_nonce = object()
        self._action_ball_continuous_observation_publication_identity = None
        self._action_ball_continuous_observation_token = None
        self._action_ball_continuous_observation_record = None
        self._action_ball_continuous_observation_common_step = None
        self._action_ball_continuous_canonical_phase = torch.full(
            (n,),
            _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                "recover_hidden"
            ],
            **long_options,
        )
        self._action_ball_continuous_canonical_phase_start_tick = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_task_identity = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_cadence_identity = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_action_uid = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_shot_index = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_outcome_identity = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_task_receipt_sha256 = torch.zeros(
            (n, 32), dtype=torch.uint8, device=self.device
        )
        self._action_ball_continuous_canonical_cadence_receipt_sha256 = torch.zeros(
            (n, 32), dtype=torch.uint8, device=self.device
        )
        self._action_ball_continuous_canonical_candidate_identity = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_contact_tick = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_launch_tick = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_chosen_horizon_tick = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_task_close_tick = torch.full(
            (n,), -1, **long_options
        )
        self._action_ball_continuous_canonical_task_valid = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_canonical_playback_started = torch.zeros(
            n, **bool_options
        )
        self._action_ball_continuous_r07_ready_owner = None
        self._action_ball_continuous_r07_ready_validator = None
        self._action_ball_continuous_r07_ready_projection = None
        # Production all-owner Motion leaf state is initialized only when an
        # exact R05 owner is explicitly bound.  Keeping this false preserves
        # the isolated scheduler-baseline tests without allowing those legacy
        # APIs into the constructed production path.
        self._action_ball_continuous_fresh_motion_lane_bound = False
        self._action_ball_continuous_transaction_module = None
        self._action_ball_continuous_transaction_owner = None
        # Production reveal ingress is a separate Device-R05 capability
        # family.  The portable owner above remains diagnostic compatibility
        # only; the two binders are deliberately mutually exclusive.
        self._action_ball_continuous_motion_device_r05_owner = None
        # The lean diagnostic path has one construction-bound epoch owner.
        # It is a direct writer-order coordinator, not a receipt or a second
        # source of task truth.  Binding is deliberately separate from the
        # Device-R05 genesis join so factories cannot silently synthesize an
        # epoch while constructing Motion.
        self._action_ball_full_mdp_motion_epoch_owner = None
        self._action_ball_full_mdp_motion_epoch_fault_latch = None
        self._action_ball_full_mdp_motion_epoch_writable_rows = None
        # Exact question chronology capabilities are Motion-owned and retain
        # the independent Physical horizon capability that supplied the
        # candidate-keyed maximum complete segment.  They are deliberately
        # separate from the cadence-close deadline state above.
        self._action_ball_motion_question_owner_nonce = object()
        self._action_ball_motion_question_records = {}
        self._action_ball_continuous_motion_boundary_module = None
        self._action_ball_continuous_motion_boundary_owner = None
        self._action_ball_continuous_motion_boundary_lane = None
        self._action_ball_continuous_motion_boundary_source_sha256 = None
        self._action_ball_continuous_motion_child_token_authority = None
        self._action_ball_continuous_motion_boundary_fault_schema = None
        self._action_ball_continuous_motion_owner_nonce = None
        self._action_ball_continuous_motion_next_serial = 0
        self._action_ball_continuous_motion_mutation_version = 0
        self._action_ball_continuous_motion_device_mutation_version = None
        self._action_ball_continuous_motion_stage = None
        self._action_ball_continuous_motion_prearmed_install = None
        self._action_ball_continuous_motion_prearmed_accept_swaps = None
        self._action_ball_continuous_motion_prearmed_censor_swaps = None
        self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_accept_refs = None
        self._action_ball_continuous_motion_prearmed_boundary_row = None
        self._action_ball_continuous_motion_armed_install = None
        self._action_ball_continuous_motion_censored_install = None
        self._action_ball_continuous_motion_armed_swaps = None
        self._action_ball_continuous_motion_armed_refs = None
        self._action_ball_continuous_motion_commit_receipt = None
        self._action_ball_continuous_motion_terminal_claim = None
        self._action_ball_continuous_motion_terminal_expectations = None
        self._action_ball_continuous_motion_terminal_token = None
        self._action_ball_continuous_motion_terminal_epoch_committed = False
        # Fresh selected reset is a different capability family from reveal.
        # In particular, none of the reveal child tokens above can be reused
        # to enter or complete this state machine.
        self._action_ball_continuous_motion_selected_reset_authority = None
        self._action_ball_continuous_motion_selected_reset_prepare_validator = None
        self._action_ball_continuous_motion_selected_reset_r05_validator = None
        self._action_ball_continuous_motion_selected_reset_r05_owner = None
        self._action_ball_continuous_motion_selected_reset_authority_api_sha256 = (
            None
        )
        self._action_ball_continuous_motion_selected_reset_diagnostic = False
        self._action_ball_continuous_motion_selected_reset_owner_nonce = None
        self._action_ball_continuous_motion_selected_reset_next_serial = 0
        self._action_ball_continuous_motion_selected_reset_stage = None
        self._action_ball_continuous_motion_selected_reset_prepared_true_reset = None
        self._action_ball_continuous_motion_selected_reset_selected_mask = None
        self._action_ball_continuous_motion_selected_reset_generation_before = None
        self._action_ball_continuous_motion_selected_reset_generation_after = None
        self._action_ball_continuous_motion_selected_reset_generation_overflow_fault = (
            None
        )
        self._action_ball_continuous_motion_selected_reset_prevalidated = None
        self._action_ball_continuous_motion_selected_reset_swaps = None
        self._action_ball_continuous_motion_selected_reset_version_after = None
        self._action_ball_continuous_motion_selected_reset_terminal_token = None
        self._action_ball_continuous_motion_selected_reset_completion_token = None
        self._action_ball_continuous_motion_selected_reset_completion_prepared = None
        self._action_ball_continuous_motion_selected_reset_committed = False
        self._action_ball_continuous_motion_poisoned = False
        self._action_ball_continuous_motion_poison_reason = None
        self._action_ball_continuous_motion_fault_count_device = None
        self._action_ball_continuous_motion_terminal_resolution_total = 0
        self._action_ball_continuous_motion_terminal_resolution_total_device = None
        self._action_ball_continuous_motion_global_drain_active = None
        self._action_ball_continuous_motion_global_drain_sequence = 0
        self._action_ball_continuous_motion_global_drain_last_update = -1
        self._action_ball_continuous_motion_global_drain_last_completed_steps = -1
        self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = -1
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True
        self._action_ball_continuous_motion_global_drain_poisoned = False
        self._action_ball_continuous_motion_global_drain_poison_reason = None
        self._action_ball_continuous_fresh_time_left_receipt = None
        # Published only after the complete Motion update for this manager
        # tick.  A later Command can therefore distinguish the declared
        # Motion->Racket ordering from a swapped or stale call.
        self._action_ball_continuous_published_common_step = None
        self._action_ball_continuous_phase = torch.full(
            (n,),
            _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE["pre_reveal_hidden"],
            **long_options,
        )
        # A selected reset deliberately leaves legacy host task references
        # untouched.  This device bit makes those references unreachable until
        # a later accepted reveal atomically installs the new task identity.
        self._action_ball_continuous_motion_reset_pending = torch.zeros(
            n, **bool_options
        )

    @property
    def action_ball_continuous_motion_enabled(self) -> bool:
        return (
            getattr(
                self,
                "_action_ball_continuous_motion_profile",
                None,
            )
            is not None
        )

    def _action_ball_continuous_public_tensor(
        self, value: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        """Never export a writable live alias from the production leaf."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation=f"{name} read"
            )
            return value.detach().clone()
        return value

    @property
    def action_ball_continuous_motion_phase(self) -> torch.Tensor:
        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        return self._action_ball_continuous_public_tensor(
            self._action_ball_continuous_phase,
            name="phase",
        )

    @property
    def action_ball_continuous_reveal_due(self) -> torch.Tensor:
        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        return self._action_ball_continuous_public_tensor(
            self._action_ball_continuous_reveal_due,
            name="reveal-due",
        )

    @property
    def action_ball_continuous_deadline_due(self) -> torch.Tensor:
        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        return self._action_ball_continuous_public_tensor(
            self._action_ball_continuous_deadline_due,
            name="deadline-due",
        )

    @property
    def action_ball_continuous_recovery_unavailable(self) -> torch.Tensor:
        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        return self._action_ball_continuous_public_tensor(
            self._action_ball_continuous_recovery_unavailable,
            name="recovery-unavailable",
        )

    def bind_action_ball_continuous_ready_authority(
        self, ready: torch.Tensor
    ) -> None:
        """Bind the future recovery conjunction without owning its formula.

        The shared boolean remains owned by the recovery/plant authority.  A
        missing binding is fail-closed (not ready); Motion never invents a
        weighted readiness score from its own imitation errors.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        if (
            not torch.is_tensor(ready)
            or ready.dtype != torch.bool
            or tuple(ready.shape) != (self.num_envs,)
            or ready.device != torch.device(self.device)
        ):
            raise ValueError(
                "continuous ready authority must be one bool tensor on Motion's device"
            )
        current = self._action_ball_continuous_ready_authority
        if current is not None and current is not ready:
            raise RuntimeError(
                "continuous ready authority may be bound exactly once"
            )
        self._action_ball_continuous_ready_authority = ready

    def bind_action_ball_continuous_r07_ready_projection(
        self,
        r07_owner: object,
        *,
        require_owned_ready_projection: object,
    ) -> None:
        """Bind the sole dwell-qualified R07 readiness capability.

        The legacy shared bool above remains diagnostic compatibility.  The
        fresh five-state lifecycle never consumes it: READY_HOLD requires an
        owner-issued projection, exact validator and stable owner identity.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "continuous R07 ready projection requires fresh cadence"
            )
        if (
            r07_owner is None
            or not callable(require_owned_ready_projection)
            or getattr(require_owned_ready_projection, "__self__", None)
            is not r07_owner
        ):
            raise TypeError(
                "continuous R07 readiness requires an owner-bound validator"
            )
        current = self._action_ball_continuous_r07_ready_owner
        if current is not None:
            if (
                current is not r07_owner
                or getattr(
                    self._action_ball_continuous_r07_ready_validator,
                    "__func__",
                    None,
                )
                is not getattr(
                    require_owned_ready_projection, "__func__", None
                )
            ):
                raise RuntimeError(
                    "continuous R07 ready projection may not be rebound"
                )
            return
        self._action_ball_continuous_r07_ready_owner = r07_owner
        self._action_ball_continuous_r07_ready_validator = (
            require_owned_ready_projection
        )

    def install_action_ball_continuous_r07_ready_projection(
        self, projection: object
    ) -> None:
        """Retain one current-tick, owner-validated R07 readiness capability."""

        validator = self._action_ball_continuous_r07_ready_validator
        if validator is None:
            raise RuntimeError(
                "continuous R07 ready projection is not construction-bound"
            )
        try:
            view = validator(projection, owner_kind="motion")
        except Exception as exc:
            raise RuntimeError(
                "continuous R07 readiness projection is not owner-issued"
            ) from exc
        if (
            getattr(view, "owner_kind", None) != "motion"
            or getattr(view, "ready_projection", None) is not projection
            or getattr(view, "ready_identity", None) is None
            or not torch.is_tensor(getattr(view, "ready", None))
            or view.ready.dtype != torch.bool
            or tuple(view.ready.shape) != (self.num_envs,)
            or view.ready.device != torch.device(self.device)
            or not torch.is_tensor(getattr(view, "control_tick", None))
            or view.control_tick.dtype != torch.int64
            or tuple(view.control_tick.shape) != (self.num_envs,)
            or view.control_tick.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "continuous R07 readiness projection shape differs"
            )
        # Retain the opaque capability, never the clone returned to this call.
        # The command update revalidates it at the actual chronology write.
        self._action_ball_continuous_r07_ready_projection = projection

    def bind_action_ball_continuous_parent_authorities(
        self,
        *,
        continuous_contract_authority_sha256,
        recovery_contract_authority_sha256,
        frozen_at_step,
        sequence_origin_step,
        first_reveal_step,
        cadence_steps,
        deadline_offset_steps,
        upcoming_action_slot=None,
        upcoming_action_uid=None,
    ) -> None:
        """Bind the external C01/C02 identities and their timing projection.

        The profile's canonical hash is only an integrity checksum.  The
        target/task owner must retain the C01 contract authority and the
        recovery owner must retain the C02 authority, then jointly bind the
        exact clock values those parents authorized before the first reset.
        This seam intentionally performs no target or ball installation.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        continuous_sha256 = self._action_ball_sha256(
            continuous_contract_authority_sha256,
            name="continuous_contract_authority_sha256",
        )
        recovery_sha256 = self._action_ball_sha256(
            recovery_contract_authority_sha256,
            name="recovery_contract_authority_sha256",
        )
        projection = {
            "frozen_at_step": self._action_ball_plain_int(
                frozen_at_step, name="frozen_at_step"
            ),
            "sequence_origin_step": self._action_ball_plain_int(
                sequence_origin_step, name="sequence_origin_step"
            ),
            "first_reveal_step": self._action_ball_plain_int(
                first_reveal_step, name="first_reveal_step"
            ),
            "cadence_steps": self._action_ball_plain_int(
                cadence_steps, name="cadence_steps", minimum=1
            ),
            "deadline_offset_steps": self._action_ball_plain_int(
                deadline_offset_steps,
                name="deadline_offset_steps",
                minimum=1,
            ),
        }
        if upcoming_action_slot is not None or upcoming_action_uid is not None:
            projection["upcoming_action_slot"] = self._action_ball_plain_int(
                upcoming_action_slot,
                name="upcoming_action_slot",
            )
            projection["upcoming_action_uid"] = self._action_ball_plain_int(
                upcoming_action_uid,
                name="upcoming_action_uid",
                minimum=1,
            )
            action_uids = (
                self._action_ball_continuous_code_owned_action_uids()
            )
            if (
                not isinstance(action_uids, tuple)
                or projection["upcoming_action_slot"] >= len(action_uids)
                or action_uids[projection["upcoming_action_slot"]]
                != projection["upcoming_action_uid"]
            ):
                raise RuntimeError(
                    "external continuous upcoming action identity differs"
                )
        if not (
            projection["sequence_origin_step"]
            <= projection["frozen_at_step"]
            < projection["first_reveal_step"]
        ):
            raise ValueError(
                "external continuous schedule must freeze before first reveal"
            )
        if (
            projection["deadline_offset_steps"]
            >= projection["cadence_steps"]
        ):
            raise ValueError(
                "external continuous deadline offset must be smaller than cadence"
            )
        profile = self._action_ball_continuous_motion_profile
        if (
            continuous_sha256
            != profile["continuous_contract_authority_sha256"]
            or recovery_sha256
            != profile["recovery_contract_authority_sha256"]
        ):
            raise RuntimeError(
                "continuous Motion profile differs from external C01/C02 authority"
            )
        immutable_projection = MappingProxyType(projection)
        binding = (
            profile,
            immutable_projection,
            continuous_sha256,
            recovery_sha256,
        )
        current = self._action_ball_continuous_parent_authority_binding
        if current is not None and current != binding:
            raise RuntimeError(
                "continuous Motion external C01/C02 authority may not drift"
            )
        if current is not None:
            return
        self._action_ball_continuous_schedule_projection = (
            immutable_projection
        )
        self._action_ball_continuous_parent_authority_binding = binding

    def _require_action_ball_continuous_parent_authorities(self) -> None:
        binding = self._action_ball_continuous_parent_authority_binding
        if (
            binding is None
            or binding[0] is not self._action_ball_continuous_motion_profile
            or binding[1]
            is not self._action_ball_continuous_schedule_projection
        ):
            raise RuntimeError(
                "continuous Motion cadence requires external C01/C02 authority binding"
            )

    def _action_ball_continuous_motion_leaf_is_active(self) -> bool:
        return any(
            value is not None
            for value in (
                getattr(self, "_action_ball_continuous_motion_stage", None),
                getattr(
                    self,
                    "_action_ball_continuous_motion_prearmed_install",
                    None,
                ),
                getattr(
                    self,
                    "_action_ball_continuous_motion_armed_install",
                    None,
                ),
                getattr(
                    self,
                    "_action_ball_continuous_motion_censored_install",
                    None,
                ),
            )
        )

    def _action_ball_continuous_motion_selected_reset_is_active(self) -> bool:
        return any(
            value is not None
            for value in (
                getattr(
                    self,
                    "_action_ball_continuous_motion_selected_reset_stage",
                    None,
                ),
                getattr(
                    self,
                    "_action_ball_continuous_motion_selected_reset_prevalidated",
                    None,
                ),
                getattr(
                    self,
                    "_action_ball_continuous_motion_selected_reset_terminal_token",
                    None,
                ),
                getattr(
                    self,
                    "_action_ball_continuous_motion_selected_reset_completion_token",
                    None,
                ),
            )
        )

    def _require_action_ball_continuous_motion_leaf_idle(
        self, *, operation: str
    ) -> None:
        """Protect the production owner while one private leaf is retained."""

        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            return
        if self._action_ball_continuous_motion_poisoned:
            raise RuntimeError(
                f"continuous Motion child owner is poisoned during {operation}"
            )
        if self._action_ball_continuous_motion_leaf_is_active():
            raise RuntimeError(
                f"continuous Motion {operation} is forbidden while its leaf lease is active"
            )
        if self._action_ball_continuous_motion_selected_reset_is_active():
            raise RuntimeError(
                f"continuous Motion {operation} is forbidden while its selected-reset lease is active"
            )
        if getattr(
            self,
            "_action_ball_continuous_motion_global_drain_active",
            None,
        ) is not None:
            raise RuntimeError(
                f"continuous Motion {operation} is forbidden while its global drain lease is active"
            )
        if (
            self._action_ball_continuous_motion_mutation_version
            >= self._ACTION_BALL_INT64_MAX
        ):
            raise RuntimeError(
                "continuous Motion owner mutation version would overflow int64"
            )

    def _increment_action_ball_continuous_motion_mutation_version(
        self,
    ) -> None:
        """Advance the host/device high-water after one complete live mutation."""

        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            return
        next_version = (
            self._action_ball_continuous_motion_mutation_version + 1
        )
        if next_version > self._ACTION_BALL_INT64_MAX:
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion owner mutation version overflowed int64"
            )
        self._action_ball_continuous_motion_device_mutation_version.add_(1)
        self._action_ball_continuous_motion_mutation_version = next_version
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True

    def _legacy_clear_action_ball_continuous_committed_refs(
        self, ids: torch.Tensor
    ) -> None:
        """Legacy-only host-list reset; production never enters this D2H path."""

        for env_id in ids.detach().cpu().tolist():
            self._action_ball_continuous_committed_task_refs[int(env_id)] = None

    def _reset_action_ball_continuous_motion_cadence(
        self, env_ids: torch.Tensor
    ) -> None:
        """Start a fresh sequence clock after a real episode reset only."""

        if not self.action_ball_continuous_motion_enabled:
            return
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="true reset"
            )
            raise RuntimeError(
                "legacy Motion cadence reset is tombstoned for the fresh selected-reset lane"
            )
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="true reset"
        )
        self._require_action_ball_continuous_parent_authorities()
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        schedule = self._action_ball_continuous_schedule_projection
        origin = int(schedule["sequence_origin_step"])
        self._action_ball_continuous_sequence_active[ids] = True
        # The first command update publishes ``sequence_origin_step``.  Reset
        # itself is outside the policy-tick tape.
        self._action_ball_continuous_episode_step[ids] = origin - 1
        self._action_ball_continuous_scheduled_ordinal[ids] = -1
        self._action_ball_continuous_current_reveal_step[ids] = -1
        self._action_ball_continuous_current_deadline_step[ids] = -1
        self._action_ball_continuous_next_reveal_step[ids] = int(
            schedule["first_reveal_step"]
        )
        self._action_ball_continuous_last_closed_ordinal[ids] = -1
        self._action_ball_continuous_opportunities_consumed[ids] = 0
        self._action_ball_continuous_policy_opportunities_created[ids] = 0
        self._action_ball_continuous_infrastructure_censors_consumed[ids] = 0
        self._action_ball_continuous_current_policy_opportunity[ids] = False
        self._action_ball_continuous_motion_active[ids] = False
        self._action_ball_continuous_suffix_complete[ids] = False
        self._action_ball_continuous_ready_reference_active[ids] = True
        self._action_ball_continuous_ready_at_reveal[ids] = False
        self._action_ball_continuous_reveal_due[ids] = False
        self._action_ball_continuous_closed_mask[ids] = False
        self._action_ball_continuous_close_reason[ids] = (
            ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE
        )
        self._action_ball_continuous_deadline_due[ids] = False
        self._action_ball_continuous_recovery_unavailable[ids] = False
        self._action_ball_continuous_task_commit_pending[ids] = False
        self._action_ball_continuous_task_commit_missed[ids] = False
        self._action_ball_continuous_task_committed[ids] = False
        self._action_ball_continuous_motion_release_pending[ids] = False
        self._action_ball_continuous_motion_release_missed[ids] = False
        # Prepared tokens are policy-tick capabilities and never cross a true
        # reset.  Revocation is metadata only; no task/history/simulator state
        # is cleared here.
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_current_projection = None
        self._action_ball_continuous_published_common_step = None
        self._invalidate_action_ball_continuous_observation_publication()
        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._legacy_clear_action_ball_continuous_committed_refs(ids)
        self._action_ball_continuous_phase[ids] = (
            _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                "pre_reveal_hidden"
            ]
        )
        self._action_ball_continuous_canonical_phase[ids] = (
            _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                "recover_hidden"
            ]
        )
        self._action_ball_continuous_canonical_phase_start_tick[ids] = (
            origin - 1
        )
        self._action_ball_continuous_canonical_task_identity[ids] = -1
        self._action_ball_continuous_canonical_cadence_identity[ids] = -1
        self._action_ball_continuous_canonical_action_uid[ids] = -1
        self._action_ball_continuous_canonical_shot_index[ids] = -1
        self._action_ball_continuous_canonical_outcome_identity[ids] = -1
        self._action_ball_continuous_canonical_task_receipt_sha256[ids] = 0
        self._action_ball_continuous_canonical_cadence_receipt_sha256[ids] = 0
        self._action_ball_continuous_canonical_candidate_identity[ids] = -1
        self._action_ball_continuous_canonical_contact_tick[ids] = -1
        self._action_ball_continuous_canonical_launch_tick[ids] = -1
        self._action_ball_continuous_canonical_chosen_horizon_tick[ids] = -1
        self._action_ball_continuous_canonical_task_close_tick[ids] = -1
        self._action_ball_continuous_canonical_task_valid[ids] = False
        self._action_ball_continuous_canonical_playback_started[ids] = False
        self._invalidate_action_ball_continuous_observation_publication()
        self._hold_action_ball_continuous_ready_reference()
        self._increment_action_ball_continuous_motion_mutation_version()

    def _hold_action_ball_continuous_ready_reference(
        self, writable_rows: torch.Tensor | None = None
    ) -> None:
        """Publish completed-action frame 0 with every reference velocity zero.

        Only command/reference tensors change.  This method deliberately has
        no simulator, action-manager, history, reset or termination writer.
        """

        if not self.action_ball_continuous_motion_enabled:
            return
        ready = self._action_ball_continuous_ready_reference_active
        if writable_rows is not None:
            ready = ready & writable_rows
        ready_steps = self.motion.seg_start[self.clip_id]
        self.time_steps.copy_(torch.where(ready, ready_steps, self.time_steps))
        self.time_steps_f.copy_(
            torch.where(
                ready,
                ready_steps.to(dtype=self.time_steps_f.dtype),
                self.time_steps_f,
            )
        )
        self.speed_scale.copy_(
            torch.where(ready, torch.zeros_like(self.speed_scale), self.speed_scale)
        )
        self.hold_counter.copy_(
            torch.where(ready, torch.ones_like(self.hold_counter), self.hold_counter)
        )
        if "in_hold" in self.metrics:
            self.metrics["in_hold"] = torch.where(
                ready,
                torch.ones_like(self.metrics["in_hold"]),
                self.metrics["in_hold"],
            )

    def _action_ball_continuous_event_rows(
        self,
        env_ids,
        scheduled_ordinals,
        *,
        operation: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        self._require_action_ball_continuous_parent_authorities()
        raw_ids = torch.as_tensor(env_ids, device=self.device)
        raw_ordinals = torch.as_tensor(scheduled_ordinals, device=self.device)
        for name, value in (("env_ids", raw_ids), ("scheduled_ordinals", raw_ordinals)):
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"continuous Motion {name} must use an integer dtype")
        ids = raw_ids.to(dtype=torch.long).reshape(-1)
        ordinals = raw_ordinals.to(dtype=torch.long).reshape(-1)
        if len(ids) == 0 or len(ids) != len(ordinals):
            raise ValueError(
                f"continuous Motion {operation} requires equal non-empty rows"
            )
        if (
            len(torch.unique(ids)) != len(ids)
            or bool((ids < 0).any())
            or bool((ids >= self.num_envs).any())
        ):
            raise ValueError(
                f"continuous Motion {operation} env ids are invalid"
            )
        return ids, ordinals

    def _require_action_ball_continuous_current_publication(
        self,
        *,
        operation: str,
    ) -> int:
        """Reject a consumer running before Motion in this manager tick."""

        common_step = getattr(self._env, "common_step_counter", None)
        if (
            type(common_step) is not int
            or common_step < 0
            or self._action_ball_continuous_published_common_step
            != common_step
        ):
            raise RuntimeError(
                f"continuous Motion {operation} is stale or Command order is swapped"
            )
        return common_step

    def _action_ball_continuous_canonical_ready(self) -> torch.Tensor:
        """Validate and clone the current dwell-qualified R07 ready lane."""

        projection = self._action_ball_continuous_r07_ready_projection
        validator = self._action_ball_continuous_r07_ready_validator
        if projection is None or validator is None:
            # Production stays HOLD until R07 exposes its registry-backed
            # child projection; a caller bool is deliberately not a fallback.
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        try:
            view = validator(projection, owner_kind="motion")
        except Exception as exc:
            raise RuntimeError(
                "continuous R07 readiness projection became stale"
            ) from exc
        ready = getattr(view, "ready", None)
        control_tick = getattr(view, "control_tick", None)
        if (
            getattr(view, "ready_projection", None) is not projection
            or getattr(view, "owner_kind", None) != "motion"
            or not torch.is_tensor(ready)
            or ready.dtype != torch.bool
            or tuple(ready.shape) != (self.num_envs,)
            or ready.device != torch.device(self.device)
            or not torch.is_tensor(control_tick)
            or control_tick.dtype != torch.int64
            or tuple(control_tick.shape) != (self.num_envs,)
            or control_tick.device != torch.device(self.device)
            or not torch.equal(
                control_tick,
                self._action_ball_continuous_episode_step,
            )
        ):
            raise RuntimeError(
                "continuous R07 readiness chronology differs from Motion"
            )
        return ready.clone()

    def _invalidate_action_ball_continuous_observation_publication(self) -> None:
        self._action_ball_continuous_observation_publication_identity = None
        self._action_ball_continuous_observation_token = None
        self._action_ball_continuous_observation_record = None
        self._action_ball_continuous_observation_common_step = None

    def _transition_action_ball_continuous_canonical_phase(
        self,
        next_phase: torch.Tensor,
    ) -> None:
        if (
            not torch.is_tensor(next_phase)
            or next_phase.dtype != torch.int64
            or tuple(next_phase.shape) != (self.num_envs,)
            or next_phase.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "continuous canonical phase after-image shape differs"
            )
        changed = next_phase != self._action_ball_continuous_canonical_phase
        self._action_ball_continuous_canonical_phase_start_tick.copy_(
            torch.where(
                changed,
                self._action_ball_continuous_episode_step,
                self._action_ball_continuous_canonical_phase_start_tick,
            )
        )
        self._action_ball_continuous_canonical_phase.copy_(next_phase)

    def _advance_action_ball_continuous_canonical_lifecycle(
        self,
        *,
        motion_active_before: torch.Tensor,
        suffix_due: torch.Tensor,
        closed_without_playback: torch.Tensor,
        writable_rows: torch.Tensor | None = None,
    ) -> None:
        """Advance the independent exact five-state lifecycle once.

        This writer consumes task/teacher chronology only.  Physical ball
        launch is intentionally absent: late launch affects ball_state_valid
        at the downstream Physical+R06 join, never Motion phase.
        """

        # Portable/legacy cadence has no authority to write R08.  The exact
        # Device-R05 construction binding is the minimum owner fact even for
        # recovery/ready transitions.
        if self._action_ball_continuous_motion_device_r05_owner is None:
            return
        if writable_rows is None:
            writable_rows = torch.ones_like(
                self._action_ball_continuous_sequence_active
            )
        current = self._action_ball_continuous_canonical_phase
        next_phase = current.clone()
        active = (
            self._action_ball_continuous_sequence_active & writable_rows
        )
        task_valid = self._action_ball_continuous_canonical_task_valid
        # Fresh Full-MDP lifecycle is owned by observable task/teacher events.
        # R07 remains recovery telemetry/reward; it cannot become a training
        # liveness dependency or an actor-phase authority.  Legacy lanes keep
        # their existing R07-qualified ready transition.
        ready = (
            torch.ones_like(active)
            if self._action_ball_continuous_fresh_motion_lane_bound
            else self._action_ball_continuous_canonical_ready()
        )
        timing = self._action_ball_task_timing_active & task_valid
        teacher_left_frame0 = self.time_steps.gt(
            self.motion.seg_start[self.clip_id]
        )
        teacher_started = (
            timing
            & teacher_left_frame0
            & writable_rows
            & (
                self._action_ball_task_age_s + 1.0e-12
                >= self._action_ball_pre_swing_wait_s
            )
        )
        playback_after = self._action_ball_continuous_canonical_playback_started
        epoch_owner = self._action_ball_full_mdp_motion_epoch_owner
        if epoch_owner is not None:
            # The scalar compatibility ``record.epoch`` deliberately remains
            # -1 in the row-wise lane.  Epoch pulls Motion's exact full-key
            # mask below; an IDLE, foreign or stale row produces an empty
            # event without becoming caller-authored selection authority.
            published_started = (
                epoch_owner.publish_motion_playback_started(owner=self)[:, 0]
                & writable_rows
            )
            teacher_started &= published_started
            playback_after = playback_after | published_started
        else:
            playback_after = playback_after | teacher_started
        lifecycle_close = self._action_ball_continuous_current_deadline_step
        if self._action_ball_continuous_fresh_motion_lane_bound:
            lifecycle_close = (
                self._action_ball_continuous_canonical_task_close_tick
            )
        swing = (
            teacher_started
            & (
                (current == ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE)
                | (current == ACTION_BALL_CONTINUOUS_CANONICAL_SWING)
            )
            & (self._action_ball_continuous_episode_step <= lifecycle_close)
        )
        follow = (
            task_valid
            & playback_after
            & (
                (current == ACTION_BALL_CONTINUOUS_CANONICAL_SWING)
                | (current == ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH)
                | (
                    (current == ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE)
                    & teacher_started
                )
            )
            & (self._action_ball_continuous_episode_step >= lifecycle_close)
            & motion_active_before
            & ~suffix_due
        )

        next_phase = torch.where(
            swing,
            torch.full_like(
                next_phase,
                _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE["swing"],
            ),
            next_phase,
        )
        next_phase = torch.where(
            follow,
            torch.full_like(
                next_phase,
                _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                    "follow_through"
                ],
            ),
            next_phase,
        )
        hidden = (
            active
            & (
                closed_without_playback
                | suffix_due
                | ~task_valid
            )
        )
        task_valid_after = task_valid & ~hidden
        playback_after &= ~hidden
        task_identity_after = (
            torch.where(
                hidden,
                torch.full_like(
                    self._action_ball_continuous_canonical_task_identity, -1
                ),
                self._action_ball_continuous_canonical_task_identity,
            )
        )
        cadence_identity_after = (
            torch.where(
                hidden,
                torch.full_like(
                    self._action_ball_continuous_canonical_cadence_identity, -1
                ),
                self._action_ball_continuous_canonical_cadence_identity,
            )
        )
        scalar_identity_after = tuple(
            torch.where(hidden, torch.full_like(destination, -1), destination)
            for destination in (
                self._action_ball_continuous_canonical_action_uid,
                self._action_ball_continuous_canonical_shot_index,
                self._action_ball_continuous_canonical_outcome_identity,
                self._action_ball_continuous_canonical_candidate_identity,
                self._action_ball_continuous_canonical_contact_tick,
                self._action_ball_continuous_canonical_launch_tick,
                self._action_ball_continuous_canonical_chosen_horizon_tick,
                self._action_ball_continuous_canonical_task_close_tick,
            )
        )
        sha_after = tuple(
            torch.where(hidden.unsqueeze(1), torch.zeros_like(destination), destination)
            for destination in (
                self._action_ball_continuous_canonical_task_receipt_sha256,
                self._action_ball_continuous_canonical_cadence_receipt_sha256,
            )
        )
        next_phase = torch.where(
            hidden,
            torch.full_like(
                next_phase,
                _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                    "recover_hidden"
                ],
            ),
            next_phase,
        )
        hidden_now = active & ~task_valid_after
        next_phase = torch.where(
            hidden_now & ready,
            torch.full_like(
                next_phase,
                _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE["ready_hold"],
            ),
            next_phase,
        )
        next_phase = torch.where(
            hidden_now & ~ready,
            torch.full_like(
                next_phase,
                _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                    "recover_hidden"
                ],
            ),
            next_phase,
        )
        # No fallible owner operation may follow this application block.
        self._action_ball_continuous_canonical_task_valid.copy_(task_valid_after)
        self._action_ball_continuous_canonical_playback_started.copy_(playback_after)
        self._action_ball_continuous_canonical_task_identity.copy_(task_identity_after)
        self._action_ball_continuous_canonical_cadence_identity.copy_(cadence_identity_after)
        for destination, after in zip(
            (
                self._action_ball_continuous_canonical_action_uid,
                self._action_ball_continuous_canonical_shot_index,
                self._action_ball_continuous_canonical_outcome_identity,
                self._action_ball_continuous_canonical_candidate_identity,
                self._action_ball_continuous_canonical_contact_tick,
                self._action_ball_continuous_canonical_launch_tick,
                self._action_ball_continuous_canonical_chosen_horizon_tick,
                self._action_ball_continuous_canonical_task_close_tick,
            ),
            scalar_identity_after,
        ):
            destination.copy_(after)
        for destination, after in zip(
            (
                self._action_ball_continuous_canonical_task_receipt_sha256,
                self._action_ball_continuous_canonical_cadence_receipt_sha256,
            ),
            sha_after,
        ):
            destination.copy_(after)
        self._transition_action_ball_continuous_canonical_phase(next_phase)

    def _publish_action_ball_continuous_observation(self) -> None:
        """Seal one owner-private post-update snapshot; no host observation."""

        if self._action_ball_continuous_motion_device_r05_owner is None:
            self._invalidate_action_ball_continuous_observation_publication()
            return
        common_step = getattr(self._env, "common_step_counter", None)
        if type(common_step) is not int or common_step < 0:
            raise RuntimeError(
                "continuous Motion observation requires manager common step"
            )
        publication_identity = object()
        token = ActionBallContinuousMotionObservationToken()

        def snapshot(value: torch.Tensor) -> torch.Tensor:
            return value.detach().clone()

        time_to_contact_remaining = (
            self._action_ball_time_to_contact_s
            - self._action_ball_task_age_s
        )
        time_to_teacher_start_remaining = torch.clamp(
            self._action_ball_pre_swing_wait_s
            - self._action_ball_task_age_s,
            min=0.0,
        )
        time_to_next_reveal = (
            self._action_ball_continuous_next_reveal_step
            - self._action_ball_continuous_episode_step
        ).to(dtype=self._action_ball_task_age_s.dtype) * float(
            self._env.step_dt
        )
        if self._action_ball_continuous_fresh_motion_lane_bound:
            schedule_exhausted = (
                self._action_ball_continuous_scheduled_ordinal
                >= _ACTION_BALL_FULL_MDP_FRESH_REFERENCE_DUE_COUNT - 1
            )
            time_to_next_reveal = torch.where(
                schedule_exhausted,
                torch.full_like(
                    time_to_next_reveal,
                    _ACTION_BALL_FULL_MDP_FRESH_SCHEDULE_EXHAUSTED_TIME_S,
                ),
                time_to_next_reveal,
            )
        record = ActionBallContinuousMotionObservationView(
            motion_owner=self,
            publication_identity=publication_identity,
            common_step=common_step,
            control_tick=snapshot(self._action_ball_continuous_episode_step),
            phase=snapshot(self._action_ball_continuous_canonical_phase),
            reset_generation=snapshot(self._action_ball_reset_generation),
            swing_generation=snapshot(self._action_ball_swing_generation),
            action_uid=snapshot(
                self._action_ball_continuous_canonical_action_uid
            ),
            task_identity=snapshot(
                self._action_ball_continuous_canonical_task_identity
            ),
            task_valid=snapshot(
                self._action_ball_continuous_canonical_task_valid
            ),
            # These three chronology values were allocated above for this
            # publication; detach is sufficient because they cannot alias an
            # owner buffer.
            time_to_contact_remaining_s=time_to_contact_remaining.detach(),
            time_to_teacher_start_remaining_s=(
                time_to_teacher_start_remaining.detach()
            ),
            time_to_next_reveal_s=time_to_next_reveal.detach(),
        )
        self._action_ball_continuous_observation_publication_identity = (
            publication_identity
        )
        self._action_ball_continuous_observation_token = token
        self._action_ball_continuous_observation_record = record
        self._action_ball_continuous_observation_common_step = common_step

    def action_ball_continuous_motion_observation_projection(
        self,
    ) -> ActionBallContinuousMotionObservationToken:
        """Return the already-published opaque R08 token without side effects."""

        self._require_action_ball_continuous_motion_leaf_idle(
            operation="observation projection"
        )
        common_step = self._require_action_ball_continuous_current_publication(
            operation="observation projection"
        )
        token = self._action_ball_continuous_observation_token
        record = self._action_ball_continuous_observation_record
        if (
            type(token) is not ActionBallContinuousMotionObservationToken
            or type(record) is not ActionBallContinuousMotionObservationView
            or record.motion_owner is not self
            or record.publication_identity
            is not self._action_ball_continuous_observation_publication_identity
            or record.common_step != common_step
            or self._action_ball_continuous_observation_common_step
            != common_step
        ):
            raise RuntimeError(
                "continuous Motion observation was not published this tick"
            )
        return token

    def issue_current_r05_cadence_if_due(
        self,
    ) -> ActionBallContinuousMotionObservationToken | None:
        """Return this tick's Motion token only on the frozen reveal cadence.

        ``None`` is the ordinary non-due result.  The decision uses only the
        manager's host common-step chronology and the construction-frozen
        cadence integers; it never observes a device bool or accepts a caller
        verdict.  The returned object is the existing Motion-owned opaque
        observation token, not a new receipt.
        """

        self._require_action_ball_continuous_motion_leaf_idle(
            operation="conditional R05 cadence issue"
        )
        common_step = self._require_action_ball_continuous_current_publication(
            operation="conditional R05 cadence issue"
        )
        schedule = self._action_ball_continuous_schedule_projection
        if not isinstance(schedule, MappingProxyType):
            raise RuntimeError(
                "conditional R05 cadence issue lacks the frozen schedule"
            )
        try:
            origin = int(schedule["sequence_origin_step"])
            first = int(schedule["first_reveal_step"])
            cadence = int(schedule["cadence_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "conditional R05 cadence schedule fields differ"
            ) from exc
        if origin < 0 or first < origin or cadence < 1:
            raise RuntimeError(
                "conditional R05 cadence schedule values differ"
            )
        control_tick = origin + common_step
        if control_tick < first or (control_tick - first) % cadence:
            return None
        if (
            self._action_ball_continuous_fresh_motion_lane_bound
            and control_tick
            > first
            + cadence
            * (_ACTION_BALL_FULL_MDP_FRESH_REFERENCE_DUE_COUNT - 1)
        ):
            return None
        return self.action_ball_continuous_motion_observation_projection()

    def require_owned_action_ball_continuous_motion_observation(
        self,
        token: ActionBallContinuousMotionObservationToken,
    ) -> ActionBallContinuousMotionObservationView:
        """Validate one current opaque token and return an isolated view."""

        self._require_action_ball_continuous_motion_leaf_idle(
            operation="observation validation"
        )
        common_step = self._require_action_ball_continuous_current_publication(
            operation="observation validation"
        )
        record = self._action_ball_continuous_observation_record
        if (
            type(token) is not ActionBallContinuousMotionObservationToken
            or token is not self._action_ball_continuous_observation_token
            or type(record) is not ActionBallContinuousMotionObservationView
            or record.motion_owner is not self
            or record.publication_identity
            is not self._action_ball_continuous_observation_publication_identity
            or record.common_step != common_step
            or self._action_ball_continuous_observation_common_step
            != common_step
        ):
            raise RuntimeError(
                "continuous Motion observation token is forged or stale"
            )
        return ActionBallContinuousMotionObservationView(
            motion_owner=self,
            publication_identity=record.publication_identity,
            common_step=record.common_step,
            control_tick=record.control_tick.clone(),
            phase=record.phase.clone(),
            reset_generation=record.reset_generation.clone(),
            swing_generation=record.swing_generation.clone(),
            action_uid=record.action_uid.clone(),
            task_identity=record.task_identity.clone(),
            task_valid=record.task_valid.clone(),
            time_to_contact_remaining_s=(
                record.time_to_contact_remaining_s.clone()
            ),
            time_to_teacher_start_remaining_s=(
                record.time_to_teacher_start_remaining_s.clone()
            ),
            time_to_next_reveal_s=record.time_to_next_reveal_s.clone(),
        )

    def _action_ball_continuous_motion_checkpoint_payload(self) -> dict:
        """Materialize the fresh Motion owner's complete mid-task state.

        This is intentionally a child payload, not the legacy Motion resume
        contract that expects an immediate full reset.  Every tensor that can
        change the next cadence, teacher reference or R08 projection lives in
        this one independently rooted schema.
        """

        self._require_action_ball_continuous_canonical_checkpoint_complete()
        acknowledged_mutation = (
            self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version
        )
        if (
            type(acknowledged_mutation) is not int
            or acknowledged_mutation
            != self._action_ball_continuous_motion_mutation_version
            or self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack
        ):
            raise RuntimeError(
                "continuous Motion checkpoint lacks the exact globally ACKed mutation frontier"
            )
        tensors = {
            field_name: self._exact_resume_cpu_tensor(getattr(self, attr_name))
            for field_name, attr_name, _nonnegative
            in _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
        }
        pending_work = bool(
            tensors["reset_ready_reference_pending"].any()
        )
        pending_count = getattr(
            self, "_action_ball_safe_ready_pending_count", None
        )
        if (
            type(pending_count) is not int
            or not 0 <= pending_count <= self.num_envs
            or (pending_count > 0) != pending_work
        ):
            raise RuntimeError(
                "continuous Motion reset-ready pending cache differs from its mask"
            )
        tensor_payload = {
            name: value.tolist() for name, value in tensors.items()
        }
        payload = {
            "checkpoint_kind": (
                _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_KIND
            ),
            "schema_version": (
                _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_SCHEMA_VERSION
            ),
            "phase": "IDLE",
            "owner_mutation_version": (
                self._action_ball_continuous_motion_mutation_version
            ),
            "device_owner_mutation_version": [
                self._action_ball_continuous_motion_mutation_version
            ],
            "next_serial": self._action_ball_continuous_motion_next_serial,
            "selected_reset_next_serial": (
                self._action_ball_continuous_motion_selected_reset_next_serial
            ),
            "terminal_resolution_total": (
                self._action_ball_continuous_motion_terminal_resolution_total
            ),
            "fault_count_device": (
                self._action_ball_continuous_motion_fault_count_device.detach()
                .cpu()
                .tolist()
            ),
            "global_drain_sequence": (
                self._action_ball_continuous_motion_global_drain_sequence
            ),
            "global_drain_last_update": (
                self._action_ball_continuous_motion_global_drain_last_update
            ),
            "global_drain_last_completed_steps": (
                self._action_ball_continuous_motion_global_drain_last_completed_steps
            ),
            "global_drain_last_acknowledged_mutation_version": (
                acknowledged_mutation
            ),
            "checkpoint_requires_global_drain_ack": False,
            "published_common_step": (
                # A publication frontier without its opaque capability would
                # let an exact-resume consumer confuse restored bytes with a
                # post-update publication.  Both frontiers are revoked.
                None
            ),
            "observation_common_step": (
                # The getter token is an owner-private capability, not
                # serializable authority.  Restore preserves the publication
                # frontier but requires a fresh post-update publication.
                None
            ),
            "tensors": tensor_payload,
            "poisoned": False,
        }
        return {
            **payload,
            "device_owner_mutation_version": self._exact_resume_cpu_tensor(
                self._action_ball_continuous_motion_device_mutation_version
            ),
            "terminal_resolution_total_device": self._exact_resume_cpu_tensor(
                self._action_ball_continuous_motion_terminal_resolution_total_device
            ),
            "fault_count_device": self._exact_resume_cpu_tensor(
                self._action_ball_continuous_motion_fault_count_device
            ),
            "tensors": tensors,
            "canonical_sha256": hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        }

    def _require_action_ball_continuous_canonical_checkpoint_complete(
        self,
    ) -> None:
        """Fail closed when an active R08 row lacks exact continuation facts."""

        phase = self._action_ball_continuous_canonical_phase
        active = phase <= ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH
        if not bool(torch.any(active)):
            return
        action_slot = self.clip_id
        start = self.motion.seg_start[action_slot]
        playback = self._action_ball_continuous_canonical_playback_started
        receipt_identity_complete = torch.ones_like(active)
        if not self._action_ball_continuous_fresh_motion_lane_bound:
            receipt_identity_complete = torch.any(
                self._action_ball_continuous_canonical_task_receipt_sha256 != 0,
                dim=1,
            ) & torch.any(
                self._action_ball_continuous_canonical_cadence_receipt_sha256 != 0,
                dim=1,
            )
        task_close_complete = torch.ones_like(active)
        if self._action_ball_continuous_fresh_motion_lane_bound:
            task_close_complete = (
                self._action_ball_continuous_canonical_task_close_tick
                >= self._action_ball_continuous_current_reveal_step
            ) & (
                self._action_ball_continuous_canonical_task_close_tick
                < self._action_ball_continuous_next_reveal_step
            )
        complete = (
            self._action_ball_continuous_canonical_task_valid
            & self._action_ball_task_timing_active
            & (self._action_ball_continuous_canonical_task_identity > 0)
            & (self._action_ball_continuous_canonical_cadence_identity > 0)
            & (self._action_ball_continuous_canonical_action_uid > 0)
            & (self._action_ball_continuous_canonical_shot_index > 0)
            & (self._action_ball_continuous_canonical_outcome_identity > 0)
            & receipt_identity_complete
            & (self._action_ball_continuous_canonical_candidate_identity > 0)
            & (self._action_ball_continuous_canonical_launch_tick >= 0)
            & (
                self._action_ball_continuous_canonical_contact_tick
                > self._action_ball_continuous_canonical_launch_tick
            )
            & (
                self._action_ball_continuous_canonical_chosen_horizon_tick
                == self._action_ball_continuous_canonical_contact_tick
                - self._action_ball_continuous_canonical_launch_tick
            )
            & task_close_complete
            & torch.isfinite(self._action_ball_task_pending_elapsed_s)
            & (self._action_ball_task_pending_elapsed_s >= 0)
            & torch.isfinite(self._action_ball_task_age_s)
            & (self._action_ball_task_age_s >= 0)
            & torch.isfinite(self._action_ball_time_to_contact_s)
            & (self._action_ball_time_to_contact_s > 0)
            & torch.isfinite(self._action_ball_teacher_rate)
            & (self._action_ball_teacher_rate > 0)
            & torch.isfinite(self._action_ball_scaled_t_hit_s)
            & (self._action_ball_scaled_t_hit_s > 0)
            & torch.isfinite(self._action_ball_scaled_t_cycle_s)
            & (self._action_ball_scaled_t_cycle_s > 0)
            & torch.isfinite(self._action_ball_pre_swing_wait_s)
            & (self._action_ball_pre_swing_wait_s >= 0)
            & (self._action_ball_continuous_canonical_phase_start_tick >= 0)
            & (
                self._action_ball_continuous_canonical_phase_start_tick
                <= self._action_ball_continuous_episode_step
            )
            & torch.where(
                phase == ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE,
                ~playback,
                playback & (self.time_steps > start),
            )
        )
        if bool(torch.any(active & ~complete)):
            raise RuntimeError(
                "Motion active canonical checkpoint lacks exact mid-task identity/timing"
            )

    def _prepare_action_ball_continuous_motion_checkpoint(
        self, leaf: object
    ) -> dict:
        """Purely parse one fresh mid-task Motion leaf before any write."""

        expected = {
            "checkpoint_kind",
            "schema_version",
            "phase",
            "owner_mutation_version",
            "device_owner_mutation_version",
            "next_serial",
            "selected_reset_next_serial",
            "terminal_resolution_total",
            "terminal_resolution_total_device",
            "fault_count_device",
            "global_drain_sequence",
            "global_drain_last_update",
            "global_drain_last_completed_steps",
            "global_drain_last_acknowledged_mutation_version",
            "checkpoint_requires_global_drain_ack",
            "published_common_step",
            "observation_common_step",
            "tensors",
            "poisoned",
            "canonical_sha256",
        }
        if type(leaf) is not dict or set(leaf) != expected:
            raise ValueError(
                "Motion fresh checkpoint fields differ from schema"
            )
        exact_nonnegative_scalars = (
            "owner_mutation_version",
            "next_serial",
            "selected_reset_next_serial",
            "terminal_resolution_total",
            "global_drain_sequence",
        )
        if (
            leaf["checkpoint_kind"]
            != _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_KIND
            or leaf["schema_version"]
            != _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_SCHEMA_VERSION
            or leaf["phase"] != "IDLE"
            or leaf["poisoned"] is not False
            or any(
                type(leaf[name]) is not int or leaf[name] < 0
                for name in exact_nonnegative_scalars
            )
            or any(
                leaf[name] > self._ACTION_BALL_INT64_MAX
                for name in (
                    "owner_mutation_version",
                    "terminal_resolution_total",
                )
            )
            or type(leaf["global_drain_last_update"]) is not int
            or leaf["global_drain_last_update"] < -1
            or type(leaf["global_drain_last_completed_steps"]) is not int
            or leaf["global_drain_last_completed_steps"] < -1
            or type(leaf["global_drain_last_acknowledged_mutation_version"])
            is not int
            or leaf["global_drain_last_acknowledged_mutation_version"]
            != leaf["owner_mutation_version"]
            or leaf["checkpoint_requires_global_drain_ack"] is not False
            or (
                (leaf["global_drain_sequence"] == 0)
                != (leaf["global_drain_last_update"] == -1)
            )
            or (
                (leaf["global_drain_sequence"] == 0)
                != (leaf["global_drain_last_completed_steps"] == -1)
            )
            or any(
                value is not None
                and (type(value) is not int or value < 0)
                for value in (
                    leaf["published_common_step"],
                    leaf["observation_common_step"],
                )
            )
            or leaf["published_common_step"] is not None
            or (
                leaf["observation_common_step"] is not None
            )
            or type(leaf["canonical_sha256"]) is not str
            or len(leaf["canonical_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in leaf["canonical_sha256"]
            )
        ):
            raise ValueError("Motion fresh checkpoint metadata is invalid")

        device_version = self._validate_exact_resume_tensor(
            leaf["device_owner_mutation_version"],
            name="action_ball.continuous_motion.device_version",
            shape=(1,),
            dtype=torch.int64,
            nonnegative=True,
        )
        terminal_total_device = self._validate_exact_resume_tensor(
            leaf["terminal_resolution_total_device"],
            name="action_ball.continuous_motion.terminal_total",
            shape=(1,),
            dtype=torch.int64,
            nonnegative=True,
        )
        fault_count_device = self._validate_exact_resume_tensor(
            leaf["fault_count_device"],
            name="action_ball.continuous_motion.fault_count",
            shape=(1,),
            dtype=torch.int64,
            nonnegative=True,
        )
        if (
            device_version.tolist()
            != [leaf["owner_mutation_version"]]
            or terminal_total_device.tolist()
            != [leaf["terminal_resolution_total"]]
        ):
            raise ValueError(
                "Motion fresh checkpoint scalar/device frontier differs"
            )

        raw_tensors = leaf["tensors"]
        expected_tensor_fields = {
            field_name
            for field_name, _attr_name, _nonnegative
            in _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
        }
        if (
            type(raw_tensors) is not dict
            or set(raw_tensors) != expected_tensor_fields
        ):
            raise ValueError(
                "Motion fresh checkpoint tensor fields differ"
            )
        tensors = {}
        for field_name, attr_name, nonnegative in (
            _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
        ):
            live = getattr(self, attr_name)
            tensors[field_name] = self._validate_exact_resume_tensor(
                raw_tensors[field_name],
                name=f"action_ball.continuous_motion.{field_name}",
                shape=tuple(live.shape),
                dtype=live.dtype,
                nonnegative=nonnegative,
            )

        canonical_phase = tensors["canonical_phase"]
        legacy_phase = tensors["legacy_phase"]
        task_valid = tensors["task_valid"]
        timing_active = tensors["timing_active"]
        playback_started = tensors["playback_started"]
        sequence_active = tensors["sequence_active"]
        control_tick = tensors["control_tick"]
        phase_start_tick = tensors["canonical_phase_start_tick"]
        scheduled_ordinal = tensors["scheduled_ordinal"]
        reveal_tick = tensors["reveal_tick"]
        deadline_tick = tensors["deadline_tick"]
        next_reveal_tick = tensors["next_reveal_tick"]
        last_closed_ordinal = tensors["last_closed_ordinal"]
        opportunities_consumed = tensors["opportunities_consumed"]
        active_canonical = canonical_phase <= ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH
        expected_last_closed = torch.where(
            control_tick >= deadline_tick,
            scheduled_ordinal,
            scheduled_ordinal - 1,
        )
        expected_opportunities_consumed = scheduled_ordinal + (
            control_tick >= deadline_tick
        ).to(dtype=scheduled_ordinal.dtype)
        contact_clock_matches = torch.isclose(
            (
                tensors["contact_tick"] - control_tick
            ).to(dtype=tensors["time_to_contact_s"].dtype)
            * float(self._env.step_dt),
            tensors["time_to_contact_s"] - tensors["task_age_s"],
            rtol=0.0,
            atol=float(self._env.step_dt) * 0.5 + 1.0e-6,
        )
        action_uid_matches_slot = torch.zeros_like(active_canonical)
        try:
            code_owned_action_uids = (
                self._action_ball_continuous_code_owned_action_uids()
            )
        except RuntimeError:
            code_owned_action_uids = ()
        if code_owned_action_uids:
            action_uids = torch.as_tensor(
                code_owned_action_uids,
                dtype=tensors["action_uid"].dtype,
            )
            valid_slot = tensors["action_slot"] < len(action_uids)
            action_uid_matches_slot = valid_slot & (
                tensors["action_uid"]
                == action_uids[
                    torch.clamp(
                        tensors["action_slot"],
                        min=0,
                        max=max(len(action_uids) - 1, 0),
                    )
                ]
            )
        receipt_identity_complete = torch.ones_like(active_canonical)
        if not self._action_ball_continuous_fresh_motion_lane_bound:
            receipt_identity_complete = torch.any(
                tensors["task_receipt_sha256"] != 0, dim=1
            ) & torch.any(
                tensors["cadence_receipt_sha256"] != 0, dim=1
            )
        task_close_complete = torch.ones_like(active_canonical)
        if self._action_ball_continuous_fresh_motion_lane_bound:
            task_close_complete = (
                tensors["task_close_tick"] >= tensors["reveal_tick"]
            ) & (
                tensors["task_close_tick"] < tensors["next_reveal_tick"]
            )
        complete_identity = (
            (tensors["task_identity"] > 0)
            & (tensors["cadence_identity"] > 0)
            & (tensors["action_uid"] > 0)
            & receipt_identity_complete
            & (tensors["candidate_identity"] > 0)
            & action_uid_matches_slot
            & contact_clock_matches
            & (tensors["launch_tick"] >= 0)
            & (tensors["contact_tick"] > tensors["launch_tick"])
            & (
                tensors["chosen_horizon_tick"]
                == tensors["contact_tick"] - tensors["launch_tick"]
            )
            & task_close_complete
        )
        if (
            bool(
                (
                    canonical_phase
                    >= len(ACTION_BALL_CONTINUOUS_CANONICAL_PHASES)
                ).any()
            )
            or bool(
                (legacy_phase >= len(ACTION_BALL_CONTINUOUS_MOTION_PHASES)).any()
            )
            or bool((phase_start_tick > control_tick).any())
            or bool(
                (
                    active_canonical
                    & (
                        ~sequence_active
                        | (control_tick < 0)
                        | (phase_start_tick < 0)
                        | (scheduled_ordinal < 0)
                        | (reveal_tick < 0)
                        | (reveal_tick > control_tick)
                        | (deadline_tick < reveal_tick)
                        | (next_reveal_tick <= deadline_tick)
                        | (last_closed_ordinal != expected_last_closed)
                        | (
                            opportunities_consumed
                            != expected_opportunities_consumed
                        )
                    )
                ).any()
            )
            or bool(
                (
                    task_valid
                    & (
                        (tensors["task_identity"] < 0)
                        | (tensors["cadence_identity"] < 0)
                        | ~timing_active
                    )
                ).any()
            )
            or bool((playback_started & ~task_valid).any())
            or bool(
                (
                    (canonical_phase <= 2)
                    & ~task_valid
                ).any()
            )
            or bool(
                (
                    (canonical_phase >= 3)
                    & task_valid
                ).any()
            )
            or bool(
                (
                    (canonical_phase == 0)
                    & (
                        playback_started
                        | (phase_start_tick != reveal_tick)
                        | (control_tick >= deadline_tick)
                    )
                ).any()
            )
            or bool(
                (
                    (canonical_phase == 1)
                    & (
                        ~playback_started
                        | (phase_start_tick < reveal_tick)
                        | (control_tick >= deadline_tick)
                    )
                ).any()
            )
            or bool(
                (
                    (canonical_phase == 2)
                    & (
                        ~playback_started
                        | (phase_start_tick != deadline_tick)
                        | (control_tick < deadline_tick)
                    )
                ).any()
            )
            or bool((active_canonical & ~complete_identity).any())
            or bool(
                (
                    active_canonical
                    & (
                        ~torch.isfinite(tensors["pending_elapsed_s"])
                        | ~torch.isfinite(tensors["task_age_s"])
                        | ~torch.isfinite(tensors["time_to_contact_s"])
                        | ~torch.isfinite(tensors["teacher_rate"])
                        | ~torch.isfinite(tensors["scaled_t_hit_s"])
                        | ~torch.isfinite(tensors["scaled_t_cycle_s"])
                        | ~torch.isfinite(tensors["pre_swing_wait_s"])
                        | (tensors["pending_elapsed_s"] < 0)
                        | (tensors["task_age_s"] < 0)
                        | (tensors["time_to_contact_s"] <= 0)
                        | (tensors["teacher_rate"] <= 0)
                        | (tensors["scaled_t_hit_s"] <= 0)
                        | (tensors["scaled_t_cycle_s"] <= 0)
                        | (tensors["pre_swing_wait_s"] < 0)
                    )
                ).any()
            )
            or bool(
                (
                    tensors["action_slot"]
                    >= int(self.motion.num_segments)
                ).any()
            )
        ):
            raise ValueError(
                "Motion fresh checkpoint lifecycle invariants differ"
            )
        canonical_payload = {
            "checkpoint_kind": leaf["checkpoint_kind"],
            "schema_version": leaf["schema_version"],
            "phase": leaf["phase"],
            "owner_mutation_version": leaf["owner_mutation_version"],
            "device_owner_mutation_version": [
                leaf["owner_mutation_version"]
            ],
            "next_serial": leaf["next_serial"],
            "selected_reset_next_serial": leaf[
                "selected_reset_next_serial"
            ],
            "terminal_resolution_total": leaf[
                "terminal_resolution_total"
            ],
            "fault_count_device": fault_count_device.tolist(),
            "global_drain_sequence": leaf["global_drain_sequence"],
            "global_drain_last_update": leaf[
                "global_drain_last_update"
            ],
            "global_drain_last_completed_steps": leaf[
                "global_drain_last_completed_steps"
            ],
            "global_drain_last_acknowledged_mutation_version": leaf[
                "global_drain_last_acknowledged_mutation_version"
            ],
            "checkpoint_requires_global_drain_ack": leaf[
                "checkpoint_requires_global_drain_ack"
            ],
            "published_common_step": leaf["published_common_step"],
            "observation_common_step": leaf["observation_common_step"],
            "tensors": {
                name: value.tolist() for name, value in tensors.items()
            },
            "poisoned": leaf["poisoned"],
        }
        if hashlib.sha256(
            _canonical_json_bytes(canonical_payload)
        ).hexdigest() != leaf["canonical_sha256"]:
            raise ValueError("Motion fresh checkpoint root differs")
        # This host value is only the zero/nonzero work cache for the device
        # mask.  It is deliberately not serialized as a second authority.
        # Derive it from the already validated CPU checkpoint tensor.
        safe_ready_pending_count = (
            self.num_envs
            if bool(tensors["reset_ready_reference_pending"].any())
            else 0
        )
        return {
            **{
                name: leaf[name]
                for name in (
                    "owner_mutation_version",
                    "next_serial",
                    "selected_reset_next_serial",
                    "terminal_resolution_total",
                    "global_drain_sequence",
                    "global_drain_last_update",
                    "global_drain_last_completed_steps",
                    "global_drain_last_acknowledged_mutation_version",
                    "checkpoint_requires_global_drain_ack",
                    "published_common_step",
                    "observation_common_step",
                )
            },
            "device_owner_mutation_version": device_version,
            "terminal_resolution_total_device": terminal_total_device,
            "fault_count_device": fault_count_device,
            "safe_ready_pending_count": safe_ready_pending_count,
            "tensors": tensors,
        }

    def action_ball_continuous_current_projection(
        self,
    ) -> ActionBallContinuousMotionProjection:
        """Return Motion's already-sealed current-tick projection.

        ``common_step`` is the publication tick, not a second cadence.  A
        caller before Motion, or one retaining the prior tick, fails instead
        of seeing a plausible but stale reveal mask.  This consumer never
        clones live Motion state: the publication writer sealed the snapshot
        before any downstream owner could run.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "action-ball continuous Motion cadence is not enabled"
            )
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="projection"
        )
        self._require_action_ball_continuous_parent_authorities()
        common_step = self._require_action_ball_continuous_current_publication(
            operation="projection",
        )
        cached = self._action_ball_continuous_current_projection
        if (
            type(cached) is not ActionBallContinuousMotionProjection
            or cached.common_step != common_step
        ):
            raise RuntimeError(
                "continuous Motion current projection was not sealed by publication"
            )
        self._require_action_ball_continuous_projection_current(cached)
        return self._clone_action_ball_continuous_projection(cached)

    def _seal_action_ball_continuous_current_projection(
        self,
        common_step: int,
    ) -> None:
        """Eagerly freeze current Motion facts at the owner publication point."""

        if (
            type(common_step) is not int
            or common_step < 0
            or self._action_ball_continuous_published_common_step != common_step
            or self._action_ball_continuous_current_projection is not None
        ):
            raise RuntimeError(
                "continuous Motion current projection seal chronology differs"
            )

        def snapshot(value: torch.Tensor) -> torch.Tensor:
            return value.detach().clone()

        projection = ActionBallContinuousMotionProjection(
            common_step=common_step,
            episode_tick=snapshot(
                self._action_ball_continuous_episode_step
            ),
            reveal_due=snapshot(
                self._action_ball_continuous_reveal_due
            ),
            closed_mask=snapshot(
                self._action_ball_continuous_closed_mask
            ),
            close_reason=snapshot(
                self._action_ball_continuous_close_reason
            ),
            deadline_due=snapshot(
                self._action_ball_continuous_deadline_due
            ),
            scheduled_ordinal=snapshot(
                self._action_ball_continuous_scheduled_ordinal
            ),
            reveal_tick=snapshot(
                self._action_ball_continuous_current_reveal_step
            ),
            deadline_tick=snapshot(
                self._action_ball_continuous_current_deadline_step
            ),
            next_reveal_tick=snapshot(
                self._action_ball_continuous_next_reveal_step
            ),
            ready_at_reveal=snapshot(
                self._action_ball_continuous_ready_at_reveal
            ),
            motion_active=snapshot(
                self._action_ball_continuous_motion_active
            ),
            ready_reference_active=snapshot(
                self._action_ball_continuous_ready_reference_active
            ),
            suffix_complete=snapshot(
                self._action_ball_continuous_suffix_complete
            ),
            reset_generation=snapshot(
                self._action_ball_reset_generation
            ),
            swing_generation=snapshot(
                self._action_ball_swing_generation
            ),
        )
        self._action_ball_continuous_current_projection = projection

    @staticmethod
    def _clone_action_ball_continuous_projection(
        projection: ActionBallContinuousMotionProjection,
    ) -> ActionBallContinuousMotionProjection:
        """Give each consumer an isolated copy of the sealed full-N row."""

        def snapshot(value: torch.Tensor) -> torch.Tensor:
            return value.detach().clone()

        return ActionBallContinuousMotionProjection(
            common_step=projection.common_step,
            episode_tick=snapshot(projection.episode_tick),
            reveal_due=snapshot(projection.reveal_due),
            closed_mask=snapshot(projection.closed_mask),
            close_reason=snapshot(projection.close_reason),
            deadline_due=snapshot(projection.deadline_due),
            scheduled_ordinal=snapshot(projection.scheduled_ordinal),
            reveal_tick=snapshot(projection.reveal_tick),
            deadline_tick=snapshot(projection.deadline_tick),
            next_reveal_tick=snapshot(projection.next_reveal_tick),
            ready_at_reveal=snapshot(projection.ready_at_reveal),
            motion_active=snapshot(projection.motion_active),
            ready_reference_active=snapshot(
                projection.ready_reference_active
            ),
            suffix_complete=snapshot(projection.suffix_complete),
            reset_generation=snapshot(projection.reset_generation),
            swing_generation=snapshot(projection.swing_generation),
        )

    def _require_action_ball_continuous_projection_current(
        self,
        projection: ActionBallContinuousMotionProjection,
    ) -> None:
        """Require Motion's sealed snapshot for the current manager tick."""

        common_step = getattr(self._env, "common_step_counter", None)
        if (
            type(projection) is not ActionBallContinuousMotionProjection
            or self._action_ball_continuous_current_projection is not projection
            or type(common_step) is not int
            or common_step != projection.common_step
            or self._action_ball_continuous_published_common_step
            != projection.common_step
        ):
            raise RuntimeError(
                "continuous Motion projection is stale or Command order is swapped"
            )

    def _action_ball_continuous_require_full_reveal_batch(
        self,
        ids: torch.Tensor,
        ordinals: torch.Tensor,
        *,
        operation: str,
    ) -> None:
        """Require one sorted, complete batch over every pending reveal row."""

        expected = torch.where(
            self._action_ball_continuous_reveal_due
        )[0]
        if not torch.equal(ids, expected):
            raise RuntimeError(
                f"continuous Motion {operation} must cover the complete "
                "strictly ordered reveal batch"
            )
        admissible = (
            self._action_ball_continuous_task_commit_pending[ids]
            & ~self._action_ball_continuous_task_committed[ids]
            & ~self._action_ball_continuous_motion_active[ids]
        )
        if not bool(admissible.all()):
            raise RuntimeError(
                f"continuous Motion {operation} reveal batch is not fully admissible"
            )
        if not bool(
            torch.eq(
                self._action_ball_continuous_scheduled_ordinal[ids],
                ordinals,
            ).all()
        ):
            raise RuntimeError(
                f"continuous Motion {operation} ordinal differs from the current reveal"
            )

    def _action_ball_continuous_require_full_release_batch(
        self,
        ids: torch.Tensor,
        ordinals: torch.Tensor,
    ) -> None:
        """Require one sorted, complete release over every ready commit."""

        expected = torch.where(
            self._action_ball_continuous_reveal_due
            & self._action_ball_continuous_ready_at_reveal
        )[0]
        if not torch.equal(ids, expected):
            raise RuntimeError(
                "continuous Motion playback release is not a ready committed "
                "reveal or complete strictly ordered ready batch"
            )
        admissible = (
            self._action_ball_continuous_task_committed[ids]
            & self._action_ball_continuous_motion_release_pending[ids]
            & ~self._action_ball_continuous_motion_active[ids]
        )
        if not bool(admissible.all()):
            raise RuntimeError(
                "continuous Motion playback release is not a ready committed reveal"
            )
        if not bool(
            torch.eq(
                self._action_ball_continuous_scheduled_ordinal[ids],
                ordinals,
            ).all()
        ):
            raise RuntimeError(
                "continuous Motion playback release ordinal differs from the current reveal"
            )

    def _action_ball_continuous_commit_tensor_receipts(
        self,
    ) -> tuple[tuple[object, int], ...]:
        tensors = (
            self._action_ball_continuous_episode_step,
            self._action_ball_continuous_scheduled_ordinal,
            self._action_ball_continuous_current_reveal_step,
            self._action_ball_continuous_current_deadline_step,
            self._action_ball_continuous_next_reveal_step,
            self._action_ball_continuous_reveal_due,
            self._action_ball_continuous_task_commit_pending,
            self._action_ball_continuous_task_committed,
            self._action_ball_continuous_motion_active,
            self._action_ball_continuous_ready_at_reveal,
            self._action_ball_continuous_motion_release_pending,
            self._action_ball_continuous_ready_reference_active,
            self._action_ball_continuous_suffix_complete,
            self._action_ball_reset_generation,
            self._action_ball_swing_generation,
            self.clip_id,
            self._action_ball_task_pending_elapsed_s,
            self._action_ball_task_age_s,
            self._action_ball_time_to_contact_s,
            self._action_ball_teacher_rate,
            self._action_ball_scaled_t_hit_s,
            self._action_ball_scaled_t_cycle_s,
            self._action_ball_pre_swing_wait_s,
            self._action_ball_task_timing_active,
        )
        receipts = tuple(
            _tensor_identity_version_receipt(tensor) for tensor in tensors
        )
        if any(receipt is None for receipt in receipts):
            raise RuntimeError(
                "continuous Motion cannot seal commit tensor receipts"
            )
        return receipts

    def _validate_action_ball_continuous_full_suffix_window(
        self,
        *,
        env_id: int,
        timing: dict[str, float],
        task_age_s: float,
    ) -> None:
        """Prove a ready row's complete suffix closes before next reveal."""

        gap_steps = int(
            self._action_ball_continuous_next_reveal_step[env_id].item()
            - self._action_ball_continuous_episode_step[env_id].item()
        )
        cycle_total_s = float(
            timing["pre_swing_wait_s"] + timing["scaled_t_cycle_s"]
        )
        age_s = float(task_age_s)
        if not math.isfinite(age_s) or age_s < 0.0:
            raise RuntimeError(
                "continuous Motion task age is invalid for suffix admission"
            )
        # Timing checks cycle-due before each increment.  The last check that
        # may latch ready before the next reveal therefore sees age
        # A + (gap - 2) * dt.  Completing on the reveal tick is too late:
        # readiness is sampled before timing advances in that update.
        latest_pre_reveal_age_s = (
            age_s + (gap_steps - 2) * float(self._env.step_dt)
        )
        if (
            gap_steps < 2
            or latest_pre_reveal_age_s + 1.0e-12 < cycle_total_s
        ):
            raise RuntimeError(
                "continuous Motion full suffix cannot complete before "
                "the next frozen scheduled reveal"
            )

    @staticmethod
    def _action_ball_continuous_motion_sha256(
        value: object, *, label: str
    ) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} is not one lowercase SHA-256")
        return value

    def bind_action_ball_continuous_motion_staging(
        self, transaction_owner: object
    ) -> None:
        """Bind the legacy portable-R05 diagnostic Motion lane."""

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "continuous Motion staging requires the fresh cadence"
            )
        if self._action_ball_continuous_motion_device_r05_owner is not None:
            raise RuntimeError(
                "portable Motion staging cannot coexist with Device-R05 reveal ingress"
            )
        import action_ball_continuous_runtime_transaction as transaction

        if (
            type(transaction_owner)
            is not transaction.ContinuousRuntimeTransactionOwner
        ):
            raise TypeError(
                "continuous Motion staging requires the exact pure R05 owner"
            )
        transaction_path = Path(transaction.__file__).resolve()
        transaction_source_sha256 = hashlib.sha256(
            transaction_path.read_bytes()
        ).hexdigest()
        if _ACTION_BALL_CONTINUOUS_R05_SOURCE_SHA256 is None:
            raise RuntimeError(
                "continuous Motion staging R05 final source pin is pending"
            )
        if (
            transaction_source_sha256
            != _ACTION_BALL_CONTINUOUS_R05_SOURCE_SHA256
        ):
            raise RuntimeError(
                "continuous Motion staging R05 source pin differs"
            )
        if (
            transaction.INTEGRATION_STATUS != "PRE_INTEGRATION_HOLD"
            or transaction.RUNTIME_WIRING_CONNECTED is not False
        ):
            raise RuntimeError(
                "continuous Motion staging cannot reinterpret R05 status"
            )
        current = self._action_ball_continuous_transaction_owner
        if current is not None and current is not transaction_owner:
            raise RuntimeError(
                "continuous Motion transaction owner may not be rebound"
            )
        if current is transaction_owner:
            return
        self._action_ball_continuous_transaction_module = transaction
        self._action_ball_continuous_transaction_owner = transaction_owner
        self._action_ball_continuous_motion_owner_nonce = object()
        self._action_ball_continuous_motion_next_serial = 0
        self._action_ball_continuous_motion_mutation_version = 0
        self._action_ball_continuous_motion_device_mutation_version = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
        )
        self._action_ball_continuous_motion_boundary_fault_schema = None
        self._action_ball_continuous_motion_stage = None
        self._action_ball_continuous_motion_prearmed_install = None
        self._action_ball_continuous_motion_prearmed_accept_swaps = None
        self._action_ball_continuous_motion_prearmed_censor_swaps = None
        self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_accept_refs = None
        self._action_ball_continuous_motion_prearmed_boundary_row = None
        self._action_ball_continuous_motion_armed_install = None
        self._action_ball_continuous_motion_censored_install = None
        self._action_ball_continuous_motion_armed_swaps = None
        self._action_ball_continuous_motion_armed_refs = None
        self._action_ball_continuous_motion_commit_receipt = None
        self._action_ball_continuous_motion_terminal_claim = None
        self._action_ball_continuous_motion_terminal_expectations = None
        self._action_ball_continuous_motion_terminal_token = None
        self._action_ball_continuous_motion_terminal_epoch_committed = False
        self._action_ball_continuous_motion_poisoned = False
        self._action_ball_continuous_motion_poison_reason = None
        self._action_ball_continuous_motion_fault_count_device = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self._action_ball_continuous_motion_terminal_resolution_total = 0
        self._action_ball_continuous_motion_terminal_resolution_total_device = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
        )
        self._action_ball_continuous_motion_global_drain_active = None
        self._action_ball_continuous_motion_global_drain_sequence = 0
        self._action_ball_continuous_motion_global_drain_last_update = -1
        self._action_ball_continuous_motion_global_drain_last_completed_steps = -1
        self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = -1
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True
        self._action_ball_continuous_motion_global_drain_poisoned = False
        self._action_ball_continuous_motion_global_drain_poison_reason = None
        self._action_ball_continuous_fresh_motion_lane_bound = True

    def bind_action_ball_continuous_motion_device_r05_reveal(
        self,
        device_r05_owner: object,
    ) -> None:
        """Construction-bind production reveal to one exact Device-R05 owner.

        This binder does not import, construct, or retain the portable R05
        owner.  The later hot stage may therefore receive only an opaque
        Device-R05 preview and its owner-issued child projections.  Keeping
        this capability family mutually exclusive with the compatibility
        binder makes a factory mistake fail at construction, before any
        preview or live Motion write exists.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError(
                "Device-R05 Motion reveal requires the fresh cadence"
            )
        import action_ball_continuous_runtime_transaction_device as device_r05

        required_methods = (
            "project_owned_genesis_for_child",
            "require_owned_genesis_projection",
            "require_owned_action_epoch_accepted",
            "require_owned_terminal_claim_for_child",
            "require_owned_terminal_receipt_for_child",
        )
        if (
            type(device_r05_owner) is not device_r05.DeviceR05Owner
            or any(
                not callable(getattr(device_r05_owner, name, None))
                or getattr(
                    getattr(device_r05_owner, name), "__self__", None
                )
                is not device_r05_owner
                or getattr(
                    getattr(device_r05_owner, name), "__func__", None
                )
                is not getattr(device_r05.DeviceR05Owner, name, None)
                for name in required_methods
            )
        ):
            raise TypeError(
                "Motion hot reveal requires the exact Device-R05 child API"
            )
        if self._action_ball_continuous_transaction_owner is not None:
            raise RuntimeError(
                "Device-R05 Motion reveal cannot coexist with portable R05 staging"
            )
        current = self._action_ball_continuous_motion_device_r05_owner
        if current is device_r05_owner:
            return
        if current is not None or self._action_ball_continuous_fresh_motion_lane_bound:
            raise RuntimeError(
                "Device-R05 Motion reveal owner may not be rebound"
            )

        schedule = self._action_ball_continuous_schedule_projection
        if not isinstance(schedule, MappingProxyType):
            raise RuntimeError(
                "Motion hot reveal genesis lacks the cold-bound parent schedule"
            )
        try:
            origin = int(schedule["sequence_origin_step"])
            first_reveal = int(schedule["first_reveal_step"])
            upcoming_action_slot = int(schedule["upcoming_action_slot"])
            upcoming_action_uid = int(schedule["upcoming_action_uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Motion hot reveal genesis parent schedule differs"
            ) from exc
        code_owned_action_uids = (
            self._action_ball_continuous_code_owned_action_uids()
        )
        if (
            origin < 0
            or first_reveal <= origin
            or upcoming_action_slot < 0
            or upcoming_action_slot >= len(code_owned_action_uids)
            or code_owned_action_uids[upcoming_action_slot]
            != upcoming_action_uid
        ):
            raise RuntimeError(
                "Motion hot reveal genesis parent schedule values differ"
            )

        # Device-R05 is constructed from the independent world-reset
        # chronology, while the legacy broker initializes Motion's destination
        # tensor to zero.  Join those two sole-writer initial states exactly
        # once, through the registry-backed genesis capability, before the top
        # closes Device-R05's construction window.  This is initialization of
        # Motion's own destination chronology, not a same-writer echo gate.
        try:
            genesis_projection = (
                device_r05_owner.project_owned_genesis_for_child(
                    owner_kind="motion"
                )
            )
            genesis_view = device_r05_owner.require_owned_genesis_projection(
                genesis_projection,
                owner_kind="motion",
            )
        except Exception as exc:
            raise RuntimeError(
                "Motion hot reveal lacks the owner-issued Device-R05 genesis join"
            ) from exc
        reset_generation = getattr(genesis_view, "reset_generation", None)
        if (
            type(genesis_view) is not device_r05.DeviceR05GenesisView
            or getattr(genesis_view, "device_r05_owner", None)
            is not device_r05_owner
            or getattr(genesis_view, "owner_kind", None) != "motion"
            or getattr(genesis_view, "world_reset_identity", None) is None
            or not torch.is_tensor(reset_generation)
            or reset_generation.dtype != torch.int64
            or reset_generation.device != torch.device(self.device)
            or tuple(reset_generation.shape) != (self.num_envs,)
        ):
            raise RuntimeError(
                "Motion hot reveal Device-R05 genesis projection differs"
            )

        # The fresh full-MDP constructor deliberately has no legacy birth
        # broker.  Allocate Motion's own reset/swing and task-timing
        # destinations exactly once, only after the owner-issued genesis has
        # passed.  A legacy broker may already have allocated the same Motion-
        # owned tensors; that path is validated and retained byte-for-byte.
        fresh_buffer_specs = (
            ("_action_ball_reset_generation", torch.int64),
            ("_action_ball_swing_generation", torch.int64),
            ("_action_ball_task_timing_active", torch.bool),
            ("_action_ball_task_pending_elapsed_s", torch.float64),
            ("_action_ball_task_age_s", torch.float64),
            ("_action_ball_time_to_contact_s", torch.float64),
            ("_action_ball_teacher_rate", torch.float64),
            ("_action_ball_scaled_t_hit_s", torch.float64),
            ("_action_ball_scaled_t_cycle_s", torch.float64),
            ("_action_ball_pre_swing_wait_s", torch.float64),
        )
        retained_fresh_buffers = tuple(
            getattr(self, name, None) for name, _dtype in fresh_buffer_specs
        )
        allocate_fresh_buffers = all(
            value is None for value in retained_fresh_buffers
        )
        if allocate_fresh_buffers:
            retained_fresh_buffers = tuple(
                torch.zeros(
                    self.num_envs,
                    dtype=dtype,
                    device=self.device,
                )
                for _name, dtype in fresh_buffer_specs
            )
        elif any(value is None for value in retained_fresh_buffers):
            raise RuntimeError(
                "Motion fresh owner buffers are only partially initialized"
            )
        if any(
            type(value) is not torch.Tensor
            or value.dtype != dtype
            or value.device != torch.device(self.device)
            or tuple(value.shape) != (self.num_envs,)
            for value, (_name, dtype) in zip(
                retained_fresh_buffers, fresh_buffer_specs
            )
        ):
            raise RuntimeError("Motion fresh owner buffer shape differs")
        fresh_buffers = {
            name: value
            for (name, _dtype), value in zip(
                fresh_buffer_specs, retained_fresh_buffers
            )
        }
        # FullMDP's first curriculum segment is balance, not action-frame
        # imitation.  Freeze one real reset-ready FK tuple after genesis so
        # joint, body and anchor teachers all describe the same state until
        # the first accepted task.  Reuse the existing generic safe-ready
        # buffers; enabling the legacy split/canonical modes would impose
        # unrelated N=1 clip contracts on the fresh catalog.
        ready_body_shape = (self.num_envs, len(self.cfg.body_names), 3)
        ready_quat_shape = (self.num_envs, len(self.cfg.body_names), 4)
        ready_pos = getattr(
            self, "_action_ball_safe_ready_body_pos_w", None
        )
        ready_quat = getattr(
            self, "_action_ball_safe_ready_body_quat_w", None
        )
        ready_pending = getattr(
            self, "_action_ball_safe_ready_reference_pending", None
        )
        if ready_pos is None and ready_quat is None:
            if ready_pending is not None and (
                type(ready_pending) is not torch.Tensor
                or tuple(ready_pending.shape) != (self.num_envs,)
                or ready_pending.dtype != torch.bool
                or ready_pending.device != torch.device(self.device)
            ):
                raise RuntimeError(
                    "Motion fresh reset-ready pending state differs"
                )
            ready_pos = torch.zeros(
                ready_body_shape,
                dtype=self.motion.body_pos_w.dtype,
                device=self.device,
            )
            ready_quat = torch.zeros(
                ready_quat_shape,
                dtype=self.motion.body_quat_w.dtype,
                device=self.device,
            )
            if ready_pending is None:
                ready_pending = torch.ones(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
        elif (
            type(ready_pos) is not torch.Tensor
            or tuple(ready_pos.shape) != ready_body_shape
            or ready_pos.dtype != self.motion.body_pos_w.dtype
            or ready_pos.device != torch.device(self.device)
            or type(ready_quat) is not torch.Tensor
            or tuple(ready_quat.shape) != ready_quat_shape
            or ready_quat.dtype != self.motion.body_quat_w.dtype
            or ready_quat.device != torch.device(self.device)
            or type(ready_pending) is not torch.Tensor
            or tuple(ready_pending.shape) != (self.num_envs,)
            or ready_pending.dtype != torch.bool
            or ready_pending.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "Motion fresh reset-ready reference buffers differ"
            )
        ready_pending.fill_(True)
        self._action_ball_safe_ready_body_pos_w = ready_pos
        self._action_ball_safe_ready_body_quat_w = ready_quat
        self._action_ball_safe_ready_reference_pending = ready_pending
        self._action_ball_safe_ready_pending_count = self.num_envs
        time_left = getattr(self, "time_left", None)
        time_left_receipt = _tensor_identity_version_receipt(time_left)
        if (
            type(time_left) is not torch.Tensor
            or not time_left.is_floating_point()
            or time_left.device != torch.device(self.device)
            or tuple(time_left.shape) != (self.num_envs,)
            or not time_left.is_contiguous()
            or time_left_receipt is None
        ):
            raise RuntimeError(
                "Motion fresh genesis requires its exact inherited resample timer"
            )

        # The independent reset genesis is the first whole-environment reset;
        # no selected-reset callback follows it.  Materialize the complete
        # cadence/reference after-image from the already cold-bound parent
        # schedule before publishing any destination or sealing the fresh
        # lane.  Thus policy tick 0 is ordinary and tick ``first_reveal`` is
        # the first D05 opportunity.  This is deliberately not routed through
        # the tombstoned legacy reset API.
        def full(value: torch.Tensor, fill_value) -> torch.Tensor:
            return torch.full_like(value, fill_value)

        upcoming_slots = full(self.clip_id, upcoming_action_slot)
        ready_steps = self.motion.seg_start[upcoming_slots]
        replacements = (
            (time_left, full(time_left, float("inf"))),
            (self.clip_id, upcoming_slots),
            (
                self._action_ball_continuous_sequence_active,
                full(self._action_ball_continuous_sequence_active, True),
            ),
            (
                self._action_ball_continuous_episode_step,
                full(self._action_ball_continuous_episode_step, origin - 1),
            ),
            (
                self._action_ball_continuous_scheduled_ordinal,
                full(self._action_ball_continuous_scheduled_ordinal, -1),
            ),
            (
                self._action_ball_continuous_current_reveal_step,
                full(self._action_ball_continuous_current_reveal_step, -1),
            ),
            (
                self._action_ball_continuous_current_deadline_step,
                full(self._action_ball_continuous_current_deadline_step, -1),
            ),
            (
                self._action_ball_continuous_next_reveal_step,
                full(
                    self._action_ball_continuous_next_reveal_step,
                    first_reveal,
                ),
            ),
            (
                self._action_ball_continuous_last_closed_ordinal,
                full(self._action_ball_continuous_last_closed_ordinal, -1),
            ),
            (
                self._action_ball_continuous_opportunities_consumed,
                full(self._action_ball_continuous_opportunities_consumed, 0),
            ),
            (
                self._action_ball_continuous_policy_opportunities_created,
                full(
                    self._action_ball_continuous_policy_opportunities_created,
                    0,
                ),
            ),
            (
                self._action_ball_continuous_infrastructure_censors_consumed,
                full(
                    self._action_ball_continuous_infrastructure_censors_consumed,
                    0,
                ),
            ),
            (
                self._action_ball_continuous_current_policy_opportunity,
                full(
                    self._action_ball_continuous_current_policy_opportunity,
                    False,
                ),
            ),
            (
                self._action_ball_continuous_motion_active,
                full(self._action_ball_continuous_motion_active, False),
            ),
            (
                self._action_ball_continuous_suffix_complete,
                full(self._action_ball_continuous_suffix_complete, False),
            ),
            (
                self._action_ball_continuous_ready_reference_active,
                full(
                    self._action_ball_continuous_ready_reference_active,
                    True,
                ),
            ),
            (
                self._action_ball_continuous_ready_at_reveal,
                full(self._action_ball_continuous_ready_at_reveal, False),
            ),
            (
                self._action_ball_continuous_reveal_due,
                full(self._action_ball_continuous_reveal_due, False),
            ),
            (
                self._action_ball_continuous_deadline_due,
                full(self._action_ball_continuous_deadline_due, False),
            ),
            (
                self._action_ball_continuous_recovery_unavailable,
                full(self._action_ball_continuous_recovery_unavailable, False),
            ),
            (
                self._action_ball_continuous_task_commit_pending,
                full(self._action_ball_continuous_task_commit_pending, False),
            ),
            (
                self._action_ball_continuous_task_commit_missed,
                full(self._action_ball_continuous_task_commit_missed, False),
            ),
            (
                self._action_ball_continuous_task_committed,
                full(self._action_ball_continuous_task_committed, False),
            ),
            (
                self._action_ball_continuous_motion_release_pending,
                full(
                    self._action_ball_continuous_motion_release_pending,
                    False,
                ),
            ),
            (
                self._action_ball_continuous_motion_release_missed,
                full(
                    self._action_ball_continuous_motion_release_missed,
                    False,
                ),
            ),
            (
                self._action_ball_continuous_phase,
                full(
                    self._action_ball_continuous_phase,
                    _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                        "pre_reveal_hidden"
                    ],
                ),
            ),
            (
                self._action_ball_continuous_canonical_phase,
                full(
                    self._action_ball_continuous_canonical_phase,
                    _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE[
                        "recover_hidden"
                    ],
                ),
            ),
            (
                self._action_ball_continuous_canonical_phase_start_tick,
                full(
                    self._action_ball_continuous_canonical_phase_start_tick,
                    origin - 1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_task_identity,
                full(
                    self._action_ball_continuous_canonical_task_identity,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_cadence_identity,
                full(
                    self._action_ball_continuous_canonical_cadence_identity,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_action_uid,
                full(self._action_ball_continuous_canonical_action_uid, -1),
            ),
            (
                self._action_ball_continuous_canonical_shot_index,
                full(self._action_ball_continuous_canonical_shot_index, -1),
            ),
            (
                self._action_ball_continuous_canonical_outcome_identity,
                full(
                    self._action_ball_continuous_canonical_outcome_identity,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_task_receipt_sha256,
                full(
                    self._action_ball_continuous_canonical_task_receipt_sha256,
                    0,
                ),
            ),
            (
                self._action_ball_continuous_canonical_cadence_receipt_sha256,
                full(
                    self._action_ball_continuous_canonical_cadence_receipt_sha256,
                    0,
                ),
            ),
            (
                self._action_ball_continuous_canonical_candidate_identity,
                full(
                    self._action_ball_continuous_canonical_candidate_identity,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_contact_tick,
                full(
                    self._action_ball_continuous_canonical_contact_tick,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_launch_tick,
                full(self._action_ball_continuous_canonical_launch_tick, -1),
            ),
            (
                self._action_ball_continuous_canonical_chosen_horizon_tick,
                full(
                    self._action_ball_continuous_canonical_chosen_horizon_tick,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_task_close_tick,
                full(
                    self._action_ball_continuous_canonical_task_close_tick,
                    -1,
                ),
            ),
            (
                self._action_ball_continuous_canonical_task_valid,
                full(self._action_ball_continuous_canonical_task_valid, False),
            ),
            (
                self._action_ball_continuous_canonical_playback_started,
                full(
                    self._action_ball_continuous_canonical_playback_started,
                    False,
                ),
            ),
            (
                fresh_buffers["_action_ball_task_timing_active"],
                full(
                    fresh_buffers["_action_ball_task_timing_active"], False
                ),
            ),
            (
                fresh_buffers["_action_ball_task_pending_elapsed_s"],
                full(
                    fresh_buffers["_action_ball_task_pending_elapsed_s"],
                    0.0,
                ),
            ),
            (
                fresh_buffers["_action_ball_task_age_s"],
                full(fresh_buffers["_action_ball_task_age_s"], 0.0),
            ),
            (
                fresh_buffers["_action_ball_time_to_contact_s"],
                full(fresh_buffers["_action_ball_time_to_contact_s"], 0.0),
            ),
            (
                fresh_buffers["_action_ball_teacher_rate"],
                full(fresh_buffers["_action_ball_teacher_rate"], 0.0),
            ),
            (
                fresh_buffers["_action_ball_scaled_t_hit_s"],
                full(fresh_buffers["_action_ball_scaled_t_hit_s"], 0.0),
            ),
            (
                fresh_buffers["_action_ball_scaled_t_cycle_s"],
                full(fresh_buffers["_action_ball_scaled_t_cycle_s"], 0.0),
            ),
            (
                fresh_buffers["_action_ball_pre_swing_wait_s"],
                full(fresh_buffers["_action_ball_pre_swing_wait_s"], 0.0),
            ),
            (
                self._action_ball_continuous_motion_reset_pending,
                full(self._action_ball_continuous_motion_reset_pending, False),
            ),
            (
                fresh_buffers["_action_ball_reset_generation"],
                reset_generation.clone(),
            ),
            (
                fresh_buffers["_action_ball_swing_generation"],
                full(fresh_buffers["_action_ball_swing_generation"], 0),
            ),
            (self.time_steps, ready_steps.clone()),
            (
                self.time_steps_f,
                ready_steps.to(dtype=self.time_steps_f.dtype),
            ),
            (self.speed_scale, full(self.speed_scale, 0.0)),
            (self.hold_counter, full(self.hold_counter, 1)),
        )
        if "in_hold" in self.metrics:
            replacements = (
                *replacements,
                (
                    self.metrics["in_hold"],
                    full(self.metrics["in_hold"], 1.0),
                ),
            )
        if allocate_fresh_buffers:
            for name, value in fresh_buffers.items():
                setattr(self, name, value)
            self._action_ball_active_task_refs = [None] * self.num_envs
            self._action_ball_diagnostic_pending_row_count = 0
        for destination, after_image in replacements:
            destination.copy_(after_image)
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_current_projection = None
        self._action_ball_continuous_published_common_step = None
        self._invalidate_action_ball_continuous_observation_publication()
        self._action_ball_continuous_motion_device_r05_owner = device_r05_owner
        self._action_ball_continuous_motion_owner_nonce = object()
        self._action_ball_continuous_motion_next_serial = 0
        self._action_ball_continuous_motion_mutation_version = 0
        self._action_ball_continuous_motion_device_mutation_version = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
        )
        self._action_ball_continuous_motion_poisoned = False
        self._action_ball_continuous_motion_poison_reason = None
        self._action_ball_continuous_motion_fault_count_device = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self._action_ball_continuous_motion_terminal_resolution_total = 0
        self._action_ball_continuous_motion_terminal_resolution_total_device = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
        )
        self._action_ball_continuous_motion_global_drain_active = None
        self._action_ball_continuous_motion_global_drain_sequence = 0
        self._action_ball_continuous_motion_global_drain_last_update = -1
        self._action_ball_continuous_motion_global_drain_last_completed_steps = -1
        self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = -1
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True
        self._action_ball_continuous_motion_global_drain_poisoned = False
        self._action_ball_continuous_motion_global_drain_poison_reason = None
        self._action_ball_continuous_fresh_time_left_receipt = (
            _tensor_identity_version_receipt(time_left)
        )
        self._action_ball_continuous_fresh_motion_lane_bound = True

    def _latch_action_ball_full_mdp_motion_epoch_row_fault(
        self,
        rows: torch.Tensor,
        *,
        reason_bit: int,
    ) -> torch.Tensor:
        """Latch one named Motion cause and return rows still safe to write.

        The fresh direct lane uses ActionEpoch's existing packed optimizer
        drain, so this adds neither a per-tick device-to-host read nor another
        safety owner.  Legacy diagnostic views without an Epoch owner still
        suppress the bad row locally instead of poisoning the CUDA context.
        """

        if (
            not torch.is_tensor(rows)
            or rows.dtype != torch.bool
            or tuple(rows.shape) != (self.num_envs,)
            or rows.device != torch.device(self.device)
            or type(reason_bit) is not int
            or reason_bit not in _ACTION_EPOCH_MOTION_ROW_FAULT_BITS
        ):
            raise RuntimeError("Motion named ActionEpoch row-fault ABI differs")
        fault_rows = rows.contiguous()
        latch = getattr(
            self, "_action_ball_full_mdp_motion_epoch_fault_latch", None
        )
        writable = getattr(
            self,
            "_action_ball_full_mdp_motion_epoch_writable_rows",
            None,
        )
        if latch is None:
            if getattr(
                self,
                "_action_ball_continuous_fresh_motion_lane_bound",
                False,
            ):
                raise RuntimeError(
                    "fresh Motion row fault requires its exact ActionEpoch owner"
                )
            return ~fault_rows
        if (
            not torch.is_tensor(writable)
            or writable.dtype != torch.bool
            or tuple(writable.shape) != (self.num_envs,)
            or writable.device != torch.device(self.device)
            or not writable.is_contiguous()
        ):
            raise RuntimeError("Motion named ActionEpoch writable-row ABI differs")
        # Preserve causal isolation inside Motion: after this producer has
        # quarantined a row, later derived predicates must not relabel the
        # same root defect with additional downstream bits.
        fault_rows = fault_rows & writable
        safe_rows = latch("motion", reason_bit, fault_rows, owner=self)
        if (
            not torch.is_tensor(safe_rows)
            or safe_rows.dtype != torch.bool
            or tuple(safe_rows.shape) != (self.num_envs,)
            or safe_rows.device != torch.device(self.device)
            or not safe_rows.is_contiguous()
        ):
            raise RuntimeError("Motion named ActionEpoch safe-row ABI differs")
        writable.logical_and_(safe_rows)
        return writable

    def bind_action_ball_full_mdp_motion_epoch_owner(
        self,
        epoch_owner: object,
    ) -> None:
        """Construction-bind the one lean epoch that orders Motion writes.

        The epoch does not prove any Motion fact.  Device-R05 remains the sole
        selected-task producer; the epoch only fixes the irreversible writer
        order and owns the packed mutation log.
        """

        if self._action_ball_continuous_motion_device_r05_owner is None:
            raise RuntimeError(
                "Motion epoch binding requires its exact Device-R05 owner first"
            )
        current = self._action_ball_full_mdp_motion_epoch_owner
        if current is epoch_owner:
            return
        if current is not None:
            raise RuntimeError("Motion epoch owner may not be rebound")
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_full_mdp_epoch as epoch

        required_methods = (
            "current",
            "bind_d05_accept_writers",
            "settle_d05_transaction",
            "require_active_d05_accepted_rows",
            "latch_runtime_row_fault",
        )
        if (
            type(epoch_owner) is not epoch.ActionEpochOwner
            or getattr(epoch, "row_identity", None)
            is not _ACTION_BALL_ROW_IDENTITY
            or epoch_owner.num_envs != self.num_envs
            or epoch_owner.shot_slot_capacity != 1
            or epoch_owner.device != torch.device(self.device)
            or epoch_owner.poisoned
            # The independent reset authority publishes the canonical genesis
            # IDLE before D05 constructs and binds its children.  Therefore the
            # fresh factory arrives at exactly one committed genesis row, not
            # an empty pre-genesis owner.  Shot/reveal work has not begun.
            or epoch_owner.commit_head != 1
            or epoch_owner.drain_frontier != 0
            or any(
                getattr(epoch, name, None) != expected
                for name, expected in _ACTION_EPOCH_MOTION_ROW_FAULT_BINDINGS
            )
            or any(
                not callable(getattr(epoch_owner, name, None))
                or getattr(getattr(epoch_owner, name), "__self__", None)
                is not epoch_owner
                or getattr(getattr(epoch_owner, name), "__func__", None)
                is not getattr(epoch.ActionEpochOwner, name, None)
                for name in required_methods
            )
        ):
            raise TypeError(
                "Motion epoch binding requires one fresh exact ActionEpochOwner"
            )
        bind_playback = getattr(epoch_owner, "bind_motion_playback_owner", None)
        if (
            not callable(bind_playback)
            or getattr(bind_playback, "__self__", None) is not epoch_owner
            or getattr(bind_playback, "__func__", None)
            is not getattr(epoch.ActionEpochOwner, "bind_motion_playback_owner", None)
        ):
            raise TypeError(
                "Motion epoch binding requires the exact playback-transition ABI"
            )
        fault_latch = epoch_owner.latch_runtime_row_fault
        writable_rows = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        bind_playback(self)
        self._action_ball_full_mdp_motion_epoch_owner = epoch_owner
        self._action_ball_full_mdp_motion_epoch_fault_latch = fault_latch
        self._action_ball_full_mdp_motion_epoch_writable_rows = writable_rows

    def action_epoch_playback_transition_mask(
        self,
        kind: str,
        projection: object,
    ) -> torch.Tensor:
        """Derive one packed playback transition solely from Motion state.

        The construction-bound epoch calls this exact method; no caller mask
        or verdict exists.  The returned fixed ``[N, S]`` tensor is safe for a
        named empty packed chronology entry and never requires a device-to-host
        decision.
        """

        epoch_owner = self._action_ball_full_mdp_motion_epoch_owner
        if epoch_owner is None or self._action_ball_continuous_motion_poisoned:
            raise RuntimeError("Motion playback transition owner is unavailable")
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_full_mdp_epoch as epoch
        row_identity = _ACTION_BALL_ROW_IDENTITY
        if (
            type(epoch_owner) is not epoch.ActionEpochOwner
            or type(projection) is not epoch.ActionEpochMotionPlaybackProjection
            or type(kind) is not str
            or kind != epoch.MOTION_PLAYBACK_STARTED
        ):
            raise RuntimeError("Motion playback transition ABI differs")
        device = torch.device(self.device)
        n = self.num_envs
        s = epoch_owner.shot_slot_capacity
        if s != 1:
            raise RuntimeError("Motion playback requires the exact single action slot")
        slots = self._action_ball_full_mdp_motion_exact_tensor(
            projection.current_task_slot,
            name="playback.current_task_slot",
            device=device,
            dtype=torch.int64,
            shape=(n,),
        )
        # The accepted single-action ABI has exactly one slot.  Read that
        # fixed column directly: clamping a damaged task slot back to zero
        # would turn corruption into apparent authority, and a per-tick
        # full-N arange would add allocation without representing semantics.
        row_slot_active = slots.eq(0)
        phase = self._action_ball_full_mdp_motion_exact_tensor(
            projection.phase,
            name="playback.phase",
            device=device,
            dtype=torch.int64,
            shape=(n, s),
        )[:, 0]
        selected = self._action_ball_full_mdp_motion_exact_tensor(
            projection.selected_mask,
            name="playback.selected_mask",
            device=device,
            dtype=torch.bool,
            shape=(n, s),
        )[:, 0]
        retained_key = row_identity.ActionEpochShotKey(
            reset_generation=self._action_ball_reset_generation[:, None],
            ball_generation=self._action_ball_swing_generation[:, None],
            action_uid=self._action_ball_continuous_canonical_action_uid[:, None],
            action_slot=self.clip_id[:, None],
            shot_index=self._action_ball_continuous_canonical_shot_index[:, None],
            task_identity=(
                self._action_ball_continuous_canonical_task_identity[:, None]
            ),
            outcome_identity=(
                self._action_ball_continuous_canonical_outcome_identity[:, None]
            ),
            ball_identity=(
                self._action_ball_continuous_canonical_cadence_identity[:, None]
            ),
        )
        public_key = row_identity.require_action_epoch_shot_key(
            projection.shot_key,
            shape=(n, s),
            device=device,
            label="Motion playback public shot_key",
        )
        retained_key = row_identity.require_action_epoch_shot_key(
            retained_key,
            shape=(n, s),
            device=device,
            label="Motion playback retained shot_key",
        )
        full_key_matches = (
            row_identity.action_epoch_shot_key_valid(public_key)
            & row_identity.action_epoch_shot_key_valid(retained_key)
            & row_identity.action_epoch_shot_key_equal(public_key, retained_key)
        )[:, 0]
        teacher_left_frame0 = self.time_steps.gt(
            self.motion.seg_start[self.clip_id]
        )
        rows = (
            row_slot_active
            & selected
            & epoch.action_epoch_open_shot_phase_mask(phase)
            & full_key_matches
            & self._action_ball_continuous_canonical_task_valid
            & self._action_ball_task_timing_active
            & self._action_ball_continuous_motion_active
            & ~self._action_ball_continuous_canonical_playback_started
            & self._action_ball_full_mdp_motion_epoch_writable_rows
            & teacher_left_frame0
            & (
                self._action_ball_task_age_s + 1.0e-12
                >= self._action_ball_pre_swing_wait_s
            )
        )
        return rows[:, None].contiguous()

    @staticmethod
    def _action_ball_full_mdp_motion_exact_tensor(
        value: object,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if (
            type(value) is not torch.Tensor
            or value.device != device
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise RuntimeError(
                f"Motion epoch {name} must be contiguous {dtype} on "
                f"{device} with shape {shape}"
            )
        return value

    def commit_action_ball_full_mdp_motion_epoch_rows(
        self,
        token: object,
    ) -> None:
        """Install D05's exact full-N ACCEPT after-image under epoch ordering.

        The caller supplies only D05's opaque transaction token.  D05 derives
        and masks the accepted rows from its private candidate; Motion neither
        accepts a row selector nor rereads ActionEpoch's public phase table.
        ActionEpoch owns the corresponding ``WRITES_COMMITTED:motion`` event,
        so an empty callback creates no second Motion transaction counter.
        """

        # The pre-D05 snapshots stop being current at the writer barrier.  A
        # successful outer settlement republishes and eagerly seals the exact
        # same-tick after-image only after every D05 writer has returned.
        self._invalidate_action_ball_continuous_observation_publication()
        self._action_ball_continuous_current_projection = None
        try:
            self._commit_action_ball_full_mdp_motion_epoch_rows_impl(token)
        except BaseException:
            self.poison_global_reveal_epoch(
                "motion_epoch_hot_reveal_post_start_failure"
            )
            raise

    def _commit_action_ball_full_mdp_motion_epoch_rows_impl(
        self,
        token: object,
    ) -> None:
        """Validate the typed view and perform only full-N masked copies."""

        device_r05_owner = self._action_ball_continuous_motion_device_r05_owner
        epoch_owner = self._action_ball_full_mdp_motion_epoch_owner
        if device_r05_owner is None or epoch_owner is None:
            raise RuntimeError(
                "Motion epoch rows require exact Device-R05 and epoch bindings"
            )
        if self._action_ball_continuous_motion_poisoned or epoch_owner.poisoned:
            raise RuntimeError("Motion epoch row owner is poisoned")
        try:
            import action_ball_continuous_runtime_transaction_device as device_r05
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_continuous_runtime_transaction_device as device_r05
            import action_ball_full_mdp_epoch as epoch
        row_identity = _ACTION_BALL_ROW_IDENTITY

        if type(device_r05_owner) is not device_r05.DeviceR05Owner:
            raise RuntimeError("Motion epoch Device-R05 owner type differs")
        if type(epoch_owner) is not epoch.ActionEpochOwner:
            raise RuntimeError("Motion epoch owner type differs")
        accepted = device_r05_owner.require_owned_action_epoch_accepted(
            token, owner_kind="motion"
        )
        if (
            type(accepted) is not device_r05.DeviceR05AcceptedRowsView
            or accepted.transaction is not token
            or type(accepted.identity) is not epoch.EpochIdentityPayload
            or type(accepted.clocks) is not epoch.EpochClockPayload
            or type(accepted.task) is not epoch.EpochTaskPayload
        ):
            raise RuntimeError("Motion epoch accepted-row view differs")

        device = torch.device(self.device)
        n = self.num_envs
        shape = (n, 1)

        def exact(
            value: object,
            *,
            name: str,
            dtype: torch.dtype,
            suffix: tuple[int, ...] = (),
        ) -> torch.Tensor:
            return self._action_ball_full_mdp_motion_exact_tensor(
                value,
                name=name,
                device=device,
                dtype=dtype,
                shape=(*shape, *suffix),
            )

        key = row_identity.require_action_epoch_shot_key(
            accepted.identity.shot_key,
            shape=shape,
            device=device,
            label="Motion accepted shot_key",
        )
        for name in (
            "scheduled_ordinal",
            "target_generation",
            "selected_cell",
            "candidate_identity",
        ):
            exact(
                getattr(accepted.identity, name),
                name="identity." + name,
                dtype=torch.int64,
            )
        for name in (
            "reveal_tick",
            "contact_tick",
            "launch_tick",
            "deadline_tick",
            "next_reveal_tick",
        ):
            exact(
                getattr(accepted.clocks, name),
                name="clocks." + name,
                dtype=torch.int64,
            )
        timing_grid = exact(
            accepted.task.task_f32,
            name="task.task_f32",
            dtype=torch.float32,
            suffix=(epoch.TASK_F32_WIDTH,),
        )
        task_valid_grid = exact(
            accepted.task.task_valid,
            name="task.task_valid",
            dtype=torch.bool,
        )
        playback_grid = exact(
            accepted.playback_admissible,
            name="playback_admissible",
            dtype=torch.bool,
        )
        exact(
            accepted.publication_ordinal,
            name="publication_ordinal",
            dtype=torch.int64,
        )
        exact(
            accepted.target_xy_m,
            name="target_xy_m",
            dtype=torch.float32,
            suffix=(2,),
        )
        exact(
            accepted.rng_counter,
            name="rng_counter",
            dtype=torch.int64,
        )

        # D05 has already neutralized every non-ACCEPT row.  Combining its
        # typed task-valid bit with the shared full-key predicate is the only
        # Motion write mask; no compact row list or caller verdict exists.
        write_rows = (
            row_identity.action_epoch_shot_key_valid(key)
            & task_valid_grid
        )[:, 0]
        if self._action_ball_full_mdp_motion_epoch_writable_rows is not None:
            write_rows &= (
                self._action_ball_full_mdp_motion_epoch_writable_rows
            )
        # Validate/latch the complete reveal reference before any D05 Motion
        # destination changes.  ``refresh`` preserves invalid cache rows and
        # updates the persistent writable mask; re-intersect so a frame/pending
        # defect cannot mutate timing, identity, counters, or playback state
        # before the packed optimizer drain observes bit 25.
        self.refresh_action_ball_revealed_body_reference(write_rows)
        persistent_writable = (
            self._action_ball_full_mdp_motion_epoch_writable_rows
        )
        if torch.is_tensor(persistent_writable):
            write_rows &= persistent_writable
        elif getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            raise RuntimeError(
                "fresh Motion D05 write requires its exact ActionEpoch quarantine"
            )
        # Task publication and teacher playback are separate facts.  A clean
        # not-ready row still receives its task/key/clock and therefore gives
        # the policy a real early opportunity; it simply stays on the ready
        # reference until playback becomes admissible.
        playback_rows = write_rows & playback_grid[:, 0]
        timing = timing_grid[:, 0, :epoch.MOTION_TASK_F32_WIDTH]
        false_rows = torch.zeros(n, dtype=torch.bool, device=device)
        true_rows = torch.ones(n, dtype=torch.bool, device=device)
        zero_f32 = torch.zeros(n, dtype=torch.float32, device=device)
        legacy_phase = torch.where(
            playback_rows,
            torch.full(
                (n,),
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "active_opportunity"
                ],
                dtype=torch.int64,
                device=device,
            ),
            torch.full(
                (n,),
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "recovery_unavailable"
                ],
                dtype=torch.int64,
                device=device,
            ),
        )
        canonical_phase = torch.full(
            (n,),
            ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE,
            dtype=torch.int64,
            device=device,
        )
        task_identity = key.task_identity[:, 0]
        # This legacy-named local field has always retained D05's ball
        # identity.  Scheduled/publication ordinal is deliberately separate.
        cadence_identity = key.ball_identity[:, 0]
        action_uid = key.action_uid[:, 0]
        shot_index = key.shot_index[:, 0]
        outcome_identity = key.outcome_identity[:, 0]
        candidate_identity = accepted.identity.candidate_identity[:, 0]
        reveal_tick = accepted.clocks.reveal_tick[:, 0]
        contact_tick = accepted.clocks.contact_tick[:, 0]
        launch_tick = accepted.clocks.launch_tick[:, 0]
        deadline_tick = accepted.clocks.deadline_tick[:, 0]

        accepted_values = (
            (self._action_ball_task_pending_elapsed_s, zero_f32),
            (self._action_ball_task_age_s, zero_f32),
            (self._action_ball_time_to_contact_s, timing[:, 0]),
            (self._action_ball_teacher_rate, timing[:, 1]),
            (self._action_ball_scaled_t_hit_s, timing[:, 2]),
            (self._action_ball_scaled_t_cycle_s, timing[:, 3]),
            (self._action_ball_pre_swing_wait_s, timing[:, 4]),
            (self._action_ball_task_timing_active, true_rows),
            (self._action_ball_continuous_task_commit_pending, false_rows),
            (self._action_ball_continuous_task_commit_missed, false_rows),
            (self._action_ball_continuous_task_committed, true_rows),
            (self._action_ball_continuous_motion_reset_pending, false_rows),
            (self._action_ball_continuous_motion_release_pending, false_rows),
            (self._action_ball_continuous_motion_release_missed, false_rows),
            (self._action_ball_continuous_motion_active, playback_rows),
            (self._action_ball_continuous_suffix_complete, false_rows),
            (
                self._action_ball_continuous_ready_reference_active,
                ~playback_rows,
            ),
            (self._action_ball_continuous_phase, legacy_phase),
            (self._action_ball_continuous_current_policy_opportunity, true_rows),
            (
                self._action_ball_continuous_policy_opportunities_created,
                self._action_ball_continuous_policy_opportunities_created + 1,
            ),
            (self._action_ball_continuous_canonical_phase, canonical_phase),
            (self._action_ball_continuous_canonical_phase_start_tick, reveal_tick),
            (self._action_ball_continuous_canonical_task_identity, task_identity),
            (
                self._action_ball_continuous_canonical_cadence_identity,
                cadence_identity,
            ),
            (self._action_ball_continuous_canonical_action_uid, action_uid),
            (self._action_ball_continuous_canonical_shot_index, shot_index),
            (
                self._action_ball_continuous_canonical_outcome_identity,
                outcome_identity,
            ),
            (
                self._action_ball_continuous_canonical_candidate_identity,
                candidate_identity,
            ),
            (
                self._action_ball_continuous_canonical_contact_tick,
                contact_tick,
            ),
            (
                self._action_ball_continuous_canonical_launch_tick,
                launch_tick,
            ),
            (
                self._action_ball_continuous_canonical_chosen_horizon_tick,
                contact_tick - launch_tick,
            ),
            (
                self._action_ball_continuous_canonical_task_close_tick,
                deadline_tick,
            ),
            (self._action_ball_continuous_canonical_task_valid, true_rows),
            (
                self._action_ball_continuous_canonical_playback_started,
                false_rows,
            ),
        )

        swaps = []
        for destination, value in accepted_values:
            row_mask = write_rows.reshape(
                (n,) + (1,) * (destination.ndim - 1)
            )
            swaps.append(
                (
                    destination,
                    torch.where(
                        row_mask,
                        value.to(dtype=destination.dtype),
                        destination,
                    ),
                )
            )
        for destination, after_image in swaps:
            destination.copy_(after_image)
        # ``refresh_action_ball_revealed_body_reference`` already installed
        # the healthy rows' selected frame-0 body cache in the preflight above.
        # The pure copies complete the same public D05 transaction here.

    def publish_action_ball_full_mdp_post_d05_observation(self) -> None:
        """Publish Motion only after the complete three-writer D05 commit.

        The Motion writer invalidates the pre-D05 observation and current-row
        projection at its barrier.  D05 invokes this no-argument owner method
        only after ``settle_d05_transaction`` has returned from every writer,
        so downstream owners receive one same-tick row-wise after-image.
        """

        try:
            self._publish_action_ball_full_mdp_post_d05_observation_impl()
        except BaseException:
            # A failed post-write publication must never leave a partially
            # installed or stale capability visible to Physical.
            self._invalidate_action_ball_continuous_observation_publication()
            self._action_ball_continuous_current_projection = None
            self.poison_global_reveal_epoch(
                "motion_post_d05_observation_publication_failed"
            )
            raise

    def _publish_action_ball_full_mdp_post_d05_observation_impl(self) -> None:
        device_r05_owner = self._action_ball_continuous_motion_device_r05_owner
        epoch_owner = self._action_ball_full_mdp_motion_epoch_owner
        if device_r05_owner is None or epoch_owner is None:
            raise RuntimeError(
                "post-D05 Motion observation requires exact owner bindings"
            )
        if self._action_ball_continuous_motion_poisoned or epoch_owner.poisoned:
            raise RuntimeError("post-D05 Motion observation owner is poisoned")
        try:
            import action_ball_continuous_runtime_transaction_device as device_r05
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
        except ImportError:
            import action_ball_continuous_runtime_transaction_device as device_r05
            import action_ball_full_mdp_epoch as epoch

        if (
            type(device_r05_owner) is not device_r05.DeviceR05Owner
            or type(epoch_owner) is not epoch.ActionEpochOwner
        ):
            raise RuntimeError("post-D05 Motion owner type differs")
        if any(
            value is not None
            for value in (
                self._action_ball_continuous_observation_publication_identity,
                self._action_ball_continuous_observation_token,
                self._action_ball_continuous_observation_record,
                self._action_ball_continuous_observation_common_step,
            )
        ):
            raise RuntimeError(
                "post-D05 Motion observation was already published"
            )
        common_step = self._require_action_ball_continuous_current_publication(
            operation="post-D05 observation publication"
        )
        if getattr(self._env, "common_step_counter", None) != common_step:
            raise RuntimeError("post-D05 Motion publication tick drifted")

        # Both publications clone owner state before exposing it.  Neither
        # one rereads ActionEpoch's row table as a second ACCEPT authority.
        self._publish_action_ball_continuous_observation()
        self._seal_action_ball_continuous_current_projection(common_step)

    def project_action_ball_full_mdp_recovery_ready_reference(
        self,
        *,
        action_epoch_snapshot: object = None,
    ) -> ActionBallFullMdpCompletedActionFrame0Reference:
        """Clone Motion's owner-selected recovery-ready frame-0 reference.

        Canonical genesis IDLE uses Motion's already scheduled upcoming
        ``clip_id`` and immutable action table, paired with a neutral shot key.
        Only the typed completed lifecycle uses the epoch-selected action and
        its exact full eight-field shot key.  REJECT/DEFER/CENSOR are event
        classifications and are deliberately not inspected here; the current
        carry lifecycle remains the sole authority.  The internal R07
        transaction may pass its one public epoch snapshot so this projection
        does not reread the same record.  The caller still supplies neither a
        slot nor a validity verdict, and no live robot pose, zero tensor or
        current-position self-reference can become the target.  Reference
        velocities are literal zero by contract.
        """

        epoch_owner = self._action_ball_full_mdp_motion_epoch_owner
        if epoch_owner is None:
            raise RuntimeError("Motion frame-0 reference requires its bound epoch")
        if self._action_ball_continuous_motion_poisoned or epoch_owner.poisoned:
            raise RuntimeError("Motion frame-0 reference owner is poisoned")
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
            from whole_body_tracking.robots.agibot_a3 import A3_UPPER_TRACKED
        except ImportError:
            import action_ball_full_mdp_epoch as epoch
            from whole_body_tracking.robots.agibot_a3 import A3_UPPER_TRACKED

        if type(epoch_owner) is not epoch.ActionEpochOwner:
            raise RuntimeError("Motion frame-0 epoch owner type differs")
        record = (
            epoch_owner.current()
            if action_epoch_snapshot is None
            else action_epoch_snapshot
        )
        if type(record) is not epoch.ActionEpochRecord:
            raise RuntimeError("Motion frame-0 epoch publication type differs")

        device = torch.device(self.device)
        n = self.num_envs
        schedule = self._action_ball_continuous_schedule_projection
        if not isinstance(schedule, MappingProxyType):
            raise RuntimeError("Motion frame-0 reference lacks its parent schedule")
        upcoming_action_slot = schedule.get("upcoming_action_slot")
        upcoming_action_uid = schedule.get("upcoming_action_uid")
        motion_clip_id = self._action_ball_full_mdp_motion_exact_tensor(
            self.clip_id,
            name="frame0.action_slot",
            device=device,
            dtype=torch.int64,
            shape=(n,),
        )
        current_slot = self._action_ball_full_mdp_motion_exact_tensor(
            record.current_task_slot,
            name="frame0.epoch.current_task_slot",
            device=device,
            dtype=torch.int64,
            shape=(n,),
        )
        code_owned_action_uids = (
            self._action_ball_continuous_code_owned_action_uids()
        )
        if tuple(self.cfg.body_names)[0] != self.robot.body_names[0]:
            raise RuntimeError("Motion frame-0 action/root binding differs")
        ordered_body_names = tuple(A3_UPPER_TRACKED)
        configured_body_names = tuple(self.cfg.body_names)
        if (
            not ordered_body_names
            or len(set(ordered_body_names)) != len(ordered_body_names)
            or any(name not in configured_body_names for name in ordered_body_names)
        ):
            raise RuntimeError("Motion frame-0 A3 upper-body order is unavailable")
        body_slots = torch.tensor(
            [configured_body_names.index(name) for name in ordered_body_names],
            dtype=torch.int64,
            device=device,
        )
        action_uids = torch.as_tensor(
            code_owned_action_uids,
            dtype=torch.int64,
            device=device,
        )
        if (
            tuple(action_uids.shape) != (self.motion.num_segments,)
            or action_uids.device != device
            or action_uids.dtype != torch.int64
        ):
            raise RuntimeError("Motion frame-0 action UID table differs")

        if epoch_owner.shot_slot_capacity != 1:
            raise RuntimeError("Motion frame-0 reference requires one action slot")
        current_slot_valid = current_slot.eq(0)
        lifecycle = record.recovery_reference_lifecycle_masks()
        if type(lifecycle) is not epoch.RecoveryReferenceLifecycleMasks:
            raise RuntimeError("Motion frame-0 epoch lifecycle type differs")
        lifecycle_upcoming = self._action_ball_full_mdp_motion_exact_tensor(
            lifecycle.upcoming,
            name="frame0.epoch.lifecycle.upcoming",
            device=device,
            dtype=torch.bool,
            shape=(n, epoch_owner.shot_slot_capacity),
        )
        lifecycle_completed = self._action_ball_full_mdp_motion_exact_tensor(
            lifecycle.completed,
            name="frame0.epoch.lifecycle.completed",
            device=device,
            dtype=torch.bool,
            shape=(n, epoch_owner.shot_slot_capacity),
        )
        lifecycle_disjoint = ~(lifecycle_upcoming & lifecycle_completed)
        upcoming_reference = lifecycle_upcoming[:, 0]
        completed_reference = lifecycle_completed[:, 0]
        selected_phase = record.phase[:, 0]
        selected = record.selected_mask[:, 0]
        row_identity = _ACTION_BALL_ROW_IDENTITY
        public_key = row_identity.require_action_epoch_shot_key(
            record.identity.shot_key,
            shape=(n, 1),
            device=device,
            label="Motion frame-0 public shot_key",
        )
        public_key_valid = row_identity.action_epoch_shot_key_valid(public_key)[
            :, 0
        ]
        public_key_neutral = public_key.reset_generation[:, 0].eq(-1)
        for key_field in fields(row_identity.ActionEpochShotKey)[1:]:
            public_key_neutral = public_key_neutral & getattr(
                public_key, key_field.name
            )[:, 0].eq(-1)
        empty_key = row_identity.empty_action_epoch_shot_key(
            (n,), device=device
        )
        reference_shot_key = row_identity.ActionEpochShotKey(
            **{
                key_field.name: torch.where(
                    completed_reference,
                    getattr(public_key, key_field.name)[:, 0],
                    getattr(empty_key, key_field.name),
                ).contiguous()
                for key_field in fields(row_identity.ActionEpochShotKey)
            }
        )
        epoch_action_slot = public_key.action_slot[:, 0]
        selected_action_slot = torch.where(
            upcoming_reference,
            motion_clip_id,
            torch.where(
                completed_reference,
                epoch_action_slot,
                torch.full_like(epoch_action_slot, -1),
            ),
        )
        action_slot_valid = (
            selected_action_slot.ge(0)
            & selected_action_slot.lt(self.motion.num_segments)
        )
        safe_action_slot = torch.clamp(
            selected_action_slot,
            min=0,
            max=self.motion.num_segments - 1,
        )
        safe_motion_clip_id = torch.clamp(
            motion_clip_id,
            min=0,
            max=self.motion.num_segments - 1,
        )
        starts = self.motion.seg_start[safe_action_slot]
        root_position = (
            self.motion.body_pos_w[starts, 0]
            + self._env.scene.env_origins
        )
        root_orientation = self.motion.body_quat_w[starts, 0]
        joint_position = self.motion.joint_pos[starts]
        body_position = (
            self.motion.body_pos_w[starts][:, body_slots]
            + self._env.scene.env_origins[:, None, :]
        )
        body_orientation = self.motion.body_quat_w[starts][:, body_slots]
        epoch_action_uid = public_key.action_uid[:, 0]
        selected_uid = torch.where(
            upcoming_reference,
            action_uids[safe_motion_clip_id],
            torch.where(
                completed_reference,
                epoch_action_uid,
                torch.full_like(epoch_action_uid, -1),
            ),
        )
        completed_identity_valid = (
            action_slot_valid
            & public_key_valid
            & selected_action_slot.eq(motion_clip_id)
            & selected_uid.eq(action_uids[safe_action_slot])
        )
        upcoming_schedule_identity_valid = (
            action_slot_valid
            & action_uids[safe_action_slot].gt(0)
            & motion_clip_id.eq(upcoming_action_slot)
            & selected_action_slot.eq(upcoming_action_slot)
            & action_uids[safe_action_slot].eq(upcoming_action_uid)
        )
        bootstrap_owner_valid = (
            current_slot_valid
            & upcoming_reference
            & lifecycle_disjoint[:, 0]
            & ~selected
            & selected_phase.eq(epoch.PHASE_IDLE)
            & public_key_neutral
            & upcoming_schedule_identity_valid
        )
        completed_owner_valid = (
            current_slot_valid
            & selected
            & completed_reference
            & lifecycle_disjoint[:, 0]
            & completed_identity_valid
        )
        owner_valid = bootstrap_owner_valid | completed_owner_valid
        numeric_valid = (
            torch.isfinite(root_position).all(dim=1)
            & torch.isfinite(root_orientation).all(dim=1)
            & torch.isfinite(joint_position).all(dim=1)
            & torch.isfinite(body_position).reshape(n, -1).all(dim=1)
            & torch.isfinite(body_orientation).reshape(n, -1).all(dim=1)
        )
        validity = owner_valid & numeric_valid
        producer_fault_bits = (
            (~owner_valid).to(dtype=torch.int64)
            | ((~numeric_valid).to(dtype=torch.int64) << 1)
        )
        reference_kind = torch.where(
            upcoming_reference,
            torch.full_like(
                selected_phase,
                _ACTION_BALL_R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
            ),
            torch.where(
                completed_reference,
                torch.full_like(
                    selected_phase,
                    _ACTION_BALL_R07_REFERENCE_COMPLETED_ACTION_FRAME0,
                ),
                torch.zeros_like(selected_phase),
            ),
        )
        return ActionBallFullMdpCompletedActionFrame0Reference(
            motion_owner=self,
            epoch_owner=epoch_owner,
            epoch_version=record.version,
            cadence_tick=(
                self._action_ball_continuous_episode_step.detach().clone()
            ),
            shot_key=reference_shot_key.clone(),
            reference_kind=reference_kind.detach().clone(),
            reference_action_slot=selected_action_slot.detach().clone(),
            reference_action_uid=selected_uid.detach().clone(),
            root_position_m=root_position.detach().clone(),
            root_orientation_wxyz=root_orientation.detach().clone(),
            joint_position_rad=joint_position.detach().clone(),
            body_position_m=body_position.detach().clone(),
            body_orientation_wxyz=body_orientation.detach().clone(),
            station_anchor_xy_m=root_position[:, :2].detach().clone(),
            validity=validity.detach().clone(),
            producer_fault_bits=producer_fault_bits.detach().clone(),
        )

    def project_action_ball_full_mdp_completed_action_frame0_reference(
        self,
    ) -> ActionBallFullMdpCompletedActionFrame0Reference:
        """Compatibility name for the same owner-derived frame-0 producer."""

        return self.project_action_ball_full_mdp_recovery_ready_reference()

    @staticmethod
    def _clone_action_ball_motion_question_view(
        record: _ActionBallMotionQuestionChronologyRecord,
        *,
        motion_owner: object,
    ) -> ActionBallMotionQuestionChronologyView:
        return ActionBallMotionQuestionChronologyView(
            motion_owner=motion_owner,
            chronology_identity=record.chronology_identity,
            selected_env_index=record.selected_env_index.clone(),
            current_tick=record.current_tick.clone(),
            candidate_identity=record.candidate_identity.clone(),
            contact_tick=record.contact_tick.clone(),
            earliest_launch_tick=record.earliest_launch_tick.clone(),
            launch_tick=record.launch_tick.clone(),
            chosen_horizon_s=record.chosen_horizon_s.clone(),
            action_uid=record.action_uid.clone(),
            action_slot=record.action_slot.clone(),
            task_identity=record.task_identity.clone(),
            cadence_identity=record.cadence_identity.clone(),
            task_receipt_sha256=record.task_receipt_sha256.clone(),
            motion_task_f32=record.motion_task_f32.clone(),
            construction_reason=record.construction_reason.clone(),
            producer_fault=record.producer_fault.clone(),
        )

    def issue_action_ball_motion_question_chronology(
        self,
        *,
        selected_env_index: torch.Tensor,
        candidate_identity: torch.Tensor,
        runtime_task_receipts: tuple,
        physical_horizon_owner: object,
        physical_horizon_receipt: object,
    ) -> ActionBallMotionQuestionChronologyReceipt:
        """Quantize real task contact requirements against Physical maxima.

        Cadence deadline is intentionally not an input.  A zero complete-tick
        horizon is ordinary construction reason 12; corrupt/non-attributable
        chronology remains a producer fault.  This capability is useful for
        negative production integration today, but cannot authorize D05 while
        its child projection omits the exact fields retained here.
        """

        if not self.action_ball_continuous_motion_enabled:
            raise RuntimeError("Motion question chronology requires fresh cadence")
        runtime = self._action_ball_runtime_module_bound
        import action_ball_physical_question_device as physical

        if (
            runtime is None
            or type(selected_env_index) is not torch.Tensor
            or selected_env_index.dtype is not torch.int64
            or selected_env_index.device != torch.device(self.device)
            or selected_env_index.ndim != 1
            or selected_env_index.numel() < 1
            or not selected_env_index.is_contiguous()
            or type(runtime_task_receipts) is not tuple
            or len(runtime_task_receipts) != selected_env_index.numel()
        ):
            raise TypeError("Motion chronology requires aligned exact task rows")
        if (
            type(physical_horizon_owner) is not physical.PhysicalQuestionNumericCore
            or type(physical_horizon_receipt)
            is not physical.PhysicalQuestionHorizonReceipt
            or getattr(
                getattr(physical_horizon_owner, "project_horizon_for_test", None),
                "__self__",
                None,
            )
            is not physical_horizon_owner
            or getattr(
                getattr(physical_horizon_owner, "project_horizon_for_test", None),
                "__func__",
                None,
            )
            is not physical.PhysicalQuestionNumericCore.project_horizon_for_test
        ):
            raise TypeError("Motion chronology requires the exact Physical horizon owner")
        horizon = physical_horizon_owner.project_horizon_for_test(
            physical_horizon_receipt
        )
        if type(horizon) is not physical.PhysicalQuestionHorizonView:
            raise RuntimeError("Physical horizon owner returned a foreign view")

        candidate = horizon.candidate_identity
        max_ticks = horizon.max_feasible_motion_ticks
        reason = horizon.construction_reason
        physical_fault = horizon.producer_fault
        k = selected_env_index.numel()
        if (
            type(candidate_identity) is not torch.Tensor
            or candidate_identity.dtype is not torch.int64
            or candidate_identity.device != torch.device(self.device)
            or not candidate_identity.is_contiguous()
            or candidate_identity.ndim != 2
            or candidate_identity.shape[0] != k
            or type(candidate) is not torch.Tensor
            or candidate.dtype is not torch.int64
            or candidate.device != torch.device(self.device)
            or tuple(candidate.shape) != tuple(candidate_identity.shape)
            or type(max_ticks) is not torch.Tensor
            or max_ticks.dtype is not torch.int64
            or max_ticks.device != torch.device(self.device)
            or tuple(max_ticks.shape) != tuple(candidate.shape)
            or type(reason) is not torch.Tensor
            or reason.dtype is not torch.int64
            or tuple(reason.shape) != tuple(candidate.shape)
            or type(physical_fault) is not torch.Tensor
            or physical_fault.dtype is not torch.int64
            or tuple(physical_fault.shape) != tuple(candidate.shape)
        ):
            raise TypeError("Motion chronology Physical horizon tensor ABI differs")

        step_dt = float(self._env.step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise RuntimeError("Motion chronology control dt must be finite and positive")
        expected_env = []
        contact_delta = []
        action_uid = []
        action_slot = []
        task_sha = []
        motion_task = []
        for receipt in runtime_task_receipts:
            if type(receipt) is not runtime.ActionBallTaskReceipt:
                raise TypeError("Motion chronology requires exact ActionBallTaskReceipt")
            canonical = runtime.ActionBallTaskReceipt.from_dict(receipt.to_dict())
            if canonical != receipt:
                raise RuntimeError("Motion chronology task receipt failed round-trip")
            delta = int(round(float(receipt.time_to_contact_s) / step_dt))
            if delta < 1:
                raise RuntimeError("Motion chronology contact requirement has no full tick")
            if receipt.contact_time_step_s is not None:
                if (
                    float(receipt.contact_time_step_s) != step_dt
                    or receipt.time_to_contact_tick != delta
                ):
                    raise RuntimeError("task receipt tick geometry differs from Motion")
            expected_env.append(receipt.env_id)
            contact_delta.append(delta)
            action_uid.append(receipt.action_uid)
            action_slot.append(receipt.action_slot)
            task_sha.append(list(bytes.fromhex(receipt.canonical_sha256)))
            motion_task.append(
                [
                    receipt.time_to_contact_s,
                    receipt.teacher_rate,
                    receipt.scaled_t_hit_s,
                    receipt.scaled_t_cycle_s,
                    receipt.pre_swing_wait_s,
                ]
            )

        expected_env_device = torch.tensor(
            expected_env, dtype=torch.int64, device=self.device
        )
        candidate_mismatch = candidate.ne(candidate_identity)
        row_unattributable = selected_env_index.ne(expected_env_device)
        current_tick = self._action_ball_continuous_episode_step[
            selected_env_index
        ].clone()
        delta_device = torch.tensor(
            contact_delta, dtype=torch.int64, device=self.device
        )
        max_i64 = torch.iinfo(torch.int64).max
        overflow = current_tick.gt(max_i64 - delta_device)
        contact_tick = torch.where(
            overflow,
            torch.full_like(current_tick, max_i64),
            current_tick + delta_device,
        ).contiguous()
        bounded_contact = torch.clamp(contact_tick, min=0).unsqueeze(1)
        chosen_ticks = torch.minimum(
            torch.clamp(max_ticks, min=0), bounded_contact
        ).contiguous()
        launch_tick = (contact_tick.unsqueeze(1) - chosen_ticks).contiguous()
        has_horizon = chosen_ticks.gt(0) & physical_fault.eq(0)
        construction_reason = torch.where(
            has_horizon,
            torch.full_like(reason, -1),
            torch.where(
                physical_fault.eq(0),
                torch.full_like(reason, _ACTION_BALL_MOTION_QUESTION_NO_COMPLETE_HORIZON),
                reason,
            ),
        ).contiguous()
        task_identity = self._action_ball_continuous_canonical_task_identity[
            selected_env_index
        ].clone()
        cadence_identity = self._action_ball_continuous_canonical_cadence_identity[
            selected_env_index
        ].clone()
        unattributable = (
            row_unattributable
            | task_identity.le(0)
            | cadence_identity.le(0)
            | candidate_mismatch.any(dim=1)
        ).unsqueeze(1)
        producer_fault = physical_fault.clone()
        producer_fault = torch.where(
            unattributable,
            torch.bitwise_or(
                producer_fault,
                torch.full_like(
                    producer_fault,
                    _ACTION_BALL_MOTION_QUESTION_FAULT_UNATTRIBUTABLE,
                ),
            ),
            producer_fault,
        )
        producer_fault = torch.where(
            overflow.unsqueeze(1),
            torch.bitwise_or(
                producer_fault,
                torch.full_like(
                    producer_fault,
                    _ACTION_BALL_MOTION_QUESTION_FAULT_TICK_OVERFLOW,
                ),
            ),
            producer_fault,
        ).contiguous()
        chosen_horizon_s = (
            chosen_ticks.to(torch.float32) * step_dt
        ).contiguous()
        finite_horizon = torch.isfinite(chosen_horizon_s)
        producer_fault = torch.where(
            finite_horizon,
            producer_fault,
            torch.bitwise_or(
                producer_fault,
                torch.full_like(
                    producer_fault,
                    _ACTION_BALL_MOTION_QUESTION_FAULT_NONFINITE,
                ),
            ),
        ).contiguous()

        receipt = object.__new__(ActionBallMotionQuestionChronologyReceipt)
        record = _ActionBallMotionQuestionChronologyRecord(
            chronology_identity=object(),
            physical_horizon_owner=physical_horizon_owner,
            physical_horizon_receipt=physical_horizon_receipt,
            selected_env_index=selected_env_index.clone(),
            current_tick=current_tick,
            candidate_identity=candidate.clone(),
            contact_tick=contact_tick,
            earliest_launch_tick=launch_tick.clone(),
            launch_tick=launch_tick,
            chosen_horizon_s=chosen_horizon_s,
            action_uid=torch.tensor(action_uid, dtype=torch.int64, device=self.device),
            action_slot=torch.tensor(action_slot, dtype=torch.int64, device=self.device),
            task_identity=task_identity,
            cadence_identity=cadence_identity,
            task_receipt_sha256=torch.tensor(
                task_sha, dtype=torch.uint8, device=self.device
            ).contiguous(),
            motion_task_f32=torch.tensor(
                motion_task, dtype=torch.float32, device=self.device
            ).contiguous(),
            construction_reason=construction_reason,
            producer_fault=producer_fault,
        )
        self._action_ball_motion_question_records[receipt] = record
        return receipt

    def require_owned_action_ball_motion_question_chronology(
        self,
        receipt: object,
    ) -> ActionBallMotionQuestionChronologyView:
        """Validate one Motion capability and publish clone-only exact fields."""

        if type(receipt) is not ActionBallMotionQuestionChronologyReceipt:
            raise TypeError("Motion question chronology receipt type differs")
        record = self._action_ball_motion_question_records.get(receipt)
        if record is None:
            raise RuntimeError("Motion question chronology receipt is foreign")
        # Re-projecting proves that the independently owned Physical receipt is
        # still live; no same-writer hash is accepted as a substitute.
        record.physical_horizon_owner.project_horizon_for_test(
            record.physical_horizon_receipt
        )
        return self._clone_action_ball_motion_question_view(
            record, motion_owner=self
        )

    def stage_action_ball_continuous_motion_device_r05_reveal(
        self,
        prepared_reveal: object,
        *,
        question_chronology: object,
    ) -> None:
        """Validate real Device-R05 ingress and then HOLD on its incomplete ABI."""

        owner = self._action_ball_continuous_motion_device_r05_owner
        if owner is None:
            raise RuntimeError("Motion production hot stage has no Device-R05 owner")
        child = owner.require_owned_prepared_reveal_for_child(
            prepared_reveal, owner_kind="motion"
        )
        chronology = self.require_owned_action_ball_motion_question_chronology(
            question_chronology
        )
        if (
            getattr(child, "owner_kind", None) != "motion"
            or getattr(child, "prepared_reveal", None) is not prepared_reveal
            or getattr(child, "numeric_f32", None) is None
            or chronology.motion_owner is not self
        ):
            raise RuntimeError("Motion production hot projection differs")
        missing = tuple(
            name
            for name in (
                "action_uid",
                "task_receipt_sha256",
                "cadence_receipt_sha256",
                "contact_tick",
                "launch_tick",
                "chosen_horizon_ticks",
                "physical_question_receipt_identity",
            )
            if getattr(child, name, None) is None
        )
        if missing:
            raise ActionBallMotionQuestionProductionHold(
                "Device-R05 Motion child projection lacks exact "
                + ", ".join(missing)
                + "; canonical R08 remains HOLD"
            )
        # Positive PREPARE publication is intentionally deferred until the
        # owner-issued terminal ACCEPT receipt exists.  A preview, even with
        # complete fields, is not a completion fact.
        raise ActionBallMotionQuestionProductionHold(
            "canonical PREPARE waits for the exact Device-R05 ACCEPT terminal receipt"
        )

    def bind_action_ball_continuous_motion_selected_reset(
        self,
        r05_owner: object,
        *,
        prepared_reset_validator: object,
        r05_receipt_validator: object,
        authority_source_sha256: str | None = None,
        diagnostic: bool = False,
    ) -> None:
        """Construction-bind the sole Device-R05 selected-reset authority.

        ``authority_source_sha256`` is a deprecated call-site compatibility
        argument and grants no authority.  Production admits only the exact
        Device-R05 owner and its two exact bound methods; explicit diagnostic
        fixtures remain non-authoritative test doubles.
        """

        if not self._action_ball_continuous_fresh_motion_lane_bound:
            raise RuntimeError(
                "Motion selected reset requires bound production staging"
            )
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="selected-reset bind"
        )
        import action_ball_continuous_runtime_transaction_device as device_r05

        if (
            r05_owner is None
            or not callable(prepared_reset_validator)
            or not callable(r05_receipt_validator)
            or getattr(prepared_reset_validator, "__self__", None) is not r05_owner
            or getattr(r05_receipt_validator, "__self__", None) is not r05_owner
        ):
            raise TypeError(
                "Motion selected reset requires exact reset and Device-R05 authorities"
            )
        if type(diagnostic) is not bool:
            raise TypeError("Motion selected-reset diagnostic flag must be exact bool")
        if diagnostic is False and (
            type(r05_owner) is not device_r05.DeviceR05Owner
            or getattr(prepared_reset_validator, "__func__", None)
            is not device_r05.DeviceR05Owner.require_owned_prepared_true_reset
            or getattr(r05_receipt_validator, "__func__", None)
            is not device_r05.DeviceR05Owner.require_owned_true_reset_receipt
            or self._action_ball_continuous_motion_device_r05_owner is not r05_owner
        ):
            raise RuntimeError(
                "Motion selected reset lacks its exact Device-R05 reveal binder"
            )
        del authority_source_sha256
        prepared_identity = (
            getattr(prepared_reset_validator, "__self__", None),
            getattr(prepared_reset_validator, "__func__", None),
        )
        r05_identity = (
            getattr(r05_receipt_validator, "__self__", None),
            getattr(r05_receipt_validator, "__func__", None),
        )
        current = self._action_ball_continuous_motion_selected_reset_authority
        if current is not None:
            if (
                current is not r05_owner
                or (
                    getattr(
                        self._action_ball_continuous_motion_selected_reset_prepare_validator,
                        "__self__",
                        None,
                    ),
                    getattr(
                        self._action_ball_continuous_motion_selected_reset_prepare_validator,
                        "__func__",
                        None,
                    ),
                )
                != prepared_identity
                or (
                    getattr(
                        self._action_ball_continuous_motion_selected_reset_r05_validator,
                        "__self__",
                        None,
                    ),
                    getattr(
                        self._action_ball_continuous_motion_selected_reset_r05_validator,
                        "__func__",
                        None,
                    ),
                )
                != r05_identity
                or self._action_ball_continuous_motion_selected_reset_diagnostic
                is not diagnostic
            ):
                raise RuntimeError(
                    "Motion selected-reset authority may not be rebound"
                )
            return
        self._action_ball_continuous_motion_selected_reset_authority = r05_owner
        self._action_ball_continuous_motion_selected_reset_r05_owner = r05_owner
        self._action_ball_continuous_motion_selected_reset_prepare_validator = (
            prepared_reset_validator
        )
        self._action_ball_continuous_motion_selected_reset_r05_validator = (
            r05_receipt_validator
        )
        self._action_ball_continuous_motion_selected_reset_authority_api_sha256 = (
            None
        )
        self._action_ball_continuous_motion_selected_reset_diagnostic = diagnostic
        self._action_ball_continuous_motion_selected_reset_owner_nonce = object()

    def _clear_action_ball_continuous_motion_selected_reset(self) -> None:
        self._action_ball_continuous_motion_selected_reset_stage = None
        self._action_ball_continuous_motion_selected_reset_prepared_true_reset = None
        self._action_ball_continuous_motion_selected_reset_selected_mask = None
        self._action_ball_continuous_motion_selected_reset_generation_before = None
        self._action_ball_continuous_motion_selected_reset_generation_after = None
        self._action_ball_continuous_motion_selected_reset_generation_overflow_fault = (
            None
        )
        self._action_ball_continuous_motion_selected_reset_prevalidated = None
        self._action_ball_continuous_motion_selected_reset_swaps = None
        self._action_ball_continuous_motion_selected_reset_version_after = None
        self._action_ball_continuous_motion_selected_reset_terminal_token = None
        self._action_ball_continuous_motion_selected_reset_committed = False

    def prepare_action_ball_continuous_motion_selected_reset(
        self, prepared_true_reset: object
    ) -> ActionBallContinuousMotionSelectedResetStage:
        """Mint a reset stage without changing live Motion state."""

        self._require_action_ball_continuous_motion_leaf_idle(
            operation="selected-reset prepare"
        )
        validator = (
            self._action_ball_continuous_motion_selected_reset_prepare_validator
        )
        if validator is None:
            raise RuntimeError("Motion selected-reset authority is not bound")
        try:
            claim = validator(prepared_true_reset, owner_kind="motion")
        except Exception as exc:
            raise RuntimeError(
                "Motion selected-reset Device-R05 prepare is not owner-issued"
            ) from exc
        selected_mask = getattr(claim, "selected_mask", None)
        generation_before = getattr(claim, "generation_before", None)
        generation_after = getattr(claim, "generation_after", None)
        generation_overflow_fault = getattr(
            claim, "generation_overflow_fault", None
        )
        if (
            getattr(claim, "prepared_true_reset", None)
            is not prepared_true_reset
            or getattr(claim, "owner_kind", None) != "motion"
            or getattr(claim, "prepared_identity", None) is None
            or getattr(claim, "reset_event_identity", None) is None
            or not torch.is_tensor(selected_mask)
            or selected_mask.dtype != torch.bool
            or selected_mask.device != torch.device(self.device)
            or tuple(selected_mask.shape) != (self.num_envs,)
            or not torch.is_tensor(generation_before)
            or not torch.is_tensor(generation_after)
            or generation_before.dtype != torch.int64
            or generation_after.dtype != torch.int64
            or generation_before.device != torch.device(self.device)
            or generation_after.device != torch.device(self.device)
            or tuple(generation_before.shape) != (self.num_envs,)
            or tuple(generation_after.shape) != (self.num_envs,)
            or not torch.is_tensor(generation_overflow_fault)
            or generation_overflow_fault.dtype != torch.bool
            or generation_overflow_fault.device != torch.device(self.device)
            or tuple(generation_overflow_fault.shape) != (self.num_envs,)
        ):
            raise RuntimeError(
                "Motion selected-reset selection claim differs"
            )
        # Device-R05 publishes clone-only tensors and the production top owner
        # has already joined them against Env and ActionEpoch in its sole
        # packed preflight.  Snapshot them again so a diagnostic validator
        # cannot mutate Motion's private lease after returning.  Do not use
        # Tensor._version as authority here: inference tensors deliberately
        # have no version counter, and an identity/version receipt would only
        # prove same-writer object stability rather than the reset fact.  Arm
        # below independently re-derives the after-generation from Motion's
        # own live generation and records any mismatch in the global drain.
        selected_mask = selected_mask.detach().clone()
        generation_before = generation_before.detach().clone()
        generation_after = generation_after.detach().clone()
        generation_overflow_fault = (
            generation_overflow_fault.detach().clone()
        )
        serial = self._action_ball_continuous_motion_selected_reset_next_serial
        payload = {
            "schema_version": 1,
            "kind": "action_ball_continuous_motion_selected_reset_stage_v1",
            "serial": serial,
            "owner_mutation_version": (
                self._action_ball_continuous_motion_mutation_version
            ),
        }
        stage = ActionBallContinuousMotionSelectedResetStage(
            _owner_nonce=(
                self._action_ball_continuous_motion_selected_reset_owner_nonce
            ),
            serial=serial,
            owner_mutation_version=(
                self._action_ball_continuous_motion_mutation_version
            ),
            stage_sha256=hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        )
        self._action_ball_continuous_motion_selected_reset_next_serial = serial + 1
        self._action_ball_continuous_motion_selected_reset_stage = stage
        self._action_ball_continuous_motion_selected_reset_prepared_true_reset = (
            prepared_true_reset
        )
        self._action_ball_continuous_motion_selected_reset_selected_mask = (
            selected_mask
        )
        self._action_ball_continuous_motion_selected_reset_generation_before = (
            generation_before
        )
        self._action_ball_continuous_motion_selected_reset_generation_after = (
            generation_after
        )
        self._action_ball_continuous_motion_selected_reset_generation_overflow_fault = (
            generation_overflow_fault
        )
        return stage

    def arm_prevalidated_action_ball_continuous_motion_selected_reset(
        self, stage: ActionBallContinuousMotionSelectedResetStage
    ) -> ActionBallContinuousMotionSelectedResetPrevalidated:
        """Finish every fallible check and materialize device after-images."""

        active = self._action_ball_continuous_motion_selected_reset_stage
        selected = (
            self._action_ball_continuous_motion_selected_reset_selected_mask
        )
        generation_before = (
            self._action_ball_continuous_motion_selected_reset_generation_before
        )
        generation_after = (
            self._action_ball_continuous_motion_selected_reset_generation_after
        )
        generation_overflow_fault = (
            self._action_ball_continuous_motion_selected_reset_generation_overflow_fault
        )
        if (
            self._action_ball_continuous_motion_poisoned
            or type(stage) is not ActionBallContinuousMotionSelectedResetStage
            or stage is not active
            or stage._owner_nonce
            is not self._action_ball_continuous_motion_selected_reset_owner_nonce
            or self._action_ball_continuous_motion_selected_reset_prevalidated
            is not None
            or self._action_ball_continuous_motion_selected_reset_terminal_token
            is not None
            or self._action_ball_continuous_motion_mutation_version
            != stage.owner_mutation_version
        ):
            raise RuntimeError(
                "Motion selected-reset stage is forged or stale"
            )
        schedule = self._action_ball_continuous_schedule_projection
        if schedule is None:
            raise RuntimeError(
                "Motion selected reset requires the frozen cadence projection"
            )
        origin = int(schedule["sequence_origin_step"])
        first_reveal = int(schedule["first_reveal_step"])
        ready_steps = self.motion.seg_start[self.clip_id]
        ready_pending = getattr(
            self, "_action_ball_safe_ready_reference_pending", None
        )
        if (
            not torch.is_tensor(ready_pending)
            or ready_pending.dtype != torch.bool
            or tuple(ready_pending.shape) != (self.num_envs,)
            or ready_pending.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "Motion selected reset requires its reset-ready capture state"
            )
        # Device validation is evidence for the sole global PPO drain, never
        # asynchronous authorization for the synchronous copies below.  A
        # bad generation claim (including MAX) must still settle every
        # selected Motion row to its safe tombstone so no stale task remains
        # reachable.  Compute the only non-wrapping after-image locally:
        # selected MAX stays MAX, while selected rows with room advance once.
        expected_overflow = selected & (
            generation_before == self._ACTION_BALL_INT64_MAX
        )
        safe_increment = (
            selected & ~expected_overflow
        ).to(dtype=torch.int64)
        safe_generation_after = generation_before + safe_increment
        validation_ok = (
            torch.any(selected)
            & torch.all(
                generation_before == self._action_ball_reset_generation
            )
            & torch.all(
                generation_overflow_fault == expected_overflow
            )
            & torch.all(generation_after == safe_generation_after)
        )
        validation_fault = (
            (~validation_ok) | torch.any(generation_overflow_fault)
        ).to(dtype=torch.int64).reshape(1)
        fault_count = self._action_ball_continuous_motion_fault_count_device
        if (
            not torch.is_tensor(fault_count)
            or fault_count.dtype != torch.int64
            or fault_count.device != torch.device(self.device)
            or tuple(fault_count.shape) != (1,)
        ):
            raise RuntimeError(
                "Motion selected-reset device fault counter differs"
            )
        fault_count.bitwise_or_(validation_fault)
        # Even a clean validation is a device fact until the exact global
        # drain ACKs it.  Portable checkpoint materialization may not promote
        # that unobserved fact into fresh-process truth.
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True

        def full(value: torch.Tensor, fill_value) -> torch.Tensor:
            return torch.full_like(value, fill_value)

        replacements = (
            (self._action_ball_continuous_sequence_active, full(self._action_ball_continuous_sequence_active, True)),
            (self._action_ball_continuous_episode_step, full(self._action_ball_continuous_episode_step, origin - 1)),
            (self._action_ball_continuous_scheduled_ordinal, full(self._action_ball_continuous_scheduled_ordinal, -1)),
            (self._action_ball_continuous_current_reveal_step, full(self._action_ball_continuous_current_reveal_step, -1)),
            (self._action_ball_continuous_current_deadline_step, full(self._action_ball_continuous_current_deadline_step, -1)),
            (self._action_ball_continuous_next_reveal_step, full(self._action_ball_continuous_next_reveal_step, first_reveal)),
            (self._action_ball_continuous_last_closed_ordinal, full(self._action_ball_continuous_last_closed_ordinal, -1)),
            (self._action_ball_continuous_opportunities_consumed, full(self._action_ball_continuous_opportunities_consumed, 0)),
            (self._action_ball_continuous_policy_opportunities_created, full(self._action_ball_continuous_policy_opportunities_created, 0)),
            (self._action_ball_continuous_infrastructure_censors_consumed, full(self._action_ball_continuous_infrastructure_censors_consumed, 0)),
            (self._action_ball_continuous_current_policy_opportunity, full(self._action_ball_continuous_current_policy_opportunity, False)),
            (self._action_ball_continuous_motion_active, full(self._action_ball_continuous_motion_active, False)),
            (self._action_ball_continuous_suffix_complete, full(self._action_ball_continuous_suffix_complete, False)),
            (self._action_ball_continuous_ready_reference_active, full(self._action_ball_continuous_ready_reference_active, True)),
            (self._action_ball_continuous_ready_at_reveal, full(self._action_ball_continuous_ready_at_reveal, False)),
            (self._action_ball_continuous_reveal_due, full(self._action_ball_continuous_reveal_due, False)),
            (self._action_ball_continuous_closed_mask, full(self._action_ball_continuous_closed_mask, False)),
            (self._action_ball_continuous_close_reason, full(self._action_ball_continuous_close_reason, ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE)),
            (self._action_ball_continuous_deadline_due, full(self._action_ball_continuous_deadline_due, False)),
            (self._action_ball_continuous_recovery_unavailable, full(self._action_ball_continuous_recovery_unavailable, False)),
            (self._action_ball_continuous_task_commit_pending, full(self._action_ball_continuous_task_commit_pending, False)),
            (self._action_ball_continuous_task_commit_missed, full(self._action_ball_continuous_task_commit_missed, False)),
            (self._action_ball_continuous_task_committed, full(self._action_ball_continuous_task_committed, False)),
            (self._action_ball_continuous_motion_release_pending, full(self._action_ball_continuous_motion_release_pending, False)),
            (self._action_ball_continuous_motion_release_missed, full(self._action_ball_continuous_motion_release_missed, False)),
            (self._action_ball_continuous_phase, full(self._action_ball_continuous_phase, _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE["pre_reveal_hidden"])),
            (self._action_ball_continuous_canonical_phase, full(self._action_ball_continuous_canonical_phase, _ACTION_BALL_CONTINUOUS_CANONICAL_PHASE_CODE["recover_hidden"])),
            (self._action_ball_continuous_canonical_phase_start_tick, full(self._action_ball_continuous_canonical_phase_start_tick, origin - 1)),
            (self._action_ball_continuous_canonical_task_identity, full(self._action_ball_continuous_canonical_task_identity, -1)),
            (self._action_ball_continuous_canonical_cadence_identity, full(self._action_ball_continuous_canonical_cadence_identity, -1)),
            (self._action_ball_continuous_canonical_action_uid, full(self._action_ball_continuous_canonical_action_uid, -1)),
            (self._action_ball_continuous_canonical_shot_index, full(self._action_ball_continuous_canonical_shot_index, -1)),
            (self._action_ball_continuous_canonical_outcome_identity, full(self._action_ball_continuous_canonical_outcome_identity, -1)),
            (self._action_ball_continuous_canonical_task_receipt_sha256, full(self._action_ball_continuous_canonical_task_receipt_sha256, 0)),
            (self._action_ball_continuous_canonical_cadence_receipt_sha256, full(self._action_ball_continuous_canonical_cadence_receipt_sha256, 0)),
            (self._action_ball_continuous_canonical_candidate_identity, full(self._action_ball_continuous_canonical_candidate_identity, -1)),
            (self._action_ball_continuous_canonical_contact_tick, full(self._action_ball_continuous_canonical_contact_tick, -1)),
            (self._action_ball_continuous_canonical_launch_tick, full(self._action_ball_continuous_canonical_launch_tick, -1)),
            (self._action_ball_continuous_canonical_chosen_horizon_tick, full(self._action_ball_continuous_canonical_chosen_horizon_tick, -1)),
            (self._action_ball_continuous_canonical_task_close_tick, full(self._action_ball_continuous_canonical_task_close_tick, -1)),
            (self._action_ball_continuous_canonical_task_valid, full(self._action_ball_continuous_canonical_task_valid, False)),
            (self._action_ball_continuous_canonical_playback_started, full(self._action_ball_continuous_canonical_playback_started, False)),
            (self._action_ball_task_timing_active, full(self._action_ball_task_timing_active, False)),
            (self._action_ball_task_pending_elapsed_s, full(self._action_ball_task_pending_elapsed_s, 0.0)),
            (self._action_ball_task_age_s, full(self._action_ball_task_age_s, 0.0)),
            (self._action_ball_time_to_contact_s, full(self._action_ball_time_to_contact_s, 0.0)),
            (self._action_ball_teacher_rate, full(self._action_ball_teacher_rate, 0.0)),
            (self._action_ball_scaled_t_hit_s, full(self._action_ball_scaled_t_hit_s, 0.0)),
            (self._action_ball_scaled_t_cycle_s, full(self._action_ball_scaled_t_cycle_s, 0.0)),
            (self._action_ball_pre_swing_wait_s, full(self._action_ball_pre_swing_wait_s, 0.0)),
            (self._action_ball_continuous_motion_reset_pending, full(self._action_ball_continuous_motion_reset_pending, True)),
            (self._action_ball_reset_generation, safe_generation_after),
            (self._action_ball_swing_generation, full(self._action_ball_swing_generation, 0)),
            (self.time_steps, ready_steps),
            (self.time_steps_f, ready_steps.to(dtype=self.time_steps_f.dtype)),
            (self.speed_scale, full(self.speed_scale, 0.0)),
            (self.hold_counter, full(self.hold_counter, 1)),
            (ready_pending, ready_pending | selected),
        )
        if "in_hold" in self.metrics:
            replacements = (
                *replacements,
                (self.metrics["in_hold"], full(self.metrics["in_hold"], 1.0)),
            )
        swaps = tuple(
            (
                destination,
                torch.where(
                    selected.reshape(
                        (self.num_envs,)
                        + (1,) * (destination.ndim - 1)
                    ),
                    replacement,
                    destination,
                ),
            )
            for destination, replacement in replacements
        )
        swaps = (
            *swaps,
            (
                self._action_ball_continuous_motion_device_mutation_version,
                torch.full_like(
                    self._action_ball_continuous_motion_device_mutation_version,
                    stage.owner_mutation_version + 1,
                ),
            ),
        )
        token = ActionBallContinuousMotionSelectedResetPrevalidated(
            _owner_nonce=(
                self._action_ball_continuous_motion_selected_reset_owner_nonce
            ),
            serial=stage.serial,
            stage_sha256=stage.stage_sha256,
        )
        self._action_ball_continuous_motion_selected_reset_swaps = swaps
        self._action_ball_continuous_motion_selected_reset_version_after = (
            stage.owner_mutation_version + 1
        )
        self._action_ball_continuous_motion_selected_reset_prevalidated = token
        return token

    def abort_prevalidated_action_ball_continuous_motion_selected_reset(
        self,
        value: (
            ActionBallContinuousMotionSelectedResetStage
            | ActionBallContinuousMotionSelectedResetPrevalidated
        ),
    ) -> None:
        """Drop an exact precommit reset lease; live Motion bytes stay intact."""

        stage = self._action_ball_continuous_motion_selected_reset_stage
        prevalidated = (
            self._action_ball_continuous_motion_selected_reset_prevalidated
        )
        if (
            self._action_ball_continuous_motion_poisoned
            or stage is None
            or not (value is stage or value is prevalidated)
            or self._action_ball_continuous_motion_selected_reset_committed
            or self._action_ball_continuous_motion_selected_reset_terminal_token
            is not None
        ):
            raise RuntimeError(
                "Motion selected-reset abort requires its exact precommit handle"
            )
        self._clear_action_ball_continuous_motion_selected_reset()

    def commit_prevalidated_action_ball_continuous_motion_selected_reset(
        self,
        prevalidated: ActionBallContinuousMotionSelectedResetPrevalidated,
    ) -> ActionBallContinuousMotionSelectedResetChildTerminalToken:
        """Pure device copy; retain the reset debt until R05-last ACK."""

        stage = self._action_ball_continuous_motion_selected_reset_stage
        active = self._action_ball_continuous_motion_selected_reset_prevalidated
        swaps = self._action_ball_continuous_motion_selected_reset_swaps
        version_after = (
            self._action_ball_continuous_motion_selected_reset_version_after
        )
        if (
            self._action_ball_continuous_motion_poisoned
            or type(prevalidated)
            is not ActionBallContinuousMotionSelectedResetPrevalidated
            or prevalidated is not active
            or stage is None
            or prevalidated._owner_nonce
            is not self._action_ball_continuous_motion_selected_reset_owner_nonce
            or prevalidated.serial != stage.serial
            or prevalidated.stage_sha256 != stage.stage_sha256
            or self._action_ball_continuous_motion_selected_reset_committed
            or not isinstance(swaps, tuple)
            or version_after != stage.owner_mutation_version + 1
        ):
            self.poison_global_reveal_epoch(
                "motion_selected_reset_commit_handle_invalid"
            )
            raise RuntimeError(
                "Motion selected-reset prevalidated handle is forged or stale"
            )
        # The private after-images cannot escape between arm and this immediate
        # opaque-handle commit.  The commit phase intentionally contains no
        # fallible same-writer seal; it is only device copies.
        terminal = ActionBallContinuousMotionSelectedResetChildTerminalToken(
            _owner_nonce=(
                self._action_ball_continuous_motion_selected_reset_owner_nonce
            ),
            serial=stage.serial,
            stage_sha256=stage.stage_sha256,
        )
        try:
            for destination, after_image in swaps:
                destination.copy_(after_image)
            # This host value is only a conservative hot-path work flag; the
            # device ``ready_pending`` mask above remains the sole row truth.
            # Avoid a selected-id readback merely to compute an exact count.
            self._action_ball_safe_ready_pending_count = self.num_envs
            self._action_ball_continuous_motion_mutation_version = version_after
            self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True
            self._action_ball_continuous_prepared_task_commit = None
            self._action_ball_continuous_prepared_task_commit_receipts = None
            self._action_ball_continuous_current_projection = None
            self._action_ball_continuous_published_common_step = None
            # R07 readiness is one whole-batch next-policy-tick capability.
            # A selected reset changes Motion chronology after the preceding
            # post-physics publication, so no row of that mixed-epoch handle
            # remains consumable.  The independent R07 producer installs a
            # fresh handle after the next real post-physics boundary.
            self._action_ball_continuous_r07_ready_projection = None
            self._invalidate_action_ball_continuous_observation_publication()
            self._action_ball_continuous_motion_selected_reset_terminal_token = (
                terminal
            )
            self._action_ball_continuous_motion_selected_reset_committed = True
            return terminal
        except BaseException:
            self.poison_global_reveal_epoch(
                "motion_selected_reset_device_commit_failed"
            )
            raise

    def complete_action_ball_continuous_motion_selected_reset_after_r05(
        self,
        child_terminal_token: ActionBallContinuousMotionSelectedResetChildTerminalToken,
        r05_reset_receipt: object,
    ) -> ActionBallContinuousMotionSelectedResetCompletionToken:
        """Validate Device-R05-last exactly, then mint one opaque child ACK."""

        stage = self._action_ball_continuous_motion_selected_reset_stage
        retained = (
            self._action_ball_continuous_motion_selected_reset_terminal_token
        )
        validator = self._action_ball_continuous_motion_selected_reset_r05_validator
        if (
            self._action_ball_continuous_motion_poisoned
            or not self._action_ball_continuous_motion_selected_reset_committed
            or type(child_terminal_token)
            is not ActionBallContinuousMotionSelectedResetChildTerminalToken
            or child_terminal_token is not retained
            or stage is None
            or child_terminal_token._owner_nonce
            is not self._action_ball_continuous_motion_selected_reset_owner_nonce
            or child_terminal_token.serial != stage.serial
            or child_terminal_token.stage_sha256 != stage.stage_sha256
            or validator is None
        ):
            self.poison_global_reveal_epoch(
                "motion_selected_reset_completion_handle_invalid"
            )
            raise RuntimeError(
                "Motion selected-reset completion is stale or duplicated"
            )
        try:
            owned = validator(
                r05_reset_receipt,
                expected_prepared_true_reset=(
                    self._action_ball_continuous_motion_selected_reset_prepared_true_reset
                ),
            )
        except Exception as exc:
            self.poison_global_reveal_epoch(
                "motion_selected_reset_r05_ack_invalid"
            )
            raise RuntimeError(
                "Motion selected-reset R05 receipt is not owner-issued"
            ) from exc
        if owned is not r05_reset_receipt:
            self.poison_global_reveal_epoch(
                "motion_selected_reset_r05_ack_differs"
            )
            raise RuntimeError(
                "Motion selected-reset R05 acknowledgement differs"
            )
        completion = ActionBallContinuousMotionSelectedResetCompletionToken(
            _owner_nonce=(
                self._action_ball_continuous_motion_selected_reset_owner_nonce
            ),
            serial=stage.serial,
            stage_sha256=stage.stage_sha256,
        )
        prepared_true_reset = (
            self._action_ball_continuous_motion_selected_reset_prepared_true_reset
        )
        self._clear_action_ball_continuous_motion_selected_reset()
        self._action_ball_continuous_motion_selected_reset_completion_token = (
            completion
        )
        self._action_ball_continuous_motion_selected_reset_completion_prepared = (
            prepared_true_reset
        )
        return completion

    def require_owned_selected_reset_commit(
        self,
        commit_token: object,
        *,
        expected_prepared_true_reset: object,
    ) -> ActionBallContinuousMotionSelectedResetChildTerminalToken:
        """Repeatably validate the exact retained pre-R05 Motion commit."""

        retained = (
            self._action_ball_continuous_motion_selected_reset_terminal_token
        )
        if (
            self._action_ball_continuous_motion_poisoned
            or not self._action_ball_continuous_motion_selected_reset_committed
            or type(commit_token)
            is not ActionBallContinuousMotionSelectedResetChildTerminalToken
            or commit_token is not retained
            or commit_token._owner_nonce
            is not self._action_ball_continuous_motion_selected_reset_owner_nonce
            or expected_prepared_true_reset
            is not self._action_ball_continuous_motion_selected_reset_prepared_true_reset
        ):
            raise RuntimeError(
                "Motion selected-reset commit token is stale or foreign"
            )
        return commit_token

    def require_owned_selected_reset_completion(
        self,
        completion: object,
        *,
        expected_prepared_true_reset: object,
    ) -> ActionBallContinuousMotionSelectedResetCompletionToken:
        """Top-owner validation of the latest exact opaque Motion ACK."""

        retained = (
            self._action_ball_continuous_motion_selected_reset_completion_token
        )
        if (
            self._action_ball_continuous_motion_poisoned
            or type(completion)
            is not ActionBallContinuousMotionSelectedResetCompletionToken
            or completion is not retained
            or completion._owner_nonce
            is not self._action_ball_continuous_motion_selected_reset_owner_nonce
            or expected_prepared_true_reset
            is not getattr(
                self,
                "_action_ball_continuous_motion_selected_reset_completion_prepared",
                None,
            )
        ):
            raise RuntimeError(
                "Motion selected-reset completion token is stale or foreign"
            )
        return completion

    def consume_owned_selected_reset_completion(
        self,
        completion: object,
        *,
        expected_prepared_true_reset: object,
    ) -> ActionBallContinuousMotionSelectedResetCompletionToken:
        """Let the top owner consume the exact Motion ACK once."""

        owned = self.require_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=expected_prepared_true_reset,
        )
        self._action_ball_continuous_motion_selected_reset_completion_token = None
        self._action_ball_continuous_motion_selected_reset_completion_prepared = (
            None
        )
        return owned

    # Top-owner-neutral aliases shared by all four selected-reset leaves.
    prepare_selected_reset = prepare_action_ball_continuous_motion_selected_reset
    arm_prevalidated_selected_reset = (
        arm_prevalidated_action_ball_continuous_motion_selected_reset
    )
    commit_prevalidated_selected_reset = (
        commit_prevalidated_action_ball_continuous_motion_selected_reset
    )
    abort_prevalidated_selected_reset = (
        abort_prevalidated_action_ball_continuous_motion_selected_reset
    )
    complete_selected_reset_after_r05 = (
        complete_action_ball_continuous_motion_selected_reset_after_r05
    )

    def action_ball_continuous_motion_boundary_child_token_authority(self):
        """Return the boundary capability backed by Motion's retained token."""

        if not self._action_ball_continuous_fresh_motion_lane_bound:
            raise RuntimeError(
                "continuous Motion child authority requires bound R05 staging"
            )
        if _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256 is None:
            raise RuntimeError(
                "continuous Motion reveal-boundary final source pin is pending"
            )
        import action_ball_full_mdp_reveal_boundary as boundary

        authority = self._action_ball_continuous_motion_child_token_authority
        if authority is None:
            authority = (
                boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority(
                    owner_kind="motion",
                    validator=(
                        self._require_action_ball_continuous_motion_prearmed_claim
                    ),
                )
            )
            self._action_ball_continuous_motion_child_token_authority = (
                authority
            )
        elif (
            type(authority)
            is not boundary.ActionBallFullMdpRevealBoundaryChildTokenAuthority
            or authority.owner_kind != "motion"
        ):
            raise RuntimeError(
                "continuous Motion child token authority drifted"
            )
        return authority

    def action_ball_continuous_motion_reveal_boundary_fault_schema(self):
        """Return Motion's one retained, frozen global-boundary schema."""

        if not self._action_ball_continuous_fresh_motion_lane_bound:
            raise RuntimeError(
                "continuous Motion fault schema requires bound R05 staging"
            )
        if _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256 is None:
            raise RuntimeError(
                "continuous Motion reveal-boundary final source pin is pending"
            )
        import action_ball_full_mdp_reveal_boundary as boundary

        boundary_path = Path(boundary.__file__).resolve()
        if (
            hashlib.sha256(boundary_path.read_bytes()).hexdigest()
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256
            or boundary.PACKET_ROW_INTEGRITY_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256
            or boundary.RECEIPT_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ):
            raise RuntimeError(
                "continuous Motion fault schema boundary pins differ"
            )
        schema = (
            self._action_ball_continuous_motion_boundary_fault_schema
        )
        if schema is None:
            schema = boundary.ActionBallFullMdpRevealBoundaryFaultSchema(
                schema_version=1,
                owner_kind="motion",
                ordered_fault_bits=((
                    _ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,
                    _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT,
                ),),
                allowed_fault_mask=(
                    _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT
                ),
                precedence=(
                    _ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,
                ),
            )
            self._action_ball_continuous_motion_boundary_fault_schema = (
                schema
            )
        elif (
            type(schema)
            is not boundary.ActionBallFullMdpRevealBoundaryFaultSchema
            or schema.schema_version != 1
            or schema.owner_kind != "motion"
            or schema.ordered_fault_bits
            != ((
                _ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,
                _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT,
            ),)
            or schema.allowed_fault_mask
            != _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT
            or schema.precedence
            != (_ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,)
        ):
            raise RuntimeError(
                "continuous Motion retained fault schema drifted"
            )
        return schema

    def bind_action_ball_continuous_motion_reveal_boundary(
        self, boundary_owner: object
    ) -> None:
        """Bind only Motion's typed lane in the neutral packed owner."""

        import action_ball_full_mdp_reveal_boundary as boundary

        if (
            type(boundary_owner)
            is not boundary.ActionBallFullMdpRevealBoundaryOwner
        ):
            raise TypeError(
                "continuous Motion requires the exact reveal-boundary owner"
            )
        boundary_path = Path(boundary.__file__).resolve()
        source_sha256 = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
        expected_source_sha256 = (
            _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256
        )
        if expected_source_sha256 is None:
            raise RuntimeError(
                "continuous Motion reveal-boundary final source pin is pending"
            )
        if (
            boundary_path.name
            != "action_ball_full_mdp_reveal_boundary.py"
            or source_sha256 != expected_source_sha256
            or boundary.PACKET_ROW_INTEGRITY_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256
            or boundary.RECEIPT_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ):
            raise RuntimeError(
                "continuous Motion reveal-boundary source pin differs"
            )
        authority = (
            self.action_ball_continuous_motion_boundary_child_token_authority()
        )
        retained_schema = (
            self.action_ball_continuous_motion_reveal_boundary_fault_schema()
        )
        lane = boundary_owner.lane_authority("motion")
        schema = lane.fault_schema
        if (
            type(lane)
            is not boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
            or lane.owner_kind != "motion"
            or lane.child_token_authority is not authority
            or boundary_owner.lane_authority("motion") is not lane
            or schema is not retained_schema
            or boundary_owner.num_envs != self.num_envs
            or boundary_owner.device != torch.device(self.device)
            or schema.owner_kind != "motion"
            or schema.schema_version != 1
            or schema.ordered_fault_bits
            != ((
                _ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,
                _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT,
            ),)
            or schema.allowed_fault_mask
            != _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT
            or schema.precedence
            != (_ACTION_BALL_CONTINUOUS_MOTION_FAULT_NAME,)
        ):
            raise RuntimeError(
                "continuous Motion reveal-boundary lane/schema differs"
            )
        current = self._action_ball_continuous_motion_boundary_owner
        if current is not None and current is not boundary_owner:
            raise RuntimeError(
                "continuous Motion reveal-boundary owner may not be rebound"
            )
        if current is boundary_owner:
            return
        self._action_ball_continuous_motion_boundary_module = boundary
        self._action_ball_continuous_motion_boundary_owner = boundary_owner
        self._action_ball_continuous_motion_boundary_lane = lane
        self._action_ball_continuous_motion_boundary_source_sha256 = (
            source_sha256
        )

    def _action_ball_continuous_motion_boundary_binding(self):
        boundary = self._action_ball_continuous_motion_boundary_module
        owner = self._action_ball_continuous_motion_boundary_owner
        lane = self._action_ball_continuous_motion_boundary_lane
        expected_source_sha256 = (
            _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256
        )
        if (
            boundary is None
            or type(owner)
            is not boundary.ActionBallFullMdpRevealBoundaryOwner
            or type(lane)
            is not boundary.ActionBallFullMdpRevealBoundaryLaneAuthority
            or lane.owner_kind != "motion"
            or owner.lane_authority("motion") is not lane
            or owner.num_envs != self.num_envs
            or owner.device != torch.device(self.device)
            or self._action_ball_continuous_motion_boundary_source_sha256
            != expected_source_sha256
            or boundary.PACKET_ROW_INTEGRITY_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256
            or boundary.RECEIPT_SCHEMA_SHA256
            != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ):
            raise RuntimeError(
                "continuous Motion has no exact reveal-boundary lane"
            )
        return boundary, owner, lane

    def _action_ball_reject_legacy_fresh_motion_lane(
        self, operation: str
    ) -> None:
        if type(operation) is not str or not operation:
            raise TypeError("fresh continuous Motion operation must be named")
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            raise RuntimeError(
                "fresh continuous Motion lane rejects legacy "
                f"{operation}; use stage/finalize/arm/commit_prevalidated"
            )

    @staticmethod
    def _validate_action_ball_continuous_motion_suffix_host(
        *,
        episode_tick: int,
        next_reveal_tick: int,
        step_dt: float,
        timing: dict[str, float],
    ) -> None:
        gap_steps = next_reveal_tick - episode_tick
        age_s = float(timing["pending_elapsed_s"])
        cycle_total_s = float(
            timing["pre_swing_wait_s"] + timing["scaled_t_cycle_s"]
        )
        latest_pre_reveal_age_s = age_s + (gap_steps - 2) * step_dt
        if (
            gap_steps < 2
            or not math.isfinite(latest_pre_reveal_age_s)
            or latest_pre_reveal_age_s + 1.0e-12 < cycle_total_s
        ):
            raise RuntimeError(
                "continuous Motion full suffix cannot complete before the next reveal"
            )

    def stage_action_ball_continuous_motion_reveal(
        self,
        reveal_final_preview: object,
        *,
        runtime_task_receipts: tuple,
    ) -> ActionBallContinuousMotionStage:
        """Validate one exact full-K R05 preview without device readback."""

        transaction = self._action_ball_continuous_transaction_module
        owner = self._action_ball_continuous_transaction_owner
        if (
            transaction is None
            or type(owner)
            is not transaction.ContinuousRuntimeTransactionOwner
            or not self._action_ball_continuous_fresh_motion_lane_bound
        ):
            raise RuntimeError(
                "continuous Motion staging has no exact bound R05 owner"
            )
        self._action_ball_continuous_motion_boundary_binding()
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="stage"
        )
        if (
            type(reveal_final_preview)
            is not transaction.RevealFinalPreviewBatch
        ):
            raise TypeError(
                "continuous Motion staging requires exact RevealFinalPreviewBatch"
            )
        preview_root = reveal_final_preview.canonical_sha256
        try:
            private_preview = owner.require_owned_active_reveal_final_preview(
                reveal_final_preview,
                expected_reveal_final_preview_sha256=preview_root,
            )
        except Exception as exc:
            raise RuntimeError(
                "continuous Motion staging requires the exact active unarmed "
                "R05 preview lease"
            ) from exc
        if (
            private_preview is reveal_final_preview
            or private_preview.canonical_sha256 != preview_root
        ):
            raise RuntimeError(
                "continuous Motion retained R05 preview image differs"
            )
        current_step = getattr(self._env, "common_step_counter", None)
        if (
            type(current_step) is not int
            or current_step < 0
            or self._action_ball_continuous_published_common_step
            != current_step
        ):
            raise RuntimeError(
                "continuous Motion staging requires current-tick publication"
            )
        env_ids = private_preview.prepared_batch.selected_env_ids
        if (
            not env_ids
            or type(runtime_task_receipts) is not tuple
            or len(runtime_task_receipts) != len(env_ids)
        ):
            raise ValueError(
                "continuous Motion stage requires aligned immutable task receipts"
            )

        runtime = self._action_ball_runtime_module_bound
        schedule = self._action_ball_continuous_schedule_projection
        cadence_steps = int(schedule["cadence_steps"])
        step_dt = float(self._env.step_dt)
        ordinals = []
        episode_ticks = []
        reveal_ticks = []
        deadline_ticks = []
        next_reveal_ticks = []
        reset_generations = []
        swing_generations = []
        action_slots = []
        ready_rows = []
        refs = []
        receipt_roots = []
        timing_rows = []
        timing_bytes = bytearray()
        payload_rows = []
        for env_id, preview_row, receipt in zip(
            env_ids,
            private_preview.reveal_final_rows,
            runtime_task_receipts,
        ):
            prepared = preview_row.prepared_reveal
            request = prepared.request
            facts = preview_row.reveal_facts
            if type(receipt) is not runtime.ActionBallTaskReceipt:
                raise TypeError(
                    "continuous Motion stage requires exact ActionBallTaskReceipt rows"
                )
            canonical_receipt = runtime.ActionBallTaskReceipt.from_dict(
                receipt.to_dict()
            )
            if canonical_receipt != receipt:
                raise RuntimeError(
                    "continuous Motion task receipt failed exact round-trip"
                )
            ordinal = request.scheduled_ordinal
            reset_generation = request.reset_generation
            swing_generation = request.runtime_swing_generation
            reveal_tick = request.scheduled_reveal_step
            deadline_tick = request.scheduled_deadline_step
            next_reveal_tick = reveal_tick + cadence_steps
            ready = facts.ready_at_reveal
            action_slot = request.action_slot
            if (
                request.env_id != env_id
                or facts.env_id != env_id
                or facts.reset_generation != reset_generation
                or facts.scheduled_ordinal != ordinal
                or facts.runtime_swing_generation != ordinal
                or swing_generation != ordinal
                or facts.reveal_step != reveal_tick
                or facts.deadline_step != deadline_tick
                or next_reveal_tick <= deadline_tick
            ):
                raise RuntimeError(
                    "continuous Motion staged identity differs from the R05 reveal"
                )
            task_ref = receipt.task_ref()
            expected_ref = prepared.selected_task_ref.runtime_dict()
            if (
                type(task_ref) is not runtime.ActionTaskReceiptRef
                or task_ref.to_dict() != expected_ref
                or receipt.canonical_sha256
                != prepared.selected_task_ref.task_sha256
                or task_ref.env_id != env_id
                or task_ref.reset_generation != reset_generation
                or task_ref.swing_generation != ordinal
                or task_ref.action_slot != action_slot
                or task_ref.action_uid != request.action_uid
                or task_ref.birth_sha256 != request.birth_sha256
            ):
                raise RuntimeError(
                    "continuous Motion staged task authority differs from R05"
                )
            previous = self._action_ball_continuous_committed_task_refs[
                env_id
            ]
            if previous is not None and (
                previous.reset_generation == reset_generation
            ):
                if (
                    task_ref.action_uid != previous.action_uid
                    or task_ref.action_slot != previous.action_slot
                    or task_ref.birth_sha256 != previous.birth_sha256
                    or task_ref.swing_generation
                    <= previous.swing_generation
                    or task_ref.sample_sha256 == previous.sample_sha256
                    or task_ref.task_sha256 == previous.task_sha256
                ):
                    raise RuntimeError(
                        "continuous Motion successor task lineage did not advance"
                    )
            if (
                self._action_ball_segment_lengths is None
                or action_slot >= len(self._action_ball_segment_lengths)
            ):
                raise RuntimeError(
                    "continuous Motion has no construction-cached segment length"
                )
            pending_elapsed_s = 0.0 if ordinal == 0 else step_dt
            timing = self._validate_action_ball_task_ref_and_receipt_host(
                task_ref,
                receipt,
                env_id=env_id,
                reset_generation=reset_generation,
                swing_generation=swing_generation,
                action_slot=action_slot,
                segment_length=self._action_ball_segment_lengths[action_slot],
                pending_elapsed_s=pending_elapsed_s,
            )
            if ready:
                self._validate_action_ball_continuous_motion_suffix_host(
                    episode_tick=reveal_tick,
                    next_reveal_tick=next_reveal_tick,
                    step_dt=step_dt,
                    timing=timing,
                )
            packed = struct.pack(
                "<6f",
                timing["pending_elapsed_s"],
                timing["time_to_contact_s"],
                timing["teacher_rate"],
                timing["scaled_t_hit_s"],
                timing["scaled_t_cycle_s"],
                timing["pre_swing_wait_s"],
            )
            canonical_timing = tuple(struct.unpack("<6f", packed))
            timing_bytes.extend(packed)
            timing_rows.append(canonical_timing)
            ordinals.append(ordinal)
            episode_ticks.append(reveal_tick)
            reveal_ticks.append(reveal_tick)
            deadline_ticks.append(deadline_tick)
            next_reveal_ticks.append(next_reveal_tick)
            reset_generations.append(reset_generation)
            swing_generations.append(swing_generation)
            action_slots.append(action_slot)
            ready_rows.append(ready)
            refs.append(task_ref)
            receipt_roots.append(receipt.canonical_sha256)
            payload_rows.append(
                {
                    "env_id": env_id,
                    "scheduled_ordinal": ordinal,
                    "episode_tick": reveal_tick,
                    "reveal_tick": reveal_tick,
                    "deadline_tick": deadline_tick,
                    "next_reveal_tick": next_reveal_tick,
                    "reset_generation": reset_generation,
                    "swing_generation": swing_generation,
                    "action_slot": action_slot,
                    "ready_at_reveal": ready,
                    "runtime_task_ref": task_ref.to_dict(),
                    "runtime_task_receipt_sha256": receipt.canonical_sha256,
                    "timing_f32": list(canonical_timing),
                }
            )
        timing_identity = {
            "schema_version": 1,
            "kind": "action_ball_continuous_motion_timing_after_image_v1",
            "dtype": "little_endian_ieee754_binary32",
            "shape": [len(env_ids), 6],
            "field_order": [
                "pending_elapsed_s",
                "time_to_contact_s",
                "teacher_rate",
                "scaled_t_hit_s",
                "scaled_t_cycle_s",
                "pre_swing_wait_s",
            ],
        }
        timing_identity_bytes = _canonical_json_bytes(timing_identity)
        timing_f32_le = bytes(timing_bytes)
        timing_root = hashlib.sha256(
            b"action_ball_continuous_motion_timing_after_image_v1\0"
            + len(timing_identity_bytes).to_bytes(8, "little")
            + timing_identity_bytes
            + timing_f32_le
        ).hexdigest()
        serial = self._action_ball_continuous_motion_next_serial
        payload = {
            "schema_version": 1,
            "kind": "action_ball_continuous_motion_prearm_child_token_v1",
            "stage_serial": serial,
            "owner_mutation_version": (
                self._action_ball_continuous_motion_mutation_version
            ),
            "common_step": current_step,
            "reveal_final_preview_schema_version": (
                transaction.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
            ),
            "reveal_final_preview_sha256": private_preview.canonical_sha256,
            "r05_all_owner_install_root_sha256": (
                private_preview.all_owner_install_root_sha256
            ),
            "prepared_batch_sha256": (
                private_preview.prepared_batch.canonical_sha256
            ),
            "timing_after_image_sha256": timing_root,
            "rows": payload_rows,
        }
        payload_json = _canonical_json_bytes(payload)
        token_root = hashlib.sha256(payload_json).hexdigest()
        stage = ActionBallContinuousMotionStage(
            _owner_nonce=self._action_ball_continuous_motion_owner_nonce,
            serial=serial,
            owner_mutation_version=(
                self._action_ball_continuous_motion_mutation_version
            ),
            common_step=current_step,
            reveal_final_preview_schema_version=(
                transaction.RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
            ),
            reveal_final_preview_sha256=private_preview.canonical_sha256,
            all_owner_install_root_sha256=(
                private_preview.all_owner_install_root_sha256
            ),
            prepared_batch_sha256=(
                private_preview.prepared_batch.canonical_sha256
            ),
            env_ids=tuple(env_ids),
            scheduled_ordinals=tuple(ordinals),
            episode_ticks=tuple(episode_ticks),
            reveal_ticks=tuple(reveal_ticks),
            deadline_ticks=tuple(deadline_ticks),
            next_reveal_ticks=tuple(next_reveal_ticks),
            reset_generations=tuple(reset_generations),
            swing_generations=tuple(swing_generations),
            action_slots=tuple(action_slots),
            ready_at_reveal=tuple(ready_rows),
            runtime_task_refs=tuple(refs),
            runtime_task_receipts=runtime_task_receipts,
            runtime_task_receipt_sha256s=tuple(receipt_roots),
            timing_after_image_sha256=timing_root,
            motion_child_token_root_sha256=token_root,
            _timing_rows=tuple(timing_rows),
            _timing_f32_le=timing_f32_le,
            _prearm_payload_json=payload_json,
            _reveal_final_public_token=reveal_final_preview,
            _reveal_final_private_token=private_preview,
        )
        self._action_ball_continuous_motion_next_serial = serial + 1
        self._action_ball_continuous_motion_stage = stage
        return stage

    def _validate_action_ball_continuous_motion_stage(
        self, stage: ActionBallContinuousMotionStage
    ) -> None:
        transaction = self._action_ball_continuous_transaction_module
        owner = self._action_ball_continuous_transaction_owner
        lease = getattr(owner, "_active_preview", None)
        current_step = getattr(self._env, "common_step_counter", None)
        if (
            self._action_ball_continuous_motion_poisoned
            or type(stage) is not ActionBallContinuousMotionStage
            or stage is not self._action_ball_continuous_motion_stage
            or stage._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or transaction is None
            or type(owner)
            is not transaction.ContinuousRuntimeTransactionOwner
            or lease is None
            or getattr(lease, "public_token", None)
            is not stage._reveal_final_public_token
            or getattr(lease, "preview_root_sha256", None)
            != stage.reveal_final_preview_sha256
            or getattr(lease, "armed_handle", None) is not None
            or type(current_step) is not int
            or current_step != stage.common_step
            or self._action_ball_continuous_published_common_step
            != stage.common_step
            or self._action_ball_continuous_motion_mutation_version
            != stage.owner_mutation_version
            or hashlib.sha256(stage._prearm_payload_json).hexdigest()
            != stage.motion_child_token_root_sha256
        ):
            raise RuntimeError(
                "continuous Motion stage is forged, stale, or no longer abortable"
            )

    @staticmethod
    def _action_ball_continuous_motion_indexed_after_image(
        destination: torch.Tensor,
        ids: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not torch.is_tensor(destination)
            or not torch.is_tensor(ids)
            or not torch.is_tensor(selected)
            or destination.device != ids.device
            or destination.device != selected.device
            or destination.dtype != selected.dtype
            or destination.ndim < 1
            or selected.ndim != destination.ndim
            or selected.shape[0] != ids.shape[0]
            or tuple(selected.shape[1:]) != tuple(destination.shape[1:])
        ):
            raise RuntimeError(
                "continuous Motion selected after-image shape/dtype/device differs"
            )
        after = destination.detach().clone()
        after.index_copy_(0, ids, selected)
        return destination, after

    @staticmethod
    def _action_ball_continuous_motion_swap_receipts(
        swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[tuple[tuple[torch.Tensor, int], tuple[torch.Tensor, int]], ...]:
        """Seal host-only tensor identity/version receipts for one branch."""

        receipts = tuple(
            (
                _tensor_identity_version_receipt(destination),
                _tensor_identity_version_receipt(after_image),
            )
            for destination, after_image in swaps
        )
        if any(
            destination_receipt is None or after_receipt is None
            for destination_receipt, after_receipt in receipts
        ):
            raise RuntimeError(
                "continuous Motion could not seal a prearmed after-image"
            )
        return receipts

    @staticmethod
    def _action_ball_continuous_motion_swaps_match_receipts(
        swaps: object,
        receipts: object,
    ) -> bool:
        """Check retained tensor epochs without reading any device value."""

        return (
            isinstance(swaps, tuple)
            and isinstance(receipts, tuple)
            and len(swaps) == len(receipts)
            and all(
                isinstance(swap, tuple)
                and len(swap) == 2
                and isinstance(receipt, tuple)
                and len(receipt) == 2
                and _tensor_matches_identity_version_receipt(
                    swap[0], receipt[0]
                )
                and _tensor_matches_identity_version_receipt(
                    swap[1], receipt[1]
                )
                for swap, receipt in zip(swaps, receipts)
            )
        )

    def _materialize_action_ball_continuous_motion_prearm(
        self, stage: ActionBallContinuousMotionStage
    ) -> tuple[
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
        list[object],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Preconstruct ACCEPT/CENSOR swaps and the owner pass row."""

        ids = torch.tensor(
            stage.env_ids, dtype=torch.long, device=self.device
        )
        timing = torch.tensor(
            stage._timing_rows,
            dtype=self._action_ball_task_age_s.dtype,
            device=self.device,
        )
        row_count = len(stage.env_ids)
        if tuple(timing.shape) != (row_count, 6):
            raise RuntimeError(
                "continuous Motion timing after-image shape differs"
            )
        ready = torch.tensor(
            stage.ready_at_reveal, dtype=torch.bool, device=self.device
        )
        ordinals = torch.tensor(
            stage.scheduled_ordinals,
            dtype=torch.long,
            device=self.device,
        )
        episode_ticks = torch.tensor(
            stage.episode_ticks, dtype=torch.long, device=self.device
        )
        reveal_ticks = torch.tensor(
            stage.reveal_ticks, dtype=torch.long, device=self.device
        )
        deadline_ticks = torch.tensor(
            stage.deadline_ticks, dtype=torch.long, device=self.device
        )
        next_reveal_ticks = torch.tensor(
            stage.next_reveal_ticks, dtype=torch.long, device=self.device
        )
        reset_generations = torch.tensor(
            stage.reset_generations,
            dtype=torch.long,
            device=self.device,
        )
        swing_generations = torch.tensor(
            stage.swing_generations,
            dtype=torch.long,
            device=self.device,
        )
        action_slots = torch.tensor(
            stage.action_slots, dtype=torch.long, device=self.device
        )
        selected_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        selected_mask.index_fill_(0, ids, True)
        unexpected_reveal = torch.any(
            self._action_ball_continuous_reveal_due & ~selected_mask
        )
        version_expected = torch.full(
            (1,),
            stage.owner_mutation_version,
            dtype=torch.int64,
            device=self.device,
        )
        selected_chronology_integrity = (
            self._action_ball_continuous_reveal_due[ids]
            & (
                self._action_ball_continuous_episode_step[ids]
                == episode_ticks
            )
            & (
                self._action_ball_continuous_scheduled_ordinal[ids]
                == ordinals
            )
            & (
                self._action_ball_continuous_current_reveal_step[ids]
                == reveal_ticks
            )
            & (
                self._action_ball_continuous_current_deadline_step[ids]
                == deadline_ticks
            )
            & (
                self._action_ball_continuous_next_reveal_step[ids]
                == next_reveal_ticks
            )
            & (self._action_ball_reset_generation[ids] == reset_generations)
            & (self._action_ball_swing_generation[ids] == swing_generations)
        )
        selected_preflight_pass = (
            self._action_ball_continuous_task_commit_pending[ids]
            & ~self._action_ball_continuous_task_committed[ids]
            & ~self._action_ball_continuous_motion_active[ids]
            & (self.clip_id[ids] == action_slots)
            & (self._action_ball_continuous_ready_at_reveal[ids] == ready)
        )
        owner_integrity_ok = (
            torch.all(selected_chronology_integrity)
            & torch.all(
                self._action_ball_continuous_motion_device_mutation_version
                == version_expected
            )
            & ~unexpected_reveal
        )
        selected_pass = selected_preflight_pass & owner_integrity_ok
        pass_mask = torch.zeros_like(selected_mask)
        pass_mask.index_copy_(0, ids, selected_pass)
        fault_bits = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        selected_fault_bits = (
            (~selected_preflight_pass).to(dtype=torch.int64)
            * _ACTION_BALL_CONTINUOUS_MOTION_FAULT_BIT
        )
        selected_fault_bits = torch.where(
            owner_integrity_ok,
            selected_fault_bits,
            torch.full_like(
                selected_fault_bits,
                _ACTION_BALL_CONTINUOUS_MOTION_UNATTRIBUTABLE_BIT,
            ),
        )
        fault_bits.index_copy_(
            0,
            ids,
            selected_fault_bits,
        )

        false_rows = torch.zeros(
            row_count, dtype=torch.bool, device=self.device
        )
        true_rows = torch.ones(
            row_count, dtype=torch.bool, device=self.device
        )
        accept_phase = torch.where(
            ready,
            torch.full(
                (row_count,),
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "active_opportunity"
                ],
                dtype=torch.long,
                device=self.device,
            ),
            torch.full(
                (row_count,),
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "recovery_unavailable"
                ],
                dtype=torch.long,
                device=self.device,
            ),
        )
        accept_selected = (
            (self._action_ball_task_pending_elapsed_s, timing[:, 0]),
            (self._action_ball_task_age_s, timing[:, 0]),
            (self._action_ball_time_to_contact_s, timing[:, 1]),
            (self._action_ball_teacher_rate, timing[:, 2]),
            (self._action_ball_scaled_t_hit_s, timing[:, 3]),
            (self._action_ball_scaled_t_cycle_s, timing[:, 4]),
            (self._action_ball_pre_swing_wait_s, timing[:, 5]),
            (self._action_ball_task_timing_active, true_rows),
            (self._action_ball_continuous_task_commit_pending, false_rows),
            (self._action_ball_continuous_task_commit_missed, false_rows),
            (self._action_ball_continuous_task_committed, true_rows),
            (self._action_ball_continuous_motion_reset_pending, false_rows),
            (self._action_ball_continuous_motion_release_pending, false_rows),
            (self._action_ball_continuous_motion_release_missed, false_rows),
            (self._action_ball_continuous_motion_active, ready),
            (self._action_ball_continuous_suffix_complete, false_rows),
            (
                self._action_ball_continuous_ready_reference_active,
                ~ready,
            ),
            (self._action_ball_continuous_phase, accept_phase),
            (
                self._action_ball_continuous_current_policy_opportunity,
                true_rows,
            ),
            (
                self._action_ball_continuous_policy_opportunities_created,
                self._action_ball_continuous_policy_opportunities_created[
                    ids
                ]
                + 1,
            ),
        )
        accept_swaps = tuple(
            self._action_ball_continuous_motion_indexed_after_image(
                destination, ids, selected
            )
            for destination, selected in accept_selected
        )

        censor_phase = torch.full(
            (row_count,),
            _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                "infrastructure_invalid"
            ],
            dtype=torch.long,
            device=self.device,
        )
        censor_selected = (
            (self._action_ball_continuous_last_closed_ordinal, ordinals),
            (self._action_ball_task_timing_active, false_rows),
            (self._action_ball_continuous_task_commit_pending, false_rows),
            (self._action_ball_continuous_task_commit_missed, true_rows),
            (self._action_ball_continuous_task_committed, false_rows),
            (self._action_ball_continuous_motion_release_pending, false_rows),
            (self._action_ball_continuous_motion_release_missed, false_rows),
            (self._action_ball_continuous_motion_active, false_rows),
            (self._action_ball_continuous_suffix_complete, false_rows),
            (self._action_ball_continuous_phase, censor_phase),
            (
                self._action_ball_continuous_current_policy_opportunity,
                false_rows,
            ),
            (
                self._action_ball_continuous_infrastructure_censors_consumed,
                self._action_ball_continuous_infrastructure_censors_consumed[
                    ids
                ]
                + 1,
            ),
        )
        censor_swaps = tuple(
            self._action_ball_continuous_motion_indexed_after_image(
                destination, ids, selected
            )
            for destination, selected in censor_selected
        )
        version_after = (
            self._action_ball_continuous_motion_device_mutation_version
            .detach()
            .clone()
        )
        version_after.add_(1)
        version_swap = (
            self._action_ball_continuous_motion_device_mutation_version,
            version_after,
        )
        accept_swaps = (*accept_swaps, version_swap)
        censor_swaps = (*censor_swaps, version_swap)
        accept_refs = list(self._action_ball_active_task_refs)
        committed_refs = list(
            self._action_ball_continuous_committed_task_refs
        )
        for env_id, task_ref in zip(
            stage.env_ids, stage.runtime_task_refs
        ):
            accept_refs[env_id] = task_ref
            committed_refs[env_id] = task_ref
        return (
            accept_swaps,
            censor_swaps,
            [accept_refs, committed_refs],
            pass_mask,
            fault_bits,
        )

    def finalize_action_ball_continuous_motion_prearm(
        self, stage: ActionBallContinuousMotionStage
    ) -> ActionBallContinuousMotionPrearmedInstall:
        """Complete all fallible materialization before the packed transfer."""

        self._validate_action_ball_continuous_motion_stage(stage)
        _boundary, _owner, lane = (
            self._action_ball_continuous_motion_boundary_binding()
        )
        if (
            self._action_ball_continuous_motion_prearmed_install is not None
            or self._action_ball_continuous_motion_armed_install is not None
            or self._action_ball_continuous_motion_censored_install is not None
        ):
            raise RuntimeError(
                "continuous Motion already has a prearmed child"
            )
        (
            accept_swaps,
            censor_swaps,
            accept_refs,
            pass_mask,
            fault_bits,
        ) = self._materialize_action_ball_continuous_motion_prearm(stage)
        accept_swap_receipts = (
            self._action_ball_continuous_motion_swap_receipts(accept_swaps)
        )
        censor_swap_receipts = (
            self._action_ball_continuous_motion_swap_receipts(censor_swaps)
        )
        prearmed = ActionBallContinuousMotionPrearmedInstall(
            _owner_nonce=self._action_ball_continuous_motion_owner_nonce,
            serial=stage.serial,
            owner_mutation_version=stage.owner_mutation_version,
            reveal_final_preview_schema_version=(
                stage.reveal_final_preview_schema_version
            ),
            reveal_final_preview_sha256=(
                stage.reveal_final_preview_sha256
            ),
            selected_env_ids=stage.env_ids,
            canonical_sha256=stage.motion_child_token_root_sha256,
        )
        self._action_ball_continuous_motion_prearmed_accept_swaps = (
            accept_swaps
        )
        self._action_ball_continuous_motion_prearmed_censor_swaps = (
            censor_swaps
        )
        self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
            accept_swap_receipts
        )
        self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
            censor_swap_receipts
        )
        self._action_ball_continuous_motion_prearmed_accept_refs = accept_refs
        self._action_ball_continuous_motion_prearmed_install = prearmed
        try:
            row = lane.mint_device_row(
                prepared_token=prearmed,
                selected_env_ids=stage.env_ids,
                pass_mask=pass_mask,
                fault_bits=fault_bits,
            )
        except Exception:
            self._action_ball_continuous_motion_prearmed_accept_swaps = None
            self._action_ball_continuous_motion_prearmed_censor_swaps = None
            self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
                None
            )
            self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
                None
            )
            self._action_ball_continuous_motion_prearmed_accept_refs = None
            self._action_ball_continuous_motion_prearmed_install = None
            raise
        self._action_ball_continuous_motion_prearmed_boundary_row = row
        return prearmed

    def _require_action_ball_continuous_motion_prearmed_claim(
        self, prepared_token: object
    ):
        boundary, _owner, _lane = (
            self._action_ball_continuous_motion_boundary_binding()
        )
        stage = self._action_ball_continuous_motion_stage
        active = self._action_ball_continuous_motion_prearmed_install
        if (
            self._action_ball_continuous_motion_poisoned
            or type(prepared_token)
            is not ActionBallContinuousMotionPrearmedInstall
            or prepared_token is not active
            or prepared_token._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or stage is None
            or prepared_token.serial != stage.serial
            or prepared_token.owner_mutation_version
            != stage.owner_mutation_version
            or prepared_token.canonical_sha256
            != stage.motion_child_token_root_sha256
            or self._action_ball_continuous_motion_mutation_version
            != stage.owner_mutation_version
            or not torch.is_tensor(
                self._action_ball_continuous_motion_device_mutation_version
            )
        ):
            raise RuntimeError(
                "continuous Motion prepared token is forged or stale"
            )
        return boundary.ActionBallFullMdpRevealBoundaryPreparedTokenClaim(
            owner_kind="motion",
            device_owner_mutation_version=(
                self._action_ball_continuous_motion_device_mutation_version
            ),
            owner_token_root_sha256=stage.motion_child_token_root_sha256,
            reveal_final_preview_schema_version=(
                stage.reveal_final_preview_schema_version
            ),
            reveal_final_preview_sha256=(
                stage.reveal_final_preview_sha256
            ),
            _prepared_token=prepared_token,
        )

    def action_ball_continuous_motion_boundary_row(
        self, prearmed_install: ActionBallContinuousMotionPrearmedInstall
    ):
        """Return only the exact row minted from Motion's retained token."""

        boundary, _owner, lane = (
            self._action_ball_continuous_motion_boundary_binding()
        )
        active = self._action_ball_continuous_motion_prearmed_install
        row = self._action_ball_continuous_motion_prearmed_boundary_row
        stage = self._action_ball_continuous_motion_stage
        if (
            type(prearmed_install)
            is not ActionBallContinuousMotionPrearmedInstall
            or prearmed_install is not active
            or stage is None
            or self._action_ball_continuous_motion_armed_install is not None
            or self._action_ball_continuous_motion_censored_install is not None
            or type(row)
            is not boundary.ActionBallFullMdpRevealBoundaryDeviceRow
            or row.owner_kind != "motion"
            or row.owner_token_root_sha256
            != stage.motion_child_token_root_sha256
            or row.reveal_final_preview_schema_version
            != stage.reveal_final_preview_schema_version
            or row.reveal_final_preview_sha256
            != stage.reveal_final_preview_sha256
            or row.selected_env_ids != stage.env_ids
        ):
            raise RuntimeError(
                "continuous Motion prearmed boundary row is forged or stale"
            )
        return lane.require_owned_device_row(
            row, expected_prepared_token=prearmed_install
        )

    def _clear_action_ball_continuous_motion_leaf(self) -> None:
        """Drop only private capability state; live Motion bytes are untouched."""

        self._action_ball_continuous_motion_stage = None
        self._action_ball_continuous_motion_prearmed_install = None
        self._action_ball_continuous_motion_prearmed_accept_swaps = None
        self._action_ball_continuous_motion_prearmed_censor_swaps = None
        self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_accept_refs = None
        self._action_ball_continuous_motion_prearmed_boundary_row = None
        self._action_ball_continuous_motion_armed_install = None
        self._action_ball_continuous_motion_censored_install = None
        self._action_ball_continuous_motion_armed_swaps = None
        self._action_ball_continuous_motion_armed_refs = None
        self._action_ball_continuous_motion_commit_receipt = None
        self._action_ball_continuous_motion_terminal_claim = None
        self._action_ball_continuous_motion_terminal_expectations = None
        self._action_ball_continuous_motion_terminal_token = None
        self._action_ball_continuous_motion_terminal_epoch_committed = False

    def _retain_action_ball_continuous_motion_terminal_epoch(self) -> None:
        """Retire install material but retain the exact R05 completion debt."""

        self._action_ball_continuous_motion_prearmed_install = None
        self._action_ball_continuous_motion_prearmed_accept_swaps = None
        self._action_ball_continuous_motion_prearmed_censor_swaps = None
        self._action_ball_continuous_motion_prearmed_accept_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_censor_swap_receipts = (
            None
        )
        self._action_ball_continuous_motion_prearmed_accept_refs = None
        self._action_ball_continuous_motion_prearmed_boundary_row = None
        self._action_ball_continuous_motion_armed_install = None
        self._action_ball_continuous_motion_censored_install = None
        self._action_ball_continuous_motion_armed_swaps = None
        self._action_ball_continuous_motion_armed_refs = None
        self._action_ball_continuous_motion_terminal_epoch_committed = True

    def _arm_action_ball_continuous_motion_prearm(
        self,
        prearmed_install: ActionBallContinuousMotionPrearmedInstall,
        global_boundary_receipt: object,
        prepared_terminal_claim: object,
        *,
        expected_decision: str,
    ) -> (
        ActionBallContinuousMotionArmedInstall
        | ActionBallContinuousMotionCensoredInstall
    ):
        """Consume Motion's exact row through one typed terminal branch."""

        boundary, boundary_owner, lane = (
            self._action_ball_continuous_motion_boundary_binding()
        )
        stage = self._action_ball_continuous_motion_stage
        active = self._action_ball_continuous_motion_prearmed_install
        row = self._action_ball_continuous_motion_prearmed_boundary_row
        accept_swaps = (
            self._action_ball_continuous_motion_prearmed_accept_swaps
        )
        censor_swaps = (
            self._action_ball_continuous_motion_prearmed_censor_swaps
        )
        accept_swap_receipts = (
            self._action_ball_continuous_motion_prearmed_accept_swap_receipts
        )
        censor_swap_receipts = (
            self._action_ball_continuous_motion_prearmed_censor_swap_receipts
        )
        accept_refs = (
            self._action_ball_continuous_motion_prearmed_accept_refs
        )
        owner = self._action_ball_continuous_transaction_owner
        lease = getattr(owner, "_active_preview", None)
        if (
            self._action_ball_continuous_motion_armed_install is not None
            or self._action_ball_continuous_motion_censored_install is not None
            or self._action_ball_continuous_motion_commit_receipt is not None
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion boundary receipt arm was duplicated"
            )
        if (
            self._action_ball_continuous_motion_poisoned
            or type(prearmed_install)
            is not ActionBallContinuousMotionPrearmedInstall
            or prearmed_install is not active
            or prearmed_install._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or stage is None
            or prearmed_install.serial != stage.serial
            or prearmed_install.owner_mutation_version
            != stage.owner_mutation_version
            or prearmed_install.canonical_sha256
            != stage.motion_child_token_root_sha256
            or self._action_ball_continuous_motion_mutation_version
            != stage.owner_mutation_version
            or not isinstance(accept_swaps, tuple)
            or not isinstance(censor_swaps, tuple)
            or type(accept_refs) is not list
            or type(row)
            is not boundary.ActionBallFullMdpRevealBoundaryDeviceRow
            or lease is None
            or getattr(lease, "public_token", None)
            is not stage._reveal_final_public_token
            or getattr(lease, "preview_root_sha256", None)
            != stage.reveal_final_preview_sha256
            or getattr(lease, "armed_handle", None) is not None
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion prearmed leaf is forged, stale, or already armed"
            )
        if (
            type(global_boundary_receipt)
            is not boundary.ActionBallFullMdpRevealBoundaryReceipt
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion global boundary packet is malformed"
            )
        decision = global_boundary_receipt.decision
        if decision not in (
            boundary.DECISION_ACCEPT,
            boundary.DECISION_CENSOR,
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion global boundary decision is malformed"
            )
        if decision != expected_decision:
            raise RuntimeError(
                "continuous Motion boundary receipt belongs to the other typed arm branch"
            )
        selected_swaps = (
            accept_swaps
            if expected_decision == boundary.DECISION_ACCEPT
            else censor_swaps
        )
        current_step = getattr(self._env, "common_step_counter", None)
        if (
            type(current_step) is not int
            or current_step != stage.common_step
            or self._action_ball_continuous_published_common_step
            != stage.common_step
            or not self._action_ball_continuous_motion_swaps_match_receipts(
                accept_swaps,
                accept_swap_receipts,
            )
            or not self._action_ball_continuous_motion_swaps_match_receipts(
                censor_swaps,
                censor_swap_receipts,
            )
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion prearmed tensor epoch or manager tick drifted"
            )
        try:
            owned_row = boundary_owner.require_owned_owner_row(
                global_boundary_receipt,
                owner_kind="motion",
                expected_device_row=row,
                expected_prepared_token=prearmed_install,
                expected_fault_schema_sha256=(
                    lane.fault_schema.schema_sha256
                ),
                expected_reveal_final_preview_schema_version=(
                    stage.reveal_final_preview_schema_version
                ),
                expected_reveal_final_preview_sha256=(
                    stage.reveal_final_preview_sha256
                ),
                expected_selected_env_ids=stage.env_ids,
                expected_packet_sha256=(
                    global_boundary_receipt.packet_sha256
                ),
                expected_decision=expected_decision,
            )
        except Exception as exc:
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion global boundary row is malformed or stale"
            ) from exc
        if (
            owned_row.owner_kind != "motion"
            or owned_row.owner_mutation_version
            != stage.owner_mutation_version
            or owned_row.owner_token_root_sha256
            != stage.motion_child_token_root_sha256
            or owned_row.fault_schema_sha256
            != lane.fault_schema.schema_sha256
            or owned_row.allowed_fault_mask
            != lane.fault_schema.allowed_fault_mask
            or len(owned_row.selected_pass) != len(stage.env_ids)
            or len(owned_row.selected_fault_bits) != len(stage.env_ids)
            or (
                decision == boundary.DECISION_ACCEPT
                and (
                    not all(owned_row.selected_pass)
                    or any(owned_row.selected_fault_bits)
                )
            )
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion decoded owner row differs"
            )
        try:
            return self._finish_action_ball_continuous_motion_arm(
                stage=stage,
                selected_swaps=selected_swaps,
                accept_refs=accept_refs,
                expected_decision=expected_decision,
                global_boundary_receipt=global_boundary_receipt,
                prepared_terminal_claim=prepared_terminal_claim,
            )
        except Exception:
            self._action_ball_continuous_motion_poisoned = True
            raise

    def _require_action_ball_continuous_motion_terminal_claim(
        self,
        prepared_terminal_claim: object,
        *,
        global_boundary_receipt: object,
        stage: ActionBallContinuousMotionStage,
        expected_decision: str,
        require_armed: bool,
    ) -> MappingProxyType:
        """Bind or revalidate R05's exact owner-issued terminal claim."""

        transaction = self._action_ball_continuous_transaction_module
        owner = self._action_ball_continuous_transaction_owner
        boundary = self._action_ball_continuous_motion_boundary_module
        if (
            transaction is None
            or type(owner)
            is not transaction.ContinuousRuntimeTransactionOwner
            or type(prepared_terminal_claim)
            is not transaction.PreparedRevealTerminalClaim
            or expected_decision
            not in (
                boundary.DECISION_ACCEPT,
                boundary.DECISION_CENSOR,
            )
        ):
            raise RuntimeError(
                "continuous Motion R05 terminal claim type/decision differs"
            )
        if require_armed:
            expectations = (
                self._action_ball_continuous_motion_terminal_expectations
            )
            if (
                prepared_terminal_claim
                is not self._action_ball_continuous_motion_terminal_claim
                or not isinstance(expectations, MappingProxyType)
                or expectations["expected_decision"] != expected_decision
                or expectations["expected_reveal_final_preview_sha256"]
                != stage.reveal_final_preview_sha256
                or expectations["expected_selected_env_ids"]
                != stage.env_ids
            ):
                raise RuntimeError(
                    "continuous Motion retained R05 terminal claim differs"
                )
            validator = owner.require_owned_armed_terminal_claim
        else:
            if (
                type(global_boundary_receipt)
                is not boundary.ActionBallFullMdpRevealBoundaryReceipt
            ):
                raise RuntimeError(
                    "continuous Motion terminal claim lacks exact boundary receipt"
                )
            terminal_kind = (
                transaction.CommittedRevealBatch.KIND
                if expected_decision == boundary.DECISION_ACCEPT
                else transaction.CensoredRevealBatch.KIND
            )
            receipt_sha256 = self._action_ball_continuous_motion_sha256(
                global_boundary_receipt.canonical_sha256,
                label="global reveal-boundary receipt root",
            )
            packet_sha256 = self._action_ball_continuous_motion_sha256(
                global_boundary_receipt.packet_sha256,
                label="global reveal-boundary packet root",
            )
            claim_sha256 = self._action_ball_continuous_motion_sha256(
                prepared_terminal_claim.canonical_sha256,
                label="R05 prepared terminal claim root",
            )
            terminal_sha256 = self._action_ball_continuous_motion_sha256(
                prepared_terminal_claim.terminal_sha256,
                label="R05 expected terminal root",
            )
            terminal_boundary_authority_sha256 = (
                self._action_ball_continuous_motion_sha256(
                    prepared_terminal_claim.terminal_boundary_authority_sha256,
                    label="R05 terminal boundary authority root",
                )
            )
            terminal_boundary_projection_sha256 = (
                self._action_ball_continuous_motion_sha256(
                    prepared_terminal_claim.terminal_boundary_projection_sha256,
                    label="R05 terminal boundary projection root",
                )
            )
            terminal_content_pin_sha256 = (
                self._action_ball_continuous_motion_sha256(
                    prepared_terminal_claim.terminal_content_pin_sha256,
                    label="R05 terminal content pin root",
                )
            )
            terminal_projection = (
                prepared_terminal_claim.terminal_boundary_projection
            )
            terminal_content_pin = prepared_terminal_claim.terminal_content_pin
            expected_motion_participant_root = (
                _ACTION_BALL_CONTINUOUS_TERMINAL_BOUNDARY_AUTHORITY_DOMAIN,
                "motion",
                stage.motion_child_token_root_sha256,
            )
            projected_motion_participant_roots = (
                ()
                if type(terminal_projection)
                is not transaction.TerminalBoundaryProjection
                else tuple(
                    (
                        participant.participant_domain,
                        participant.participant_kind,
                        participant.participant_root_sha256,
                    )
                    for participant in (
                        terminal_projection.ordered_participant_roots
                    )
                    if (
                        participant.participant_domain
                        == terminal_projection.authority_domain
                        and participant.participant_kind == "motion"
                    )
                )
            )
            expectations = MappingProxyType(
                {
                    "expected_claim_sha256": claim_sha256,
                    "expected_decision": expected_decision,
                    "expected_reveal_final_preview_sha256": (
                        stage.reveal_final_preview_sha256
                    ),
                    "expected_global_boundary_receipt_sha256": (
                        receipt_sha256
                    ),
                    "expected_global_boundary_packet_sha256": packet_sha256,
                    "expected_terminal_boundary_authority_sha256": (
                        terminal_boundary_authority_sha256
                    ),
                    "expected_terminal_boundary_projection_sha256": (
                        terminal_boundary_projection_sha256
                    ),
                    "expected_terminal_content_pin_sha256": (
                        terminal_content_pin_sha256
                    ),
                    "expected_terminal_kind": terminal_kind,
                    "expected_terminal_sha256": terminal_sha256,
                    "expected_selected_env_ids": stage.env_ids,
                }
            )
            if (
                prepared_terminal_claim.schema_version != 1
                or prepared_terminal_claim.kind
                != transaction.PREPARED_REVEAL_TERMINAL_CLAIM_KIND
                or prepared_terminal_claim.decision != expected_decision
                or prepared_terminal_claim.selected_env_ids != stage.env_ids
                or prepared_terminal_claim.reveal_final_preview_schema_version
                != stage.reveal_final_preview_schema_version
                or prepared_terminal_claim.reveal_final_preview_sha256
                != stage.reveal_final_preview_sha256
                or prepared_terminal_claim.global_boundary_receipt_kind
                != boundary.RECEIPT_KIND
                or prepared_terminal_claim.global_boundary_receipt_sha256
                != receipt_sha256
                or prepared_terminal_claim.global_boundary_packet_schema_version
                != boundary.PACKET_SCHEMA_VERSION
                or prepared_terminal_claim.global_boundary_packet_sha256
                != packet_sha256
                or prepared_terminal_claim.terminal_boundary_authority_sha256
                != terminal_boundary_authority_sha256
                or prepared_terminal_claim.terminal_boundary_projection_sha256
                != terminal_boundary_projection_sha256
                or prepared_terminal_claim.terminal_content_pin_sha256
                != terminal_content_pin_sha256
                or type(terminal_projection)
                is not transaction.TerminalBoundaryProjection
                or terminal_projection.canonical_sha256
                != terminal_boundary_projection_sha256
                or terminal_projection.authority_domain
                != _ACTION_BALL_CONTINUOUS_TERMINAL_BOUNDARY_AUTHORITY_DOMAIN
                or terminal_projection.authority_schema_sha256
                != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
                or terminal_projection.authority_source_sha256
                != _ACTION_BALL_CONTINUOUS_REVEAL_BOUNDARY_SOURCE_SHA256
                or terminal_projection.decision_mapping_schema_version
                != transaction.TERMINAL_BOUNDARY_DECISION_MAPPING_SCHEMA_VERSION
                or terminal_projection.source_decision
                != (
                    transaction.TERMINAL_BOUNDARY_SOURCE_DECISION_PASS
                    if expected_decision == boundary.DECISION_ACCEPT
                    else transaction.TERMINAL_DECISION_CENSOR
                )
                or terminal_projection.decision != expected_decision
                or terminal_projection.reveal_final_preview_schema_version
                != stage.reveal_final_preview_schema_version
                or terminal_projection.reveal_final_preview_sha256
                != stage.reveal_final_preview_sha256
                or terminal_projection.selected_env_ids != stage.env_ids
                or terminal_projection.boundary_receipt_kind
                != boundary.RECEIPT_KIND
                or terminal_projection.boundary_receipt_sha256
                != receipt_sha256
                or terminal_projection.boundary_packet_schema_version
                != boundary.PACKET_SCHEMA_VERSION
                or terminal_projection.boundary_packet_sha256 != packet_sha256
                or projected_motion_participant_roots
                != (expected_motion_participant_root,)
                or type(terminal_content_pin)
                is not transaction.PreparedTerminalContentPin
                or terminal_content_pin.canonical_sha256
                != terminal_content_pin_sha256
                or terminal_content_pin.terminal_kind != terminal_kind
                or terminal_content_pin.terminal_canonical_sha256
                != terminal_sha256
                or prepared_terminal_claim.terminal_kind != terminal_kind
                or prepared_terminal_claim.terminal_sha256
                != terminal_sha256
            ):
                raise RuntimeError(
                    "continuous Motion R05 terminal claim facts differ"
                )
            validator = owner.require_owned_prepared_terminal_claim
        owned = validator(
            prepared_terminal_claim,
            expected_claim_sha256=expectations["expected_claim_sha256"],
            expected_decision=expectations["expected_decision"],
            expected_reveal_final_preview_sha256=expectations[
                "expected_reveal_final_preview_sha256"
            ],
            expected_global_boundary_receipt_sha256=expectations[
                "expected_global_boundary_receipt_sha256"
            ],
            expected_global_boundary_packet_sha256=expectations[
                "expected_global_boundary_packet_sha256"
            ],
            expected_terminal_boundary_authority_sha256=expectations[
                "expected_terminal_boundary_authority_sha256"
            ],
            expected_terminal_boundary_projection_sha256=expectations[
                "expected_terminal_boundary_projection_sha256"
            ],
            expected_terminal_content_pin_sha256=expectations[
                "expected_terminal_content_pin_sha256"
            ],
            expected_terminal_kind=expectations["expected_terminal_kind"],
            expected_terminal_sha256=expectations[
                "expected_terminal_sha256"
            ],
            expected_selected_env_ids=expectations[
                "expected_selected_env_ids"
            ],
        )
        if owned is not prepared_terminal_claim:
            raise RuntimeError(
                "continuous Motion R05 terminal claim identity differs"
            )
        return expectations

    def _finish_action_ball_continuous_motion_arm(
        self,
        *,
        stage: ActionBallContinuousMotionStage,
        selected_swaps: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        accept_refs: list[object],
        expected_decision: str,
        global_boundary_receipt: object,
        prepared_terminal_claim: object,
    ) -> (
        ActionBallContinuousMotionArmedInstall
        | ActionBallContinuousMotionCensoredInstall
    ):
        """Build private terminal material after the row is authenticated."""

        boundary = self._action_ball_continuous_motion_boundary_module
        terminal_expectations = (
            self._require_action_ball_continuous_motion_terminal_claim(
                prepared_terminal_claim,
                global_boundary_receipt=global_boundary_receipt,
                stage=stage,
                expected_decision=expected_decision,
                require_armed=False,
            )
        )
        receipt_sha256 = self._action_ball_continuous_motion_sha256(
            global_boundary_receipt.canonical_sha256,
            label="global reveal-boundary receipt root",
        )
        packet_sha256 = self._action_ball_continuous_motion_sha256(
            global_boundary_receipt.packet_sha256,
            label="global reveal-boundary packet root",
        )
        next_version = stage.owner_mutation_version + 1
        if next_version > self._ACTION_BALL_INT64_MAX:
            raise RuntimeError(
                "continuous Motion mutation version would overflow int64"
            )
        accept = expected_decision == boundary.DECISION_ACCEPT
        if accept:
            receipt = ActionBallContinuousMotionCommitReceipt(
                schema_version=1,
                kind="action_ball_continuous_motion_child_commit_receipt_v1",
                decision=expected_decision,
                reveal_final_preview_sha256=(
                    stage.reveal_final_preview_sha256
                ),
                global_boundary_receipt_sha256=receipt_sha256,
                global_boundary_packet_sha256=packet_sha256,
                motion_child_token_root_sha256=(
                    stage.motion_child_token_root_sha256
                ),
                prepared_r05_terminal_claim_sha256=(
                    terminal_expectations["expected_claim_sha256"]
                ),
                expected_r05_terminal_kind=(
                    terminal_expectations["expected_terminal_kind"]
                ),
                expected_r05_terminal_sha256=(
                    terminal_expectations["expected_terminal_sha256"]
                ),
                timing_after_image_sha256=stage.timing_after_image_sha256,
                selected_env_ids=stage.env_ids,
                owner_mutation_version_before=stage.owner_mutation_version,
                owner_mutation_version_after=next_version,
                installed_count=len(stage.env_ids),
                censored_count=0,
                policy_opportunity_created=True,
                runtime_integrated=False,
                launch_authorized=False,
            )
            armed = ActionBallContinuousMotionArmedInstall(
                _owner_nonce=self._action_ball_continuous_motion_owner_nonce,
                serial=stage.serial,
            )
        else:
            receipt = ActionBallContinuousMotionCensorReceipt(
                schema_version=1,
                kind="action_ball_continuous_motion_child_censor_receipt_v1",
                decision=expected_decision,
                reveal_final_preview_sha256=(
                    stage.reveal_final_preview_sha256
                ),
                global_boundary_receipt_sha256=receipt_sha256,
                global_boundary_packet_sha256=packet_sha256,
                motion_child_token_root_sha256=(
                    stage.motion_child_token_root_sha256
                ),
                prepared_r05_terminal_claim_sha256=(
                    terminal_expectations["expected_claim_sha256"]
                ),
                expected_r05_terminal_kind=(
                    terminal_expectations["expected_terminal_kind"]
                ),
                expected_r05_terminal_sha256=(
                    terminal_expectations["expected_terminal_sha256"]
                ),
                selected_env_ids=stage.env_ids,
                owner_mutation_version_before=stage.owner_mutation_version,
                owner_mutation_version_after=next_version,
                censored_count=len(stage.env_ids),
                policy_opportunity_created=False,
                runtime_integrated=False,
                launch_authorized=False,
            )
            armed = ActionBallContinuousMotionCensoredInstall(
                _owner_nonce=self._action_ball_continuous_motion_owner_nonce,
                serial=stage.serial,
            )
        self._action_ball_continuous_motion_sha256(
            receipt.canonical_sha256,
            label="Motion child commit receipt root",
        )
        terminal_token = ActionBallContinuousMotionChildTerminalToken(
            _owner_nonce=self._action_ball_continuous_motion_owner_nonce,
            serial=stage.serial,
            decision=expected_decision,
        )
        self._action_ball_continuous_motion_armed_swaps = selected_swaps
        self._action_ball_continuous_motion_armed_refs = (
            accept_refs if accept else None
        )
        self._action_ball_continuous_motion_commit_receipt = receipt
        self._action_ball_continuous_motion_terminal_claim = (
            prepared_terminal_claim
        )
        self._action_ball_continuous_motion_terminal_expectations = (
            terminal_expectations
        )
        self._action_ball_continuous_motion_terminal_token = terminal_token
        if accept:
            self._action_ball_continuous_motion_armed_install = armed
        else:
            self._action_ball_continuous_motion_censored_install = armed
        return armed

    def arm_action_ball_continuous_motion_prearm(
        self,
        prearmed_install: ActionBallContinuousMotionPrearmedInstall,
        global_boundary_receipt: object,
        prepared_terminal_claim: object,
    ) -> ActionBallContinuousMotionArmedInstall:
        """Arm only the owner-issued global ACCEPT branch."""

        boundary = self._action_ball_continuous_motion_boundary_module
        return self._arm_action_ball_continuous_motion_prearm(
            prearmed_install,
            global_boundary_receipt,
            prepared_terminal_claim,
            expected_decision=boundary.DECISION_ACCEPT,
        )

    def arm_censored_action_ball_continuous_motion_prearm(
        self,
        prearmed_install: ActionBallContinuousMotionPrearmedInstall,
        global_boundary_receipt: object,
        prepared_terminal_claim: object,
    ) -> ActionBallContinuousMotionCensoredInstall:
        """Arm only the owner-issued global CENSOR chronology."""

        boundary = self._action_ball_continuous_motion_boundary_module
        return self._arm_action_ball_continuous_motion_prearm(
            prearmed_install,
            global_boundary_receipt,
            prepared_terminal_claim,
            expected_decision=boundary.DECISION_CENSOR,
        )

    def _commit_prevalidated_action_ball_continuous_motion(
        self,
        armed_install: (
            ActionBallContinuousMotionArmedInstall
            | ActionBallContinuousMotionCensoredInstall
        ),
        *,
        expected_decision: str,
    ) -> (
        ActionBallContinuousMotionChildTerminalToken
    ):
        boundary = self._action_ball_continuous_motion_boundary_module
        accept = expected_decision == boundary.DECISION_ACCEPT
        active = (
            self._action_ball_continuous_motion_armed_install
            if accept
            else self._action_ball_continuous_motion_censored_install
        )
        swaps = self._action_ball_continuous_motion_armed_swaps
        refs = self._action_ball_continuous_motion_armed_refs
        receipt = self._action_ball_continuous_motion_commit_receipt
        stage = self._action_ball_continuous_motion_stage
        terminal_claim = self._action_ball_continuous_motion_terminal_claim
        terminal_expectations = (
            self._action_ball_continuous_motion_terminal_expectations
        )
        terminal_token = self._action_ball_continuous_motion_terminal_token
        if (
            self._action_ball_continuous_motion_poisoned
            or self._action_ball_continuous_motion_terminal_epoch_committed
            or active is None
            or armed_install is not active
            or type(armed_install)
            is not (
                ActionBallContinuousMotionArmedInstall
                if accept
                else ActionBallContinuousMotionCensoredInstall
            )
            or armed_install._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or not isinstance(swaps, tuple)
            or type(receipt)
            is not (
                ActionBallContinuousMotionCommitReceipt
                if accept
                else ActionBallContinuousMotionCensorReceipt
            )
            or receipt.decision != expected_decision
            or stage is None
            or terminal_claim is None
            or not isinstance(terminal_expectations, MappingProxyType)
            or type(terminal_token)
            is not ActionBallContinuousMotionChildTerminalToken
            or terminal_token._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or terminal_token.serial != armed_install.serial
            or terminal_token.decision != expected_decision
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion armed install is not the active opaque handle"
            )
        try:
            for destination, after_image in swaps:
                destination.copy_(after_image)
            if refs is not None:
                self._action_ball_active_task_refs = refs[0]
                self._action_ball_continuous_committed_task_refs = refs[1]
            self._action_ball_continuous_motion_mutation_version = (
                receipt.owner_mutation_version_after
            )
            self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = True
            self._action_ball_continuous_current_projection = None
            self._invalidate_action_ball_continuous_observation_publication()
            self._retain_action_ball_continuous_motion_terminal_epoch()
            return terminal_token
        except Exception:
            self._action_ball_continuous_motion_poisoned = True
            self._clear_action_ball_continuous_motion_leaf()
            raise

    def commit_prevalidated_action_ball_continuous_motion(
        self, armed_install: ActionBallContinuousMotionArmedInstall
    ) -> ActionBallContinuousMotionChildTerminalToken:
        """Pure-copy ACCEPT and return only an opaque completion token."""

        boundary = self._action_ball_continuous_motion_boundary_module
        return self._commit_prevalidated_action_ball_continuous_motion(
            armed_install,
            expected_decision=boundary.DECISION_ACCEPT,
        )

    def commit_censored_prevalidated_action_ball_continuous_motion(
        self, armed_install: ActionBallContinuousMotionCensoredInstall
    ) -> ActionBallContinuousMotionChildTerminalToken:
        """Pure-copy CENSOR and return only an opaque completion token."""

        boundary = self._action_ball_continuous_motion_boundary_module
        return self._commit_prevalidated_action_ball_continuous_motion(
            armed_install,
            expected_decision=boundary.DECISION_CENSOR,
        )

    def poison_global_reveal_epoch(self, reason: str) -> None:
        """Idempotently fail-stop this owner during a global poison broadcast."""

        normalized = (
            reason
            if type(reason) is str and reason.strip()
            else "invalid_global_reveal_poison_reason"
        )
        if (
            getattr(
                self,
                "_action_ball_continuous_motion_poison_reason",
                None,
            )
            is None
        ):
            self._action_ball_continuous_motion_poison_reason = normalized
        fault_count = getattr(
            self, "_action_ball_continuous_motion_fault_count_device", None
        )
        if torch.is_tensor(fault_count):
            fault_count.fill_(1)
        self._action_ball_continuous_motion_poisoned = True

    @staticmethod
    def _action_ball_continuous_motion_global_drain_row(
        owner_row: object,
    ) -> tuple[tuple[str, int], ...]:
        """Parse only the exact frozen Motion row from the global receipt."""

        field_names = (
            "mutation_version",
            "fault_count",
            "invariant_count",
            "terminal_resolution_total",
        )
        if getattr(owner_row, "owner_kind", None) != "motion":
            raise RuntimeError("global drain owner row is not the Motion row")
        values = getattr(owner_row, "values", None)
        if (
            type(values) is not tuple
            or len(values) != len(field_names)
            or any(
                type(row) is not tuple
                or len(row) != 2
                or row[0] != name
                or type(row[1]) is not int
                for row, name in zip(values, field_names)
            )
        ):
            raise RuntimeError("global drain Motion row schema differs")
        return values

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Freeze Motion counters for the one global device-to-host packet."""

        if (
            self._action_ball_continuous_motion_poisoned
            or self._action_ball_continuous_motion_global_drain_poisoned
        ):
            raise RuntimeError(
                "continuous Motion is poisoned at the global PPO boundary"
            )
        if (
            not getattr(
                self, "_action_ball_continuous_fresh_motion_lane_bound", False
            )
            or self._action_ball_continuous_motion_global_drain_active
            is not None
            or self._action_ball_continuous_motion_leaf_is_active()
            or self._action_ball_continuous_motion_selected_reset_is_active()
        ):
            raise RuntimeError(
                "continuous Motion is not IDLE at the global PPO boundary"
            )
        if (
            type(update_index) is not int
            or update_index
            <= self._action_ball_continuous_motion_global_drain_last_update
            or type(completed_environment_steps) is not int
            or completed_environment_steps
            <= self._action_ball_continuous_motion_global_drain_last_completed_steps
        ):
            raise RuntimeError(
                "continuous Motion global drain chronology did not advance"
            )
        field_names = (
            "mutation_version",
            "fault_count",
            "invariant_count",
            "terminal_resolution_total",
        )
        if (
            getattr(authority, "owner_kind", None) != "motion"
            or tuple(getattr(authority, "field_names", ())) != field_names
            or getattr(authority, "expected_width", None) != len(field_names)
        ):
            raise RuntimeError(
                "continuous Motion global drain authority differs"
            )
        mint = getattr(authority, "mint_device_pack", None)
        require_owned_ack = getattr(authority, "require_owned_ack", None)
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_ppo_drain as drain,
            )
        except (ImportError, ModuleNotFoundError):
            import action_ball_full_mdp_ppo_drain as drain
        if (
            type(authority) is not drain.LeafDevicePackAuthority
            or not callable(mint)
            or getattr(mint, "__self__", None) is not authority
            or getattr(mint, "__func__", None)
            is not drain.LeafDevicePackAuthority.mint_device_pack
            or not callable(require_owned_ack)
            or getattr(require_owned_ack, "__self__", None) is not authority
            or getattr(require_owned_ack, "__func__", None)
            is not drain.LeafDevicePackAuthority.require_owned_ack
        ):
            raise RuntimeError(
                "continuous Motion global drain exact authority API differs"
            )
        device_version = (
            self._action_ball_continuous_motion_device_mutation_version
        )
        fault_count = (
            self._action_ball_continuous_motion_fault_count_device
        )
        terminal_total = (
            self._action_ball_continuous_motion_terminal_resolution_total_device
        )
        opportunities_consumed = (
            self._action_ball_continuous_opportunities_consumed
        )
        policy_created = (
            self._action_ball_continuous_policy_opportunities_created
        )
        infrastructure_censored = (
            self._action_ball_continuous_infrastructure_censors_consumed
        )
        scheduled_ordinal = (
            self._action_ball_continuous_scheduled_ordinal
        )
        current_policy_opportunity = (
            self._action_ball_continuous_current_policy_opportunity
        )
        scalar_counters = (device_version, fault_count, terminal_total)
        env_counters = (
            opportunities_consumed,
            policy_created,
            infrastructure_censored,
            scheduled_ordinal,
        )
        if (
            any(not torch.is_tensor(value) for value in scalar_counters)
            or any(not torch.is_tensor(value) for value in env_counters)
            or not torch.is_tensor(current_policy_opportunity)
            or any(value.dtype != torch.int64 for value in scalar_counters)
            or any(value.dtype != torch.int64 for value in env_counters)
            or current_policy_opportunity.dtype != torch.bool
            or any(
                value.device != torch.device(self.device)
                for value in (*scalar_counters, *env_counters)
            )
            or current_policy_opportunity.device != torch.device(self.device)
            or any(tuple(value.shape) != (1,) for value in scalar_counters)
            or any(
                tuple(value.shape) != (self.num_envs,)
                for value in env_counters
            )
            or tuple(current_policy_opportunity.shape) != (self.num_envs,)
        ):
            raise RuntimeError(
                "continuous Motion global drain counters differ"
            )
        # These are live device chronology relationships, not a hard-coded
        # zero and not a host mirror echo.  ``terminal_total`` counts selected
        # environments while ``device_version`` counts batches, so comparing
        # their magnitudes would reject every legal K>1 terminal.
        scalar_chronology_failure = (
            (device_version < 0)
            | (terminal_total < 0)
        ).to(dtype=torch.int64)
        ordinal_overflow = scheduled_ordinal >= self._ACTION_BALL_INT64_MAX
        ordinal_capacity = torch.where(
            (scheduled_ordinal >= -1) & ~ordinal_overflow,
            scheduled_ordinal + 1,
            torch.zeros_like(scheduled_ordinal),
        )
        negative_lane = (
            (scheduled_ordinal < -1)
            | (opportunities_consumed < 0)
            | (policy_created < 0)
            | (infrastructure_censored < 0)
        )
        opportunity_over_capacity = (
            opportunities_consumed > ordinal_capacity
        )
        policy_over_capacity = policy_created > ordinal_capacity
        censor_over_capacity = infrastructure_censored > ordinal_capacity
        safe_policy = torch.minimum(
            torch.clamp_min(policy_created, 0), ordinal_capacity
        )
        safe_censor = torch.minimum(
            torch.clamp_min(infrastructure_censored, 0), ordinal_capacity
        )
        double_resolution = safe_policy > (ordinal_capacity - safe_censor)
        safe_consumed = torch.minimum(
            torch.clamp_min(opportunities_consumed, 0), ordinal_capacity
        )
        current_without_unconsumed_resolution = (
            current_policy_opportunity
            & (safe_censor <= safe_consumed)
            & (safe_policy <= (safe_consumed - safe_censor))
        )
        lane_failure = (
            negative_lane
            | ordinal_overflow
            | opportunity_over_capacity
            | policy_over_capacity
            | censor_over_capacity
            | double_resolution
            | current_without_unconsumed_resolution
        )
        invariant_count = scalar_chronology_failure + torch.sum(
            lane_failure, dtype=torch.int64
        ).reshape(1)
        values = torch.cat(
            (
                device_version,
                fault_count,
                invariant_count,
                terminal_total,
            )
        ).contiguous()
        # Seal the actual scalar pack sources.  The per-env tensors are read
        # only to derive ``invariant_count``; mutating them after prepare does
        # not change the already-cloned authority pack and is not, by itself,
        # a partial ACK ambiguity.  Normal Motion mutations are independently
        # blocked while this global lease is active.
        source_tensors = (device_version, fault_count, terminal_total)
        source_receipts = tuple(
            _tensor_identity_version_receipt(value)
            for value in source_tensors
        )
        if any(receipt is None for receipt in source_receipts):
            raise RuntimeError(
                "continuous Motion global drain source epoch cannot be sealed"
            )
        pack = mint(leaf=self, values=values)
        self._action_ball_continuous_motion_global_drain_active = (
            _ActionBallContinuousMotionGlobalDrainLease(
                pack=pack,
                authority=authority,
                update_index=update_index,
                completed_environment_steps=completed_environment_steps,
                owner_mutation_version=(
                    self._action_ball_continuous_motion_mutation_version
                ),
                terminal_resolution_total=(
                    self._action_ball_continuous_motion_terminal_resolution_total
                ),
                expected_values=(
                    self._action_ball_continuous_motion_mutation_version,
                    0,
                    0,
                    self._action_ball_continuous_motion_terminal_resolution_total,
                ),
                source_tensor_receipts=source_receipts,
            )
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(
        self, *, pack: object
    ) -> None:
        """Release the exact pre-transfer lease without a business write."""

        active = self._action_ball_continuous_motion_global_drain_active
        if (
            active is None
            or active.stage != "prepared"
            or pack is not active.pack
        ):
            raise RuntimeError(
                "continuous Motion global drain abort pack is stale or foreign"
            )
        if (
            self._action_ball_continuous_motion_mutation_version
            != active.owner_mutation_version
            or self._action_ball_continuous_motion_terminal_resolution_total
            != active.terminal_resolution_total
            or any(
                not _tensor_matches_identity_version_receipt(row[0], row)
                for row in active.source_tensor_receipts
            )
        ):
            self.poison_pre_optimizer_ppo_boundary(
                reason="continuous Motion mutated during global drain prepare"
            )
            raise RuntimeError(
                "continuous Motion global drain pre-transfer image drifted"
            )
        active.stage = "aborted"
        self._action_ball_continuous_motion_global_drain_active = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Consume Motion's exact row after the global optimizer boundary."""

        if self._action_ball_continuous_motion_global_drain_poisoned:
            raise RuntimeError("continuous Motion global drain is poisoned")
        active = self._action_ball_continuous_motion_global_drain_active
        if active is None:
            raise RuntimeError(
                "continuous Motion global drain acknowledgement has no active pack"
            )
        # Prove the exact construction-bound coordinator, optimizer-return
        # window, receipt and decoded row before reading any business facts.
        active.authority.require_owned_ack(
            leaf=self,
            pack=pack,
            receipt=receipt,
            owner_row=owner_row,
        )
        if (
            active.stage != "prepared"
            or pack is not active.pack
            or self._action_ball_continuous_motion_mutation_version
            != active.owner_mutation_version
            or getattr(receipt, "update_index", None) != active.update_index
            or getattr(receipt, "completed_environment_steps", None)
            != active.completed_environment_steps
            or getattr(receipt, "device_to_host_transfers", None) != 1
            or getattr(receipt, "drain_sequence", None)
            != self._action_ball_continuous_motion_global_drain_sequence + 1
            or any(
                not _tensor_matches_identity_version_receipt(row[0], row)
                for row in active.source_tensor_receipts
            )
        ):
            raise RuntimeError(
                "continuous Motion global drain acknowledgement differs"
            )
        values = self._action_ball_continuous_motion_global_drain_row(
            owner_row
        )
        decoded = tuple(value for _name, value in values)
        if (
            decoded != active.expected_values
            or values[1][1] != 0
            or values[2][1] != 0
            or self._action_ball_continuous_motion_terminal_resolution_total
            != active.terminal_resolution_total
        ):
            raise RuntimeError(
                "continuous Motion global drain row differs from its device snapshot"
            )
        self._action_ball_continuous_motion_global_drain_sequence += 1
        self._action_ball_continuous_motion_global_drain_last_update = (
            active.update_index
        )
        self._action_ball_continuous_motion_global_drain_last_completed_steps = (
            active.completed_environment_steps
        )
        self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = (
            active.owner_mutation_version
        )
        self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = False
        active.stage = "acknowledged"
        self._action_ball_continuous_motion_global_drain_active = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason: object) -> None:
        """Sticky reason-only fail-stop for post-transfer/partial ACK faults."""

        if self._action_ball_continuous_motion_global_drain_poison_reason is None:
            self._action_ball_continuous_motion_global_drain_poison_reason = (
                reason
                if type(reason) is str and bool(reason)
                else "unspecified continuous Motion global PPO drain failure"
            )
        self._action_ball_continuous_motion_global_drain_poisoned = True
        self.poison_global_reveal_epoch(
            self._action_ball_continuous_motion_global_drain_poison_reason
        )
        active = self._action_ball_continuous_motion_global_drain_active
        if active is not None:
            active.stage = "poisoned"

    def complete_global_reveal_epoch(
        self,
        child_terminal_token: ActionBallContinuousMotionChildTerminalToken,
        r05_terminal_receipt: object,
    ) -> (
        ActionBallContinuousMotionCommitReceipt
        | ActionBallContinuousMotionCensorReceipt
    ):
        """Publish Motion's receipt only after R05's exact terminal last."""

        transaction = self._action_ball_continuous_transaction_module
        owner = self._action_ball_continuous_transaction_owner
        boundary = self._action_ball_continuous_motion_boundary_module
        stage = self._action_ball_continuous_motion_stage
        claim = self._action_ball_continuous_motion_terminal_claim
        expectations = (
            self._action_ball_continuous_motion_terminal_expectations
        )
        retained_receipt = (
            self._action_ball_continuous_motion_commit_receipt
        )
        retained_token = self._action_ball_continuous_motion_terminal_token
        current_step = getattr(self._env, "common_step_counter", None)
        if (
            self._action_ball_continuous_motion_poisoned
            or not self._action_ball_continuous_motion_terminal_epoch_committed
            or transaction is None
            or type(owner)
            is not transaction.ContinuousRuntimeTransactionOwner
            or stage is None
            or claim is None
            or not isinstance(expectations, MappingProxyType)
            or type(child_terminal_token)
            is not ActionBallContinuousMotionChildTerminalToken
            or child_terminal_token is not retained_token
            or child_terminal_token._owner_nonce
            is not self._action_ball_continuous_motion_owner_nonce
            or child_terminal_token.serial != stage.serial
            or child_terminal_token.decision
            != expectations["expected_decision"]
            or type(current_step) is not int
            or current_step != stage.common_step
            or self._action_ball_continuous_published_common_step
            != stage.common_step
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion global reveal epoch completion is stale or duplicated"
            )
        accept = expectations["expected_decision"] == boundary.DECISION_ACCEPT
        expected_child_type = (
            ActionBallContinuousMotionCommitReceipt
            if accept
            else ActionBallContinuousMotionCensorReceipt
        )
        expected_r05_type = (
            transaction.CommittedRevealBatch
            if accept
            else transaction.CensoredRevealBatch
        )
        if (
            type(retained_receipt) is not expected_child_type
            or type(r05_terminal_receipt) is not expected_r05_type
            or retained_receipt.decision
            != expectations["expected_decision"]
            or retained_receipt.prepared_r05_terminal_claim_sha256
            != expectations["expected_claim_sha256"]
            or retained_receipt.expected_r05_terminal_kind
            != expectations["expected_terminal_kind"]
            or retained_receipt.expected_r05_terminal_sha256
            != expectations["expected_terminal_sha256"]
        ):
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion global reveal epoch terminal type/root differs"
            )
        try:
            owned = owner.require_owned_terminal_receipt(
                claim,
                r05_terminal_receipt,
                expected_claim_sha256=expectations[
                    "expected_claim_sha256"
                ],
                expected_decision=expectations["expected_decision"],
                expected_reveal_final_preview_sha256=expectations[
                    "expected_reveal_final_preview_sha256"
                ],
                expected_global_boundary_receipt_sha256=expectations[
                    "expected_global_boundary_receipt_sha256"
                ],
                expected_global_boundary_packet_sha256=expectations[
                    "expected_global_boundary_packet_sha256"
                ],
                expected_terminal_boundary_authority_sha256=expectations[
                    "expected_terminal_boundary_authority_sha256"
                ],
                expected_terminal_boundary_projection_sha256=expectations[
                    "expected_terminal_boundary_projection_sha256"
                ],
                expected_terminal_content_pin_sha256=expectations[
                    "expected_terminal_content_pin_sha256"
                ],
                expected_terminal_kind=expectations[
                    "expected_terminal_kind"
                ],
                expected_terminal_sha256=expectations[
                    "expected_terminal_sha256"
                ],
                expected_selected_env_ids=expectations[
                    "expected_selected_env_ids"
                ],
            )
            if owned is not r05_terminal_receipt:
                raise RuntimeError(
                    "continuous Motion R05 terminal receipt identity differs"
                )
        except Exception as exc:
            self._action_ball_continuous_motion_poisoned = True
            raise RuntimeError(
                "continuous Motion R05 terminal receipt is not owner-issued"
            ) from exc
        portable_receipt = retained_receipt
        terminal_total = (
            self._action_ball_continuous_motion_terminal_resolution_total_device
        )
        if (
            not torch.is_tensor(terminal_total)
            or terminal_total.dtype != torch.int64
            or terminal_total.device != torch.device(self.device)
            or tuple(terminal_total.shape) != (1,)
        ):
            self.poison_global_reveal_epoch(
                "motion_terminal_resolution_counter_invalid"
            )
            raise RuntimeError(
                "continuous Motion terminal-resolution counter differs"
            )
        try:
            resolution_count = len(
                expectations["expected_selected_env_ids"]
            )
            if (
                resolution_count <= 0
                or self._action_ball_continuous_motion_terminal_resolution_total
                > self._ACTION_BALL_INT64_MAX - resolution_count
            ):
                raise RuntimeError(
                    "continuous Motion terminal-resolution counter overflowed"
                )
            device_overflow = terminal_total > (
                self._ACTION_BALL_INT64_MAX - resolution_count
            )
            terminal_total.copy_(
                torch.where(
                    device_overflow,
                    torch.full_like(
                        terminal_total, self._ACTION_BALL_INT64_MAX
                    ),
                    terminal_total + resolution_count,
                )
            )
            self._action_ball_continuous_motion_fault_count_device.add_(
                device_overflow.to(dtype=torch.int64)
            )
            self._action_ball_continuous_motion_terminal_resolution_total += (
                resolution_count
            )
        except BaseException:
            self.poison_global_reveal_epoch(
                "motion_terminal_resolution_counter_overflow"
            )
            raise
        self._clear_action_ball_continuous_motion_leaf()
        return portable_receipt

    def abort_action_ball_continuous_motion_prearm(
        self,
        value: (
            ActionBallContinuousMotionStage
            | ActionBallContinuousMotionPrearmedInstall
        ),
        *,
        boundary_abort_capability: object = None,
    ) -> None:
        """Discard one exact pre-transfer leaf while R05 remains unarmed.

        A stage with no minted row is purely local.  Once finalize has minted
        Motion's retained-token row, the boundary lane must consume either its
        local abort permission or the exact capability returned after the
        coordinator aborts an active four-row attempt.  A transferred row,
        ACCEPT handle, or CENSOR handle is permanently non-abortable.
        """

        if self._action_ball_continuous_motion_poisoned:
            raise RuntimeError(
                "continuous Motion poisoned owner cannot abort"
        )
        stage = self._action_ball_continuous_motion_stage
        prearmed = self._action_ball_continuous_motion_prearmed_install
        row = self._action_ball_continuous_motion_prearmed_boundary_row
        owner = self._action_ball_continuous_transaction_owner
        lease = getattr(owner, "_active_preview", None)
        if (
            stage is None
            or not (
                value is stage
                or (prearmed is not None and value is prearmed)
            )
            or self._action_ball_continuous_motion_terminal_claim is not None
            or self._action_ball_continuous_motion_terminal_epoch_committed
            or lease is None
            or getattr(lease, "public_token", None)
            is not stage._reveal_final_public_token
            or getattr(lease, "preview_root_sha256", None)
            != stage.reveal_final_preview_sha256
            or getattr(lease, "armed_handle", None) is not None
        ):
            raise RuntimeError(
                "continuous Motion abort requires its exact active pre-transfer child and an unarmed R05 lease"
            )
        if prearmed is None:
            if row is not None or boundary_abort_capability is not None:
                raise RuntimeError(
                    "continuous Motion local stage has no boundary abort capability"
                )
        else:
            _boundary, _boundary_owner, lane = (
                self._action_ball_continuous_motion_boundary_binding()
            )
            try:
                lane.require_abortable_device_row(
                    row,
                    expected_prepared_token=prearmed,
                    abort_capability=boundary_abort_capability,
                )
            except Exception as exc:
                raise RuntimeError(
                    "continuous Motion retained row is not pre-transfer abortable"
                ) from exc
        self._clear_action_ball_continuous_motion_leaf()

    def prepare_action_ball_continuous_task_commit(
        self,
        env_ids,
        scheduled_ordinals,
        task_refs,
        task_receipts,
    ) -> ActionBallContinuousTaskCommitToken:
        """Validate a complete reveal batch without reading live Racket state.

        Task receipts are the other owner's staged immutable candidates.  The
        existing Motion host validator closes task identity, birth/action
        lineage, timing algebra, admitted suffix and episode horizon before
        any Motion task/timing field changes.  The returned capability is
        valid only for this exact policy tick and exact owner state.
        """

        self._action_ball_reject_legacy_fresh_motion_lane("task prepare")

        # This also proves Motion already ran for the current manager tick.
        projection = self.action_ball_continuous_current_projection()
        ids, ordinals = self._action_ball_continuous_event_rows(
            env_ids,
            scheduled_ordinals,
            operation="task prepare",
        )
        self._action_ball_continuous_require_full_reveal_batch(
            ids,
            ordinals,
            operation="task prepare",
        )
        if self._action_ball_continuous_prepared_task_commit is not None:
            raise RuntimeError(
                "continuous Motion already has an unconsumed prepared task token"
            )
        if (
            type(task_refs) is not tuple
            or type(task_receipts) is not tuple
            or len(task_refs) != len(ids)
            or len(task_receipts) != len(ids)
        ):
            raise ValueError(
                "continuous Motion task prepare requires aligned immutable ref/receipt tuples"
            )

        runtime = self._action_ball_runtime_module_bound
        env_rows = tuple(int(value) for value in ids.detach().cpu().tolist())
        ordinal_rows = tuple(
            int(value) for value in ordinals.detach().cpu().tolist()
        )
        timing_rows = []
        for env_id, ordinal, task_ref, receipt in zip(
            env_rows,
            ordinal_rows,
            task_refs,
            task_receipts,
        ):
            if type(task_ref) is not runtime.ActionTaskReceiptRef:
                raise ValueError(
                    "continuous Motion task prepare requires exact ActionTaskReceiptRef rows"
                )
            if type(receipt) is not runtime.ActionBallTaskReceipt:
                raise ValueError(
                    "continuous Motion task prepare requires exact ActionBallTaskReceipt rows"
                )
            if task_ref.env_id != env_id or task_ref.swing_generation != ordinal:
                raise RuntimeError(
                    "continuous Motion staged task ref differs from scheduled env/ordinal"
                )
            previous = self._action_ball_continuous_committed_task_refs[
                env_id
            ]
            if previous is not None:
                same_reset = (
                    task_ref.reset_generation
                    == previous.reset_generation
                )
                same_lineage = (
                    task_ref.env_id == previous.env_id
                    and same_reset
                    and task_ref.action_uid == previous.action_uid
                    and task_ref.action_slot == previous.action_slot
                    and task_ref.birth_sha256 == previous.birth_sha256
                )
                if (
                    (
                        same_reset
                        and (
                            not same_lineage
                            or task_ref.swing_generation
                            != previous.swing_generation + 1
                            or task_ref.sample_sha256
                            == previous.sample_sha256
                            or task_ref.task_sha256
                            == previous.task_sha256
                        )
                    )
                    or (
                        not same_reset
                        and (
                            task_ref.reset_generation
                            != previous.reset_generation + 1
                            or task_ref.swing_generation != 0
                        )
                    )
                ):
                    raise RuntimeError(
                        "continuous Motion successor/reset task identity did not advance exactly once"
                    )
            action_slot = int(self.clip_id[env_id].item())
            # A successor is produced after Motion advances its generation on
            # the reveal tick, matching the repository's existing formal and
            # diagnostic wrap-timing convention.
            pending_elapsed_s = (
                0.0 if ordinal == 0 else float(self._env.step_dt)
            )
            timing = self._validate_action_ball_task_ref_and_receipt_host(
                task_ref,
                receipt,
                env_id=env_id,
                reset_generation=int(
                    self._action_ball_reset_generation[env_id].item()
                ),
                swing_generation=int(
                    self._action_ball_swing_generation[env_id].item()
                ),
                action_slot=action_slot,
                segment_length=int(self.motion.seg_len[action_slot].item()),
                pending_elapsed_s=pending_elapsed_s,
            )
            if bool(
                self._action_ball_continuous_ready_at_reveal[env_id]
            ):
                # Cross-owner publication may start only after this exact
                # staged task has proved it can satisfy Motion's final release
                # guard.  Release repeats the check against live state solely
                # as a drift revalidation.
                self._validate_action_ball_continuous_full_suffix_window(
                    env_id=env_id,
                    timing=timing,
                    task_age_s=timing["pending_elapsed_s"],
                )
            timing_rows.append(
                (
                    timing["pending_elapsed_s"],
                    timing["time_to_contact_s"],
                    timing["teacher_rate"],
                    timing["scaled_t_hit_s"],
                    timing["scaled_t_cycle_s"],
                    timing["pre_swing_wait_s"],
                )
            )

        selected = ids
        serial = self._action_ball_continuous_next_commit_token_serial
        token = ActionBallContinuousTaskCommitToken(
            _owner_nonce=self._action_ball_continuous_commit_owner_nonce,
            serial=serial,
            common_step=projection.common_step,
            env_ids=env_rows,
            scheduled_ordinals=ordinal_rows,
            episode_ticks=tuple(
                int(value)
                for value in projection.episode_tick[selected].tolist()
            ),
            reveal_ticks=tuple(
                int(value)
                for value in projection.reveal_tick[selected].tolist()
            ),
            deadline_ticks=tuple(
                int(value)
                for value in projection.deadline_tick[selected].tolist()
            ),
            next_reveal_ticks=tuple(
                int(value)
                for value in projection.next_reveal_tick[selected].tolist()
            ),
            reset_generations=tuple(
                int(value)
                for value in projection.reset_generation[selected].tolist()
            ),
            swing_generations=tuple(
                int(value)
                for value in projection.swing_generation[selected].tolist()
            ),
            task_refs=task_refs,
            _timing_rows=tuple(timing_rows),
            _active_task_refs_before=tuple(
                self._action_ball_active_task_refs[env_id]
                for env_id in env_rows
            ),
            _committed_task_refs_before=tuple(
                self._action_ball_continuous_committed_task_refs[
                    env_id
                ]
                for env_id in env_rows
            ),
        )
        tensor_receipts = (
            self._action_ball_continuous_commit_tensor_receipts()
        )
        self._action_ball_continuous_next_commit_token_serial = serial + 1
        self._action_ball_continuous_prepared_task_commit = token
        self._action_ball_continuous_prepared_task_commit_receipts = (
            tensor_receipts
        )
        return token

    def _validate_action_ball_continuous_task_commit_token(
        self,
        token: ActionBallContinuousTaskCommitToken,
        *,
        operation: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Authenticate one current, unmodified, single-use task token."""

        current = self._action_ball_continuous_prepared_task_commit
        if (
            type(token) is not ActionBallContinuousTaskCommitToken
            or token._owner_nonce
            is not self._action_ball_continuous_commit_owner_nonce
            or current is not token
        ):
            raise RuntimeError(
                "continuous Motion task token is forged, stale, or already consumed"
            )
        common_step = getattr(self._env, "common_step_counter", None)
        if (
            type(common_step) is not int
            or common_step != token.common_step
            or self._action_ball_continuous_published_common_step
            != token.common_step
        ):
            raise RuntimeError(
                "continuous Motion task token is stale for the current policy tick"
            )
        expected_receipts = (
            self._action_ball_continuous_prepared_task_commit_receipts
        )
        current_receipts = (
            self._action_ball_continuous_commit_tensor_receipts()
        )
        if (
            not isinstance(expected_receipts, tuple)
            or len(current_receipts) != len(expected_receipts)
            or any(
                not _tensor_matches_identity_version_receipt(
                    tensor, expected
                )
                for (tensor, _version), expected in zip(
                    current_receipts, expected_receipts
                )
            )
        ):
            raise RuntimeError(
                "continuous Motion task token owner state drifted after prepare"
            )
        if tuple(
            self._action_ball_active_task_refs[env_id]
            for env_id in token.env_ids
        ) != token._active_task_refs_before:
            raise RuntimeError(
                "continuous Motion active task refs drifted after prepare"
            )
        if tuple(
            self._action_ball_continuous_committed_task_refs[env_id]
            for env_id in token.env_ids
        ) != token._committed_task_refs_before:
            raise RuntimeError(
                "continuous Motion committed task refs drifted after prepare"
            )

        ids = torch.tensor(
            token.env_ids, dtype=torch.long, device=self.device
        )
        ordinals = torch.tensor(
            token.scheduled_ordinals,
            dtype=torch.long,
            device=self.device,
        )
        self._action_ball_continuous_require_full_reveal_batch(
            ids,
            ordinals,
            operation=operation,
        )
        current_rows = (
            tuple(
                int(value)
                for value in self._action_ball_continuous_episode_step[
                    ids
                ].detach().cpu().tolist()
            ),
            tuple(
                int(value)
                for value in self._action_ball_continuous_current_reveal_step[
                    ids
                ].detach().cpu().tolist()
            ),
            tuple(
                int(value)
                for value in self._action_ball_continuous_current_deadline_step[
                    ids
                ].detach().cpu().tolist()
            ),
            tuple(
                int(value)
                for value in self._action_ball_continuous_next_reveal_step[
                    ids
                ].detach().cpu().tolist()
            ),
            tuple(
                int(value)
                for value in self._action_ball_reset_generation[
                    ids
                ].detach().cpu().tolist()
            ),
            tuple(
                int(value)
                for value in self._action_ball_swing_generation[
                    ids
                ].detach().cpu().tolist()
            ),
        )
        expected_rows = (
            token.episode_ticks,
            token.reveal_ticks,
            token.deadline_ticks,
            token.next_reveal_ticks,
            token.reset_generations,
            token.swing_generations,
        )
        if current_rows != expected_rows:
            raise RuntimeError(
                "continuous Motion task token identity/timing rows drifted after prepare"
            )
        return ids, ordinals

    def commit_prepared_action_ball_continuous_task(
        self,
        token: ActionBallContinuousTaskCommitToken,
    ) -> None:
        """Publish one previously validated Motion batch exactly once."""

        self._action_ball_reject_legacy_fresh_motion_lane(
            "prepared task commit"
        )

        ids, _ordinals = (
            self._validate_action_ball_continuous_task_commit_token(
                token,
                operation="task commit",
            )
        )

        timing = torch.tensor(
            token._timing_rows,
            dtype=self._action_ball_task_age_s.dtype,
            device=self.device,
        )
        if tuple(timing.shape) != (len(ids), 6):
            raise RuntimeError(
                "continuous Motion prepared timing batch shape changed"
            )
        active_refs = list(self._action_ball_active_task_refs)
        committed_refs = list(
            self._action_ball_continuous_committed_task_refs
        )
        for env_id, task_ref in zip(token.env_ids, token.task_refs):
            active_refs[env_id] = task_ref
            committed_refs[env_id] = task_ref

        # All validation and device materialization completed above.  These
        # indexed writes are the sole Motion publication phase.
        self._action_ball_task_pending_elapsed_s[ids] = timing[:, 0]
        self._action_ball_task_age_s[ids] = timing[:, 0]
        self._action_ball_time_to_contact_s[ids] = timing[:, 1]
        self._action_ball_teacher_rate[ids] = timing[:, 2]
        self._action_ball_scaled_t_hit_s[ids] = timing[:, 3]
        self._action_ball_scaled_t_cycle_s[ids] = timing[:, 4]
        self._action_ball_pre_swing_wait_s[ids] = timing[:, 5]
        self._action_ball_task_timing_active[ids] = True
        self._action_ball_active_task_refs = active_refs
        self._action_ball_continuous_committed_task_refs = committed_refs
        self._action_ball_continuous_task_commit_pending[ids] = False
        self._action_ball_continuous_task_commit_missed[ids] = False
        self._action_ball_continuous_task_committed[ids] = True
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_current_projection = None

    def acknowledge_action_ball_continuous_infrastructure_invalid(
        self,
        env_ids,
        scheduled_ordinals,
        reveal_context,
    ) -> None:
        """Consume one current unpublishable reveal without cadence drift.

        A successfully prepared batch must present its exact single-use token.
        If preflight itself failed before a token existed, the caller presents
        the isolated current Motion projection it read for this reveal.  Its
        manager tick, plus Motion's still-current internal publication, closes
        chronology without treating mutable tensor identity as authority.
        """

        self._action_ball_reject_legacy_fresh_motion_lane(
            "infrastructure-invalid acknowledgement"
        )

        ids, ordinals = self._action_ball_continuous_event_rows(
            env_ids,
            scheduled_ordinals,
            operation="infrastructure-invalid acknowledgement",
        )
        self._action_ball_continuous_require_full_reveal_batch(
            ids,
            ordinals,
            operation="infrastructure-invalid acknowledgement",
        )
        prepared = self._action_ball_continuous_prepared_task_commit
        if type(reveal_context) is ActionBallContinuousTaskCommitToken:
            token_ids, token_ordinals = (
                self._validate_action_ball_continuous_task_commit_token(
                    reveal_context,
                    operation="infrastructure-invalid acknowledgement",
                )
            )
            if not torch.equal(ids, token_ids) or not torch.equal(
                ordinals, token_ordinals
            ):
                raise RuntimeError(
                    "continuous Motion infrastructure acknowledgement differs from prepared token"
                )
        elif type(reveal_context) is ActionBallContinuousMotionProjection:
            if prepared is not None:
                raise RuntimeError(
                    "continuous Motion prepared infrastructure failure requires its exact token"
                )
            if (
                reveal_context.common_step
                != self._action_ball_continuous_published_common_step
            ):
                raise RuntimeError(
                    "continuous Motion infrastructure projection is stale"
                )
            self._require_action_ball_continuous_projection_current(
                self._action_ball_continuous_current_projection
            )
        else:
            raise ValueError(
                "continuous Motion infrastructure acknowledgement requires "
                "an exact task token or current Motion projection"
            )
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_task_commit_pending[ids] = False
        self._action_ball_continuous_task_committed[ids] = False
        self._action_ball_continuous_task_commit_missed[ids] = True
        self._action_ball_continuous_motion_release_pending[ids] = False
        self._action_ball_continuous_motion_release_missed[ids] = False
        self._publish_action_ball_continuous_phase()
        self._action_ball_continuous_current_projection = None

    def commit_action_ball_continuous_task(
        self,
        env_ids,
        scheduled_ordinals,
        task_refs,
    ) -> None:
        """Commit every admitted reveal's full task ref and Motion timing.

        Readiness does not authorize skipping a question.  A not-ready row
        still installs a new task identity/timing tuple and closes at its
        frozen deadline; it merely keeps playback on the ready reference.
        Target and ball installation remain the future R05 owner's separate
        transaction and are not claimed by this Motion-side receipt.
        """

        self._action_ball_reject_legacy_fresh_motion_lane("direct task commit")

        ids, ordinals = self._action_ball_continuous_event_rows(
            env_ids,
            scheduled_ordinals,
            operation="task commit",
        )
        self._require_action_ball_continuous_current_publication(
            operation="task commit",
        )
        self._action_ball_continuous_require_full_reveal_batch(
            ids,
            ordinals,
            operation="task commit",
        )
        if self._action_ball_continuous_prepared_task_commit is not None:
            raise RuntimeError(
                "continuous Motion prepared task token must be consumed explicitly"
            )
        if type(task_refs) is not tuple or len(task_refs) != len(ids):
            raise ValueError(
                "continuous Motion task commit requires one immutable task ref per row"
            )
        admissible = (
            self._action_ball_continuous_reveal_due[ids]
            & self._action_ball_continuous_task_commit_pending[ids]
            & ~self._action_ball_continuous_task_committed[ids]
            & ~self._action_ball_continuous_motion_active[ids]
            & (
                self._action_ball_continuous_scheduled_ordinal[ids]
                == ordinals
            )
        )
        if not bool(admissible.all()):
            raise RuntimeError(
                "continuous Motion task commit is not the current scheduled reveal"
            )
        runtime = self._action_ball_runtime_module_bound
        staged_refs = []
        for env_id, ordinal, task_ref in zip(
            ids.detach().cpu().tolist(),
            ordinals.detach().cpu().tolist(),
            task_refs,
        ):
            env_id = int(env_id)
            ordinal = int(ordinal)
            if type(task_ref) is not runtime.ActionTaskReceiptRef:
                raise ValueError(
                    "continuous Motion task commit requires exact ActionTaskReceiptRef rows"
                )
            if task_ref.env_id != env_id or task_ref.swing_generation != ordinal:
                raise RuntimeError(
                    "continuous Motion task ref differs from scheduled env/ordinal"
                )
            previous = self._action_ball_continuous_committed_task_refs[
                env_id
            ]
            if previous is not None:
                same_lineage = (
                    task_ref.env_id == previous.env_id
                    and task_ref.reset_generation
                    == previous.reset_generation
                    and task_ref.action_uid == previous.action_uid
                    and task_ref.action_slot == previous.action_slot
                    and task_ref.birth_sha256 == previous.birth_sha256
                )
                if (
                    not same_lineage
                    or task_ref.swing_generation
                    != previous.swing_generation + 1
                    or task_ref.sample_sha256 == previous.sample_sha256
                    or task_ref.task_sha256 == previous.task_sha256
                ):
                    raise RuntimeError(
                        "continuous Motion successor task identity did not advance exactly once"
                    )
            self._validate_action_ball_continuous_task_timing_binding(
                env_id=env_id,
                task_ref=task_ref,
            )
            staged_refs.append((env_id, task_ref))
        for env_id, task_ref in staged_refs:
            self._action_ball_continuous_committed_task_refs[env_id] = task_ref
        self._action_ball_continuous_task_commit_pending[ids] = False
        self._action_ball_continuous_task_committed[ids] = True
        self._action_ball_continuous_current_projection = None

    def _validate_action_ball_continuous_task_timing_binding(
        self,
        *,
        env_id: int,
        task_ref,
    ) -> dict[str, float]:
        """Revalidate the immutable task authority and its installed timing."""

        runtime = self._action_ball_runtime_module_bound
        if type(task_ref) is not runtime.ActionTaskReceiptRef:
            raise ValueError(
                "continuous Motion timing binding requires exact ActionTaskReceiptRef"
            )
        if not bool(self._action_ball_task_timing_active[env_id]):
            raise RuntimeError(
                "continuous Motion task timing was not atomically installed"
            )
        if self._action_ball_active_task_refs[env_id] != task_ref:
            raise RuntimeError(
                "continuous Motion task timing is not owned by the committed task ref"
            )
        live_ref = self._action_ball_task_ref_for_env(env_id)
        if live_ref != task_ref:
            raise RuntimeError(
                "continuous Motion committed task ref differs from live task authority"
            )
        receipt = self._action_ball_task_receipt_resolver(task_ref)
        timing = self._validate_action_ball_task_ref_and_receipt(
            task_ref,
            receipt,
            env_id=env_id,
        )
        actual_timing = {
            "pending_elapsed_s": (
                self._action_ball_task_pending_elapsed_s,
                timing["pending_elapsed_s"],
            ),
            "task_age_s": (
                self._action_ball_task_age_s,
                timing["pending_elapsed_s"],
            ),
            "time_to_contact_s": (
                self._action_ball_time_to_contact_s,
                timing["time_to_contact_s"],
            ),
            "teacher_rate": (
                self._action_ball_teacher_rate,
                timing["teacher_rate"],
            ),
            "scaled_t_hit_s": (
                self._action_ball_scaled_t_hit_s,
                timing["scaled_t_hit_s"],
            ),
            "scaled_t_cycle_s": (
                self._action_ball_scaled_t_cycle_s,
                timing["scaled_t_cycle_s"],
            ),
            "pre_swing_wait_s": (
                self._action_ball_pre_swing_wait_s,
                timing["pre_swing_wait_s"],
            ),
        }
        for name, (tensor, expected_value) in actual_timing.items():
            expected = torch.as_tensor(
                expected_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if not bool(torch.eq(tensor[env_id], expected)):
                raise RuntimeError(
                    "continuous Motion task timing retained a stale "
                    f"{name} value"
                )
        return timing

    def release_action_ball_continuous_motion_playback(
        self,
        env_ids,
        scheduled_ordinals,
    ) -> None:
        """Release playback only for a ready, already-committed reveal."""

        self._action_ball_reject_legacy_fresh_motion_lane(
            "separate playback release"
        )

        ids, ordinals = self._action_ball_continuous_event_rows(
            env_ids,
            scheduled_ordinals,
            operation="playback release",
        )
        self._require_action_ball_continuous_current_publication(
            operation="playback release",
        )
        self._action_ball_continuous_require_full_release_batch(
            ids,
            ordinals,
        )
        releasable = (
            self._action_ball_continuous_reveal_due[ids]
            & self._action_ball_continuous_ready_at_reveal[ids]
            & self._action_ball_continuous_task_committed[ids]
            & self._action_ball_continuous_motion_release_pending[ids]
            & ~self._action_ball_continuous_motion_active[ids]
            & (
                self._action_ball_continuous_scheduled_ordinal[ids]
                == ordinals
            )
        )
        if not bool(releasable.all()):
            raise RuntimeError(
                "continuous Motion playback release is not a ready committed reveal"
            )
        for env_id, ordinal in zip(
            ids.detach().cpu().tolist(),
            ordinals.detach().cpu().tolist(),
        ):
            env_id = int(env_id)
            ordinal = int(ordinal)
            task_ref = self._action_ball_continuous_committed_task_refs[
                env_id
            ]
            if task_ref is None or task_ref.swing_generation != ordinal:
                raise RuntimeError(
                    "continuous Motion playback release lost its committed task identity"
                )
            timing = self._validate_action_ball_continuous_task_timing_binding(
                env_id=env_id,
                task_ref=task_ref,
            )
            self._validate_action_ball_continuous_full_suffix_window(
                env_id=env_id,
                timing=timing,
                task_age_s=float(
                    self._action_ball_task_age_s[env_id].item()
                ),
            )
        self._action_ball_continuous_motion_active[ids] = True
        self._action_ball_continuous_motion_release_pending[ids] = False
        self._action_ball_continuous_suffix_complete[ids] = False
        self._action_ball_continuous_ready_reference_active[ids] = False
        self._action_ball_continuous_phase[ids] = (
            _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                "active_opportunity"
            ]
        )
        self._action_ball_continuous_current_projection = None

    def acknowledge_action_ball_continuous_motion_task(
        self,
        env_ids,
        scheduled_ordinals,
        task_refs,
    ) -> None:
        """Legacy convenience wrapper: commit all rows, release ready rows."""

        self._action_ball_reject_legacy_fresh_motion_lane(
            "task acknowledgement wrapper"
        )

        ids, ordinals = self._action_ball_continuous_event_rows(
            env_ids,
            scheduled_ordinals,
            operation="task acknowledgement",
        )

        self.commit_action_ball_continuous_task(
            ids,
            ordinals,
            task_refs,
        )
        ready = self._action_ball_continuous_ready_at_reveal[ids]
        if bool(ready.any()):
            self.release_action_ball_continuous_motion_playback(
                ids[ready],
                ordinals[ready],
            )

    def _publish_action_ball_continuous_phase(
        self, writable_rows: torch.Tensor | None = None
    ) -> None:
        step = self._action_ball_continuous_episode_step
        ordinal = self._action_ball_continuous_scheduled_ordinal
        deadline = self._action_ball_continuous_current_deadline_step
        active = self._action_ball_continuous_sequence_active
        before_first = active & (ordinal < 0)
        within_opportunity = active & (ordinal >= 0) & (step <= deadline)
        unavailable = within_opportunity & ~self._action_ball_continuous_ready_at_reveal
        infrastructure_invalid = (
            within_opportunity
            & (
                self._action_ball_continuous_task_commit_missed
                | self._action_ball_continuous_motion_release_missed
            )
        )
        active_opportunity = (
            within_opportunity & ~unavailable & ~infrastructure_invalid
        )
        suffix = (
            active
            & self._action_ball_continuous_motion_active
            & (step > deadline)
        )
        ready_reference = (
            active & self._action_ball_continuous_ready_reference_active
        )
        ready_authority = self._action_ball_continuous_ready_authority
        live_ready = (
            torch.ones_like(active)
            if self._action_ball_continuous_fresh_motion_lane_bound
            else (
                torch.zeros_like(active)
                if ready_authority is None
                else ready_authority
            )
        )
        phase = torch.full_like(
            self._action_ball_continuous_phase,
            _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                "recovery_hidden"
            ],
        )
        phase = torch.where(
            before_first,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "pre_reveal_hidden"
                ],
            ),
            phase,
        )
        phase = torch.where(
            ready_reference & live_ready & ~before_first,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE["ready_hold"],
            ),
            phase,
        )
        phase = torch.where(
            active_opportunity,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "active_opportunity"
                ],
            ),
            phase,
        )
        phase = torch.where(
            unavailable,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "recovery_unavailable"
                ],
            ),
            phase,
        )
        phase = torch.where(
            infrastructure_invalid,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "infrastructure_invalid"
                ],
            ),
            phase,
        )
        phase = torch.where(
            suffix,
            torch.full_like(
                phase,
                _ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE[
                    "post_deadline_suffix"
                ],
            ),
            phase,
        )
        if writable_rows is not None:
            phase = torch.where(
                writable_rows,
                phase,
                self._action_ball_continuous_phase,
            )
        self._action_ball_continuous_phase.copy_(phase)

    def _advance_action_ball_continuous_motion_cadence(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the frozen clock independently of contact and readiness."""

        self._require_action_ball_continuous_motion_leaf_idle(
            operation="cadence advance"
        )
        self._require_action_ball_continuous_parent_authorities()
        # A prepared capability is valid for exactly one Motion publication
        # tick.  Advancing first revokes it, while absolute cadence continues.
        self._action_ball_continuous_prepared_task_commit = None
        self._action_ball_continuous_prepared_task_commit_receipts = None
        self._action_ball_continuous_current_projection = None
        self._action_ball_continuous_published_common_step = None
        if (
            self._action_ball_birth_broker is None
            and not self._action_ball_continuous_fresh_motion_lane_bound
        ):
            raise RuntimeError(
                "continuous Motion cadence requires the action-ball birth authority"
            )
        schedule = self._action_ball_continuous_schedule_projection
        active_sequence = self._action_ball_continuous_sequence_active
        if self._action_ball_swing_generation is None:
            raise RuntimeError(
                "continuous Motion cadence has no swing-generation authority"
            )
        fresh_schedule_exhausted = torch.zeros_like(active_sequence)
        if self._action_ball_continuous_fresh_motion_lane_bound:
            fresh_schedule_exhausted = (
                self._action_ball_continuous_scheduled_ordinal
                >= _ACTION_BALL_FULL_MDP_FRESH_REFERENCE_DUE_COUNT - 1
            )
        prospective_step = self._action_ball_continuous_episode_step + (
            active_sequence.to(dtype=torch.long)
        )
        prospective_reveal = (
            active_sequence
            & ~fresh_schedule_exhausted
            & prospective_step.eq(
                self._action_ball_continuous_next_reveal_step
            )
        )
        overdue = (
            active_sequence
            & ~fresh_schedule_exhausted
            & prospective_step.gt(
                self._action_ball_continuous_next_reveal_step
            )
        )
        writable_rows = (
            self._latch_action_ball_full_mdp_motion_epoch_row_fault(
                overdue.contiguous(),
                reason_bit=(
                    _ACTION_EPOCH_ROW_FAULT_MOTION_CADENCE_OVERDUE
                ),
            )
        )
        successor_candidate = prospective_reveal & (
            self._action_ball_continuous_scheduled_ordinal >= 0
        )
        generation_writable = (
            self._latch_action_ball_full_mdp_motion_epoch_row_fault(
                (
                    successor_candidate
                    & self._action_ball_swing_generation.ge(
                        self._ACTION_BALL_INT64_MAX
                    )
                ).contiguous(),
                reason_bit=(
                    _ACTION_EPOCH_ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW
                ),
            )
        )
        writable_rows = writable_rows & generation_writable
        if self._action_ball_continuous_fresh_motion_lane_bound:
            prospective_task_close = (
                active_sequence
                & self._action_ball_continuous_canonical_task_valid
                & prospective_step.eq(
                    self._action_ball_continuous_canonical_task_close_tick
                )
                & ~self._action_ball_continuous_canonical_playback_started
            )
            timing_advance = (
                self._action_ball_continuous_motion_active
                & ~prospective_task_close
                & writable_rows
            )
            reference_writable = (
                self._latch_action_ball_full_mdp_motion_epoch_row_fault(
                    (
                        timing_advance
                        & ~self._action_ball_task_timing_active
                    ).contiguous(),
                    reason_bit=(
                        _ACTION_EPOCH_ROW_FAULT_MOTION_TASK_TIMING_CONTRACT
                    ),
                )
            )
            writable_rows = writable_rows & reference_writable
        active_sequence = active_sequence & writable_rows
        # A reveal not committed by the next Motion update, or a ready reveal
        # committed without releasing playback, was not an atomic handoff.
        # Record either missing half, then keep the frozen cadence moving;
        # target/ball integration will classify it as infrastructure.
        self._action_ball_continuous_task_commit_missed |= (
            self._action_ball_continuous_task_commit_pending & writable_rows
        )
        self._action_ball_continuous_task_commit_pending.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_motion_release_missed |= (
            self._action_ball_continuous_motion_release_pending & writable_rows
        )
        self._action_ball_continuous_motion_release_pending.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_reveal_due.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_closed_mask.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_close_reason.masked_fill_(
            writable_rows, ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE
        )
        self._action_ball_continuous_deadline_due.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_recovery_unavailable.masked_fill_(
            writable_rows, False
        )
        self._action_ball_continuous_episode_step.add_(
            active_sequence.to(dtype=torch.long)
        )
        step = self._action_ball_continuous_episode_step
        reveal = prospective_reveal & writable_rows
        self._action_ball_continuous_reveal_due.logical_or_(reveal)
        self._action_ball_continuous_current_policy_opportunity[reveal] = False
        self._action_ball_continuous_task_commit_missed[reveal] = False
        self._action_ball_continuous_task_committed[reveal] = False
        self._action_ball_continuous_task_commit_pending[reveal] = True
        self._action_ball_continuous_motion_release_missed[reveal] = False
        successor = reveal & (
            self._action_ball_continuous_scheduled_ordinal >= 0
        )
        self._action_ball_swing_generation.add_(successor.to(torch.long))
        self._action_ball_continuous_scheduled_ordinal[reveal] += 1
        self._action_ball_continuous_current_reveal_step[reveal] = step[
            reveal
        ]
        self._action_ball_continuous_current_deadline_step[reveal] = (
            step[reveal] + int(schedule["deadline_offset_steps"])
        )
        self._action_ball_continuous_next_reveal_step[reveal] += int(
            schedule["cadence_steps"]
        )
        if self._action_ball_continuous_fresh_motion_lane_bound:
            # The boundary after the fourth reveal retires that opportunity;
            # it is not a fifth task.  Once retirement is reached, park the
            # next-reveal clock at the shared episode horizon so the final
            # ready-hold tail remains non-negative and cannot schedule again.
            retired_final_opportunity = (
                active_sequence
                & fresh_schedule_exhausted
                & step.eq(self._action_ball_continuous_next_reveal_step)
                & self._action_ball_continuous_next_reveal_step.lt(
                    _ACTION_BALL_FULL_MDP_FRESH_EPISODE_HORIZON_TICKS
                )
            )
            self._action_ball_continuous_next_reveal_step[
                retired_final_opportunity
            ] = _ACTION_BALL_FULL_MDP_FRESH_EPISODE_HORIZON_TICKS
        # A fresh training row earns task exposure by surviving the balance
        # prefix to this due tick.  R07 remains post-shot recovery evidence;
        # using its 13-component all-of projection here would circularly make
        # full-motion imitation a prerequisite for seeing full motion.
        if self._action_ball_continuous_fresh_motion_lane_bound:
            live_ready = torch.ones_like(reveal)
        else:
            ready_authority = self._action_ball_continuous_ready_authority
            live_ready = (
                torch.zeros_like(reveal)
                if ready_authority is None
                else ready_authority
            )
        ready = (
            reveal
            & live_ready
            & self._action_ball_continuous_ready_reference_active
            & ~self._action_ball_continuous_motion_active
        )
        self._action_ball_continuous_ready_at_reveal[reveal] = ready[
            reveal
        ]
        self._action_ball_continuous_motion_release_pending[ready] = True
        unavailable = reveal & ~ready
        self._action_ball_continuous_recovery_unavailable.logical_or_(
            unavailable
        )

        current_ordinal = self._action_ball_continuous_scheduled_ordinal
        deadline = (
            active_sequence
            & (current_ordinal >= 0)
            & (
                step
                == self._action_ball_continuous_current_deadline_step
            )
            & (
                self._action_ball_continuous_last_closed_ordinal
                < current_ordinal
            )
        )
        self._action_ball_continuous_deadline_due.logical_or_(deadline)
        self._action_ball_continuous_last_closed_ordinal[deadline] = (
            current_ordinal[deadline]
        )
        self._action_ball_continuous_opportunities_consumed[deadline] += 1
        if not self._action_ball_continuous_fresh_motion_lane_bound:
            self._action_ball_continuous_current_policy_opportunity[deadline] = (
                False
            )
        self._action_ball_continuous_task_commit_pending[deadline] = False
        self._action_ball_continuous_motion_release_pending[deadline] = False

        motion_active = (
            self._action_ball_continuous_motion_active & writable_rows
        )
        closed_without_playback = deadline & ~motion_active
        if self._action_ball_continuous_fresh_motion_lane_bound:
            task_close_due = (
                active_sequence
                & self._action_ball_continuous_canonical_task_valid
                & step.eq(
                    self._action_ball_continuous_canonical_task_close_tick
                )
            )
            closed_without_playback = (
                task_close_due
                & ~self._action_ball_continuous_canonical_playback_started
            )
            self._action_ball_continuous_current_policy_opportunity[
                task_close_due
            ] = False
        self._action_ball_task_timing_active[closed_without_playback] = False
        timing_advance = motion_active & ~closed_without_playback
        held, suffix_due = self._advance_action_ball_task_timing(
            advance_mask=timing_advance,
            resolve_pending=False,
        )
        suffix_due &= timing_advance
        self._action_ball_continuous_motion_active[suffix_due] = False
        self._action_ball_continuous_motion_active[
            closed_without_playback
        ] = False
        self._action_ball_continuous_suffix_complete[suffix_due] = True
        self._action_ball_continuous_ready_reference_active[
            suffix_due | closed_without_playback
        ] = True
        self._action_ball_task_timing_active[suffix_due] = False
        self._write_action_ball_continuous_close_edge(
            suffix_due=suffix_due,
            closed_without_playback=closed_without_playback,
            writable_rows=writable_rows,
        )
        self._advance_action_ball_continuous_canonical_lifecycle(
            motion_active_before=motion_active,
            suffix_due=suffix_due,
            closed_without_playback=closed_without_playback,
            writable_rows=writable_rows,
        )
        self._hold_action_ball_continuous_ready_reference(writable_rows)
        held = held | (
            self._action_ball_continuous_ready_reference_active
            & writable_rows
        )
        self._publish_action_ball_continuous_phase(writable_rows)
        if "action_ball_continuous_phase" in self.metrics:
            self.metrics["action_ball_continuous_phase"] = (
                self._action_ball_continuous_phase.to(
                    dtype=self.metrics["action_ball_continuous_phase"].dtype
                )
            )
            self.metrics["action_ball_continuous_reveal_due"] = (
                self._action_ball_continuous_reveal_due.float()
            )
            self.metrics["action_ball_continuous_deadline_due"] = (
                self._action_ball_continuous_deadline_due.float()
            )
            self.metrics[
                "action_ball_continuous_recovery_unavailable"
            ] = self._action_ball_continuous_recovery_unavailable.float()
            self.metrics[
                "action_ball_continuous_task_commit_missed"
            ] = self._action_ball_continuous_task_commit_missed.float()
            self.metrics[
                "action_ball_continuous_motion_release_missed"
            ] = self._action_ball_continuous_motion_release_missed.float()
            self.metrics[
                "action_ball_continuous_opportunities_consumed"
            ] = self._action_ball_continuous_opportunities_consumed.to(
                dtype=self.speed_scale.dtype
            )
            self.metrics[
                "action_ball_continuous_policy_opportunities_created"
            ] = self._action_ball_continuous_policy_opportunities_created.to(
                dtype=self.speed_scale.dtype
            )
            self.metrics[
                "action_ball_continuous_infrastructure_censors_consumed"
            ] = (
                self._action_ball_continuous_infrastructure_censors_consumed.to(
                    dtype=self.speed_scale.dtype
                )
            )
        common_step = getattr(self._env, "common_step_counter", None)
        if type(common_step) is int and common_step >= 0:
            # Publish last, after every current-tick cadence/reference writer.
            self._action_ball_continuous_published_common_step = common_step
            self._publish_action_ball_continuous_observation()
            self._seal_action_ball_continuous_current_projection(common_step)
        self._increment_action_ball_continuous_motion_mutation_version()
        return held, suffix_due

    def _write_action_ball_continuous_close_edge(
        self,
        *,
        suffix_due: torch.Tensor,
        closed_without_playback: torch.Tensor,
        writable_rows: torch.Tensor | None = None,
    ) -> None:
        """Publish this tick's row-wise close mechanics before identity clear."""

        closed_mask = suffix_due | closed_without_playback
        close_reason = torch.where(
                suffix_due,
                torch.full_like(
                    self._action_ball_continuous_close_reason,
                    ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX,
                ),
                torch.where(
                    closed_without_playback,
                    torch.full_like(
                        self._action_ball_continuous_close_reason,
                        ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED,
                    ),
                    torch.full_like(
                        self._action_ball_continuous_close_reason,
                        ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE,
                    ),
                ),
            )
        if writable_rows is not None:
            closed_mask = torch.where(
                writable_rows,
                closed_mask,
                self._action_ball_continuous_closed_mask,
            )
            close_reason = torch.where(
                writable_rows,
                close_reason,
                self._action_ball_continuous_close_reason,
            )
        self._action_ball_continuous_closed_mask.copy_(closed_mask)
        self._action_ball_continuous_close_reason.copy_(close_reason)

    @staticmethod
    def _range_is_exact_zero_pair(value) -> bool:
        try:
            pair = tuple(value)
        except TypeError:
            return False
        return len(pair) == 2 and all(
            type(item) in (int, float) and math.isfinite(float(item)) and float(item) == 0.0
            for item in pair
        )

    @staticmethod
    def _mapping_ranges_are_exact_zero(mapping) -> bool:
        return isinstance(mapping, dict) and all(
            MotionCommand._range_is_exact_zero_pair(value) for value in mapping.values()
        )

    @staticmethod
    def _canonical_registry_module():
        """Load the repository registry only for the explicitly enabled formal path."""

        import importlib.util
        import sys

        global _CANONICAL_REGISTRY_RUNTIME_MODULE
        module_name = "_hope_canonical_motion_registry_runtime"
        script = (
            Path(__file__).resolve().parents[6]
            / "scripts"
            / "canonical_motion_registry.py"
        )
        if _CANONICAL_REGISTRY_RUNTIME_MODULE is not None:
            if (
                Path(_CANONICAL_REGISTRY_RUNTIME_MODULE.__file__).resolve()
                != script
            ):
                raise ValueError(
                    "cached canonical registry module resolved to a different file"
                )
            return _CANONICAL_REGISTRY_RUNTIME_MODULE
        if not script.is_file():
            raise ValueError(f"canonical registry loader is missing: {script}")
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot create canonical registry loader spec for {script}")
        module = importlib.util.module_from_spec(spec)
        # Do not reuse a caller-preloaded sys.modules object, even when it
        # spoofs __file__.  The first trusted load in this process always
        # executes the exact repository bytes through the standard file loader.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if Path(module.__file__).resolve() != script:
            sys.modules.pop(module_name, None)
            raise ValueError("canonical registry executed from a wrong file")
        _CANONICAL_REGISTRY_RUNTIME_MODULE = module
        return module

    @staticmethod
    def _training_contract_module():
        """Load the shared execution-contract helpers from their repository bytes.

        ``commands.py`` is imported standalone by the dependency-light tests, so
        an ordinary package import is not available here.  Reuse the canonical
        registry loader's trust pattern: always execute the exact repository
        file, never a caller-preloaded ``sys.modules`` object.
        """

        import importlib.util
        import sys

        global _TRAINING_CONTRACT_RUNTIME_MODULE
        module_name = "_hope_training_contract_runtime"
        script = (
            Path(__file__).resolve().parents[3]
            / "utils"
            / "training_contract.py"
        )
        if _TRAINING_CONTRACT_RUNTIME_MODULE is not None:
            if (
                Path(_TRAINING_CONTRACT_RUNTIME_MODULE.__file__).resolve()
                != script
            ):
                raise ValueError(
                    "cached training-contract module resolved to a different file"
                )
            return _TRAINING_CONTRACT_RUNTIME_MODULE
        if not script.is_file():
            raise ValueError(f"training contract module is missing: {script}")
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ValueError(
                f"cannot create training-contract loader spec for {script}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if Path(module.__file__).resolve() != script:
            sys.modules.pop(module_name, None)
            raise ValueError("training contract executed from a wrong file")
        _TRAINING_CONTRACT_RUNTIME_MODULE = module
        return module

    @staticmethod
    def _exact_config_sha256(value, label: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                f"{label} must be one exact 64-character lowercase SHA-256"
            )
        return value

    @staticmethod
    def _configured_motion_files(value) -> tuple[str, ...]:
        if isinstance(value, str):
            files = (value,)
        else:
            try:
                files = tuple(value)
            except TypeError as exc:
                raise ValueError(
                    "canonical motion_file must be one path or an ordered path sequence"
                ) from exc
        if not files or any(type(path) is not str or not path for path in files):
            raise ValueError(
                "canonical motion_file entries must be non-empty path strings"
            )
        try:
            return tuple(str(Path(path).expanduser().resolve(strict=True)) for path in files)
        except OSError as exc:
            raise ValueError(f"cannot resolve canonical motion_file: {exc}") from exc

    @staticmethod
    def _exact_numeric_tuple(value, label: str) -> tuple[float, ...]:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be an explicit ordered sequence") from exc
        if any(
            isinstance(item, bool)
            or type(item) not in (int, float)
            or not math.isfinite(float(item))
            for item in raw
        ):
            raise ValueError(f"{label} must contain only finite real numbers")
        return tuple(float(item) for item in raw)

    def _load_and_validate_canonical_registry(self, env):
        """Bind all five runtime columns to one pinned, training-authorized registry."""

        registry_path = getattr(self.cfg, "canonical_registry_path", "")
        if type(registry_path) is not str or not registry_path.strip():
            raise ValueError(
                "canonical_ready_mode requires canonical_registry_path"
            )
        expected_registry = self._exact_config_sha256(
            getattr(self.cfg, "canonical_registry_sha256", ""),
            "canonical_registry_sha256",
        )
        expected_alignment = self._exact_config_sha256(
            getattr(self.cfg, "canonical_registry_alignment_sha256", ""),
            "canonical_registry_alignment_sha256",
        )
        expected_ready = self._exact_config_sha256(
            getattr(self.cfg, "canonical_ready_sha256", ""),
            "canonical_ready_sha256",
        )
        expected_ready_fk = self._exact_config_sha256(
            getattr(self.cfg, "canonical_ready_fk_sha256", ""),
            "canonical_ready_fk_sha256",
        )
        promotion_certificate_path = getattr(
            self.cfg, "canonical_promotion_certificate_path", ""
        )
        if (
            type(promotion_certificate_path) is not str
            or not promotion_certificate_path.strip()
        ):
            raise ValueError(
                "canonical_ready_mode requires "
                "canonical_promotion_certificate_path"
            )
        repo_root_value = getattr(self.cfg, "canonical_registry_repo_root", "")
        if type(repo_root_value) is not str:
            raise ValueError("canonical_registry_repo_root must be a path string")
        repo_root = repo_root_value.strip() or None

        registry_module = self._canonical_registry_module()
        try:
            registry = registry_module.load_canonical_motion_bank_registry(
                registry_path,
                repo_root=repo_root,
                expected_registry_sha256=expected_registry,
            )
            admission = registry_module.verify_registry_promotion_certificate(
                registry,
                promotion_certificate_path,
                authorization_purpose="training",
            )
            promotion_binding = registry_module.bank_promotion_binding(
                registry,
                authorization_purpose="training",
            )
            registry_module.motion_admission.require_matching_admission(
                admission, promotion_binding
            )
            tables = registry_module.adapt_registry_for_runtime(
                registry,
                expected_alignment_sha256=expected_alignment,
                expected_canonical_ready_sha256=expected_ready,
                expected_canonical_ready_fk_sha256=expected_ready_fk,
                authorization_purpose="training",
                admission=admission,
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(f"invalid canonical motion registry: {exc}") from exc

        if tables.canonical_ready_sha256 != expected_ready:
            raise ValueError(
                "canonical_ready_sha256 differs from the pinned registry ready: "
                f"config={expected_ready} registry={tables.canonical_ready_sha256}"
            )
        if tables.canonical_ready_fk_sha256 != expected_ready_fk:
            raise ValueError(
                "canonical_ready_fk_sha256 differs from the pinned registry FK truth: "
                f"config={expected_ready_fk} "
                f"registry={tables.canonical_ready_fk_sha256}"
            )
        actual_files = self._motion_files
        if actual_files != tuple(tables.motion_file):
            raise ValueError(
                "canonical motion_file order differs from registry motion_ids: "
                f"ids={tables.motion_ids} config={actual_files} "
                f"registry={tables.motion_file}"
            )
        families = getattr(self.cfg, "clip_family_per_clip", None)
        if (
            not isinstance(families, (tuple, list))
            or tuple(families) != tuple(tables.clip_family_per_clip)
        ):
            raise ValueError(
                "canonical clip_family_per_clip must exactly equal the registry table"
            )

        commands_cfg = getattr(getattr(env, "cfg", None), "commands", None)
        racket_cfg = getattr(commands_cfg, "racket_target", None)
        if racket_cfg is None:
            raise ValueError(
                "canonical_ready_mode requires env.cfg.commands.racket_target "
                "for an atomic phase/face table check"
            )
        phases = self._exact_numeric_tuple(
            getattr(racket_cfg, "strike_phase_per_clip", ()),
            "racket_target.strike_phase_per_clip",
        )
        if phases != tuple(tables.strike_phase_per_clip):
            raise ValueError(
                "racket_target.strike_phase_per_clip differs from the registry table"
            )
        signs = self._exact_numeric_tuple(
            getattr(racket_cfg, "mount_normal_sign_per_clip", ()),
            "racket_target.mount_normal_sign_per_clip",
        )
        if signs != tuple(tables.mount_normal_sign_per_clip):
            raise ValueError(
                "racket_target.mount_normal_sign_per_clip differs from the registry table"
            )

        self.canonical_motion_ids = tuple(tables.motion_ids)
        self.canonical_registry_sha256 = tables.registry_sha256
        self.canonical_registry_alignment_sha256 = tables.alignment_sha256
        self.canonical_ready_sha256 = tables.canonical_ready_sha256
        self.canonical_ready_fk_sha256 = tables.canonical_ready_fk_sha256
        self.canonical_contact_opportunity_frames = tuple(
            tables.contact_opportunity_frames_per_clip
        )
        self.canonical_source_manifest_sha256_per_clip = tuple(
            tables.source_manifest_sha256_per_clip
        )
        self.canonical_build_manifest_sha256_per_clip = tuple(
            tables.build_manifest_sha256_per_clip
        )
        self.canonical_applicability_manifest_sha256_per_clip = tuple(
            tables.applicability_manifest_sha256_per_clip
        )
        self.canonical_evidence_level_per_clip = tuple(
            tables.evidence_level_per_clip
        )
        self.canonical_evidence_manifest_sha256_per_clip = tuple(
            tables.evidence_manifest_sha256_per_clip
        )
        self.canonical_question_bank_sha256_per_clip = tuple(
            tables.question_bank_sha256_per_clip
        )
        self.canonical_training_config_sha256_per_clip = tuple(
            tables.training_config_sha256_per_clip
        )
        self.canonical_onnx_model_sha256_per_clip = tuple(
            tables.onnx_model_sha256_per_clip
        )
        self.canonical_adoption_manifest_sha256_per_clip = tuple(
            tables.adoption_manifest_sha256_per_clip
        )
        # Keep the actual opaque capability and the exact object it authorizes.  The action-ball
        # manifest may repeat these hashes for identity, but it can never mint or replace this
        # code-rooted training admission.
        self._canonical_motion_registry = registry
        self._canonical_motion_admission = admission
        self._canonical_motion_promotion_binding = promotion_binding
        self._canonical_motion_registry_module = registry_module
        return tables

    def _snapshot_canonical_motion_bytes(self) -> tuple[bytes, ...]:
        """Bind MotionLoader to the exact registry-authorized NPZ bytes."""

        payloads: list[bytes] = []
        digests: list[str] = []
        for index, path in enumerate(self._motion_files):
            try:
                payload = Path(path).read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"cannot snapshot canonical motion_file[{index}]: {exc}"
                ) from exc
            payloads.append(payload)
            digests.append(hashlib.sha256(payload).hexdigest())
        expected = tuple(self._canonical_registry_tables.npz_sha256_per_clip)
        if tuple(digests) != expected:
            raise ValueError(
                "canonical motion bytes changed after trusted registry admission"
            )
        return tuple(payloads)

    def _snapshot_diagnostic_motion_bytes(self) -> tuple[bytes, ...]:
        """Bind an unauthorized diagnostic to the exact bytes its loader adopts."""

        payloads: list[bytes] = []
        digests: list[str] = []
        for index, path in enumerate(self._motion_files):
            try:
                payload = Path(path).read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"cannot snapshot diagnostic motion_file[{index}]: {exc}"
                ) from exc
            payloads.append(payload)
            digests.append(hashlib.sha256(payload).hexdigest())
        if tuple(digests) != tuple(self._motion_file_sha256):
            raise ValueError(
                "diagnostic motion bytes changed between initial hashing and "
                "MotionLoader adoption"
            )
        return tuple(payloads)

    def _validate_canonical_registry_motion_bytes(self) -> None:
        """Check schema and the immutable snapshots used by MotionLoader."""

        if not self.motion.kinematics_contract_exact:
            raise ValueError(
                "canonical_ready_mode requires every clip to use exact schema-2 kinematics "
                "with the runtime body order bound"
            )
        tables = self._canonical_registry_tables
        if int(self.motion.num_segments) != len(tables.motion_ids):
            raise ValueError(
                "canonical MotionLoader segment count differs from the five registry rows"
            )
        actual_hashes = tuple(
            hashlib.sha256(payload).hexdigest()
            for payload in self._motion_payloads
        )
        if actual_hashes != tuple(tables.npz_sha256_per_clip):
            raise ValueError(
                "canonical motion bytes changed between registry validation and MotionLoader adoption"
            )

    def _configure_start_pose_ramp(self) -> None:
        """Normalize and bind the declared start-pose ramp (or the legacy identity).

        ``cfg.start_pose_ramp is None`` is the literal pre-ramp path: the
        resolved payload is the all-zero identity, ``_start_pose_ramp_enabled``
        is False, and every effective range below returns the static config
        unchanged.  A present declaration is re-validated here rather than
        trusted from ``train.py`` — a directly constructed cfg (tests, exports,
        a future launcher) must meet the same law.
        """

        contract = self._training_contract_module()

        raw = getattr(self.cfg, "start_pose_ramp", None)
        self._start_pose_ramp = contract.validate_action_ball_start_pose_ramp(
            raw, name="MotionCommandCfg.start_pose_ramp"
        )
        self._start_pose_ramp_enabled = bool(self._start_pose_ramp["enabled"])
        self._start_pose_ramp_sha256 = (
            contract.action_ball_start_pose_ramp_sha256(self._start_pose_ramp)
        )
        if not self._start_pose_ramp_enabled:
            return
        # 静态种子 = ramp 在 progress=0 处的取值,必须落在 [0, 终点] 内。
        # 这条和 train.py 的硬门读同一个函数,不是两份互相抄的实现。
        offenders: list[str] = []
        for field, static in (
            ("pose_range", self.cfg.pose_range),
            ("velocity_range", self.cfg.velocity_range),
        ):
            for axis, pair in dict(static or {}).items():
                if str(axis) not in self._start_pose_ramp[field]:
                    offenders.append(f"{field}.{axis} has no declared endpoint")
                    continue
                if not contract.action_ball_start_pose_ramp_seed_within_endpoint(
                    self._start_pose_ramp,
                    field=field,
                    axis=str(axis),
                    static=pair,
                ):
                    offenders.append(
                        f"{field}.{axis}={list(pair)!r} leaves endpoint "
                        f"{self._start_pose_ramp[field][str(axis)]!r}"
                    )
        joint_static = tuple(
            float(value) for value in (self.cfg.joint_position_range or ())
        )
        joint_endpoint = self._start_pose_ramp["joint_position_range"]
        if len(joint_static) != 2:
            offenders.append("joint_position_range must be a [lo,hi] pair")
        else:
            for seed, end in zip(joint_static, joint_endpoint):
                if seed == 0.0:
                    continue
                if seed * end < 0.0 or abs(seed) > abs(end):
                    offenders.append(
                        f"joint_position_range={list(joint_static)!r} leaves "
                        f"endpoint {list(joint_endpoint)!r}"
                    )
                    break
        if offenders:
            raise ValueError(
                "start_pose_ramp static seeds must lie inside [0, endpoint]: "
                + "; ".join(offenders)
            )
        print(
            "[MotionCommand] start_pose_ramp enabled: "
            f"ramp_steps={self._start_pose_ramp['ramp_steps']} "
            f"pose_x={self._start_pose_ramp['pose_range']['x']} "
            f"pose_y={self._start_pose_ramp['pose_range']['y']} "
            f"pose_yaw={self._start_pose_ramp['pose_range']['yaw']} "
            f"sha={self._start_pose_ramp_sha256}",
            flush=True,
        )

    def start_pose_ramp_progress(self) -> float:
        """Return this control step's ramp fraction in ``[0, 1]``."""

        ramp = getattr(self, "_start_pose_ramp", None)
        if ramp is None or not ramp.get("enabled", False):
            return 0.0
        step = getattr(self._env, "common_step_counter", None)
        if type(step) is not int or step < 0:
            raise RuntimeError(
                "start_pose_ramp requires a non-negative integer "
                "env.common_step_counter"
            )
        return self._training_contract_module(
        ).action_ball_start_pose_ramp_progress(ramp, step)

    def _effective_reset_range_list(
        self, field: str, progress: float
    ) -> list[tuple[float, float]]:
        """Return the six ordered axis ranges after the ramp interpolation."""

        static = getattr(self.cfg, field, None) or {}
        ramp = getattr(self, "_start_pose_ramp", None)
        out = []
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            seed = static.get(axis, (0.0, 0.0))
            if ramp is None or not ramp.get("enabled", False):
                out.append((float(seed[0]), float(seed[1])))
                continue
            out.append(
                self._training_contract_module(
                ).action_ball_start_pose_ramp_axis_range(
                    ramp, field=field, axis=axis, static=seed, progress=progress
                )
            )
        return out

    def _effective_joint_position_range(
        self, progress: float
    ) -> tuple[float, float]:
        """Return the reset joint-noise range after the ramp interpolation."""

        seed = tuple(
            float(value) for value in (self.cfg.joint_position_range or (0.0, 0.0))
        )
        ramp = getattr(self, "_start_pose_ramp", None)
        if ramp is None or not ramp.get("enabled", False):
            return (seed[0], seed[1])
        endpoint = ramp["joint_position_range"]
        return (
            seed[0] + (endpoint[0] - seed[0]) * progress,
            seed[1] + (endpoint[1] - seed[1]) * progress,
        )

    def _effective_hold_steps_range(self, progress: float) -> tuple[int, int]:
        """Return the legacy motion hold window after the ramp interpolation."""

        ramp = getattr(self, "_start_pose_ramp", None)
        static = tuple(int(value) for value in self.cfg.hold_steps_range)
        if ramp is None or not ramp.get("enabled", False):
            return static
        return self._training_contract_module(
        ).action_ball_start_pose_ramp_hold_window(
            ramp, static=static, progress=progress
        )

    def _validate_canonical_ready_config(self) -> None:
        """Reject reset curricula that would silently bypass the formal ready contract."""

        conflicts: list[str] = []
        if float(self.cfg.stand_start_prob) != 1.0:
            conflicts.append("stand_start_prob must be 1.0")
        if float(self.cfg.post_swing_start_prob) != 0.0:
            conflicts.append("post_swing_start_prob must be 0.0")
        if str(getattr(self.cfg, "post_swing_teacher_receipt", "") or "").strip():
            conflicts.append("post_swing_teacher_receipt must be empty")
        if any(
            bool(getattr(self.cfg, name, False))
            for name in (
                "post_swing_require_ready_at_init",
                "post_swing_fail_fast_first_reset",
                "post_swing_first_reset_require_readback",
            )
        ):
            conflicts.append("post-swing first-reset/replay gates must be disabled")
        if bool(self.cfg.wrap_teleport):
            conflicts.append("wrap_teleport must be false")
        if float(self.cfg.clip_switch_prob) != 0.0:
            conflicts.append("clip_switch_prob must be 0 (switch only at shared-ready wrap)")
        if (
            str(getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED))
            != EVENT_TIMING_MODE_DISABLED
        ):
            conflicts.append(
                "event_timing_mode must be disabled (no mid-stroke ready-reference jump)"
            )
        if int(getattr(self.cfg, "rsi_skip_settle_frames", 0)) != 0:
            conflicts.append("rsi_skip_settle_frames must be 0")
        # 起点扰动:ramp 未启用时,下面四条仍然是逐字节不变的旧硬门。启用时,
        # 静态种子已经在 _configure_start_pose_ramp 里按 [0, 终点] 校验过,
        # 这里放行的是"有一条已声明、已规范化、已入哈希的斜坡在背书"这件事,
        # 而不是"随便什么非零值都行"。stand_start_yaw_range 不在斜坡范围内:
        # 出生朝向由 pose_range.yaw 拥有,两个 yaw 源不许同时活着。
        ramp_enabled = bool(getattr(self, "_start_pose_ramp_enabled", False))
        if not self._range_is_exact_zero_pair(self.cfg.joint_position_range):
            conflicts.append("joint_position_range must be (0, 0)")
        if not self._range_is_exact_zero_pair(self.cfg.stand_start_yaw_range):
            conflicts.append("stand_start_yaw_range must be (0, 0)")
        if not ramp_enabled:
            if not self._mapping_ranges_are_exact_zero(self.cfg.pose_range):
                conflicts.append("all pose_range entries must be (0, 0)")
            if not self._mapping_ranges_are_exact_zero(self.cfg.velocity_range):
                conflicts.append("all velocity_range entries must be (0, 0)")
        if conflicts:
            raise ValueError(
                "canonical_ready_mode is the formal all-true-reset ready-entry path and is "
                "incompatible with RSI/post-swing/noised reset curricula: "
                + "; ".join(conflicts)
            )

    @staticmethod
    def _first_tensor_mismatch(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[int, float]:
        """Return mismatch count/max error for an exact runtime-float32 comparison."""

        unequal = reference != candidate
        count = int(torch.count_nonzero(unequal).item())
        if count == 0:
            return 0, 0.0
        ref64 = reference.to(dtype=torch.float64)
        cand64 = candidate.to(dtype=torch.float64)
        max_abs = float(torch.max(torch.abs(ref64 - cand64)).item())
        return count, max_abs

    def _validate_canonical_ready_clips(self) -> None:
        """Require one literal runtime-ready pose at each clip's own boundaries.

        ``MotionLoader`` intentionally converts references to the consumer's float32 dtype.
        The gate is exact in that runtime dtype (no hidden tolerance): every clip must start
        and end on one identical joint/body pose, including the same quaternion hemisphere;
        all six endpoint velocity channels must be literal zero; every boundary value
        (including the ready root Z) must be finite; and the frame-0 root quaternion must be
        unit length (1e-6, the hope_commands convention).  Yaw-only roots are deliberately
        NOT required: real compiled ready stances carry roll/pitch (fivebind shared ready
        pitch ~-11.2 deg, ChingMu73 ~+8..12 deg measured 2026-07-28), and the per-slot
        action-ball birth frame is the yaw PROJECTION of this root, not the root itself.

        Scope is deliberately PER CLIP (coordinator ruling, 2026-07-28): the runtime's
        per-slot ready machinery (``_action_ball_ready_yaw/quat/z`` captured from each
        clip's own frame 0, B_yaw ball offsets anchored to that per-slot yaw, per-slot
        ready-Z contract in the profile adapter) is the design; aim-rotated canonical
        clips legitimately differ across clips in world orientation.  The former
        cross-clip clause ("all clip starts/ends share one exact world-frame ready
        pose") was a leftover of the single-shared-ready ideal, contradicted that
        per-slot machinery, and is deliberately removed.  Raw capture segments
        (ChingMu73-style units) are still rejected by the per-clip clauses: their own
        endpoints match neither in pose nor in velocity.

        One separately branded measured-N1 diagnostic has a different, narrower
        contract: the immutable clip is a single professional stroke rather than
        a ready-to-ready loop.  Its physical birth comes from the independently
        validated dynamic-ready binding, while frame 0 remains the teacher held
        during the receipt-owned transition.  That mode retains all shape,
        finiteness and unit-quaternion checks.  When its first three poses are
        byte-identical in the source float32 arrays, a microscopic source-side angular-velocity
        roundoff residue is admitted at the teacher start; joint and linear
        start velocity remain literal-zero requirements.  The runtime
        velocity properties below synthesize literal zeros for every held row
        without mutating the immutable clip, and a real moving start is still
        rejected.  The diagnostic deliberately does not claim an equal/
        zero-speed end and consequently terminates after one stroke instead
        of wrapping.
        """

        split_ready_teacher = bool(
            getattr(
                self,
                "action_ball_diagnostic_split_ready_teacher",
                False,
            )
        )

        runtime_joint_count = int(self.robot.data.default_joint_pos.shape[-1])
        motion_joint_count = int(self.motion.joint_pos.shape[-1])
        if (
            runtime_joint_count != _A3_CANONICAL_READY_JOINT_COUNT
            or motion_joint_count != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise ValueError(
                "canonical_ready_mode is bound to the Agibot A3 31-joint articulation: "
                f"runtime={runtime_joint_count}, motion={motion_joint_count}"
            )
        if int(self.body_indexes[0].item()) != 0:
            raise ValueError(
                "canonical_ready_mode requires body_names[0] to be the articulation root body "
                "so one clip frame can atomically seed root and joint state"
            )

        starts = self.motion.seg_start
        ends = starts + self.motion.seg_len - 1
        pose_channels = (
            ("joint_pos", self.motion.joint_pos),
            ("body_pos_w", self.motion._body_pos_w),
            ("body_quat_w", self.motion._body_quat_w),
        )
        velocity_channels = (
            ("joint_vel", self.motion.joint_vel),
            ("body_lin_vel_w", self.motion._body_lin_vel_w),
            ("body_ang_vel_w", self.motion._body_ang_vel_w),
        )
        raw_static_prefixes = getattr(
            self.motion,
            "split_ready_raw_prefix_pose_bytes_static",
            (),
        )
        if split_ready_teacher and (
            type(raw_static_prefixes) is not tuple
            or len(raw_static_prefixes) != int(self.motion.num_segments)
            or any(type(value) is not bool for value in raw_static_prefixes)
        ):
            raise ValueError(
                "measured N=1 diagnostic requires an exact tuple with one boolean "
                "source-byte static-prefix receipt per clip"
            )
        for channel_name, channel in (*pose_channels, *velocity_channels):
            endpoint_values = channel[torch.cat((starts, ends))]
            if not bool(torch.isfinite(endpoint_values).all()):
                raise ValueError(
                    f"canonical_ready_mode found non-finite {channel_name} at a clip boundary"
                )

        for clip_index in range(int(self.motion.num_segments)):
            start_index = int(starts[clip_index].item())
            end_index = int(ends[clip_index].item())
            split_static_first_three = bool(
                split_ready_teacher and raw_static_prefixes[clip_index]
            )
            if not split_ready_teacher:
                for channel_name, channel in pose_channels:
                    mismatch_count, max_abs = self._first_tensor_mismatch(
                        channel[start_index], channel[end_index]
                    )
                    if mismatch_count:
                        raise ValueError(
                            "canonical_ready_mode requires each clip to start and end on one "
                            "exact runtime-float32 ready pose: "
                            f"clip={clip_index} channel={channel_name} "
                            f"mismatches={mismatch_count} max_abs={max_abs:.9g}"
                        )
            velocity_boundaries = (
                (("start", start_index),)
                if split_ready_teacher
                else (("start", start_index), ("end", end_index))
            )
            for boundary_name, boundary_index in velocity_boundaries:
                for channel_name, channel in velocity_channels:
                    value = channel[boundary_index]
                    if int(torch.count_nonzero(value).item()) != 0:
                        max_abs = float(torch.max(torch.abs(value)).item())
                        if (
                            split_static_first_three
                            and channel_name == "body_ang_vel_w"
                            and max_abs
                            <= self._SPLIT_READY_TEACHER_START_BODY_ANG_ROUNDOFF_MAX
                        ):
                            # Do not rewrite MotionLoader storage: frame 0 is
                            # part of the immutable measured playback.  The
                            # held-reference accessors return freshly-created
                            # literal zeros, so only the wait-time teacher is
                            # canonicalized and playback bytes stay intact.
                            continue
                        contract = (
                            "measured N=1 diagnostic rejects moving teacher-start velocities"
                            if split_ready_teacher
                            else "canonical_ready_mode requires literal zero endpoint velocities"
                        )
                        raise ValueError(
                            f"{contract}: "
                            f"clip={clip_index} boundary={boundary_name} channel={channel_name} "
                            f"max_abs={max_abs:.9g}"
                        )
            root_quat = self.motion._body_quat_w[start_index, 0]
            norm = math.sqrt(
                sum(float(value) ** 2 for value in root_quat.tolist())
            )
            if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-6:
                raise ValueError(
                    "canonical_ready_mode requires a unit frame-0 root quaternion (the "
                    "per-slot birth frame is its yaw projection): "
                    f"clip={clip_index} norm={norm:.9g}"
                )

    @staticmethod
    def _action_ball_dynamic_ready_sha256(value: object) -> str:
        """Hash one in-memory runtime binding without filesystem/path ambiguity."""

        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _action_ball_dynamic_ready_exact_sha256(
        value: object, *, name: str
    ) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be one lowercase SHA-256 digest")
        return value

    @staticmethod
    def _action_ball_dynamic_ready_vector(
        value: object, *, name: str, length: int
    ) -> tuple[float, ...]:
        if not isinstance(value, (tuple, list)) or len(value) != length:
            raise ValueError(f"{name} must contain exactly {length} values")
        parsed: list[float] = []
        for index, raw in enumerate(value):
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError(f"{name}[{index}] must be finite numeric")
            parsed.append(float(raw))
        return tuple(parsed)

    @classmethod
    def _validate_action_ball_dynamic_ready_plant_v2(
        cls, value: object, *, name: str
    ) -> None:
        """Validate the schema-v2 plant identity retained in the sealed binding."""

        expected_keys = {
            "joint_names",
            "articulation_joint_names",
            "action_joint_ids",
            "joint_stiffness",
            "joint_damping",
            "joint_effort_limits",
            "joint_velocity_limits",
            "joint_armature",
            "default_joint_pos_rad",
            "action_scale_rad",
            "qdes_joint_pos_limits",
            "physx_control_position_limits",
            "physics_step_dt_s",
            "policy_step_dt_s",
            "control_decimation",
            "control_step_action_delay",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ValueError(f"{name} must contain the exact schema-v2 fields")
        joint_names = value["joint_names"]
        if (
            not isinstance(joint_names, list)
            or len(joint_names) != _A3_CANONICAL_READY_JOINT_COUNT
            or len(set(joint_names)) != _A3_CANONICAL_READY_JOINT_COUNT
            or any(type(joint) is not str or not joint for joint in joint_names)
            or value["articulation_joint_names"] != joint_names
            or value["action_joint_ids"]
            != list(range(_A3_CANONICAL_READY_JOINT_COUNT))
        ):
            raise ValueError(f"{name} must bind one exact 31-joint action order")

        vectors = {
            key: cls._action_ball_dynamic_ready_vector(
                value[key],
                name=f"{name}.{key}",
                length=_A3_CANONICAL_READY_JOINT_COUNT,
            )
            for key in (
                "joint_stiffness",
                "joint_damping",
                "joint_effort_limits",
                "joint_velocity_limits",
                "joint_armature",
                "default_joint_pos_rad",
                "action_scale_rad",
            )
        }
        if (
            any(item <= 0.0 for item in vectors["joint_stiffness"])
            or any(item < 0.0 for item in vectors["joint_damping"])
            or any(item <= 0.0 for item in vectors["joint_effort_limits"])
            or any(item <= 0.0 for item in vectors["joint_velocity_limits"])
            or any(item < 0.0 for item in vectors["joint_armature"])
            or any(item <= 0.0 for item in vectors["action_scale_rad"])
        ):
            raise ValueError(f"{name} contains an invalid actuator value")

        raw_limits = value["qdes_joint_pos_limits"]
        if (
            not isinstance(raw_limits, list)
            or len(raw_limits) != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise ValueError(f"{name}.qdes_joint_pos_limits must have 31 rows")
        limits = tuple(
            cls._action_ball_dynamic_ready_vector(
                row, name=f"{name}.qdes_joint_pos_limits[{index}]", length=2
            )
            for index, row in enumerate(raw_limits)
        )
        if any(lower >= upper for lower, upper in limits):
            raise ValueError(f"{name}.qdes_joint_pos_limits contains an empty row")

        hctrl = value["physx_control_position_limits"]
        expected_hctrl_keys = {
            "schema_version",
            "backend",
            "inset_fraction_per_side_hard_span",
            "selected_joint_names",
            "mechanical_joint_pos_limits",
            "control_joint_pos_limits",
            "unselected_joint_count",
            "unselected_limits_equal_mechanical",
            "articulation_mechanical_ledger_unchanged",
            "soft_qdes_ledger_unchanged",
        }
        expected_selected_names = list(
            _A3_PHYSX_CONTROL_POSITION_LIMIT_JOINT_NAMES
        )
        selected_indices = [
            index
            for index, joint_name in enumerate(joint_names)
            if joint_name in expected_selected_names
        ]
        if (
            type(hctrl) is not dict
            or set(hctrl) != expected_hctrl_keys
            or hctrl["schema_version"] != 1
            or hctrl["backend"] != "physx_root_view_dof_limits"
            or type(hctrl["inset_fraction_per_side_hard_span"]) is not float
            or hctrl["inset_fraction_per_side_hard_span"] != 0.02
            or hctrl["selected_joint_names"] != expected_selected_names
            or [joint_names[index] for index in selected_indices]
            != expected_selected_names
            or type(hctrl["unselected_joint_count"]) is not int
            or hctrl["unselected_joint_count"] != 27
            or hctrl["unselected_limits_equal_mechanical"] is not True
            or hctrl["articulation_mechanical_ledger_unchanged"] is not True
            or hctrl["soft_qdes_ledger_unchanged"] is not True
        ):
            raise ValueError(
                f"{name}.physx_control_position_limits identity is invalid"
            )

        def limit_matrix(raw: object, *, field: str) -> tuple[tuple[float, ...], ...]:
            if not isinstance(raw, list) or len(raw) != 31:
                raise ValueError(
                    f"{name}.physx_control_position_limits.{field} must have 31 rows"
                )
            result = tuple(
                cls._action_ball_dynamic_ready_vector(
                    row,
                    name=(
                        f"{name}.physx_control_position_limits.{field}[{index}]"
                    ),
                    length=2,
                )
                for index, row in enumerate(raw)
            )
            if any(lower >= upper for lower, upper in result):
                raise ValueError(
                    f"{name}.physx_control_position_limits.{field} contains an empty row"
                )
            return result

        mechanical = limit_matrix(
            hctrl["mechanical_joint_pos_limits"],
            field="mechanical_joint_pos_limits",
        )
        control = limit_matrix(
            hctrl["control_joint_pos_limits"],
            field="control_joint_pos_limits",
        )
        selected = set(selected_indices)
        for index, (hard, constrained, qdes) in enumerate(
            zip(mechanical, control, limits)
        ):
            if index not in selected:
                if constrained != hard:
                    raise ValueError(
                        f"{name}.physx_control_position_limits unselected "
                        "H_ctrl must equal H_mech"
                    )
            else:
                span = hard[1] - hard[0]
                if not (
                    math.isclose(
                        constrained[0], hard[0] + 0.02 * span,
                        rel_tol=0.0, abs_tol=2.0e-7,
                    )
                    and math.isclose(
                        constrained[1], hard[1] - 0.02 * span,
                        rel_tol=0.0, abs_tol=2.0e-7,
                    )
                    and hard[0] < constrained[0] < constrained[1] < hard[1]
                ):
                    raise ValueError(
                        f"{name}.physx_control_position_limits selected H_ctrl "
                        "must be two percent per side inside H_mech"
                    )
            if not (constrained[0] <= qdes[0] < qdes[1] <= constrained[1]):
                raise ValueError(
                    f"{name}.qdes_joint_pos_limits[{index}] must remain inside H_ctrl"
                )

        physics_dt = value["physics_step_dt_s"]
        policy_dt = value["policy_step_dt_s"]
        decimation = value["control_decimation"]
        if (
            isinstance(physics_dt, bool)
            or not isinstance(physics_dt, (int, float))
            or not math.isfinite(float(physics_dt))
            or float(physics_dt) <= 0.0
            or isinstance(policy_dt, bool)
            or not isinstance(policy_dt, (int, float))
            or not math.isfinite(float(policy_dt))
            or float(policy_dt) <= 0.0
            or type(decimation) is not int
            or decimation <= 0
            or not math.isclose(
                float(policy_dt),
                float(physics_dt) * decimation,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(f"{name} contains inconsistent control timing")

        delay = value["control_step_action_delay"]
        expected_delay_keys = {
            "schema_version",
            "enabled",
            "semantic_unit",
            "sample_timing",
            "distribution",
            "min_steps",
            "max_steps",
            "shared_across_all_31_joints",
            "history_fill",
        }
        if (
            type(delay) is not dict
            or set(delay) != expected_delay_keys
            or delay["schema_version"] != 1
            or type(delay["enabled"]) is not bool
            or delay["semantic_unit"] != "policy_control_step"
            or delay["sample_timing"] != "once_per_episode_reset"
            or delay["distribution"] != "discrete_uniform_inclusive"
            or type(delay["min_steps"]) is not int
            or type(delay["max_steps"]) is not int
            or delay["min_steps"] < 0
            or delay["max_steps"] < delay["min_steps"]
            or delay["enabled"] != (delay["max_steps"] > 0)
            or delay["shared_across_all_31_joints"] is not True
            or delay["history_fill"]
            != "safe_default_or_action_specific_hold"
        ):
            raise ValueError(f"{name}.control_step_action_delay is invalid")

    def _configure_action_ball_dynamic_ready(self) -> None:
        """Validate and device-materialize the action-specific A3 reset/hold binding.

        The binding is deliberately passed as an already materialized mapping by
        ``train.py``.  Motion owns immutable clip bytes and can therefore close the
        two important identities here: ordered motion SHA-256 values and exact
        runtime-float32 frame-0 physical state.  The associated action term owns
        actor/action/q_des state installation at true reset.
        """

        self._action_ball_dynamic_ready_binding_sha256 = None
        self._action_ball_dynamic_ready_action_order = None
        self._action_ball_dynamic_ready_physical_root_pos_w_m = None
        self._action_ball_dynamic_ready_physical_root_quat_wxyz = None
        self._action_ball_dynamic_ready_physical_joint_pos_rad = None
        self._action_ball_dynamic_ready_physical_joint_vel_radps = None
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = None
        self._action_ball_dynamic_ready_normalized_actor_action = None
        self._action_ball_dynamic_ready_action_term = None

        binding = getattr(self.cfg, "action_ball_dynamic_ready", None)
        if binding is None:
            return
        if not self.canonical_ready_mode:
            raise ValueError(
                "action_ball_dynamic_ready requires canonical_ready_mode"
            )
        expected_top_keys = {
            "schema_version",
            "kind",
            "binding_sha256",
            "action_order",
            "motion_sha256_per_action",
            "rows",
        }
        if type(binding) is not dict or set(binding) != expected_top_keys:
            raise ValueError(
                "action_ball_dynamic_ready must be one exact schema-1/2 runtime binding"
            )
        schema_version = binding["schema_version"]
        expected_kind = {
            1: "action_ball_dynamic_ready_runtime_binding_v1",
            2: "action_ball_dynamic_ready_runtime_binding_v2",
        }.get(schema_version)
        if type(schema_version) is not int or binding["kind"] != expected_kind:
            raise ValueError(
                "action_ball_dynamic_ready schema_version/kind mismatch"
            )
        binding_sha256 = self._action_ball_dynamic_ready_exact_sha256(
            binding["binding_sha256"],
            name="action_ball_dynamic_ready.binding_sha256",
        )
        unsigned = dict(binding)
        del unsigned["binding_sha256"]
        actual_binding_sha256 = self._action_ball_dynamic_ready_sha256(
            unsigned
        )
        if actual_binding_sha256 != binding_sha256:
            raise ValueError(
                "action_ball_dynamic_ready binding SHA-256 mismatch: "
                f"{actual_binding_sha256} != {binding_sha256}"
            )

        action_order_raw = binding["action_order"]
        if (
            not isinstance(action_order_raw, list)
            or not action_order_raw
            or any(
                not isinstance(action_id, str) or not action_id
                for action_id in action_order_raw
            )
            or len(set(action_order_raw)) != len(action_order_raw)
        ):
            raise ValueError(
                "action_ball_dynamic_ready.action_order must contain unique "
                "non-empty action ids"
            )
        action_order = tuple(action_order_raw)
        action_count = int(self.motion.num_segments)
        if len(action_order) != action_count:
            raise ValueError(
                "action_ball_dynamic_ready action count differs from loaded motion: "
                f"{len(action_order)} != {action_count}"
            )
        motion_sha_raw = binding["motion_sha256_per_action"]
        if not isinstance(motion_sha_raw, list):
            raise ValueError(
                "action_ball_dynamic_ready.motion_sha256_per_action must be a list"
            )
        motion_sha256_per_action = tuple(
            self._action_ball_dynamic_ready_exact_sha256(
                value,
                name=(
                    "action_ball_dynamic_ready.motion_sha256_per_action"
                    f"[{index}]"
                ),
            )
            for index, value in enumerate(motion_sha_raw)
        )
        if motion_sha256_per_action != tuple(self._motion_file_sha256):
            raise ValueError(
                "action_ball_dynamic_ready ordered motion SHA-256 values differ "
                "from the immutable MotionLoader inputs"
            )
        canonical_motion_ids = getattr(self, "canonical_motion_ids", None)
        if (
            canonical_motion_ids is not None
            and tuple(canonical_motion_ids) != action_order
        ):
            raise ValueError(
                "action_ball_dynamic_ready.action_order differs from the "
                "canonical registry motion ids"
            )

        rows = binding["rows"]
        if not isinstance(rows, list) or len(rows) != action_count:
            raise ValueError(
                "action_ball_dynamic_ready.rows must have one row per action"
            )
        expected_row_keys = {
            "action_id",
            "physical_ready",
            "hold_qdes_joint_pos_rad",
            "normalized_actor_action",
            "artifact",
            "nominal_hold_receipt",
        }
        if schema_version == 2:
            expected_row_keys.add("runtime_plant_identity")
            expected_row_keys.add("nonterminal_prefix_evidence")
        expected_physical_keys = {
            "root_pos_w_m",
            "root_quat_wxyz",
            "joint_pos_rad",
            "joint_vel_radps",
        }
        expected_pin_keys = {"path", "sha256", "content_sha256"}
        root_pos_rows: list[tuple[float, ...]] = []
        root_quat_rows: list[tuple[float, ...]] = []
        joint_pos_rows: list[tuple[float, ...]] = []
        joint_vel_rows: list[tuple[float, ...]] = []
        hold_qdes_rows: list[tuple[float, ...]] = []
        normalized_action_rows: list[tuple[float, ...]] = []
        for action_slot, row in enumerate(rows):
            if type(row) is not dict or set(row) != expected_row_keys:
                raise ValueError(
                    f"action_ball_dynamic_ready.rows[{action_slot}] has "
                    "unexpected or missing fields"
                )
            if row["action_id"] != action_order[action_slot]:
                raise ValueError(
                    "action_ball_dynamic_ready row order differs from action_order "
                    f"at slot {action_slot}"
                )
            if schema_version == 2:
                self._validate_action_ball_dynamic_ready_plant_v2(
                    row["runtime_plant_identity"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].runtime_plant_identity"
                    ),
                )
            for pin_name in ("artifact", "nominal_hold_receipt"):
                pin = row[pin_name]
                if type(pin) is not dict or set(pin) != expected_pin_keys:
                    raise ValueError(
                        "action_ball_dynamic_ready "
                        f"rows[{action_slot}].{pin_name} must contain exact "
                        "path/file/content pins"
                    )
                if not isinstance(pin["path"], str) or not pin["path"]:
                    raise ValueError(
                        "action_ball_dynamic_ready "
                        f"rows[{action_slot}].{pin_name}.path must be non-empty"
                    )
                for digest_name in ("sha256", "content_sha256"):
                    self._action_ball_dynamic_ready_exact_sha256(
                        pin[digest_name],
                        name=(
                            "action_ball_dynamic_ready."
                            f"rows[{action_slot}].{pin_name}.{digest_name}"
                        ),
                    )

            physical = row["physical_ready"]
            if (
                type(physical) is not dict
                or set(physical) != expected_physical_keys
            ):
                raise ValueError(
                    "action_ball_dynamic_ready physical_ready must contain "
                    "exact root/joint state fields"
                )
            physical_root_pos = self._action_ball_dynamic_ready_vector(
                physical["root_pos_w_m"],
                name=(
                    "action_ball_dynamic_ready."
                    f"rows[{action_slot}].physical_ready.root_pos_w_m"
                ),
                length=3,
            )
            # 人话:这个字段名里的 `_w_` 会让人以为它是 HOPE 世界系(原点在近端左桌角),
            # 但它其实是机器人局部地面系。照字面理解会把机器人放到台面上去,所以这里
            # 显式钉住它的系。
            #
            # The artifact carries no frame declaration of its own and its SHA is pinned by the
            # lineage, so the frame contract has to live here.  Values are in
            # ``a3_robot_origin_ground_z0`` (robot local ground origin, ground z=0), NOT the HOPE
            # ``world`` frame whose origin is the near-left table corner with the table SURFACE at
            # z=0.  The documented bridge is a pure translation with no rotation:
            #   p_a3_robot_origin_ground_z0 = p_world + [0.5, 0.7625, 0.76]
            # so a HOPE-world reading of the same numbers would place the pelvis about 0.15 m ONTO
            # the table at 1.07 m above its surface.  See docs/interfaces/frames_and_coordinates.md.
            #
            # The bounds below are deliberately loose: they only have to separate the two frames,
            # not certify the stance.  A HOPE-world vector for this robot would have x <= 0 and
            # y around -0.76, both of which fall outside these ranges.
            _ROBOT_GROUND_FRAME_BOUNDS = (
                (-1.0, 1.0),  # x: forward of the robot ground origin
                (-1.0, 1.0),  # y: lateral about the robot ground origin
                (0.3, 1.6),  # z: pelvis above the FLOOR, never a table-surface-relative height
            )
            for _axis, (_lo, _hi) in enumerate(_ROBOT_GROUND_FRAME_BOUNDS):
                _value = float(physical_root_pos[_axis])
                if not _lo <= _value <= _hi:
                    raise ValueError(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.root_pos_w_m is declared in the "
                        "a3_robot_origin_ground_z0 robot-local ground frame, but axis "
                        f"{_axis} = {_value!r} is outside [{_lo}, {_hi}]; a HOPE-world vector "
                        "here would spawn the robot on the table surface"
                    )
            root_pos_rows.append(physical_root_pos)
            root_quat_rows.append(
                self._action_ball_dynamic_ready_vector(
                    physical["root_quat_wxyz"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.root_quat_wxyz"
                    ),
                    length=4,
                )
            )
            joint_pos_rows.append(
                self._action_ball_dynamic_ready_vector(
                    physical["joint_pos_rad"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.joint_pos_rad"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )
            joint_vel = self._action_ball_dynamic_ready_vector(
                physical["joint_vel_radps"],
                name=(
                    "action_ball_dynamic_ready."
                    f"rows[{action_slot}].physical_ready.joint_vel_radps"
                ),
                length=_A3_CANONICAL_READY_JOINT_COUNT,
            )
            if any(value != 0.0 for value in joint_vel):
                raise ValueError(
                    "action_ball_dynamic_ready physical joint velocities "
                    "must be literal zero"
                )
            joint_vel_rows.append(joint_vel)
            hold_qdes_rows.append(
                self._action_ball_dynamic_ready_vector(
                    row["hold_qdes_joint_pos_rad"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].hold_qdes_joint_pos_rad"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )
            normalized_action_rows.append(
                self._action_ball_dynamic_ready_vector(
                    row["normalized_actor_action"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].normalized_actor_action"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )

        starts = self.motion.seg_start
        physical_root_pos = torch.tensor(
            root_pos_rows,
            dtype=self.motion.body_pos_w.dtype,
            device=self.motion.body_pos_w.device,
        )
        physical_root_quat = torch.tensor(
            root_quat_rows,
            dtype=self.motion.body_quat_w.dtype,
            device=self.motion.body_quat_w.device,
        )
        physical_joint_pos = torch.tensor(
            joint_pos_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        physical_joint_vel = torch.tensor(
            joint_vel_rows,
            dtype=self.motion.joint_vel.dtype,
            device=self.motion.joint_vel.device,
        )
        split_ready_teacher = bool(
            getattr(
                self,
                "action_ball_diagnostic_split_ready_teacher",
                False,
            )
        )
        if split_ready_teacher:
            # The measured diagnostic intentionally binds two different states:
            # a statically validated physical birth and immutable motion frame 0
            # as the teacher.  They must share a world-yaw frame so task XY/yaw
            # remains coherent, but tilt, root Z and joints are allowed to differ.
            physical_quat_norm = torch.linalg.vector_norm(
                physical_root_quat.to(dtype=torch.float64), dim=-1
            )
            if not bool(
                torch.isfinite(physical_quat_norm).all()
                and torch.all(torch.abs(physical_quat_norm - 1.0) <= 1.0e-6)
            ):
                raise ValueError(
                    "split-ready physical root quaternion must be unit length"
                )
            teacher_root_quat = self.motion.body_quat_w[starts, 0]
            for action_slot in range(action_count):
                physical_values = physical_root_quat[action_slot].tolist()
                teacher_values = teacher_root_quat[action_slot].tolist()
                pw, px, py, pz = (float(value) for value in physical_values)
                tw, tx, ty, tz = (float(value) for value in teacher_values)
                physical_yaw = math.atan2(
                    2.0 * (pw * pz + px * py),
                    1.0 - 2.0 * (py * py + pz * pz),
                )
                teacher_yaw = math.atan2(
                    2.0 * (tw * tz + tx * ty),
                    1.0 - 2.0 * (ty * ty + tz * tz),
                )
                yaw_error = abs(
                    math.atan2(
                        math.sin(physical_yaw - teacher_yaw),
                        math.cos(physical_yaw - teacher_yaw),
                    )
                )
                if yaw_error > 1.0e-6:
                    raise ValueError(
                        "split-ready physical/teacher root yaw mismatch: "
                        f"slot={action_slot} error_rad={yaw_error:.9g}"
                    )
        else:
            exact_frame0 = (
                (
                    "root_pos_w_m",
                    physical_root_pos,
                    self.motion.body_pos_w[starts, 0],
                ),
                (
                    "root_quat_wxyz",
                    physical_root_quat,
                    self.motion.body_quat_w[starts, 0],
                ),
                (
                    "joint_pos_rad",
                    physical_joint_pos,
                    self.motion.joint_pos[starts],
                ),
                (
                    "joint_vel_radps",
                    physical_joint_vel,
                    self.motion.joint_vel[starts],
                ),
            )
            for name, supplied, motion_value in exact_frame0:
                if not torch.equal(supplied, motion_value):
                    mismatch_count, max_abs = self._first_tensor_mismatch(
                        motion_value, supplied
                    )
                    raise ValueError(
                        "action_ball_dynamic_ready physical frame-0 mismatch: "
                        f"channel={name} mismatches={mismatch_count} "
                        f"max_abs={max_abs:.9g}"
                    )

        self._action_ball_dynamic_ready_binding_sha256 = binding_sha256
        self._action_ball_dynamic_ready_action_order = action_order
        self._action_ball_dynamic_ready_physical_root_pos_w_m = (
            physical_root_pos
        )
        self._action_ball_dynamic_ready_physical_root_quat_wxyz = (
            physical_root_quat
        )
        self._action_ball_dynamic_ready_physical_joint_pos_rad = (
            physical_joint_pos
        )
        self._action_ball_dynamic_ready_physical_joint_vel_radps = (
            physical_joint_vel
        )
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = torch.tensor(
            hold_qdes_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        self._action_ball_dynamic_ready_normalized_actor_action = torch.tensor(
            normalized_action_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        # ActionManager constructs after CommandManager in Isaac Lab.  Keep all
        # pre-scene identity/physical validation above, but resolve the decoder
        # term only at the first true reset, before any simulator state write.
        self._action_ball_dynamic_ready_action_term = None

    def _bind_action_ball_dynamic_ready_action_term(self):
        """Resolve the decoder handshake after ActionManager exists."""

        if self._action_ball_dynamic_ready_binding_sha256 is None:
            return None
        if self._action_ball_dynamic_ready_action_term is not None:
            return self._action_ball_dynamic_ready_action_term
        action_manager = getattr(self._env, "action_manager", None)
        get_term = getattr(action_manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError(
                "action_ball_dynamic_ready requires ActionManager.get_term "
                "before its first true reset"
            )
        try:
            action_term = get_term("joint_pos")
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "action_ball_dynamic_ready requires the joint_pos action term"
            ) from exc
        for method_name in (
            "install_action_ball_dynamic_ready_state",
            "install_action_ball_physical_birth_controller_history",
            "restore_action_ball_dynamic_ready_state",
        ):
            if not callable(getattr(action_term, method_name, None)):
                raise RuntimeError(
                    "action_ball_dynamic_ready joint_pos action term lacks "
                    f"{method_name}"
                )
        processed = getattr(action_term, "processed_actions", None)
        if (
            not torch.is_tensor(processed)
            or processed.ndim != 2
            or processed.shape[1] != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise RuntimeError(
                "action_ball_dynamic_ready requires an identity-ordered "
                "31-D joint_pos decoder"
            )
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = (
            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad.to(
                dtype=processed.dtype, device=processed.device
            )
        )
        self._action_ball_dynamic_ready_normalized_actor_action = (
            self._action_ball_dynamic_ready_normalized_actor_action.to(
                dtype=processed.dtype, device=processed.device
            )
        )
        self._action_ball_dynamic_ready_action_term = action_term
        return action_term

    def action_ball_full_mdp_restore_physical_birth_controller_history(
        self, env_ids: torch.Tensor
    ) -> None:
        """Reinstall the birth hold erased by native ActionManager reset.

        FullMDP writes the physical birth before native manager resets.  The
        ActionManager reset must retain its zero policy action/rate semantics,
        while the controller's previous executable q_des must still describe
        the plant that was just born.  Keep that distinction in one narrow
        handoff instead of replaying the broad dynamic-ready transaction.
        """

        if (
            type(env_ids) is not torch.Tensor
            or env_ids.ndim != 1
            or env_ids.dtype != torch.int64
            or env_ids.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "FullMDP controller-history restore requires selected int64 env_ids"
            )
        action_term = self._bind_action_ball_dynamic_ready_action_term()
        if action_term is None:
            raise RuntimeError(
                "FullMDP controller-history restore lacks dynamic-ready action term"
            )
        action_slots = self.clip_id[env_ids]
        result = action_term.install_action_ball_physical_birth_controller_history(
            env_ids,
            self._action_ball_dynamic_ready_normalized_actor_action[action_slots],
            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad[action_slots],
        )
        if result is not None:
            raise RuntimeError(
                "FullMDP controller-history restore must return None"
            )

    def _canonical_ready_steps(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        clips = self.clip_id if env_ids is None else self.clip_id[env_ids]
        return self.motion.seg_start[clips]

    def _require_canonical_ready_boundary(
        self, env_ids: torch.Tensor, operation: str
    ) -> None:
        """Allow an in-episode clip retarget only at a proven zero-speed ready endpoint."""

        if not self.canonical_ready_mode or len(env_ids) == 0:
            return
        clips = self.clip_id[env_ids]
        starts = self.motion.seg_start[clips]
        ends = starts + self.motion.seg_len[clips] - 1
        steps = self.time_steps[env_ids]
        if self.action_ball_diagnostic_split_ready_teacher:
            # A measured diagnostic clip ends in a moving post-strike pose;
            # only its immutable teacher start is a legal install boundary.
            at_ready_boundary = steps == starts
        else:
            at_ready_boundary = (steps == starts) | (steps == ends)
        if not bool(torch.all(at_ready_boundary)):
            bad = env_ids[~at_ready_boundary].detach().cpu().tolist()
            raise ValueError(
                f"{operation} cannot change canonical clip mid-stroke; "
                f"envs {bad} are not at a legal canonical ready boundary"
            )

    def _pose_reference_steps(self) -> torch.Tensor:
        if not self.canonical_ready_mode:
            return self.time_steps
        return torch.where(self.in_hold, self._canonical_ready_steps(), self.time_steps)

    def _action_ball_full_mdp_safe_pose_reference_steps(self) -> torch.Tensor:
        """Return legal gather indices while a named row fault drains.

        ActionEpoch's writable mask is the sole quarantine fact.  Once a
        fresh row faults, actor/reward materialization must remain total until
        the synchronous optimizer drain observes that cause; masking only the
        destination after an out-of-range gather would be too late.  Healthy
        rows keep their exact reference step, while quarantined rows read the
        selected clip's legal frame 0 and cannot mutate published caches.
        """

        steps = self._pose_reference_steps()
        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            return steps
        writable = getattr(
            self,
            "_action_ball_full_mdp_motion_epoch_writable_rows",
            None,
        )
        if writable is None:
            # Construction may query observation shapes before ActionEpoch is
            # bound.  No runtime row fault can exist at that point.
            return steps
        if (
            not torch.is_tensor(writable)
            or writable.dtype != torch.bool
            or tuple(writable.shape) != (self.num_envs,)
            or writable.device != torch.device(self.device)
        ):
            raise RuntimeError(
                "fresh Motion quarantine writable-row ABI differs"
            )
        frame_zero = self.motion.seg_start[self.clip_id]
        return torch.where(writable, steps, frame_zero)

    @classmethod
    def _action_ball_plain_int(
        cls, value, *, name: str, minimum: int = 0
    ) -> int:
        if type(value) is not int or not minimum <= value <= cls._ACTION_BALL_INT64_MAX:
            raise ValueError(
                f"{name} must be a plain integer in "
                f"[{minimum}, {cls._ACTION_BALL_INT64_MAX}]"
            )
        return value

    @staticmethod
    def _action_ball_sha256(value, *, name: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"{name} must be exactly 64 lowercase hexadecimal characters"
            )
        return value

    def _action_ball_continuous_code_owned_action_uids(
        self,
    ) -> tuple[int, ...]:
        """Return the one immutable slot/UID order available before broker bind.

        The fresh diagnostic constructor retains the exact pinned catalog
        before the pre-command factory freezes cadence.  The birth broker is
        allowed to publish later, but only if its complete tuple equals that
        catalog.  Legacy/non-catalog construction continues to require the
        broker-owned tuple directly.
        """

        catalog = getattr(
            self, "_action_ball_full_mdp_diagnostic_catalog_table", None
        )
        catalog_action_uids = None
        if catalog is not None:
            if type(catalog) is not ActionBallFullMdpDiagnosticCatalogTable:
                raise RuntimeError(
                    "Motion code-owned diagnostic catalog type differs"
                )
            catalog_action_uids = catalog.action_uids
        broker_action_uids = getattr(self, "_action_ball_action_uids", None)
        if (
            broker_action_uids is not None
            and catalog_action_uids is not None
            and broker_action_uids != catalog_action_uids
        ):
            raise RuntimeError(
                "Motion broker action identity differs from its code-owned catalog"
            )
        action_uids = (
            broker_action_uids
            if broker_action_uids is not None
            else catalog_action_uids
        )
        action_count = int(self.motion.num_segments)
        if (
            type(action_uids) is not tuple
            or len(action_uids) != action_count
            or len(set(action_uids)) != action_count
            or any(type(uid) is not int or uid <= 0 for uid in action_uids)
        ):
            raise RuntimeError("Motion code-owned action identity table differs")
        return action_uids

    @classmethod
    def _action_ball_runtime_module(cls):
        """Return the exact repository runtime module that minted the broker classes."""

        import importlib
        import importlib.util
        import sys

        global _ACTION_BALL_RUNTIME_MODULE
        script = Path(__file__).resolve().with_name("action_ball_runtime.py")
        module_name = (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
        )
        if _ACTION_BALL_RUNTIME_MODULE is None:
            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    # CPU unit tests load this package from file under a namespace-only stub.
                    # Execute the same repository bytes under the canonical name so the classes
                    # used by Motion and the test broker still have one exact identity.
                    spec = importlib.util.spec_from_file_location(
                        module_name, script
                    )
                    if spec is None or spec.loader is None:
                        raise ValueError(
                            "cannot create action-ball runtime module spec"
                        )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception:
                        sys.modules.pop(module_name, None)
                        raise
            _ACTION_BALL_RUNTIME_MODULE = module
        module = _ACTION_BALL_RUNTIME_MODULE
        try:
            module_file = Path(module.__file__).resolve(strict=True)
        except (AttributeError, OSError) as exc:
            raise ValueError(
                "action-ball runtime module has no exact repository file"
            ) from exc
        if module_file != script:
            raise ValueError(
                "action-ball runtime module resolved to a different file"
            )
        if (
            getattr(module, "BROKER_STATE_SCHEMA_VERSION", None) != 4
            or getattr(module, "SCHEMA_VERSION", None) != 3
            or getattr(module, "SAMPLER_SCHEMA_VERSION", None) != 3
        ):
            raise ValueError(
                "unsupported action-ball runtime/broker/sampler schema"
            )
        cls._action_ball_sha256(
            getattr(module, "ARM_CATALOG_SHA256", None),
            name="action-ball runtime arm catalog SHA",
        )
        for name in (
            "ActionBinding",
            "ActionBirthBroker",
            "ActionBirthReceipt",
            "ActionBallTaskReceipt",
            "ActionTaskReceiptRef",
            "BirthReserveRequest",
            "BirthCommitRequest",
        ):
            if not isinstance(getattr(module, name, None), type):
                raise ValueError(
                    f"action-ball runtime is missing exact {name}"
                )
        return module

    @staticmethod
    def _action_ball_vector(
        value, *, name: str, length: int
    ) -> tuple[float, ...]:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (tuple, list))
            or len(value) != length
        ):
            raise ValueError(f"{name} must be an exact length-{length} tuple/list")
        result = []
        for index, component in enumerate(value):
            if (
                isinstance(component, bool)
                or type(component) not in (int, float)
                or not math.isfinite(float(component))
            ):
                raise ValueError(f"{name}[{index}] must be a plain finite number")
            result.append(float(component))
        return tuple(result)

    @staticmethod
    def _action_ball_resolve_root(value) -> Path:
        if isinstance(value, bool):
            raise ValueError("trusted_repo_root must be one explicit absolute path")
        try:
            raw = Path(os.fspath(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trusted_repo_root must be one explicit absolute path"
            ) from exc
        if not raw.is_absolute() or raw.is_symlink():
            raise ValueError(
                "trusted_repo_root must be absolute and must not be a symlink"
            )
        try:
            resolved = raw.resolve(strict=True)
        except OSError as exc:
            raise ValueError("trusted_repo_root cannot be resolved") from exc
        try:
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise ValueError("trusted_repo_root cannot be stat'ed") from exc
        if not stat.S_ISDIR(mode):
            raise ValueError("trusted_repo_root must be a regular directory")
        return resolved

    @staticmethod
    def _action_ball_file_receipt(
        repo_root: Path,
        relative_path: str,
        *,
        name: str,
        expected_sha256: str | None = None,
    ) -> tuple[Path, str]:
        """Resolve one normalized repo-relative regular file without following symlinks."""

        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
        ):
            raise ValueError(f"{name} must be one normalized repo-relative POSIX path")
        parts = tuple(relative_path.split("/"))
        if (
            any(part in ("", ".", "..") for part in parts)
            or Path(relative_path).is_absolute()
        ):
            raise ValueError(f"{name} must not escape the trusted repository root")
        cursor = repo_root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"{name} must not contain a symlink")
        try:
            resolved = cursor.resolve(strict=True)
            resolved.relative_to(repo_root)
            mode = resolved.stat().st_mode
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{name} cannot be resolved inside trusted_repo_root"
            ) from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"{name} must resolve to a regular file")
        try:
            payload = resolved.read_bytes()
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            expected = MotionCommand._action_ball_sha256(
                expected_sha256, name=f"{name}.expected_sha256"
            )
            if digest != expected:
                raise ValueError(f"{name} SHA-256 changed after admission")
        return resolved, digest

    @classmethod
    def _action_ball_repo_file_receipt(
        cls,
        repo_root: Path,
        path,
        *,
        name: str,
        expected_sha256: str | None = None,
    ) -> tuple[str, str]:
        try:
            resolved = Path(path).resolve(strict=True)
            relative = resolved.relative_to(repo_root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError(f"{name} escaped trusted_repo_root") from exc
        checked, digest = cls._action_ball_file_receipt(
            repo_root,
            relative,
            name=name,
            expected_sha256=expected_sha256,
        )
        if checked != resolved:
            raise ValueError(f"{name} changed during path admission")
        return relative, digest

    def _require_action_ball_motion_admission(
        self, repo_root: Path
    ) -> None:
        registry = self._canonical_motion_registry
        admission = self._canonical_motion_admission
        binding = self._canonical_motion_promotion_binding
        module = self._canonical_motion_registry_module
        if (
            registry is None
            or admission is None
            or binding is None
            or module is None
        ):
            raise ValueError(
                "action-ball requires the code-rooted opaque canonical motion admission"
            )
        if Path(registry.repo_root).resolve(strict=True) != repo_root:
            raise ValueError(
                "action-ball trusted_repo_root differs from canonical motion admission"
            )
        try:
            module.motion_admission.require_matching_admission(
                admission, binding
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(
                "opaque canonical motion admission failed revalidation"
            ) from exc

    def bind_action_ball_birth_broker(
        self, broker, *, trusted_repo_root
    ) -> None:
        """Bind one exact schema-v4 broker to already-admitted motion bytes.

        ``trusted_repo_root`` is deliberately mandatory.  Runtime manifest paths are relative to
        that explicit root and cannot depend on the process working directory.  The manifest only
        supplies identity rows; authorization is re-proved from MotionCommand's retained opaque
        promotion capability.
        """

        if self._action_ball_birth_broker is not None:
            raise ValueError("action-ball birth broker may be bound exactly once")
        if not self.canonical_ready_mode:
            raise ValueError(
                "action-ball birth requires canonical_ready_mode=true"
            )
        if bool(self.cfg.wrap_teleport):
            raise ValueError("action-ball birth requires wrap_teleport=false")
        runtime = self._action_ball_runtime_module()
        if type(broker) is not runtime.ActionBirthBroker:
            raise ValueError(
                "action-ball birth broker must be the exact repository ActionBirthBroker"
            )
        if (
            broker.diagnostic_fast_path
            != self._canonical_diagnostic_unauthorized
        ):
            raise ValueError(
                "action-ball broker diagnostic mode differs from Motion"
            )
        repo_root = self._action_ball_resolve_root(trusted_repo_root)
        if self._canonical_diagnostic_unauthorized:
            print(
                "[MotionCommand] WARN action-ball birth broker bound WITHOUT "
                "canonical motion admission (diagnostic_unauthorized=true)",
                flush=True,
            )
        else:
            self._require_action_ball_motion_admission(repo_root)
        for method_name in (
            "binding_for_slot",
            "reserve_many_true_reset",
            "pending_receipt",
            "commit_many_true_reset",
            "state_dict",
            "load_state_dict",
        ):
            if not callable(getattr(broker, method_name, None)):
                raise ValueError(
                    f"action-ball schema-v4 broker must implement {method_name}()"
                )
        broker_state = broker.state_dict()
        if (
            type(broker_state) is not dict
            or broker_state.get("schema_version")
            != runtime.BROKER_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "action-ball broker/provider/domain authority are not fully bound"
            )

        action_count = self._action_ball_plain_int(
            broker.action_count,
            name="broker.action_count",
            minimum=1,
        )
        if action_count != int(self.motion.num_segments):
            raise ValueError(
                "action-ball action count must equal the loaded motion segment count"
            )
        action_uids = tuple(
            self._action_ball_plain_int(
                uid, name=f"broker.ordered_action_uids[{slot}]", minimum=1
            )
            for slot, uid in enumerate(broker.ordered_action_uids)
        )
        if (
            len(action_uids) != action_count
            or len(set(action_uids)) != action_count
        ):
            raise ValueError(
                "broker.ordered_action_uids must contain one unique UID per slot"
            )
        diagnostic_catalog = self._action_ball_full_mdp_diagnostic_catalog_table
        if (
            diagnostic_catalog is not None
            and (
                type(diagnostic_catalog.action_uids) is not tuple
                or action_uids != diagnostic_catalog.action_uids
            )
        ):
            raise ValueError(
                "broker.ordered_action_uids differs from the code-owned "
                "diagnostic catalog"
            )

        motion_sha256: list[str] = []
        for slot in range(action_count):
            binding = broker.binding_for_slot(slot)
            if type(binding) is not runtime.ActionBinding:
                raise ValueError(
                    f"action-ball binding[{slot}] has a forged runtime type"
                )
            if (
                binding.action_slot != slot
                or binding.action_uid != action_uids[slot]
            ):
                raise ValueError(
                    f"action-ball binding[{slot}] does not match its ordered slot/UID"
                )
            resolved, digest = self._action_ball_file_receipt(
                repo_root,
                binding.motion_path,
                name=f"action-ball binding[{slot}].motion_path",
                expected_sha256=binding.motion_sha256,
            )
            if (
                str(resolved) != self._motion_files[slot]
                or digest != self._motion_file_sha256[slot]
                or digest
                != hashlib.sha256(self._motion_payloads[slot]).hexdigest()
            ):
                raise ValueError(
                    f"action-ball binding[{slot}] does not match the admitted loaded clip bytes"
                )
            motion_sha256.append(digest)

        ready_steps = self.motion.seg_start.to(
            device=self.motion.body_pos_w.device, dtype=torch.long
        )
        if self.action_ball_diagnostic_split_ready_teacher:
            ready_root_z_tensor = (
                self._action_ball_dynamic_ready_physical_root_pos_w_m[:, 2]
            )
            ready_root_quat_tensor = (
                self._action_ball_dynamic_ready_physical_root_quat_wxyz
            )
        else:
            ready_root_z_tensor = self.motion.body_pos_w[
                ready_steps, 0, 2
            ]
            ready_root_quat_tensor = self.motion.body_quat_w[
                ready_steps, 0
            ]
        ready_root_z = tuple(
            float(value)
            for value in ready_root_z_tensor.detach().cpu().tolist()
        )
        ready_root_quat = tuple(
            tuple(float(component) for component in row)
            for row in ready_root_quat_tensor.detach().cpu().tolist()
        )
        # Cache this immutable admitted-motion fact at construction.  The
        # production reveal leaf must never add a per-reveal device readback
        # merely to recover a segment length already known here.
        segment_lengths = tuple(
            int(value)
            for value in self.motion.seg_len.detach().cpu().tolist()
        )
        if (
            len(segment_lengths) != action_count
            or any(length < 3 for length in segment_lengths)
        ):
            raise ValueError(
                "action-ball admitted motions require one interior frame per action"
            )
        # Publish only after every opaque admission and file/broker row has passed.  The hard
        # receipt immediately reopens the capability and all implementation sources once more.
        self._action_ball_birth_broker = broker
        self._action_ball_runtime_module_bound = runtime
        self._action_ball_trusted_repo_root = repo_root
        self._action_ball_action_uids = action_uids
        self._action_ball_motion_sha256 = tuple(motion_sha256)
        self._action_ball_segment_lengths = segment_lengths
        self._action_ball_ready_root_z = ready_root_z
        self._action_ball_ready_root_quat = ready_root_quat
        self._action_ball_reset_generation = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._action_ball_swing_generation = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._action_ball_birth_receipt_sha256 = [None] * self.num_envs
        self._action_ball_seen_birth_receipts = set()
        self._action_ball_active_task_refs = [None] * self.num_envs
        self._action_ball_task_timing_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Racket owns task reveal, but Motion owns every teacher/mimic tensor.
        # Racket binds its exact public task-valid tensor after construction so
        # RESET_WAIT can use the physical safe-ready reference and the reveal
        # tick can switch atomically to the measured clip frame 0.  No
        # safe-ready -> frame0 interpolation is a runtime authority.
        self._action_ball_public_task_valid = None
        body_shape = (self.num_envs, len(self.body_indexes), 3)
        quat_shape = (self.num_envs, len(self.body_indexes), 4)
        self._action_ball_safe_ready_body_pos_w = torch.zeros(
            body_shape,
            dtype=self.motion.body_pos_w.dtype,
            device=self.device,
        )
        self._action_ball_safe_ready_body_quat_w = torch.zeros(
            quat_shape,
            dtype=self.motion.body_quat_w.dtype,
            device=self.device,
        )
        self._action_ball_safe_ready_reference_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._action_ball_safe_ready_pending_count = 0
        self._action_ball_single_stroke_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Diagnostic Racket resolves every reset/wrap selection synchronously in
        # the same command-manager pass.  Keep that host-known batch state here
        # so ordinary active steps do not rediscover an empty pending set with a
        # device-wide ``torch.where`` and host length check.
        self._action_ball_diagnostic_pending_row_count = 0
        timing_shape = (self.num_envs,)
        timing_options = {
            "dtype": torch.float64,
            "device": self.device,
        }
        self._action_ball_task_pending_elapsed_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_task_age_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_time_to_contact_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_teacher_rate = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_scaled_t_hit_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_scaled_t_cycle_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_pre_swing_wait_s = torch.zeros(
            timing_shape, **timing_options
        )
        try:
            receipt = self.action_ball_motion_admission_hard_contract()
        except Exception:
            self._action_ball_birth_broker = None
            self._action_ball_runtime_module_bound = None
            self._action_ball_trusted_repo_root = None
            self._action_ball_action_uids = None
            self._action_ball_motion_sha256 = None
            self._action_ball_segment_lengths = None
            self._action_ball_ready_root_z = None
            self._action_ball_ready_root_quat = None
            self._action_ball_reset_generation = None
            self._action_ball_swing_generation = None
            self._action_ball_birth_receipt_sha256 = None
            self._action_ball_seen_birth_receipts = None
            self._action_ball_active_task_refs = None
            self._action_ball_task_timing_active = None
            self._action_ball_public_task_valid = None
            self._action_ball_safe_ready_body_pos_w = None
            self._action_ball_safe_ready_body_quat_w = None
            self._action_ball_safe_ready_reference_pending = None
            self._action_ball_safe_ready_pending_count = None
            self._action_ball_single_stroke_complete = None
            self._action_ball_diagnostic_pending_row_count = None
            self._action_ball_task_pending_elapsed_s = None
            self._action_ball_task_age_s = None
            self._action_ball_time_to_contact_s = None
            self._action_ball_teacher_rate = None
            self._action_ball_scaled_t_hit_s = None
            self._action_ball_scaled_t_cycle_s = None
            self._action_ball_pre_swing_wait_s = None
            raise
        self._action_ball_motion_admission_receipt_sha256 = receipt[
            "canonical_sha256"
        ]

    def bind_action_ball_public_task_valid(
        self, task_valid: torch.Tensor
    ) -> None:
        """Share Racket's sole WAIT/reveal bit with teacher-reference accessors."""

        if not self.action_ball_diagnostic_split_ready_teacher:
            return
        if (
            not torch.is_tensor(task_valid)
            or task_valid.dtype != torch.bool
            or tuple(task_valid.shape) != (self.num_envs,)
            or task_valid.device != torch.device(self.device)
        ):
            raise ValueError(
                "split-ready task_valid must be one bool tensor on Motion's device"
            )
        if (
            self._action_ball_public_task_valid is not None
            and self._action_ball_public_task_valid is not task_valid
        ):
            raise RuntimeError(
                "split-ready task_valid authority may be bound exactly once"
            )
        self._action_ball_public_task_valid = task_valid

    def refresh_action_ball_revealed_body_reference(
        self, reveal: torch.Tensor
    ) -> None:
        """Refresh cached aligned bodies on an exact public reveal tick.

        Motion updates before RacketTargetCommand.  Racket owns ``task_valid``
        and may therefore reveal a row after Motion has already materialized
        ``body_*_relative_w`` from the safe-ready tuple for this policy tick.
        The fresh direct lane has the same ordering at D05's writer barrier.
        Joint and anchor accessors select lazily, but body rewards consume
        cached aligned tensors.  Refreshing only newly accepted rows keeps the
        complete teacher tuple on measured frame 0 during the same public tick.
        """

        fresh_direct = getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        )
        if not (
            self.action_ball_diagnostic_split_ready_teacher or fresh_direct
        ):
            return
        if (
            not torch.is_tensor(reveal)
            or reveal.dtype != torch.bool
            or tuple(reveal.shape) != (self.num_envs,)
            or reveal.device != torch.device(self.device)
        ):
            raise ValueError(
                "teacher reveal must be one bool tensor on Motion's device"
            )
        pending = getattr(
            self, "_action_ball_safe_ready_reference_pending", None
        )
        if pending is None:
            raise RuntimeError(
                "teacher reveal requires bound safe-ready state"
            )

        # Keep this full-batch and mask-select below.  A CUDA ``nonzero`` /
        # one-argument ``where`` would add a host synchronization to every
        # policy tick merely to discover that most reveal masks are empty.
        steps = self._pose_reference_steps()
        frame_zero = self.motion.seg_start[self.clip_id]
        reveal_fault = reveal & (pending | steps.ne(frame_zero))
        fault_latch = getattr(
            self,
            "_latch_action_ball_full_mdp_motion_epoch_row_fault",
            None,
        )
        if callable(fault_latch):
            writable_rows = fault_latch(
                reveal_fault.contiguous(),
                reason_bit=(
                    _ACTION_EPOCH_ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT
                ),
            )
        elif fresh_direct:
            raise RuntimeError(
                "fresh Motion reveal reference requires its exact ActionEpoch owner"
            )
        else:
            # Extracted legacy diagnostic fixtures intentionally do not carry
            # the production ActionEpoch helper.  They retain the old local
            # quarantine semantics; only the fresh direct lane is fail-closed.
            writable_rows = ~reveal_fault
        reveal = reveal & writable_rows
        # The named fault intentionally accepts an invalid/out-of-range step
        # as input.  Never pass that rejected index to a CUDA gather: rows not
        # being refreshed use a legal frame-0 placeholder whose values are
        # masked away below.
        safe_steps = torch.where(reveal, steps, frame_zero)

        body_pos_w = (
            self.motion.body_pos_w[safe_steps]
            + self._env.scene.env_origins[:, None, :]
        )
        body_quat_w = self.motion.body_quat_w[safe_steps]
        anchor_pos_w = body_pos_w[:, self.motion_anchor_body_index]
        anchor_quat_w = body_quat_w[:, self.motion_anchor_body_index]
        robot_anchor_pos_w = self.robot.data.body_pos_w[
            :, self.robot_anchor_body_index
        ]
        robot_anchor_quat_w = self.robot.data.body_quat_w[
            :, self.robot_anchor_body_index
        ]
        (
            measured_body_quat_relative_w,
            measured_body_pos_relative_w,
        ) = _motion_anchor_relative_body_transform(
            anchor_pos_w,
            anchor_quat_w,
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            body_pos_w,
            body_quat_w,
            expected_body_count=len(self.cfg.body_names),
        )
        self.body_quat_relative_w = torch.where(
            reveal[:, None, None],
            measured_body_quat_relative_w,
            self.body_quat_relative_w,
        )
        self.body_pos_relative_w = torch.where(
            reveal[:, None, None],
            measured_body_pos_relative_w,
            self.body_pos_relative_w,
        )

    def _action_ball_full_mdp_initial_balance_reference_mask(
        self,
    ) -> torch.Tensor:
        """Rows whose teacher is the episode's frozen reset-ready tuple.

        The existing policy-opportunity counter is reset by the selected-reset
        transaction and increments only under D05 ACCEPT.  It therefore gives
        the exact curriculum boundary without a new selector, caller verdict
        or task gate: censored reveals remain balance rows; accepted, playback
        and completed-recovery rows use the selected action reference.
        """

        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        created = getattr(
            self,
            "_action_ball_continuous_policy_opportunities_created",
            None,
        )
        if (
            type(created) is not torch.Tensor
            or created.dtype != torch.int64
            or created.device != torch.device(self.device)
            or tuple(created.shape) != (self.num_envs,)
        ):
            raise RuntimeError(
                "fresh Motion balance-reference counter ABI differs"
            )
        return created.eq(0)

    def _action_ball_safe_ready_wait_mask(self) -> torch.Tensor:
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            # Reset observations may read teacher properties before the next
            # command update.  Capture once from the settled plant, never from
            # the policy's moving current pose thereafter.
            self._capture_action_ball_safe_ready_reference()
            return self._action_ball_full_mdp_initial_balance_reference_mask()
        if not self.action_ball_diagnostic_split_ready_teacher:
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        # Reset observations can consume a teacher property before the next
        # command-manager update.  Materialize the settled physical tuple at
        # the first such access, independent of observation-term ordering.
        self._capture_action_ball_safe_ready_reference()
        task_valid = getattr(self, "_action_ball_public_task_valid", None)
        if task_valid is None:
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if self.action_ball_continuous_motion_enabled:
            # The pre-Q0 hidden wait still uses the separately admitted
            # physical safe-ready tuple.  After a real shot has completed its
            # full suffix, however, recovery is supervised against that
            # completed action's frame 0 with zero reference velocity.  Hiding
            # the future task must not silently switch the teacher back to the
            # episode-birth tuple.
            completed_action_ready = (
                self._action_ball_continuous_suffix_complete
                & self._action_ball_continuous_ready_reference_active
            )
            return (~task_valid) & ~completed_action_ready
        return ~task_valid

    def action_ball_full_mdp_playback_active_mask(self) -> torch.Tensor:
        """Return Motion's sole FullMDP playback-started row mask.

        This zero-copy public owner view is consumed by reward and telemetry.
        It exposes no new lifecycle state and does not infer playback from task
        validity, reward values, reference steps, physical outcome, or
        recovery.  Motion turns it on only after the teacher leaves selected
        frame 0, and clears it at the suffix-hidden transition.  Prepare,
        ready, and recovery are therefore false; canonical swing and
        follow-through are true.
        """

        if not getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            raise RuntimeError(
                "FullMDP playback-active telemetry requires the fresh Motion lane"
            )
        active = getattr(
            self, "_action_ball_continuous_canonical_playback_started", None
        )
        if (
            type(active) is not torch.Tensor
            or active.dtype != torch.bool
            or active.device != torch.device(self.device)
            or tuple(active.shape) != (self.num_envs,)
            or not active.is_contiguous()
        ):
            raise RuntimeError(
                "FullMDP playback-started Motion tensor changed ABI"
            )
        return active

    def action_ball_split_ready_hold_command(self):
        """WAIT-phase ``(mask, executable q_des)`` for anything that DRIVES the plant.

        人话:等待阶段能把出生姿态撑住的是"保持 q_des",不是"参考姿态"本身。
        谁在等待阶段直接驱动机器人,就必须发这个,不能把参考姿态当指令发下去。

        ``joint_pos`` deliberately returns the physical BIRTH configuration during the
        pre-task WAIT — that is where the robot is supposed to physically BE, so it is
        the correct imitation REFERENCE.  It is not a command.  A zero-PPO driver that
        sends it as ``q_des`` makes every PD error exactly zero, so the gravity-holding
        torque the contract's LP solved for is never produced (for
        ``take_061_unit04_bh`` that discards 36.5 N*m at ``right_hip_roll``, 18.7 N*m at
        ``waist_pitch``, 15.7 N*m at ``left_ankle_pitch``) and the robot sags out of its
        stance inside half a second.  ``hold_qdes_joint_pos_rad`` is that same LP
        solution expressed as a ``q_des`` — ``kp * (hold_qdes - q_birth)`` reproduces the
        artifact's holding torque exactly — and is already what the reset bootstrap
        installs into the action term.  A driver must keep sending it for the whole WAIT.

        Returns ``None`` when this is not a split-ready diagnostic: there is then no
        separate birth pose and nothing to substitute.  Raises when it IS one but the
        hold ``q_des`` binding is absent, because such a driver has no safe command.
        """

        if not self.action_ball_diagnostic_split_ready_teacher:
            return None
        hold_qdes = getattr(
            self, "_action_ball_dynamic_ready_hold_qdes_joint_pos_rad", None
        )
        if not torch.is_tensor(hold_qdes):
            raise RuntimeError(
                "split-ready diagnostic has no dynamic-ready hold q_des to command "
                "during the pre-task WAIT"
            )
        return self._action_ball_safe_ready_wait_mask(), hold_qdes[self.clip_id]

    def action_ball_teacher_start_frozen_steps(self) -> torch.Tensor:
        """How many MORE control steps the teacher reference stays frozen.

        人话:揭示之后老师还要在 frame0 上冻结几步 —— 这就是"桥接窗口"还剩多长。
        谁在这段里驱动机器人,就该用它把指令铺开,而不是一步跨过去。

        This is the integer form of the clock :attr:`teacher_start_wait_remaining_s`
        already publishes to the actor, and it is the same counter
        :meth:`_install_event_motion` writes at the atomic reveal
        (``hold_counter = step.install_hold_steps``).  While it is positive
        :attr:`joint_pos` returns measured frame 0 unchanged, so a driver that reads
        it knows exactly how many steps it has to travel the split-ready -> frame 0
        gap before the clip starts advancing.  Zero means the reference advances on
        the next step: there is no window left and a command must already be there.

        Returned per environment as ``torch.long``, same ordering as
        :meth:`action_ball_split_ready_hold_command`.
        """

        if self.hold_counter.shape != (self.num_envs,):
            raise RuntimeError(
                "MotionCommand hold_counter must have one scalar per environment"
            )
        if self.hold_counter.dtype != torch.long:
            raise RuntimeError("MotionCommand hold_counter must use torch.long steps")
        return self.hold_counter.clamp_min(0)

    def _capture_action_ball_safe_ready_reference(self) -> None:
        """Freeze FK body targets after the physical safe-ready reset settles."""

        pending = getattr(
            self, "_action_ball_safe_ready_reference_pending", None
        )
        count = getattr(self, "_action_ball_safe_ready_pending_count", None)
        if count is None and pending is None:
            # 人话:两者同时是 None 表示**任务权威还没绑定** —— 这是构造期(以及绑定
            # 失败后的清零态)的合法状态,此时"没有待冻结的 env",不是"待冻结但掩码
            # 丢了"。原来用 `count == 0` 判空,而 None != 0,于是 ObservationManager
            # 在 gym.make 里探测观测维度的那次干调用就被误判成"掩码缺失"并抛错 ——
            # A211/C211 四格从 materialize 走到 recipe 就是死在这一行,一次都没建成过环境。
            # 只有一边是 None 仍然是硬错:那才是真正的半初始化。
            return
        if pending is None or count is None:
            raise RuntimeError(
                "safe-ready reference pending state is half-initialized"
            )
        if count == 0:
            return
        writable = getattr(
            self,
            "_action_ball_full_mdp_motion_epoch_writable_rows",
            None,
        )
        capture_rows = (
            pending
            if not torch.is_tensor(writable)
            else (pending & writable)
        )
        ids = torch.where(capture_rows)[0]
        body_pos = self.robot.data.body_pos_w[ids][:, self.body_indexes]
        body_quat = self.robot.data.body_quat_w[ids][:, self.body_indexes]
        if (
            tuple(body_pos.shape)
            != tuple(self._action_ball_safe_ready_body_pos_w[ids].shape)
            or tuple(body_quat.shape)
            != tuple(self._action_ball_safe_ready_body_quat_w[ids].shape)
            or not bool(torch.isfinite(body_pos).all())
            or not bool(torch.isfinite(body_quat).all())
        ):
            raise RuntimeError(
                "physical FK reset-ready reference is unavailable after reset"
            )
        self._action_ball_safe_ready_body_pos_w[ids] = body_pos
        self._action_ball_safe_ready_body_quat_w[ids] = body_quat
        # Reset may request its first observation before Motion's next command
        # update.  Keep the materialized body cache on the same newly captured
        # physical tuple so body and paddle consumers cannot see the previous
        # episode while joint targets already expose the new safe-ready pose.
        self.body_pos_relative_w[ids] = body_pos
        self.body_quat_relative_w[ids] = body_quat
        # A terminal fault row must not mutate its cached teacher.  The packed
        # optimizer drain will stop the run, so the one-shot request is cleared
        # for all rows after healthy rows have been frozen.
        pending.zero_()
        self._action_ball_safe_ready_pending_count = 0

    def bind_action_ball_task_authority(
        self, *, task_ref_for_env, resolve_task_ref, shared_state_sha256
    ) -> None:
        """Bind Racket's opaque ref/resolve and shared-digest authority seams exactly once."""

        if self._action_ball_birth_broker is None:
            raise RuntimeError(
                "action-ball task authority requires the birth broker first"
            )
        if (
            self._action_ball_task_ref_for_env is not None
            or self._action_ball_task_receipt_resolver is not None
            or self._action_ball_shared_state_sha256_accessor is not None
        ):
            raise ValueError("action-ball task authority may be bound exactly once")
        if (
            not callable(task_ref_for_env)
            or getattr(task_ref_for_env, "__name__", None)
            != "action_ball_task_ref_for_env"
            or not callable(resolve_task_ref)
            or getattr(resolve_task_ref, "__name__", None)
            != "action_ball_resolve_task_ref"
            or not callable(shared_state_sha256)
            or getattr(shared_state_sha256, "__name__", None)
            != "action_ball_shared_state_sha256"
        ):
            raise ValueError(
                "action-ball task authority requires the exact public Racket accessors"
            )
        ref_owner = getattr(task_ref_for_env, "__self__", None)
        resolver_owner = getattr(resolve_task_ref, "__self__", None)
        digest_owner = getattr(shared_state_sha256, "__self__", None)
        if (
            ref_owner is None
            or ref_owner is not resolver_owner
            or ref_owner is not digest_owner
        ):
            raise ValueError(
                "action-ball task ref, resolver and shared digest must have one bound owner"
            )
        if self.planner_revision_enabled:
            raise ValueError(
                "action-ball receipt timing is the sole phase/deadline owner"
            )
        if self._speed_per_clip is not None or tuple(
            float(value) for value in self.cfg.speed_scale_range
        ) != (1.0, 1.0):
            raise ValueError(
                "action-ball teacher_rate requires native generic speed configuration"
            )
        if (
            tuple(int(value) for value in self.cfg.hold_steps_range)
            != (0, 0)
            or int(self.cfg.stand_start_min_hold) != 0
            or int(self.cfg.post_swing_min_hold) != 0
            or bool(self.cfg.stagger_initial_clock)
        ):
            raise ValueError(
                "action-ball task receipt owns preparation wait; legacy hold/stagger must be zero"
            )
        self._action_ball_task_ref_for_env = task_ref_for_env
        self._action_ball_task_receipt_resolver = resolve_task_ref
        self._action_ball_shared_state_sha256_accessor = shared_state_sha256
        # Dynamic teacher rates use the existing audited velocity-scaling lane, but their values
        # come only from the current immutable task receipt (never the generic speed sampler).
        self.retiming_active = True
        self._action_ball_expected_shared_racket_state_sha256 = None

    def bind_action_ball_fixed_view_timing(
        self,
        *,
        fixed_view_identity_sha256: str,
        timing_row: tuple,
        broker_exact_state,
    ) -> None:
        """Bind one prevalidated immutable-N1 timing row without task receipts.

        Birth receipts remain the episode/root identity authority.  This seam
        replaces only the per-swing ``ActionBallTaskReceipt`` resolver for the
        explicitly unauthorized, single-action immutable-tape diagnostic.
        """

        if (
            self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
            or self._action_ball_task_ref_for_env is None
            or self._action_ball_task_receipt_resolver is None
            or self._action_ball_shared_state_sha256_accessor is None
        ):
            raise RuntimeError(
                "fixed-view timing requires the diagnostic ActionBall birth/task binding"
            )
        if (
            self._action_ball_fixed_view_identity_sha256 is not None
            or self._action_ball_fixed_view_timing_row is not None
            or self._action_ball_fixed_view_timing_row_device is not None
            or self._action_ball_fixed_view_broker_state_accessor is not None
        ):
            raise ValueError("fixed-view timing may be bound exactly once")
        if (
            self._action_ball_action_uids is None
            or len(self._action_ball_action_uids) != 1
            or self._action_ball_segment_lengths is None
            or len(self._action_ball_segment_lengths) != 1
        ):
            raise ValueError("fixed-view timing requires exact ActionBall N=1")
        identity = self._action_ball_sha256(
            fixed_view_identity_sha256,
            name="fixed_view_identity_sha256",
        )
        validated = self._validate_action_ball_fixed_view_timing_row(timing_row)
        if not callable(broker_exact_state):
            raise TypeError(
                "fixed-view timing requires a callable exact broker-state accessor"
            )
        task_owner = getattr(
            self._action_ball_task_ref_for_env, "__self__", None
        )
        broker_state_owner = getattr(broker_exact_state, "__self__", None)
        if task_owner is None or broker_state_owner is not task_owner:
            raise ValueError(
                "fixed-view timing and exact broker state require one Racket owner"
            )
        self._action_ball_fixed_view_identity_sha256 = identity
        self._action_ball_fixed_view_timing_row = validated
        self._action_ball_fixed_view_timing_row_device = torch.tensor(
            validated,
            dtype=self._action_ball_task_age_s.dtype,
            device=self.device,
        ).reshape(1, 5)
        self._action_ball_fixed_view_broker_state_accessor = (
            broker_exact_state
        )

    @property
    def action_ball_fixed_view_enabled(self) -> bool:
        return self._action_ball_fixed_view_identity_sha256 is not None

    def validate_action_ball_task_authority_binding(self) -> None:
        """Probe the shared Racket digest after both runtime owners are published."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="task authority binding validation"
            )
        self.action_ball_shared_racket_state_sha256()

    @property
    def action_ball_enabled(self) -> bool:
        return self._action_ball_birth_broker is not None

    @property
    def action_ball_ordered_action_uids(self) -> tuple[int, ...]:
        if self._action_ball_action_uids is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_action_uids

    @property
    def action_ball_reset_generation(self) -> torch.Tensor:
        if self._action_ball_reset_generation is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_continuous_public_tensor(
            self._action_ball_reset_generation,
            name="reset-generation",
        )

    @property
    def action_ball_episode_generation(self) -> torch.Tensor:
        """Alias documenting that a reset generation identifies one physical episode."""

        return self.action_ball_reset_generation

    @property
    def action_ball_swing_generation(self) -> torch.Tensor:
        if self._action_ball_swing_generation is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_continuous_public_tensor(
            self._action_ball_swing_generation,
            name="swing-generation",
        )

    def action_ball_action_uid_for_envs(self, env_ids) -> torch.Tensor:
        if self._action_ball_action_uids is None:
            raise RuntimeError("action-ball birth broker is not bound")
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        uid_table = torch.tensor(
            self._action_ball_action_uids,
            dtype=torch.long,
            device=self.device,
        )
        return uid_table[self.clip_id[ids]]

    def action_ball_birth_receipt_sha256(self, env_id: int) -> str:
        env_id = self._action_ball_plain_int(
            env_id, name="env_id", minimum=0
        )
        if env_id >= self.num_envs:
            raise ValueError("env_id is outside the environment batch")
        if self._action_ball_birth_receipt_sha256 is None:
            raise RuntimeError("action-ball birth broker is not bound")
        receipt = self._action_ball_birth_receipt_sha256[env_id]
        if receipt is None:
            raise RuntimeError("environment has no committed action-ball birth")
        return receipt

    def action_ball_motion_admission_hard_contract(self) -> dict:
        """Reopen the opaque training admission and emit a content-addressed receipt."""

        if (
            self._action_ball_birth_broker is None
            or self._action_ball_trusted_repo_root is None
            or self._action_ball_runtime_module_bound is None
        ):
            raise RuntimeError("action-ball motion admission is not bound")
        if self._canonical_diagnostic_unauthorized:
            # No admission exists to reopen.  Emit a content-addressed
            # unauthorized binding receipt so exact-resume and hard-contract
            # identities can still pin the immutable bytes without mistaking
            # them for a training capability.
            payload = {
                "schema_version": 1,
                "kind": (
                    "whole_body_tracking.MotionCommand."
                    "action_ball_motion_diagnostic_binding"
                ),
                "diagnostic_unauthorized": True,
                "motion_file_sha256": list(self._motion_file_sha256),
                "training_authorized": False,
            }
            payload["canonical_sha256"] = hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest()
            return payload
        repo_root = self._action_ball_trusted_repo_root
        self._require_action_ball_motion_admission(repo_root)
        registry = self._canonical_motion_registry
        admission = self._canonical_motion_admission
        promotion_binding = self._canonical_motion_promotion_binding
        registry_module = self._canonical_motion_registry_module
        runtime = self._action_ball_runtime_module_bound

        registry_path, registry_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.path,
            name="canonical registry",
            expected_sha256=registry.registry_sha256,
        )
        ready_path, ready_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.canonical_ready_path,
            name="canonical ready",
            expected_sha256=registry.canonical_ready_sha256,
        )
        ready_fk_path, ready_fk_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.canonical_ready_fk_path,
            name="canonical ready FK",
            expected_sha256=registry.canonical_ready_fk_sha256,
        )
        certificate_path, certificate_sha = (
            self._action_ball_repo_file_receipt(
                repo_root,
                getattr(admission, "_certificate_path", ""),
                name="canonical promotion certificate",
                expected_sha256=admission.certificate_sha256,
            )
        )
        binding_sha = registry_module.motion_admission._binding_sha256(
            promotion_binding
        )

        motion_rows = []
        for slot, motion_id in enumerate(self.canonical_motion_ids):
            binding = self._action_ball_birth_broker.binding_for_slot(slot)
            resolved, digest = self._action_ball_file_receipt(
                repo_root,
                binding.motion_path,
                name=f"canonical motion[{slot}]",
                expected_sha256=binding.motion_sha256,
            )
            if (
                str(resolved) != self._motion_files[slot]
                or digest != self._motion_file_sha256[slot]
            ):
                raise RuntimeError(
                    "action-ball motion bytes changed after opaque admission"
                )
            motion_rows.append(
                {
                    "motion_id": motion_id,
                    "action_uid": binding.action_uid,
                    "action_slot": binding.action_slot,
                    "motion_path": binding.motion_path,
                    "motion_sha256": digest,
                    "profile_sha256": binding.profile_sha256,
                }
            )

        source_paths = {
            "commands": Path(__file__).resolve(strict=True),
            "action_ball_runtime": Path(runtime.__file__).resolve(strict=True),
            "canonical_motion_registry": Path(
                registry_module.__file__
            ).resolve(strict=True),
            "canonical_motion_admission": Path(
                registry_module.motion_admission.__file__
            ).resolve(strict=True),
        }
        implementation_sources = {}
        for name, path in source_paths.items():
            relative, digest = self._action_ball_repo_file_receipt(
                repo_root,
                path,
                name=f"implementation source {name}",
            )
            implementation_sources[name] = {
                "path": relative,
                "sha256": digest,
            }

        payload = {
            "schema_version": 1,
            "kind": (
                "whole_body_tracking.MotionCommand."
                "action_ball_motion_admission"
            ),
            "authorization_purpose": "training",
            "trusted_repo_root": str(repo_root),
            "opaque_capability": {
                "capability_type": type(admission).__name__,
                "purpose": admission.purpose,
                "promotion_binding_sha256": binding_sha,
                "certificate_path": certificate_path,
                "certificate_sha256": certificate_sha,
            },
            "canonical_bank": {
                "bank_id": registry.bank_id,
                "scope": registry.scope,
                "registry_path": registry_path,
                "registry_sha256": registry_sha,
                "alignment_sha256": (
                    self.canonical_registry_alignment_sha256
                ),
                "canonical_ready_path": ready_path,
                "canonical_ready_sha256": ready_sha,
                "canonical_ready_fk_path": ready_fk_path,
                "canonical_ready_fk_sha256": ready_fk_sha,
                "motion_rows": motion_rows,
            },
            "runtime_binding": {
                "runtime_contract_sha256": runtime.RUNTIME_CONTRACT_SHA256,
                "broker_state_schema_version": (
                    runtime.BROKER_STATE_SCHEMA_VERSION
                ),
                "broker_registry_sha256": (
                    self._action_ball_birth_broker.registry_sha256
                ),
                "provider_state_owner_sha256": (
                    self._action_ball_birth_broker.provider_state_owner_sha256
                ),
                "ordered_action_uids": list(
                    self._action_ball_action_uids
                ),
                "manifest_rows_are_identity_only": True,
            },
            "implementation_sources": implementation_sources,
        }
        payload["canonical_sha256"] = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        return payload

    def _validate_action_ball_birth_receipt(
        self,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        action_slot: int,
        action_uid: int,
    ) -> tuple[
        str,
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        runtime = self._action_ball_runtime_module_bound
        if type(receipt) is not runtime.ActionBirthReceipt:
            raise ValueError("action-ball birth receipt has a forged runtime type")
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
        ):
            raise ValueError(
                "action-ball birth does not match the batched reset request"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(action_slot)
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
            or receipt.profile_sha256 != binding.profile_sha256
        ):
            raise ValueError(
                "action-ball birth does not match its broker motion/profile registry"
            )
        receipt_sha = self._action_ball_sha256(
            receipt.canonical_sha256,
            name="birth.canonical_sha256",
        )
        if receipt_sha in self._action_ball_seen_birth_receipts:
            raise ValueError("action-ball birth receipt replay detected")
        spawn = self._action_ball_vector(
            receipt.base_spawn_w_m,
            name="birth.base_spawn_w_m",
            length=3,
        )
        quat = self._action_ball_vector(
            receipt.base_quat_wxyz,
            name="birth.base_quat_wxyz",
            length=4,
        )
        ready_z = self._action_ball_ready_root_z[action_slot]
        if not math.isclose(
            spawn[2], ready_z, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise ValueError(
                "action-ball birth Z differs from canonical-ready root Z"
            )
        # ``base_quat_wxyz`` is the yaw-only B_yaw frame used by the sampler
        # and solver.  The physical floating-base reset keeps the admitted
        # clip's complete ready quaternion (including real roll/pitch), so
        # compare the receipt with that quaternion's yaw projection here.
        # Conflating these two frames strips the ready pitch and moves the
        # paddle/feet by centimetres even though the joint vector is exact.
        ready_root_quat = self._action_ball_ready_root_quat[action_slot]
        rw, rx, ry, rz = ready_root_quat
        ready_yaw = math.atan2(
            2.0 * (rw * rz + rx * ry),
            1.0 - 2.0 * (ry * ry + rz * rz),
        )
        ready_frame_quat = (
            math.cos(0.5 * ready_yaw),
            0.0,
            0.0,
            math.sin(0.5 * ready_yaw),
        )
        direct = max(abs(a - b) for a, b in zip(quat, ready_frame_quat))
        negated = max(abs(a + b) for a, b in zip(quat, ready_frame_quat))
        if min(direct, negated) > 1.0e-6:
            raise ValueError(
                "action-ball birth yaw frame differs from canonical-ready root yaw"
            )
        return receipt_sha, spawn, quat

    def _rollback_action_ball_broker(
        self, state: dict, *, original_error: BaseException
    ) -> None:
        try:
            self._action_ball_birth_broker.load_state_dict(state)
            if self._action_ball_birth_broker.state_dict() != state:
                raise RuntimeError(
                    "broker rollback did not restore exact state"
                )
        except Exception as rollback_error:
            raise RuntimeError(
                "action-ball batch failed and broker/provider/domain rollback failed"
            ) from rollback_error

    def _reserve_action_ball_true_reset(
        self, env_ids: torch.Tensor
    ) -> dict:
        if self._action_ball_birth_broker is None:
            raise RuntimeError("action-ball birth broker is not bound")
        if env_ids.ndim != 1 or len(env_ids) == 0:
            raise ValueError(
                "action-ball reset batch requires unique non-empty env ids"
            )
        env_rows = tuple(
            int(value) for value in env_ids.detach().cpu().tolist()
        )
        if len(set(env_rows)) != len(env_rows):
            raise ValueError(
                "action-ball reset batch requires unique non-empty env ids"
            )
        current = self._action_ball_reset_generation[env_ids]
        if bool((current >= self._ACTION_BALL_INT64_MAX).any()):
            raise OverflowError("action-ball reset generation exhausted")
        next_generation = current + 1
        runtime = self._action_ball_runtime_module_bound
        action_slot_rows = tuple(
            int(value)
            for value in self.clip_id[env_ids].detach().cpu().tolist()
        )
        generation_rows = tuple(
            int(value)
            for value in next_generation.detach().cpu().tolist()
        )
        requests = []
        request_rows = []
        for env_id, action_slot, generation in zip(
            env_rows, action_slot_rows, generation_rows
        ):
            action_uid = self._action_ball_action_uids[action_slot]
            requests.append(
                runtime.BirthReserveRequest(
                    env_id=env_id,
                    reset_generation=generation,
                    action_uid=action_uid,
                    action_slot=action_slot,
                )
            )
            request_rows.append(
                (env_id, generation, action_slot, action_uid)
            )

        diagnostic_fast_path = (
            self._action_ball_birth_broker.diagnostic_fast_path
        )
        broker_state_before = (
            None
            if diagnostic_fast_path
            else self._action_ball_birth_broker.state_dict()
        )
        try:
            receipts = self._action_ball_birth_broker.reserve_many_true_reset(
                tuple(requests), reset_kind="true_reset"
            )
            if (
                type(receipts) is not tuple
                or len(receipts) != len(requests)
            ):
                raise ValueError(
                    "action-ball broker returned a partial reset batch"
                )
            receipt_sha256 = []
            spawn_rows = []
            quat_rows = []
            for receipt, request_row in zip(receipts, request_rows):
                env_id, generation, action_slot, action_uid = request_row
                receipt_sha, spawn, quat = (
                    self._validate_action_ball_birth_receipt(
                        receipt,
                        env_id=env_id,
                        reset_generation=generation,
                        action_slot=action_slot,
                        action_uid=action_uid,
                    )
                )
                pending = self._action_ball_birth_broker.pending_receipt(
                    env_id=env_id,
                    reset_generation=generation,
                    action_uid=action_uid,
                    action_slot=action_slot,
                    reset_kind="true_reset",
                )
                if pending is not receipt:
                    raise ValueError(
                        "action-ball broker changed a reserved receipt object"
                    )
                receipt_sha256.append(receipt_sha)
                spawn_rows.append(spawn)
                quat_rows.append(quat)
            if len(set(receipt_sha256)) != len(receipt_sha256):
                raise ValueError(
                    "action-ball broker replayed one birth within a reset batch"
                )
            spawn = torch.tensor(
                spawn_rows,
                dtype=self.motion.body_pos_w.dtype,
                device=self.device,
            )
            quat = torch.tensor(
                quat_rows,
                dtype=self.motion.body_quat_w.dtype,
                device=self.device,
            )
            if (
                tuple(spawn.shape) != (len(env_ids), 3)
                or tuple(quat.shape) != (len(env_ids), 4)
                or not bool(torch.isfinite(spawn).all())
                or not bool(torch.isfinite(quat).all())
            ):
                raise ValueError(
                    "action-ball broker returned a malformed root batch"
                )
        except Exception as exc:
            if not diagnostic_fast_path:
                self._rollback_action_ball_broker(
                    broker_state_before, original_error=exc
                )
            raise
        if diagnostic_fast_path:
            # A diagnostic exception poisons the run instead of restoring it.
            # Do not clone formal Motion rollback tensors or whole-run
            # containers on every successful short episode.
            return {
                "receipts": receipts,
                "receipt_sha256": tuple(receipt_sha256),
                "request_rows": tuple(request_rows),
                "next_generation": next_generation,
                "spawn": spawn,
                "quat": quat,
            }
        # Preserve the formal transaction's exact field order and values.
        return {
            "broker_state_before": broker_state_before,
            "receipts": receipts,
            "receipt_sha256": tuple(receipt_sha256),
            "request_rows": tuple(request_rows),
            "next_generation": next_generation,
            "spawn": spawn,
            "quat": quat,
            "motion_reset_generation_before": current.clone(),
            "motion_swing_generation_before": (
                self._action_ball_swing_generation[env_ids].clone()
            ),
            "motion_birth_receipts_before": list(
                self._action_ball_birth_receipt_sha256
            ),
            "motion_seen_receipts_before": set(
                self._action_ball_seen_birth_receipts
            ),
        }

    def _rollback_action_ball_true_reset(
        self,
        env_ids: torch.Tensor,
        transaction: dict,
        *,
        original_error: BaseException,
    ) -> None:
        if self._action_ball_birth_broker.diagnostic_fast_path:
            raise RuntimeError(
                "diagnostic action-ball reset is fail-stop and cannot be "
                "rolled back or retried"
            ) from original_error
        rollback_error = None
        broker_state_before = transaction["broker_state_before"]
        if broker_state_before is not None:
            try:
                self._rollback_action_ball_broker(
                    broker_state_before,
                    original_error=original_error,
                )
            except Exception as exc:
                rollback_error = exc
        # Restore Motion's publication fields even when a broken callback prevents broker
        # rollback, so no prefix of the batch is presented as a committed local episode.
        self._action_ball_reset_generation[env_ids] = transaction[
            "motion_reset_generation_before"
        ]
        self._action_ball_swing_generation[env_ids] = transaction[
            "motion_swing_generation_before"
        ]
        self._action_ball_birth_receipt_sha256 = list(
            transaction["motion_birth_receipts_before"]
        )
        self._action_ball_seen_birth_receipts = set(
            transaction["motion_seen_receipts_before"]
        )
        if rollback_error is not None:
            raise RuntimeError(
                "action-ball reset failed and exact transaction rollback failed"
            ) from rollback_error

    def _commit_action_ball_true_reset(
        self, env_ids: torch.Tensor, transaction: dict
    ) -> None:
        runtime = self._action_ball_runtime_module_bound
        receipt_sha256 = transaction["receipt_sha256"]
        next_generation = transaction["next_generation"]
        request_rows = transaction["request_rows"]
        if (
            len(env_ids) != len(receipt_sha256)
            or len(request_rows) != len(receipt_sha256)
            or tuple(next_generation.shape) != (len(env_ids),)
        ):
            raise RuntimeError(
                "action-ball commit batch is internally inconsistent"
            )
        requests = tuple(
            runtime.BirthCommitRequest(
                env_id=env_id,
                reset_generation=generation,
                receipt_sha256=receipt_sha256[index],
            )
            for index, (
                env_id,
                generation,
                _action_slot,
                _action_uid,
            ) in enumerate(request_rows)
        )
        # Validate every pending identity before the simulator mutation is declared committed.
        for index, (
            env_id,
            generation,
            action_slot,
            action_uid,
        ) in enumerate(request_rows):
            pending = self._action_ball_birth_broker.pending_receipt(
                env_id=env_id,
                reset_generation=generation,
                action_uid=action_uid,
                action_slot=action_slot,
                reset_kind="true_reset",
            )
            if pending.canonical_sha256 != receipt_sha256[index]:
                raise RuntimeError(
                    "action-ball pending receipt drifted before atomic commit"
                )
        self._action_ball_birth_broker.commit_many_true_reset(
            requests, reset_kind="true_reset"
        )

        updated_receipts = list(self._action_ball_birth_receipt_sha256)
        updated_seen = set(self._action_ball_seen_birth_receipts)
        for (env_id, _generation, _slot, _uid), receipt_sha in zip(
            request_rows, receipt_sha256
        ):
            if self._action_ball_birth_broker.diagnostic_fast_path:
                previous = updated_receipts[env_id]
                if previous is not None:
                    updated_seen.discard(previous)
            updated_receipts[env_id] = receipt_sha
            updated_seen.add(receipt_sha)
        self._action_ball_reset_generation[env_ids] = next_generation
        self._action_ball_swing_generation[env_ids] = 0
        self._action_ball_birth_receipt_sha256 = updated_receipts
        self._action_ball_seen_birth_receipts = updated_seen

    def _advance_action_ball_wrap_generation(
        self, env_ids: torch.Tensor
    ) -> None:
        current = self._action_ball_swing_generation[env_ids]
        if bool((current >= self._ACTION_BALL_INT64_MAX).any()):
            raise OverflowError("action-ball swing generation exhausted")
        self._action_ball_swing_generation[env_ids] = current + 1

    def _begin_action_ball_task_pending(
        self, env_ids: torch.Tensor, *, elapsed_s: float
    ) -> None:
        """Invalidate the prior swing locally until Racket publishes the new frozen receipt."""

        if (
            self._action_ball_task_ref_for_env is None
            or self._action_ball_task_receipt_resolver is None
        ):
            raise RuntimeError(
                "action-ball reset reached task timing before Racket authority was bound"
            )
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "action-ball pending task elapsed time must be finite and non-negative"
            )
        diagnostic_fast_path = (
            self._action_ball_birth_broker is not None
            and self._action_ball_birth_broker.diagnostic_fast_path
        )
        if not diagnostic_fast_path:
            # Formal keeps the opaque host ref lifecycle and its existing
            # selected-id readback unchanged.
            env_rows = tuple(
                int(value) for value in env_ids.detach().cpu().tolist()
            )
            for env_id in env_rows:
                self._action_ball_active_task_refs[env_id] = None
        self._action_ball_task_timing_active[env_ids] = False
        if not diagnostic_fast_path:
            self._action_ball_task_pending_elapsed_s[env_ids] = elapsed
            self._action_ball_task_age_s[env_ids] = 0.0
            self._action_ball_time_to_contact_s[env_ids] = 0.0
            self._action_ball_teacher_rate[env_ids] = 0.0
            self._action_ball_scaled_t_hit_s[env_ids] = 0.0
            self._action_ball_scaled_t_cycle_s[env_ids] = 0.0
            self._action_ball_pre_swing_wait_s[env_ids] = 0.0
        # Until the exact task arrives the admitted canonical-ready pose is the only safe target.
        self.time_steps[env_ids] = self.motion.seg_start[
            self.clip_id[env_ids]
        ]
        self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
        self.speed_scale[env_ids] = 0.0
        self.hold_counter[env_ids] = 1
        self.metrics["in_hold"][env_ids] = 1.0
        if self.action_ball_diagnostic_split_ready_teacher:
            if self._action_ball_safe_ready_reference_pending is None:
                raise RuntimeError(
                    "split-ready safe reference buffers are not bound"
                )
            self._action_ball_safe_ready_reference_pending[env_ids] = True
            self._action_ball_safe_ready_pending_count += int(env_ids.numel())
        if diagnostic_fast_path:
            # Racket will reuse its one identity D2H to replace the inactive
            # host refs and install every final timing column.  Until then the
            # false active mask makes the previous numeric rows unreachable.
            self._action_ball_diagnostic_pending_row_count += int(
                env_ids.numel()
            )

    @property
    def action_ball_single_stroke_complete(self) -> torch.Tensor:
        """Latched terminal mask for the scoped measured non-looping stroke."""

        complete = getattr(
            self, "_action_ball_single_stroke_complete", None
        )
        if not torch.is_tensor(complete) or complete.shape != (
            self.num_envs,
        ):
            raise RuntimeError(
                "action-ball single-stroke completion latch is unavailable"
            )
        if not (
            self.action_ball_diagnostic_split_ready_teacher
            and self.action_ball_single_stroke_timeout_enabled
        ):
            return torch.zeros_like(complete)
        return complete

    @staticmethod
    def _action_ball_finite_float(
        value, *, name: str, minimum: float | None = None
    ) -> float:
        if (
            isinstance(value, bool)
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a plain finite number")
        result = float(value)
        if minimum is not None and result < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return result

    @staticmethod
    def _action_ball_close_float(
        actual: float, expected: float, *, name: str
    ) -> None:
        if not math.isclose(
            actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"{name} is inconsistent: actual={actual}, expected={expected}"
            )

    def _validate_action_ball_fixed_view_timing_row(
        self, timing_row: object
    ) -> tuple[float, float, float, float, float]:
        """Validate Motion's immutable timing algebra once at fixed-view bind."""

        if (
            not isinstance(timing_row, tuple)
            or len(timing_row) != 15
            or any(
                isinstance(value, bool)
                or type(value) not in (int, float)
                or not math.isfinite(float(value))
                for value in timing_row
            )
        ):
            raise ValueError(
                "fixed-view timing row must contain exactly 15 finite plain numbers"
            )
        (
            time_to_contact,
            reference_t_hit,
            reference_t_cycle,
            reference_speed,
            required_speed,
            teacher_rate_min,
            teacher_rate_max,
            teacher_rate,
            scaled_t_hit,
            scaled_t_cycle,
            pre_swing_wait,
            reaction_margin,
            site_vx,
            site_vy,
            site_vz,
        ) = tuple(float(value) for value in timing_row)
        if (
            time_to_contact <= 0.0
            or reference_t_hit <= 0.0
            or reference_t_cycle <= reference_t_hit
            or reference_speed <= 0.0
            or required_speed <= 0.0
            or teacher_rate_min <= 0.0
            or teacher_rate_max < teacher_rate_min
            or teacher_rate <= 0.0
            or scaled_t_hit <= 0.0
            or scaled_t_cycle <= scaled_t_hit
            or pre_swing_wait < 0.0
            or reaction_margin < 0.0
            or not teacher_rate_min <= 1.0 <= teacher_rate_max
        ):
            raise ValueError("fixed-view timing row has a range/order violation")
        runtime = self._action_ball_runtime_module_bound
        contact_geometry = runtime._contact_geometry
        try:
            canonical_teacher_rate = (
                contact_geometry.canonical_teacher_rate_from_site_speed(
                    required_speed,
                    reference_speed,
                    teacher_rate_min,
                    teacher_rate_max,
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as exc:
            raise ValueError(
                "fixed-view teacher_rate is outside its certified range"
            ) from exc
        self._action_ball_close_float(
            teacher_rate,
            canonical_teacher_rate,
            name="fixed-view canonical teacher_rate",
        )
        self._action_ball_close_float(
            required_speed,
            math.sqrt(site_vx * site_vx + site_vy * site_vy + site_vz * site_vz),
            name="fixed-view required racket-site speed",
        )
        self._action_ball_close_float(
            teacher_rate,
            required_speed / reference_speed,
            name="fixed-view teacher_rate=required/reference",
        )
        self._action_ball_close_float(
            scaled_t_hit,
            reference_t_hit / teacher_rate,
            name="fixed-view scaled_t_hit_s",
        )
        self._action_ball_close_float(
            scaled_t_cycle,
            reference_t_cycle / teacher_rate,
            name="fixed-view scaled_t_cycle_s",
        )
        self._action_ball_close_float(
            pre_swing_wait,
            time_to_contact - scaled_t_hit,
            name="fixed-view pre_swing_wait_s",
        )
        if (
            pre_swing_wait + 1.0e-12 < reaction_margin
            or pre_swing_wait > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "fixed-view pre-swing wait violates reaction/one-second bounds"
            )
        policy_dt = float(self._env.step_dt)
        if (
            pre_swing_wait
            + scaled_t_cycle
            + policy_dt
            > int(self._env.max_episode_length) * policy_dt + 1.0e-12
        ):
            raise ValueError(
                "fixed-view task cycle plus close tick exceeds runtime episode horizon"
            )
        segment_length = int(self._action_ball_segment_lengths[0])
        self._action_ball_close_float(
            reference_t_cycle,
            (segment_length - 1) * policy_dt,
            name="fixed-view reference_t_cycle_s vs admitted motion",
        )
        hit_frame = reference_t_hit / policy_dt
        self._action_ball_close_float(
            hit_frame,
            float(round(hit_frame)),
            name="fixed-view reference_t_hit_s policy-frame alignment",
        )
        if not 0 < round(hit_frame) < segment_length - 1:
            raise ValueError(
                "fixed-view task hit frame is outside the admitted motion interior"
            )
        return (
            time_to_contact,
            teacher_rate,
            scaled_t_hit,
            scaled_t_cycle,
            pre_swing_wait,
        )

    def _validate_action_ball_task_ref_and_receipt_host(
        self,
        task_ref,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        swing_generation: int,
        action_slot: int,
        segment_length: int,
        pending_elapsed_s: float,
    ) -> dict:
        """Validate one task entirely from immutable host identity and timing rows."""

        runtime = self._action_ball_runtime_module_bound
        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        reset_generation = self._action_ball_plain_int(
            reset_generation,
            name="task.reset_generation",
            minimum=1,
        )
        swing_generation = self._action_ball_plain_int(
            swing_generation,
            name="task.swing_generation",
        )
        action_slot = self._action_ball_plain_int(
            action_slot, name="task.action_slot"
        )
        if (
            self._action_ball_action_uids is None
            or action_slot >= len(self._action_ball_action_uids)
        ):
            raise ValueError("task.action_slot is outside Motion's action manifest")
        segment_length = self._action_ball_plain_int(
            segment_length,
            name="task.segment_length",
            minimum=3,
        )
        pending_elapsed = self._action_ball_finite_float(
            pending_elapsed_s,
            name="task.pending_elapsed_s",
            minimum=0.0,
        )
        if type(task_ref) is not runtime.ActionTaskReceiptRef:
            raise ValueError("action-ball task ref has a forged runtime type")
        if type(receipt) is not runtime.ActionBallTaskReceipt:
            raise ValueError("action-ball task receipt has a forged runtime type")
        if self._action_ball_birth_broker.diagnostic_fast_path:
            if (
                task_ref.env_id != receipt.env_id
                or task_ref.reset_generation != receipt.reset_generation
                or task_ref.swing_generation != receipt.swing_generation
                or task_ref.action_uid != receipt.action_uid
                or task_ref.action_slot != receipt.action_slot
                or task_ref.birth_sha256 != receipt.birth_sha256
                or task_ref.sample_sha256 != receipt.sample_sha256
                or task_ref.task_sha256 != receipt.canonical_sha256
            ):
                raise ValueError(
                    "diagnostic action-ball task ref changed receipt identity"
                )
            self._action_ball_sha256(
                task_ref.task_sha256, name="task_ref.task_sha256"
            )
        else:
            canonical_ref = receipt.task_ref()
            if type(canonical_ref) is not runtime.ActionTaskReceiptRef:
                raise ValueError(
                    "action-ball task receipt emitted a forged canonical ref"
                )
            if canonical_ref != task_ref:
                raise ValueError(
                    "action-ball task resolver changed the requested "
                    "immutable ref"
                )
        action_uid = self._action_ball_action_uids[action_slot]
        birth_sha256 = self.action_ball_birth_receipt_sha256(env_id)
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.swing_generation != swing_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
            or receipt.birth_sha256 != birth_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
        ):
            raise ValueError(
                "action-ball task receipt disagrees with Motion birth/action generation"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(
            action_slot
        )
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.profile_sha256 != binding.profile_sha256
            or receipt.arm_catalog_sha256
            != runtime.ARM_CATALOG_SHA256
        ):
            raise ValueError(
                "action-ball task receipt disagrees with broker registry/profile/arm catalog"
            )
        self._action_ball_sha256(
            receipt.sample_sha256, name="task.sample_sha256"
        )
        self._action_ball_sha256(
            receipt.canonical_sha256, name="task.canonical_sha256"
        )

        time_to_contact = self._action_ball_finite_float(
            receipt.time_to_contact_s,
            name="task.time_to_contact_s",
            minimum=0.0,
        )
        reference_t_hit = self._action_ball_finite_float(
            receipt.reference_t_hit_s,
            name="task.reference_t_hit_s",
            minimum=0.0,
        )
        reference_t_cycle = self._action_ball_finite_float(
            receipt.reference_t_cycle_s,
            name="task.reference_t_cycle_s",
            minimum=0.0,
        )
        reference_speed = self._action_ball_finite_float(
            receipt.reference_racket_site_speed_mps,
            name="task.reference_racket_site_speed_mps",
            minimum=0.0,
        )
        required_speed = self._action_ball_finite_float(
            receipt.required_racket_site_speed_mps,
            name="task.required_racket_site_speed_mps",
            minimum=0.0,
        )
        teacher_rate = self._action_ball_finite_float(
            receipt.teacher_rate,
            name="task.teacher_rate",
            minimum=0.0,
        )
        teacher_rate_min = self._action_ball_finite_float(
            receipt.teacher_rate_min,
            name="task.teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = self._action_ball_finite_float(
            receipt.teacher_rate_max,
            name="task.teacher_rate_max",
            minimum=0.0,
        )
        scaled_t_hit = self._action_ball_finite_float(
            receipt.scaled_t_hit_s,
            name="task.scaled_t_hit_s",
            minimum=0.0,
        )
        scaled_t_cycle = self._action_ball_finite_float(
            receipt.scaled_t_cycle_s,
            name="task.scaled_t_cycle_s",
            minimum=0.0,
        )
        pre_swing_wait = self._action_ball_finite_float(
            receipt.pre_swing_wait_s,
            name="task.pre_swing_wait_s",
            minimum=0.0,
        )
        reaction_margin = self._action_ball_finite_float(
            receipt.reaction_margin_s,
            name="task.reaction_margin_s",
            minimum=0.0,
        )
        if (
            time_to_contact <= 0.0
            or reference_t_hit <= 0.0
            or reference_t_cycle <= reference_t_hit
            or reference_speed <= 0.0
            or required_speed <= 0.0
            or teacher_rate <= 0.0
            or teacher_rate_min <= 0.0
            or teacher_rate_max < teacher_rate_min
            or scaled_t_hit <= 0.0
            or scaled_t_cycle <= scaled_t_hit
        ):
            raise ValueError("action-ball task timing has a non-positive/order violation")
        if not teacher_rate_min <= 1.0 <= teacher_rate_max:
            raise ValueError(
                "action-ball certified teacher-rate bounds must contain native rate 1"
            )
        contact_geometry = runtime._contact_geometry
        try:
            contact_geometry.canonical_teacher_rate_from_site_speed(
                teacher_rate,
                1.0,
                teacher_rate_min,
                teacher_rate_max,
            )
            canonical_teacher_rate = (
                contact_geometry.canonical_teacher_rate_from_site_speed(
                required_speed,
                reference_speed,
                teacher_rate_min,
                teacher_rate_max,
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as exc:
            # Keep the consumer on the producer's one SHA-bound float32 seam.
            # This remains fail-closed outside the canonical 5e-7 boundary
            # tolerance and never clips or retimes the task.
            raise ValueError(
                "action-ball teacher_rate is outside its certified range"
            ) from exc
        self._action_ball_close_float(
            teacher_rate,
            canonical_teacher_rate,
            name="task canonical teacher_rate",
        )
        required_vector = self._action_ball_vector(
            receipt.racket_site_velocity_w_mps,
            name="task.racket_site_velocity_w_mps",
            length=3,
        )
        self._action_ball_close_float(
            required_speed,
            math.sqrt(sum(value * value for value in required_vector)),
            name="task required racket-site speed",
        )
        self._action_ball_close_float(
            teacher_rate,
            required_speed / reference_speed,
            name="task teacher_rate=required/reference",
        )
        self._action_ball_close_float(
            scaled_t_hit,
            reference_t_hit / teacher_rate,
            name="task scaled_t_hit_s",
        )
        self._action_ball_close_float(
            scaled_t_cycle,
            reference_t_cycle / teacher_rate,
            name="task scaled_t_cycle_s",
        )
        self._action_ball_close_float(
            pre_swing_wait,
            time_to_contact - scaled_t_hit,
            name="task pre_swing_wait_s",
        )
        if (
            pre_swing_wait + 1.0e-12 < reaction_margin
            or pre_swing_wait > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "action-ball pre-swing wait violates reaction/one-second bounds"
            )
        runtime_episode_length = (
            int(self._env.max_episode_length) * float(self._env.step_dt)
        )
        if (
            pre_swing_wait
            + scaled_t_cycle
            + float(self._env.step_dt)
            > runtime_episode_length + 1.0e-12
        ):
            raise ValueError(
                "action-ball task cycle plus close tick exceeds runtime episode horizon"
            )
        native_cycle = (segment_length - 1) * float(self._env.step_dt)
        self._action_ball_close_float(
            reference_t_cycle,
            native_cycle,
            name="task reference_t_cycle_s vs admitted motion",
        )
        hit_frame = reference_t_hit / float(self._env.step_dt)
        self._action_ball_close_float(
            hit_frame,
            float(round(hit_frame)),
            name="task reference_t_hit_s policy-frame alignment",
        )
        if not 0 < round(hit_frame) < segment_length - 1:
            raise ValueError(
                "action-ball task hit frame is outside the admitted motion interior"
            )
        if pending_elapsed > pre_swing_wait + 1.0e-12:
            raise RuntimeError(
                "action-ball task arrived after its certified ready-wait ended"
            )
        return {
            "time_to_contact_s": time_to_contact,
            "teacher_rate": teacher_rate,
            "scaled_t_hit_s": scaled_t_hit,
            "scaled_t_cycle_s": scaled_t_cycle,
            "pre_swing_wait_s": pre_swing_wait,
            "pending_elapsed_s": pending_elapsed,
        }

    def _validate_action_ball_task_ref_and_receipt(
        self, task_ref, receipt, *, env_id: int
    ) -> dict:
        """Resolve formal/live device identity before the common host validator."""

        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        action_slot = int(self.clip_id[env_id].item())
        return self._validate_action_ball_task_ref_and_receipt_host(
            task_ref,
            receipt,
            env_id=env_id,
            reset_generation=int(
                self._action_ball_reset_generation[env_id].item()
            ),
            swing_generation=int(
                self._action_ball_swing_generation[env_id].item()
            ),
            action_slot=action_slot,
            segment_length=int(
                self.motion.seg_len[action_slot].item()
            ),
            pending_elapsed_s=float(
                self._action_ball_task_pending_elapsed_s[env_id].item()
            ),
        )

    def _validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
        self,
        task_ref,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        swing_generation: int,
        action_slot: int,
        segment_length: int,
        pending_elapsed_s: float,
    ) -> dict:
        """Validate diagnostic runtime identity with lean consumer-owned algebra.

        The diagnostic pool and Racket have already admitted these exact frozen
        runtime objects.  Motion still owns the current-generation identity,
        timing fields it consumes, admitted-motion, episode-horizon, and
        pending-wait checks.  The formal path continues to use the complete
        validator above.
        """

        runtime = self._action_ball_runtime_module_bound
        if (
            self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
        ):
            raise RuntimeError(
                "prevalidated action-ball task validation is diagnostic-only"
            )
        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        reset_generation = self._action_ball_plain_int(
            reset_generation,
            name="task.reset_generation",
            minimum=1,
        )
        swing_generation = self._action_ball_plain_int(
            swing_generation,
            name="task.swing_generation",
        )
        action_slot = self._action_ball_plain_int(
            action_slot, name="task.action_slot"
        )
        if (
            self._action_ball_action_uids is None
            or action_slot >= len(self._action_ball_action_uids)
        ):
            raise ValueError("task.action_slot is outside Motion's action manifest")
        segment_length = self._action_ball_plain_int(
            segment_length,
            name="task.segment_length",
            minimum=3,
        )
        pending_elapsed = self._action_ball_finite_float(
            pending_elapsed_s,
            name="task.pending_elapsed_s",
            minimum=0.0,
        )
        if type(task_ref) is not runtime.ActionTaskReceiptRef:
            raise ValueError("action-ball task ref has a forged runtime type")
        if type(receipt) is not runtime.ActionBallTaskReceipt:
            raise ValueError("action-ball task receipt has a forged runtime type")
        if (
            task_ref.env_id != receipt.env_id
            or task_ref.reset_generation != receipt.reset_generation
            or task_ref.swing_generation != receipt.swing_generation
            or task_ref.action_uid != receipt.action_uid
            or task_ref.action_slot != receipt.action_slot
            or task_ref.birth_sha256 != receipt.birth_sha256
            or task_ref.sample_sha256 != receipt.sample_sha256
            or task_ref.task_sha256 != receipt.canonical_sha256
        ):
            raise ValueError(
                "diagnostic action-ball task ref changed receipt identity"
            )
        self._action_ball_sha256(
            task_ref.task_sha256, name="task_ref.task_sha256"
        )

        action_uid = self._action_ball_action_uids[action_slot]
        birth_sha256 = self.action_ball_birth_receipt_sha256(env_id)
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.swing_generation != swing_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
            or receipt.birth_sha256 != birth_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
        ):
            raise ValueError(
                "action-ball task receipt disagrees with Motion birth/action generation"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(
            action_slot
        )
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.profile_sha256 != binding.profile_sha256
            or receipt.arm_catalog_sha256
            != runtime.ARM_CATALOG_SHA256
        ):
            raise ValueError(
                "action-ball task receipt disagrees with broker registry/profile/arm catalog"
            )
        self._action_ball_sha256(
            receipt.sample_sha256, name="task.sample_sha256"
        )
        self._action_ball_sha256(
            receipt.canonical_sha256, name="task.canonical_sha256"
        )

        time_to_contact = self._action_ball_finite_float(
            receipt.time_to_contact_s,
            name="task.time_to_contact_s",
            minimum=0.0,
        )
        reference_t_hit = self._action_ball_finite_float(
            receipt.reference_t_hit_s,
            name="task.reference_t_hit_s",
            minimum=0.0,
        )
        reference_t_cycle = self._action_ball_finite_float(
            receipt.reference_t_cycle_s,
            name="task.reference_t_cycle_s",
            minimum=0.0,
        )
        reference_speed = self._action_ball_finite_float(
            receipt.reference_racket_site_speed_mps,
            name="task.reference_racket_site_speed_mps",
            minimum=0.0,
        )
        required_speed = self._action_ball_finite_float(
            receipt.required_racket_site_speed_mps,
            name="task.required_racket_site_speed_mps",
            minimum=0.0,
        )
        teacher_rate = self._action_ball_finite_float(
            receipt.teacher_rate,
            name="task.teacher_rate",
            minimum=0.0,
        )
        teacher_rate_min = self._action_ball_finite_float(
            receipt.teacher_rate_min,
            name="task.teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = self._action_ball_finite_float(
            receipt.teacher_rate_max,
            name="task.teacher_rate_max",
            minimum=0.0,
        )
        scaled_t_hit = self._action_ball_finite_float(
            receipt.scaled_t_hit_s,
            name="task.scaled_t_hit_s",
            minimum=0.0,
        )
        scaled_t_cycle = self._action_ball_finite_float(
            receipt.scaled_t_cycle_s,
            name="task.scaled_t_cycle_s",
            minimum=0.0,
        )
        pre_swing_wait = self._action_ball_finite_float(
            receipt.pre_swing_wait_s,
            name="task.pre_swing_wait_s",
            minimum=0.0,
        )
        reaction_margin = self._action_ball_finite_float(
            receipt.reaction_margin_s,
            name="task.reaction_margin_s",
            minimum=0.0,
        )
        if (
            time_to_contact <= 0.0
            or reference_t_hit <= 0.0
            or reference_t_cycle <= reference_t_hit
            or reference_speed <= 0.0
            or required_speed <= 0.0
            or teacher_rate <= 0.0
            or teacher_rate_min <= 0.0
            or teacher_rate_max < teacher_rate_min
            or scaled_t_hit <= 0.0
            or scaled_t_cycle <= scaled_t_hit
        ):
            raise ValueError("action-ball task timing has a non-positive/order violation")
        if not teacher_rate_min <= 1.0 <= teacher_rate_max:
            raise ValueError(
                "action-ball certified teacher-rate bounds must contain native rate 1"
            )
        contact_geometry = runtime._contact_geometry
        try:
            contact_geometry.canonical_teacher_rate_from_site_speed(
                teacher_rate,
                1.0,
                teacher_rate_min,
                teacher_rate_max,
            )
            canonical_teacher_rate = (
                contact_geometry.canonical_teacher_rate_from_site_speed(
                    required_speed,
                    reference_speed,
                    teacher_rate_min,
                    teacher_rate_max,
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as exc:
            raise ValueError(
                "action-ball teacher_rate is outside its certified range"
            ) from exc
        self._action_ball_close_float(
            teacher_rate,
            canonical_teacher_rate,
            name="task canonical teacher_rate",
        )
        required_vector = self._action_ball_vector(
            receipt.racket_site_velocity_w_mps,
            name="task.racket_site_velocity_w_mps",
            length=3,
        )
        self._action_ball_close_float(
            required_speed,
            math.sqrt(sum(value * value for value in required_vector)),
            name="task required racket-site speed",
        )
        # These are the O(1) relations Motion consumes.  Rechecking them is
        # intentionally cheap and prevents a forged frozen receipt from
        # retiming the teacher after pool/Racket admission.  It keeps the
        # vector/rate seams used by the full validator while still avoiding
        # canonical receipt serialization and resolver replay in this path.
        self._action_ball_close_float(
            teacher_rate,
            required_speed / reference_speed,
            name="task teacher_rate=required/reference",
        )
        self._action_ball_close_float(
            scaled_t_hit,
            reference_t_hit / teacher_rate,
            name="task scaled_t_hit_s",
        )
        self._action_ball_close_float(
            scaled_t_cycle,
            reference_t_cycle / teacher_rate,
            name="task scaled_t_cycle_s",
        )
        self._action_ball_close_float(
            pre_swing_wait,
            time_to_contact - scaled_t_hit,
            name="task pre_swing_wait_s",
        )
        if (
            pre_swing_wait + 1.0e-12 < reaction_margin
            or pre_swing_wait > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "action-ball pre-swing wait violates reaction/one-second bounds"
            )
        runtime_episode_length = (
            int(self._env.max_episode_length) * float(self._env.step_dt)
        )
        if (
            pre_swing_wait
            + scaled_t_cycle
            + float(self._env.step_dt)
            > runtime_episode_length + 1.0e-12
        ):
            raise ValueError(
                "action-ball task cycle plus close tick exceeds runtime episode horizon"
            )
        native_cycle = (segment_length - 1) * float(self._env.step_dt)
        self._action_ball_close_float(
            reference_t_cycle,
            native_cycle,
            name="task reference_t_cycle_s vs admitted motion",
        )
        hit_frame = reference_t_hit / float(self._env.step_dt)
        self._action_ball_close_float(
            hit_frame,
            float(round(hit_frame)),
            name="task reference_t_hit_s policy-frame alignment",
        )
        if not 0 < round(hit_frame) < segment_length - 1:
            raise ValueError(
                "action-ball task hit frame is outside the admitted motion interior"
            )
        if pending_elapsed > pre_swing_wait + 1.0e-12:
            raise RuntimeError(
                "action-ball task arrived after its certified ready-wait ended"
            )
        return {
            "time_to_contact_s": time_to_contact,
            "teacher_rate": teacher_rate,
            "scaled_t_hit_s": scaled_t_hit,
            "scaled_t_cycle_s": scaled_t_cycle,
            "pre_swing_wait_s": pre_swing_wait,
            "pending_elapsed_s": pending_elapsed,
        }

    def _resolve_action_ball_task_timing_diagnostic_selected(
        self,
        *,
        host_identity_rows: tuple,
        receipts: tuple,
        task_refs: tuple,
    ) -> None:
        """Install one diagnostic timing batch without per-env device reads.

        Racket calls this only after validating and installing every issued
        receipt.  All identities, Motion-owned timing algebra/runtime
        constraints, buffer shapes, and tensor materialization complete before
        any Motion buffer changes.  Producer-owned vector/contact geometry was
        already validated before the frozen receipt entered this handoff.
        Because the diagnostic pool has already issued, any failure is terminal
        for that run and must never be caught and retried.
        Formal runs retain the opaque ref/resolver path below.
        """

        if (
            self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
        ):
            raise RuntimeError(
                "direct action-ball task timing install is diagnostic-only"
            )
        if (
            type(host_identity_rows) is not tuple
            or type(receipts) is not tuple
            or type(task_refs) is not tuple
            or not host_identity_rows
            or len(receipts) != len(host_identity_rows)
            or len(task_refs) != len(host_identity_rows)
        ):
            raise ValueError(
                "diagnostic action-ball timing requires one non-empty aligned tuple batch"
        )
        if (
            self._action_ball_segment_lengths is None
            or self._action_ball_action_uids is None
            or len(self._action_ball_segment_lengths)
            != len(self._action_ball_action_uids)
        ):
            raise RuntimeError(
                "diagnostic action-ball timing lacks admitted segment lengths"
            )

        env_rows: list[int] = []
        device_rows: list[tuple[float, ...]] = []
        validated_refs: list[object] = []
        seen_envs: set[int] = set()
        step_dt = float(self._env.step_dt)
        for row_index, (identity, receipt, task_ref) in enumerate(
            zip(host_identity_rows, receipts, task_refs)
        ):
            if type(identity) is not tuple or len(identity) != 7:
                raise ValueError(
                    "diagnostic action-ball timing identity row must have seven fields"
                )
            (
                raw_env_id,
                raw_action_slot,
                raw_action_uid,
                raw_reset_generation,
                raw_swing_generation,
                raw_previous_swing_generation,
                active_before_install,
            ) = identity
            env_id = self._action_ball_plain_int(
                raw_env_id,
                name=f"host_identity_rows[{row_index}].env_id",
            )
            if env_id >= self.num_envs or env_id in seen_envs:
                raise ValueError(
                    "diagnostic action-ball timing env ids are out of range or repeated"
                )
            seen_envs.add(env_id)
            action_slot = self._action_ball_plain_int(
                raw_action_slot,
                name=f"host_identity_rows[{row_index}].action_slot",
            )
            if action_slot >= len(self._action_ball_action_uids):
                raise ValueError(
                    "diagnostic action-ball timing action slot is outside the manifest"
                )
            action_uid = self._action_ball_plain_int(
                raw_action_uid,
                name=f"host_identity_rows[{row_index}].action_uid",
                minimum=1,
            )
            reset_generation = self._action_ball_plain_int(
                raw_reset_generation,
                name=f"host_identity_rows[{row_index}].reset_generation",
                minimum=1,
            )
            swing_generation = self._action_ball_plain_int(
                raw_swing_generation,
                name=f"host_identity_rows[{row_index}].swing_generation",
            )
            previous_swing_generation = self._action_ball_plain_int(
                raw_previous_swing_generation,
                name=(
                    f"host_identity_rows[{row_index}]"
                    ".previous_swing_generation"
                ),
                minimum=-1,
            )
            if (
                swing_generation > 0
                and swing_generation != previous_swing_generation + 1
            ):
                raise ValueError(
                    "diagnostic action-ball wrap generation did not advance exactly once"
                )
            # Racket already consumed this flag while closing the prior attempt.
            # A true reset may legitimately replace either an active or an
            # inactive env, so Motion only preserves its exact boolean shape.
            if type(active_before_install) is not bool:
                raise ValueError(
                    "diagnostic action-ball timing active flag must be boolean"
                )
            if action_uid != self._action_ball_action_uids[action_slot]:
                raise ValueError(
                    "diagnostic action-ball timing action UID/slot binding changed"
                )

            pending_elapsed = 0.0 if swing_generation == 0 else step_dt
            timing = (
                self
                ._validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
                    task_ref,
                    receipt,
                    env_id=env_id,
                    reset_generation=reset_generation,
                    swing_generation=swing_generation,
                    action_slot=action_slot,
                    segment_length=self._action_ball_segment_lengths[
                        action_slot
                    ],
                    pending_elapsed_s=pending_elapsed,
                )
            )
            env_rows.append(env_id)
            validated_refs.append(task_ref)
            device_rows.append(
                (
                    env_id,
                    timing["pending_elapsed_s"],
                    timing["time_to_contact_s"],
                    timing["teacher_rate"],
                    timing["scaled_t_hit_s"],
                    timing["scaled_t_cycle_s"],
                    timing["pre_swing_wait_s"],
                )
            )

        timing_buffers = (
            self._action_ball_task_pending_elapsed_s,
            self._action_ball_task_age_s,
            self._action_ball_time_to_contact_s,
            self._action_ball_teacher_rate,
            self._action_ball_scaled_t_hit_s,
            self._action_ball_scaled_t_cycle_s,
            self._action_ball_pre_swing_wait_s,
            self._action_ball_task_timing_active,
        )
        if (
            type(self._action_ball_active_task_refs) is not list
            or len(self._action_ball_active_task_refs) != self.num_envs
            or any(
                buffer is None
                or tuple(buffer.shape) != (self.num_envs,)
                for buffer in timing_buffers
            )
        ):
            raise RuntimeError(
                "diagnostic action-ball timing buffers are not fully bound"
            )
        pending_row_count = (
            self._action_ball_diagnostic_pending_row_count
        )
        if pending_row_count <= 0:
            raise RuntimeError(
                "diagnostic action-ball timing has no pending selected batch"
            )
        if len(env_rows) != pending_row_count:
            raise RuntimeError(
                "diagnostic action-ball timing selected row count does not "
                "match the pending row count"
            )

        # Tensor construction is still staging: no Motion state changes until
        # the complete host batch and its sole H2D payload exist.  Environment
        # ids are exactly representable in the float64 timing dtype and become
        # the indexed-write column on device.
        staged = torch.tensor(
            device_rows,
            dtype=self._action_ball_task_age_s.dtype,
            device=self.device,
        )
        if tuple(staged.shape) != (len(env_rows), 7):
            raise RuntimeError(
                "diagnostic action-ball timing staging shape changed"
            )
        ids = staged[:, 0].to(dtype=torch.long)
        values = staged[:, 1:]
        # The host packet length protects omissions; this same-device guard
        # protects a forged same-length substitution of an already-active row.
        # Diagnostic failures poison the process, so an async device assertion
        # preserves fail-closed semantics without reintroducing a host barrier.
        torch._assert_async(
            torch.all(~self._action_ball_task_timing_active[ids])
        )

        self._action_ball_task_pending_elapsed_s[ids] = values[:, 0]
        self._action_ball_task_age_s[ids] = values[:, 0]
        self._action_ball_time_to_contact_s[ids] = values[:, 1]
        self._action_ball_teacher_rate[ids] = values[:, 2]
        self._action_ball_scaled_t_hit_s[ids] = values[:, 3]
        self._action_ball_scaled_t_cycle_s[ids] = values[:, 4]
        self._action_ball_pre_swing_wait_s[ids] = values[:, 5]
        self._action_ball_task_timing_active[ids] = True
        for env_id, task_ref in zip(env_rows, validated_refs):
            self._action_ball_active_task_refs[env_id] = task_ref
        self._action_ball_diagnostic_pending_row_count -= len(env_rows)
        torch._assert_async(
            torch.all(
                self._action_ball_task_timing_active[
                    self._action_ball_reset_generation > 0
                ]
            )
        )

    # ``install_action_ball_task_timing_diagnostic_many`` was deleted 2026-08-06.
    # It called itself a "compatibility wrapper" and had zero callers -- callers
    # already use ``_resolve_action_ball_task_timing_diagnostic_selected``
    # directly, so the wrapper was compatibility with nothing.

    def install_action_ball_fixed_view_timing_now(
        self,
        *,
        env_ids: torch.Tensor,
        host_identity_rows: tuple,
        fixed_view_identity_sha256: str,
    ) -> None:
        """Activate one shared immutable timing row without task receipts."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="fixed-view timing install"
            )
        if (
            not self.action_ball_fixed_view_enabled
            or self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
            or fixed_view_identity_sha256
            != self._action_ball_fixed_view_identity_sha256
        ):
            raise RuntimeError(
                "fixed-view timing install is outside its immutable N1 diagnostic authority"
            )
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if (
            type(host_identity_rows) is not tuple
            or not host_identity_rows
            or len(host_identity_rows) != len(ids)
            or self._action_ball_diagnostic_pending_row_count != len(ids)
        ):
            raise ValueError(
                "fixed-view timing requires one aligned non-empty pending reset batch"
            )
        seen_envs = set()
        for row_index, row in enumerate(host_identity_rows):
            if type(row) is not tuple or len(row) != 7:
                raise ValueError(
                    "fixed-view timing identity row must have seven fields"
                )
            (
                env_id,
                action_slot,
                action_uid,
                reset_generation,
                swing_generation,
                previous_swing_generation,
                active_before_install,
            ) = row
            if (
                type(env_id) is not int
                or env_id < 0
                or env_id >= self.num_envs
                or env_id in seen_envs
                or type(action_slot) is not int
                or action_slot != 0
                or type(action_uid) is not int
                or action_uid != self._action_ball_action_uids[0]
                or type(reset_generation) is not int
                or reset_generation < 1
                or type(swing_generation) is not int
                or swing_generation < 0
                or type(previous_swing_generation) is not int
                or previous_swing_generation < -1
                or type(active_before_install) is not bool
            ):
                raise ValueError(
                    f"fixed-view timing identity row {row_index} is invalid"
                )
            if swing_generation > 0 and (
                swing_generation != previous_swing_generation + 1
            ):
                raise ValueError(
                    "fixed-view wrap generation did not advance exactly once"
                )
            seen_envs.add(env_id)
        # The caller's device ids are already the reset selection.  Validate
        # their host identity/generation handoff with one compact H2D packet;
        # never copy the ids back to the host (that would add a second reset
        # synchronization after Racket's existing identity packet).
        expected_identity = torch.tensor(
            [
                (row[0], row[3], row[4])
                for row in host_identity_rows
            ],
            dtype=torch.long,
            device=self.device,
        )
        if tuple(expected_identity.shape) != (len(ids), 3):
            raise RuntimeError(
                "fixed-view timing identity staging returned the wrong shape"
            )
        torch._assert_async(torch.all(ids == expected_identity[:, 0]))
        torch._assert_async(
            torch.all(
                self._action_ball_reset_generation[ids]
                == expected_identity[:, 1]
            )
        )
        torch._assert_async(
            torch.all(
                self._action_ball_swing_generation[ids]
                == expected_identity[:, 2]
            )
        )
        torch._assert_async(torch.all(self.clip_id[ids] == 0))
        timing = self._action_ball_fixed_view_timing_row_device.expand(
            len(ids), 5
        )
        if tuple(timing.shape) != (len(ids), 5):
            raise RuntimeError("fixed-view timing broadcast returned the wrong shape")
        swing_generation = self._action_ball_swing_generation[ids]
        pending_elapsed = torch.where(
            swing_generation == 0,
            torch.zeros(
                len(ids),
                dtype=self._action_ball_task_age_s.dtype,
                device=self.device,
            ),
            torch.full(
                (len(ids),),
                float(self._env.step_dt),
                dtype=self._action_ball_task_age_s.dtype,
                device=self.device,
            ),
        )
        torch._assert_async(
            torch.all(~self._action_ball_task_timing_active[ids])
        )
        torch._assert_async(
            torch.all(pending_elapsed <= timing[:, 4] + 1.0e-12)
        )
        self._action_ball_task_pending_elapsed_s[ids] = pending_elapsed
        self._action_ball_task_age_s[ids] = pending_elapsed
        self._action_ball_time_to_contact_s[ids] = timing[:, 0]
        self._action_ball_teacher_rate[ids] = timing[:, 1]
        self._action_ball_scaled_t_hit_s[ids] = timing[:, 2]
        self._action_ball_scaled_t_cycle_s[ids] = timing[:, 3]
        self._action_ball_pre_swing_wait_s[ids] = timing[:, 4]
        self._action_ball_task_timing_active[ids] = True
        self._action_ball_diagnostic_pending_row_count -= len(ids)
        # A strict load deliberately keeps timing inactive until its deferred
        # first true reset.  The first successful fixed-view install consumes
        # that checkpoint-only allowance; subsequent snapshots must again
        # observe a fully installed handoff.
        self._action_ball_expected_shared_racket_state_sha256 = None
        torch._assert_async(
            torch.all(
                self._action_ball_task_timing_active[
                    self._action_ball_reset_generation > 0
                ]
            )
        )

    def _resolve_pending_action_ball_tasks(self) -> None:
        if self._action_ball_task_ref_for_env is None:
            raise RuntimeError("action-ball task ref authority is not bound")
        if (
            self._action_ball_birth_broker is not None
            and self._action_ball_birth_broker.diagnostic_fast_path
            and self._action_ball_diagnostic_pending_row_count == 0
        ):
            return
        pending_ids = torch.where(
            (self._action_ball_reset_generation > 0)
            & (~self._action_ball_task_timing_active)
        )[0]
        if len(pending_ids) == 0:
            if self._action_ball_birth_broker.diagnostic_fast_path:
                self._action_ball_diagnostic_pending_row_count = 0
            return
        for env_id in (
            int(value) for value in pending_ids.detach().cpu().tolist()
        ):
            task_ref = self._action_ball_task_ref_for_env(env_id)
            if task_ref is None:
                raise RuntimeError(
                    "action-ball Racket authority did not publish the current task ref"
                )
            receipt = self._action_ball_task_receipt_resolver(task_ref)
            timing = self._validate_action_ball_task_ref_and_receipt(
                task_ref, receipt, env_id=env_id
            )
            self._action_ball_active_task_refs[env_id] = task_ref
            self._action_ball_task_age_s[env_id] = timing[
                "pending_elapsed_s"
            ]
            self._action_ball_time_to_contact_s[env_id] = timing[
                "time_to_contact_s"
            ]
            self._action_ball_teacher_rate[env_id] = timing[
                "teacher_rate"
            ]
            self._action_ball_scaled_t_hit_s[env_id] = timing[
                "scaled_t_hit_s"
            ]
            self._action_ball_scaled_t_cycle_s[env_id] = timing[
                "scaled_t_cycle_s"
            ]
            self._action_ball_pre_swing_wait_s[env_id] = timing[
                "pre_swing_wait_s"
            ]
            self._action_ball_task_timing_active[env_id] = True
        if self._action_ball_birth_broker.diagnostic_fast_path:
            self._action_ball_diagnostic_pending_row_count = 0

    def resolve_action_ball_task_timing_now(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        diagnostic_host_identity_rows: tuple | None = None,
        diagnostic_receipts: tuple | None = None,
        diagnostic_task_refs: tuple | None = None,
    ) -> None:
        """Resolve newly published Racket receipts before reset observation.

        CommandManager resets Motion before Racket.  Without this handoff,
        formal rows remain locally pending until the following policy step and
        the first actor observation would report a false zero teacher-start
        clock.  Formal calls retain the opaque device-id resolver.  Diagnostic
        calls pass Racket's already-host-visible selected batch and install it
        through one H2D without advancing task age or teacher phase.
        """

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="task timing resolve"
            )

        diagnostic_requested = any(
            value is not None
            for value in (
                diagnostic_host_identity_rows,
                diagnostic_receipts,
                diagnostic_task_refs,
            )
        )
        if diagnostic_requested:
            if env_ids is not None:
                raise ValueError(
                    "diagnostic action-ball timing selection is owned by host identity rows"
                )
            self._resolve_action_ball_task_timing_diagnostic_selected(
                host_identity_rows=diagnostic_host_identity_rows,
                receipts=diagnostic_receipts,
                task_refs=diagnostic_task_refs,
            )
            return
        if env_ids is None:
            raise ValueError(
                "formal action-ball timing resolution requires selected env ids"
            )
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        self._resolve_pending_action_ball_tasks()
        if bool((~self._action_ball_task_timing_active[ids]).any()):
            raise RuntimeError(
                "action-ball task timing was not active before reset "
                "observation"
            )

    @property
    def action_ball_task_timing_active(self) -> torch.Tensor:
        if self._action_ball_task_timing_active is None:
            raise RuntimeError("action-ball task timing is not bound")
        return self._action_ball_continuous_public_tensor(
            self._action_ball_task_timing_active,
            name="task-timing-active",
        )

    @property
    def action_ball_time_to_contact_remaining_s(self) -> torch.Tensor:
        """Signed task deadline; inactive rows use a large fail-closed positive sentinel."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="time-to-contact remaining read"
            )
        if self._action_ball_task_timing_active is None:
            raise RuntimeError("action-ball task timing is not bound")
        remaining = (
            self._action_ball_time_to_contact_s
            - self._action_ball_task_age_s
        )
        return torch.where(
            self._action_ball_task_timing_active,
            remaining,
            torch.full_like(remaining, 1.0e6),
        )

    @property
    def action_ball_pre_swing_wait_remaining_s(self) -> torch.Tensor:
        """Time until this row's teacher leaves its frozen ready frame.

        This is the exact live phase-governor clock, not a value reconstructed
        by the actor from time-to-contact, action identity and requested site
        speed.  Inactive rows expose zero; a valid ActionBall rollout resolves
        every row before the policy observation is consumed.
        """

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="pre-swing wait remaining read"
            )
        if (
            self._action_ball_task_timing_active is None
            or self._action_ball_pre_swing_wait_s is None
            or self._action_ball_task_age_s is None
        ):
            raise RuntimeError("action-ball task timing is not bound")
        remaining = torch.clamp(
            self._action_ball_pre_swing_wait_s
            - self._action_ball_task_age_s,
            min=0.0,
        )
        return torch.where(
            self._action_ball_task_timing_active,
            remaining,
            torch.zeros_like(remaining),
        )

    def _advance_action_ball_task_timing(
        self,
        advance_mask: torch.Tensor | None = None,
        *,
        resolve_pending: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance receipt time analytically; return held and cycle-due-before masks."""

        # A task resolved during this command-manager compute has not driven a
        # physics tick yet.  Only rows that were already active on entry may
        # consume this update's elapsed policy interval.  WRAP carries one dt
        # in ``pending_elapsed_s`` because its replacement task is installed
        # later in the previous compute and does drive the intervening tick.
        if advance_mask is not None and (
            not torch.is_tensor(advance_mask)
            or advance_mask.dtype != torch.bool
            or tuple(advance_mask.shape) != (self.num_envs,)
            or advance_mask.device != torch.device(self.device)
        ):
            raise ValueError(
                "action-ball task timing advance_mask must be one bool tensor "
                "on Motion's device"
            )
        if type(resolve_pending) is not bool:
            raise ValueError("resolve_pending must be an exact boolean")
        if not resolve_pending and advance_mask is None:
            raise ValueError(
                "skipping pending resolution requires an explicit advance_mask"
            )
        active_before_resolve = self._action_ball_task_timing_active.clone()
        if advance_mask is not None:
            active_before_resolve &= advance_mask
        if resolve_pending:
            self._resolve_pending_action_ball_tasks()
        active_all = self._action_ball_task_timing_active
        writable_rows = torch.ones_like(active_all)
        if resolve_pending:
            if self._action_ball_birth_broker.diagnostic_fast_path:
                if self._action_ball_diagnostic_pending_row_count != 0:
                    raise RuntimeError(
                        "diagnostic action-ball task timing remained unresolved"
                    )
            elif bool(
                (
                    (self._action_ball_reset_generation > 0)
                    & (
                        torch.ones_like(active_all)
                        if advance_mask is None
                        else advance_mask
                    )
                    & ~active_all
                )
                .any()
            ):
                raise RuntimeError(
                    "action-ball task timing remained unresolved"
                )
        else:
            # The continuous transaction releases a row only after the other
            # command owner has atomically installed timing and acknowledged
            # the same reveal.  Re-running the legacy formal pending resolver
            # here would perform a device-wide ``where`` plus D2H scan on every
            # policy tick.  Latch the exact task-timing cause into the
            # existing packed Epoch drain and suppress the bad row locally.
            writable_rows = (
                self._latch_action_ball_full_mdp_motion_epoch_row_fault(
                    (advance_mask & ~active_all).contiguous(),
                    reason_bit=(
                        _ACTION_EPOCH_ROW_FAULT_MOTION_TASK_TIMING_CONTRACT
                    ),
                )
            )
            advance_mask = advance_mask & writable_rows
            active_before_resolve &= writable_rows
        active = (
            active_all
            if advance_mask is None
            else (active_all & advance_mask)
        )
        cycle_total = (
            self._action_ball_pre_swing_wait_s
            + self._action_ball_scaled_t_cycle_s
        )
        cycle_due_before = active & (
            self._action_ball_task_age_s + 1.0e-12 >= cycle_total
        )
        advancing = active_before_resolve & ~cycle_due_before
        self._action_ball_task_age_s[advancing] += float(
            self._env.step_dt
        )
        active_motion_s = torch.clamp(
            self._action_ball_task_age_s
            - self._action_ball_pre_swing_wait_s,
            min=0.0,
        )
        active_motion_s = torch.minimum(
            active_motion_s, self._action_ball_scaled_t_cycle_s
        )
        clip_starts = self.motion.seg_start[self.clip_id].to(
            dtype=torch.float64
        )
        phase_frames = (
            active_motion_s
            * self._action_ball_teacher_rate
            / float(self._env.step_dt)
        )
        final_frames = (
            self.motion.seg_len[self.clip_id] - 1
        ).to(dtype=torch.float64)
        phase_frames = torch.minimum(phase_frames, final_frames)
        next_time_steps_f = (clip_starts + phase_frames).to(
            self.time_steps_f.dtype
        )
        self.time_steps_f.copy_(
            torch.where(
                writable_rows,
                next_time_steps_f,
                self.time_steps_f,
            )
        )
        rounded = self.time_steps_f.round().long()
        final_steps = (
            self.motion.seg_start[self.clip_id]
            + self.motion.seg_len[self.clip_id]
            - 1
        )
        self.time_steps.copy_(
            torch.where(
                writable_rows,
                torch.minimum(rounded, final_steps),
                self.time_steps,
            )
        )
        self.speed_scale.copy_(
            torch.where(
                writable_rows,
                torch.where(
                    active,
                    self._action_ball_teacher_rate.to(self.speed_scale.dtype),
                    torch.zeros_like(self.speed_scale),
                ),
                self.speed_scale,
            )
        )
        held = active & (active_motion_s <= 1.0e-12)
        self.hold_counter.copy_(
            torch.where(
                writable_rows,
                held.to(self.hold_counter.dtype),
                self.hold_counter,
            )
        )
        prior_in_hold = self.metrics.get("in_hold")
        if (
            torch.is_tensor(prior_in_hold)
            and tuple(prior_in_hold.shape) == (self.num_envs,)
            and prior_in_hold.device == torch.device(self.device)
        ):
            self.metrics["in_hold"] = torch.where(
                writable_rows,
                held.to(prior_in_hold.dtype),
                prior_in_hold,
            )
        else:
            self.metrics["in_hold"] = held.float()
        self.metrics["playback_speed"] = self.speed_scale.clone()
        self.metrics["action_ball_task_age_s"] = (
            self._action_ball_task_age_s.to(self.speed_scale.dtype)
        )
        self.metrics["action_ball_time_to_contact_s"] = (
            self.action_ball_time_to_contact_remaining_s.to(
                self.speed_scale.dtype
            )
        )
        self.metrics["action_ball_teacher_rate"] = (
            self._action_ball_teacher_rate.to(self.speed_scale.dtype)
        )
        self.metrics["action_ball_pre_swing_wait_remaining_s"] = (
            self.action_ball_pre_swing_wait_remaining_s.to(
                self.speed_scale.dtype
            )
        )
        return held, cycle_due_before

    def _write_canonical_ready_state(
        self,
        env_ids: torch.Tensor,
        *,
        action_ball_base_spawn_w_m: torch.Tensor | None = None,
        action_ball_base_quat_wxyz: torch.Tensor | None = None,
    ) -> dict | None:
        """Write one clip-owned ready transaction: root + 31 joints, all velocities zero.

        In standard action-ball mode the provider-issued birth owns the
        environment-local XYZ and supplies a yaw-only B_yaw frame for
        validation.  The physical root quaternion and joint pose remain the
        selected opaque-admitted clip's literal ready state.  The separately
        branded measured-N1 diagnostic instead consumes receipt XY plus the
        dynamic-ready binding's physical Z/quaternion/joints.  A horizontal
        solver frame must never erase physical roll/pitch in either path.
        """

        ready_steps = self._canonical_ready_steps(env_ids)
        action_slots = self.clip_id[env_ids]
        dynamic_ready_enabled = (
            getattr(
                self,
                "_action_ball_dynamic_ready_binding_sha256",
                None,
            )
            is not None
        )
        split_ready_teacher = bool(
            getattr(
                self,
                "action_ball_diagnostic_split_ready_teacher",
                False,
            )
        )
        if split_ready_teacher:
            if not dynamic_ready_enabled:
                raise RuntimeError(
                    "split-ready true reset requires its validated dynamic-ready binding"
                )
            root_pos = (
                self._action_ball_dynamic_ready_physical_root_pos_w_m[
                    action_slots
                ]
                + self._env.scene.env_origins[env_ids]
            )
            root_quat = (
                self._action_ball_dynamic_ready_physical_root_quat_wxyz[
                    action_slots
                ]
            )
            joint_pos = (
                self._action_ball_dynamic_ready_physical_joint_pos_rad[
                    action_slots
                ]
            )
            joint_vel = (
                self._action_ball_dynamic_ready_physical_joint_vel_radps[
                    action_slots
                ]
            )
        else:
            root_pos = (
                self.motion.body_pos_w[ready_steps, 0]
                + self._env.scene.env_origins[env_ids]
            )
            root_quat = self.motion.body_quat_w[ready_steps, 0]
            joint_pos = self.motion.joint_pos[ready_steps]
            joint_vel = torch.zeros_like(joint_pos)
        action_ball_write = action_ball_base_spawn_w_m is not None
        if action_ball_write != (action_ball_base_quat_wxyz is not None):
            raise ValueError(
                "action-ball root spawn and quaternion must be supplied together"
            )
        if action_ball_write:
            spawn = action_ball_base_spawn_w_m
            frame_quat = action_ball_base_quat_wxyz
            if (
                not torch.is_tensor(spawn)
                or tuple(spawn.shape) != (len(env_ids), 3)
                or not bool(torch.isfinite(spawn).all())
                or not torch.is_tensor(frame_quat)
                or tuple(frame_quat.shape) != (len(env_ids), 4)
                or not bool(torch.isfinite(frame_quat).all())
            ):
                raise ValueError(
                    "action-ball root must be finite [N,3] spawn + [N,4] yaw-frame tensors"
                )
            spawn = spawn.to(dtype=root_pos.dtype, device=root_pos.device)
            # The yaw-frame tensor was already checked against the selected
            # clip by _validate_action_ball_birth_receipt.  Convert here only
            # to fail on an incompatible device/dtype before mutating PhysX;
            # it is deliberately not the physical root quaternion.
            frame_quat = frame_quat.to(
                dtype=root_quat.dtype, device=root_quat.device
            )
            if split_ready_teacher:
                # The task receipt owns the environment-local XY/yaw frame, while
                # the independently validated physical-ready state owns physical
                # root Z/tilt/joints.  Recenter XY without replacing the hold-pass
                # root height with the lower measured-teacher frame-0 height.
                root_pos = root_pos.clone()
                root_pos[:, :2] = (
                    self._env.scene.env_origins[env_ids, :2].to(root_pos.dtype)
                    + spawn[:, :2]
                )
            else:
                # ``base_spawn_w_m`` is an environment-local world-frame position, not an offset
                # from the historical clip root.  The standard canonical path keeps the receipt
                # as the sole physical translation truth.
                root_pos = (
                    self._env.scene.env_origins[env_ids].to(root_pos.dtype)
                    + spawn
                )
        # --- 起点扰动斜坡:收据之外、显式声明的物理出生偏移 -------------------
        #
        # 收据仍然是"这一拍的任务"的唯一真相:base_goal / 接触点 / B_yaw 目标框
        # 一个字节都不动。这里改的只是**机器人物理出生在哪、朝哪**。于是目标不动、
        # 起点动 —— 这正是"要走过去才够得着"的步法训练,而不是偷偷换了一道题。
        #
        # 只动 XY 与 yaw:
        #   * 出生高度 Z 归 canonical-ready 的 root-Z 合同拥有,不许碰;
        #   * roll/pitch 是物理 ready 姿态的一部分(fivebind ~-11.2 deg、
        #     ChingMu73 ~+8..12 deg 实测),不是"站位",碰了就等于换了个 ready。
        # 偏移量按 ramp 进度插值,progress=0 时逐字节等于过去的收据出生。
        ramp_progress = self.start_pose_ramp_progress()
        if bool(getattr(self, "_start_pose_ramp_enabled", False)):
            if not action_ball_write:
                raise RuntimeError(
                    "start_pose_ramp may only perturb an action-ball "
                    "true-reset transaction"
                )
            ramp_ranges = self._effective_reset_range_list(
                "pose_range", ramp_progress
            )
            axis_index = {
                axis: index
                for index, axis in enumerate(
                    ("x", "y", "z", "roll", "pitch", "yaw")
                )
            }
            for axis in ("z", "roll", "pitch"):
                lo, hi = ramp_ranges[axis_index[axis]]
                if lo != 0.0 or hi != 0.0:
                    raise RuntimeError(
                        "start_pose_ramp may not perturb the canonical-ready "
                        f"{axis} axis; the ready root height and physical "
                        "roll/pitch belong to the ready contract"
                    )
            velocity_ranges = self._effective_reset_range_list(
                "velocity_range", ramp_progress
            )
            if any(pair != (0.0, 0.0) for pair in velocity_ranges):
                raise RuntimeError(
                    "start_pose_ramp may not give a canonical-ready birth a "
                    "non-zero root velocity; the split-ready physical reset is "
                    "a stationary safe-ready state"
                )
            lo_x, hi_x = ramp_ranges[axis_index["x"]]
            lo_y, hi_y = ramp_ranges[axis_index["y"]]
            lo_yaw, hi_yaw = ramp_ranges[axis_index["yaw"]]
            bounds = torch.tensor(
                [[lo_x, hi_x], [lo_y, hi_y], [lo_yaw, hi_yaw]],
                dtype=root_pos.dtype,
                device=root_pos.device,
            )
            samples = sample_uniform(
                bounds[:, 0],
                bounds[:, 1],
                (len(env_ids), 3),
                device=root_pos.device,
            ).to(dtype=root_pos.dtype)
            root_pos = root_pos.clone()
            root_pos[:, 0] += samples[:, 0]
            root_pos[:, 1] += samples[:, 1]
            zero = torch.zeros_like(samples[:, 2])
            yaw_delta = quat_from_euler_xyz(zero, zero, samples[:, 2]).to(
                dtype=root_quat.dtype
            )
            root_quat = quat_mul(yaw_delta, root_quat)
            if "start_pose_ramp_dx_m" in self.metrics:
                self.metrics["start_pose_ramp_progress"][env_ids] = float(
                    ramp_progress
                )
                self.metrics["start_pose_ramp_dx_m"][env_ids] = samples[
                    :, 0
                ].to(self.metrics["start_pose_ramp_dx_m"].dtype)
                self.metrics["start_pose_ramp_dy_m"][env_ids] = samples[
                    :, 1
                ].to(self.metrics["start_pose_ramp_dy_m"].dtype)
                self.metrics["start_pose_ramp_dyaw_rad"][env_ids] = samples[
                    :, 2
                ].to(self.metrics["start_pose_ramp_dyaw_rad"].dtype)
        root_velocity = torch.zeros(
            len(env_ids), 6, dtype=root_pos.dtype, device=root_pos.device
        )
        root_state = torch.cat((root_pos, root_quat, root_velocity), dim=-1)
        diagnostic_fast_path = (
            action_ball_write
            and bool(
                getattr(
                    getattr(self, "_action_ball_birth_broker", None),
                    "diagnostic_fast_path",
                    False,
                )
            )
        )
        rollback_state = None
        if action_ball_write and not diagnostic_fast_path:
            # Isaac exposes separate setters.  Snapshot only for rollback; these live tensors are
            # never used to derive a birth.  All new payloads above came from admitted clip bytes
            # and the provider-issued receipt before the first simulator mutation.
            rollback_state = {
                "root_state": self.robot.data.root_state_w[env_ids].clone(),
                "joint_pos": self.robot.data.joint_pos[env_ids].clone(),
                "joint_vel": self.robot.data.joint_vel[env_ids].clone(),
            }
        if dynamic_ready_enabled and not action_ball_write:
            raise RuntimeError(
                "action_ball_dynamic_ready may install only inside an "
                "action-ball true-reset transaction"
            )
        dynamic_ready_action_term = (
            self._bind_action_ball_dynamic_ready_action_term()
            if dynamic_ready_enabled
            else None
        )
        try:
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids
            )
            if dynamic_ready_action_term is not None:
                if diagnostic_fast_path:
                    action_rollback_state = (
                        dynamic_ready_action_term
                        .install_action_ball_dynamic_ready_state(
                            env_ids,
                            self._action_ball_dynamic_ready_normalized_actor_action[
                                action_slots
                            ],
                            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad[
                                action_slots
                            ],
                            capture_rollback=False,
                        )
                    )
                    if action_rollback_state is not None:
                        raise RuntimeError(
                            "diagnostic dynamic-ready install unexpectedly "
                            "returned rollback state"
                        )
                else:
                    action_rollback_state = (
                        dynamic_ready_action_term
                        .install_action_ball_dynamic_ready_state(
                            env_ids,
                            self._action_ball_dynamic_ready_normalized_actor_action[
                                action_slots
                            ],
                            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad[
                                action_slots
                            ],
                        )
                    )
                    rollback_state["action_state"] = action_rollback_state
        except Exception as exc:
            if rollback_state is not None:
                try:
                    self._restore_action_ball_sim_state(
                        env_ids, rollback_state
                    )
                except Exception as rollback_error:
                    raise RuntimeError(
                        "action-ball root write failed and simulator rollback failed"
                    ) from rollback_error
            raise
        return rollback_state

    def action_ball_full_mdp_physical_reset_state(
        self, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project the one validated dynamic-ready plant birth for reset.

        The fresh FullMDP reset event is the sole simulator writer.  Motion
        owns the selected action and the dynamic-ready binding, so it exposes
        only the already-validated physical root/joint state here.  Teacher
        frame 0, policy q_des and task success are deliberately absent.
        """

        if (
            type(env_ids) is not torch.Tensor
            or env_ids.ndim != 1
            or env_ids.dtype != torch.int64
            or env_ids.device != torch.device(self.device)
        ):
            raise ValueError(
                "FullMDP physical reset requires selected int64 env_ids"
            )
        if (
            not bool(
                getattr(
                    self,
                    "action_ball_diagnostic_split_ready_teacher",
                    False,
                )
            )
            or getattr(
                self,
                "_action_ball_dynamic_ready_binding_sha256",
                None,
            )
            is None
        ):
            raise RuntimeError(
                "FullMDP physical reset requires the split dynamic-ready binding"
            )
        action_slots = self.clip_id[env_ids]
        root_pos = (
            self._action_ball_dynamic_ready_physical_root_pos_w_m[
                action_slots
            ]
            + self._env.scene.env_origins[env_ids]
        )
        root_quat = (
            self._action_ball_dynamic_ready_physical_root_quat_wxyz[
                action_slots
            ]
        )
        joint_pos = self._action_ball_dynamic_ready_physical_joint_pos_rad[
            action_slots
        ]
        joint_vel = self._action_ball_dynamic_ready_physical_joint_vel_radps[
            action_slots
        ]
        root_velocity = torch.zeros(
            len(env_ids),
            6,
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        result = {
            "root_state": torch.cat(
                (root_pos, root_quat, root_velocity), dim=-1
            ),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        if (
            tuple(result["root_state"].shape) != (len(env_ids), 13)
            or tuple(result["joint_pos"].shape)
            != (len(env_ids), _A3_CANONICAL_READY_JOINT_COUNT)
            or tuple(result["joint_vel"].shape)
            != (len(env_ids), _A3_CANONICAL_READY_JOINT_COUNT)
            or any(
                not bool(torch.isfinite(value).all())
                for value in result.values()
            )
        ):
            raise RuntimeError(
                "FullMDP dynamic-ready physical reset state differs"
            )
        return result

    def _restore_action_ball_sim_state(
        self, env_ids: torch.Tensor, rollback_state: dict
    ) -> None:
        legacy_keys = {"root_state", "joint_pos", "joint_vel"}
        if (
            type(rollback_state) is not dict
            or set(rollback_state) not in (
                legacy_keys,
                legacy_keys | {"action_state"},
            )
        ):
            raise RuntimeError("action-ball simulator rollback state is malformed")
        self.robot.write_root_state_to_sim(
            rollback_state["root_state"], env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            rollback_state["joint_pos"],
            rollback_state["joint_vel"],
            env_ids=env_ids,
        )
        if "action_state" in rollback_state:
            action_term = getattr(
                self, "_action_ball_dynamic_ready_action_term", None
            )
            if action_term is None:
                raise RuntimeError(
                    "action-ball rollback contains action state without its "
                    "dynamic-ready action term"
                )
            action_term.restore_action_ball_dynamic_ready_state(
                env_ids, rollback_state["action_state"]
            )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def clip_family_is_forehand(self) -> torch.Tensor:
        """[num_segments] bool 表:True = 该 clip 属正手家族(False = 反手)。

        人话:配了 clip_family_per_clip 用配置表(__init__ 已整表校验);没配按老规矩懒推导——
        单 clip 当正手、恰好 2 clip = (正手, 反手),和写死的 ``clips == 0`` 判断逐字节同值;
        推不出(≥3 clip 没配表)当场报错,绝不猜(见 resolve_clip_family_is_forehand)。
        """
        if self._clip_family_is_forehand_t is None:
            self._clip_family_is_forehand_t = torch.tensor(
                resolve_clip_family_is_forehand(
                    getattr(self.cfg, "clip_family_per_clip", None),
                    int(self.motion.num_segments),
                ),
                dtype=torch.bool,
                device=self.device,
            )
        return self._clip_family_is_forehand_t

    @property
    def in_hold(self) -> torch.Tensor:
        """Bool mask for the *current control step's* pre-swing hold.

        ``_update_command`` snapshots ``held`` and then decrements ``hold_counter``.  Looking only
        at the post-decrement counter made the final frozen-reference step appear unheld to
        rewards/terminations (an off-by-one reference death at release).  The metric stores that
        snapshot; OR it with the counter so the contract is also correct immediately after a
        reset/wrap resample, before the next update.
        """
        counter_hold = self.hold_counter > 0
        metric_hold = self.metrics.get("in_hold")
        return counter_hold if metric_hold is None else (counter_hold | metric_hold.bool())

    @property
    def imitation_eligible(self) -> torch.Tensor:
        """Rows allowed to receive body-imitation income this control step.

        Ordinary hold/recovery rows stay excluded.  The measured split-ready
        diagnostic is different: hidden RESET_WAIT supervises the frozen
        physical safe-ready reference, then atomic reveal exposes measured
        frame 0.  Body pose and stationary velocity imitation therefore stay
        active while ``in_hold`` protects reset-relative termination semantics.
        """

        if self.action_ball_diagnostic_split_ready_teacher:
            return torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        return ~self.in_hold

    @property
    def teacher_start_wait_remaining_s(self) -> torch.Tensor:
        """Seconds until legacy/Stage-1 teacher playback next advances.

        This is deliberately owned by ``MotionCommand`` rather than by ActionBall's task receipt.
        On the normal command-manager path ``_update_command`` has already consumed the current
        control step and decremented ``hold_counter`` before the actor observes the next state.
        Therefore the post-decrement counter is the exact number of *future* frozen-reference
        steps.  In particular, the final frozen step leaves a zero counter and correctly reports
        zero to the next action; OR-ing with :attr:`in_hold` here would introduce one extra step.

        Immediately after reset (before the first update), the undecremented counter still equals
        the number of future frozen steps, so the same expression also covers construction/reset.
        """

        if self.hold_counter.shape != (self.num_envs,):
            raise RuntimeError(
                "MotionCommand hold_counter must have one scalar per environment"
            )
        if self.hold_counter.dtype != torch.long:
            raise RuntimeError("MotionCommand hold_counter must use torch.long steps")
        policy_dt_s = float(self._env.step_dt)
        if not math.isfinite(policy_dt_s) or policy_dt_s <= 0.0:
            raise RuntimeError("MotionCommand policy step_dt must be finite and positive")
        remaining = self.hold_counter.clamp_min(0).to(
            dtype=self.time_steps_f.dtype
        ) * policy_dt_s
        torch._assert_async(torch.isfinite(remaining).all())
        return remaining

    @property
    def event_timing_enabled(self) -> bool:
        return self._event_scheduler is not None

    @property
    def event_just_installed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.event_just_installed

    @property
    def event_installed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.row_installed

    @property
    def event_exact_strike_allowed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.exact_strike_allowed

    @property
    def event_deadline_ticks_remaining(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.deadline_ticks_remaining

    @property
    def event_current_clip_id(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.current_clip_id

    @property
    def event_current_bank_row(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.current_bank_row

    @property
    def event_schedule(self):
        return self._event_schedule

    def event_timing_hard_contract(self) -> dict:
        """Stable timing facts embedded in every checkpoint contract."""

        if self._event_schedule is None:
            return {"mode": EVENT_TIMING_MODE_DISABLED}
        return {
            "mode": EVENT_TIMING_MODE_POST_STRIKE_T1,
            "schedule": self._event_schedule.hard_contract(),
            "sequence_assignment": "env_id_mod_sequence_count_v1",
            "repeat_within_episode": False,
            "clock_origin": "accepted_exact_strike_opportunity",
            "install_trigger": "immutable_post_strike_reveal_tick",
            "deadline_origin": "previous_scheduled_deadline_after_first_origin",
            "deadline_shift_allowed": False,
            "miss_consumes_opportunity": True,
            "carry_state": True,
            "reset_robot_or_last_action_on_install": False,
            "reset_history_or_noise_on_install": False,
            "event_playback": "native_clip_start_plus_exact_hold_no_retime",
        }

    def bind_event_native_strike_ticks(
        self, native_strike_ticks_by_clip: Sequence[int] | torch.Tensor
    ) -> None:
        """Bind RacketTargetCommand's audited per-clip strike frames exactly once."""

        if self._event_scheduler is None:
            return
        raw = torch.as_tensor(native_strike_ticks_by_clip, device=self.device)
        if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
            raise ValueError("event native strike ticks must use an integer dtype")
        values = raw.to(dtype=torch.long).reshape(-1)
        if len(values) != int(self.motion.num_segments) or torch.any(values <= 0):
            raise ValueError(
                "event native strike timing must contain one positive offset per motion clip"
            )
        if self._event_native_strike_ticks is not None:
            if not torch.equal(self._event_native_strike_ticks, values):
                raise RuntimeError("event native strike timing was rebound with different values")
            return
        self._event_native_strike_ticks = values.clone()

    def record_event_exact_strike(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return self._event_scheduler.record_exact_strike(env_ids)

    def finalize_event_deadlines(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return self._event_scheduler.finalize_deadlines()

    def planner_revision_hard_contract(self) -> dict | None:
        """Return the complete runtime contract, or ``None`` for the byte-identical OFF path."""

        if not self.planner_revision_enabled:
            return None
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision enabled without a validated profile")
        initial_tts = tuple(
            float(value) for value in self.cfg.planner_revision_initial_tts_range_s
        )
        return {
            "enabled": True,
            "revision_schema_version": PLANNER_TASK_REVISION_SCHEMA_VERSION,
            "governor": profile.hard_contract(),
            "initial_tts_range_s": list(initial_tts),
        }

    def planner_revision_training_hard_contract(self) -> dict | None:
        """Return canonical training-only planner facts from the validated runtime objects.

        Hydra may retain mapping-shaped values as ``DictConfig`` instances.  The generic legacy
        hard-contract converter intentionally preserves its historical behavior, so feeding the
        raw config through it would serialize a mapping as a list of keys.  The parsed
        ``InitialTtsMixture`` is already the runtime authority; publishing its canonical document
        here keeps the producer and validator on one representation without changing any legacy
        OFF-path contract bytes.
        """

        if not self.planner_revision_enabled:
            return None
        mixture = self._planner_initial_tts_mixture
        if mixture is None:
            raise RuntimeError(
                "planner revision enabled without a validated initial-TTS mixture"
            )
        return {"initial_tts_mixture": mixture.document()}

    def begin_planner_task(
        self,
        env_ids: torch.Tensor,
        *,
        control_epoch: torch.Tensor,
        task_id: torch.Tensor,
        strike_step: torch.Tensor,
        initial_tts: torch.Tensor,
        target_position: torch.Tensor,
        target_velocity: torch.Tensor,
        target_normal: torch.Tensor,
    ) -> None:
        """Install one new immutable physical-ball identity for each selected environment."""

        if not self.planner_revision_enabled:
            raise RuntimeError("begin_planner_task called while planner revisions are disabled")
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if len(ids) == 0:
            return
        epoch = control_epoch.to(device=self.device, dtype=torch.long).reshape(-1)
        tasks = task_id.to(device=self.device, dtype=torch.long).reshape(-1)
        strike = strike_step.to(device=self.device, dtype=torch.float32).reshape(-1)
        raw_tts = initial_tts.to(device=self.device, dtype=torch.float32).reshape(-1)
        tts = self._planner_canonicalize_tts(raw_tts, profile)
        pos = target_position.to(device=self.device, dtype=torch.float32)
        vel = target_velocity.to(device=self.device, dtype=torch.float32)
        normal = target_normal.to(device=self.device, dtype=torch.float32)
        start = self.time_steps_f[ids]
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        minimum_tts = self._planner_minimum_finish_time(
            torch.ones_like(tts),
            torch.zeros_like(tts),
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        valid = (
            (epoch > 0)
            & (tasks > 0)
            & torch.isfinite(strike)
            & (strike > start)
            & torch.isfinite(raw_tts)
            & (raw_tts + profile.early_deadline_tolerance_s >= profile.min_tts_s)
            & (raw_tts - profile.early_deadline_tolerance_s <= profile.max_tts_s)
            & (tts + profile.early_deadline_tolerance_s >= minimum_tts)
            & torch.isfinite(pos).all(dim=-1)
            & torch.isfinite(vel).all(dim=-1)
            & torch.isfinite(normal).all(dim=-1)
            & ((normal_norm - 1.0).abs() <= profile.normal_unit_tolerance)
        )
        if not bool(valid.all()):
            raise ValueError("begin_planner_task received an invalid or partial atomic task tuple")
        self._planner_active[ids] = True
        self._planner_control_epoch[ids] = epoch
        self._planner_task_id[ids] = tasks
        self._planner_task_revision[ids] = 1
        self._planner_start_step[ids] = start
        self._planner_strike_step[ids] = strike
        self._planner_phase_rate[ids] = 0.0
        self._planner_slow_only_next[ids] = False
        self._planner_desired_tts[ids] = tts
        self._planner_begin_tts[ids] = tts
        self._planner_truth_tts[ids] = tts
        self._planner_truth_tts_signed[ids] = tts
        self._planner_begin_target_pos[ids] = pos
        self._planner_begin_target_vel[ids] = vel
        self._planner_begin_target_normal[ids] = normal
        self.speed_scale[ids] = 0.0

    @staticmethod
    def _planner_canonicalize_tts(
        tts: torch.Tensor, profile: PhaseGovernorProfile
    ) -> torch.Tensor:
        """Snap only the profile-bound float32 edge bands to their canonical values."""

        tolerance = profile.early_deadline_tolerance_s
        minimum = torch.full_like(tts, profile.min_tts_s)
        maximum = torch.full_like(tts, profile.max_tts_s)
        snapped = torch.where((tts - minimum).abs() <= tolerance, minimum, tts)
        return torch.where((snapped - maximum).abs() <= tolerance, maximum, snapped)

    @staticmethod
    def _planner_minimum_finish_time(
        distance: torch.Tensor,
        initial_rate: torch.Tensor,
        maximum_rate: float,
        maximum_acceleration: float,
    ) -> torch.Tensor:
        """Vector form of planner_revision._minimum_finish_time."""

        rate = initial_rate.clamp(min=0.0, max=maximum_rate)
        accelerate_time = (maximum_rate - rate).clamp(min=0.0) / maximum_acceleration
        accelerate_distance = (
            rate * accelerate_time + 0.5 * maximum_acceleration * accelerate_time.square()
        )
        triangular = (-rate + torch.sqrt(
            (rate.square() + 2.0 * maximum_acceleration * distance.clamp(min=0.0))
        )) / maximum_acceleration
        trapezoidal = accelerate_time + (
            distance - accelerate_distance
        ).clamp(min=0.0) / maximum_rate
        return torch.where(distance <= accelerate_distance, triangular, trapezoidal)

    def submit_planner_revision(
        self,
        env_ids: torch.Tensor,
        *,
        control_epoch: torch.Tensor,
        task_id: torch.Tensor,
        task_revision: torch.Tensor,
        desired_tts: torch.Tensor,
        target_position: torch.Tensor,
        target_velocity: torch.Tensor,
        target_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Atomically accept/reject same-task revisions; rejected rows preserve the old ledger."""

        if not self.planner_revision_enabled:
            raise RuntimeError("submit_planner_revision called while planner revisions are disabled")
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if len(ids) == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        epoch = control_epoch.to(device=self.device, dtype=torch.long).reshape(-1)
        tasks = task_id.to(device=self.device, dtype=torch.long).reshape(-1)
        revisions = task_revision.to(device=self.device, dtype=torch.long).reshape(-1)
        raw_tts = desired_tts.to(device=self.device, dtype=torch.float32).reshape(-1)
        tts = self._planner_canonicalize_tts(raw_tts, profile)
        pos = target_position.to(device=self.device, dtype=torch.float32)
        vel = target_velocity.to(device=self.device, dtype=torch.float32)
        normal = target_normal.to(device=self.device, dtype=torch.float32)
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        # Envelope the proposed *absolute* deadline relative to the immutable task-begin deadline.
        # The visible-to-proposed delta is still needed only for the one-step slow-only rule.  These
        # are deliberately separate: a latest-value mailbox may skip revisions, but every accepted
        # snapshot must remain inside the same begin-bound envelope.
        elapsed_since_begin = (
            self._planner_begin_tts[ids] - self._planner_truth_tts[ids]
        ).clamp(min=0.0)
        deadline_delta_from_begin = (
            elapsed_since_begin + tts - self._planner_begin_tts[ids]
        )
        deadline_delta_from_visible = tts - self._planner_desired_tts[ids]
        span = (self._planner_strike_step[ids] - self._planner_start_step[ids]).clamp(min=1.0e-6)
        phase = ((self.time_steps_f[ids] - self._planner_start_step[ids]) / span).clamp(0.0, 1.0)
        minimum_tts = self._planner_minimum_finish_time(
            1.0 - phase,
            self._planner_phase_rate[ids],
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        valid = (
            self._planner_active[ids]
            & (self.time_steps_f[ids] < self._planner_strike_step[ids])
            & (epoch == self._planner_control_epoch[ids])
            & (tasks == self._planner_task_id[ids])
            & (revisions > self._planner_task_revision[ids])
            & torch.isfinite(raw_tts)
            & (raw_tts + profile.early_deadline_tolerance_s >= profile.min_tts_s)
            & (raw_tts - profile.early_deadline_tolerance_s <= profile.max_tts_s)
            & (
                deadline_delta_from_begin.abs()
                <= profile.max_deadline_revision_delta_s
            )
            & (tts + profile.early_deadline_tolerance_s >= minimum_tts)
            & torch.isfinite(pos).all(dim=-1)
            & torch.isfinite(vel).all(dim=-1)
            & torch.isfinite(normal).all(dim=-1)
            & ((normal_norm - 1.0).abs() <= profile.normal_unit_tolerance)
            & (
                torch.linalg.vector_norm(
                    pos - self._planner_begin_target_pos[ids], dim=-1
                )
                <= profile.max_position_revision_delta_m
            )
            & (
                torch.linalg.vector_norm(
                    vel - self._planner_begin_target_vel[ids], dim=-1
                )
                <= profile.max_velocity_revision_delta_mps
            )
            & (
                torch.acos(
                    (normal * self._planner_begin_target_normal[ids])
                    .sum(dim=-1)
                    .clamp(-1.0, 1.0)
                )
                <= profile.max_normal_revision_delta_rad
            )
        )
        accepted_ids = ids[valid]
        if len(accepted_ids) > 0:
            self._planner_task_revision[accepted_ids] = revisions[valid]
            self._planner_desired_tts[accepted_ids] = tts[valid]
            self._planner_slow_only_next[accepted_ids] = (
                deadline_delta_from_visible[valid]
                > profile.early_deadline_tolerance_s
            )
        # CommandTerm.reset() indexes every metric with GLOBAL environment ids. ``valid`` is
        # intentionally compact (one row per currently eligible environment), so rebinding either
        # metric to ``valid.float()`` corrupts the mandatory [num_envs] shape as soon as the first
        # short-preparation task leaves the pre-contact set. Keep the registered per-env buffers
        # stable and scatter the compact decision back through its original ids.
        self.metrics["planner_revision_accepted"][ids] = valid.float()
        self.metrics["planner_revision_rejected"][ids] = (~valid).float()
        return valid

    def _advance_planner_phase(self, held: torch.Tensor) -> torch.Tensor:
        """Advance active planner-owned clocks and return their exact clip-frame delta."""

        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        # These are per-step indicators, not held episode metrics. Clearing the full registered
        # tensors here also covers a step with no eligible revision submission; submit() then
        # scatters only the environments that actually attempted a revision.
        self.metrics["planner_revision_accepted"].zero_()
        self.metrics["planner_revision_rejected"].zero_()
        active = self._planner_active
        dt = profile.policy_dt_s
        self._planner_truth_tts[active] = (
            self._planner_truth_tts[active] - dt
        ).clamp(min=0.0)
        # 孪生时钟不截断:触球后转负,供击球窗掩码在 +window 处如约关闭。
        self._planner_truth_tts_signed[active] = (
            self._planner_truth_tts_signed[active] - dt
        )
        remaining_deadline = (self._planner_desired_tts - dt).clamp(min=0.0)
        span = (self._planner_strike_step - self._planner_start_step).clamp(min=1.0e-6)
        phase = ((self.time_steps_f - self._planner_start_step) / span).clamp(0.0, 1.0)
        prestrike = active & (phase < 1.0)
        requested = torch.where(
            remaining_deadline > profile.early_deadline_tolerance_s,
            (1.0 - phase) / remaining_deadline.clamp(min=dt),
            torch.full_like(phase, profile.max_phase_rate_per_s),
        ).clamp(min=0.0, max=profile.max_phase_rate_per_s)
        # Mirror planner_revision.advance_phase / PpPhaseGovernor::Advance exactly.  Near the
        # reachability boundary, dividing remaining phase by the nominal deadline under-requests
        # the rate because it ignores the acceleration ramp; force the cap before applying a
        # one-step slow-only deadline extension.  Without this branch training and deployment
        # diverge specifically on the short-preparation cases this curriculum is meant to expose.
        earliest = self._planner_minimum_finish_time(
            (1.0 - phase).clamp(min=0.0),
            self._planner_phase_rate,
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        requested = torch.where(
            remaining_deadline <= earliest + dt,
            torch.full_like(requested, profile.max_phase_rate_per_s),
            requested,
        )
        requested = torch.where(
            self._planner_slow_only_next,
            torch.minimum(requested, self._planner_phase_rate),
            requested,
        )
        max_delta = profile.max_phase_acceleration_per_s2 * dt
        rate_delta = (requested - self._planner_phase_rate).clamp(
            min=-max_delta, max=max_delta
        )
        new_rate = (self._planner_phase_rate + rate_delta).clamp(
            min=0.0, max=profile.max_phase_rate_per_s
        )
        # Once contact is reached, smoothly return to the native one-frame clock for follow-through.
        native_rate = (1.0 / (span * dt)).clamp(
            max=profile.max_phase_rate_per_s
        )
        post_delta = (native_rate - self._planner_phase_rate).clamp(
            min=-max_delta, max=max_delta
        )
        new_rate = torch.where(
            prestrike,
            new_rate,
            (self._planner_phase_rate + post_delta).clamp(min=0.0),
        )
        new_rate = torch.where(held & active, torch.zeros_like(new_rate), new_rate)
        frame_delta = 0.5 * (self._planner_phase_rate + new_rate) * dt * span
        remaining_frames = (self._planner_strike_step - self.time_steps_f).clamp(min=0.0)
        frame_delta = torch.where(prestrike, torch.minimum(frame_delta, remaining_frames), frame_delta)
        # Keep one full actor interval in reserve whenever the task still has
        # positive time after this update.  The racket command runs later in
        # the same command-manager step, so this is what leaves the final
        # policy_dt target/TTS revision pre-contact and actor-visible.  The
        # following step has remaining_deadline==0 and may reach contact.
        next_rate = torch.minimum(
            torch.full_like(new_rate, profile.max_phase_rate_per_s),
            new_rate + max_delta,
        )
        reserved_phase_distance = 0.5 * (new_rate + next_rate) * dt
        precontact_delta_cap = (
            remaining_frames - reserved_phase_distance * span
        ).clamp(min=0.0)
        reserve_last_actor_interval = prestrike & (
            remaining_deadline > profile.early_deadline_tolerance_s
        )
        frame_delta = torch.where(
            reserve_last_actor_interval,
            torch.minimum(frame_delta, precontact_delta_cap),
            frame_delta,
        )
        frame_delta = torch.where(active, frame_delta.clamp(min=0.0), torch.zeros_like(frame_delta))
        self._planner_phase_rate = torch.where(active, new_rate, self._planner_phase_rate)
        self._planner_slow_only_next[active] = False
        self._planner_desired_tts[active] = remaining_deadline[active]
        self.metrics["planner_phase_rate_per_s"] = self._planner_phase_rate.clone()
        self.metrics["planner_truth_tts_s"] = self._planner_truth_tts.clone()
        return frame_delta

    def _install_event_motion(self, step) -> None:
        """Install clip/start/hold only; carry all physical and policy state across the event."""

        ids = step.install_env_ids
        if len(ids) == 0:
            return
        clips = step.install_clip_ids
        holds = step.install_hold_steps
        # Deliberately no _resample_command, adaptive sampling, simulator write, action write,
        # history reset, or teleport here.  The current robot state and last action continue.
        self._require_canonical_ready_boundary(ids, "event motion install")
        self.clip_id[ids] = clips
        starts = self.motion.seg_start[clips]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        self.speed_scale[ids] = 1.0
        self.hold_counter[ids] = holds
        self.metrics["in_hold"][ids] = (holds > 0).float()
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")

    @property
    def joint_pos(self) -> torch.Tensor:
        # HOLD imitates the READY STAND, not the windup crouch (2026-07-05, pragmatic
        # P2.0): clip frame 0 is an asymmetric mid-crouch (knee 0.62/0.52 vs stand 0.25,
        # left hip_roll +0.14) — imitating it all hold long produced the splayed-feet
        # crouch-stand seen in Gate 2.5/3. During hold the joint reference is the
        # default stand pose; the release (stand -> windup) is exactly the trained
        # stand_start transition. C++ mirrors this (pp_policy: refs.joint_pos =
        # default_q at level 0) — keep them in lockstep.
        if self.canonical_ready_mode:
            measured = self.motion.joint_pos[
                self._action_ball_full_mdp_safe_pose_reference_steps()
            ]
            if (
                self.action_ball_diagnostic_split_ready_teacher
                and self._action_ball_public_task_valid is not None
            ):
                wait = self._action_ball_safe_ready_wait_mask()
                safe = (
                    self._action_ball_dynamic_ready_physical_joint_pos_rad[
                        self.clip_id
                    ]
                )
                measured = torch.where(wait[:, None], safe, measured)
            return measured
        jp = self.motion.joint_pos[
            self._action_ball_full_mdp_safe_pose_reference_steps()
        ]
        dq = self.robot.data.default_joint_pos
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            # Fresh FullMDP freezes selected frame 0 during prepare/recovery;
            # only the initial balance segment uses the runtime default pose.
            # The generic legacy HOLD rule would otherwise keep joint teacher
            # at default while body/anchor already exposed frame 0.
            use_default = (
                self._action_ball_full_mdp_initial_balance_reference_mask()
            )
        else:
            use_default = self.in_hold
        return torch.where(use_default[:, None], dq, jp)

    @property
    def joint_vel(self) -> torch.Tensor:
        # HOLD = a STATIONARY reference (2026-07-05): clip frame 0 is a mid-crouch
        # TRANSIENT (knee +7.8 rad/s, torso -1.11 m/s DOWN in the hopex clips). Feeding
        # its raw velocities through the whole hold taught the policy to fight a phantom
        # squat at soft gains and made "sink slowly" the velocity-reward optimum — the
        # AGI-sim / hardware bare-hold fall (Gate 2.5 P2, 3-5 s tip). A frozen reference
        # is not moving: zero its velocities on held envs. The C++ runner mirrors this
        # (pp_policy zeroes refs.joint_vel in its hold states) — keep them in lockstep.
        jv = self.motion.joint_vel[
            self._action_ball_full_mdp_safe_pose_reference_steps()
        ]
        # R14: at playback speed s the reference joints traverse the same poses s× as fast.
        if self.retiming_active:
            jv = jv * self.speed_scale[:, None]
        stationary = self.in_hold | self._action_ball_safe_ready_wait_mask()
        return torch.where(stationary[:, None], torch.zeros_like(jv), jv)

    @property
    def body_pos_w(self) -> torch.Tensor:
        wait = self._action_ball_safe_ready_wait_mask()
        steps = self._action_ball_full_mdp_safe_pose_reference_steps()
        measured = (
            self.motion.body_pos_w[steps]
            + self._env.scene.env_origins[:, None, :]
        )
        if self._action_ball_safe_ready_body_pos_w is not None:
            measured = torch.where(
                wait[:, None, None],
                self._action_ball_safe_ready_body_pos_w,
                measured,
            )
        return measured

    @property
    def body_quat_w(self) -> torch.Tensor:
        wait = self._action_ball_safe_ready_wait_mask()
        measured = self.motion.body_quat_w[
            self._action_ball_full_mdp_safe_pose_reference_steps()
        ]
        if self._action_ball_safe_ready_body_quat_w is not None:
            measured = torch.where(
                wait[:, None, None],
                self._action_ball_safe_ready_body_quat_w,
                measured,
            )
        return measured

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        # Zeroed during hold — see joint_vel. Un-gated motion_body_lin_vel otherwise
        # pays for tracking frame-0's -1.11 m/s DOWNWARD torso velocity all hold long.
        # R14 retiming composes: scale by playback speed first, then hold-zero wins.
        v = self.motion.body_lin_vel_w[
            self._action_ball_full_mdp_safe_pose_reference_steps()
        ]
        if self.retiming_active:
            v = v * self.speed_scale[:, None, None]
        stationary = self.in_hold | self._action_ball_safe_ready_wait_mask()
        return torch.where(stationary[:, None, None], torch.zeros_like(v), v)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        v = self.motion.body_ang_vel_w[
            self._action_ball_full_mdp_safe_pose_reference_steps()
        ]
        if self.retiming_active:
            v = v * self.speed_scale[:, None, None]
        stationary = self.in_hold | self._action_ball_safe_ready_wait_mask()
        return torch.where(stationary[:, None, None], torch.zeros_like(v), v)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        alv = self.motion.body_lin_vel_w[
            self._action_ball_full_mdp_safe_pose_reference_steps(),
            self.motion_anchor_body_index,
        ]
        if self.retiming_active:
            alv = alv * self.speed_scale[:, None]
        if self.canonical_ready_mode or getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            stationary = self.in_hold | self._action_ball_safe_ready_wait_mask()
            alv = torch.where(stationary[:, None], torch.zeros_like(alv), alv)
        return alv

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        aav = self.motion.body_ang_vel_w[
            self._action_ball_full_mdp_safe_pose_reference_steps(),
            self.motion_anchor_body_index,
        ]
        if self.retiming_active:
            aav = aav * self.speed_scale[:, None]
        if self.canonical_ready_mode or getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            stationary = self.in_hold | self._action_ball_safe_ready_wait_mask()
            aav = torch.where(stationary[:, None], torch.zeros_like(aav), aav)
        return aav

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        anchor_pos_err = self.anchor_pos_w - self.robot_anchor_pos_w
        anchor_rot_err = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        anchor_lin_vel_err = self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w
        anchor_ang_vel_err = self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(anchor_pos_err, dim=-1)
        self.metrics["error_anchor_rot"] = anchor_rot_err
        self.metrics["error_anchor_lin_vel"] = torch.norm(anchor_lin_vel_err, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(anchor_ang_vel_err, dim=-1)
        self.metrics["error_anchor_rot_deg"] = anchor_rot_err * (180.0 / math.pi)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        joint_pos_err = self.joint_pos - self.robot_joint_pos
        joint_vel_err = self.joint_vel - self.robot_joint_vel
        self.metrics["error_joint_pos"] = torch.norm(joint_pos_err, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(joint_vel_err, dim=-1)
        self.metrics["error_joint_pos_mean_abs"] = torch.mean(torch.abs(joint_pos_err), dim=-1)
        self.metrics["error_joint_pos_max_abs"] = torch.max(torch.abs(joint_pos_err), dim=-1).values
        self.metrics["error_joint_vel_mean_abs"] = torch.mean(torch.abs(joint_vel_err), dim=-1)
        self.metrics["error_joint_vel_max_abs"] = torch.max(torch.abs(joint_vel_err), dim=-1).values

        # Log anchor states in an env-origin-relative frame so cross-env averages remain meaningful.
        anchor_ref_rel = self.anchor_pos_w - self._env.scene.env_origins
        anchor_robot_rel = self.robot_anchor_pos_w - self._env.scene.env_origins
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"reference_anchor_pos_{axis}"] = anchor_ref_rel[:, axis_idx]
            self.metrics[f"robot_anchor_pos_{axis}"] = anchor_robot_rel[:, axis_idx]
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = self.anchor_lin_vel_w[:, axis_idx]
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = self.robot_anchor_lin_vel_w[:, axis_idx]

        self.metrics["reference_anchor_speed"] = torch.norm(self.anchor_lin_vel_w, dim=-1)
        self.metrics["robot_anchor_speed"] = torch.norm(self.robot_anchor_lin_vel_w, dim=-1)
        if self._multiseg:
            seg_start = self.motion.seg_start[self.clip_id]
            seg_len = self.motion.seg_len[self.clip_id].clamp(min=2)
            self.metrics["motion_phase"] = (self.time_steps - seg_start).float() / (seg_len - 1).float()
        else:
            self.metrics["motion_phase"] = self.time_steps.float() / max(self.motion.time_step_total - 1, 1)

    def _action_ball_select_or_rewind_action(
        self, env_ids: Sequence[int]
    ) -> None:
        """Select one episode action, or rewind the frozen action at a natural wrap."""

        n = len(env_ids)
        if n == 0:
            return
        ids = (
            env_ids
            if torch.is_tensor(env_ids)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        ids = ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if self._resampling_from_wrap:
            # An action-ball episode has exactly one action identity.  A natural clip wrap starts
            # a new ball/swing against the same birth; it is never a selector opportunity.
            selected = self.clip_id[ids].clone()
        elif int(self.motion.num_segments) == 1:
            selected = torch.zeros(n, dtype=torch.long, device=self.device)
            self.clip_id[ids] = selected
        else:
            if self._balanced_clip_sampler is not None:
                selected = self._balanced_clip_sampler.sample(n)
            else:
                selected = torch.randint(
                    0, int(self.motion.num_segments), (n,), device=self.device
                )
            self.clip_id[ids] = selected

        starts = self.motion.seg_start[selected]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        if self._action_ball_task_ref_for_env is not None:
            # The selected task receipt will install teacher_rate after Racket solves this swing.
            # Do not consume generic retiming RNG, even when its configured range is [1, 1].
            self.speed_scale[ids] = 0.0
        elif self.retiming_active:
            if self._speed_per_clip is not None:
                self.speed_scale[ids] = self._speed_per_clip[selected]
            else:
                speed_lo, speed_hi = self.cfg.speed_scale_range
                self.speed_scale[ids] = sample_uniform(
                    float(speed_lo), float(speed_hi), (n,), device=self.device
                )

        counts = torch.bincount(
            self.clip_id, minlength=int(self.motion.num_segments)
        ).float()
        probabilities = counts / counts.sum().clamp(min=1.0)
        entropy = -(
            probabilities * (probabilities + 1.0e-12).log()
        ).sum()
        self.metrics["sampling_entropy"][:] = entropy / math.log(
            max(int(self.motion.num_segments), 2)
        )
        top_probability, top_index = probabilities.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = top_probability
        self.metrics["sampling_top1_bin"][:] = (
            top_index.float() / max(int(self.motion.num_segments), 1)
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self._action_ball_birth_broker is not None:
            self._action_ball_select_or_rewind_action(env_ids)
            return
        if self._multiseg:
            # HITTER unified policy: each new swing uniformly samples the swing TYPE (clip) and starts at
            # that clip's first frame (reference-state-init at the swing start). The adaptive failure-bin
            # curriculum is single-clip BeyondMimic machinery and is bypassed here.
            n = len(env_ids)
            if n > 0:
                if self._balanced_clip_sampler is not None:
                    new_clip = self._balanced_clip_sampler.sample(n)
                else:
                    new_clip = torch.randint(0, self.motion.num_segments, (n,), device=self.device)
                self.clip_id[env_ids] = new_clip
                # R-c(i) rsi_skip_settle_frames: enter every swing N frames past the clip start —
                # the v5 clips carry a 3-4 frame IK cold-start transient at frame 0 (7.4-15.9 rad/s
                # phantom joint velocities). Wraps go through this same path, so the reference is
                # live-trimmed for the whole run, not only at RSI births. Clamped to the clip's
                # last frame so a short clip can never index out of its segment. 0 (default) = off.
                _skip = int(getattr(self.cfg, "rsi_skip_settle_frames", 0))
                if _skip > 0:
                    self.time_steps[env_ids] = torch.minimum(
                        self.motion.seg_start[new_clip] + _skip,
                        self.motion.seg_start[new_clip] + self.motion.seg_len[new_clip] - 1,
                    )
                else:
                    self.time_steps[env_ids] = self.motion.seg_start[new_clip]
                if self.retiming_active:
                    # R14: re-base the float clock and draw this swing's playback speed.
                    self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
                    if self._speed_per_clip is not None:
                        self.speed_scale[env_ids] = self._speed_per_clip[new_clip]
                    else:
                        s_lo, s_hi = self.cfg.speed_scale_range
                        self.speed_scale[env_ids] = sample_uniform(float(s_lo), float(s_hi), (n,), device=self.device)
            # Report the REAL clip-sampling distribution (repurpose the bin-sampling metrics for clips):
            # entropy of the per-clip env fraction (1.0 = balanced), and the most-sampled clip + its share.
            counts = torch.bincount(self.clip_id, minlength=self.motion.num_segments).float()
            probs = counts / counts.sum().clamp(min=1.0)
            H = -(probs * (probs + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = H / math.log(max(self.motion.num_segments, 2))
            pmax, imax = probs.max(dim=0)
            self.metrics["sampling_top1_prob"][:] = pmax
            self.metrics["sampling_top1_bin"][:] = imax.float() / max(self.motion.num_segments, 1)
            return
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        # R-c(i) rsi_skip_settle_frames (single-clip path): clamp the sampled entry frame to >= N,
        # so the failure-adaptive sampler can never place a birth on the frame-0 IK transient
        # ("越摔越采"的止血). Guarded against clips shorter than N. 0 (default) = off.
        _skip = int(getattr(self.cfg, "rsi_skip_settle_frames", 0))
        if _skip > 0:
            self.time_steps[env_ids] = self.time_steps[env_ids].clamp(
                min=min(_skip, max(int(self.motion.time_step_total) - 1, 0))
            )
        if self.retiming_active:
            # R14: re-base the float clock and draw this swing's playback speed (single-clip path).
            self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
            if self._speed_per_clip is not None:
                self.speed_scale[env_ids] = self._speed_per_clip[self.clip_id[env_ids]]
            else:
                s_lo, s_hi = self.cfg.speed_scale_range
                self.speed_scale[env_ids] = sample_uniform(
                    float(s_lo), float(s_hi), (len(env_ids),), device=self.device
                )

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def balanced_clip_sampler_state_dict(self) -> dict | None:
        """Return exact-resume state for the optional balanced clip sampler."""
        if self._balanced_clip_sampler is None:
            return None
        return self._balanced_clip_sampler.state_dict()

    def load_balanced_clip_sampler_state_dict(self, state: dict | None):
        """Restore balanced clip allocation, rejecting incompatible clip identity/order."""
        if self._balanced_clip_sampler is None:
            if state is not None:
                raise ValueError(
                    "checkpoint contains balanced clip sampler state but "
                    "balanced_clip_sampling is disabled"
                )
            return
        if state is None:
            raise ValueError(
                "balanced_clip_sampling is enabled but checkpoint sampler state is missing"
            )
        self._balanced_clip_sampler.load_state_dict(state)

    @staticmethod
    def _exact_resume_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu").clone()

    def _exact_resume_identity(self) -> dict:
        teacher_contract = self._post_swing_teacher_hard_contract
        teacher_contract_sha256 = (
            None
            if teacher_contract is None
            else hashlib.sha256(_canonical_json_bytes(teacher_contract)).hexdigest()
        )
        identity = {
            "motion": {
                "num_segments": int(self.motion.num_segments),
                "clip_order": tuple(self._motion_files),
                "clip_sha256": tuple(self._motion_file_sha256),
                "segment_lengths": tuple(
                    int(value) for value in self.motion.seg_len.detach().cpu().tolist()
                ),
                "body_names": tuple(str(value) for value in self.cfg.body_names),
                "joint_names": tuple(str(value) for value in self.robot.data.joint_names),
            },
            "adaptive_sampling_config": {
                "bin_count": int(self.bin_count),
                "adaptive_kernel_size": int(self.cfg.adaptive_kernel_size),
                "adaptive_lambda": float(self.cfg.adaptive_lambda),
                "adaptive_uniform_ratio": float(self.cfg.adaptive_uniform_ratio),
                "adaptive_alpha": float(self.cfg.adaptive_alpha),
            },
            "post_swing_replay_config": {
                "post_swing_start_prob": float(self.cfg.post_swing_start_prob),
                "post_swing_buffer_size": int(self.cfg.post_swing_buffer_size),
                "post_swing_min_fill": int(self.cfg.post_swing_min_fill),
                "post_swing_min_hold": int(self.cfg.post_swing_min_hold),
                "post_swing_teacher_hard_contract_sha256": teacher_contract_sha256,
                "post_swing_fail_fast_first_reset": bool(
                    self._post_swing_fail_fast_first_reset
                ),
                "post_swing_first_reset_min_adopted_count": int(
                    self._post_swing_first_reset_min_adopted_count
                ),
                "post_swing_first_reset_min_adopted_fraction": float(
                    self._post_swing_first_reset_min_adopted_fraction
                ),
                "post_swing_first_reset_selection_tolerance": float(
                    self._post_swing_first_reset_selection_tolerance
                ),
                "post_swing_first_reset_require_readback": bool(
                    self._post_swing_first_reset_require_readback
                ),
            },
        }
        if self._action_ball_birth_broker is not None:
            admission_receipt = (
                self.action_ball_motion_admission_hard_contract()
            )
            action_ball_identity = {
                "runtime_contract_sha256": (
                    self._action_ball_runtime_module_bound.RUNTIME_CONTRACT_SHA256
                ),
                "broker_state_schema_version": (
                    self._action_ball_runtime_module_bound.BROKER_STATE_SCHEMA_VERSION
                ),
                "broker_registry_sha256": (
                    self._action_ball_birth_broker.registry_sha256
                ),
                "ordered_action_uids": tuple(
                    self._action_ball_action_uids
                ),
                "trusted_repo_root": str(
                    self._action_ball_trusted_repo_root
                ),
                "motion_admission_receipt_sha256": (
                    admission_receipt["canonical_sha256"]
                ),
                "timing_authority": (
                    (
                        "immutable_n1_fixed_view_row_zero"
                        if self.action_ball_fixed_view_enabled
                        else self._action_ball_runtime_module_bound
                        .TASK_RECEIPT_TIMING_AUTHORITY
                    )
                ),
                "policy_dt_s": float(self._env.step_dt),
                "episode_length_s": (
                    int(self._env.max_episode_length)
                    * float(self._env.step_dt)
                ),
            }
            # Keep the established online/banded identity byte-for-byte
            # stable.  Only the explicitly bound fixed view owns this key.
            if self.action_ball_fixed_view_enabled:
                action_ball_identity["fixed_view_identity_sha256"] = (
                    self._action_ball_fixed_view_identity_sha256
                )
            if self.action_ball_continuous_motion_enabled:
                self._require_action_ball_continuous_parent_authorities()
                profile = self._action_ball_continuous_motion_profile
                action_ball_identity[
                    "continuous_motion_projection_sha256"
                ] = profile[
                    "canonical_sha256"
                ]
                action_ball_identity[
                    "continuous_contract_authority_sha256"
                ] = profile["continuous_contract_authority_sha256"]
                action_ball_identity[
                    "recovery_contract_authority_sha256"
                ] = profile["recovery_contract_authority_sha256"]
                action_ball_identity[
                    "continuous_motion_schedule_projection"
                ] = dict(
                    self._action_ball_continuous_schedule_projection
                )
            identity["action_ball"] = action_ball_identity
        return identity

    def _action_ball_exact_resume_state_dict(self) -> dict:
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="exact checkpoint"
            )
            if (
                self._action_ball_continuous_motion_global_drain_active
                is not None
            ):
                raise RuntimeError(
                    "continuous Motion exact checkpoint is forbidden during global drain"
                )
        if self.action_ball_fixed_view_enabled:
            pending_count = self._action_ball_diagnostic_pending_row_count
            active_generation = self._action_ball_reset_generation > 0
            timing_complete = bool(
                torch.all(
                    ~active_generation
                    | self._action_ball_task_timing_active
                )
            )
            if pending_count != 0 or (
                not timing_complete
                and self._action_ball_expected_shared_racket_state_sha256
                is None
            ):
                raise RuntimeError(
                    "fixed-view exact resume cannot snapshot an incomplete "
                    "Motion/Racket timing handoff"
                )
        broker_state = (
            self._action_ball_fixed_view_broker_state_accessor()
            if self.action_ball_fixed_view_enabled
            else self._action_ball_birth_broker.state_dict()
        )
        self._action_ball_sha256(
            broker_state.get("integrity_sha256"),
            name="broker.integrity_sha256",
        )
        pending = broker_state.get("pending")
        if not isinstance(pending, list) or any(
            not isinstance(row, dict) or row.get("status") != "committed"
            for row in pending
        ):
            raise RuntimeError(
                "action-ball exact resume cannot snapshot an in-flight reserve transaction"
            )
        transcript = {}
        for row in broker_state.get("consumed_receipts", ()):
            if type(row) is not dict:
                raise RuntimeError("action-ball broker transcript is malformed")
            transcript[(row["env_id"], row["reset_generation"])] = row[
                "canonical_sha256"
            ]
        for pending_row in pending:
            row = pending_row["receipt"]
            transcript[(row["env_id"], row["reset_generation"])] = row[
                "canonical_sha256"
            ]
        expected_seen = (
            {
                transcript[(env_id, generation)]
                for env_id, generation in (
                    (int(env), int(generation))
                    for env, generation in broker_state.get(
                        "consumed_generations", ()
                    )
                )
            }
            if self.action_ball_fixed_view_enabled
            else set(transcript.values())
        )
        if expected_seen != self._action_ball_seen_birth_receipts:
            raise RuntimeError(
                "Motion/broker committed birth transcript diverged"
            )
        last = {
            int(env): int(generation)
            for env, generation in broker_state.get("last_generations", ())
        }
        reset_generation = [
            int(value)
            for value in self._action_ball_reset_generation.detach()
            .cpu()
            .tolist()
        ]
        runtime = self._action_ball_runtime_module_bound
        task_ref_rows = []
        for env_id, generation in enumerate(reset_generation):
            receipt_sha = self._action_ball_birth_receipt_sha256[env_id]
            task_ref = self._action_ball_active_task_refs[env_id]
            if generation == 0:
                if (
                    env_id in last
                    or receipt_sha is not None
                    or task_ref is not None
                ):
                    raise RuntimeError(
                        "zero-generation env has broker/birth/task state"
                    )
                task_ref_rows.append(None)
                continue
            if (
                last.get(env_id) != generation
                or transcript.get((env_id, generation)) != receipt_sha
            ):
                raise RuntimeError(
                    "Motion current generation/receipt differs from broker transcript"
                )
            if self.action_ball_fixed_view_enabled:
                if task_ref is not None:
                    raise RuntimeError(
                        "fixed-view Motion state must not contain a legacy task ref"
                    )
                task_ref_rows.append(None)
            else:
                if type(task_ref) is not runtime.ActionTaskReceiptRef:
                    raise RuntimeError(
                        "positive-generation env lacks an exact active task ref"
                    )
                live_ref = self._action_ball_task_ref_for_env(env_id)
                if live_ref != task_ref:
                    raise RuntimeError(
                        "Motion active task ref differs from Racket authority"
                    )
                resolved = self._action_ball_task_receipt_resolver(task_ref)
                self._validate_action_ball_task_ref_and_receipt(
                    task_ref, resolved, env_id=env_id
                )
                task_ref_rows.append(task_ref.to_dict())
        admission_receipt = (
            self.action_ball_motion_admission_hard_contract()
        )
        result = {
            "runtime_contract_sha256": (
                self._action_ball_runtime_module_bound.RUNTIME_CONTRACT_SHA256
            ),
            "broker_registry_sha256": (
                self._action_ball_birth_broker.registry_sha256
            ),
            "motion_admission_receipt_sha256": (
                admission_receipt["canonical_sha256"]
            ),
            # Racket is the sole owner of sampler/provider/domain/broker/pool/task bytes.  Motion
            # stores only its canonical full-state digest plus opaque per-env task references.
            "shared_racket_state_sha256": (
                self.action_ball_shared_racket_state_sha256()
            ),
            "reset_generation": self._exact_resume_cpu_tensor(
                self._action_ball_reset_generation
            ),
            "swing_generation": self._exact_resume_cpu_tensor(
                self._action_ball_swing_generation
            ),
            "birth_receipt_sha256": list(
                self._action_ball_birth_receipt_sha256
            ),
            "seen_birth_receipts": sorted(
                self._action_ball_seen_birth_receipts
            ),
            "active_task_refs": task_ref_rows,
        }
        if self.action_ball_fixed_view_enabled:
            result["fixed_view_identity_sha256"] = (
                self._action_ball_fixed_view_identity_sha256
            )
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            result["continuous_motion_leaf"] = (
                self._action_ball_continuous_motion_checkpoint_payload()
            )
        return result

    # ``action_ball_shared_broker_state_sha256`` was deleted 2026-08-06: zero
    # callers.  Its docstring advertised "for runner ordering checks", but no
    # runner ever read it -- the digest the runner does read is
    # ``action_ball_shared_racket_state_sha256`` below, which covers the broker
    # bytes along with every other shared action-ball authority byte.

    def action_ball_shared_racket_state_sha256(self) -> str:
        """Return Racket's digest over every shared action-ball authority byte."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="shared Racket digest read"
            )
        if self._action_ball_shared_state_sha256_accessor is None:
            raise RuntimeError("action-ball shared Racket digest is not bound")
        return self._action_ball_sha256(
            self._action_ball_shared_state_sha256_accessor(),
            name="Racket.action_ball_shared_state_sha256",
        )

    def finalize_action_ball_exact_resume(self) -> None:
        """Verify Racket-first shared restore against Motion's staged digest and refs."""

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="exact resume finalization"
            )
        expected = self._action_ball_expected_shared_racket_state_sha256
        if expected is None:
            raise RuntimeError(
                "Motion action-ball exact resume has no staged Racket digest"
            )
        if self.action_ball_shared_racket_state_sha256() != expected:
            raise RuntimeError(
                "live Racket state differs from Motion exact-resume handoff"
            )
        # Re-run broker transcript plus opaque task-ref resolution on the live Racket restore.
        snapshot = self._action_ball_exact_resume_state_dict()
        if snapshot["shared_racket_state_sha256"] != expected:
            raise RuntimeError(
                "live Racket task/birth refs differ after exact resume"
            )

    def exact_resume_state_dict(self) -> dict:
        """Return every persistent MotionCommand state that shapes the next rollout."""
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            # Fail before identity serialization, CPU copies, or replay-ring
            # clones when a production reveal epoch is retained.
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="public exact checkpoint"
            )
        # Per-env clip/hold/planner/event clocks are deliberately absent: the runner performs one
        # full env reset after loading. The two stagger pending flags are also construction state,
        # not curriculum state—the documented resume path must re-spread that freshly reset cohort.
        # The fields below are the state that survives episode boundaries and changes later draws.
        ring_values = (
            self._post_swing_root,
            self._post_swing_joint_pos,
            self._post_swing_joint_vel,
        )
        if any(value is None for value in ring_values) and not all(
            value is None for value in ring_values
        ):
            raise RuntimeError("post-swing replay ring is only partially allocated")
        state = {
            "state_kind": self._EXACT_RESUME_STATE_KIND,
            "schema_version": (
                self._ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION
                if self._action_ball_birth_broker is not None
                else self._EXACT_RESUME_STATE_SCHEMA_VERSION
            ),
            "identity": self._exact_resume_identity(),
            "adaptive_sampling": {
                "bin_failed_count": self._exact_resume_cpu_tensor(
                    self.bin_failed_count
                ),
                "current_bin_failed": self._exact_resume_cpu_tensor(
                    self._current_bin_failed
                ),
            },
            "post_swing_replay": {
                # Explicit null is part of the schema: an unallocated/disabled ring is state,
                # not a missing key that a loader may silently reinterpret.
                "root": (
                    None
                    if self._post_swing_root is None
                    else self._exact_resume_cpu_tensor(self._post_swing_root)
                ),
                "joint_pos": (
                    None
                    if self._post_swing_joint_pos is None
                    else self._exact_resume_cpu_tensor(self._post_swing_joint_pos)
                ),
                "joint_vel": (
                    None
                    if self._post_swing_joint_vel is None
                    else self._exact_resume_cpu_tensor(self._post_swing_joint_vel)
                ),
                "ptr": int(self._post_swing_ptr),
                "count": int(self._post_swing_count),
                "first_reset_checked": bool(
                    self._post_swing_first_reset_checked
                ),
            },
            "balanced_clip_sampler": self.balanced_clip_sampler_state_dict(),
        }
        if self._action_ball_birth_broker is not None:
            # Diagnostic A211/C211 has no promotion authority, but its sampler, question/cache,
            # WAIT and command continuation state are still real mutable bytes.  Authorization and
            # recoverability are independent axes: serialize the full schema-5 handoff and keep the
            # diagnostic brand inside Racket's hard contract.
            #
            # 人话勘误(2026-08-07):这段以前的结尾是"而不是悄悄把每个诊断
            # checkpoint 变成只能重开"。现在它**就是**只能重开的 —— 但关键在
            # "不是悄悄"。诊断跑的三份存档各自自陈范围:solver 的
            # ``task_transcript_scope=diagnostic_live_births_only``、池子的
            # ``pool_state_scope=diagnostic_live_births_only``、broker 的
            # ``birth_transcript_scope=live_envs_only``;三者的载入端都点名拒绝,
            # 并说明理由。存档照存不误(权重、取证数据、训练连续性都在),
            # 只是不承载"从第 N 步接着跑"所需的逐出生历史 —— 那半份材料这条
            # 快路本来就故意不写,让它悄悄变成零才是真正的错。
            state["action_ball_birth"] = (
                self._action_ball_exact_resume_state_dict()
            )
        return state

    @staticmethod
    def _validate_exact_resume_tensor(
        value,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        nonnegative: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be serialized on the CPU")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} shape/dtype mismatch: checkpoint={tuple(value.shape)}/"
                f"{value.dtype}, runtime={shape}/{dtype}"
            )
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        if nonnegative and bool((value < 0).any()):
            raise ValueError(f"{name} contains negative counts")
        return value.detach().clone()

    def _prepare_action_ball_exact_resume_state(
        self, value
    ) -> dict:
        expected = {
            "runtime_contract_sha256",
            "broker_registry_sha256",
            "motion_admission_receipt_sha256",
            "shared_racket_state_sha256",
            "reset_generation",
            "swing_generation",
            "birth_receipt_sha256",
            "seen_birth_receipts",
            "active_task_refs",
        }
        if self.action_ball_fixed_view_enabled:
            expected.add("fixed_view_identity_sha256")
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            expected.add("continuous_motion_leaf")
        if type(value) is not dict or set(value) != expected:
            raise ValueError(
                "Motion action-ball exact-resume state keys do not match the strict schema"
            )
        runtime = self._action_ball_runtime_module_bound
        admission_receipt = (
            self.action_ball_motion_admission_hard_contract()
        )
        if (
            value["runtime_contract_sha256"]
            != runtime.RUNTIME_CONTRACT_SHA256
            or value["broker_registry_sha256"]
            != self._action_ball_birth_broker.registry_sha256
            or value["motion_admission_receipt_sha256"]
            != admission_receipt["canonical_sha256"]
            or (
                self.action_ball_fixed_view_enabled
                and value["fixed_view_identity_sha256"]
                != self._action_ball_fixed_view_identity_sha256
            )
        ):
            raise ValueError(
                "Motion action-ball exact-resume immutable identity differs"
            )
        shared_racket_state_sha256 = self._action_ball_sha256(
            value["shared_racket_state_sha256"],
            name="action_ball.shared_racket_state_sha256",
        )
        reset_generation = self._validate_exact_resume_tensor(
            value["reset_generation"],
            name="action_ball.reset_generation",
            shape=(self.num_envs,),
            dtype=self._action_ball_reset_generation.dtype,
            nonnegative=True,
        )
        swing_generation = self._validate_exact_resume_tensor(
            value["swing_generation"],
            name="action_ball.swing_generation",
            shape=(self.num_envs,),
            dtype=self._action_ball_swing_generation.dtype,
            nonnegative=True,
        )
        if (
            bool((reset_generation > self._ACTION_BALL_INT64_MAX).any())
            or bool((swing_generation > self._ACTION_BALL_INT64_MAX).any())
        ):
            raise ValueError("Motion action-ball generation exceeds int64")

        current = value["birth_receipt_sha256"]
        seen = value["seen_birth_receipts"]
        task_ref_rows = value["active_task_refs"]
        if (
            type(current) is not list
            or len(current) != self.num_envs
            or type(seen) is not list
            or type(task_ref_rows) is not list
            or len(task_ref_rows) != self.num_envs
        ):
            raise ValueError(
                "Motion action-ball receipt state has invalid container shape"
            )
        current_receipts = []
        for index, digest in enumerate(current):
            if digest is not None:
                digest = self._action_ball_sha256(
                    digest,
                    name=f"action_ball.birth_receipt_sha256[{index}]",
                )
            current_receipts.append(digest)
        seen_receipts = [
            self._action_ball_sha256(
                digest, name=f"action_ball.seen_birth_receipts[{index}]"
            )
            for index, digest in enumerate(seen)
        ]
        if (
            seen_receipts != sorted(seen_receipts)
            or len(set(seen_receipts)) != len(seen_receipts)
        ):
            raise ValueError(
                "Motion action-ball seen receipt list must be sorted and unique"
            )

        reset_rows = [
            int(item) for item in reset_generation.tolist()
        ]
        seen_receipt_set = set(seen_receipts)
        task_refs = []
        for env_id, generation in enumerate(reset_rows):
            digest = current_receipts[env_id]
            ref_row = task_ref_rows[env_id]
            task_ref = None
            if ref_row is not None:
                if self.action_ball_fixed_view_enabled:
                    raise ValueError(
                        "fixed-view Motion resume must not contain task refs"
                    )
                task_ref = runtime.ActionTaskReceiptRef.from_dict(ref_row)
            if generation == 0:
                if digest is not None or task_ref is not None:
                    raise ValueError(
                        "zero-generation Motion env has a birth/task ref"
                    )
            else:
                if digest is None or digest not in seen_receipt_set:
                    raise ValueError(
                        "positive-generation Motion env lacks a seen birth ref"
                    )
                if self.action_ball_fixed_view_enabled:
                    if task_ref is not None:
                        raise ValueError(
                            "positive-generation fixed-view env has a task ref"
                        )
                elif (
                    type(task_ref) is not runtime.ActionTaskReceiptRef
                    or task_ref.env_id != env_id
                    or task_ref.reset_generation != generation
                    or task_ref.swing_generation
                    != int(swing_generation[env_id].item())
                    or task_ref.birth_sha256 != digest
                    or not (
                        0
                        <= task_ref.action_slot
                        < len(self._action_ball_action_uids)
                    )
                    or task_ref.action_uid
                    != self._action_ball_action_uids[
                        task_ref.action_slot
                    ]
                ):
                    raise ValueError(
                        "positive-generation Motion env has a mismatched task ref"
                    )
            task_refs.append(task_ref)
        prepared = {
            "shared_racket_state_sha256": shared_racket_state_sha256,
            "reset_generation": reset_generation,
            "swing_generation": swing_generation,
            "birth_receipt_sha256": current_receipts,
            "seen_birth_receipts": set(seen_receipts),
            "active_task_refs": task_refs,
        }
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            prepared["continuous_motion_leaf"] = (
                self._prepare_action_ball_continuous_motion_checkpoint(
                    value["continuous_motion_leaf"]
                )
            )
        return prepared

    def validate_exact_resume_state_dict(
        self, state: dict, *, strict: bool = True
    ) -> None:
        """Read-only strict parser used by runner-wide atomic preflight."""

        self.load_exact_resume_state_dict(
            state, strict=strict, _validate_only=True
        )

    def load_exact_resume_state_dict(
        self,
        state: dict,
        strict: bool = True,
        *,
        _validate_only: bool = False,
    ) -> None:
        """Restore only an exact schema/config/clip identity match."""
        if strict is not True:
            raise ValueError("MotionCommand exact resume supports only strict=True")
        if type(state) is not dict:
            raise ValueError("MotionCommand exact resume state must be a dictionary")
        expected_keys = {
            "state_kind",
            "schema_version",
            "identity",
            "adaptive_sampling",
            "post_swing_replay",
            "balanced_clip_sampler",
        }
        action_ball_bound = self._action_ball_birth_broker is not None
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="exact checkpoint restore"
            )
            if (
                self._action_ball_continuous_motion_global_drain_active
                is not None
            ):
                raise RuntimeError(
                    "continuous Motion exact restore is forbidden during global drain"
                )
        expected_schema = (
            self._ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION
            if action_ball_bound
            else self._EXACT_RESUME_STATE_SCHEMA_VERSION
        )
        if action_ball_bound:
            expected_keys.add("action_ball_birth")
        if set(state) != expected_keys:
            raise ValueError(
                "MotionCommand exact resume state keys do not match the strict schema"
            )
        if state["state_kind"] != self._EXACT_RESUME_STATE_KIND:
            raise ValueError("MotionCommand exact resume state_kind does not match")
        if state["schema_version"] != expected_schema:
            raise ValueError(
                "MotionCommand exact resume schema_version is unsupported"
            )
        if state["identity"] != self._exact_resume_identity():
            raise ValueError(
                "MotionCommand exact resume motion/config/clip identity does not match"
            )
        action_ball_state = (
            self._prepare_action_ball_exact_resume_state(
                state["action_ball_birth"]
            )
            if action_ball_bound
            else None
        )

        adaptive = state["adaptive_sampling"]
        if type(adaptive) is not dict or set(adaptive) != {
            "bin_failed_count",
            "current_bin_failed",
        }:
            raise ValueError(
                "MotionCommand adaptive_sampling state does not match the strict schema"
            )
        bin_shape = tuple(self.bin_failed_count.shape)
        bin_failed_count = self._validate_exact_resume_tensor(
            adaptive["bin_failed_count"],
            name="bin_failed_count",
            shape=bin_shape,
            dtype=self.bin_failed_count.dtype,
            nonnegative=True,
        )
        current_bin_failed = self._validate_exact_resume_tensor(
            adaptive["current_bin_failed"],
            name="current_bin_failed",
            shape=tuple(self._current_bin_failed.shape),
            dtype=self._current_bin_failed.dtype,
            nonnegative=True,
        )

        replay = state["post_swing_replay"]
        replay_keys = {
            "root",
            "joint_pos",
            "joint_vel",
            "ptr",
            "count",
            "first_reset_checked",
        }
        if type(replay) is not dict or set(replay) != replay_keys:
            raise ValueError(
                "MotionCommand post_swing_replay state does not match the strict schema"
            )
        ptr = replay["ptr"]
        count = replay["count"]
        first_reset_checked = replay["first_reset_checked"]
        if (
            type(ptr) is not int
            or type(count) is not int
            or type(first_reset_checked) is not bool
        ):
            raise ValueError(
                "post-swing replay ptr/count/first_reset_checked have invalid types"
            )
        size = int(self.cfg.post_swing_buffer_size)
        if not (0 <= ptr < size) or not (0 <= count <= size):
            raise ValueError("post-swing replay ptr/count are outside the configured ring")
        ring_values = (replay["root"], replay["joint_pos"], replay["joint_vel"])
        ring_is_none = tuple(value is None for value in ring_values)
        if any(ring_is_none) and not all(ring_is_none):
            raise ValueError("post-swing replay ring is only partially serialized")
        if all(ring_is_none):
            if ptr != 0 or count != 0:
                raise ValueError(
                    "unallocated post-swing replay ring requires ptr=count=0"
                )
            if self._post_swing_teacher_hard_contract is not None:
                raise ValueError(
                    "configured post-swing teacher cannot restore an unallocated ring"
                )
            root = joint_pos = joint_vel = None
        else:
            joint_count = int(self.robot.data.joint_pos.shape[-1])
            root = self._validate_exact_resume_tensor(
                replay["root"],
                name="post_swing_root",
                shape=(size, 13),
                dtype=self.robot.data.root_state_w.dtype,
            )
            joint_pos = self._validate_exact_resume_tensor(
                replay["joint_pos"],
                name="post_swing_joint_pos",
                shape=(size, joint_count),
                dtype=self.robot.data.joint_pos.dtype,
            )
            joint_vel = self._validate_exact_resume_tensor(
                replay["joint_vel"],
                name="post_swing_joint_vel",
                shape=(size, joint_count),
                dtype=self.robot.data.joint_vel.dtype,
            )

        sampler_state = state["balanced_clip_sampler"]
        if self._balanced_clip_sampler is None:
            if sampler_state is not None:
                raise ValueError(
                    "checkpoint contains balanced clip sampler state but "
                    "balanced_clip_sampling is disabled"
                )
        else:
            if sampler_state is None:
                raise ValueError(
                    "balanced_clip_sampling is enabled but checkpoint sampler state is missing"
                )
            self._balanced_clip_sampler.validate_state_dict(sampler_state)

        # Everything above is pure parsing/staging.  Runner preflight calls this branch for every
        # command before it allows the first live mutation anywhere in the checkpoint graph.
        if _validate_only:
            return

        # Complete every potentially fallible CPU->device transfer before the
        # first live Motion byte changes.  The application block below is then
        # same-device copy-only; a CUDA allocation/transfer failure cannot
        # strand a mixed live/checkpoint state or require a second allocation
        # while rolling back.
        bin_failed_count_after = bin_failed_count.to(
            device=self.bin_failed_count.device
        )
        current_bin_failed_after = current_bin_failed.to(
            device=self._current_bin_failed.device
        )
        root_after = None if root is None else root.to(device=self.device)
        joint_pos_after = (
            None if joint_pos is None else joint_pos.to(device=self.device)
        )
        joint_vel_after = (
            None if joint_vel is None else joint_vel.to(device=self.device)
        )
        continuous_motion_leaf_after = None
        action_ball_reset_generation_after = None
        action_ball_swing_generation_after = None
        if action_ball_bound:
            action_ball_reset_generation_after = action_ball_state[
                "reset_generation"
            ].to(device=self.device)
            action_ball_swing_generation_after = action_ball_state[
                "swing_generation"
            ].to(device=self.device)
        if (
            action_ball_bound
            and getattr(
                self,
                "_action_ball_continuous_fresh_motion_lane_bound",
                False,
            )
        ):
            leaf_state = action_ball_state["continuous_motion_leaf"]
            continuous_motion_leaf_after = {
                "device_owner_mutation_version": leaf_state[
                    "device_owner_mutation_version"
                ].to(device=self.device),
                "terminal_resolution_total_device": leaf_state[
                    "terminal_resolution_total_device"
                ].to(device=self.device),
                "fault_count_device": leaf_state["fault_count_device"].to(
                    device=self.device
                ),
                "safe_ready_pending_count": leaf_state[
                    "safe_ready_pending_count"
                ],
                "tensors": {
                    field_name: leaf_state["tensors"][field_name].to(
                        device=getattr(self, attr_name).device
                    )
                    for field_name, attr_name, _nonnegative in (
                        _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
                    )
                },
            }

        # Racket owns and restores the shared evaluator/curriculum/provider/domain/broker/pool/task
        # graph before this local load.  Motion never restores those bytes; it stages their full
        # digest plus opaque local refs for the runner's post-load finalize.
        sampler_before = self.balanced_clip_sampler_state_dict()
        bin_before = self.bin_failed_count.clone()
        current_bin_before = self._current_bin_failed.clone()
        replay_before = (
            self._post_swing_root,
            self._post_swing_joint_pos,
            self._post_swing_joint_vel,
            self._post_swing_ptr,
            self._post_swing_count,
            self._post_swing_first_reset_checked,
        )
        if action_ball_bound:
            action_ball_before = (
                self._action_ball_reset_generation.clone(),
                self._action_ball_swing_generation.clone(),
                list(self._action_ball_birth_receipt_sha256),
                set(self._action_ball_seen_birth_receipts),
                list(self._action_ball_active_task_refs),
                self._action_ball_task_timing_active.clone(),
                self._action_ball_task_pending_elapsed_s.clone(),
                self._action_ball_task_age_s.clone(),
                self._action_ball_time_to_contact_s.clone(),
                self._action_ball_teacher_rate.clone(),
                self._action_ball_scaled_t_hit_s.clone(),
                self._action_ball_scaled_t_cycle_s.clone(),
                self._action_ball_pre_swing_wait_s.clone(),
                self._action_ball_expected_shared_racket_state_sha256,
                self.clip_id.clone(),
            )
            continuous_motion_leaf_before = (
                (
                    self._action_ball_continuous_motion_mutation_version,
                    self._action_ball_continuous_motion_device_mutation_version.clone(),
                    self._action_ball_continuous_motion_next_serial,
                    self._action_ball_continuous_motion_selected_reset_next_serial,
                    self._action_ball_continuous_motion_terminal_resolution_total,
                    self._action_ball_continuous_motion_terminal_resolution_total_device.clone(),
                    self._action_ball_continuous_motion_fault_count_device.clone(),
                    self._action_ball_continuous_motion_global_drain_sequence,
                    self._action_ball_continuous_motion_global_drain_last_update,
                    self._action_ball_continuous_motion_global_drain_last_completed_steps,
                    self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version,
                    self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack,
                    self._action_ball_continuous_published_common_step,
                    self._action_ball_safe_ready_pending_count,
                    tuple(
                        (
                            attr_name,
                            getattr(self, attr_name).clone(),
                        )
                        for _field_name, attr_name, _nonnegative in (
                            _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
                        )
                    ),
                )
                if getattr(
                    self,
                    "_action_ball_continuous_fresh_motion_lane_bound",
                    False,
                )
                else None
            )
        else:
            action_ball_before = None
            continuous_motion_leaf_before = None
        try:
            self.load_balanced_clip_sampler_state_dict(
                state["balanced_clip_sampler"]
            )
            self.bin_failed_count.copy_(bin_failed_count_after)
            self._current_bin_failed.copy_(current_bin_failed_after)
            self._post_swing_root = root_after
            self._post_swing_joint_pos = joint_pos_after
            self._post_swing_joint_vel = joint_vel_after
            self._post_swing_ptr = ptr
            self._post_swing_count = count
            self._post_swing_first_reset_checked = first_reset_checked
            if action_ball_bound:
                self._action_ball_reset_generation.copy_(
                    action_ball_reset_generation_after
                )
                self._action_ball_swing_generation.copy_(
                    action_ball_swing_generation_after
                )
                self._action_ball_birth_receipt_sha256 = list(
                    action_ball_state["birth_receipt_sha256"]
                )
                self._action_ball_seen_birth_receipts = set(
                    action_ball_state["seen_birth_receipts"]
                )
                self._action_ball_active_task_refs = list(
                    action_ball_state["active_task_refs"]
                )
                # Positive-generation refs are Motion's checkpoint-local
                # action authority.  Reconstruct clip_id from them before the
                # post-load finalizer validates each live Racket receipt.
                # This is local tensor state only: no RNG or simulator write.
                for env_id, task_ref in enumerate(
                    self._action_ball_active_task_refs
                ):
                    if task_ref is not None:
                        self.clip_id[env_id] = task_ref.action_slot
                if self.action_ball_fixed_view_enabled:
                    self.clip_id.zero_()
                # Legacy resume still resets these clocks.  Fresh R08 restores
                # the complete owner payload below; silently zeroing it would
                # turn a mid-task checkpoint into a different chronology.
                if not getattr(
                    self,
                    "_action_ball_continuous_fresh_motion_lane_bound",
                    False,
                ):
                    self._action_ball_task_timing_active.zero_()
                    self._action_ball_task_pending_elapsed_s.zero_()
                    self._action_ball_task_age_s.zero_()
                    self._action_ball_time_to_contact_s.zero_()
                    self._action_ball_teacher_rate.zero_()
                    self._action_ball_scaled_t_hit_s.zero_()
                    self._action_ball_scaled_t_cycle_s.zero_()
                    self._action_ball_pre_swing_wait_s.zero_()
                self._action_ball_expected_shared_racket_state_sha256 = (
                    action_ball_state["shared_racket_state_sha256"]
                )
                if getattr(
                    self,
                    "_action_ball_continuous_fresh_motion_lane_bound",
                    False,
                ):
                    leaf_state = action_ball_state[
                        "continuous_motion_leaf"
                    ]
                    self._action_ball_continuous_motion_device_mutation_version.copy_(
                        continuous_motion_leaf_after[
                            "device_owner_mutation_version"
                        ]
                    )
                    self._action_ball_continuous_motion_mutation_version = (
                        leaf_state["owner_mutation_version"]
                    )
                    self._action_ball_continuous_motion_next_serial = (
                        leaf_state["next_serial"]
                    )
                    self._action_ball_continuous_motion_selected_reset_next_serial = (
                        leaf_state["selected_reset_next_serial"]
                    )
                    self._action_ball_continuous_motion_terminal_resolution_total = (
                        leaf_state["terminal_resolution_total"]
                    )
                    self._action_ball_continuous_motion_terminal_resolution_total_device.copy_(
                        continuous_motion_leaf_after[
                            "terminal_resolution_total_device"
                        ]
                    )
                    self._action_ball_continuous_motion_fault_count_device.copy_(
                        continuous_motion_leaf_after[
                            "fault_count_device"
                        ]
                    )
                    self._action_ball_continuous_motion_global_drain_sequence = (
                        leaf_state["global_drain_sequence"]
                    )
                    self._action_ball_continuous_motion_global_drain_last_update = (
                        leaf_state["global_drain_last_update"]
                    )
                    self._action_ball_continuous_motion_global_drain_last_completed_steps = (
                        leaf_state["global_drain_last_completed_steps"]
                    )
                    self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = (
                        leaf_state[
                            "global_drain_last_acknowledged_mutation_version"
                        ]
                    )
                    self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = (
                        leaf_state["checkpoint_requires_global_drain_ack"]
                    )
                    self._action_ball_continuous_published_common_step = (
                        leaf_state["published_common_step"]
                    )
                    self._action_ball_safe_ready_pending_count = (
                        continuous_motion_leaf_after[
                            "safe_ready_pending_count"
                        ]
                    )
                    leaf_tensors = leaf_state["tensors"]
                    for field_name, attr_name, _nonnegative in (
                        _ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
                    ):
                        getattr(self, attr_name).copy_(
                            continuous_motion_leaf_after["tensors"][
                                field_name
                            ]
                        )
                    self._invalidate_action_ball_continuous_observation_publication()
        except Exception:
            # Restore live state without invoking reset/resample or any simulator setter.
            self.load_balanced_clip_sampler_state_dict(sampler_before)
            self.bin_failed_count.copy_(bin_before)
            self._current_bin_failed.copy_(current_bin_before)
            (
                self._post_swing_root,
                self._post_swing_joint_pos,
                self._post_swing_joint_vel,
                self._post_swing_ptr,
                self._post_swing_count,
                self._post_swing_first_reset_checked,
            ) = replay_before
            if action_ball_bound:
                (
                    reset_before,
                    swing_before,
                    receipt_before,
                    seen_before,
                    task_refs_before,
                    timing_active_before,
                    pending_elapsed_before,
                    task_age_before,
                    time_to_contact_before,
                    teacher_rate_before,
                    scaled_t_hit_before,
                    scaled_t_cycle_before,
                    pre_swing_wait_before,
                    expected_racket_before,
                    clip_id_before,
                ) = action_ball_before
                self._action_ball_reset_generation.copy_(reset_before)
                self._action_ball_swing_generation.copy_(swing_before)
                self._action_ball_birth_receipt_sha256 = receipt_before
                self._action_ball_seen_birth_receipts = seen_before
                self._action_ball_active_task_refs = task_refs_before
                self._action_ball_task_timing_active.copy_(
                    timing_active_before
                )
                self._action_ball_task_pending_elapsed_s.copy_(
                    pending_elapsed_before
                )
                self._action_ball_task_age_s.copy_(task_age_before)
                self._action_ball_time_to_contact_s.copy_(
                    time_to_contact_before
                )
                self._action_ball_teacher_rate.copy_(teacher_rate_before)
                self._action_ball_scaled_t_hit_s.copy_(scaled_t_hit_before)
                self._action_ball_scaled_t_cycle_s.copy_(
                    scaled_t_cycle_before
                )
                self._action_ball_pre_swing_wait_s.copy_(
                    pre_swing_wait_before
                )
                self._action_ball_expected_shared_racket_state_sha256 = (
                    expected_racket_before
                )
                self.clip_id.copy_(clip_id_before)
                if continuous_motion_leaf_before is not None:
                    (
                        leaf_version_before,
                        leaf_device_version_before,
                        leaf_serial_before,
                        leaf_selected_reset_serial_before,
                        leaf_terminal_total_before,
                        leaf_terminal_device_before,
                        leaf_fault_count_before,
                        leaf_drain_sequence_before,
                        leaf_drain_update_before,
                        leaf_drain_steps_before,
                        leaf_drain_acknowledged_mutation_before,
                        leaf_checkpoint_requires_drain_ack_before,
                        leaf_published_common_step_before,
                        leaf_safe_ready_pending_count_before,
                        leaf_tensors_before,
                    ) = continuous_motion_leaf_before
                    self._action_ball_continuous_motion_device_mutation_version.copy_(
                        leaf_device_version_before
                    )
                    self._action_ball_continuous_motion_mutation_version = (
                        leaf_version_before
                    )
                    self._action_ball_continuous_motion_next_serial = (
                        leaf_serial_before
                    )
                    self._action_ball_continuous_motion_selected_reset_next_serial = (
                        leaf_selected_reset_serial_before
                    )
                    self._action_ball_continuous_motion_terminal_resolution_total = (
                        leaf_terminal_total_before
                    )
                    self._action_ball_continuous_motion_terminal_resolution_total_device.copy_(
                        leaf_terminal_device_before
                    )
                    self._action_ball_continuous_motion_fault_count_device.copy_(
                        leaf_fault_count_before
                    )
                    self._action_ball_continuous_motion_global_drain_sequence = (
                        leaf_drain_sequence_before
                    )
                    self._action_ball_continuous_motion_global_drain_last_update = (
                        leaf_drain_update_before
                    )
                    self._action_ball_continuous_motion_global_drain_last_completed_steps = (
                        leaf_drain_steps_before
                    )
                    self._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = (
                        leaf_drain_acknowledged_mutation_before
                    )
                    self._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = (
                        leaf_checkpoint_requires_drain_ack_before
                    )
                    self._action_ball_continuous_published_common_step = (
                        leaf_published_common_step_before
                    )
                    self._action_ball_safe_ready_pending_count = (
                        leaf_safe_ready_pending_count_before
                    )
                    for attr_name, tensor_before in leaf_tensors_before:
                        getattr(self, attr_name).copy_(tensor_before)
            raise

    def _capture_post_swing_states(self, env_ids: torch.Tensor):
        """A8: snapshot end-of-swing robot states (wrap envs only) into the ring buffer.

        Wrapped envs necessarily completed their swing physically (no teleport happened and they
        reached the clip's final frame), so every buffer entry is a genuine follow-through state.
        Root position is stored origin-relative; write pairs root_state_w <->
        write_root_state_to_sim (com-frame velocities) to match the stand/RSI branches.
        """
        # Receipt-backed science pairs keep one identical exogenous reset distribution.  Letting
        # each arm overwrite it with policy-owned wraps would reintroduce the treatment-dependent
        # curriculum that this cold-start path exists to remove.
        if self._post_swing_teacher_hard_contract is not None:
            return
        n = len(env_ids)
        if n == 0:
            return
        root = self.robot.data.root_state_w[env_ids].clone()
        root[:, :3] -= self._env.scene.env_origins[env_ids]
        jp = self.robot.data.joint_pos[env_ids].clone()
        jv = self.robot.data.joint_vel[env_ids].clone()
        if self._post_swing_capture_output_dir is not None and not self._post_swing_capture_complete:
            if (
                self._post_swing_capture_runtime_hard_contract_sha256 is None
                or self._post_swing_capture_claim_sha256 is None
            ):
                raise RuntimeError(
                    "natural-wrap capture cannot step before runtime-contract equality is bound"
                )
            # This is the sole source path that populates the capture accumulator: arrays are read
            # directly from the live articulation tensors above.  No writer/capability API accepts
            # arbitrary caller-owned arrays.  The caller remains ordinary Python, so the artifact
            # records reviewed-source/O_EXCL evidence rather than claiming cryptographic callback
            # provenance.  CPU conversion also synchronizes the CUDA producer before publication.
            root_np = root.detach().to(device="cpu", dtype=torch.float32).numpy()
            joint_pos_np = jp.detach().to(device="cpu", dtype=torch.float32).numpy()
            joint_vel_np = jv.detach().to(device="cpu", dtype=torch.float32).numpy()
            rows = min(
                root_np.shape[0] if root_np.ndim == 2 else 0,
                self._post_swing_capture_target_count - self._post_swing_capture_count,
            )
            if (
                rows <= 0
                or root_np.dtype != np.float32
                or joint_pos_np.dtype != np.float32
                or joint_vel_np.dtype != np.float32
                or root_np.shape[1:] != (13,)
                or joint_pos_np.shape
                != (root_np.shape[0], len(self._post_swing_capture_joint_names))
                or joint_vel_np.shape != joint_pos_np.shape
                or not np.isfinite(root_np).all()
                or not np.isfinite(joint_pos_np).all()
                or not np.isfinite(joint_vel_np).all()
            ):
                raise RuntimeError("natural-wrap source path produced an invalid runtime state batch")
            self._post_swing_capture_roots.append(np.array(root_np[:rows], copy=True))
            self._post_swing_capture_joint_pos.append(np.array(joint_pos_np[:rows], copy=True))
            self._post_swing_capture_joint_vel.append(np.array(joint_vel_np[:rows], copy=True))
            self._post_swing_capture_count += rows
            if self._post_swing_capture_count == self._post_swing_capture_target_count:
                self._publish_post_swing_capture()
        size = int(self.cfg.post_swing_buffer_size)
        if self._post_swing_root is None:
            self._post_swing_root = torch.zeros(size, 13, device=self.device)
            self._post_swing_joint_pos = torch.zeros(size, jp.shape[1], device=self.device)
            self._post_swing_joint_vel = torch.zeros(size, jv.shape[1], device=self.device)
        # ring write (n < size in practice; wrap the slot indices just in case)
        slots = (self._post_swing_ptr + torch.arange(n, device=self.device)) % size
        self._post_swing_root[slots] = root
        self._post_swing_joint_pos[slots] = jp
        self._post_swing_joint_vel[slots] = jv
        self._post_swing_ptr = int((self._post_swing_ptr + n) % size)
        self._post_swing_count = min(self._post_swing_count + n, size)

    def _load_post_swing_teacher_if_configured(self) -> None:
        """Seed the replay ring from one immutable natural-wrap teacher receipt."""

        receipt_path = str(
            getattr(self.cfg, "post_swing_teacher_receipt", "") or ""
        ).strip()
        receipt_sha = str(
            getattr(self.cfg, "post_swing_teacher_receipt_sha256", "") or ""
        ).strip().lower()
        authorization_path = str(
            getattr(self.cfg, "post_swing_teacher_retry_authorization", "") or ""
        ).strip()
        authorization_sha = str(
            getattr(
                self.cfg,
                "post_swing_teacher_retry_authorization_sha256",
                "",
            )
            or ""
        ).strip().lower()
        probability = float(self.cfg.post_swing_start_prob)
        configured_identity = tuple(
            bool(value)
            for value in (
                receipt_path,
                receipt_sha,
                authorization_path,
                authorization_sha,
            )
        )
        if any(configured_identity) and not all(configured_identity):
            raise ValueError(
                "post-swing teacher receipt and retry authorization paths/SHA-256 values "
                "must be provided together"
            )
        if (
            receipt_path
            or self._post_swing_require_ready_at_init
            or self._post_swing_fail_fast_first_reset
        ) and probability <= 0.0:
            raise ValueError(
                "post-swing teacher/activation gates require post_swing_start_prob > 0"
            )
        if (
            self._post_swing_require_ready_at_init
            or self._post_swing_fail_fast_first_reset
        ) and not receipt_path:
            raise ValueError(
                "ready-at-init, frozen teacher, and activation fail-fast modes require an "
                "immutable post_swing_teacher_receipt"
            )
        if not receipt_path:
            return

        motion_files = self.cfg.motion_file
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        else:
            motion_files = list(motion_files)
        try:
            joint_velocity_limits = self.robot.data.joint_vel_limits
            if joint_velocity_limits.ndim == 2:
                joint_velocity_limits = joint_velocity_limits[0]
            if joint_velocity_limits.ndim != 1:
                raise ValueError("runtime joint velocity limits have an unexpected shape")
            teacher = load_post_swing_teacher_states(
                receipt_path,
                receipt_sha,
                retry_authorization_path=authorization_path,
                expected_retry_authorization_sha256=authorization_sha,
                expected_motion_sha256=[sha256_file(path) for path in motion_files],
                expected_joint_names=self.robot.data.joint_names,
                expected_joint_velocity_limits=[
                    float(value) for value in joint_velocity_limits.detach().cpu().tolist()
                ],
                expected_root_linear_velocity_limit_mps=float(
                    self.cfg.post_swing_teacher_root_linear_velocity_limit_mps
                ),
                expected_root_angular_velocity_limit_radps=float(
                    self.cfg.post_swing_teacher_root_angular_velocity_limit_radps
                ),
                min_fill=int(self.cfg.post_swing_min_fill),
                buffer_size=int(self.cfg.post_swing_buffer_size),
            )
        except (OSError, PostSwingTeacherError) as exc:
            raise ValueError(f"invalid post-swing teacher receipt: {exc}") from exc

        joint_pos = torch.as_tensor(teacher.joint_pos, device=self.device)
        limits = self.robot.data.soft_joint_pos_limits
        if limits.ndim != 3 or limits.shape[-1] != 2:
            raise ValueError("runtime soft joint-position limits have an unexpected shape")
        lower = limits[0, :, 0].to(device=self.device)
        upper = limits[0, :, 1].to(device=self.device)
        if joint_pos.shape[1] != lower.numel() or torch.any(joint_pos < lower) or torch.any(
            joint_pos > upper
        ):
            raise ValueError(
                "post-swing teacher joint positions violate runtime articulation limits"
            )

        count = int(teacher.root_state_origin_relative.shape[0])
        size = int(self.cfg.post_swing_buffer_size)
        self._post_swing_root = torch.zeros(size, 13, device=self.device)
        self._post_swing_joint_pos = torch.zeros(
            size, joint_pos.shape[1], device=self.device
        )
        self._post_swing_joint_vel = torch.zeros_like(self._post_swing_joint_pos)
        self._post_swing_root[:count] = torch.as_tensor(
            teacher.root_state_origin_relative, device=self.device
        )
        self._post_swing_joint_pos[:count] = joint_pos
        self._post_swing_joint_vel[:count] = torch.as_tensor(
            teacher.joint_vel, device=self.device
        )
        self._post_swing_count = count
        self._post_swing_ptr = count % size
        self._post_swing_teacher_hard_contract = teacher.hard_contract
        if self._post_swing_count < int(self.cfg.post_swing_min_fill):
            # The pure loader already rejects this; retain a local invariant at the simulator
            # adoption boundary so a future loader refactor cannot weaken ready-at-init.
            raise ValueError("post-swing teacher did not make the replay buffer ready")

    def _configure_post_swing_capture_if_requested(self) -> None:
        """Atomically claim one inference-only natural-wrap capture namespace, default off."""

        output_dir = str(getattr(self.cfg, "post_swing_capture_output_dir", "") or "").strip()
        target_count = getattr(self.cfg, "post_swing_capture_target_count", 0)
        if not output_dir:
            if type(target_count) is not int or target_count != 0:
                raise ValueError(
                    "post_swing_capture_target_count requires post_swing_capture_output_dir"
                )
            return
        if self._post_swing_teacher_hard_contract is not None:
            raise ValueError("teacher consumption and teacher capture are mutually exclusive")
        if type(target_count) is not int or target_count <= 0:
            raise ValueError("post_swing_capture_target_count must be a positive integer")
        if bool(self.cfg.wrap_teleport):
            raise ValueError("natural-wrap teacher capture requires wrap_teleport=false")
        if float(self.cfg.post_swing_start_prob) <= 0.0:
            raise ValueError("natural-wrap teacher capture requires post_swing_start_prob > 0")
        motion_files = self.cfg.motion_file
        motion_files = [motion_files] if isinstance(motion_files, str) else list(motion_files)
        capture_dir = Path(output_dir)
        if capture_dir.is_symlink() or not capture_dir.is_dir():
            raise ValueError("natural-wrap capture output must be an existing regular directory")
        for name in (CAPTURE_STATE_NAME, CAPTURE_RESULT_NAME):
            if os.path.lexists(capture_dir / name):
                raise ValueError(
                    "natural-wrap capture output already exists; one-shot replay is forbidden"
                )
        joint_names = [str(value) for value in self.robot.data.joint_names]
        if (
            not joint_names
            or any(not value for value in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("capture joint names must be non-empty and unique")
        try:
            motion_clips = [
                {"index": index, "sha256": sha256_file(path)}
                for index, path in enumerate(motion_files)
            ]
            producer_source_sha256 = sha256_file(__file__)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            claim_fd = os.open(capture_dir / CAPTURE_CLAIM_NAME, flags, 0o600)
        except (OSError, PostSwingTeacherError) as exc:
            raise ValueError(f"cannot arm natural-wrap teacher capture: {exc}") from exc
        self._post_swing_capture_output_dir = capture_dir
        self._post_swing_capture_target_count = target_count
        self._post_swing_capture_motion_clips = motion_clips
        self._post_swing_capture_joint_names = joint_names
        self._post_swing_capture_producer_source_sha256 = producer_source_sha256
        self._post_swing_capture_claim_fd = claim_fd

    def post_swing_capture_complete(self) -> bool:
        """Return whether the one-shot source-owned result was durably published."""

        return self._post_swing_capture_complete

    def _bind_post_swing_capture_runtime_contract(self, sha256: str) -> None:
        if self._post_swing_capture_output_dir is None or self._post_swing_capture_claim_fd is None:
            raise RuntimeError("post-swing capture is not armed")
        if (
            self._post_swing_capture_count != 0
            or self._post_swing_capture_runtime_hard_contract_sha256 is not None
            or self._post_swing_capture_claim_sha256 is not None
        ):
            raise RuntimeError("capture runtime contract may be bound exactly once before stepping")
        if (
            type(sha256) is not str
            or len(sha256) != 64
            or any(value not in "0123456789abcdef" for value in sha256)
        ):
            raise RuntimeError("capture runtime hard-contract SHA-256 is invalid")
        claim = {
            "schema_version": 1,
            "artifact_kind": CAPTURE_CLAIM_KIND,
            "producer_source_sha256": self._post_swing_capture_producer_source_sha256,
            "runtime_hard_contract_sha256": sha256,
            "target_count": self._post_swing_capture_target_count,
            "motion_clips": list(self._post_swing_capture_motion_clips),
            "joint_names": list(self._post_swing_capture_joint_names),
            "exclusive_create": True,
        }
        raw = _canonical_json_bytes(claim)
        view = memoryview(raw)
        while view:
            written = os.write(self._post_swing_capture_claim_fd, view)
            if written <= 0:
                raise RuntimeError("cannot write the exclusive natural-wrap capture claim")
            view = view[written:]
        os.fsync(self._post_swing_capture_claim_fd)
        self._post_swing_capture_runtime_hard_contract_sha256 = sha256
        self._post_swing_capture_claim_sha256 = hashlib.sha256(raw).hexdigest()

    def _publish_post_swing_capture(self) -> None:
        """Publish accumulated live articulation snapshots; accepts no caller arrays."""

        if (
            self._post_swing_capture_output_dir is None
            or self._post_swing_capture_producer_source_sha256 is None
            or self._post_swing_capture_runtime_hard_contract_sha256 is None
            or self._post_swing_capture_claim_sha256 is None
            or self._post_swing_capture_claim_fd is None
            or self._post_swing_capture_count != self._post_swing_capture_target_count
        ):
            raise RuntimeError("natural-wrap capture publication invariants are not satisfied")
        root = np.concatenate(self._post_swing_capture_roots, axis=0)
        joint_pos = np.concatenate(self._post_swing_capture_joint_pos, axis=0)
        joint_vel = np.concatenate(self._post_swing_capture_joint_vel, axis=0)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            root_state_origin_relative=root,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        state_bytes = buffer.getvalue()
        _publish_bytes_no_clobber(
            self._post_swing_capture_output_dir / CAPTURE_STATE_NAME,
            state_bytes,
            "natural-wrap state payload",
        )
        result = {
            "schema_version": 2,
            "artifact_kind": CAPTURE_RESULT_KIND,
            "capture_contract": dict(CAPTURE_CONTRACT),
            "evidence": {
                "producer_source_sha256": self._post_swing_capture_producer_source_sha256,
                "runtime_hard_contract_sha256": (
                    self._post_swing_capture_runtime_hard_contract_sha256
                ),
                "exclusive_claim_sha256": self._post_swing_capture_claim_sha256,
                "exclusive_claim_relative_path": CAPTURE_CLAIM_NAME,
                "no_clobber": True,
            },
            "motion_clips": list(self._post_swing_capture_motion_clips),
            "states": {
                "relative_path": CAPTURE_STATE_NAME,
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
                "count": self._post_swing_capture_count,
                "root_shape": list(root.shape),
                "joint_pos_shape": list(joint_pos.shape),
                "joint_vel_shape": list(joint_vel.shape),
                "joint_names": list(self._post_swing_capture_joint_names),
            },
        }
        _publish_bytes_no_clobber(
            self._post_swing_capture_output_dir / CAPTURE_RESULT_NAME,
            _canonical_json_bytes(result),
            "natural-wrap capture result",
        )
        os.close(self._post_swing_capture_claim_fd)
        self._post_swing_capture_claim_fd = None
        self._post_swing_capture_complete = True

    def post_swing_replay_hard_contract(self) -> dict:
        """Return exact cold-start semantics for checkpoint lineage binding."""

        return {
            "teacher_receipt": self._post_swing_teacher_hard_contract,
            "teacher_distribution": "immutable",
            "require_ready_at_init": self._post_swing_require_ready_at_init,
            "fail_fast_first_reset": self._post_swing_fail_fast_first_reset,
            "first_reset_acceptance": {
                "min_adopted_count": self._post_swing_first_reset_min_adopted_count,
                "min_adopted_fraction": self._post_swing_first_reset_min_adopted_fraction,
                "selection_probability_abs_tolerance": self._post_swing_first_reset_selection_tolerance,
                "require_state_readback": self._post_swing_first_reset_require_readback,
            },
        }

    def _write_post_swing_states(self, env_ids: torch.Tensor):
        """A8: initialize `env_ids` from random buffered end-of-swing states (origin re-based)."""
        picks = torch.randint(0, self._post_swing_count, (len(env_ids),), device=self.device)
        root = self._post_swing_root[picks].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self._post_swing_joint_pos[picks].clone()
        joint_vel = self._post_swing_joint_vel[picks].clone()
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if self._post_swing_first_reset_require_readback:
            try:
                observed = (
                    ("root", self.robot.data.root_state_w[env_ids], root),
                    ("joint position", self.robot.data.joint_pos[env_ids], joint_pos),
                    ("joint velocity", self.robot.data.joint_vel[env_ids], joint_vel),
                )
            except (AttributeError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "post-swing replay readback is unavailable on this runtime"
                ) from exc
            for label, actual, expected in observed:
                if actual.shape != expected.shape or not torch.allclose(
                    actual, expected, rtol=0.0, atol=1.0e-6
                ):
                    raise RuntimeError(f"post-swing replay {label} readback differs from write")

    def consume_post_swing_activation_counters(self) -> dict[str, torch.Tensor]:
        """Return one PPO update's replay-start counts and atomically reset the window.

        The training runner calls this once after each rollout/update.  Returning cloned device
        scalars keeps the completed window stable while the live counters are zeroed for the next
        update.  With ``post_swing_start_prob == 0`` every counter remains exactly zero and this
        instrumentation performs no sampling or simulator write.
        """

        snapshot = {
            name: value.detach().clone()
            for name, value in self._post_swing_activation_counters.items()
        }
        for value in self._post_swing_activation_counters.values():
            value.zero_()
        return snapshot

    def consume_training_activation_counters(self) -> dict[str, torch.Tensor]:
        """Snapshot and reset every integer activation counter for one PPO update.

        ``MotionOnPolicyRunner`` prefers this aggregate consumer over the legacy post-swing-only
        consumer.  Keeping the latter public preserves the original narrow API for diagnostic
        callers while this method guarantees that post-swing, V1 and V2 share one logger
        transaction and cannot be reset at different update boundaries.
        """

        snapshot = {
            name: value.detach().clone()
            for counters in (
                self._post_swing_activation_counters,
                self._reward_activation_counters,
            )
            for name, value in counters.items()
        }
        for counters in (
            self._post_swing_activation_counters,
            self._reward_activation_counters,
        ):
            for value in counters.values():
                value.zero_()
        return snapshot

    def record_v1_velocity_mimic_activation(
        self, eligible_sample_count: int | torch.Tensor, *, held_wrist_excluded: bool
    ) -> None:
        """Record real V1 linear-velocity imitation evaluations.

        The explicit config activation bit is written only by the V1 training override.  When it
        is disabled this method is a strict no-op.  The denominator is recorded before checking
        the resolved body list, so a miswired V1 run produces a positive denominator and a zero
        exclusion numerator instead of a false green.
        """

        if not bool(self.cfg.v1_free_wrist_vel_mimic_activation):
            return
        counters = self._reward_activation_counters
        counters["v1_velocity_mimic_eligible_sample_count"].add_(
            eligible_sample_count
        )
        if held_wrist_excluded:
            counters["v1_held_wrist_excluded_sample_count"].add_(
                eligible_sample_count
            )

    def record_v2_strike_window_scale_activation(
        self, strike_window: torch.Tensor, *, actual_window_scale: float
    ) -> None:
        """Record real V2 reward applications inside the wide strike window.

        The denominator counts wide-window samples reaching ``torch.where`` in the imitation
        reward path.  The numerator counts the same samples only when both the explicit V2
        activation contract and the actually applied reward parameter are exactly ``0.25``.
        Thus a missing/mismatched scale cannot pass by merely exposing a strike-window mask.
        """

        configured_scale = self.cfg.v2_motion_scale_in_window_activation
        if configured_scale is None:
            return
        eligible_sample_count = strike_window.to(dtype=torch.bool).sum(
            dtype=torch.long
        )
        counters = self._reward_activation_counters
        counters["v2_strike_window_eligible_imitation_sample_count"].add_(
            eligible_sample_count
        )
        if (
            float(configured_scale) == 0.25
            and float(actual_window_scale) == 0.25
        ):
            counters[
                "v2_quarter_scaled_strike_window_imitation_sample_count"
            ].add_(eligible_sample_count)

    def _action_ball_reset_motion_snapshot(
        self, env_ids: torch.Tensor
    ) -> dict:
        device = torch.device(self.device)
        if device.type == "cuda":
            rng_state = torch.cuda.get_rng_state(device)
        else:
            rng_state = torch.random.get_rng_state()
        metric_names = (
            "in_hold",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        )
        return {
            "clip_id": self.clip_id[env_ids].clone(),
            "time_steps": self.time_steps[env_ids].clone(),
            "time_steps_f": self.time_steps_f[env_ids].clone(),
            "speed_scale": self.speed_scale[env_ids].clone(),
            "hold_counter": self.hold_counter[env_ids].clone(),
            "metrics": {
                name: self.metrics[name].clone()
                for name in metric_names
                if name in self.metrics
            },
            "balanced_sampler": self.balanced_clip_sampler_state_dict(),
            "stagger_pending": (
                None
                if self._stagger_hold_pending is None
                else self._stagger_hold_pending[env_ids].clone()
            ),
            "active_task_refs": list(self._action_ball_active_task_refs),
            "diagnostic_pending_row_count": (
                getattr(
                    self,
                    "_action_ball_diagnostic_pending_row_count",
                    0,
                )
            ),
            "task_timing_active": self._action_ball_task_timing_active[
                env_ids
            ].clone(),
            "task_pending_elapsed_s": (
                self._action_ball_task_pending_elapsed_s[env_ids].clone()
            ),
            "task_age_s": self._action_ball_task_age_s[env_ids].clone(),
            "time_to_contact_s": self._action_ball_time_to_contact_s[
                env_ids
            ].clone(),
            "teacher_rate": self._action_ball_teacher_rate[env_ids].clone(),
            "scaled_t_hit_s": self._action_ball_scaled_t_hit_s[
                env_ids
            ].clone(),
            "scaled_t_cycle_s": self._action_ball_scaled_t_cycle_s[
                env_ids
            ].clone(),
            "pre_swing_wait_s": self._action_ball_pre_swing_wait_s[
                env_ids
            ].clone(),
            "rng_state": rng_state,
        }

    def _restore_action_ball_reset_motion_snapshot(
        self, env_ids: torch.Tensor, state: dict
    ) -> None:
        self.load_balanced_clip_sampler_state_dict(
            state["balanced_sampler"]
        )
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng_state"], device)
        else:
            torch.random.set_rng_state(state["rng_state"])
        self.clip_id[env_ids] = state["clip_id"]
        self.time_steps[env_ids] = state["time_steps"]
        self.time_steps_f[env_ids] = state["time_steps_f"]
        self.speed_scale[env_ids] = state["speed_scale"]
        self.hold_counter[env_ids] = state["hold_counter"]
        self._action_ball_active_task_refs = list(
            state["active_task_refs"]
        )
        self._action_ball_diagnostic_pending_row_count = state[
            "diagnostic_pending_row_count"
        ]
        self._action_ball_task_timing_active[env_ids] = state[
            "task_timing_active"
        ]
        self._action_ball_task_pending_elapsed_s[env_ids] = state[
            "task_pending_elapsed_s"
        ]
        self._action_ball_task_age_s[env_ids] = state["task_age_s"]
        self._action_ball_time_to_contact_s[env_ids] = state[
            "time_to_contact_s"
        ]
        self._action_ball_teacher_rate[env_ids] = state["teacher_rate"]
        self._action_ball_scaled_t_hit_s[env_ids] = state[
            "scaled_t_hit_s"
        ]
        self._action_ball_scaled_t_cycle_s[env_ids] = state[
            "scaled_t_cycle_s"
        ]
        self._action_ball_pre_swing_wait_s[env_ids] = state[
            "pre_swing_wait_s"
        ]
        for name, value in state["metrics"].items():
            self.metrics[name].copy_(value)
        if self._stagger_hold_pending is not None:
            if state["stagger_pending"] is None:
                raise RuntimeError(
                    "action-ball reset snapshot lost stagger state"
                )
            self._stagger_hold_pending[env_ids] = state[
                "stagger_pending"
            ]

    def _resample_command(self, env_ids: Sequence[int]):
        """Run one formal atomic or diagnostic fail-stop action-ball true reset."""

        if len(env_ids) == 0:
            return
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            # This guard is intentionally before adaptive sampling, broker
            # reserve, simulator setters and every Motion tensor write.  The
            # constructed top owner has separate initial-install and selected-
            # reset paths; generic CommandManager resample is authority for
            # neither one.
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="command resample"
            )
            raise RuntimeError(
                "legacy Motion command resample is tombstoned for the fresh full-MDP lane"
            )
        if self.action_ball_continuous_motion_enabled:
            # This guard must precede the reset body: broker, simulator and
            # Motion writes inside that body are already too late to discover
            # a retained production reveal epoch.
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="command resample"
            )
        if (
            self.action_ball_diagnostic_split_ready_teacher
            and self.action_ball_single_stroke_timeout_enabled
            and self._resampling_from_wrap
        ):
            raise RuntimeError(
                "measured non-looping N=1 stroke may not enter the wrap path"
            )
        if (
            self._action_ball_birth_broker is None
            or self._resampling_from_wrap
        ):
            result = self._resample_command_body(env_ids)
            if (
                self.action_ball_continuous_motion_enabled
                and not self._resampling_from_wrap
            ):
                self._reset_action_ball_continuous_motion_cadence(
                    torch.as_tensor(
                        env_ids,
                        dtype=torch.long,
                        device=self.device,
                    ).reshape(-1)
                )
            return result
        env_ids_t = (
            env_ids
            if torch.is_tensor(env_ids)
            else torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        )
        env_ids_t = env_ids_t.to(
            device=self.device, dtype=torch.long
        ).reshape(-1)
        if (
            self.action_ball_diagnostic_split_ready_teacher
            and self.action_ball_single_stroke_timeout_enabled
        ):
            self._action_ball_single_stroke_complete[env_ids_t] = False
        if self._action_ball_birth_broker.diagnostic_fast_path:
            # Diagnostic broker/provider/domain state is intentionally not
            # recoverable after a true-reset exception.  Let the one attempt
            # either publish normally or poison the whole run; a formal Motion
            # snapshot here is both unused and a dominant short-episode tax.
            result = self._resample_command_body(env_ids_t)
            if self.action_ball_continuous_motion_enabled:
                self._reset_action_ball_continuous_motion_cadence(
                    env_ids_t
                )
            return result
        snapshot = self._action_ball_reset_motion_snapshot(env_ids_t)
        try:
            result = self._resample_command_body(env_ids_t)
            if self.action_ball_continuous_motion_enabled:
                self._reset_action_ball_continuous_motion_cadence(
                    env_ids_t
                )
            return result
        except Exception:
            self._restore_action_ball_reset_motion_snapshot(
                env_ids_t, snapshot
            )
            raise

    def _resample_command_body(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # A true episode boundary starts the same immutable sequence from an unarmed ledger.  An
        # intra-episode wrap before the initial origin is not a sequence boundary and must not
        # rewrite scheduler time.  Once armed, T1 suppresses natural wraps entirely.
        if self._event_scheduler is not None and not self._resampling_from_wrap:
            self._event_scheduler.reset(env_ids)
        self._adaptive_sampling(env_ids)

        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)
        if (
            self._action_ball_birth_broker is not None
            and self._resampling_from_wrap
        ):
            # WRAP is a new swing against the same physical episode birth.  It freezes both the
            # action and root spawn, performs no broker transaction and writes no simulator root.
            self._advance_action_ball_wrap_generation(env_ids_t)

        if self.planner_revision_enabled:
            # A new ball starts from the action's explicit zero-rate entry frame.  Failure-adaptive
            # RSI may still choose the clip, but it may not start halfway through the swing: that
            # would erase preparation time and can place the phase past contact.  Clearing active
            # before any optional RSI simulator write also makes joint/body reference velocities
            # exactly zero until RacketTargetCommand installs the complete new task.
            if self._multiseg:
                starts = self.motion.seg_start[self.clip_id[env_ids_t]]
            else:
                starts = torch.zeros_like(env_ids_t)
            self.time_steps[env_ids_t] = starts
            self.time_steps_f[env_ids_t] = starts.float()
            self.speed_scale[env_ids_t] = 0.0
            self._planner_active[env_ids_t] = False
            # 孪生时钟回哨兵:新任务安装前窗保持关闭(fail-closed)。
            self._planner_truth_tts_signed[env_ids_t] = 1.0e6

        # A formal canonical-ready clip is self-contained: frame 0 is the shared waiting pose and
        # the following frames are its already-integrated path toward contact.  Always re-base a
        # newly selected clip on that frame; never serialize an extra ready->historical-frame-0
        # bridge and never release from adaptive RSI halfway through the stroke.
        if self.canonical_ready_mode:
            ready_steps = self._canonical_ready_steps(env_ids_t)
            self.time_steps[env_ids_t] = ready_steps
            self.time_steps_f[env_ids_t] = ready_steps.float()

        # Pre-swing HOLD (Phase A): freeze the reference at the swing's first frame for a random
        # number of control steps ("the ball is not reaching yet"). Applies to resets AND wraps.
        if self._action_ball_birth_broker is None:
            lo, hi = self._effective_hold_steps_range(
                self.start_pose_ramp_progress()
            )
            self.hold_counter[env_ids_t] = torch.randint(
                int(lo),
                int(hi) + 1,
                (len(env_ids_t),),
                device=self.device,
            )
        else:
            # The per-swing receipt is the sole wait owner.  Keep a one-step fail-closed ready
            # sentinel until Racket's public accessor exposes the exact task later in this reset
            # (or, for WRAP, later in this command-manager update).
            self.hold_counter[env_ids_t] = 1
        # A wrap can resample a new hold late inside _update_command. Publish its state now so
        # downstream rewards/terminations on this same control step do not see the old swing mask.
        self.metrics["in_hold"][env_ids_t] = (self.hold_counter[env_ids_t] > 0).float()

        # stagger (a): each env's FIRST true reset adds a uniform hold bias, spreading the swing/
        # strike phases of a same-instant reset cohort across ~one swing period. One-shot per env;
        # wraps and every later reset draw the plain hold range, so steady-state behavior is
        # unchanged. The stand/post-swing min-hold clamps below are min= clamps — the bias
        # survives them. Default OFF (see cfg.stagger_initial_clock): no RNG draw, byte-identical.
        if (
            self._action_ball_birth_broker is None
            and self._stagger_hold_pending is not None
            and not self._resampling_from_wrap
        ):
            _pend_ids = env_ids_t[self._stagger_hold_pending[env_ids_t]]
            if len(_pend_ids) > 0:
                _mx = int(self.cfg.stagger_hold_max_steps)
                if _mx > 0:
                    self.hold_counter[_pend_ids] += torch.randint(
                        0, _mx + 1, (len(_pend_ids),), device=self.device
                    )
                self._stagger_hold_pending[_pend_ids] = False

        # Intra-episode clip WRAP: no teleport (deploy case) — the policy must physically carry
        # the body from the previous swing's end into the new swing's windup. The imitation
        # targets are anchor-relative, so the new reference re-anchors to the robot where it is.
        # Teleporting at a wrap (legacy RSI behavior) requires wrap_teleport=True.
        if self._resampling_from_wrap and not self.cfg.wrap_teleport:
            if self._action_ball_birth_broker is not None:
                self._begin_action_ball_task_pending(
                    env_ids_t, elapsed_s=float(self._env.step_dt)
                )
            return

        if self.canonical_ready_mode:
            # Every true episode reset belongs to the formal ready-entry distribution.  RSI and
            # post-swing replay are rejected at boot rather than silently surviving as alternate
            # reset routes.  The selected clip is immaterial to pose because startup validation
            # proved every start/end shares the same literal runtime ready.
            if self._action_ball_birth_broker is None:
                self._write_canonical_ready_state(env_ids_t)
            else:
                transaction = self._reserve_action_ball_true_reset(env_ids_t)
                sim_rollback_state = None
                try:
                    sim_rollback_state = self._write_canonical_ready_state(
                        env_ids_t,
                        action_ball_base_spawn_w_m=transaction["spawn"],
                        action_ball_base_quat_wxyz=transaction["quat"],
                    )
                    self._commit_action_ball_true_reset(
                        env_ids_t, transaction
                    )
                    self.hold_counter[env_ids_t] = torch.clamp(
                        self.hold_counter[env_ids_t],
                        min=int(self.cfg.stand_start_min_hold),
                    )
                    self.metrics["in_hold"][env_ids_t] = (
                        self.hold_counter[env_ids_t] > 0
                    ).float()
                    self._begin_action_ball_task_pending(
                        env_ids_t, elapsed_s=0.0
                    )
                except Exception as exc:
                    if self._action_ball_birth_broker.diagnostic_fast_path:
                        # Diagnostic transactions carry no rollback fields.
                        # Any failure after reserve poisons the run and must
                        # escape unchanged; retrying could reuse an advanced
                        # provider/RNG tape under the same logical reset.
                        raise
                    # _write_canonical_ready_state already restores a failed setter.  A later
                    # commit failure needs the same physical rollback before rewinding the exact
                    # broker/provider/domain tape.
                    if sim_rollback_state is not None:
                        try:
                            self._restore_action_ball_sim_state(
                                env_ids_t, sim_rollback_state
                            )
                        except Exception as rollback_error:
                            # Still restore the broker tape before surfacing the simulator failure.
                            self._rollback_action_ball_true_reset(
                                env_ids_t,
                                transaction,
                                original_error=exc,
                            )
                            raise RuntimeError(
                                "action-ball commit failed and simulator rollback failed"
                            ) from rollback_error
                    self._rollback_action_ball_true_reset(
                        env_ids_t,
                        transaction,
                        original_error=exc,
                    )
                    raise
                return
            self.hold_counter[env_ids_t] = torch.clamp(
                self.hold_counter[env_ids_t], min=int(self.cfg.stand_start_min_hold)
            )
            self.metrics["in_hold"][env_ids_t] = (
                self.hold_counter[env_ids_t] > 0
            ).float()
            return

        # TRUE episode reset: three-way split — DEFAULT STAND (deploy entry) / POST-SWING buffer
        # (A8: the policy's own end-of-swing states) / legacy RSI teleport onto the (noised)
        # reference frame. One uniform draw per env: u < stand_p -> stand; stand_p <= u <
        # stand_p + post_p -> post-swing (only once the buffer has post_swing_min_fill entries);
        # else RSI.
        u = torch.rand(len(env_ids_t), device=self.device)
        stand_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)
        post_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)
        post_selected_count: torch.Tensor | None = None
        if not self._resampling_from_wrap:
            stand_p = float(self.cfg.stand_start_prob)
            post_p = float(self.cfg.post_swing_start_prob)
            if stand_p > 0.0:
                stand_mask = u < stand_p
            if post_p > 0.0:
                buffer_ready = self._post_swing_count >= int(self.cfg.post_swing_min_fill)
                if buffer_ready:
                    eligible_count = len(env_ids_t)
                    post_mask = (u >= stand_p) & (u < stand_p + post_p)
                    post_selected_count = post_mask.sum(dtype=torch.long)
                    counters = self._post_swing_activation_counters
                    counters["post_swing_replay_eligible_reset_count"].add_(eligible_count)
                    counters["post_swing_replay_selected_reset_count"].add_(
                        post_selected_count
                    )
                    counters["post_swing_replay_random_not_selected_reset_count"].add_(
                        eligible_count - post_selected_count
                    )
                else:
                    self._post_swing_activation_counters[
                        "post_swing_replay_buffer_not_ready_reset_count"
                    ].add_(len(env_ids_t))
        stand_ids = env_ids_t[stand_mask]
        post_ids = env_ids_t[post_mask]
        rsi_ids = env_ids_t[~(stand_mask | post_mask)]

        if len(stand_ids) > 0:
            default_root = self.robot.data.default_root_state[stand_ids].clone()
            default_root[:, :3] += self._env.scene.env_origins[stand_ids]
            default_root[:, 7:] = 0.0  # zero lin/ang velocity
            # Optional heading-recovery curriculum: deploy follow-throughs can enter the
            # recovery hold yawed, so square-only stand starts leave that state unseen.
            yaw = _stand_start_yaw_samples(
                self.cfg.stand_start_yaw_range, len(stand_ids), self.device
            )
            if yaw is not None:
                zero = torch.zeros_like(yaw)
                yaw_delta = quat_from_euler_xyz(zero, zero, yaw)
                default_root[:, 3:7] = quat_mul(yaw_delta, default_root[:, 3:7])
            self.robot.write_root_state_to_sim(default_root, env_ids=stand_ids)
            self.robot.write_joint_state_to_sim(
                self.robot.data.default_joint_pos[stand_ids],
                torch.zeros_like(self.robot.data.default_joint_vel[stand_ids]),
                env_ids=stand_ids,
            )
            # Give the stand-started envs time to travel stand -> windup before the clip runs.
            self.hold_counter[stand_ids] = torch.clamp(
                self.hold_counter[stand_ids], min=int(self.cfg.stand_start_min_hold)
            )

        if len(post_ids) > 0:
            if post_selected_count is None:
                raise RuntimeError(
                    "post-swing replay ids exist without an activation selected count"
                )
            self._write_post_swing_states(post_ids)
            # Count a replay as started only after both root and joint state writes return.  A
            # selected count without a started count therefore exposes a failed adoption path
            # instead of silently treating the random draw as a real replay start.
            self._post_swing_activation_counters[
                "post_swing_replay_started_reset_count"
            ].add_(post_selected_count)
            # Settle follow-through -> windup before the clip runs.
            self.hold_counter[post_ids] = torch.clamp(
                self.hold_counter[post_ids], min=int(self.cfg.post_swing_min_hold)
            )

        if self._post_swing_fail_fast_first_reset and not self._post_swing_first_reset_checked:
            # CommandManager invokes this true-reset path while constructing/resetting the
            # environment, before PPO can collect or optimize its first rollout.  Requiring one
            # successful adoption here catches a dead/endogenous cold start without burning a
            # +200 checkpoint.  The draw is still the configured Bernoulli draw; scientific
            # queues should use a large enough initial cohort that selected>0 is deterministic in
            # practice (4096 envs at p=0.25 in the registered pair).
            if self._post_swing_count < int(self.cfg.post_swing_min_fill):
                raise RuntimeError(
                    "post-swing first-reset fail-fast: teacher buffer is not ready"
                )
            selected = 0 if post_selected_count is None else int(post_selected_count.item())
            eligible = len(env_ids_t)
            selected_fraction = selected / eligible if eligible > 0 else 0.0
            if selected < self._post_swing_first_reset_min_adopted_count:
                raise RuntimeError(
                    "post-swing first-reset fail-fast: adopted count below the frozen minimum "
                    f"({selected} < {self._post_swing_first_reset_min_adopted_count})"
                )
            if selected_fraction < self._post_swing_first_reset_min_adopted_fraction:
                raise RuntimeError(
                    "post-swing first-reset fail-fast: adopted fraction below the frozen minimum "
                    f"({selected_fraction} < {self._post_swing_first_reset_min_adopted_fraction})"
                )
            if abs(selected_fraction - float(self.cfg.post_swing_start_prob)) > (
                self._post_swing_first_reset_selection_tolerance
            ):
                raise RuntimeError(
                    "post-swing first-reset fail-fast: selected fraction differs from the "
                    "configured Bernoulli probability beyond tolerance"
                )
            # Reaching here means _write_post_swing_states returned after both root and joint
            # state writes, and started was incremented from the same selected scalar.
            self._post_swing_first_reset_checked = True

        # stand/post-start clamps may have promoted an initially zero draw to a real hold.
        self.metrics["in_hold"][env_ids_t] = (self.hold_counter[env_ids_t] > 0).float()

        if len(rsi_ids) == 0:
            return
        env_ids = rsi_ids

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # R-c(ii) rsi_hold_root_stand_z: a HELD RSI birth (hold_counter>0, drawn above — ~100/101
        # of RSI births at hold_steps_range [0,100]) writes STAND joints (the joint_pos property's
        # hold gate) but the reference frame's CROUCH root z (~0.78 m; body_pos_w has NO hold
        # gate) — stand legs at crouch height put the feet ~0.29 m under the floor and PhysX
        # depenetration kicks the robot out at birth. Fix: give held-RSI births the DEFAULT-STAND
        # root height (default_root_state z, 1.0684 m on the A3 — read at runtime, never
        # hardcoded); xy + yaw stay the reference frame's. Velocities are already hold-zeroed by
        # the body_*_vel_w properties. Default False = byte-identical.
        if bool(getattr(self.cfg, "rsi_hold_root_stand_z", False)):
            held_rsi = env_ids[self.hold_counter[env_ids] > 0]
            if len(held_rsi) > 0:
                root_pos[held_rsi, 2] = (
                    self.robot.data.default_root_state[held_rsi, 2]
                    + self._env.scene.env_origins[held_rsi, 2]
                )

        # 起点扰动斜坡:ramp 关闭时 _effective_* 原样返回静态配置,采样调用和
        # RNG 消耗与过去逐字节相同;打开后同一批调用读的是插值后的范围。
        ramp_progress = self.start_pose_ramp_progress()
        range_list = self._effective_reset_range_list("pose_range", ramp_progress)
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = self._effective_reset_range_list(
            "velocity_range", ramp_progress
        )
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            *self._effective_joint_position_range(ramp_progress),
            joint_pos.shape,
            joint_pos.device,
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def install_external_exam_timing(
        self,
        env_ids: Sequence[int],
        clip_ids: torch.Tensor,
        hold_steps: torch.Tensor,
    ) -> None:
        """Install one evaluator-owned, immutable BankExam item per environment.

        This is deliberately a runtime seam rather than a config field: training still owns its
        normal random clip/hold sampler, while the formal evaluator may replace the *current*
        command only after it has independently validated an exam-split bank and schedule.  The
        method does not reset robot state; callers must first perform the documented nominal-stand
        reset and then refresh observations after installing both motion timing and racket targets.
        """

        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            self._require_action_ball_continuous_motion_leaf_idle(
                operation="external exam timing install"
            )
        raw_ids = torch.as_tensor(env_ids, device=self.device)
        raw_clips = torch.as_tensor(clip_ids, device=self.device)
        raw_holds = torch.as_tensor(hold_steps, device=self.device)
        for name, value in (("env_ids", raw_ids), ("clip_ids", raw_clips),
                            ("hold_steps", raw_holds)):
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"external exam {name} must use an integer dtype")
        ids = raw_ids.to(dtype=torch.long).reshape(-1)
        clips = raw_clips.to(dtype=torch.long).reshape(-1)
        holds = raw_holds.to(dtype=torch.long).reshape(-1)
        if len(ids) == 0 or len(ids) != len(clips) or len(ids) != len(holds):
            raise ValueError(
                "external exam timing requires equal, non-empty env/clip/hold vectors"
            )
        if len(torch.unique(ids)) != len(ids) or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("external exam env ids must be unique and in range")
        if torch.any(clips < 0) or torch.any(clips >= int(self.motion.num_segments)):
            raise ValueError("external exam clip ids are outside the loaded motion segments")
        if torch.any(holds < 0):
            raise ValueError("external exam hold steps must be non-negative")
        if bool(self.cfg.stagger_initial_clock) or float(self.cfg.clip_switch_prob) != 0.0:
            raise ValueError(
                "external BankExam requires stagger_initial_clock=false and clip_switch_prob=0"
            )
        if self._speed_per_clip is not None or tuple(float(v) for v in self.cfg.speed_scale_range) != (1.0, 1.0):
            raise ValueError(
                "external BankExam currently requires native one-frame-per-step playback"
            )

        self._require_canonical_ready_boundary(ids, "external BankExam install")
        self.clip_id[ids] = clips
        starts = self.motion.seg_start[clips]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        self.speed_scale[ids] = 1.0
        self.hold_counter[ids] = holds
        self.metrics["in_hold"][ids] = (holds > 0).float()
        self.just_resampled[ids] = False
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")
        if self._stagger_hold_pending is not None:
            self._stagger_hold_pending[ids] = False
        self._stagger_ep_pending = False

    def compute(self, dt: float):
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="manager compute"
        )
        if getattr(
            self,
            "_action_ball_continuous_fresh_motion_lane_bound",
            False,
        ):
            if not _tensor_matches_identity_version_receipt(
                getattr(self, "time_left", None),
                getattr(
                    self,
                    "_action_ball_continuous_fresh_time_left_receipt",
                    None,
                ),
            ):
                self.poison_global_reveal_epoch(
                    "fresh Motion inherited resample timer drifted"
                )
                raise RuntimeError(
                    "fresh Motion inherited resample timer drifted"
                )
            # Preserve IsaacLab's metric/timer/update order but omit the
            # inherited expiry scan entirely.  Genesis installed literal
            # infinity and no legacy resampler owns this lane; cadence/D05
            # remain the only Motion business writers.
            self._update_metrics()
            self.time_left -= dt
            receipt = _tensor_identity_version_receipt(self.time_left)
            if receipt is None:
                self.poison_global_reveal_epoch(
                    "fresh Motion inherited resample timer became unsealable"
                )
                raise RuntimeError(
                    "fresh Motion inherited resample timer became unsealable"
                )
            self._action_ball_continuous_fresh_time_left_receipt = receipt
            self._update_command()
            return
        if _compute_without_disabled_time_resampling_scan(self, dt):
            return
        super().compute(dt)

    def _resample(self, env_ids: Sequence[int]):
        # Observe config drift before IsaacLab may write a short timer.  The
        # poison is intentionally permanent: a later sentinel round-trip
        # cannot prove that every per-env ``time_left`` is sentinel-origin.
        _poison_disabled_time_resampling_fast_path_on_drift(self)
        _revoke_disabled_time_resampling_fast_path_on_timer_drift(self)
        super()._resample(env_ids)
        _arm_disabled_time_resampling_fast_path_after_resample(
            self, env_ids
        )

    def _update_command(self):
        self._require_action_ball_continuous_motion_leaf_idle(
            operation="command update"
        )
        # An exception, changed command order, or repeated direct call must not
        # leave a prior tick's empty-wrap proof consumable.
        self._split_ready_empty_wrap_receipt = None
        # stagger (b): ONE-SHOT at the first step after construction (fresh run OR resume — both
        # are the same-instant cohort the metric-sync forensics caught): advance every env's
        # episode clock by U[0, max_episode_length) so the first timeouts, and every episode
        # boundary after them, spread out instead of firing in one synchronized wave. Guarded on
        # the env exposing the clock (defensive: metrics must never crash training).
        if self._stagger_ep_pending:
            self._stagger_ep_pending = False
            _ep_buf = getattr(self._env, "episode_length_buf", None)
            _max_len = int(getattr(self._env, "max_episode_length", 0) or 0)
            if _ep_buf is not None and _max_len > 1:
                _ep_buf.add_(torch.randint(0, _max_len, (self.num_envs,), device=_ep_buf.device))
        # The first command update after a physical split-ready reset sees the
        # simulator's settled FK tensors.  Freeze them once for RESET_WAIT;
        # later policy motion cannot turn the mimic target into a moving copy
        # of the robot.
        if (
            self.action_ball_diagnostic_split_ready_teacher
            or getattr(
                self,
                "_action_ball_continuous_fresh_motion_lane_bound",
                False,
            )
        ):
            self._capture_action_ball_safe_ready_reference()

        # Pre-swing HOLD: action-ball owns a continuous receipt deadline (including a possible
        # fractional first motion tick); legacy paths retain their integer random hold counter.
        legacy_action_ball_active = self._action_ball_birth_broker is not None
        fresh_action_ball_active = (
            self.action_ball_continuous_motion_enabled
            and self._action_ball_continuous_fresh_motion_lane_bound
            and isinstance(
                self._action_ball_continuous_schedule_projection,
                MappingProxyType,
            )
            and self._action_ball_continuous_motion_device_r05_owner is not None
        )
        action_ball_active = (
            legacy_action_ball_active or fresh_action_ball_active
        )
        if self.action_ball_continuous_motion_enabled and not action_ball_active:
            raise RuntimeError(
                "continuous Motion cadence requires the action-ball birth authority"
            )
        if action_ball_active and self._event_scheduler is not None:
            # bind_action_ball_birth_broker requires canonical-ready mode, whose
            # boot contract rejects every non-disabled event timing mode.
            raise RuntimeError(
                "action-ball/event timing mutual exclusion drifted after binding"
            )
        if action_ball_active:
            if self.action_ball_continuous_motion_enabled:
                held, action_ball_cycle_due = (
                    self._advance_action_ball_continuous_motion_cadence()
                )
            else:
                held, action_ball_cycle_due = (
                    self._advance_action_ball_task_timing()
                )
        else:
            held = self.hold_counter > 0
            self.hold_counter = torch.clamp(
                self.hold_counter - 1, min=0
            )
            self.metrics["in_hold"] = held.float()
        if "clip_switch_count" not in self.metrics:
            self.metrics["clip_switch_count"] = torch.zeros(self.num_envs, device=self.device)
        if self.planner_revision_enabled:
            # The same-ball governor owns the sole reference clock.  Non-active rows can only
            # occur during construction/reset ordering and remain frozen until RacketTargetCommand
            # installs their first complete task tuple.
            frame_delta = self._advance_planner_phase(held)
            self.speed_scale = torch.where(
                self._planner_active, frame_delta, torch.zeros_like(frame_delta)
            )
            self.time_steps_f += self.speed_scale
            self.time_steps = self.time_steps_f.round().long()
            self.metrics["playback_speed"] = self.speed_scale.clone()
        elif action_ball_active:
            # _advance_action_ball_task_timing analytically installed the current receipt phase.
            pass
        elif self.retiming_active:
            # R14: fractional clock — advance s frames per unheld control step; the integer index is
            # derived by round(), mirroring the deploy clock's nearest-frame mapping (torch rounds
            # half-to-even vs C++ half-away-from-zero — differs only on exact .5 ties, measure-zero
            # for continuous speed ranges).
            self.time_steps_f += (~held).float() * self.speed_scale
            self.time_steps = self.time_steps_f.round().long()
            self.metrics["playback_speed"] = self.speed_scale.clone()
        else:
            self.time_steps += (~held).long()
        if not action_ball_active:
            event_owned = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if not action_ball_active and self._event_scheduler is not None:
            native = self._event_native_strike_ticks
            if native is None:
                if bool(self._event_scheduler.armed.any()):
                    raise RuntimeError(
                        "post_strike_t1 armed before RacketTargetCommand bound native strike timing"
                    )
                # Before the first exact-strike origin no row can reveal and absolute scheduler
                # time has no meaning.  RacketTargetCommand binds the real vector in the same
                # command-manager step that can accept the initial exact strike.
            else:
                event_step = self._event_scheduler.advance(native)
                self._install_event_motion(event_step)
            event_owned = self._event_scheduler.armed
            self.metrics["event_timing_armed"] = event_owned.float()
            self.metrics["event_question_installed"] = (
                self._event_scheduler.event_just_installed.float()
            )
            self.metrics["event_question_unavailable"] = (
                self._event_scheduler.event_just_unavailable.float()
            )
            self.metrics["event_question_infeasible"] = (
                self._event_scheduler.event_just_infeasible.float()
            )
            self.metrics["event_deadline_due"] = (
                self._event_scheduler.deadline_just_due.float()
            )
            self.metrics["event_opportunities_consumed"] = (
                self._event_scheduler.opportunities_consumed.float()
            )
        if action_ball_active:
            # Receipt timing is the sole ActionBall wrap owner.  The bind-time
            # event exclusion above makes clamp/event reductions both
            # semantically impossible and an avoidable host synchronization.
            if self.action_ball_continuous_motion_enabled:
                # The frozen cadence publishes a separate reveal handoff; a
                # suffix boundary is only a reference-phase change.  Never
                # route it through legacy just_resampled/_resample_command,
                # which would redraw a question immediately and erase the
                # recovery interval.
                env_ids = torch.empty(
                    0, dtype=torch.long, device=self.device
                )
                wrap_ids = env_ids
            elif (
                self.action_ball_diagnostic_split_ready_teacher
                and self.action_ball_single_stroke_timeout_enabled
            ):
                # A measured capture is one non-looping professional stroke.
                # Latch completion for the diagnostic-only timeout term and
                # hold the final teacher frame until the environment performs
                # a true reset; never reinterpret its moving end as ready.
                self._action_ball_single_stroke_complete |= (
                    action_ball_cycle_due
                )
                env_ids = torch.empty(
                    0, dtype=torch.long, device=self.device
                )
                wrap_ids = env_ids
            else:
                env_ids = torch.where(action_ball_cycle_due)[0]
                wrap_ids = env_ids
        elif self._multiseg:
            # Wrap at the END of the env's current clip/segment, not the global concatenated end.
            seg_end = self.motion.seg_start[self.clip_id] + self.motion.seg_len[self.clip_id]
            # Once an exact-strike origin arms T1, natural clip completion is a carry-state wait,
            # not permission to draw or teleport to another question.  Clamp the old reference at
            # its final native frame until the immutable reveal installs the next clip.
            clamp = event_owned & (self.time_steps >= seg_end)
            if bool(clamp.any()):
                self.time_steps[clamp] = seg_end[clamp] - 1
                self.time_steps_f[clamp] = self.time_steps[clamp].float()
            wrap_ids = torch.where(
                (~event_owned) & (self.time_steps >= seg_end)
            )[0]
            # DEPLOY-PARITY CLIP SWITCH (venue falls 2026-07-04): the runner's reference clock flips
            # clip_id whenever the planner re-sides the target — at an ARBITRARY mid-swing moment —
            # and the reference jumps to the new clip's first frame (pp_reference_clock.hpp clamps
            # tts-large to seg_start). Training previously only switched clips at clip END, so the
            # policy never saw that discontinuity and falls at 准备/正手/反手 switches on hardware.
            # With per-step prob clip_switch_prob an env aborts its swing operator-style and routes
            # through the SAME wrap-resample path (uniform new clip, frame 0, hold, fresh target).
            # NOTE: aborted swings count as uncompleted starts (slight completion-rate deflation).
            if (
                float(self.cfg.clip_switch_prob) > 0.0
            ):
                sw = torch.rand(self.num_envs, device=self.device) < float(self.cfg.clip_switch_prob)
                sw[wrap_ids] = False
                self.metrics["clip_switch_count"] = sw.float()
                switch_ids = torch.where(sw)[0]
                env_ids = torch.cat([wrap_ids, switch_ids]) if len(switch_ids) > 0 else wrap_ids
            else:
                env_ids = wrap_ids
        else:
            clamp = event_owned & (self.time_steps >= self.motion.time_step_total)
            if bool(clamp.any()):
                self.time_steps[clamp] = int(self.motion.time_step_total) - 1
                self.time_steps_f[clamp] = self.time_steps[clamp].float()
            env_ids = torch.where(
                (~event_owned)
                & (self.time_steps >= self.motion.time_step_total)
            )[0]
            wrap_ids = env_ids
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) > 0:
            self.just_resampled[env_ids] = True
            # A8: only envs that physically COMPLETED a swing (true wraps — passed the strike alive,
            # not teleported, not aborted-by-switch) feed the post-swing ring buffer.
            if self.cfg.post_swing_start_prob > 0.0 and len(wrap_ids) > 0:
                self._capture_post_swing_states(wrap_ids)
        # Wrap-path resample: skips the RSI teleport (cfg.wrap_teleport=False) so the policy
        # physically transitions swing -> swing. True resets go through reset()/manager instead.
        if getattr(
            self, "_action_ball_continuous_fresh_motion_lane_bound", False
        ):
            if len(env_ids) != 0:
                self._action_ball_continuous_motion_poisoned = True
                raise RuntimeError(
                    "fresh continuous Motion produced a legacy wrap batch"
                )
        else:
            self._resampling_from_wrap = True
            try:
                self._resample_command(env_ids)
            finally:
                self._resampling_from_wrap = False

        (
            next_body_quat_relative_w,
            next_body_pos_relative_w,
        ) = _motion_anchor_relative_body_transform(
            self.anchor_pos_w,
            self.anchor_quat_w,
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.body_pos_w,
            self.body_quat_w,
            expected_body_count=len(self.cfg.body_names),
        )
        writable_rows = getattr(
            self,
            "_action_ball_full_mdp_motion_epoch_writable_rows",
            None,
        )
        if (
            fresh_action_ball_active
            and torch.is_tensor(writable_rows)
            and writable_rows.dtype == torch.bool
            and tuple(writable_rows.shape) == (self.num_envs,)
            and writable_rows.device == torch.device(self.device)
        ):
            body_write = writable_rows[:, None, None]
            self.body_quat_relative_w = torch.where(
                body_write,
                next_body_quat_relative_w,
                self.body_quat_relative_w,
            )
            self.body_pos_relative_w = torch.where(
                body_write,
                next_body_pos_relative_w,
                self.body_pos_relative_w,
            )
        else:
            self.body_quat_relative_w = next_body_quat_relative_w
            self.body_pos_relative_w = next_body_pos_relative_w

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        if (
            action_ball_active
            and self.action_ball_diagnostic_split_ready_teacher
            and self.action_ball_single_stroke_timeout_enabled
        ):
            token = getattr(self._env, "common_step_counter", None)
            if type(token) is int:
                # Published last, after every current Motion writer of
                # ``just_resampled``.  The tensor identity closes replacement
                # drift; the host step token closes stale/order drift.
                self._split_ready_empty_wrap_receipt = (
                    token,
                    _tensor_identity_version_receipt(
                        self.just_resampled
                    ),
                )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    # Historical diagnostic replay only. Formal paths migrate these untagged finite-difference
    # link-origin velocities to the schema-2 COM-point contract instead of enabling this escape.
    allow_legacy_link_origin_velocity: bool = False
    # Formal canonical-library consumer.  Default OFF preserves every historical hold/reset,
    # reference value and RNG draw.  ON requires exact schema-2 clips whose starts and ends share
    # one literal runtime-float32 ready pose with zero endpoint velocities.  Every hold reference
    # (joint/body/anchor) and every true reset then comes from that same clip frame; RSI,
    # post-swing replay, reset noise, yaw perturbation and wrap teleport are rejected instead of
    # silently creating a second entry distribution.
    canonical_ready_mode: bool = False
    # Diagnostic-only measured N=1 bridge.  The robot true-resets from a separately validated
    # physical-ready state while the immutable measured clip frame 0 is held as the teacher.
    # This flag owns only that physical-ready -> measured-frame0 bridge; wrap/timeout ownership is
    # deliberately separate.  Formal/canonical-library runs must leave this false.
    action_ball_diagnostic_split_ready_teacher: bool = False
    # Internal route marker for the exact fresh full-MDP EnvCfg.  It is not
    # accepted from task YAML and is not admission: the live Motion constructor
    # re-reads the pinned table and snapshots the exact active NPZ payload before
    # handing them to the existing MotionLoader.
    action_ball_full_mdp_diagnostic_catalog: str | None = None
    # Diagnostic single-stroke lifecycle switch.  Exact true keeps the historical measured-N1
    # behavior: cycle completion latches one terminal mask and forbids natural wrap.  Missing or
    # false never latches that mask and lets the existing wrap path run; it does not by itself
    # provide recovery waits, distinct next questions, or a continuous-task contract.
    action_ball_single_stroke_timeout_enabled: bool = False
    # Fresh successor Motion bridge. ``None`` is the exact legacy/single-shot path. A non-null
    # mapping is only a C01/C02-bound projection of the episode-tick cadence and completed-action
    # frame-0 zero-velocity reference; its self-hash is not a schedule authority. Target, ball,
    # receipt admission, outcome and Reward remain separate owners and must acknowledge the same
    # reveal before Motion may start a shot.
    action_ball_continuous_motion_cadence: dict | None = None
    # Optional train.py-materialized action-specific reset/hold binding.  ``None`` is the literal
    # legacy path.  The runtime mapping is validated after immutable motion bytes are loaded, then
    # its normalized actor action and hold q_des are installed atomically with every ActionBall
    # true reset so physical spawn, last-action observation and controller state begin coherently.
    action_ball_dynamic_ready: dict | None = None
    # Formal mode has no raw-file escape hatch: one exact registry must authorize and atomically
    # bind the ordered five motion paths, family/phase/face tables, shared ready, and artifact
    # hashes.  All strings remain inert while canonical_ready_mode is false.
    canonical_registry_path: str = ""
    canonical_registry_repo_root: str = ""
    canonical_registry_sha256: str = ""
    canonical_registry_alignment_sha256: str = ""
    canonical_ready_sha256: str = ""
    canonical_ready_fk_sha256: str = ""
    # Path selection is configurable, authority is not: exact certificate bytes
    # must hash to a digest in canonical_motion_admission.py's code-owned set.
    canonical_promotion_certificate_path: str = ""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    # Execution-ledger activation bits.  ``train.py`` writes these only when the corresponding
    # V1/V2 override is explicitly present.  Defaults keep both ledgers inert; they do not change
    # reward values, simulator state, or random-number consumption.
    v1_free_wrist_vel_mimic_activation: bool = False
    v2_motion_scale_in_window_activation: float | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # 起点扰动斜坡(2026-08-05)。``None`` = 逐字节的旧路径:上面三个静态范围就是
    # 全部的复位噪声。装上一份规范化 payload 后,每次真复位读
    # ``min(1, common_step_counter / ramp_steps)`` 把静态种子线性插值到声明的终点,
    # 于是"第 0 步和过去完全一样、之后按 ramp 张开"是一条可审计的法则而不是手改配置。
    # payload 由 ``training_contract.validate_action_ball_start_pose_ramp`` 生成/校验,
    # train.py 与本类读的是同一份代码,谁都不能自己另写一套插值。
    start_pose_ramp: dict | None = None

    # --- Phase A (2026-07-02): swing ENTRY / TRANSITION / WAITING coverage --------------------
    # Deploy enters every swing from a NOMINAL STAND, waits at the windup while the ball is not
    # yet reaching, and must physically transition between swings — none of which the pure-RSI
    # scheme ever produced (teleport at every episode start AND every clip wrap). These knobs
    # close that gap; the imitation targets are anchor-RELATIVE (re-anchored to the robot's
    # current xy+yaw every step), so no-teleport starts/wraps are well-posed.
    # Fraction of TRUE episode resets that start from the robot's DEFAULT STAND pose (zero
    # velocities) instead of teleporting onto the reference clip frame (RSI).
    stand_start_prob: float = 0.25
    # Teleport the robot onto the new clip's start frame at intra-episode wraps (legacy RSI
    # behavior). False = the policy must physically transition swing->swing (the deploy case).
    wrap_teleport: bool = False
    # Pre-swing HOLD: on every swing (re)start, freeze the reference at the clip's first frame
    # for U[lo,hi] control steps (50 Hz). While held, time_to_strike sits at its per-clip
    # maximum — exactly the deploy runner's clamped "waiting for the ball" pairing.
    hold_steps_range: tuple[int, int] = (0, 100)
    # Stand-started envs get at least this much hold (they must travel stand -> windup first).
    stand_start_min_hold: int = 25
    # Uniform world-yaw perturbation (rad) for stand starts. Pair a nonzero range with a
    # hold-only heading-recovery objective; (0, 0) preserves the legacy square start.
    stand_start_yaw_range: tuple[float, float] = (0.0, 0.0)
    # --- A8 (Ace recipe): post-swing initial-state distribution ------------------------------
    # Fraction of TRUE episode resets initialized from a ring buffer of the policy's OWN
    # end-of-swing states (captured at every intra-episode clip wrap — envs that physically
    # completed a swing). Teaches "start the next swing from wherever the last one left you"
    # even for single-swing episodes. Drawn AFTER stand_start_prob from the remaining resets;
    # falls back to RSI while the buffer has fewer than post_swing_min_fill entries.
    post_swing_start_prob: float = 0.0
    post_swing_buffer_size: int = 4096
    post_swing_min_fill: int = 256
    # Post-swing-started envs get at least this much hold (settle follow-through -> windup).
    post_swing_min_hold: int = 25
    # Optional exogenous cold start.  The receipt contains only states captured at natural clip
    # wraps and binds teacher checkpoint/source/contract, exact motion bytes and runtime joint
    # order.  Empty/default preserves the historical policy-owned live buffer exactly.
    post_swing_teacher_receipt: str = ""
    post_swing_teacher_receipt_sha256: str = ""
    post_swing_teacher_retry_authorization: str = ""
    post_swing_teacher_retry_authorization_sha256: str = ""
    # Explicit runtime limits accepted by the attestor and rechecked before simulator adoption.
    # A floating base has no actuator limit in PhysX, so the capture contract must pin both norms.
    post_swing_teacher_root_linear_velocity_limit_mps: float = 0.0
    post_swing_teacher_root_angular_velocity_limit_radps: float = 0.0
    # Explicit scientific pairs can refuse endogenous cold starts at process startup and require
    # the initial true-reset cohort to adopt at least one teacher state before the first policy
    # rollout/update.  Both default off so existing checkpoints/queues keep exact behavior.
    post_swing_require_ready_at_init: bool = False
    post_swing_fail_fast_first_reset: bool = False
    post_swing_first_reset_min_adopted_count: int = 1
    post_swing_first_reset_min_adopted_fraction: float = 0.0
    post_swing_first_reset_selection_tolerance: float = 1.0
    post_swing_first_reset_require_readback: bool = False
    # Inference-only producer seam.  It emits a raw natural-wrap callback result; it cannot mint
    # a teacher receipt or attest a checkpoint.  Defaults preserve historical training exactly.
    post_swing_capture_output_dir: str = ""
    post_swing_capture_target_count: int = 0
    # Per-step per-env probability of an operator-style mid-swing clip switch (deploy parity —
    # see the venue-falls note in _update_command). 0.002 ~ one switch per ~3-4 swings. Default off.
    clip_switch_prob: float = 0.0
    # Deterministic exactly balanced multi-clip allocation. OFF keeps the historical
    # torch.randint call and global-RNG consumption byte-identical. ON cycles through one
    # locally seeded permutation, so across any prefix (and across differently sized resample
    # calls) every clip's cumulative assignment count differs by at most one. The cursor,
    # permutation and resolved clip order are exposed by MotionCommand's explicit state hooks.
    balanced_clip_sampling: bool = False
    balanced_clip_sampling_seed: int = 0
    # T1 post-strike event timing.  Disabled is the byte-identical current scheduler.  The enabled
    # path requires a materialized immutable schedule whose exact UTF-8 JSON bytes match the
    # configured SHA-256; rows are assigned deterministically by env id and never repeat inside an
    # episode.  It is intentionally incompatible with random clip switching, stagger, retiming,
    # wrap teleport, and RSI frame skipping.
    event_timing_mode: str = EVENT_TIMING_MODE_DISABLED
    event_timing_schedule: str = ""
    event_timing_schedule_sha256: str = ""
    event_timing_repeat: bool = False
    # P2.4/R14 retiming: per-swing reference playback speed, uniform-sampled from this range at
    # every swing entry (wrap, mid-swing clip switch, and true reset). At speed s the clip clock
    # advances s frames per control step, reference velocities read ×s, time_to_strike runs ÷s,
    # and the racket velocity target scales ×s (hope_commands) — the (frame, tts, velocity)
    # pairing stays consistent, unlike the deploy runner's swing_speed knob which retimes the
    # clock but NOT the velocities (pp_policy.hpp). Default (1.0, 1.0) = OFF: the integer-clock
    # path below is byte-identical to before this flag existed.
    speed_scale_range: tuple[float, float] = (1.0, 1.0)
    # FIXED per-clip playback speed (2026-07-08 backhand-fix ablation): one entry per clip in
    # motion order, e.g. (1.0, 0.8) = forehand 1.0x, backhand 0.8x. Deterministic (no per-swing
    # randomness); overrides speed_scale_range when set. None = OFF (byte-identical default).
    # Question-bank targets are NOT rescaled (bank overrides target sampling downstream) — the
    # reference swing slows, the physical answer stays the answer.
    speed_scale_per_clip: tuple[float, ...] | None = None
    # 每 clip 的挥拍家族标签("forehand"/"backhand"),顺序 = motion_file 拼接后的 clip 顺序(同
    # strike_phase_per_clip / mount_normal_sign_per_clip)。用途:6-clip 变速烤入列表(正手
    # 0.8/1.0/1.2 + 反手 0.8/1.0/1.1)里,正手 1.0/1.2 变体不再被"clips==0 才是正手"的硬编码误判成
    # 反手(spdmix v2 可行性备忘 2026-07-22 硬绑定一:swing_sign、swing_type 观测、uniform 目标 y 侧
    # 全按这张表取)。None(默认)= 现役行为逐字节不变:内部按"单 clip 正手 / 恰好 2 clip = (正手,
    # 反手)"推导,>2 clip 缺表在查表时 fail-loud——那正是会悄悄训错的场景。显式给出时开机整表校验:
    # 长度必须 == clip 数、值只认这两个字符串、正反手至少各一个,错了当场报错(见
    # resolve_clip_family_is_forehand)。
    clip_family_per_clip: tuple[str, ...] | None = None

    # Same-ball task revision + phase-governor contract.  The disabled path allocates no buffers,
    # draws no RNG and preserves the historical reference clock.  When enabled the complete
    # profile is mandatory (no defaults/partial profiles); train.py installs the same mapping in
    # MotionCommand and RacketTargetCommand from one top-level task.planner_revision block.
    planner_revision_enabled: bool = False
    planner_revision_profile: dict | None = None
    planner_revision_initial_tts_range_s: tuple[float, float] = (0.5, 1.5)
    # Training-only weighted preparation-time distribution.  The complete document is bound in
    # planner_task_revision_training; deployment consumes only the enclosing runtime range above.
    planner_revision_initial_tts_mixture: dict | None = None

    # --- R-c RSI birth fixes (reward_staged_design 2026-07-08 §⑥; defaults OFF = byte-identical) --
    # (i) Skip the first N frames of every swing entry (RSI reset AND wrap — both go through
    # _adaptive_sampling): the v5 GMR clips carry a 3-4 frame IK cold-start transient at frame 0
    # (7.4-15.9 rad/s phantom joint velocities), so births teleported onto frame 0 inherit an
    # instant over-speed reference. N=6 (0.12 s @50 fps) is the design stopgap; once the GMR
    # warm-up source fix lands, N returns to 0 and this flag retires. 人话:出生别传送到 IK 瞬态
    # 帧上,参考从第 N 帧起播。
    rsi_skip_settle_frames: int = 0
    # (ii) Held-RSI births (hold_counter>0) write the DEFAULT-STAND root height instead of the
    # reference frame-0 crouch z: the hold gate already substitutes STAND joints, but the root
    # kept the crouch height (0.78 m vs stand 1.0684 m) -> feet ~0.29 m under the floor -> PhysX
    # depenetration kick at birth. This makes the birth state self-consistent; it is a
    # correctness fix, not an incentive change. 人话:站姿关节配站姿身高,脚不再穿地被弹飞。
    rsi_hold_root_stand_z: bool = False

    # --- 防同步 stagger_initial_clock (metric-sync fix 2026-07-09; default OFF = byte-identical) --
    # 4096 envs resumed at the same instant + low fall rate => synchronized mass timeouts
    # (episode_length sawtooth 52->485) => every EMA metric reads a queue oscillation. ON adds two
    # ONE-SHOT uniform biases (see MotionCommand.__init__ / _resample_command / _update_command):
    # (a) first true reset per env: hold += U[0, stagger_hold_max_steps] (swing phases spread);
    # (b) first step after construction: episode clock += U[0, max_episode_length) (episode
    # boundaries spread, permanently). 人话:把所有 env 的节拍随机错开,治 EMA 指标同步振荡;
    # 默认关=现役可比,新点火臂建议开。
    stagger_initial_clock: bool = False
    # (a) 的偏置上限(控制步): 默认 150 步 = 3 s @ 50 Hz ≈ 一个 hold+挥拍 周期。
    stagger_hold_max_steps: int = 150

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
