"""Cold action-timing owner for the full-MDP single-action lean diagnostic.

This module does not accept a manifest path, a timing mapping, or caller
numeric tensors.  It starts from the exact live ``RacketTargetCommand``,
follows its exact bound ``MotionCommand``, and admits only the ordered motion
bytes that Motion froze at construction when those bytes are ordered members
of the code-pinned ChingMu 0807-A3P diagnostic catalog.

The catalog is diagnostic provenance, not formal motion admission: its
teachers are not mechanically admitted and this owner cannot authorize a
launch, promotion, checkpoint, export, or deployment.  The runtime policy
step is the one-tick attempt-close margin; the actual environment horizon is
``max_episode_length * step_dt``.  Both are read from the exact live env, not
from the catalog or a caller echo.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import NoReturn

import torch

from whole_body_tracking.tasks.tracking.mdp import action_ball_manifest as _manifest
from whole_body_tracking.tasks.tracking.mdp import commands as _commands
from whole_body_tracking.tasks.tracking.mdp import hope_commands as _racket


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
FORMAL_ADMISSION = False

PINNED_DIAGNOSTIC_MANIFEST_RELATIVE_PATH = (
    "configs/action_ball_chingmu73_measured_a3p0807_f10_20260819.json"
)
PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256 = (
    "7176fa6448094eaa5dba9640c3e7c74fcd947f36208c434813820a5161dd24a4"
)
PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256 = (
    "f530165013baa570e0bf6bbbebcd7eef0c5c54db6ff7d51afccdc24e170f8cd5"
)
ATTEMPT_CLOSE_SEMANTICS = "one_exact_live_racket_policy_step"
EPISODE_HORIZON_SEMANTICS = "exact_live_max_episode_length_times_step_dt"
DIAGNOSTIC_POLICY_STEP_S = 0.02

_I64_MAX = (1 << 63) - 1

_REPO_ROOT = Path(__file__).resolve().parents[4]


class DiagnosticActionTimingError(RuntimeError):
    """The cold live source cannot form the exact diagnostic timing row."""


class DiagnosticActionTimingProductionHold(DiagnosticActionTimingError):
    """No formally admitted production action-timing source exists."""


@dataclass(frozen=True, eq=False, repr=False)
class DiagnosticActionTimingStaticTableProjection:
    """Cold all-action timing/profile rows for a recurring hot gather."""

    action_uid: torch.Tensor
    time_to_contact_ticks: torch.Tensor
    teacher_rate_min: torch.Tensor
    teacher_rate_max: torch.Tensor
    reference_t_hit_s: torch.Tensor
    reference_t_cycle_s: torch.Tensor
    reaction_margin_s: torch.Tensor
    mount_normal_sign: torch.Tensor
    manifest_file_sha256: str
    manifest_canonical_sha256: str
    diagnostic_unauthorized: bool = True
    runtime_integrated: bool = False
    launch_authorized: bool = False
    formal_admission: bool = False



def _load_pinned_catalog() -> _manifest.LoadedActionBallManifest:
    path = (_REPO_ROOT / PINNED_DIAGNOSTIC_MANIFEST_RELATIVE_PATH).resolve()
    try:
        path.relative_to(_REPO_ROOT)
        loaded = _manifest.load_action_ball_manifest(
            path,
            expected_sha256=PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256,
            verify_referenced_assets=False,
            require_formal_admission=False,
        )
    except Exception as exc:
        raise DiagnosticActionTimingError(
            "code-pinned diagnostic action catalog is absent or changed"
        ) from exc
    if loaded.canonical_sha256 != PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256:
        raise DiagnosticActionTimingError(
            "code-pinned diagnostic action catalog canonical content changed"
        )
    return loaded


def _exact_live_owners(
    racket_owner: object,
) -> tuple[_racket.RacketTargetCommand, _commands.MotionCommand, object]:
    if type(racket_owner) is not _racket.RacketTargetCommand:
        raise DiagnosticActionTimingError(
            "diagnostic action timing requires the exact RacketTargetCommand"
        )
    num_envs = getattr(racket_owner, "num_envs", None)
    if (
        getattr(racket_owner, "_action_ball_full_mdp_enabled", None) is not True
        or getattr(racket_owner, "_action_ball_enabled", None) is not False
        or type(num_envs) is not int
        or num_envs <= 0
    ):
        raise DiagnosticActionTimingError(
            "diagnostic action timing requires a fresh positive exact-N Racket owner"
        )
    # `_motion_term` is the exact field consumed by the Racket FK projector.
    # Taking that field directly (rather than a shadowable `_motion()` call)
    # makes it impossible for timing/catalog rows to come from Motion A while
    # FK/cadence rows come from Motion B.
    motion_owner = getattr(racket_owner, "_motion_term", None)
    if type(motion_owner) is not _commands.MotionCommand:
        raise DiagnosticActionTimingError(
            "diagnostic action timing requires one exact Racket-bound MotionCommand"
        )
    if (
        getattr(motion_owner, "_canonical_diagnostic_unauthorized", None)
        is not True
    ):
        raise DiagnosticActionTimingError(
            "diagnostic action timing requires the explicit unauthorized "
            "live Motion lane"
        )
    env = getattr(racket_owner, "_env", None)
    if (
        env is None
        or getattr(motion_owner, "_env", None) is not env
        or getattr(motion_owner, "num_envs", None) != num_envs
        or getattr(env, "num_envs", None) != num_envs
        or torch.device(getattr(racket_owner, "device", "cpu"))
        != torch.device(getattr(motion_owner, "device", "cpu"))
        or torch.device(getattr(env, "device", "cpu"))
        != torch.device(getattr(motion_owner, "device", "cpu"))
    ):
        raise DiagnosticActionTimingError(
            "diagnostic action timing Racket/Motion/env identity differs"
        )
    loader = getattr(motion_owner, "motion", None)
    if (
        type(loader) is not _commands.MotionLoader
        or getattr(loader, "kinematics_contract_exact", None) is not True
    ):
        raise DiagnosticActionTimingError(
            "diagnostic action timing requires exact schema-2 MotionLoader bytes"
        )
    return racket_owner, motion_owner, env


def _live_catalog_rows(
    racket_owner: _racket.RacketTargetCommand,
    motion_owner: _commands.MotionCommand,
    env: object,
) -> tuple[tuple[_manifest.ActionBallAction, ...], float, float]:
    loaded = _load_pinned_catalog()
    by_sha = {
        action.motion_sha256: (index, action)
        for index, action in enumerate(loaded.manifest.actions)
    }
    frozen_files = getattr(motion_owner, "_motion_files", None)
    frozen_sha256 = getattr(motion_owner, "_motion_file_sha256", None)
    adopted_payloads = getattr(motion_owner, "_motion_payloads", None)
    action_count = getattr(motion_owner.motion, "num_segments", None)
    if (
        type(frozen_files) is not tuple
        or type(frozen_sha256) is not tuple
        or type(action_count) is not int
        or action_count < 1
        or len(frozen_files) != action_count
        or len(frozen_sha256) != action_count
        or len(set(frozen_sha256)) != action_count
        or type(adopted_payloads) is not tuple
        or len(adopted_payloads) != action_count
    ):
        raise DiagnosticActionTimingError(
            "live Motion has no exact ordered construction-time byte identity"
        )
    # MotionLoader adopted ``_motion_payloads`` rather than re-opening the
    # paths, so bind those exact immutable construction snapshots first.  The
    # current path is re-read as a TOCTOU guard as well: a valid catalog digest
    # spliced over a different path, or a later path replacement, must not
    # pass catalog membership.
    for slot, (motion_file, frozen_sha, adopted_payload) in enumerate(
        zip(frozen_files, frozen_sha256, adopted_payloads)
    ):
        if (
            type(motion_file) is not str
            or type(frozen_sha) is not str
            or type(adopted_payload) is not bytes
            or hashlib.sha256(adopted_payload).hexdigest() != frozen_sha
        ):
            raise DiagnosticActionTimingError(
                "live MotionLoader adopted-byte identity differs"
            )
        try:
            resolved = Path(motion_file).resolve(strict=True)
            actual_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (OSError, RuntimeError) as exc:
            raise DiagnosticActionTimingError(
                f"live Motion bytes are unavailable at slot {slot}"
            ) from exc
        if actual_sha != frozen_sha:
            raise DiagnosticActionTimingError(
                f"live Motion bytes changed after construction at slot {slot}"
            )
    try:
        indexed = tuple(by_sha[value] for value in frozen_sha256)
    except (KeyError, TypeError) as exc:
        raise DiagnosticActionTimingError(
            "live Motion bytes are not members of the pinned diagnostic catalog"
        ) from exc
    indices = tuple(index for index, _action in indexed)
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise DiagnosticActionTimingError(
            "live Motion order differs from the pinned diagnostic catalog"
        )

    step_dt = getattr(env, "step_dt", None)
    max_episode_length = getattr(env, "max_episode_length", None)
    if (
        type(step_dt) is not float
        or not math.isfinite(step_dt)
        or step_dt <= 0.0
        or type(max_episode_length) is not int
        or max_episode_length <= 0
    ):
        raise DiagnosticActionTimingError(
            "live Racket policy step or episode horizon is zero/nonfinite"
        )
    episode_length_s = max_episode_length * step_dt
    if not math.isfinite(episode_length_s) or episode_length_s <= step_dt:
        raise DiagnosticActionTimingError(
            "live Racket episode horizon is zero/nonfinite"
        )

    seg_len = getattr(motion_owner.motion, "seg_len", None)
    if (
        type(seg_len) is not torch.Tensor
        or seg_len.dtype != torch.int64
        or tuple(seg_len.shape) != (action_count,)
    ):
        raise DiagnosticActionTimingError(
            "live Motion segment-length source differs"
        )
    # This is an explicitly cold construction boundary.  The host observation
    # validates immutable clip metadata once; the selected K-row gather below
    # remains on the live device and no hot callback imports this module.
    lengths = tuple(int(value) for value in seg_len.detach().cpu().tolist())
    try:
        phases_cfg = _racket.RacketTargetCommand._strike_phases_cfg(
            racket_owner, action_count
        )
        phases = (
            tuple(float(value) for value in phases_cfg)
            if phases_cfg
            else tuple(
                float(getattr(racket_owner.cfg, "strike_phase"))
                for _ in range(action_count)
            )
        )
    except Exception as exc:
        raise DiagnosticActionTimingError(
            "live Racket strike-frame source is absent or invalid"
        ) from exc
    if len(phases) != action_count:
        raise DiagnosticActionTimingError(
            "live Racket strike-frame table does not match live Motion order"
        )
    rows = tuple(action for _index, action in indexed)
    try:
        signs_cfg = tuple(racket_owner.cfg.mount_normal_sign_per_clip)
        if signs_cfg and len(signs_cfg) != action_count:
            raise ValueError("configured face-sign count differs")
        if any(float(value) not in (-1.0, 1.0) for value in signs_cfg):
            raise ValueError("configured face sign is not +/-1")
        if str(
            getattr(
                racket_owner.cfg,
                "motion_teacher_racket_source",
                "robot_fk",
            )
        ) == "measured_channel":
            measured_signs = tuple(
                int(value)
                for value in getattr(
                    motion_owner.motion,
                    "measured_racket_mount_normal_sign_per_clip",
                    (),
                )
            )
            if (
                len(measured_signs) != action_count
                or any(value not in (-1, 1) for value in measured_signs)
                or (
                    signs_cfg
                    and tuple(int(float(value)) for value in signs_cfg)
                    != measured_signs
                )
            ):
                raise ValueError("measured face-sign authority differs")
            live_signs = measured_signs
        else:
            live_signs = (
                tuple(int(float(value)) for value in signs_cfg)
                if signs_cfg
                else tuple(
                    int(getattr(racket_owner.cfg, "mount_normal_sign"))
                    for _ in range(action_count)
                )
            )
        if len(live_signs) != action_count or any(
            value not in (-1, 1) for value in live_signs
        ):
            raise ValueError("resolved face-sign authority differs")
    except Exception as exc:
        raise DiagnosticActionTimingError(
            "live Racket striking-face source is absent or invalid"
        ) from exc
    manifest_signs = tuple(int(action.mount_normal_sign) for action in rows)
    if live_signs != manifest_signs:
        raise DiagnosticActionTimingError(
            "live Racket striking-face table differs from the pinned "
            "diagnostic Motion order"
        )
    _strict_time_to_contact_ticks(rows, policy_step_s=step_dt)
    for slot, (action, length, phase) in enumerate(zip(rows, lengths, phases)):
        if type(length) is not int or length < 3:
            raise DiagnosticActionTimingError(
                f"live Motion segment length is invalid at slot {slot}"
            )
        live_t_cycle_s = (length - 1) * step_dt
        live_t_hit_s = round(float(phase) * (length - 1)) * step_dt
        if (
            not math.isclose(
                action.reference_t_cycle_s,
                live_t_cycle_s,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            or not math.isclose(
                action.reference_t_hit_s,
                live_t_hit_s,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise DiagnosticActionTimingError(
                "live Racket strike frame or Motion cycle differs from the "
                f"pinned diagnostic action at slot {slot}"
            )
    return rows, step_dt, episode_length_s


def _strict_time_to_contact_ticks(
    rows: tuple[_manifest.ActionBallAction, ...],
    *,
    policy_step_s: float,
) -> tuple[int, ...]:
    """Quantize the pinned incoming-ball arrival centre at a cold boundary."""

    ticks = []
    for slot, action in enumerate(rows):
        profile = action.ball_profile
        center_s = profile.time_to_contact_center_s
        minimum_s = profile.time_to_contact_min_s
        maximum_s = profile.time_to_contact_max_s
        tick_ratio = center_s / policy_step_s
        rounded_ticks = round(tick_ratio)
        if (
            not math.isfinite(center_s)
            or not math.isfinite(minimum_s)
            or not math.isfinite(maximum_s)
            or center_s < minimum_s
            or center_s > maximum_s
            or rounded_ticks < 1
            or rounded_ticks > _I64_MAX
            or not math.isclose(
                tick_ratio,
                float(rounded_ticks),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise DiagnosticActionTimingError(
                "pinned diagnostic time-to-contact center is outside its "
                f"profile or live policy tick grid at slot {slot}"
            )
        ticks.append(rounded_ticks)
    return tuple(ticks)


def _ceil_eps_ticks(duration_s: float, *, policy_step_s: float) -> int:
    """Match Motion's due-before-add clock at a cold scalar boundary."""

    if (
        type(duration_s) is not float
        or not math.isfinite(duration_s)
        or duration_s <= 0.0
        or type(policy_step_s) is not float
        or not math.isfinite(policy_step_s)
        or policy_step_s <= 0.0
    ):
        raise DiagnosticActionTimingError(
            "diagnostic task-close duration or policy step is invalid"
        )
    ticks = math.ceil(duration_s / policy_step_s - 1.0e-12)
    if ticks < 1 or ticks > _I64_MAX:
        raise DiagnosticActionTimingError(
            "diagnostic task-close tick count is outside int64"
        )
    return ticks


def diagnostic_catalog_max_task_close_ticks() -> int:
    """Return the pinned v4 worst-case reveal-to-task-close duration.

    This cold helper accepts no path, policy step, action slot, or caller
    timing.  The worst case uses each action's catalog arrival centre and
    minimum admitted teacher rate.  It is therefore a code-owned lower bound
    for the recurring diagnostic cadence, not a runtime task verdict.
    """

    rows = tuple(_load_pinned_catalog().manifest.actions)
    if not rows:
        raise DiagnosticActionTimingError(
            "pinned diagnostic action catalog is empty"
        )
    _strict_time_to_contact_ticks(
        rows, policy_step_s=DIAGNOSTIC_POLICY_STEP_S
    )
    close_ticks = []
    for slot, action in enumerate(rows):
        suffix_s = (
            action.reference_t_cycle_s - action.reference_t_hit_s
        ) / action.teacher_rate_min
        duration_s = action.ball_profile.time_to_contact_center_s + suffix_s
        if (
            not math.isfinite(suffix_s)
            or suffix_s <= 0.0
            or not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            raise DiagnosticActionTimingError(
                "pinned diagnostic task-close timing is invalid at slot "
                f"{slot}"
            )
        close_ticks.append(
            _ceil_eps_ticks(
                float(duration_s),
                policy_step_s=DIAGNOSTIC_POLICY_STEP_S,
            )
        )
    maximum = max(close_ticks)
    if maximum != 214:
        raise DiagnosticActionTimingError(
            "pinned diagnostic catalog task-close maximum differs"
        )
    return maximum


def construct_action_ball_full_mdp_diagnostic_action_timing_static_table(
    *,
    racket_owner: object,
) -> DiagnosticActionTimingStaticTableProjection:
    """Build the reusable cold catalog once; accepts no cadence or tensors."""

    racket_owner, motion_owner, env = _exact_live_owners(racket_owner)
    rows, policy_dt_s, _episode_length_s = _live_catalog_rows(
        racket_owner, motion_owner, env
    )
    device = torch.device(racket_owner.device)

    def column(name: str) -> torch.Tensor:
        return torch.tensor(
            [float(getattr(row, name)) for row in rows],
            dtype=torch.float32,
            device=device,
        ).contiguous()

    return DiagnosticActionTimingStaticTableProjection(
        action_uid=torch.tensor(
            [int(row.action_uid) for row in rows],
            dtype=torch.int64,
            device=device,
        ).contiguous(),
        time_to_contact_ticks=torch.tensor(
            _strict_time_to_contact_ticks(
                rows, policy_step_s=policy_dt_s
            ),
            dtype=torch.int64,
            device=device,
        ).contiguous(),
        teacher_rate_min=column("teacher_rate_min"),
        teacher_rate_max=column("teacher_rate_max"),
        reference_t_hit_s=column("reference_t_hit_s"),
        reference_t_cycle_s=column("reference_t_cycle_s"),
        reaction_margin_s=column("reaction_margin_s"),
        mount_normal_sign=column("mount_normal_sign"),
        manifest_file_sha256=PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256,
        manifest_canonical_sha256=PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256,
    )


def construct_production_action_timing_owner() -> NoReturn:
    """Keep production blocked on formally admitted motion/action identity."""

    raise DiagnosticActionTimingProductionHold(
        "production action timing lacks a formally admitted action catalog "
        "bound to the exact live Motion/Racket owner"
    )


__all__ = (
    "ATTEMPT_CLOSE_SEMANTICS",
    "DIAGNOSTIC_POLICY_STEP_S",
    "DIAGNOSTIC_UNAUTHORIZED",
    "DiagnosticActionTimingError",
    "DiagnosticActionTimingProductionHold",
    "DiagnosticActionTimingStaticTableProjection",
    "EPISODE_HORIZON_SEMANTICS",
    "FORMAL_ADMISSION",
    "LAUNCH_AUTHORIZED",
    "PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256",
    "PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256",
    "RUNTIME_INTEGRATED",
    "construct_action_ball_full_mdp_diagnostic_action_timing_static_table",
    "diagnostic_catalog_max_task_close_ticks",
    "construct_production_action_timing_owner",
)
