"""Strict A211/C211 observation ABI for the native MuJoCo lane.

This module mirrors the *ordered* 211-D actor and 319-D privileged-critic
contracts owned by the fresh Isaac A211/C211 consumers.  It deliberately does
not manufacture plant or measured-mimic observations.  A caller must provide
every named group and bind the three source authorities before a tensor can be
materialized; missing plant/mimic/task sources fail closed instead of being
represented by zero-filled columns.

The only zeroing performed here is the authorized RESET_WAIT semantic mask:
when ``task_valid == 0``, the already-present task 9-D tuple, base goal, and two
clocks are hidden.  Plant and measured-mimic groups remain untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTOR_WIDTH = 211
CRITIC_WIDTH = 319

# Exact mirror pin.  The A leaf currently ships question-source contract v5
# (online solver plus semantic question cache) and the C leaf direct-ball; the
# ordered 211/319 layouts did not change.  Every launcher reopens these exact
# bytes and fails closed before constructing an env if either source drifts.
#
# 人话:这四行只是"字节没变"的收据,不是"语义跟上了"的证明。SHA 谁都能重钉一行,
# 5ed998f1 就是这么让复刻停在原地两天没人发现的。所以重钉的同时必须过
# ``live_source_parity_blockers``:它把下面这份镜像逐符号跟活的 Isaac 叶子对一遍,
# 光重钉 SHA 不再能让布局漂移蒙混过关。
#
# A211 repinned 20260806 for 5c4ced66 (the validity-mask gate now compares the
# cfg sequence element by element instead of ``list == tuple``).  That edit is
# Isaac-cfg-only -- the native lane has no env_cfg to validate -- so nothing in
# the mirror needed to move with it; ``live_source_parity_blockers`` is what
# says so out loud instead of leaving it to the next reader's memory.
A211_SOURCE_SHA256 = "101a18c3f379b59c2a8c429a9a13b8b71e843d4da3a8ba0781f342203af77e93"
C211_SOURCE_SHA256 = "95652ae9c1e27e400eef6162f8bfeecd0be564619350c8491567c27f4eda15cb"
A211_TASK_LEAF_SHA256 = "0cf619caa7ee69650bd7e10cdcfc8de958fae9d3c3f6eccfec6d863da4032224"
C211_TASK_LEAF_SHA256 = "adf4574f262185fc0d71a7d186e9d8de40697b310c91e1157ad4b43f30c6c44c"

WAIT_MASK_CONTRACT_IDENTITY = "action_ball_211_wait_task_base_clocks_mask_v1"
PLANT_OBSERVATION_AUTHORITY_KIND = (
    "action_ball_211_mujoco_plant_observation_provider_v1"
)
MEASURED_MIMIC_AUTHORITY_KIND = (
    "action_ball_211_mujoco_full_body_measured_mimic_provider_v1"
)
TASK_QUESTION_AUTHORITY_KIND = "action_ball_211_fixed_question_provider_v1"


class ActionBall211ABIError(RuntimeError):
    """The ordered 211/319 ABI or one of its values is invalid."""


class ActionBall211AuthorityBlocked(ActionBall211ABIError):
    """Required plant, measured-mimic, or task authority is unavailable."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(str(item) for item in blockers)
        super().__init__(
            "A211/C211 observation construction blocked: " + ",".join(self.blockers)
        )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionBall211ABIError(f"{name} must be one lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ObservationField:
    """One ordered vector field and the reason it exists in the ABI."""

    name: str
    width: int
    purpose: str
    authority: str
    mask_when_task_invalid: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ActionBall211ABIError("observation field name must be non-empty")
        if type(self.width) is not int or self.width < 1:
            raise ActionBall211ABIError(f"field {self.name!r} width must be positive")
        if self.purpose not in {
            "plant_state",
            "achieved_outcome",
            "measured_mimic",
            "privileged_mimic",
            "plant_mimic_relation",
            "task_exogenous",
            "task_clock",
            "task_validity",
        }:
            raise ActionBall211ABIError(f"field {self.name!r} purpose differs")
        if self.authority not in {"plant", "mimic", "plant+mimic", "task"}:
            raise ActionBall211ABIError(f"field {self.name!r} authority differs")
        if self.mask_when_task_invalid and self.authority != "task":
            raise ActionBall211ABIError(
                f"non-task field {self.name!r} cannot use the RESET_WAIT task mask"
            )


@dataclass(frozen=True)
class ObservationLane:
    """One ordered actor or critic lane."""

    name: str
    fields: tuple[ObservationField, ...]
    expected_width: int

    def __post_init__(self) -> None:
        if self.name not in {"actor", "critic"}:
            raise ActionBall211ABIError("observation lane must be actor or critic")
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise ActionBall211ABIError(f"{self.name} field names must be unique")
        if self.width != self.expected_width:
            raise ActionBall211ABIError(
                f"{self.name} width {self.width} differs from {self.expected_width}"
            )
        if names[-1:] != ("task_valid",):
            raise ActionBall211ABIError(
                f"{self.name} task_valid must be the final field"
            )
        valid = self.fields[-1]
        if valid.width != 1 or valid.purpose != "task_validity":
            raise ActionBall211ABIError(f"{self.name} task_valid field differs")
        masked_width = sum(
            field.width for field in self.fields if field.mask_when_task_invalid
        )
        if masked_width != 13:
            raise ActionBall211ABIError(
                f"{self.name} RESET_WAIT task/base/clocks mask must be 13-D"
            )

    @property
    def width(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def layout(self) -> tuple[tuple[str, int], ...]:
        return tuple((field.name, field.width) for field in self.fields)

    @property
    def offsets(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for field in self.fields:
            result[field.name] = slice(start, start + field.width)
            start += field.width
        return result

    @property
    def task_mask_indices(self) -> tuple[int, ...]:
        offsets = self.offsets
        values = []
        for field in self.fields:
            if field.mask_when_task_invalid:
                span = offsets[field.name]
                values.extend(range(span.start, span.stop))
        return tuple(values)

    @property
    def task_valid_index(self) -> int:
        return self.offsets["task_valid"].start

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "lane": self.name,
                "expected_width": self.expected_width,
                "fields": [
                    {
                        "name": field.name,
                        "width": field.width,
                        "purpose": field.purpose,
                        "authority": field.authority,
                        "mask_when_task_invalid": field.mask_when_task_invalid,
                    }
                    for field in self.fields
                ],
                "wait_mask_contract": WAIT_MASK_CONTRACT_IDENTITY,
            }
        )


@dataclass(frozen=True)
class ActionBall211Profile:
    """A complete A211 or C211 actor/critic and normalizer identity."""

    label: str
    actor_contract: str
    critic_contract: str
    trainability_contract: str
    actor_normalizer_identity: str
    critic_normalizer_identity: str
    source_sha256: str
    task_leaf_sha256: str
    actor: ObservationLane
    critic: ObservationLane

    def __post_init__(self) -> None:
        if self.label not in {"A211", "C211"}:
            raise ActionBall211ABIError("profile label must be A211 or C211")
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.task_leaf_sha256, "task_leaf_sha256")
        for name in (
            "actor_contract",
            "critic_contract",
            "trainability_contract",
            "actor_normalizer_identity",
            "critic_normalizer_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ActionBall211ABIError(f"{name} must be non-empty")
        if self.actor.name != "actor" or self.critic.name != "critic":
            raise ActionBall211ABIError("profile actor/critic lanes are swapped")

    @property
    def wait_mask_contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "identity": WAIT_MASK_CONTRACT_IDENTITY,
                "actor_mask_indices": self.actor.task_mask_indices,
                "critic_mask_indices": self.critic.task_mask_indices,
                "actor_task_valid_index": self.actor.task_valid_index,
                "critic_task_valid_index": self.critic.task_valid_index,
                "invalid_value": 0,
                "active_value": 1,
            }
        )

    @property
    def observation_contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "label": self.label,
                "actor_contract": self.actor_contract,
                "critic_contract": self.critic_contract,
                "trainability_contract": self.trainability_contract,
                "actor_normalizer_identity": self.actor_normalizer_identity,
                "critic_normalizer_identity": self.critic_normalizer_identity,
                "source_sha256": self.source_sha256,
                "task_leaf_sha256": self.task_leaf_sha256,
                "required_authority_kinds": {
                    "plant": PLANT_OBSERVATION_AUTHORITY_KIND,
                    "mimic": MEASURED_MIMIC_AUTHORITY_KIND,
                    "task": TASK_QUESTION_AUTHORITY_KIND,
                },
                "actor_layout_sha256": self.actor.content_sha256,
                "critic_layout_sha256": self.critic.content_sha256,
                "wait_mask_contract_sha256": self.wait_mask_contract_sha256,
            }
        )

    def trainer_config_kwargs(self) -> dict[str, Any]:
        """Return the asymmetric dimensions and normalization-time mask ABI."""

        return {
            "observation_dim": self.actor.width,
            "critic_observation_dim": self.critic.width,
            "actor_normalizer_identity": self.actor_normalizer_identity,
            "critic_normalizer_identity": self.critic_normalizer_identity,
            "actor_task_mask_indices": self.actor.task_mask_indices,
            "critic_task_mask_indices": self.critic.task_mask_indices,
            "actor_task_valid_index": self.actor.task_valid_index,
            "critic_task_valid_index": self.critic.task_valid_index,
            "expected_profile_observation_contract_sha256": (
                self.observation_contract_sha256
            ),
        }


def _field(
    name: str,
    width: int,
    purpose: str,
    authority: str,
    *,
    wait_mask: bool = False,
) -> ObservationField:
    return ObservationField(name, width, purpose, authority, wait_mask)


def _actor_fields(task_names: tuple[str, str, str]) -> tuple[ObservationField, ...]:
    return (
        _field("actual_base_pose_lin_vel_world", 12, "plant_state", "plant"),
        _field("base_ang_vel_body", 3, "plant_state", "plant"),
        _field("joint_pos", 31, "plant_state", "plant"),
        _field("joint_vel", 31, "plant_state", "plant"),
        _field("actions", 31, "plant_state", "plant"),
        _field("racket_site_achieved_now_heading", 9, "achieved_outcome", "plant"),
        _field("teacher_joint_pos", 31, "measured_mimic", "mimic"),
        _field("teacher_joint_vel", 31, "measured_mimic", "mimic"),
        _field("racket_site_teacher_now_heading", 9, "measured_mimic", "mimic"),
        _field(
            "racket_site_teacher_at_reference_hit_heading",
            9,
            "measured_mimic",
            "mimic",
        ),
        *(
            _field(name, 3, "task_exogenous", "task", wait_mask=True)
            for name in task_names
        ),
        _field("desired_base_xy_world", 2, "task_exogenous", "task", wait_mask=True),
        _field("time_to_contact", 1, "task_clock", "task", wait_mask=True),
        _field("time_to_teacher_start", 1, "task_clock", "task", wait_mask=True),
        _field("task_valid", 1, "task_validity", "task"),
    )


def _critic_fields(task_names: tuple[str, str, str]) -> tuple[ObservationField, ...]:
    return (
        _field("command", 62, "privileged_mimic", "mimic"),
        _field("motion_anchor_pos_b", 3, "plant_mimic_relation", "plant+mimic"),
        _field("motion_anchor_ori_b", 6, "plant_mimic_relation", "plant+mimic"),
        _field("body_pos", 42, "plant_state", "plant"),
        _field("body_ori", 84, "plant_state", "plant"),
        _field("base_lin_vel", 3, "plant_state", "plant"),
        _field("base_ang_vel", 3, "plant_state", "plant"),
        _field("joint_pos", 31, "plant_state", "plant"),
        _field("joint_vel", 31, "plant_state", "plant"),
        _field("actions", 31, "plant_state", "plant"),
        _field(
            "racket_site_teacher_at_reference_hit_heading",
            9,
            "measured_mimic",
            "mimic",
        ),
        *(
            _field(name, 3, "task_exogenous", "task", wait_mask=True)
            for name in task_names
        ),
        _field("desired_base_xy_world", 2, "task_exogenous", "task", wait_mask=True),
        _field("time_to_contact", 1, "task_clock", "task", wait_mask=True),
        _field("time_to_teacher_start", 1, "task_clock", "task", wait_mask=True),
        _field("task_valid", 1, "task_validity", "task"),
    )


_A_TASK_NAMES = (
    "task_desired_contact_position_heading",
    "task_desired_contact_velocity_heading",
    "task_desired_contact_face_heading",
)
_C_TASK_NAMES = (
    "incoming_ball_contact_position_heading",
    "incoming_ball_contact_velocity_heading",
    "incoming_ball_contact_spin_heading",
)

A211_PROFILE = ActionBall211Profile(
    label="A211",
    actor_contract="action_ball_a211",
    critic_contract="action_ball_a211_critic_v1",
    trainability_contract="action_ball_a211_fixed_question_learnability_v2",
    actor_normalizer_identity="action_ball_a211_actor_norm_v2",
    critic_normalizer_identity="action_ball_a211_critic_norm_v1",
    source_sha256=A211_SOURCE_SHA256,
    task_leaf_sha256=A211_TASK_LEAF_SHA256,
    actor=ObservationLane("actor", _actor_fields(_A_TASK_NAMES), ACTOR_WIDTH),
    critic=ObservationLane("critic", _critic_fields(_A_TASK_NAMES), CRITIC_WIDTH),
)

C211_PROFILE = ActionBall211Profile(
    label="C211",
    actor_contract="action_ball_c211",
    critic_contract="action_ball_c211_critic_v1",
    trainability_contract="action_ball_c211_fixed_midpoint_learnability_v2",
    actor_normalizer_identity="action_ball_c211_actor_norm_v2",
    critic_normalizer_identity="action_ball_c211_critic_norm_v1",
    source_sha256=C211_SOURCE_SHA256,
    task_leaf_sha256=C211_TASK_LEAF_SHA256,
    actor=ObservationLane("actor", _actor_fields(_C_TASK_NAMES), ACTOR_WIDTH),
    critic=ObservationLane("critic", _critic_fields(_C_TASK_NAMES), CRITIC_WIDTH),
)

PROFILES = {"A211": A211_PROFILE, "C211": C211_PROFILE}


# --------------------------------------------------------------------------
# Live-source semantic parity
#
# 人话:上面每一行都是手抄的。SHA 只证明"源文件字节没动过",一旦源文件动了,
# 把 SHA 重钉成新值是一行的事,而手抄件是不是跟着动了没人查 —— 这正是
# 5ed998f1 那次复刻停在原地的机制。下面这组检查把手抄件逐符号跟活的 Isaac
# 叶子对一遍,所以"只重钉 SHA"从今往后不再够。
# --------------------------------------------------------------------------

#: The observation row after which every remaining non-``task_valid`` row is a
#: task-authority row, i.e. exactly the RESET_WAIT-masked block.  Derived from
#: the live layout rather than hardcoded so a new task row cannot be added
#: upstream and silently left unmasked here.
WAIT_MASK_TAIL_ANCHOR_FIELD = "racket_site_teacher_at_reference_hit_heading"

#: (profile attribute, live module symbol suffix) for every identity string the
#: mirror hand-copies out of the Isaac trainability leaf.
MIRRORED_IDENTITY_SYMBOLS = (
    ("actor_contract", "ACTOR_CONTRACT"),
    ("critic_contract", "CRITIC_CONTRACT"),
    ("trainability_contract", "TRAINABILITY_CONTRACT"),
    ("actor_normalizer_identity", "ACTOR_NORMALIZER_IDENTITY"),
    ("critic_normalizer_identity", "CRITIC_NORMALIZER_IDENTITY"),
)

_ABSENT = object()


def load_live_trainability_module(source_path: Any) -> Any:
    """Host-load one Isaac trainability leaf straight off disk.

    The A/C leaves import only ``math``/``typing`` (C falls back to loading its
    A sibling by path), so this works on a plain host without isaaclab.
    """

    path = Path(source_path)
    unique = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"action_ball_211_live_mirror_{path.stem}_{unique}", path
    )
    if spec is None or spec.loader is None:
        raise ActionBall211ABIError(f"cannot host-load live trainability leaf {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _live_layout(module: Any, symbol: str) -> Any:
    value = getattr(module, symbol, _ABSENT)
    if value is _ABSENT:
        return _ABSENT
    try:
        return tuple((str(name), int(width)) for name, width in value)
    except (TypeError, ValueError):
        return None


def live_source_parity_blockers(
    profile: ActionBall211Profile, source_path: Any
) -> tuple:
    """List every way this mirror disagrees with the live Isaac leaf.

    Empty means the hand-copied identities, the *ordered* per-row layouts, both
    widths and the RESET_WAIT mask block all still match what the leaf actually
    ships.  Order and per-row width are compared row by row on purpose: a swap
    of two same-width rows, or moving a dimension from one row to its
    neighbour, leaves the total width and the name set untouched and is
    invisible to every coarser check.
    """

    label = profile.label
    prefix = label.lower()
    try:
        module = load_live_trainability_module(source_path)
    except Exception as exc:  # noqa: BLE001 - any load failure must fail closed
        return (f"{prefix}_live_trainability_source_unloadable:{exc}",)

    blockers = []
    for attribute, suffix in MIRRORED_IDENTITY_SYMBOLS:
        symbol = f"{label}_{suffix}"
        live = getattr(module, symbol, _ABSENT)
        if live is _ABSENT:
            blockers.append(f"{prefix}_live_symbol_absent:{symbol}")
            continue
        mirrored = getattr(profile, attribute)
        if live != mirrored:
            blockers.append(
                f"{prefix}_{attribute}_differs:live={live!r} mirror={mirrored!r}"
            )

    for lane_name, lane, suffix in (
        ("actor", profile.actor, "ACTOR"),
        ("critic", profile.critic, "CRITIC"),
    ):
        symbol = f"{label}_{suffix}_LAYOUT"
        live_layout = _live_layout(module, symbol)
        if live_layout is _ABSENT:
            blockers.append(f"{prefix}_live_symbol_absent:{symbol}")
            continue
        if live_layout is None:
            blockers.append(f"{prefix}_live_symbol_malformed:{symbol}")
            continue
        if live_layout != lane.layout:
            blockers.append(
                f"{prefix}_{lane_name}_layout_differs:live={live_layout!r} "
                f"mirror={lane.layout!r}"
            )
        width_symbol = f"{label}_{suffix}_WIDTH"
        live_width = getattr(module, width_symbol, _ABSENT)
        if live_width is _ABSENT:
            blockers.append(f"{prefix}_live_symbol_absent:{width_symbol}")
        elif int(live_width) != lane.width:
            blockers.append(
                f"{prefix}_{lane_name}_width_differs:live={live_width!r} "
                f"mirror={lane.width}"
            )
        expected_mask = _wait_mask_tail_names(live_layout)
        if expected_mask is None:
            blockers.append(
                f"{prefix}_{lane_name}_live_layout_has_no_task_tail_block"
            )
            continue
        mirrored_mask = tuple(
            field.name for field in lane.fields if field.mask_when_task_invalid
        )
        if expected_mask != mirrored_mask:
            blockers.append(
                f"{prefix}_{lane_name}_wait_mask_differs:live_tail={expected_mask!r} "
                f"mirror_masked={mirrored_mask!r}"
            )
    return tuple(blockers)


def _wait_mask_tail_names(layout: Sequence) -> Any:
    """Rows strictly between the last mimic row and the trailing ``task_valid``."""

    names = [name for name, _width in layout]
    if names[-1:] != ["task_valid"] or WAIT_MASK_TAIL_ANCHOR_FIELD not in names:
        return None
    start = len(names) - 1 - names[::-1].index(WAIT_MASK_TAIL_ANCHOR_FIELD)
    return tuple(names[start + 1 : -1])


@dataclass(frozen=True)
class ObservationAuthorities:
    """Content identities for the three providers needed to build all columns."""

    plant_observation_sha256: str
    measured_mimic_sha256: str
    task_question_sha256: str
    plant_observation_kind: str = PLANT_OBSERVATION_AUTHORITY_KIND
    measured_mimic_kind: str = MEASURED_MIMIC_AUTHORITY_KIND
    task_question_kind: str = TASK_QUESTION_AUTHORITY_KIND

    def __post_init__(self) -> None:
        _sha256(self.plant_observation_sha256, "plant_observation_sha256")
        _sha256(self.measured_mimic_sha256, "measured_mimic_sha256")
        _sha256(self.task_question_sha256, "task_question_sha256")
        expected = {
            "plant_observation_kind": PLANT_OBSERVATION_AUTHORITY_KIND,
            "measured_mimic_kind": MEASURED_MIMIC_AUTHORITY_KIND,
            "task_question_kind": TASK_QUESTION_AUTHORITY_KIND,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ActionBall211ABIError(
                    f"{name} differs from the provider contract"
                )

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "plant_observation_sha256": self.plant_observation_sha256,
                "measured_mimic_sha256": self.measured_mimic_sha256,
                "task_question_sha256": self.task_question_sha256,
                "plant_observation_kind": self.plant_observation_kind,
                "measured_mimic_kind": self.measured_mimic_kind,
                "task_question_kind": self.task_question_kind,
            }
        )


def runtime_authority_blockers(
    profile: ActionBall211Profile,
    *,
    plant_observation_sha256: str | None,
    measured_mimic_sha256: str | None,
    task_question_sha256: str | None,
) -> tuple[str, ...]:
    """List absent/invalid source identities without fabricating observations."""

    values = (
        (
            "plant_observation_authority_unavailable",
            plant_observation_sha256,
            "plant_observation_sha256",
        ),
        (
            "full_body_measured_mimic_authority_unavailable",
            measured_mimic_sha256,
            "measured_mimic_sha256",
        ),
        (
            "task_question_authority_unavailable",
            task_question_sha256,
            "task_question_sha256",
        ),
    )
    blockers = []
    for suffix, value, name in values:
        if value is None:
            blockers.append(f"{profile.label.lower()}_{suffix}")
            continue
        try:
            _sha256(value, name)
        except ActionBall211ABIError:
            blockers.append(f"{profile.label.lower()}_{suffix}")
    return tuple(blockers)


def require_runtime_authorities(
    profile: ActionBall211Profile,
    *,
    plant_observation_sha256: str | None,
    measured_mimic_sha256: str | None,
    task_question_sha256: str | None,
) -> ObservationAuthorities:
    blockers = runtime_authority_blockers(
        profile,
        plant_observation_sha256=plant_observation_sha256,
        measured_mimic_sha256=measured_mimic_sha256,
        task_question_sha256=task_question_sha256,
    )
    if blockers:
        raise ActionBall211AuthorityBlocked(blockers)
    return ObservationAuthorities(
        plant_observation_sha256=plant_observation_sha256,
        measured_mimic_sha256=measured_mimic_sha256,
        task_question_sha256=task_question_sha256,
    )


def _task_valid_rows(value: Any, batch_size: int) -> np.ndarray:
    rows = np.asarray(value)
    if rows.shape == (batch_size, 1):
        rows = rows[:, 0]
    if rows.shape != (batch_size,):
        raise ActionBall211ABIError(
            f"task_valid sideband must have shape [{batch_size}] or [{batch_size},1]"
        )
    if rows.dtype == np.bool_:
        return rows.copy()
    try:
        numeric = rows.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ActionBall211ABIError(
            "task_valid sideband must contain only 0/1"
        ) from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ActionBall211ABIError("task_valid sideband must contain only 0/1")
    return numeric == 1.0


def flatten_lane_groups(
    lane: ObservationLane,
    groups: Mapping[str, Any],
    *,
    task_valid: Any,
) -> np.ndarray:
    """Flatten one exact ordered lane and apply its authorized WAIT mask.

    Mapping insertion order is part of the ABI.  Rejecting a different order
    prevents same-width historical or reordered rows from being consumed.
    """

    if not isinstance(groups, Mapping):
        raise ActionBall211ABIError(f"{lane.name} groups must be a mapping")
    expected_names = tuple(field.name for field in lane.fields)
    actual_names = tuple(groups.keys())
    if actual_names != expected_names:
        raise ActionBall211ABIError(
            f"{lane.name} ordered fields differ: expected={expected_names!r} "
            f"actual={actual_names!r}"
        )
    arrays = []
    batch_size = None
    for field in lane.fields:
        try:
            array = np.asarray(groups[field.name], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ActionBall211ABIError(
                f"{lane.name}.{field.name} cannot be represented as float32"
            ) from exc
        if array.ndim != 2 or array.shape[1] != field.width:
            raise ActionBall211ABIError(
                f"{lane.name}.{field.name} must have shape [N,{field.width}]"
            )
        if batch_size is None:
            batch_size = int(array.shape[0])
        elif array.shape[0] != batch_size:
            raise ActionBall211ABIError(f"{lane.name} group batch sizes differ")
        if not np.isfinite(array).all():
            raise ActionBall211ABIError(
                f"{lane.name}.{field.name} contains non-finite values"
            )
        arrays.append(array)
    if batch_size is None or batch_size < 1:
        raise ActionBall211ABIError(f"{lane.name} batch must be non-empty")
    validity = _task_valid_rows(task_valid, batch_size)
    task_valid_group = arrays[-1][:, 0]
    if not np.array_equal(task_valid_group, validity.astype(np.float32)):
        raise ActionBall211ABIError(
            f"{lane.name}.task_valid differs from the atomic sideband"
        )
    result = np.concatenate(arrays, axis=1)
    if result.shape != (batch_size, lane.expected_width):
        raise ActionBall211ABIError(f"{lane.name} flattened width differs")
    invalid_rows = np.logical_not(validity)
    if invalid_rows.any():
        result[np.ix_(invalid_rows, lane.task_mask_indices)] = 0.0
    if not np.isfinite(result).all():
        raise ActionBall211ABIError(f"{lane.name} flattened tensor is non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def flatten_profile_groups(
    profile: ActionBall211Profile,
    *,
    actor_groups: Mapping[str, Any],
    critic_groups: Mapping[str, Any],
    task_valid: Any,
    authorities: ObservationAuthorities,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize both lanes only after all source authorities are bound."""

    if not isinstance(authorities, ObservationAuthorities):
        raise ActionBall211AuthorityBlocked(
            runtime_authority_blockers(
                profile,
                plant_observation_sha256=None,
                measured_mimic_sha256=None,
                task_question_sha256=None,
            )
        )
    actor = flatten_lane_groups(profile.actor, actor_groups, task_valid=task_valid)
    critic = flatten_lane_groups(profile.critic, critic_groups, task_valid=task_valid)
    if actor.shape[0] != critic.shape[0]:
        raise ActionBall211ABIError("actor/critic batch sizes differ")
    if not np.array_equal(
        actor[:, profile.actor.task_valid_index],
        critic[:, profile.critic.task_valid_index],
    ):
        raise ActionBall211ABIError("actor/critic task_valid columns differ")
    return actor, critic


def construction_receipt(
    profile: ActionBall211Profile,
    *,
    num_envs: int,
    plant_observation_sha256: str | None = None,
    measured_mimic_sha256: str | None = None,
    task_question_sha256: str | None = None,
) -> dict[str, Any]:
    """Describe shape construction and the exact blockers before allocation."""

    if type(num_envs) is not int or num_envs < 1:
        raise ActionBall211ABIError("num_envs must be a positive plain integer")
    authority_blockers = runtime_authority_blockers(
        profile,
        plant_observation_sha256=plant_observation_sha256,
        measured_mimic_sha256=measured_mimic_sha256,
        task_question_sha256=task_question_sha256,
    )
    # ABI-only construction never reopens runtime provider bytes.  A211 and
    # C211 now each have a strict real adapter, but only the executable launcher
    # may clear this provider-reopen boundary.
    adapter_available = profile.label in {"A211", "C211"}
    adapter_blocker = (
        f"{profile.label.lower()}_runtime_providers_not_reopened_by_abi_construction_only"
    )
    blockers = (*authority_blockers, adapter_blocker)
    result = {
        "schema_version": 1,
        "kind": "action_ball_211_mujoco_abi_construction_receipt_v1",
        "profile": profile.label,
        "num_envs": num_envs,
        "actor_shape": [num_envs, profile.actor.width],
        "critic_shape": [num_envs, profile.critic.width],
        "actor_layout_sha256": profile.actor.content_sha256,
        "critic_layout_sha256": profile.critic.content_sha256,
        "observation_contract_sha256": profile.observation_contract_sha256,
        "wait_mask_contract_sha256": profile.wait_mask_contract_sha256,
        "actor_normalizer_identity": profile.actor_normalizer_identity,
        "critic_normalizer_identity": profile.critic_normalizer_identity,
        "runtime_tensor_materialized": False,
        "native_real_vecenv_adapter_available": adapter_available,
        "source_digests_syntactically_bound": not authority_blockers,
        "runtime_ready": False,
        "blockers": list(blockers),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    result["content_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "ACTOR_WIDTH",
    "CRITIC_WIDTH",
    "A211_PROFILE",
    "C211_PROFILE",
    "PROFILES",
    "PLANT_OBSERVATION_AUTHORITY_KIND",
    "MEASURED_MIMIC_AUTHORITY_KIND",
    "TASK_QUESTION_AUTHORITY_KIND",
    "ActionBall211ABIError",
    "ActionBall211AuthorityBlocked",
    "ActionBall211Profile",
    "ObservationAuthorities",
    "ObservationField",
    "ObservationLane",
    "construction_receipt",
    "flatten_lane_groups",
    "flatten_profile_groups",
    "require_runtime_authorities",
    "runtime_authority_blockers",
]
