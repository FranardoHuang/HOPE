"""Immutable one-question tape for diagnostic ActionBall N1 ablations.

The ordinary ActionBall producer samples and solves a new task when a birth
needs a receipt.  That is the correct path once the ball/task distribution is
live, but it is needless work for the first N1 learnability experiment where
the base pose, incoming ball and landing aim are intentionally constant.

This module freezes that constant question once and materializes reset rows by
index/copy only.  It deliberately does not import ``continuous_questions`` and
has no callback through which LM can run.  The five contact-target recipes are
stored in one container; they share the exact same base-question SHA and fixed
width, and differ only in their target columns, producer SHA and validity mask.

The tape remains diagnostic-only.  Its logical sample indices/draw ranges keep
the existing pool conservation protocol usable, but ``physical_rng_draws`` is
always zero: no claim is made that those compatibility slots were random draws.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

try:
    from . import action_ball_runtime as _runtime
except ImportError:
    _runtime_path = Path(__file__).with_name("action_ball_runtime.py")
    _runtime_spec = importlib.util.spec_from_file_location(
        "_fixed_question_action_ball_runtime", _runtime_path
    )
    if _runtime_spec is None or _runtime_spec.loader is None:
        raise
    _runtime = importlib.util.module_from_spec(_runtime_spec)
    sys.modules[_runtime_spec.name] = _runtime
    _runtime_spec.loader.exec_module(_runtime)


SCHEMA_VERSION = 1
KIND = "action_ball_n1_immutable_single_question_tape"

TARGET_RECIPES = (
    "current_lm",
    "analytic_full",
    "analytic_no_velocity",
    "teacher_pos_face_no_velocity",
    "outcome_dense_only",
)

TARGET_VALIDITY_BY_RECIPE = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}

QUESTION_LAYOUT = (
    ("base_goal_w_m", 3),
    ("ball_contact_w_m", 3),
    ("time_to_contact_s", 1),
    ("incoming_velocity_w_mps", 3),
    ("incoming_spin_w_radps", 3),
    ("landing_aim_w_xy_m", 2),
)
TARGET_LAYOUT = (
    ("desired_racket_site_w_m", 3),
    ("desired_racket_face_center_velocity_w_mps", 3),
    ("desired_racket_face_normal_w", 3),
)
VALIDITY_LAYOUT = (
    ("desired_position_valid", 1),
    ("desired_velocity_valid", 1),
    ("desired_face_valid", 1),
)
INSTALL_LAYOUT = (
    ("ball_contact_w_m", 3),
    ("racket_site_target_w_m", 3),
    ("base_goal_w_m", 3),
    ("racket_face_center_velocity_w_mps", 3),
    ("racket_site_velocity_w_mps", 3),
    ("racket_command_quat_wxyz", 4),
    ("racket_normal_w", 3),
    ("incoming_velocity_w_mps", 3),
    ("incoming_spin_w_radps", 3),
    ("landing_aim_w_xy_m", 2),
    ("time_to_contact_s", 1),
)
# Motion owns teacher phase/deadline validation.  This runtime-only row is
# derived from the already canonical, construction-validated tape payload; it
# is deliberately not added to the on-disk schema (which would invalidate
# existing immutable-tape SHAs).  The fixed-view consumer uploads it once and
# broadcasts row zero just like the install and observation rows.
TIMING_LAYOUT = (
    ("time_to_contact_s", 1),
    ("reference_t_hit_s", 1),
    ("reference_t_cycle_s", 1),
    ("reference_racket_site_speed_mps", 1),
    ("required_racket_site_speed_mps", 1),
    ("teacher_rate_min", 1),
    ("teacher_rate_max", 1),
    ("teacher_rate", 1),
    ("scaled_t_hit_s", 1),
    ("scaled_t_cycle_s", 1),
    ("pre_swing_wait_s", 1),
    ("reaction_margin_s", 1),
    ("racket_site_velocity_w_mps", 3),
)
QUESTION_WIDTH = sum(width for _name, width in QUESTION_LAYOUT)
TARGET_WIDTH = sum(width for _name, width in TARGET_LAYOUT)
INSTALL_WIDTH = sum(width for _name, width in INSTALL_LAYOUT)
TIMING_WIDTH = sum(width for _name, width in TIMING_LAYOUT)
OBSERVATION_WIDTH = (
    QUESTION_WIDTH
    + TARGET_WIDTH
)
# Validity is a recipe/config constant and is deliberately not appended to the
# actor.  The current five diagnostic arms therefore retain the already-frozen
# actor/critic widths; invalid p/v/face groups are exact zeros in the existing
# columns.  ``VALIDITY_LAYOUT`` remains in the artifact/lineage so checkpoint
# consumers can prove which groups were masked.

_TARGET_RUNTIME_FIELDS = (
    "racket_site_target_w_m",
    "mount_normal_sign",
    "racket_normal_w",
    "reference_racket_quat_wxyz",
    "reference_racket_angular_velocity_w_radps",
    "racket_command_quat_wxyz",
    "racket_face_center_velocity_w_mps",
    "racket_site_velocity_w_mps",
    "racket_command_angular_velocity_w_radps",
    "geometry_source_sha256",
    "reference_t_hit_s",
    "reference_t_cycle_s",
    "reference_racket_site_speed_mps",
    "required_racket_site_speed_mps",
    "reaction_margin_s",
    "teacher_rate_min",
    "teacher_rate_max",
    "teacher_rate",
    "scaled_t_hit_s",
    "scaled_t_cycle_s",
    "pre_swing_wait_s",
    "solver_residual_m",
)

_QUESTION_RECEIPT_FIELDS = (
    "base_goal_w_m",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "ball_contact_w_m",
    "time_to_contact_s",
    "incoming_speed_mps",
    "incoming_direction_b_yaw",
    "incoming_velocity_w_mps",
    "spin_magnitude_radps",
    "spin_direction_b_yaw",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "counter_rally_task",
)

_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _finite_tuple(value: object, *, name: str, width: int) -> Tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (tuple, list))
        or len(value) != width
    ):
        raise ValueError(f"{name} must be a length-{width} tuple/list")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _flatten_layout(
    payload: Mapping[str, object],
    layout: Sequence[Tuple[str, int]],
) -> Tuple[float, ...]:
    values = []
    for name, width in layout:
        raw = payload[name]
        if width == 1 and type(raw) in (int, float):
            row = (float(raw),)
        else:
            row = _finite_tuple(raw, name=name, width=width)
        values.extend(row)
    return tuple(values)


def _layout_payload(layout: Sequence[Tuple[str, int]]) -> list[dict]:
    return [{"name": name, "width": width} for name, width in layout]


def _question_payload(receipt: object) -> dict:
    if not isinstance(receipt, _runtime.ActionBallTaskReceipt):
        raise TypeError("question source must be ActionBallTaskReceipt")
    return {
        "action_uid": int(receipt.action_uid),
        "action_slot": int(receipt.action_slot),
        "profile_sha256": str(receipt.profile_sha256),
        "motion_sha256": str(receipt.motion_sha256),
        "physics_sha256": str(receipt.physics_sha256),
        "mobility_mode": str(receipt.mobility_mode),
        "base_yaw_rad": float(receipt.base_yaw_rad),
        "base_quat_wxyz": list(receipt.base_quat_wxyz),
        "base_spawn_w_m": list(receipt.base_spawn_w_m),
        "base_goal_w_m": list(receipt.base_goal_w_m),
        "base_spawn_latent_w_m": list(receipt.base_spawn_latent_w_m),
        "base_travel_latent_b_yaw_m": list(
            receipt.base_travel_latent_b_yaw_m
        ),
        "contact_offset_from_base_goal_b_yaw_m": list(
            receipt.contact_offset_from_base_goal_b_yaw_m
        ),
        "ball_contact_w_m": list(receipt.ball_contact_w_m),
        "time_to_contact_s": float(receipt.time_to_contact_s),
        "incoming_speed_mps": float(receipt.incoming_speed_mps),
        "incoming_direction_b_yaw": list(receipt.incoming_direction_b_yaw),
        "incoming_velocity_w_mps": list(receipt.incoming_velocity_w_mps),
        "spin_magnitude_radps": float(receipt.spin_magnitude_radps),
        "spin_direction_b_yaw": list(receipt.spin_direction_b_yaw),
        "incoming_spin_w_radps": list(receipt.incoming_spin_w_radps),
        "landing_aim_w_xy_m": list(receipt.landing_aim_w_xy_m),
        "counter_rally_task": (
            None
            if receipt.counter_rally_task is None
            else receipt.counter_rally_task.to_dict()
        ),
    }


def _runtime_target_payload(receipt: object) -> dict:
    if not isinstance(receipt, _runtime.ActionBallTaskReceipt):
        raise TypeError("target source must be ActionBallTaskReceipt")
    payload = {}
    for name in _TARGET_RUNTIME_FIELDS:
        value = getattr(receipt, name)
        payload[name] = list(value) if isinstance(value, tuple) else value
    return payload


def _vectors_close(
    left: Sequence[float], right: Sequence[float], *, tolerance: float
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(
            float(a), float(b), rel_tol=0.0, abs_tol=tolerance
        )
        for a, b in zip(left, right)
    )


def _validate_runtime_target(
    question_receipt: object, target: "TargetVariant"
) -> None:
    """Re-prove serialized geometry/timing once, before the hot reset path."""

    runtime = target.runtime_target
    geometry = _runtime._contact_geometry.solve_exact_face_contact(
        ball_contact_w_m=question_receipt.ball_contact_w_m,
        racket_face_center_velocity_w_mps=runtime[
            "racket_face_center_velocity_w_mps"
        ],
        solved_raw_a_normal_w=runtime["racket_normal_w"],
        mount_normal_sign=runtime["mount_normal_sign"],
        reference_racket_quat_wxyz=runtime[
            "reference_racket_quat_wxyz"
        ],
        reference_racket_angular_velocity_w_radps=runtime[
            "reference_racket_angular_velocity_w_radps"
        ],
        reference_racket_site_speed_mps=runtime[
            "reference_racket_site_speed_mps"
        ],
        teacher_rate_min=runtime["teacher_rate_min"],
        teacher_rate_max=runtime["teacher_rate_max"],
    )
    for name in (
        "racket_site_target_w_m",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
    ):
        if not _vectors_close(
            runtime[name], getattr(geometry, name), tolerance=1.0e-10
        ):
            raise ValueError(
                f"target {target.recipe!r} runtime field {name} fails "
                "the exact racket geometry proof"
            )
    if not _vectors_close(
        runtime["racket_command_quat_wxyz"],
        geometry.racket_command_quat_wxyz,
        tolerance=1.0e-12,
    ):
        raise ValueError(
            f"target {target.recipe!r} command quaternion fails exact geometry"
        )
    if runtime["geometry_source_sha256"] != geometry.geometry_source_sha256:
        raise ValueError(
            f"target {target.recipe!r} geometry source SHA mismatch"
        )
    timing = _runtime.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=runtime["racket_site_velocity_w_mps"],
        time_to_contact_s=question_receipt.time_to_contact_s,
        reference_t_hit_s=runtime["reference_t_hit_s"],
        reference_t_cycle_s=runtime["reference_t_cycle_s"],
        reference_racket_site_speed_mps=runtime[
            "reference_racket_site_speed_mps"
        ],
        reaction_margin_s=runtime["reaction_margin_s"],
        teacher_rate_min=runtime["teacher_rate_min"],
        teacher_rate_max=runtime["teacher_rate_max"],
    )
    declared = (
        runtime["required_racket_site_speed_mps"],
        runtime["teacher_rate"],
        runtime["scaled_t_hit_s"],
        runtime["scaled_t_cycle_s"],
        runtime["pre_swing_wait_s"],
    )
    expected = (
        timing.required_racket_site_speed_mps,
        timing.teacher_rate,
        timing.scaled_t_hit_s,
        timing.scaled_t_cycle_s,
        timing.pre_swing_wait_s,
    )
    if tuple(float(value) for value in declared) != expected:
        raise ValueError(f"target {target.recipe!r} timing proof mismatch")
    runtime_groups = (
        runtime["racket_site_target_w_m"],
        runtime["racket_face_center_velocity_w_mps"],
        runtime["racket_normal_w"],
    )
    desired_groups = (
        target.desired_racket_site_w_m,
        target.desired_racket_face_center_velocity_w_mps,
        target.desired_racket_face_normal_w,
    )
    for valid, desired, runtime_value in zip(
        target.validity_mask, desired_groups, runtime_groups
    ):
        if valid and not _vectors_close(
            desired, runtime_value, tolerance=1.0e-12
        ):
            raise ValueError(
                f"target {target.recipe!r} valid desired columns differ "
                "from its runtime target"
            )


@dataclass(frozen=True)
class TargetVariant:
    """One target producer over the shared immutable ball question."""

    recipe: str
    producer_sha256: str
    desired_racket_site_w_m: Tuple[float, float, float]
    desired_racket_face_center_velocity_w_mps: Tuple[float, float, float]
    desired_racket_face_normal_w: Tuple[float, float, float]
    runtime_target: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.recipe not in TARGET_RECIPES:
            raise ValueError(f"unknown target recipe {self.recipe!r}")
        object.__setattr__(
            self,
            "producer_sha256",
            _sha256(self.producer_sha256, name="producer_sha256"),
        )
        for name in (
            "desired_racket_site_w_m",
            "desired_racket_face_center_velocity_w_mps",
            "desired_racket_face_normal_w",
        ):
            object.__setattr__(
                self,
                name,
                _finite_tuple(getattr(self, name), name=name, width=3),
            )
        if not isinstance(self.runtime_target, Mapping):
            raise ValueError("runtime_target must be a mapping")
        if set(self.runtime_target) != set(_TARGET_RUNTIME_FIELDS):
            raise ValueError(
                "runtime_target fields differ from the fixed receipt target contract"
            )
        frozen = json.loads(
            json.dumps(
                dict(self.runtime_target),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        object.__setattr__(self, "runtime_target", frozen)

    @property
    def validity_mask(self) -> Tuple[bool, bool, bool]:
        return TARGET_VALIDITY_BY_RECIPE[self.recipe]

    @property
    def column_payload(self) -> dict:
        return {
            "layout": _layout_payload(TARGET_LAYOUT),
            "values": list(self.raw_target_values),
            "validity_layout": _layout_payload(VALIDITY_LAYOUT),
            "validity_mask": list(self.validity_mask),
        }

    @property
    def column_sha256(self) -> str:
        return _sha256_json(self.column_payload)

    @property
    def raw_target_values(self) -> Tuple[float, ...]:
        return (
            *self.desired_racket_site_w_m,
            *self.desired_racket_face_center_velocity_w_mps,
            *self.desired_racket_face_normal_w,
        )

    @property
    def masked_target_values(self) -> Tuple[float, ...]:
        result = []
        for valid, start in zip(self.validity_mask, (0, 3, 6)):
            values = self.raw_target_values[start : start + 3]
            result.extend(values if valid else (0.0, 0.0, 0.0))
        return tuple(result)

    def to_dict(self) -> dict:
        return {
            "recipe": self.recipe,
            "producer_sha256": self.producer_sha256,
            "column_sha256": self.column_sha256,
            "desired_racket_site_w_m": list(self.desired_racket_site_w_m),
            "desired_racket_face_center_velocity_w_mps": list(
                self.desired_racket_face_center_velocity_w_mps
            ),
            "desired_racket_face_normal_w": list(
                self.desired_racket_face_normal_w
            ),
            "validity_mask": list(self.validity_mask),
            "runtime_target": dict(self.runtime_target),
        }

    @classmethod
    def from_dict(cls, value: object) -> "TargetVariant":
        if not isinstance(value, Mapping):
            raise ValueError("target variant must be a mapping")
        required = {
            "recipe",
            "producer_sha256",
            "column_sha256",
            "desired_racket_site_w_m",
            "desired_racket_face_center_velocity_w_mps",
            "desired_racket_face_normal_w",
            "validity_mask",
            "runtime_target",
        }
        if set(value) != required:
            raise ValueError("target variant has unexpected fields")
        variant = cls(
            recipe=str(value["recipe"]),
            producer_sha256=str(value["producer_sha256"]),
            desired_racket_site_w_m=tuple(value["desired_racket_site_w_m"]),
            desired_racket_face_center_velocity_w_mps=tuple(
                value["desired_racket_face_center_velocity_w_mps"]
            ),
            desired_racket_face_normal_w=tuple(
                value["desired_racket_face_normal_w"]
            ),
            runtime_target=value["runtime_target"],
        )
        if list(variant.validity_mask) != list(value["validity_mask"]):
            raise ValueError("target validity mask differs from recipe authority")
        if variant.column_sha256 != value["column_sha256"]:
            raise ValueError("target column SHA mismatch")
        return variant

    @classmethod
    def from_receipt(
        cls,
        *,
        recipe: str,
        producer_sha256: str,
        receipt: object,
        desired_racket_site_w_m: Sequence[float] | None = None,
        desired_racket_face_center_velocity_w_mps: Sequence[float] | None = None,
        desired_racket_face_normal_w: Sequence[float] | None = None,
    ) -> "TargetVariant":
        return cls(
            recipe=recipe,
            producer_sha256=producer_sha256,
            desired_racket_site_w_m=tuple(
                receipt.racket_site_target_w_m
                if desired_racket_site_w_m is None
                else desired_racket_site_w_m
            ),
            desired_racket_face_center_velocity_w_mps=tuple(
                receipt.racket_face_center_velocity_w_mps
                if desired_racket_face_center_velocity_w_mps is None
                else desired_racket_face_center_velocity_w_mps
            ),
            desired_racket_face_normal_w=tuple(
                receipt.racket_normal_w
                if desired_racket_face_normal_w is None
                else desired_racket_face_normal_w
            ),
            runtime_target=_runtime_target_payload(receipt),
        )


@dataclass(frozen=True)
class ImmutableN1QuestionTape:
    """One physical question plus every ablation target projection."""

    source_receipt: object
    targets: Mapping[str, TargetVariant]

    def __post_init__(self) -> None:
        if not isinstance(self.source_receipt, _runtime.ActionBallTaskReceipt):
            raise TypeError("source_receipt must be ActionBallTaskReceipt")
        if self.source_receipt.mobility_mode != "no_move":
            raise ValueError("immutable N1 tape currently requires mobility_mode='no_move'")
        if set(self.targets) != set(TARGET_RECIPES):
            raise ValueError(
                "one tape container must carry exactly all five target recipes"
            )
        frozen_targets = {}
        for recipe in TARGET_RECIPES:
            target = self.targets[recipe]
            if not isinstance(target, TargetVariant) or target.recipe != recipe:
                raise ValueError(f"target mapping mismatch for {recipe}")
            _validate_runtime_target(self.source_receipt, target)
            frozen_targets[recipe] = target
        object.__setattr__(self, "targets", frozen_targets)

    @property
    def question_payload(self) -> dict:
        return _question_payload(self.source_receipt)

    @property
    def question_sha256(self) -> str:
        return _sha256_json(self.question_payload)

    @property
    def question_values(self) -> Tuple[float, ...]:
        return _flatten_layout(self.question_payload, QUESTION_LAYOUT)

    @property
    def canonical_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "diagnostic_unauthorized": True,
            "row_count": 1,
            "question_shape": [1, QUESTION_WIDTH],
            "install_shape_per_recipe": [1, INSTALL_WIDTH],
            "observation_shape_per_recipe": [1, OBSERVATION_WIDTH],
            "question_layout": _layout_payload(QUESTION_LAYOUT),
            "target_layout": _layout_payload(TARGET_LAYOUT),
            "validity_layout": _layout_payload(VALIDITY_LAYOUT),
            "install_layout": _layout_payload(INSTALL_LAYOUT),
            "question_sha256": self.question_sha256,
            "question_values": list(self.question_values),
            "question": self.question_payload,
            "source_task_receipt": self.source_receipt.to_dict(),
            "targets": {
                recipe: self.targets[recipe].to_dict()
                for recipe in TARGET_RECIPES
            },
            "reset_semantics": {
                "selection": "constant_row_zero",
                "online_sampler_calls": 0,
                "online_lm_calls": 0,
                "physical_rng_draws": 0,
                "logical_sample_draw_slots_per_issue": (
                    _runtime.SAMPLER_SAMPLE_DRAW_COUNT
                ),
                "proposal_disposition": "one_proposed_one_admitted",
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha256_json(self.canonical_payload)

    def to_dict(self) -> dict:
        return {
            **self.canonical_payload,
            "canonical_sha256": self.canonical_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ImmutableN1QuestionTape":
        if not isinstance(value, Mapping):
            raise ValueError("immutable tape must be a mapping")
        declared = value.get("canonical_sha256")
        source = _runtime.ActionBallTaskReceipt.from_dict(
            value.get("source_task_receipt")
        )
        raw_targets = value.get("targets")
        if not isinstance(raw_targets, Mapping):
            raise ValueError("immutable tape targets must be a mapping")
        tape = cls(
            source_receipt=source,
            targets={
                recipe: TargetVariant.from_dict(raw_targets[recipe])
                for recipe in TARGET_RECIPES
            },
        )
        expected = tape.to_dict()
        if dict(value) != expected:
            if declared != tape.canonical_sha256:
                raise ValueError("immutable tape canonical SHA mismatch")
            raise ValueError("immutable tape document is not canonical")
        return tape

    @classmethod
    def from_receipts(
        cls,
        *,
        question_receipt: object,
        target_receipts: Mapping[str, object],
        target_producer_sha256: Mapping[str, str],
    ) -> "ImmutableN1QuestionTape":
        if set(target_receipts) != set(TARGET_RECIPES):
            raise ValueError("target_receipts must contain all five recipes")
        if set(target_producer_sha256) != set(TARGET_RECIPES):
            raise ValueError("target producer SHAs must contain all five recipes")
        question_sha = _sha256_json(_question_payload(question_receipt))
        targets = {}
        for recipe in TARGET_RECIPES:
            receipt = target_receipts[recipe]
            if _sha256_json(_question_payload(receipt)) != question_sha:
                raise ValueError(
                    f"target recipe {recipe!r} does not share the base question"
                )
            targets[recipe] = TargetVariant.from_receipt(
                recipe=recipe,
                producer_sha256=target_producer_sha256[recipe],
                receipt=receipt,
            )
        return cls(source_receipt=question_receipt, targets=targets)

    def observation_row(self, recipe: str) -> Tuple[float, ...]:
        target = self.targets[recipe]
        row = (
            *self.question_values,
            *target.masked_target_values,
        )
        if len(row) != OBSERVATION_WIDTH:
            raise RuntimeError("immutable tape observation width drifted")
        return row

    def target_lineage(self, recipe: str) -> dict:
        target = self.targets[recipe]
        return {
            "base_question_sha256": self.question_sha256,
            "target_recipe": recipe,
            "target_producer_sha256": target.producer_sha256,
            "target_column_sha256": target.column_sha256,
            "target_validity_mask": list(target.validity_mask),
            "tape_canonical_sha256": self.canonical_sha256,
        }

    def install_row(self, recipe: str) -> Tuple[float, ...]:
        """Return the prevalidated 31-D runtime row copied by reset.

        This is exactly the existing diagnostic ActionBall install order.  A
        future shared-code hook can build one device tensor from this single
        row and expand/index row zero for every resetting environment, without
        constructing per-env receipts or touching ``LazyActionTaskPool``.
        """

        payload = {
            **self.question_payload,
            **self.targets[recipe].runtime_target,
        }
        row = _flatten_layout(payload, INSTALL_LAYOUT)
        if len(row) != INSTALL_WIDTH:
            raise RuntimeError("immutable tape install width drifted")
        return row

    def timing_row(self, recipe: str) -> Tuple[float, ...]:
        """Return Motion's prevalidated fixed teacher-timing row."""

        payload = {
            **self.question_payload,
            **self.targets[recipe].runtime_target,
        }
        row = _flatten_layout(payload, TIMING_LAYOUT)
        if len(row) != TIMING_WIDTH:
            raise RuntimeError("immutable tape timing width drifted")
        return row

    def reset_batch_view(
        self, recipe: str, *, batch_size: int
    ) -> "ImmutableResetBatchView":
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive plain integer")
        return ImmutableResetBatchView(
            batch_size=batch_size,
            row_index=0,
            install_row=self.install_row(recipe),
            observation_row=self.observation_row(recipe),
            timing_row=self.timing_row(recipe),
            lineage=self.target_lineage(recipe),
        )


@dataclass(frozen=True)
class MaterializedTapeIssue:
    task_receipt: object
    observation_row: Tuple[float, ...]
    lineage: Mapping[str, object]


@dataclass(frozen=True)
class ImmutableResetBatchView:
    """O(1) view of N resets selecting immutable row zero.

    The consumer should upload ``install_row``/``observation_row`` once and
    expand or index row zero on device.  ``batch_size`` is metadata, not a
    request to allocate N Python rows.
    """

    batch_size: int
    row_index: int
    install_row: Tuple[float, ...]
    observation_row: Tuple[float, ...]
    timing_row: Tuple[float, ...]
    lineage: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive plain integer")
        if self.row_index != 0:
            raise ValueError("single-question tape row_index must be zero")
        if len(self.install_row) != INSTALL_WIDTH:
            raise ValueError("immutable reset install row has wrong width")
        if len(self.observation_row) != OBSERVATION_WIDTH:
            raise ValueError("immutable reset observation row has wrong width")
        if len(self.timing_row) != TIMING_WIDTH:
            raise ValueError("immutable reset timing row has wrong width")

    def expand_install_rows(self, install_row_tensor: object) -> object:
        """Expand a cached row-zero device tensor without allocating N rows.

        This method deliberately has no Torch import.  A consumer uploads each
        immutable install row once, then passes the cached 1-D or
        ``[1, INSTALL_WIDTH]`` device tensor here.  The return value is an
        ordinary zero-stride ``expand`` view; this method never iterates over
        environments or Python values.
        """

        return _expand_device_row_zero(
            install_row_tensor,
            batch_size=self.batch_size,
            width=INSTALL_WIDTH,
            name="install_row_tensor",
        )

    def expand_observation_rows(self, observation_row_tensor: object) -> object:
        """Broadcast the cached observation row as a zero-stride view."""

        return _expand_device_row_zero(
            observation_row_tensor,
            batch_size=self.batch_size,
            width=OBSERVATION_WIDTH,
            name="observation_row_tensor",
        )

    def expand_timing_rows(self, timing_row_tensor: object) -> object:
        """Broadcast the cached Motion timing row as a zero-stride view."""

        return _expand_device_row_zero(
            timing_row_tensor,
            batch_size=self.batch_size,
            width=TIMING_WIDTH,
            name="timing_row_tensor",
        )


def _expand_device_row_zero(
    row_tensor: object,
    *,
    batch_size: int,
    width: int,
    name: str,
) -> object:
    """Return ``[batch_size, width]`` through a tensor-like ``expand`` view."""

    shape = getattr(row_tensor, "shape", None)
    try:
        shape = tuple(shape)
    except TypeError as error:
        raise TypeError(f"{name} must expose a tensor-like shape") from error
    if shape == (width,):
        unsqueeze = getattr(row_tensor, "unsqueeze", None)
        if not callable(unsqueeze):
            raise TypeError(f"{name} must provide unsqueeze()")
        row_zero = unsqueeze(0)
    elif shape == (1, width):
        row_zero = row_tensor
    else:
        raise ValueError(
            f"{name} must have shape ({width},) or (1, {width}), got {shape}"
        )
    expand = getattr(row_zero, "expand", None)
    if not callable(expand):
        raise TypeError(f"{name} must provide expand()")
    rows = expand(batch_size, width)
    if tuple(getattr(rows, "shape", ())) != (batch_size, width):
        raise ValueError(f"{name}.expand() returned the wrong shape")
    return rows


def _sample_identity(
    *,
    birth: object,
    question: Mapping[str, object],
    sample_index: int,
    draw_start: int,
) -> dict:
    draw_end = draw_start + _runtime.SAMPLER_SAMPLE_DRAW_COUNT
    return {
        "schema_version": _runtime.SAMPLER_SCHEMA_VERSION,
        "kind": "swing_sample",
        "sampler_contract_sha256": birth.sampler_sha256,
        "arm_catalog_sha256": birth.arm_catalog_sha256,
        "sample_index": sample_index,
        "action_uid": birth.action_uid,
        "domain_epoch": birth.domain_epoch,
        "domain_levels": birth.domain_levels.to_dict(),
        "birth_id": birth.sampler_birth_sha256,
        "profile_sha256": birth.profile_sha256,
        "levels_sha256": birth.levels_sha256,
        "draw_start": draw_start,
        "draw_end": draw_end,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": birth.base_yaw_rad,
        "base_start_w_m": list(birth.base_spawn_w_m),
        "base_spawn_latent_w_m": question["base_spawn_latent_w_m"],
        "base_travel_latent_b_yaw_m": question[
            "base_travel_latent_b_yaw_m"
        ],
        "base_goal_w_m": question["base_goal_w_m"],
        "contact_offset_from_base_goal_b_yaw_m": question[
            "contact_offset_from_base_goal_b_yaw_m"
        ],
        "contact_w_m": question["ball_contact_w_m"],
        "time_to_contact_s": question["time_to_contact_s"],
        "incoming_speed_mps": question["incoming_speed_mps"],
        "incoming_direction_b_yaw": question["incoming_direction_b_yaw"],
        "incoming_direction_w": list(
            _runtime._rotate_yaw(
                question["incoming_direction_b_yaw"], birth.base_yaw_rad
            )
        ),
        "incoming_velocity_w_mps": question["incoming_velocity_w_mps"],
        "spin_magnitude_radps": question["spin_magnitude_radps"],
        "spin_direction_b_yaw": question["spin_direction_b_yaw"],
        "spin_direction_w": list(
            _runtime._rotate_yaw(
                question["spin_direction_b_yaw"], birth.base_yaw_rad
            )
        ),
        "spin_w_radps": question["incoming_spin_w_radps"],
        "landing_aim_w_xy_m": question["landing_aim_w_xy_m"],
    }


class FixedQuestionTapeSolver:
    """Pool-compatible materializer that cannot invoke an online solver."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        tape: ImmutableN1QuestionTape,
        target_recipe: str,
        solver_contract_sha256: str,
    ) -> None:
        if not isinstance(tape, ImmutableN1QuestionTape):
            raise TypeError("tape must be ImmutableN1QuestionTape")
        if target_recipe not in TARGET_RECIPES:
            raise ValueError(f"unknown target recipe {target_recipe!r}")
        self.tape = tape
        self.target_recipe = target_recipe
        self.solver_contract_sha256 = _sha256(
            solver_contract_sha256, name="solver_contract_sha256"
        )
        self.state_owner_sha256 = _sha256_json(
            {
                "kind": "immutable_n1_tape_state_owner",
                "schema_version": self.STATE_SCHEMA_VERSION,
                "tape_canonical_sha256": tape.canonical_sha256,
                "target_recipe": target_recipe,
                "solver_contract_sha256": self.solver_contract_sha256,
            }
        )
        self._emitted = []
        self._assignments: Dict[Tuple[int, int], Tuple[str, int]] = {}
        self._highwater: Dict[int, Tuple[int, int]] = {}
        # Build and hash the immutable row/lineage once at construction.  The
        # reset hot path only changes batch-size metadata around these shared
        # tuples; it does not rebuild payloads or receipts.
        base_reset_batch_template = tape.reset_batch_view(
            target_recipe, batch_size=1
        )
        self._reset_batch_template = ImmutableResetBatchView(
            batch_size=1,
            row_index=base_reset_batch_template.row_index,
            install_row=base_reset_batch_template.install_row,
            observation_row=base_reset_batch_template.observation_row,
            timing_row=base_reset_batch_template.timing_row,
            lineage=MappingProxyType(
                {
                    **base_reset_batch_template.lineage,
                    "solver_contract_sha256": self.solver_contract_sha256,
                    "state_owner_sha256": self.state_owner_sha256,
                }
            ),
        )

    @property
    def online_lm_calls(self) -> int:
        return 0

    @property
    def physical_rng_draws(self) -> int:
        return 0

    def reset_batch_view(self, *, batch_size: int) -> ImmutableResetBatchView:
        """Return the O(1) fixed-row producer view for a reset batch.

        This is intentionally separate from the legacy pool-compatible
        ``solve_many`` API.  Reading the view is stateless: task generations,
        logical counters, replay authority, and receipt materialization remain
        unchanged until the consumer commits through its chosen protocol.
        """

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive plain integer")
        template = self._reset_batch_template
        return ImmutableResetBatchView(
            batch_size=batch_size,
            row_index=template.row_index,
            install_row=template.install_row,
            observation_row=template.observation_row,
            timing_row=template.timing_row,
            lineage=template.lineage,
        )

    def _assert_birth_matches_question(self, birth: object) -> None:
        question = self.tape.question_payload
        source = self.tape.source_receipt
        expected = (
            source.action_uid,
            source.action_slot,
            source.profile_sha256,
            source.motion_sha256,
            source.physics_sha256,
            source.mobility_mode,
            source.base_yaw_rad,
            source.base_quat_wxyz,
            source.base_spawn_w_m,
            source.domain_levels,
        )
        actual = (
            birth.action_uid,
            birth.action_slot,
            birth.profile_sha256,
            birth.motion_sha256,
            birth.physics_sha256,
            birth.mobility_mode,
            birth.base_yaw_rad,
            birth.base_quat_wxyz,
            birth.base_spawn_w_m,
            birth.domain_levels,
        )
        if actual != expected:
            raise ValueError(
                "birth differs from immutable tape action/base/fixed-domain identity"
            )
        if tuple(question["base_goal_w_m"]) != tuple(birth.base_spawn_w_m):
            raise ValueError("fixed no-move tape base goal differs from birth spawn")

    def _materialize(
        self,
        *,
        birth: object,
        swing_generation: int,
        sample_index: int,
        draw_start: int,
    ) -> MaterializedTapeIssue:
        self._assert_birth_matches_question(birth)
        question = self.tape.question_payload
        target = self.tape.targets[self.target_recipe]
        sample_identity = _sample_identity(
            birth=birth,
            question=question,
            sample_index=sample_index,
            draw_start=draw_start,
        )
        counter_rally = question["counter_rally_task"]
        if counter_rally is not None:
            counter_rally = _runtime.CounterRallyTaskIdentity.from_dict(
                counter_rally
            )
        kwargs = {
            "sample_sha256": _sha256_json(sample_identity),
            "sample_index": sample_index,
            "sample_draw_start": draw_start,
            "sample_draw_end": (
                draw_start + _runtime.SAMPLER_SAMPLE_DRAW_COUNT
            ),
            "swing_generation": swing_generation,
            "base_goal_w_m": tuple(question["base_goal_w_m"]),
            "base_spawn_latent_w_m": tuple(
                question["base_spawn_latent_w_m"]
            ),
            "base_travel_latent_b_yaw_m": tuple(
                question["base_travel_latent_b_yaw_m"]
            ),
            "contact_offset_from_base_goal_b_yaw_m": tuple(
                question["contact_offset_from_base_goal_b_yaw_m"]
            ),
            "ball_contact_w_m": tuple(question["ball_contact_w_m"]),
            "time_to_contact_s": float(question["time_to_contact_s"]),
            "incoming_speed_mps": float(question["incoming_speed_mps"]),
            "incoming_direction_b_yaw": tuple(
                question["incoming_direction_b_yaw"]
            ),
            "incoming_velocity_w_mps": tuple(
                question["incoming_velocity_w_mps"]
            ),
            "spin_magnitude_radps": float(question["spin_magnitude_radps"]),
            "spin_direction_b_yaw": tuple(question["spin_direction_b_yaw"]),
            "incoming_spin_w_radps": tuple(
                question["incoming_spin_w_radps"]
            ),
            "landing_aim_w_xy_m": tuple(question["landing_aim_w_xy_m"]),
            "contact_time_step_s": None,
            "time_to_contact_tick": None,
            "birth_index": -1,
            "birth_sampling_stratum": "domain",
            "birth_sampling_levels": None,
            "birth_frontier_arm": None,
            "sampling_mixture": None,
            "sampling_stratum": "domain",
            "sampling_levels": None,
            "frontier_arm": None,
            "counter_rally_task": counter_rally,
            **target.runtime_target,
        }
        receipt = _runtime._diagnostic_prevalidated_task_receipt_from_birth(
            birth, **kwargs
        )
        receipt._assert_sample_relations_without_rehash()
        lineage = {
            **self.tape.target_lineage(self.target_recipe),
            "task_receipt_sha256": receipt.canonical_sha256,
            "sample_sha256": receipt.sample_sha256,
            "logical_sample_index": sample_index,
            "logical_sample_draw_start": draw_start,
            "logical_sample_draw_end": receipt.sample_draw_end,
            "physical_rng_draws": 0,
            "online_lm_calls": 0,
        }
        return MaterializedTapeIssue(
            task_receipt=receipt,
            observation_row=self.tape.observation_row(self.target_recipe),
            lineage=lineage,
        )

    def materialize_many(self, requests: Sequence[object]):
        requests = tuple(requests)
        if not requests:
            raise ValueError("immutable tape request batch must be non-empty")
        staged_emitted = list(self._emitted)
        staged_assignments = dict(self._assignments)
        staged_highwater = dict(self._highwater)
        batches = []
        issues = []
        for request in requests:
            if request.action_uid != self.tape.source_receipt.action_uid:
                raise ValueError("immutable N1 tape received a different action UID")
            receipts = []
            proposal_indices = []
            for offset in range(request.minimum_receipts):
                last_index, last_draw_end = staged_highwater.get(
                    request.action_uid, (-1, 0)
                )
                sample_index = last_index + 1
                draw_start = max(
                    last_draw_end,
                    int(request.birth.sampler_draw_end),
                )
                issue = self._materialize(
                    birth=request.birth,
                    swing_generation=request.swing_generation_start + offset,
                    sample_index=sample_index,
                    draw_start=draw_start,
                )
                key = (request.action_uid, sample_index)
                if key in staged_assignments:
                    raise ValueError("immutable tape sample index replayed")
                staged_assignments[key] = (
                    request.birth.canonical_sha256,
                    request.refill_index,
                )
                staged_highwater[request.action_uid] = (
                    sample_index,
                    issue.task_receipt.sample_draw_end,
                )
                staged_emitted.append(issue.task_receipt)
                receipts.append(issue.task_receipt)
                proposal_indices.append(sample_index)
                issues.append(issue)
            batches.append(
                _runtime.ActionPoolRefillBatch(
                    action_uid=request.action_uid,
                    proposed_count=len(proposal_indices),
                    proposal_sample_indices=tuple(proposal_indices),
                    receipts=tuple(receipts),
                )
            )
        self._emitted = staged_emitted
        self._assignments = staged_assignments
        self._highwater = staged_highwater
        return tuple(batches), tuple(issues)

    def __call__(self, request: object):
        batches, _issues = self.materialize_many((request,))
        return batches[0]

    def solve_many(self, requests: Sequence[object]):
        batches, _issues = self.materialize_many(requests)
        return batches

    def lineage_for_task(self, receipt: object) -> dict:
        for emitted in self._emitted:
            if emitted.canonical_sha256 == receipt.canonical_sha256:
                if emitted != receipt:
                    break
                return {
                    **self.tape.target_lineage(self.target_recipe),
                    "task_receipt_sha256": receipt.canonical_sha256,
                    "sample_sha256": receipt.sample_sha256,
                }
        raise ValueError("task receipt was not emitted by immutable tape")

    def state_dict(self) -> dict:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "tape_canonical_sha256": self.tape.canonical_sha256,
            "target_recipe": self.target_recipe,
            "solver_contract_sha256": self.solver_contract_sha256,
            "state_owner_sha256": self.state_owner_sha256,
            "physical_rng_draws": 0,
            "online_lm_calls": 0,
            "highwaters": [
                [uid, index, draw_end]
                for uid, (index, draw_end) in sorted(self._highwater.items())
            ],
            "assignments": [
                [uid, sample_index, birth_sha, refill_index]
                for (uid, sample_index), (
                    birth_sha,
                    refill_index,
                ) in sorted(self._assignments.items())
            ],
            "emitted_tasks": [receipt.to_dict() for receipt in self._emitted],
        }

    def load_state_dict(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("immutable tape solver state must be a mapping")
        fixed = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "tape_canonical_sha256": self.tape.canonical_sha256,
            "target_recipe": self.target_recipe,
            "solver_contract_sha256": self.solver_contract_sha256,
            "state_owner_sha256": self.state_owner_sha256,
            "physical_rng_draws": 0,
            "online_lm_calls": 0,
        }
        if any(state.get(name) != value for name, value in fixed.items()):
            raise ValueError("immutable tape solver state identity mismatch")
        emitted = [
            _runtime.ActionBallTaskReceipt.from_dict(row)
            for row in state.get("emitted_tasks", ())
        ]
        highwater = {}
        for row in state.get("highwaters", ()):
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ValueError("invalid immutable tape highwater row")
            uid, index, draw_end = row
            if (
                type(uid) is not int
                or type(index) is not int
                or type(draw_end) is not int
                or uid in highwater
            ):
                raise ValueError("invalid immutable tape highwater values")
            highwater[uid] = (index, draw_end)
        assignments = {}
        for row in state.get("assignments", ()):
            if not isinstance(row, (tuple, list)) or len(row) != 4:
                raise ValueError("invalid immutable tape assignment row")
            uid, sample_index, birth_sha, refill_index = row
            key = (uid, sample_index)
            if key in assignments:
                raise ValueError("duplicate immutable tape assignment")
            assignments[key] = (
                _sha256(birth_sha, name="assignment birth SHA"),
                refill_index,
            )
        expected_indices = {(r.action_uid, r.sample_index) for r in emitted}
        if set(assignments) != expected_indices:
            raise ValueError("immutable tape assignments do not cover emitted tasks")
        derived_highwater = {}
        for receipt in emitted:
            prior = derived_highwater.get(receipt.action_uid, (-1, 0))
            derived_highwater[receipt.action_uid] = (
                max(prior[0], receipt.sample_index),
                max(prior[1], receipt.sample_draw_end),
            )
        if highwater != derived_highwater:
            raise ValueError("immutable tape highwater disagrees with emitted tasks")
        for receipt in emitted:
            if receipt.action_uid != self.tape.source_receipt.action_uid:
                raise ValueError("restored immutable tape task has wrong action")
        self._emitted = emitted
        self._assignments = assignments
        self._highwater = highwater

    def assert_emitted_sample(self, receipt: object) -> None:
        if not any(
            emitted.sample_sha256 == receipt.sample_sha256
            and emitted == receipt
            for emitted in self._emitted
        ):
            raise ValueError("sample was not emitted by immutable tape")

    def assert_emitted_tasks(self, receipts: Sequence[object]) -> None:
        authority = {
            receipt.canonical_sha256: receipt for receipt in self._emitted
        }
        for receipt in receipts:
            if authority.get(receipt.canonical_sha256) != receipt:
                raise ValueError("task was not emitted by immutable tape")

    def emitted_task_count_for(self, action_uid: int) -> int:
        return sum(receipt.action_uid == action_uid for receipt in self._emitted)

    def task_transcript_for_birth(self, birth_sha256: str):
        digests = [
            receipt.canonical_sha256
            for receipt in self._emitted
            if receipt.birth_sha256 == birth_sha256
        ]
        return (
            len(digests),
            _runtime.task_transcript_sha256(birth_sha256, digests),
        )

    def assert_proposal_assignments(self, assignments: Sequence[object]) -> None:
        for assignment in assignments:
            expected = (
                assignment.birth.canonical_sha256,
                assignment.refill_index,
            )
            for sample_index in assignment.proposal_sample_indices:
                if self._assignments.get(
                    (assignment.birth.action_uid, sample_index)
                ) != expected:
                    raise ValueError(
                        "immutable tape proposal assignment mismatch"
                    )

    def sample_highwater_for(self, action_uid: int):
        return self._highwater.get(action_uid, (-1, 0))


def load_immutable_n1_tape(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> ImmutableN1QuestionTape:
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _sha256(
        expected_file_sha256, name="action_ball_immutable_tape_sha256"
    ):
        raise ValueError(
            "immutable N1 tape file SHA differs from configured authority"
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("immutable N1 tape is not canonical UTF-8 JSON") from error
    return ImmutableN1QuestionTape.from_dict(document)


__all__ = [
    "SCHEMA_VERSION",
    "KIND",
    "TARGET_RECIPES",
    "TARGET_VALIDITY_BY_RECIPE",
    "QUESTION_LAYOUT",
    "TARGET_LAYOUT",
    "VALIDITY_LAYOUT",
    "INSTALL_LAYOUT",
    "TIMING_LAYOUT",
    "QUESTION_WIDTH",
    "TARGET_WIDTH",
    "INSTALL_WIDTH",
    "TIMING_WIDTH",
    "OBSERVATION_WIDTH",
    "TargetVariant",
    "ImmutableN1QuestionTape",
    "MaterializedTapeIssue",
    "ImmutableResetBatchView",
    "FixedQuestionTapeSolver",
    "load_immutable_n1_tape",
]
