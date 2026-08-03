"""Dependency-free common-question identity for matched ActionBall arms.

The digest intentionally excludes policy ABI tails, desired-contact targets,
target-validity masks, and rewards.  A and C may differ on those treatment
axes only after both launch lineages bind the exact same receipt produced here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Union


SCHEMA_VERSION = 1
KIND = "action_ball_ac_comparison_question_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ComparisonQuestionError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ComparisonQuestionError("comparison question is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ComparisonQuestionError(f"{name} must be one lowercase SHA-256")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise ComparisonQuestionError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ComparisonQuestionError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, *, positive: bool = False) -> Union[float, int]:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ComparisonQuestionError(f"{name} must be finite")
    if positive and float(value) <= 0.0:
        raise ComparisonQuestionError(f"{name} must be positive")
    return value


def _vector(value: Any, size: int, name: str) -> list[Union[float, int]]:
    if type(value) is not list or len(value) != size:
        raise ComparisonQuestionError(f"{name} must be a {size}-vector")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonQuestionError(f"{name} must be an object")
    return value


def build_comparison_question_receipt(
    tape: Mapping[str, Any],
    *,
    tape_bytes: bytes,
    action_id: str,
    teacher_id: str,
    target_recipe: str,
    tape_row_index: int = 0,
) -> dict[str, Any]:
    """Project one immutable tape row onto the exact A/C common treatment base."""

    tape = _mapping(tape, "tape")
    if type(tape_bytes) is not bytes or tape_bytes != _canonical_bytes(tape) + b"\n":
        raise ComparisonQuestionError(
            "tape bytes must be the canonical supplied tape object plus newline"
        )
    if tape.get("row_count") != 1 or tape_row_index != 0:
        raise ComparisonQuestionError("v1 comparison requires immutable tape row zero of one")
    question = _mapping(tape.get("question"), "tape.question")
    source = _mapping(tape.get("source_task_receipt"), "tape.source_task_receipt")
    targets = _mapping(tape.get("targets"), "tape.targets")
    selected_target = _mapping(
        targets.get(_text(target_recipe, "target_recipe")),
        f"tape.targets.{target_recipe}",
    )
    if selected_target.get("recipe") != target_recipe:
        raise ComparisonQuestionError("selected target recipe label differs")
    runtime_target = _mapping(
        selected_target.get("runtime_target"),
        f"tape.targets.{target_recipe}.runtime_target",
    )
    question_sha = _digest(question)
    if tape.get("question_sha256") != question_sha:
        raise ComparisonQuestionError("tape question_sha256 does not match question bytes")
    tape_unsigned = dict(tape)
    tape_declared_sha = tape_unsigned.pop("canonical_sha256", None)
    tape_canonical_sha = _digest(tape_unsigned)
    if tape_declared_sha != tape_canonical_sha:
        raise ComparisonQuestionError("tape canonical_sha256 does not match tape payload")
    source_unsigned = dict(source)
    source_declared_sha = source_unsigned.pop("canonical_sha256", None)
    if source_declared_sha != _digest(source_unsigned):
        raise ComparisonQuestionError(
            "source task receipt canonical_sha256 does not match its payload"
        )

    # The compact tape question and its full source receipt must agree on every
    # shared field.  This prevents a relabelled source receipt from supplying
    # the clocks/base state while another row supplies the incoming ball.
    for field in (
        "action_uid",
        "motion_sha256",
        "ball_contact_w_m",
        "incoming_velocity_w_mps",
        "incoming_spin_w_radps",
        "base_spawn_w_m",
        "base_goal_w_m",
        "landing_aim_w_xy_m",
        "time_to_contact_s",
    ):
        if question.get(field) != source.get(field):
            raise ComparisonQuestionError(
                f"tape question/source common field differs: {field}"
            )

    dt = _finite(source.get("contact_time_step_s"), "contact_time_step_s", positive=True)
    tick = _integer(source.get("time_to_contact_tick"), "time_to_contact_tick")
    t_contact = _finite(source.get("time_to_contact_s"), "time_to_contact_s", positive=True)
    t_hit = _finite(source.get("scaled_t_hit_s"), "scaled_t_hit_s", positive=True)
    pre_wait = _finite(source.get("pre_swing_wait_s"), "pre_swing_wait_s")
    reference_t_hit = _finite(
        source.get("reference_t_hit_s"), "reference_t_hit_s", positive=True
    )
    if float(pre_wait) < 0.0:
        raise ComparisonQuestionError("pre_swing_wait_s must be non-negative")
    tolerance = 2.0e-6
    if (
        abs(float(tick) * float(dt) - float(t_contact)) > tolerance
        or abs(float(pre_wait) + float(t_hit) - float(t_contact)) > tolerance
    ):
        raise ComparisonQuestionError("t_hit/pre_wait/control-step clock does not close")
    # The target recipe is a treatment axis and therefore is not hashed.  Its
    # runtime clock is nevertheless an input to construction: an arm may not
    # claim the common receipt unless the target it actually consumes agrees
    # with the shared source clock.
    for field, common_value in (
        ("reference_t_hit_s", reference_t_hit),
        ("scaled_t_hit_s", t_hit),
        ("pre_swing_wait_s", pre_wait),
    ):
        target_value = _finite(
            runtime_target.get(field),
            f"tape.targets.{target_recipe}.runtime_target.{field}",
            positive=field != "pre_swing_wait_s",
        )
        if float(target_value) < 0.0 or abs(float(target_value) - float(common_value)) > tolerance:
            raise ComparisonQuestionError(
                f"selected target runtime clock differs from common source: {field}"
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "identity": {
            "action_id": _text(action_id, "action_id"),
            "action_uid": _integer(source.get("action_uid"), "action_uid", minimum=1),
            "action_slot": _integer(source.get("action_slot"), "action_slot"),
            "teacher_id": _text(teacher_id, "teacher_id"),
            "motion_sha256": _sha(source.get("motion_sha256"), "motion_sha256"),
            "tape_row_index": tape_row_index,
            "tape_file_sha256": hashlib.sha256(tape_bytes).hexdigest(),
            "tape_canonical_sha256": tape_canonical_sha,
            "tape_question_sha256": question_sha,
        },
        "incoming_ball": {
            "contact_center_w_m": _vector(
                source.get("ball_contact_w_m"), 3, "ball_contact_w_m"
            ),
            "velocity_w_mps": _vector(
                source.get("incoming_velocity_w_mps"), 3, "incoming_velocity_w_mps"
            ),
            "spin_w_radps": _vector(
                source.get("incoming_spin_w_radps"), 3, "incoming_spin_w_radps"
            ),
        },
        "base": {
            "spawn_w_m": _vector(source.get("base_spawn_w_m"), 3, "base_spawn_w_m"),
            "goal_w_m": _vector(source.get("base_goal_w_m"), 3, "base_goal_w_m"),
        },
        "landing_aim_w_xy_m": _vector(
            source.get("landing_aim_w_xy_m"), 2, "landing_aim_w_xy_m"
        ),
        "clock": {
            "reference_t_hit_s": reference_t_hit,
            "scaled_t_hit_s": t_hit,
            "pre_swing_wait_s": pre_wait,
            "time_to_contact_s": t_contact,
            "time_to_contact_tick": tick,
            "control_step_s": dt,
        },
    }
    return {**payload, "comparison_question_sha256": _digest(payload)}


def validate_comparison_question_receipt(value: Any) -> dict[str, Any]:
    receipt = dict(_mapping(value, "comparison question receipt"))
    digest = _sha(
        receipt.pop("comparison_question_sha256", None),
        "comparison_question_sha256",
    )
    if set(receipt) != {
        "schema_version",
        "kind",
        "identity",
        "incoming_ball",
        "base",
        "landing_aim_w_xy_m",
        "clock",
    } or receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("kind") != KIND:
        raise ComparisonQuestionError("comparison question receipt schema differs")
    identity = _mapping(receipt.get("identity"), "identity")
    if set(identity) != {
        "action_id",
        "action_uid",
        "action_slot",
        "teacher_id",
        "motion_sha256",
        "tape_row_index",
        "tape_file_sha256",
        "tape_canonical_sha256",
        "tape_question_sha256",
    }:
        raise ComparisonQuestionError("comparison question identity schema differs")
    _text(identity.get("action_id"), "identity.action_id")
    _integer(identity.get("action_uid"), "identity.action_uid", minimum=1)
    _integer(identity.get("action_slot"), "identity.action_slot")
    _text(identity.get("teacher_id"), "identity.teacher_id")
    _sha(identity.get("motion_sha256"), "identity.motion_sha256")
    if identity.get("tape_row_index") != 0:
        raise ComparisonQuestionError("comparison question tape row must be zero")
    for key in (
        "tape_file_sha256",
        "tape_canonical_sha256",
        "tape_question_sha256",
    ):
        _sha(identity.get(key), f"identity.{key}")
    incoming = _mapping(receipt.get("incoming_ball"), "incoming_ball")
    if set(incoming) != {"contact_center_w_m", "velocity_w_mps", "spin_w_radps"}:
        raise ComparisonQuestionError("comparison question incoming-ball schema differs")
    _vector(incoming.get("contact_center_w_m"), 3, "incoming_ball.contact_center_w_m")
    _vector(incoming.get("velocity_w_mps"), 3, "incoming_ball.velocity_w_mps")
    _vector(incoming.get("spin_w_radps"), 3, "incoming_ball.spin_w_radps")
    base = _mapping(receipt.get("base"), "base")
    if set(base) != {"spawn_w_m", "goal_w_m"}:
        raise ComparisonQuestionError("comparison question base schema differs")
    _vector(base.get("spawn_w_m"), 3, "base.spawn_w_m")
    _vector(base.get("goal_w_m"), 3, "base.goal_w_m")
    _vector(receipt.get("landing_aim_w_xy_m"), 2, "landing_aim_w_xy_m")
    clock = _mapping(receipt.get("clock"), "clock")
    if set(clock) != {
        "reference_t_hit_s",
        "scaled_t_hit_s",
        "pre_swing_wait_s",
        "time_to_contact_s",
        "time_to_contact_tick",
        "control_step_s",
    }:
        raise ComparisonQuestionError("comparison question clock schema differs")
    dt = _finite(clock.get("control_step_s"), "clock.control_step_s", positive=True)
    tick = _integer(clock.get("time_to_contact_tick"), "clock.time_to_contact_tick")
    contact = _finite(clock.get("time_to_contact_s"), "clock.time_to_contact_s", positive=True)
    hit = _finite(clock.get("scaled_t_hit_s"), "clock.scaled_t_hit_s", positive=True)
    wait = _finite(clock.get("pre_swing_wait_s"), "clock.pre_swing_wait_s")
    _finite(clock.get("reference_t_hit_s"), "clock.reference_t_hit_s", positive=True)
    if float(wait) < 0.0 or abs(float(tick) * float(dt) - float(contact)) > 2.0e-6 or abs(float(wait) + float(hit) - float(contact)) > 2.0e-6:
        raise ComparisonQuestionError("comparison question clock does not close")
    if _digest(receipt) != digest:
        raise ComparisonQuestionError("comparison_question_sha256 differs")
    return {**receipt, "comparison_question_sha256": digest}


def require_same_comparison_question(a: Any, c: Any) -> str:
    """Fail unless two independently supplied lineage receipts are identical."""

    left = validate_comparison_question_receipt(a)
    right = validate_comparison_question_receipt(c)
    if left != right:
        raise ComparisonQuestionError("A/C comparison questions are not identical")
    return left["comparison_question_sha256"]
