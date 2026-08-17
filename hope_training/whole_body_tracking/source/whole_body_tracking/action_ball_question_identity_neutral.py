#!/usr/bin/env python3
"""Engine-neutral deterministic identity checks for ActionBall questions.

This module intentionally owns no RNG, curriculum state, broker state, Torch,
NumPy, Isaac, or MuJoCo object.  It is the shared deterministic byte boundary
used by the existing Isaac sampler/broker/cache producers and by the MuJoCo GPU
question projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
QUESTION_HASH_SCHEMA_VERSION = 2
SAMPLER_SCHEMA_VERSION = 3
SAMPLER_BIRTH_DRAW_COUNT = 3
SAMPLER_SAMPLE_DRAW_COUNT = 18
ARM_CATALOG_SHA256 = (
    "2cbc6673119e0a816b0ee5081b403e5f4598437e4e8bf2eaa1e8a3db88f91d1b"
)
ARM_KEYS = (
    "time_to_contact_lower",
    "time_to_contact_upper",
    "contact_x_lower",
    "contact_x_upper",
    "contact_y_lower",
    "contact_y_upper",
    "contact_z_lower",
    "contact_z_upper",
    "incoming_speed_lower",
    "incoming_speed_upper",
    "spin_magnitude_lower",
    "spin_magnitude_upper",
    "base_spawn_x_lower",
    "base_spawn_x_upper",
    "base_spawn_y_lower",
    "base_spawn_y_upper",
    "base_travel_x_lower",
    "base_travel_x_upper",
    "base_travel_y_lower",
    "base_travel_y_upper",
    "landing_aim_x_lower",
    "landing_aim_x_upper",
    "landing_aim_y_lower",
    "landing_aim_y_upper",
    "incoming_direction_u_neg",
    "incoming_direction_u_pos",
    "incoming_direction_v_neg",
    "incoming_direction_v_pos",
    "spin_direction_u_neg",
    "spin_direction_u_pos",
    "spin_direction_v_neg",
    "spin_direction_v_pos",
)

_SAMPLE_BASE_KEYS = (
    "schema_version",
    "kind",
    "sampler_contract_sha256",
    "arm_catalog_sha256",
    "sample_index",
    "action_uid",
    "domain_epoch",
    "domain_levels",
    "birth_id",
    "profile_sha256",
    "levels_sha256",
    "draw_start",
    "draw_end",
    "mobility_mode",
    "base_yaw_rad",
    "base_start_w_m",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "base_goal_w_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "contact_w_m",
    "time_to_contact_s",
    "incoming_speed_mps",
    "incoming_direction_b_yaw",
    "incoming_direction_w",
    "incoming_velocity_w_mps",
    "spin_magnitude_radps",
    "spin_direction_b_yaw",
    "spin_direction_w",
    "spin_w_radps",
    "landing_aim_w_xy_m",
)
_SAMPLE_MIXTURE_KEYS = (
    "birth_index",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "contact_time_step_s",
    "time_to_contact_tick",
)
_BIRTH_BASE_KEYS = (
    "schema_version",
    "runtime_contract_sha256",
    "registry_sha256",
    "env_id",
    "reset_generation",
    "action_uid",
    "action_slot",
    "domain_epoch",
    "domain_claim_sha256",
    "domain_authority_sha256",
    "domain_levels",
    "arm_catalog_sha256",
    "levels_sha256",
    "sampler_birth_sha256",
    "sampler_birth_index",
    "sampler_draw_start",
    "sampler_draw_end",
    "mobility_mode",
    "base_yaw_rad",
    "base_quat_wxyz",
    "base_spawn_w_m",
    "manifest_sha256",
    "sampler_sha256",
    "profile_sha256",
    "motion_sha256",
    "physics_sha256",
    "solver_sha256",
)
_BIRTH_MIXTURE_KEYS = (
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "initial_center_single_question",
)
_SEMANTIC_QUESTION_KEYS = (
    "action_uid",
    "action_slot",
    "domain_epoch",
    "domain_levels",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "base_yaw_rad",
    "base_quat_wxyz",
    "base_spawn_w_m",
    "base_goal_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_w_m",
    "time_to_contact_s",
    "incoming_velocity_w_mps",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "manifest_sha256",
    "profile_sha256",
    "motion_sha256",
    "physics_sha256",
    "solver_sha256",
    "mount_normal_sign",
)


def canonical_json_sha256(value: object) -> str:
    """Return the repository's canonical JSON SHA-256 bytes."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def exact_question_sha256(payload: Mapping[str, object]) -> str:
    """Hash one complete semantic question with the canonical cache wrapper."""

    if not isinstance(payload, Mapping):
        raise TypeError("question payload must be a mapping")
    return canonical_json_sha256(
        {
            "schema_version": QUESTION_HASH_SCHEMA_VERSION,
            "kind": "action_ball_exact_curriculum_question",
            "payload": dict(payload),
        }
    )


def verify_sample_identity_hash(
    sample_id: object, identity_payload: Mapping[str, object]
) -> None:
    """Shared byte check used by ``BallBaseSample.verify_sample_id``."""

    declared = _sha256(sample_id, label="sample_id")
    if not isinstance(identity_payload, Mapping):
        raise TypeError("sample identity payload must be a mapping")
    if canonical_json_sha256(dict(identity_payload)) != declared:
        raise ValueError("sample_id does not match canonical identity")


def verify_canonical_receipt_hash(
    receipt: Mapping[str, object], *, digest_key: str, label: str
) -> None:
    """Verify a canonical JSON envelope without interpreting its state machine."""

    if not isinstance(receipt, Mapping):
        raise TypeError("%s must be a mapping" % label)
    if digest_key not in receipt:
        raise ValueError("%s is missing %s" % (label, digest_key))
    declared = _sha256(receipt[digest_key], label="%s.%s" % (label, digest_key))
    payload = {key: value for key, value in receipt.items() if key != digest_key}
    if canonical_json_sha256(payload) != declared:
        raise ValueError("%s canonical SHA differs" % label)


def _exact_mapping(
    value: object, expected_keys: Sequence[str], *, label: str
) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % label)
    expected = set(expected_keys)
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "%s keys differ: missing=%s unknown=%s"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )
    return dict(value)


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("%s must be a lowercase SHA-256" % label)
    return value


def _plain_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("%s must be an exact int >= %d" % (label, minimum))
    return value


def _finite(value: object, *, label: str, minimum=None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("%s must be finite" % label)
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError("%s is below its minimum" % label)
    return result


def _vector(value: object, width: int, *, label: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError("%s must be a width-%d row" % (label, width))
    return tuple(_finite(item, label="%s[]" % label) for item in value)


def _levels(value: object, *, label: str, all_zero: bool) -> dict:
    row = _exact_mapping(value, ARM_KEYS, label=label)
    result = {
        key: _finite(row[key], label="%s.%s" % (label, key), minimum=0.0)
        for key in ARM_KEYS
    }
    if all_zero and any(amount != 0.0 for amount in result.values()):
        raise ValueError("%s must be the exact all-zero center" % label)
    return result


def verify_sample_identity_receipt(value: object) -> dict:
    """Verify complete mixed-sampler ``BallBaseSample`` identity bytes."""

    row = _exact_mapping(
        value,
        ("sample_id",) + _SAMPLE_BASE_KEYS + _SAMPLE_MIXTURE_KEYS,
        label="BallBaseSample identity receipt",
    )
    sample_id = _sha256(row.pop("sample_id"), label="sample_id")
    if row["schema_version"] != SAMPLER_SCHEMA_VERSION:
        raise ValueError("sample schema_version differs")
    if row["kind"] != "swing_sample":
        raise ValueError("sample kind differs")
    verify_sample_identity_hash(sample_id, row)
    _sha256(row["sampler_contract_sha256"], label="sampler_contract_sha256")
    if row["arm_catalog_sha256"] != ARM_CATALOG_SHA256:
        raise ValueError("sample arm catalog differs")
    _plain_int(row["sample_index"], label="sample_index")
    _plain_int(row["birth_index"], label="birth_index")
    _plain_int(row["action_uid"], label="action_uid", minimum=1)
    _plain_int(row["domain_epoch"], label="domain_epoch")
    _levels(row["domain_levels"], label="sample.domain_levels", all_zero=True)
    _levels(
        row["birth_sampling_levels"],
        label="sample.birth_sampling_levels",
        all_zero=True,
    )
    _levels(
        row["sampling_levels"],
        label="sample.sampling_levels",
        all_zero=True,
    )
    for label in ("birth_id", "profile_sha256", "levels_sha256"):
        _sha256(row[label], label=label)
    draw_start = _plain_int(row["draw_start"], label="sample.draw_start")
    draw_end = _plain_int(row["draw_end"], label="sample.draw_end")
    if draw_end - draw_start != SAMPLER_SAMPLE_DRAW_COUNT:
        raise ValueError("sample RNG draw range differs")
    if row["sampling_stratum"] != "center" or row["frontier_arm"] is not None:
        raise ValueError("sample is not the literal center stratum")
    if (
        row["birth_sampling_stratum"] != "center"
        or row["birth_frontier_arm"] is not None
    ):
        raise ValueError("sample birth is not the literal center stratum")
    if row["mobility_mode"] not in ("no_move", "move"):
        raise ValueError("sample mobility mode differs")
    for label in (
        "base_start_w_m",
        "base_spawn_latent_w_m",
        "base_travel_latent_b_yaw_m",
        "base_goal_w_m",
        "contact_offset_from_base_goal_b_yaw_m",
        "contact_w_m",
        "incoming_direction_b_yaw",
        "incoming_direction_w",
        "incoming_velocity_w_mps",
        "spin_direction_b_yaw",
        "spin_direction_w",
        "spin_w_radps",
    ):
        _vector(row[label], 3, label="sample.%s" % label)
    _vector(row["landing_aim_w_xy_m"], 2, label="sample.landing_aim_w_xy_m")
    _finite(row["base_yaw_rad"], label="sample.base_yaw_rad")
    for label in (
        "time_to_contact_s",
        "contact_time_step_s",
        "incoming_speed_mps",
        "spin_magnitude_radps",
    ):
        _finite(row[label], label="sample.%s" % label, minimum=0.0)
    tick = _plain_int(
        row["time_to_contact_tick"], label="time_to_contact_tick", minimum=1
    )
    if row["time_to_contact_s"] != tick * row["contact_time_step_s"]:
        raise ValueError("sample TTC seconds/tick/step identity differs")
    return {"sample_id": sample_id, **row}


def verify_birth_receipt(value: object) -> dict:
    """Verify complete initial-center ``ActionBirthReceipt`` identity bytes."""

    row = _exact_mapping(
        value,
        _BIRTH_BASE_KEYS + _BIRTH_MIXTURE_KEYS + ("canonical_sha256",),
        label="ActionBirthReceipt",
    )
    declared = _sha256(row.pop("canonical_sha256"), label="birth canonical_sha256")
    if canonical_json_sha256(row) != declared:
        raise ValueError("birth canonical SHA differs")
    if row["schema_version"] != 3:
        raise ValueError("birth schema_version differs")
    _sha256(row["runtime_contract_sha256"], label="runtime_contract_sha256")
    _sha256(row["registry_sha256"], label="registry_sha256")
    _plain_int(row["env_id"], label="birth.env_id")
    _plain_int(row["reset_generation"], label="birth.reset_generation", minimum=1)
    _plain_int(row["action_uid"], label="birth.action_uid", minimum=1)
    _plain_int(row["action_slot"], label="birth.action_slot")
    _plain_int(row["domain_epoch"], label="birth.domain_epoch")
    _plain_int(row["sampler_birth_index"], label="sampler_birth_index")
    start = _plain_int(row["sampler_draw_start"], label="sampler_draw_start")
    end = _plain_int(row["sampler_draw_end"], label="sampler_draw_end")
    if end - start != SAMPLER_BIRTH_DRAW_COUNT:
        raise ValueError("birth RNG draw range differs")
    if row["arm_catalog_sha256"] != ARM_CATALOG_SHA256:
        raise ValueError("birth arm catalog differs")
    for label in (
        "domain_claim_sha256",
        "domain_authority_sha256",
        "levels_sha256",
        "sampler_birth_sha256",
        "manifest_sha256",
        "sampler_sha256",
        "profile_sha256",
        "motion_sha256",
        "physics_sha256",
        "solver_sha256",
    ):
        _sha256(row[label], label="birth.%s" % label)
    _levels(row["domain_levels"], label="birth.domain_levels", all_zero=True)
    _levels(row["sampling_levels"], label="birth.sampling_levels", all_zero=True)
    if row["initial_center_single_question"] is not True:
        raise ValueError("birth lacks initial-center single-question brand")
    if row["sampling_stratum"] != "center" or row["frontier_arm"] is not None:
        raise ValueError("birth is not the literal center stratum")
    _finite(row["base_yaw_rad"], label="birth.base_yaw_rad")
    _vector(row["base_quat_wxyz"], 4, label="birth.base_quat_wxyz")
    _vector(row["base_spawn_w_m"], 3, label="birth.base_spawn_w_m")
    return {"canonical_sha256": declared, **row}


def verify_semantic_question(
    payload: object, declared_sha256: object
) -> dict:
    """Verify the complete exact-question payload and its cache digest."""

    row = _exact_mapping(payload, _SEMANTIC_QUESTION_KEYS, label="semantic question")
    declared = _sha256(declared_sha256, label="semantic question SHA")
    if exact_question_sha256(row) != declared:
        raise ValueError("semantic question SHA differs")
    return row


def verify_fixed_n1_question_row(
    *,
    sample_identity_receipt: object,
    birth_receipt: object,
    semantic_question_payload: object,
    semantic_question_sha256: object,
    expected_action_uid: int,
    expected_action_slot: int,
    expected_sampler_sha256: str,
    expected_broker_registry_sha256: str,
    expected_profile_sha256: str,
    expected_manifest_sha256: str,
    expected_motion_sha256: str,
    expected_physics_sha256: str,
    expected_solver_sha256: str,
    mount_normal_sign: int,
) -> dict:
    """Cross-bind one canonical sample, broker birth, and solver question."""

    sample = verify_sample_identity_receipt(sample_identity_receipt)
    birth = verify_birth_receipt(birth_receipt)
    question = verify_semantic_question(
        semantic_question_payload, semantic_question_sha256
    )
    if type(mount_normal_sign) is not int or mount_normal_sign not in (-1, 1):
        raise ValueError("mount_normal_sign must be -1 or +1")
    expected_pins = (
        expected_action_uid,
        expected_action_slot,
        _sha256(expected_sampler_sha256, label="expected sampler SHA"),
        _sha256(
            expected_broker_registry_sha256,
            label="expected broker registry SHA",
        ),
        _sha256(expected_profile_sha256, label="expected profile SHA"),
        _sha256(expected_manifest_sha256, label="expected manifest SHA"),
        _sha256(expected_motion_sha256, label="expected motion SHA"),
        _sha256(expected_physics_sha256, label="expected physics SHA"),
        _sha256(expected_solver_sha256, label="expected solver SHA"),
    )
    observed_pins = (
        birth["action_uid"],
        birth["action_slot"],
        birth["sampler_sha256"],
        birth["registry_sha256"],
        birth["profile_sha256"],
        birth["manifest_sha256"],
        birth["motion_sha256"],
        birth["physics_sha256"],
        birth["solver_sha256"],
    )
    if observed_pins != expected_pins:
        raise ValueError("birth authority pins differ from H1 authority")
    relations = (
        (sample["action_uid"], birth["action_uid"], "sample action_uid"),
        (sample["domain_epoch"], birth["domain_epoch"], "sample domain epoch"),
        (sample["domain_levels"], birth["domain_levels"], "sample domain levels"),
        (sample["levels_sha256"], birth["levels_sha256"], "sample levels SHA"),
        (sample["birth_id"], birth["sampler_birth_sha256"], "sampler birth SHA"),
        (sample["sampler_contract_sha256"], birth["sampler_sha256"], "sampler SHA"),
        (sample["profile_sha256"], birth["profile_sha256"], "profile SHA"),
        (sample["base_yaw_rad"], birth["base_yaw_rad"], "base yaw"),
        (sample["base_start_w_m"], birth["base_spawn_w_m"], "base start"),
    )
    for left, right, label in relations:
        if left != right:
            raise ValueError("%s relation differs" % label)
    expected_question = {
        "action_uid": birth["action_uid"],
        "action_slot": birth["action_slot"],
        "domain_epoch": birth["domain_epoch"],
        "domain_levels": birth["domain_levels"],
        "birth_sampling_stratum": sample["birth_sampling_stratum"],
        "birth_sampling_levels": sample["birth_sampling_levels"],
        "birth_frontier_arm": sample["birth_frontier_arm"],
        "sampling_stratum": sample["sampling_stratum"],
        "sampling_levels": sample["sampling_levels"],
        "frontier_arm": sample["frontier_arm"],
        "base_yaw_rad": birth["base_yaw_rad"],
        "base_quat_wxyz": birth["base_quat_wxyz"],
        "base_spawn_w_m": birth["base_spawn_w_m"],
        "base_goal_w_m": sample["base_goal_w_m"],
        "base_travel_latent_b_yaw_m": sample[
            "base_travel_latent_b_yaw_m"
        ],
        "contact_w_m": sample["contact_w_m"],
        "time_to_contact_s": sample["time_to_contact_s"],
        "incoming_velocity_w_mps": sample["incoming_velocity_w_mps"],
        "incoming_spin_w_radps": sample["spin_w_radps"],
        "landing_aim_w_xy_m": sample["landing_aim_w_xy_m"],
        "manifest_sha256": birth["manifest_sha256"],
        "profile_sha256": birth["profile_sha256"],
        "motion_sha256": birth["motion_sha256"],
        "physics_sha256": birth["physics_sha256"],
        "solver_sha256": birth["solver_sha256"],
        "mount_normal_sign": mount_normal_sign,
    }
    if question != expected_question:
        raise ValueError("semantic question differs from sample/birth identity")
    return {
        "env_id": birth["env_id"],
        "reset_generation": birth["reset_generation"],
        "action_uid": birth["action_uid"],
        "action_slot": birth["action_slot"],
        "sample_id": sample["sample_id"],
        "broker_birth_sha256": birth["canonical_sha256"],
        "sampler_birth_sha256": birth["sampler_birth_sha256"],
        "semantic_question_sha256": _sha256(
            semantic_question_sha256, label="semantic question SHA"
        ),
        "sample": sample,
        "birth": birth,
        "question": question,
    }
