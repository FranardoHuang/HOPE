#!/usr/bin/env python3
"""Build C211 oracle sidecars only from observed runtime episode facts.

This module is deliberately independent from the launcher and from Hydra.  The
C211 zero-PPO runner-before-oracle hook hands it the exact per-episode facts
after stepping the real environment.  It never accepts a producer-authored PASS or a
precomputed selected-rubber classification: classification and every content
SHA are derived here, and publication is canonical/no-clobber.

The helper is not an authorization surface.  The C211 launcher remains the
consumer and derives the oracle verdict after revalidating these artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


RAW_KIND = "action_ball_c211_oracle_raw_evidence_v3"
PREFLIGHT_KIND = "action_ball_c211_runner_preflight_evidence_v1"
SELECTED_KIND = "action_ball_c211_selected_rubber_contact_evidence_v1"
INPUT_KIND = "action_ball_c211_observed_oracle_bundle_v3"
OUTPUT_KIND = "action_ball_c211_oracle_evidence_publication_v3"
OBSERVED_BUNDLE_MARKER = "ACTION_BALL_C211_OBSERVED_ORACLE_BUNDLE_JSON"
CLASSIFIER_CONTRACT = "runtime_contact_pair_selected_rubber_v1"
PREFLIGHT_MARKER = "ACTION_BALL_C211_TRAINABILITY_PREFLIGHT_JSON"

ACTOR_CONTRACT = "action_ball_c211"
CRITIC_CONTRACT = "action_ball_c211_critic_v1"
TRAINABILITY_CONTRACT = "action_ball_c211_fixed_midpoint_learnability_v2"
ACTOR_NORMALIZER_IDENTITY = "action_ball_c211_actor_norm_v2"
CRITIC_NORMALIZER_IDENTITY = "action_ball_c211_critic_norm_v1"
ACTOR_WIDTH = 211
CRITIC_WIDTH = 319
EPISODES = 32
INCOMING_FIELDS = (
    "incoming_ball_contact_position_heading",
    "incoming_ball_contact_velocity_heading",
    "incoming_ball_contact_spin_heading",
)
HARD_TERMINATIONS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
OBSERVED_CONTACT_KEYS = (
    "episode",
    "runtime_control_step",
    "task_valid",
    "eligible_closed_swing",
    "exact_strike",
    "selected_face_sweep_contact",
    "selected_face_bracketed",
    "selected_face_edge_safe",
    "selected_face_geometry_finite",
    "selected_face_closing_speed_positive",
    "selected_face_normal_speed_consistent",
    "wrong_surface_contact",
    "edge_or_rim_ambiguous",
    "between_planes_ambiguous",
)
SHA_KEYS = (
    "oracle_launch_claim_sha256",
    "lineage_sha256",
    "recipe_contract_sha256",
    "reward_contract_sha256",
    "runtime_effective_reward_sha256",
    "runtime_policy_recipe_sha256",
    "hard_contract_sha256",
    "motion_sha256",
    "manifest_sha256",
    "dynamic_ready_artifact_sha256",
    "dynamic_ready_nominal_receipt_sha256",
)
QUESTION_RNG = {
    "owner": "runtime_curriculum_sampler",
    "cadence": "every_episode_reset",
    "draw_count_authority": "sample_receipt_draw_end_minus_draw_start",
    "zero_draw_claim_permitted": False,
    "checkpoint_resume": "exact_sampler_and_curriculum_state",
}
TASK_WAIT_CONTRACT = {
    "identity": "action_ball_pre_task_wait_schedule_v1",
    "policy_dt_s": 0.02,
    "seed": 20260804,
    "min_wait_ticks": 5,
    "max_wait_ticks": 25,
    "episode_horizon_ticks": 500,
    "required_active_ticks": 200,
    "schedule_canonical_sha256": (
        "58aa7bb62406d301df619caf7026af8d595f4b8cd9594ea8441b4c89997d400e"
    ),
    "task_valid_actor_and_critic": True,
    "wait_task_ball_base_and_clocks_masked": True,
    "wait_remaining_observed": False,
}
QUESTION_SOURCE_CONTRACT = {
    "identity": "action_ball_211_question_source_scope_v5",
    "family": "C211",
    "question_sampler": {
        "source": "runtime_curriculum_sampler",
        "cadence": "every_episode_reset",
        "curriculum_domain_levels_consulted_every_reset": True,
        "sampler_runs_every_reset": True,
        "initial_center_single_question": True,
        "initial_center_activation": "all_32_domain_levels_exact_zero",
        "initial_center_physical_support": "literal_profile_center_point",
        "initial_center_rng_draws": "normal_fixed_budget",
        "post_promotion_support": "zero_to_manifest_max_width_per_promoted_arm",
        "sampler_rng_reused_by_target_provider": False,
        "physical_rng_draw_count_authority": (
            "sample_receipt_draw_end_minus_draw_start"
        ),
        "zero_physical_rng_draw_claim_permitted": False,
        "selection": "sample_current_domain_levels",
        "checkpoint_resume": "exact_sampler_and_curriculum_state",
    },
    "target_provider": {
        "source": "direct_ball",
        "desired_contact_inverse": False,
        "exact_question_answer_cache": {"enabled": False},
        "online_inverse_solves_per_reset": 0,
        "online_inverse_solves_per_step": 0,
    },
}


class EvidenceError(RuntimeError):
    """Fail-closed C211 evidence construction error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceError("%s must be exact lowercase SHA-256" % name)
    return value


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise EvidenceError("%s keys differ" % name)
    return dict(value)


def _plain_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError("%s must be a plain integer >= %d" % (name, minimum))
    return value


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise EvidenceError("%s must be finite" % name)
    result = float(value)
    if minimum is not None and result < minimum:
        raise EvidenceError("%s is below its minimum" % name)
    return result


def _stable_regular(path: Path, *, name: str) -> None:
    try:
        first = path.lstat()
        second = path.stat()
    except OSError as exc:
        raise EvidenceError("%s is unavailable" % name) from exc
    if (
        not stat.S_ISREG(first.st_mode)
        or stat.S_ISLNK(first.st_mode)
        or (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino)
    ):
        raise EvidenceError("%s must be one real regular file" % name)


def _sealed(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "content_sha256": canonical_sha256(row)}


def _geometry_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    )
    _stable_regular(source, name="C211 geometry authority")
    spec = importlib.util.spec_from_file_location(
        "_c211_oracle_racket_contact_geometry", source
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise EvidenceError("cannot import C211 geometry authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry_authority_sha256() -> str:
    module = _geometry_module()
    return _sha(
        getattr(module, "GEOMETRY_SOURCE_SHA256", None),
        name="geometry authority SHA",
    )


def classifier_source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def classify_observed_contact(value: Any, *, episode: int) -> tuple[str, str]:
    """Classify one runtime observation; never consume a verdict string."""

    row = _exact_dict(value, OBSERVED_CONTACT_KEYS, name="observed contact")
    if row["episode"] != episode:
        raise EvidenceError("observed contact episode order differs")
    _plain_int(row["runtime_control_step"], name="runtime control step", minimum=1)
    for key in OBSERVED_CONTACT_KEYS[2:]:
        if type(row[key]) is not bool:
            raise EvidenceError("observed contact %s must be an exact bool" % key)
    if not row["eligible_closed_swing"]:
        raise EvidenceError("C211 oracle evidence requires an eligible closed swing")

    selected = all(
        row[key]
        for key in (
            "task_valid",
            "exact_strike",
            "selected_face_sweep_contact",
            "selected_face_bracketed",
            "selected_face_edge_safe",
            "selected_face_geometry_finite",
            "selected_face_closing_speed_positive",
            "selected_face_normal_speed_consistent",
        )
    )
    competing = (
        row["wrong_surface_contact"],
        row["edge_or_rim_ambiguous"],
        row["between_planes_ambiguous"],
    )
    if sum(bool(flag) for flag in competing) > 1 or (selected and any(competing)):
        raise EvidenceError("observed contact classifications overlap")
    if row["between_planes_ambiguous"]:
        classification = "between_planes_ambiguous"
    elif row["edge_or_rim_ambiguous"]:
        classification = "edge_or_rim_ambiguous"
    elif row["wrong_surface_contact"]:
        classification = "wrong_surface"
    elif not row["selected_face_geometry_finite"]:
        classification = "unknown"
    elif selected:
        classification = "selected_rubber"
    else:
        classification = "no_contact"
    return classification, canonical_sha256(row)


def _validate_runner_facts(value: Any) -> dict[str, Any]:
    keys = (
        "trainability_contract",
        "actor_contract",
        "critic_contract",
        "actor_width",
        "critic_width",
        "actor_normalizer_identity",
        "critic_normalizer_identity",
        "fresh_normalizers_required",
        "symmetric_critic_fallback_forbidden",
        "task_valid_required",
        "task_wait_contract",
        "question_source_contract",
        "contact_target_absent",
        "c225_reward_contract",
        "runner_actor_width",
        "runner_critic_width",
        "actor_normalizer_attribute",
        "critic_normalizer_attribute",
    )
    facts = _exact_dict(value, keys, name="runner preflight facts")
    expected = {
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_contract": ACTOR_CONTRACT,
        "critic_contract": CRITIC_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_width": CRITIC_WIDTH,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "task_valid_required": True,
        "task_wait_contract": TASK_WAIT_CONTRACT,
        "question_source_contract": QUESTION_SOURCE_CONTRACT,
        "contact_target_absent": True,
        "runner_actor_width": ACTOR_WIDTH,
        "runner_critic_width": CRITIC_WIDTH,
    }
    if any(facts[key] != wanted for key, wanted in expected.items()):
        raise EvidenceError("runner preflight C211 ABI differs")
    actor_attr = facts["actor_normalizer_attribute"]
    critic_attr = facts["critic_normalizer_attribute"]
    if (
        type(actor_attr) is not str
        or not actor_attr
        or type(critic_attr) is not str
        or not critic_attr
        or actor_attr == critic_attr
    ):
        raise EvidenceError("runner preflight lacks distinct normalizer attributes")
    reward = facts["c225_reward_contract"]
    if (
        type(reward) is not dict
        or reward.get("identity") != "action_ball_c211_achieved_outcome_reward_v3"
        or reward.get("task_valid_required") is not True
        or type(reward.get("strike_bridge")) is not dict
        or reward["strike_bridge"].get("weight") != 240.0
        or reward.get("landing", {}).get("weight") != 700.0
        or reward["strike_bridge"].get("eligibility")
        != "task_valid_active_swing_single_exact_strike_tick"
    ):
        raise EvidenceError("runner preflight reward-v3/task_valid contract differs")
    return facts


def build_runner_preflight(
    *, launch_claim_sha256: str, hard_contract_sha256: str, facts: Any
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "kind": PREFLIGHT_KIND,
        "diagnostic_unauthorized": True,
        "oracle_launch_claim_sha256": _sha(
            launch_claim_sha256, name="oracle launch claim SHA"
        ),
        "hard_contract_sha256": _sha(
            hard_contract_sha256, name="hard contract SHA"
        ),
        "marker": PREFLIGHT_MARKER,
        "facts": _validate_runner_facts(facts),
    }
    return _sealed(unsigned)


def build_selected_rubber(
    *,
    launch_claim_sha256: str,
    action_id: str,
    action_uid: int,
    motion_sha256: str,
    observed_contacts: Sequence[Any],
) -> tuple[dict[str, Any], list[str]]:
    if type(action_id) is not str or not action_id:
        raise EvidenceError("action_id must be a non-empty string")
    _plain_int(action_uid, name="action_uid", minimum=1)
    if type(observed_contacts) not in (list, tuple) or len(observed_contacts) != EPISODES:
        raise EvidenceError("selected-rubber evidence requires exactly 32 observations")
    rows = []
    row_sha256 = []
    for episode, observed in enumerate(observed_contacts):
        classification, evidence_sha = classify_observed_contact(
            observed, episode=episode
        )
        row = {
            "episode": episode,
            "eligible_closed_swing": True,
            "classification": classification,
            "contact_evidence_sha256": evidence_sha,
        }
        rows.append(row)
        row_sha256.append(canonical_sha256(row))
    unsigned = {
        "schema_version": 1,
        "kind": SELECTED_KIND,
        "diagnostic_unauthorized": True,
        "oracle_launch_claim_sha256": _sha(
            launch_claim_sha256, name="oracle launch claim SHA"
        ),
        "action_id": action_id,
        "action_uid": action_uid,
        "motion_sha256": _sha(motion_sha256, name="motion SHA"),
        "classifier_contract": CLASSIFIER_CONTRACT,
        "classifier_source_sha256": classifier_source_sha256(),
        "geometry_authority_sha256": geometry_authority_sha256(),
        "denominator_kind": "eligible_closed_swings",
        "episodes": rows,
    }
    return _sealed(unsigned), row_sha256


def _validate_question(value: Any) -> dict[str, Any]:
    keys = (
        "target_source",
        "question_source",
        "target_recipe",
        "target_validity_mask",
        "target_observation_noise",
        "incoming_ball_fields",
        "desired_contact_fields_observed",
        "reset_inverse_solve",
        "online_solver_calls",
        "online_lm_calls",
        "question_rng",
    )
    row = _exact_dict(value, keys, name="question contract")
    expected = {
        "target_source": "direct_ball",
        "question_source": "runtime_curriculum_sampler",
        "target_recipe": "outcome_dense_only",
        "target_validity_mask": [False, False, False],
        "target_observation_noise": False,
        "incoming_ball_fields": list(INCOMING_FIELDS),
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "question_rng": QUESTION_RNG,
    }
    if row != expected:
        raise EvidenceError("question contract is not C211 runtime-sampled direct-ball")
    return row


def _finite_vec3(value: Any, *, name: str) -> list[float | int]:
    if type(value) is not list or len(value) != 3:
        raise EvidenceError("%s must be a length-3 list" % name)
    for component in value:
        _finite_number(component, name=name)
    return list(value)


def _finite_vec2(value: Any, *, name: str) -> list[float | int]:
    if type(value) is not list or len(value) != 2:
        raise EvidenceError("%s must be a length-2 list" % name)
    for component in value:
        _finite_number(component, name=name)
    return list(value)


def _validate_achieved_analytic_flight(value: Any, *, selected: bool) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "evaluated", "finite", "landing_xy_m", "landing_valid",
            "net_crossed", "net_clear", "on_opponent_table", "source",
        ),
        name="achieved analytic flight",
    )
    for key in (
        "evaluated", "finite", "landing_valid", "net_crossed", "net_clear",
        "on_opponent_table",
    ):
        if type(row[key]) is not bool:
            raise EvidenceError("achieved analytic flight %s must be bool" % key)
    if row["evaluated"] is not selected:
        raise EvidenceError("analytic flight evaluation differs from selected contact")
    if selected:
        if (
            row["source"]
            != "runtime_vb_one_shot_from_achieved_selected_rubber_contact"
            or row["finite"] is not True
        ):
            raise EvidenceError("analytic flight authority/finite gate differs")
        row["landing_xy_m"] = _finite_vec2(
            row["landing_xy_m"], name="achieved analytic landing"
        )
        if row["net_clear"] and not row["net_crossed"]:
            raise EvidenceError("analytic net-clear cannot precede net crossing")
        if row["on_opponent_table"] and not row["landing_valid"]:
            raise EvidenceError("opponent-table outcome requires valid analytic landing")
    elif (
        row["finite"] or row["landing_xy_m"] is not None
        or row["landing_valid"] or row["net_crossed"] or row["net_clear"]
        or row["on_opponent_table"] or row["source"] is not None
    ):
        raise EvidenceError("no-contact row carries hypothetical analytic flight")
    return row


def _validate_predicted_outcome(
    value: Any, *, selected: bool, flight: Mapping[str, Any]
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "evaluated", "predicted_net_clear", "predicted_legal_landing",
            "predicted_landing_xy_m", "source",
        ),
        name="predicted analytic outcome",
    )
    if type(row["evaluated"]) is not bool or row["evaluated"] is not selected:
        raise EvidenceError("predicted outcome evaluation differs from selected contact")
    if selected:
        if row["source"] != "runtime_c225_achieved_flight_prediction_one_shot":
            raise EvidenceError("predicted outcome authority differs")
        if type(row["predicted_net_clear"]) is not bool or type(
            row["predicted_legal_landing"]
        ) is not bool:
            raise EvidenceError("predicted outcome booleans differ")
        row["predicted_landing_xy_m"] = _finite_vec2(
            row["predicted_landing_xy_m"], name="predicted analytic landing"
        )
        expected_legal = bool(
            flight["landing_valid"] and flight["net_crossed"]
            and flight["net_clear"] and flight["on_opponent_table"]
        )
        if (
            row["predicted_landing_xy_m"] != flight["landing_xy_m"]
            or row["predicted_net_clear"] is not flight["net_clear"]
            or row["predicted_legal_landing"] is not expected_legal
        ):
            raise EvidenceError("predicted outcome differs from achieved analytic flight")
    elif any(
        row[key] is not None
        for key in (
            "predicted_net_clear", "predicted_legal_landing",
            "predicted_landing_xy_m", "source",
        )
    ):
        raise EvidenceError("no-contact row carries hypothetical predicted outcome")
    return row


def _validate_episode(value: Any, *, episode: int, selected_row_sha256: str) -> dict[str, Any]:
    keys = (
        "episode",
        "control_steps",
        "terminal_phase",
        "termination_reasons",
        "sampler_sample_index",
        "sampler_sample_sha256",
        "sampler_draw_start",
        "sampler_draw_end",
        "incoming_ball_observation",
        "observed_selected_rubber_contact",
        "achieved_analytic_flight",
        "predicted_outcome",
        "safety",
        "teacher_qdes",
    )
    row = _exact_dict(value, keys, name="runtime oracle episode")
    if row["episode"] != episode:
        raise EvidenceError("runtime oracle episode order differs")
    _plain_int(row["control_steps"], name="episode control steps", minimum=1)
    _plain_int(row["sampler_sample_index"], name="sampler sample index", minimum=0)
    _sha(row["sampler_sample_sha256"], name="sampler sample SHA")
    draw_start = _plain_int(
        row["sampler_draw_start"], name="sampler draw start", minimum=0
    )
    draw_end = _plain_int(
        row["sampler_draw_end"], name="sampler draw end", minimum=1
    )
    if draw_end <= draw_start:
        raise EvidenceError("sampler receipt must prove positive physical RNG draws")
    incoming = _exact_dict(
        row["incoming_ball_observation"],
        ("source", "actor", "critic"),
        name="incoming-ball observation",
    )
    if incoming["source"] != "runtime_actor_and_critic_observation_terms":
        raise EvidenceError("incoming-ball source is not the runtime observation terms")
    actor = _exact_dict(incoming["actor"], INCOMING_FIELDS, name="actor incoming ball")
    critic = _exact_dict(incoming["critic"], INCOMING_FIELDS, name="critic incoming ball")
    for field in INCOMING_FIELDS:
        actor[field] = _finite_vec3(actor[field], name="actor %s" % field)
        critic[field] = _finite_vec3(critic[field], name="critic %s" % field)
    if actor != critic:
        raise EvidenceError("actor/critic incoming-ball rows differ")
    _classification, evidence_sha = classify_observed_contact(
        row["observed_selected_rubber_contact"], episode=episode
    )
    if canonical_sha256(
        {
            "episode": episode,
            "eligible_closed_swing": True,
            "classification": _classification,
            "contact_evidence_sha256": evidence_sha,
        }
    ) != selected_row_sha256:
        raise EvidenceError("selected-rubber row binding differs")
    selected = _classification == "selected_rubber"
    flight = _validate_achieved_analytic_flight(
        row["achieved_analytic_flight"], selected=selected
    )
    _validate_predicted_outcome(
        row["predicted_outcome"], selected=selected, flight=flight
    )
    safety = _exact_dict(
        row["safety"],
        (
            "hard_termination_by_reason",
            "robot_table_contact_count",
            "projection_nonfinite_count",
            "projection_observed_sample_count",
            "qdes_observed_sample_count",
            "actual_observed_sample_count",
            "reference_guard_sample_count",
        ),
        name="episode safety",
    )
    hard = _exact_dict(
        safety["hard_termination_by_reason"], HARD_TERMINATIONS,
        name="episode hard terminations",
    )
    for key, count in {**hard, **{k: safety[k] for k in safety if k != "hard_termination_by_reason"}}.items():
        _plain_int(count, name="episode safety %s" % key)
    teacher = _exact_dict(
        row["teacher_qdes"],
        ("preclamp_max_abs_error_rad", "teleport_used"),
        name="episode teacher qdes",
    )
    _finite_number(
        teacher["preclamp_max_abs_error_rad"],
        name="teacher qdes error", minimum=0.0,
    )
    if teacher["teleport_used"] is not False:
        raise EvidenceError("teacher-qdes oracle must not teleport")
    return row


def validate_rollout_census(value: Any, *, closed_attempts: int) -> dict[str, int]:
    """Require the WAIT-only exclusion denominator to close over the rollout.

    The 32 closed attempts are not the whole run: a reset that dies inside the
    hidden WAIT never becomes an attempt and never reaches this evidence.  That
    exclusion is legitimate, but it must stay countable, otherwise a clean-looking
    32-episode census can hide an arbitrarily large pile of discarded resets.
    """

    row = _exact_dict(
        value,
        (
            "source_episodes_consumed",
            "wait_only_reset_excluded",
            "closed_attempts",
        ),
        name="rollout census",
    )
    counts = {
        key: _plain_int(row[key], name="rollout census " + key) for key in row
    }
    if (
        counts["closed_attempts"] != closed_attempts
        or counts["source_episodes_consumed"]
        != counts["closed_attempts"] + counts["wait_only_reset_excluded"]
    ):
        raise EvidenceError("rollout census does not close over its own resets")
    return counts


def build_raw_oracle(
    *,
    bindings: Any,
    training_contract_artifact: Mapping[str, str],
    runner_preflight_artifact: Mapping[str, str],
    selected_rubber_artifact: Mapping[str, str],
    question_contract: Any,
    episodes: Sequence[Any],
    selected_row_sha256: Sequence[str],
    observed_oracle_bundle_content_sha256: str,
    rollout_census: Any,
) -> dict[str, Any]:
    binding = _exact_dict(bindings, SHA_KEYS, name="raw oracle bindings")
    for key in SHA_KEYS:
        _sha(binding[key], name=key)
    _sha(
        observed_oracle_bundle_content_sha256,
        name="observed oracle bundle content SHA",
    )
    question = _validate_question(question_contract)
    if type(episodes) not in (list, tuple) or len(episodes) != EPISODES:
        raise EvidenceError("raw oracle requires exactly 32 runtime episodes")
    if len(selected_row_sha256) != EPISODES:
        raise EvidenceError("selected-rubber row SHA denominator differs")
    census = validate_rollout_census(rollout_census, closed_attempts=EPISODES)

    raw_episodes = []
    control_steps = 0
    safety_totals = {
        "hard_termination_by_reason": {name: 0 for name in HARD_TERMINATIONS},
        "robot_table_contact_count": 0,
        "projection_nonfinite_count": 0,
        "projection_observed_sample_count": 0,
        "qdes_observed_sample_count": 0,
        "actual_observed_sample_count": 0,
        "reference_guard_sample_count": 0,
    }
    teacher_max = 0.0
    by_reason: dict[str, int] = {}
    phase_by_reason: dict[str, dict[str, int]] = {
        "post_strike": {},
        "pre_strike_or_same_step_unknown": {},
    }
    sample_indices: set[int] = set()
    sample_sha256: set[str] = set()
    draw_intervals: set[tuple[int, int]] = set()
    for index, value in enumerate(episodes):
        row = _validate_episode(
            value, episode=index, selected_row_sha256=selected_row_sha256[index]
        )
        sample_identity = row["sampler_sample_index"]
        sample_digest = row["sampler_sample_sha256"]
        draw_interval = (row["sampler_draw_start"], row["sampler_draw_end"])
        if (
            sample_identity in sample_indices
            or sample_digest in sample_sha256
            or draw_interval in draw_intervals
        ):
            raise EvidenceError(
                "runtime episodes do not prove one distinct sampler receipt per reset"
            )
        sample_indices.add(sample_identity)
        sample_sha256.add(sample_digest)
        draw_intervals.add(draw_interval)
        control_steps += row["control_steps"]
        for reason in row["termination_reasons"]:
            if type(reason) is not str or not reason:
                raise EvidenceError("termination reason must be a non-empty string")
            by_reason[reason] = by_reason.get(reason, 0) + 1
            phase = row["terminal_phase"]
            if phase not in phase_by_reason:
                raise EvidenceError("terminal phase differs")
            phase_by_reason[phase][reason] = phase_by_reason[phase].get(reason, 0) + 1
        episode_safety = row["safety"]
        for name in HARD_TERMINATIONS:
            safety_totals["hard_termination_by_reason"][name] += (
                episode_safety["hard_termination_by_reason"][name]
            )
        for key in safety_totals:
            if key != "hard_termination_by_reason":
                safety_totals[key] += episode_safety[key]
        teacher_max = max(
            teacher_max,
            float(row["teacher_qdes"]["preclamp_max_abs_error_rad"]),
        )
        raw_episodes.append(
            {
                "episode": index,
                "control_steps": row["control_steps"],
                "terminal_phase": row["terminal_phase"],
                "termination_reasons": list(row["termination_reasons"]),
                "sampler_sample_index": row["sampler_sample_index"],
                "sampler_sample_sha256": row["sampler_sample_sha256"],
                "sampler_draw_start": row["sampler_draw_start"],
                "sampler_draw_end": row["sampler_draw_end"],
                "incoming_ball_observation": row["incoming_ball_observation"],
                "selected_rubber_evidence_sha256": selected_row_sha256[index],
                "achieved_analytic_flight": row["achieved_analytic_flight"],
                "predicted_outcome": row["predicted_outcome"],
            }
        )
    unexpected = {
        key: value
        for key, value in by_reason.items()
        if key != "action_ball_single_stroke_complete"
    }
    unsigned = {
        "schema_version": 3,
        "kind": RAW_KIND,
        "diagnostic_unauthorized": True,
        "bindings": binding,
        "observed_oracle_bundle_content_sha256": (
            observed_oracle_bundle_content_sha256
        ),
        "training_contract_artifact": dict(training_contract_artifact),
        "runner_preflight_artifact": dict(runner_preflight_artifact),
        "question_contract": question,
        "completion": {
            "requested": EPISODES,
            "terminal": len(raw_episodes),
            "single_stroke": by_reason.get("action_ball_single_stroke_complete", 0),
            "control_steps": control_steps,
        },
        "episodes": raw_episodes,
        "desired_contact_metrics": {
            "status": "INELIGIBLE",
            "reason": "target_validity_000_contact_target_absent",
        },
        "rollout_census": census,
        "termination": {
            "allowed_reason": "action_ball_single_stroke_complete",
            "by_reason": by_reason,
            "unexpected_by_reason": unexpected,
            "phase_by_reason": phase_by_reason,
        },
        "safety": {
            "control_step_denominator": control_steps,
            **safety_totals,
        },
        "selected_rubber_contact_artifact": dict(selected_rubber_artifact),
        "teacher_qdes": {
            "control_step_denominator": control_steps,
            "preclamp_max_abs_error_rad": teacher_max,
            "teleport_used": False,
        },
    }
    return _sealed(unsigned)


def _encoded(document: Any) -> bytes:
    return canonical_bytes(document) + b"\n"


def _publish_no_clobber(path: Path, document: Any) -> dict[str, str]:
    encoded = _encoded(document)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise EvidenceError("no-clobber publication refused for %s" % path) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A partial exclusive file permanently spends this namespace.  Do not
        # unlink it and accidentally permit a different evidence set to retry.
        raise
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def _planned_pin(path: Path, document: Any) -> dict[str, str]:
    encoded = _encoded(document)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def publish_bundle(bundle: Any, *, namespace: Path) -> dict[str, Any]:
    keys = (
        "schema_version",
        "kind",
        "diagnostic_unauthorized",
        "identity",
        "bindings",
        "training_contract_path",
        "runner_preflight_facts",
        "question_contract",
        "rollout_census",
        "episodes",
    )
    row = _exact_dict(bundle, keys, name="observed oracle bundle")
    if (
        row["schema_version"] != 3
        or row["kind"] != INPUT_KIND
        or row["diagnostic_unauthorized"] is not True
    ):
        raise EvidenceError("observed oracle bundle identity differs")
    identity = _exact_dict(
        row["identity"], ("action_id", "action_uid", "motion_sha256"),
        name="C211 action identity",
    )
    binding = _exact_dict(row["bindings"], SHA_KEYS, name="C211 bindings")
    for key in SHA_KEYS:
        _sha(binding[key], name=key)
    requested_root = namespace.absolute()
    root_info = requested_root.lstat()
    root = requested_root.resolve(strict=True)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or root != requested_root
        or root.name == ""
    ):
        raise EvidenceError("namespace must be one existing real directory")
    params = root / "params"
    if os.path.lexists(params):
        params_info = params.lstat()
        if (
            not stat.S_ISDIR(params_info.st_mode)
            or stat.S_ISLNK(params_info.st_mode)
            or params.resolve(strict=True) != params
        ):
            raise EvidenceError("namespace params must be one real directory")
    else:
        params.mkdir(mode=0o755)
    if params.resolve(strict=True).parent != root:
        raise EvidenceError("namespace params path escaped")

    configured_contract_source = Path(row["training_contract_path"])
    _stable_regular(configured_contract_source, name="runtime training contract")
    contract_source = configured_contract_source.resolve(strict=True)
    if sha256_file(contract_source) != binding["hard_contract_sha256"]:
        raise EvidenceError("runtime training contract SHA differs")
    contract_target = params / "training_contract.json"
    contract_copy_required = contract_source != contract_target.resolve(strict=False)
    if contract_copy_required and os.path.lexists(contract_target):
        raise EvidenceError(
            "training contract no-clobber target is already spent"
        )
    training_pin = {
        "path": str(contract_target),
        "sha256": binding["hard_contract_sha256"],
    }

    preflight = build_runner_preflight(
        launch_claim_sha256=binding["oracle_launch_claim_sha256"],
        hard_contract_sha256=binding["hard_contract_sha256"],
        facts=row["runner_preflight_facts"],
    )
    selected, selected_rows = build_selected_rubber(
        launch_claim_sha256=binding["oracle_launch_claim_sha256"],
        action_id=identity["action_id"],
        action_uid=identity["action_uid"],
        motion_sha256=identity["motion_sha256"],
        observed_contacts=[
            episode["observed_selected_rubber_contact"]
            if type(episode) is dict
            and "observed_selected_rubber_contact" in episode
            else None
            for episode in row["episodes"]
        ],
    )
    preflight_path = params / "c211_runner_preflight.json"
    selected_path = params / "c211_selected_rubber_contact.json"
    raw_path = root / "teacher_qdes_oracle_32ep.json"
    preflight_pin = _planned_pin(
        preflight_path, preflight
    )
    selected_pin = _planned_pin(
        selected_path, selected
    )
    raw = build_raw_oracle(
        bindings=binding,
        training_contract_artifact=training_pin,
        runner_preflight_artifact=preflight_pin,
        selected_rubber_artifact=selected_pin,
        question_contract=row["question_contract"],
        episodes=row["episodes"],
        selected_row_sha256=selected_rows,
        observed_oracle_bundle_content_sha256=canonical_sha256(row),
        rollout_census=row["rollout_census"],
    )
    # Complete every semantic validation before the first sidecar write.  A
    # filesystem race can still spend part of the namespace, which is safe;
    # semantic input failure cannot.
    if contract_copy_required:
        encoded_contract = contract_source.read_bytes()
        try:
            descriptor = os.open(
                contract_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
        except OSError as exc:
            raise EvidenceError("training contract no-clobber copy refused") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded_contract)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        _stable_regular(contract_target, name="runtime training contract target")
    if sha256_file(contract_target) != training_pin["sha256"]:
        raise EvidenceError("published training contract SHA differs")
    if _publish_no_clobber(preflight_path, preflight) != preflight_pin:
        raise EvidenceError("published runner preflight pin differs")
    if _publish_no_clobber(selected_path, selected) != selected_pin:
        raise EvidenceError("published selected-rubber pin differs")
    raw_pin = _publish_no_clobber(raw_path, raw)
    unsigned = {
        "schema_version": 3,
        "kind": OUTPUT_KIND,
        "diagnostic_unauthorized": True,
        "oracle_launch_claim_sha256": binding["oracle_launch_claim_sha256"],
        "training_contract_artifact": training_pin,
        "runner_preflight_artifact": preflight_pin,
        "selected_rubber_contact_artifact": selected_pin,
        "raw_oracle_artifact": raw_pin,
        "raw_oracle_content_sha256": raw["content_sha256"],
    }
    return _sealed(unsigned)


def _load_canonical(path: Path) -> Any:
    _stable_regular(path, name="observed oracle bundle")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("observed oracle bundle is not strict JSON") from exc
    if raw != _encoded(value):
        raise EvidenceError("observed oracle bundle must be canonical JSON plus newline")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--namespace", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = publish_bundle(
            _load_canonical(args.bundle.resolve(strict=True)),
            namespace=args.namespace,
        )
    except (EvidenceError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
