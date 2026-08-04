#!/usr/bin/env python3
"""Collect a strict C211 000 live-oracle bundle from an already-built runner.

The collector deliberately owns neither Hydra construction nor C211 admission.
Its caller must first run ``validate_action_ball_c211_runner`` and then hand the
same runner/env pair to :func:`collect_live_oracle_bundle`.  A small runner-side
adapter supplies *observed* episode rows through
``collect_action_ball_c211_oracle_episodes``; this module validates and projects
those rows into the exact input consumed by ``action_ball_c211_oracle_evidence``.

WAIT-only resets are source facts but are not task opportunities.  They are
therefore excluded before the closed-attempt denominator is formed.  A closed
TASK_ACTIVE miss remains one denominator row, so a zero-hit run cannot collapse
to a zero denominator.  C211 is a ``physical_ball=false`` learnability
ablation: its outcome authority is the one-shot analytic flight produced from
an actual achieved selected-rubber contact, never a PhysX observed landing and
never a hypothetical no-contact rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


INPUT_KIND = "action_ball_c211_observed_oracle_bundle_v2"
LIVE_EPISODE_KIND = "action_ball_c211_live_oracle_episode_v2"
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
BINDING_KEYS = (
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
QUESTION_KEYS = (
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
QUESTION_RNG = {
    "owner": "runtime_curriculum_sampler",
    "cadence": "every_episode_reset",
    "draw_count_authority": "sample_receipt_draw_end_minus_draw_start",
    "zero_draw_claim_permitted": False,
    "checkpoint_resume": "exact_sampler_and_curriculum_state",
}
OBSERVED_CONTACT_KEYS = (
    "runtime_control_step",
    "task_valid",
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
LIVE_EPISODE_KEYS = (
    "schema_version",
    "kind",
    "source_episode",
    "control_steps",
    "wait_control_steps",
    "task_valid_control_steps",
    "sampler_sample_index",
    "sampler_sample_sha256",
    "sampler_draw_start",
    "sampler_draw_end",
    "incoming_ball_observation",
    "actual_contact",
    "achieved_analytic_flight",
    "predicted_outcome",
    "attempt_closure",
    "safety",
    "teacher_qdes",
)
RUNTIME_ADAPTER_FIELDS = {
    "task_valid": "racket_target._action_ball_task_valid[0]",
    "exact_strike": "racket_target.metrics['exact_strike_hit_rate'][0] > 0.5",
    "actual_selected_rubber_contact": "racket_target.vb_fired[0]",
    "analytic_landing_xy_m": "racket_target.vb_landing_xy[0]",
    "analytic_landing_valid": "racket_target.vb_landing_valid[0]",
    "analytic_net_crossed": "racket_target.vb_net_crossed[0]",
    "analytic_net_clear": "racket_target.vb_net_clear[0]",
    "analytic_on_opponent_table": "racket_target.vb_on_opponent[0]",
    "closed_attempt_h": "racket_target._action_ball_attempt_hit[0]",
    "closed_attempt_c": "racket_target._action_ball_ledger[C, action_slot] delta",
    "terminal_reasons": "termination_manager active-term snapshot before auto-reset",
    "incoming_actor_critic": "runtime C211 actor/critic observation term snapshots",
    "sampler_receipt": "racket_target._action_ball_task_receipt_for_env(0)",
}


class LiveOracleError(RuntimeError):
    """The live source was incomplete, contradictory, or non-canonical."""


def runtime_adapter_contract() -> dict[str, Any]:
    """Return the minimal live Isaac field contract for the train hook.

    ``vb_fired`` is already the achieved selected-rubber swept-contact gate.
    The five ``vb_*`` outcome caches are read only on that one-shot hit tick.
    On a miss they must not be read into a flight row; the helper emits no
    analytic flight/prediction for it.  ``_reset_idx`` interception is needed
    only to preserve terminal latches/reasons across ManagerBasedRLEnv auto-reset.
    """

    payload = {
        "schema_version": 2,
        "kind": "action_ball_c211_live_runtime_adapter_contract_v2",
        "physical_ball": False,
        "fields": dict(RUNTIME_ADAPTER_FIELDS),
        "contact_authority": "achieved_selected_rubber_swept_contact_gate",
        "outcome_authority": "c225_analytic_flight_from_achieved_contact_one_shot",
        "no_contact_policy": "closed_attempt_counts_C_without_flight_or_prediction",
        "wait_policy": "task_valid_false_does_not_count_C_or_H",
    }
    return {**payload, "content_sha256": canonical_sha256(payload)}


def build_achieved_analytic_evidence(
    *,
    selected_rubber_contact: bool,
    landing_xy_m: Any = None,
    landing_valid: Any = None,
    net_crossed: Any = None,
    net_clear: Any = None,
    on_opponent_table: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build flight/prediction rows from the current one-shot runtime caches."""

    if type(selected_rubber_contact) is not bool:
        raise LiveOracleError("selected_rubber_contact must be bool")
    supplied = (
        landing_xy_m, landing_valid, net_crossed, net_clear, on_opponent_table
    )
    if not selected_rubber_contact:
        if any(value is not None for value in supplied):
            raise LiveOracleError("no-contact row must not consume hypothetical flight caches")
        return (
            {
                "evaluated": False,
                "finite": False,
                "landing_xy_m": None,
                "landing_valid": False,
                "net_crossed": False,
                "net_clear": False,
                "on_opponent_table": False,
                "source": None,
            },
            {
                "evaluated": False,
                "predicted_net_clear": None,
                "predicted_legal_landing": None,
                "predicted_landing_xy_m": None,
                "source": None,
            },
        )
    xy = _finite_vec2(landing_xy_m, name="runtime analytic landing")
    for name, value in (
        ("landing_valid", landing_valid),
        ("net_crossed", net_crossed),
        ("net_clear", net_clear),
        ("on_opponent_table", on_opponent_table),
    ):
        if type(value) is not bool:
            raise LiveOracleError("runtime analytic %s must be bool" % name)
    flight = {
        "evaluated": True,
        "finite": True,
        "landing_xy_m": xy,
        "landing_valid": landing_valid,
        "net_crossed": net_crossed,
        "net_clear": net_clear,
        "on_opponent_table": on_opponent_table,
        "source": "runtime_vb_one_shot_from_achieved_selected_rubber_contact",
    }
    prediction = {
        "evaluated": True,
        "predicted_net_clear": net_clear,
        "predicted_legal_landing": bool(
            landing_valid and net_crossed and net_clear and on_opponent_table
        ),
        "predicted_landing_xy_m": list(xy),
        "source": "runtime_c225_achieved_flight_prediction_one_shot",
    }
    # Reuse the consumer-facing validators so the convenience builder cannot
    # drift from the live-row contract.
    validated_flight = _achieved_analytic_flight(flight, selected=True)
    _predicted_outcome(prediction, selected=True, flight=validated_flight)
    return flight, prediction


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


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise LiveOracleError("%s keys differ" % name)
    return dict(value)


def _plain_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LiveOracleError("%s must be a plain integer >= %d" % (name, minimum))
    return value


def _sha(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LiveOracleError("%s must be exact lowercase SHA-256" % name)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_contract(value: Any, *, bindings: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact_dict(value, QUESTION_KEYS, name="C211 000 question contract")
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
        raise LiveOracleError(
            "question contract is not runtime-sampled C211 direct-ball"
        )
    return row


def _finite_vec3(value: Any, *, name: str) -> list[float | int]:
    if type(value) is not list or len(value) != 3:
        raise LiveOracleError("%s must be a length-3 JSON list" % name)
    if any(
        type(component) not in (int, float)
        or not math.isfinite(float(component))
        for component in value
    ):
        raise LiveOracleError("%s must contain finite JSON numbers" % name)
    return list(value)


def _finite_vec2(value: Any, *, name: str) -> list[float | int]:
    if type(value) is not list or len(value) != 2:
        raise LiveOracleError("%s must be a length-2 JSON list" % name)
    if any(
        type(component) not in (int, float)
        or not math.isfinite(float(component))
        for component in value
    ):
        raise LiveOracleError("%s must contain finite JSON numbers" % name)
    return list(value)


def _incoming(value: Any) -> dict[str, Any]:
    row = _exact_dict(
        value, ("source", "actor", "critic"), name="incoming-ball observation"
    )
    if row["source"] != "runtime_actor_and_critic_observation_terms":
        raise LiveOracleError("incoming-ball observation source differs")
    for group in ("actor", "critic"):
        fields = _exact_dict(row[group], INCOMING_FIELDS, name=group + " incoming ball")
        row[group] = {
            name: _finite_vec3(fields[name], name=group + " " + name)
            for name in INCOMING_FIELDS
        }
    if row["actor"] != row["critic"]:
        raise LiveOracleError("actor/critic incoming-ball facts differ")
    return row


def _contact(value: Any, *, control_steps: int) -> dict[str, Any]:
    row = _exact_dict(value, OBSERVED_CONTACT_KEYS, name="actual contact")
    step = _plain_int(
        row["runtime_control_step"], name="actual contact control step", minimum=1
    )
    if step > control_steps:
        raise LiveOracleError("actual contact control step exceeds its episode")
    for name in OBSERVED_CONTACT_KEYS[1:]:
        if type(row[name]) is not bool:
            raise LiveOracleError("actual contact %s must be an exact bool" % name)
    selected = all(
        row[name]
        for name in (
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
        raise LiveOracleError("actual contact classifications overlap")
    if row["selected_face_sweep_contact"] and not row["task_valid"]:
        raise LiveOracleError("WAIT contact cannot enter actual selected-rubber evidence")
    return row


def _selected_rubber(contact: Mapping[str, Any]) -> bool:
    return all(
        contact[name]
        for name in (
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


def _achieved_analytic_flight(value: Any, *, selected: bool) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "evaluated",
            "finite",
            "landing_xy_m",
            "landing_valid",
            "net_crossed",
            "net_clear",
            "on_opponent_table",
            "source",
        ),
        name="achieved analytic flight",
    )
    for name in (
        "evaluated", "finite", "landing_valid", "net_crossed", "net_clear",
        "on_opponent_table",
    ):
        if type(row[name]) is not bool:
            raise LiveOracleError("achieved analytic flight %s must be bool" % name)
    if row["evaluated"] is not selected:
        raise LiveOracleError(
            "achieved analytic flight evaluation differs from selected-rubber contact"
        )
    if selected:
        if (
            row["source"]
            != "runtime_vb_one_shot_from_achieved_selected_rubber_contact"
            or not row["finite"]
        ):
            raise LiveOracleError("achieved analytic flight authority/finite gate differs")
        row["landing_xy_m"] = _finite_vec2(
            row["landing_xy_m"], name="achieved analytic landing"
        )
        if row["net_clear"] and not row["net_crossed"]:
            raise LiveOracleError("analytic net-clear cannot precede net crossing")
        if row["on_opponent_table"] and not row["landing_valid"]:
            raise LiveOracleError("opponent-table result requires a valid analytic landing")
    elif (
        row["finite"]
        or row["landing_xy_m"] is not None
        or row["landing_valid"]
        or row["net_crossed"]
        or row["net_clear"]
        or row["on_opponent_table"]
        or row["source"] is not None
    ):
        raise LiveOracleError("no-contact row carries hypothetical analytic flight")
    return row


def _predicted_outcome(
    value: Any,
    *,
    selected: bool,
    flight: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "evaluated",
            "predicted_net_clear",
            "predicted_legal_landing",
            "predicted_landing_xy_m",
            "source",
        ),
        name="predicted analytic outcome",
    )
    if type(row["evaluated"]) is not bool or row["evaluated"] is not selected:
        raise LiveOracleError("predicted outcome evaluation differs from actual contact")
    if selected:
        if row["source"] != "runtime_c225_achieved_flight_prediction_one_shot":
            raise LiveOracleError("predicted analytic outcome source differs")
        for name in ("predicted_net_clear", "predicted_legal_landing"):
            if type(row[name]) is not bool:
                raise LiveOracleError("predicted outcome %s must be bool" % name)
        row["predicted_landing_xy_m"] = _finite_vec2(
            row["predicted_landing_xy_m"], name="predicted landing"
        )
        expected_legal = bool(
            flight["landing_valid"]
            and flight["net_crossed"]
            and flight["net_clear"]
            and flight["on_opponent_table"]
        )
        if (
            row["predicted_landing_xy_m"] != flight["landing_xy_m"]
            or row["predicted_net_clear"] is not flight["net_clear"]
            or row["predicted_legal_landing"] is not expected_legal
        ):
            raise LiveOracleError("predicted outcome differs from achieved analytic flight")
    elif any(
        row[name] is not None
        for name in (
            "predicted_net_clear", "predicted_legal_landing",
            "predicted_landing_xy_m", "source",
        )
    ):
        raise LiveOracleError("miss carries hypothetical predicted outcome")
    return row


def _safety(value: Any, *, control_steps: int) -> dict[str, Any]:
    keys = (
        "hard_termination_by_reason",
        "robot_table_contact_count",
        "projection_nonfinite_count",
        "projection_observed_sample_count",
        "qdes_observed_sample_count",
        "actual_observed_sample_count",
        "reference_guard_sample_count",
    )
    row = _exact_dict(value, keys, name="live episode safety")
    hard = _exact_dict(
        row["hard_termination_by_reason"], HARD_TERMINATIONS,
        name="live hard termination counts",
    )
    row["hard_termination_by_reason"] = {
        name: _plain_int(hard[name], name="hard termination " + name)
        for name in HARD_TERMINATIONS
    }
    for name in keys[1:]:
        row[name] = _plain_int(row[name], name="safety " + name)
    for name in (
        "projection_observed_sample_count",
        "qdes_observed_sample_count",
        "actual_observed_sample_count",
        "reference_guard_sample_count",
    ):
        if row[name] != control_steps:
            raise LiveOracleError("safety %s denominator differs" % name)
    return row


def _teacher_qdes(value: Any) -> dict[str, Any]:
    row = _exact_dict(
        value, ("preclamp_max_abs_error_rad", "teleport_used"),
        name="live teacher qdes",
    )
    error = row["preclamp_max_abs_error_rad"]
    if type(error) not in (int, float) or not math.isfinite(float(error)) or error < 0:
        raise LiveOracleError("teacher qdes error must be finite and non-negative")
    if row["teleport_used"] is not False:
        raise LiveOracleError("live C211 oracle must not teleport")
    return row


def _attempt_closure(value: Any) -> dict[str, Any]:
    row = _exact_dict(
        value, ("closed_attempt", "terminal_phase", "termination_reasons"),
        name="attempt closure",
    )
    if type(row["closed_attempt"]) is not bool:
        raise LiveOracleError("attempt closure closed_attempt must be bool")
    reasons = row["termination_reasons"]
    if (
        type(reasons) is not list
        or any(type(reason) is not str or not reason for reason in reasons)
        or len(set(reasons)) != len(reasons)
    ):
        raise LiveOracleError("attempt closure reasons must be unique non-empty strings")
    if row["closed_attempt"]:
        if row["terminal_phase"] not in (
            "post_strike", "pre_strike_or_same_step_unknown"
        ) or not reasons:
            raise LiveOracleError("closed attempt lacks terminal evidence")
    elif row["terminal_phase"] is not None:
        raise LiveOracleError("excluded WAIT outcome cannot claim a task terminal phase")
    return row


def _project_live_episode(value: Any, *, output_episode: int) -> dict[str, Any] | None:
    row = _exact_dict(value, LIVE_EPISODE_KEYS, name="live C211 episode")
    if row["schema_version"] != 2 or row["kind"] != LIVE_EPISODE_KIND:
        raise LiveOracleError("live C211 episode identity differs")
    _plain_int(row["source_episode"], name="source episode")
    control_steps = _plain_int(row["control_steps"], name="control steps", minimum=1)
    wait_steps = _plain_int(row["wait_control_steps"], name="WAIT control steps")
    task_steps = _plain_int(
        row["task_valid_control_steps"], name="task-valid control steps"
    )
    if wait_steps + task_steps != control_steps:
        raise LiveOracleError("WAIT/TASK control-step partition does not close")
    contact = _contact(row["actual_contact"], control_steps=control_steps)
    selected = _selected_rubber(contact)
    flight = _achieved_analytic_flight(
        row["achieved_analytic_flight"], selected=selected
    )
    prediction = _predicted_outcome(
        row["predicted_outcome"],
        selected=selected,
        flight=flight,
    )
    outcome = _attempt_closure(row["attempt_closure"])
    safety = _safety(row["safety"], control_steps=control_steps)
    teacher = _teacher_qdes(row["teacher_qdes"])

    if not outcome["closed_attempt"]:
        if task_steps != 0 or contact["task_valid"]:
            raise LiveOracleError("excluded reset contains TASK_ACTIVE evidence")
        return None
    if task_steps < 1:
        raise LiveOracleError("closed attempt has no TASK_ACTIVE control step")
    if contact["task_valid"] is not True:
        # A miss is still observed during a valid swing; it must not be encoded
        # as an absent/WAIT observation merely because no contact occurred.
        raise LiveOracleError("closed attempt contact observation is not task-valid")

    sample_index = _plain_int(
        row["sampler_sample_index"], name="sampler sample index"
    )
    sample_sha = _sha(row["sampler_sample_sha256"], name="sampler sample SHA")
    draw_start = _plain_int(row["sampler_draw_start"], name="sampler draw start")
    draw_end = _plain_int(
        row["sampler_draw_end"], name="sampler draw end", minimum=1
    )
    if draw_end <= draw_start:
        raise LiveOracleError("sampler receipt must prove positive physical RNG draws")
    observed_contact = {
        "episode": output_episode,
        "runtime_control_step": contact["runtime_control_step"],
        "task_valid": contact["task_valid"],
        "eligible_closed_swing": True,
        **{name: contact[name] for name in OBSERVED_CONTACT_KEYS[2:]},
    }
    return {
        "episode": output_episode,
        "control_steps": control_steps,
        "terminal_phase": outcome["terminal_phase"],
        "termination_reasons": list(outcome["termination_reasons"]),
        "sampler_sample_index": sample_index,
        "sampler_sample_sha256": sample_sha,
        "sampler_draw_start": draw_start,
        "sampler_draw_end": draw_end,
        "incoming_ball_observation": _incoming(row["incoming_ball_observation"]),
        "observed_selected_rubber_contact": observed_contact,
        "achieved_analytic_flight": flight,
        "predicted_outcome": prediction,
        "safety": safety,
        "teacher_qdes": teacher,
    }


EpisodeSource = Callable[..., Iterable[Mapping[str, Any]]]
RuntimeStepAdapter = Callable[..., Mapping[str, Any]]


def run_live_policy_episodes(
    runner: Any,
    env: Any,
    *,
    episodes: int = EPISODES,
    runtime_step_adapter: RuntimeStepAdapter,
) -> list[Mapping[str, Any]]:
    """Drive the built runner/env while a runtime adapter captures real facts.

    The adapter, rather than Gym ``extras``, is the evidence boundary.  It must
    call the real ``env.step(actions)`` and return exactly
    ``{"next_observation", "completed_episode"}``.  ``completed_episode`` is
    either ``None`` or one :data:`LIVE_EPISODE_KIND` row captured across the
    auto-reset boundary.  A concrete Isaac adapter therefore wraps ``_reset_idx``
    long enough to snapshot ``task_valid``, the one-shot ``vb_fired`` selected-
    rubber gate, and the ``vb_landing_*``/``vb_net_*`` achieved-flight caches;
    it must restore that wrapper in ``finally``.

    This loop owns the one initial reset and policy stepping.  It never calls a
    legacy solver/LM, teleports state, or treats an ``extras`` label as evidence.
    """

    if episodes != EPISODES:
        raise LiveOracleError("C211 live policy rollout requires exactly 32 episodes")
    if not callable(runtime_step_adapter):
        raise LiveOracleError("runtime_step_adapter must be callable")
    reset = env.reset()
    if type(reset) is not tuple or len(reset) != 2 or type(reset[1]) is not dict:
        raise LiveOracleError("live policy initial reset must return (observation, info)")
    observation = reset[0]
    get_policy = getattr(runner, "get_inference_policy", None)
    if not callable(get_policy):
        raise LiveOracleError("validated runner has no inference policy")
    device = getattr(getattr(env, "unwrapped", None), "device", None)
    policy = get_policy(device=device)
    if not callable(policy):
        raise LiveOracleError("runner inference policy is not callable")

    rows = []
    closed_attempts = 0
    source_episode = 0
    control_step = 0
    # The adapter is required to close every source episode no later than the
    # environment horizon.  Keep one extra step for a terminal emitted exactly
    # on the horizon and include WAIT-only resets in the cap.
    base = getattr(env, "unwrapped", None)
    horizon = getattr(base, "max_episode_length", None)
    if type(horizon) is not int or horizon < 1:
        raise LiveOracleError("live policy env lacks a positive episode horizon")
    max_steps = episodes * (horizon + 1) * 4
    total_steps = 0
    while closed_attempts < episodes:
        if total_steps >= max_steps:
            raise LiveOracleError("live policy rollout exhausted its bounded step cap")
        actions = policy(observation)
        packet = runtime_step_adapter(
            env=env,
            actions=actions,
            source_episode=source_episode,
            runtime_control_step=control_step + 1,
        )
        packet = _exact_dict(
            packet, ("next_observation", "completed_episode"),
            name="runtime step adapter packet",
        )
        observation = packet["next_observation"]
        total_steps += 1
        control_step += 1
        completed = packet["completed_episode"]
        if completed is None:
            continue
        if type(completed) is not dict:
            raise LiveOracleError("runtime step adapter completed row must be a dict")
        if completed.get("source_episode") != source_episode:
            raise LiveOracleError("runtime adapter source episode order differs")
        if completed.get("control_steps") != control_step:
            raise LiveOracleError("runtime adapter control-step count differs")
        rows.append(completed)
        closure = completed.get("attempt_closure")
        if type(closure) is not dict or type(closure.get("closed_attempt")) is not bool:
            raise LiveOracleError("runtime adapter completion lacks exact closure flag")
        if closure["closed_attempt"]:
            closed_attempts += 1
        source_episode += 1
        control_step = 0
    return rows


def _resolve_episode_source(runner: Any, source: EpisodeSource | None) -> EpisodeSource:
    if source is not None:
        if not callable(source):
            raise LiveOracleError("episode_source must be callable")
        return source
    resolved = getattr(
        runner, "collect_action_ball_c211_oracle_episodes", None
    )
    if not callable(resolved):
        raise LiveOracleError(
            "runner lacks collect_action_ball_c211_oracle_episodes live adapter"
        )
    return resolved


def collect_live_oracle_bundle(
    runner: Any,
    env: Any,
    *,
    identity: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_contract_path: str | Path,
    runner_preflight_facts: Mapping[str, Any],
    question_contract: Mapping[str, Any],
    episodes: int = EPISODES,
    episode_source: EpisodeSource | None = None,
    runtime_step_adapter: RuntimeStepAdapter | None = None,
) -> dict[str, Any]:
    """Run the live adapter and return the current producer's canonical payload.

    ``runner_preflight_facts`` must be the exact non-``None`` result returned by
    ``validate_action_ball_c211_runner(runner)`` for this runner.  The helper
    keeps it opaque because the evidence producer independently validates the
    full 211/319 ABI and normalizer proof.
    """

    if episodes != EPISODES:
        raise LiveOracleError("C211 live oracle requires exactly 32 closed attempts")
    if type(runner_preflight_facts) is not dict:
        raise LiveOracleError("validated runner preflight facts must be one dict")
    identity_row = _exact_dict(
        identity, ("action_id", "action_uid", "motion_sha256"),
        name="C211 live identity",
    )
    if type(identity_row["action_id"]) is not str or not identity_row["action_id"]:
        raise LiveOracleError("action_id must be a non-empty string")
    _plain_int(identity_row["action_uid"], name="action_uid", minimum=1)
    _sha(identity_row["motion_sha256"], name="motion SHA")
    binding_row = _exact_dict(bindings, BINDING_KEYS, name="C211 live bindings")
    for name in BINDING_KEYS:
        _sha(binding_row[name], name=name)
    if binding_row["motion_sha256"] != identity_row["motion_sha256"]:
        raise LiveOracleError("identity motion SHA differs from live bindings")
    question_row = _question_contract(question_contract, bindings=binding_row)
    contract_path = Path(training_contract_path)
    if not contract_path.is_file() or contract_path.is_symlink():
        raise LiveOracleError("training contract path must be one real file")
    if _sha256_file(contract_path) != binding_row["hard_contract_sha256"]:
        raise LiveOracleError("training contract file SHA differs from live bindings")

    if episode_source is not None and runtime_step_adapter is not None:
        raise LiveOracleError("choose episode_source or runtime_step_adapter, not both")
    if runtime_step_adapter is not None:
        observed = run_live_policy_episodes(
            runner, env, episodes=episodes,
            runtime_step_adapter=runtime_step_adapter,
        )
    else:
        source = _resolve_episode_source(runner, episode_source)
        try:
            observed = source(env=env, episodes=episodes)
        except TypeError as exc:
            raise LiveOracleError("live episode adapter call contract differs") from exc
    if isinstance(observed, (str, bytes, dict)) or not isinstance(observed, Iterable):
        raise LiveOracleError("live episode adapter must return an episode iterable")

    projected = []
    source_indices = set()
    for value in observed:
        if type(value) is not dict:
            raise LiveOracleError("live episode source row must be a dict")
        source_index = value.get("source_episode")
        if type(source_index) is not int or source_index in source_indices:
            raise LiveOracleError("live source episode indices must be unique integers")
        source_indices.add(source_index)
        episode = _project_live_episode(value, output_episode=len(projected))
        if episode is not None:
            projected.append(episode)
        if len(projected) == episodes:
            break
    if len(projected) != episodes:
        raise LiveOracleError("live source did not produce 32 closed TASK_ACTIVE attempts")
    sample_indices = [row["sampler_sample_index"] for row in projected]
    sample_sha256 = [row["sampler_sample_sha256"] for row in projected]
    draw_intervals = [
        (row["sampler_draw_start"], row["sampler_draw_end"])
        for row in projected
    ]
    if (
        len(set(sample_indices)) != episodes
        or len(set(sample_sha256)) != episodes
        or len(set(draw_intervals)) != episodes
    ):
        raise LiveOracleError(
            "closed attempts do not prove one distinct sampler receipt per reset"
        )
    # H is derived solely from the exact selected-rubber conjunction.  This is
    # intentionally explicit even though per-row booleans already imply it.
    hits = sum(
        _selected_rubber(row["observed_selected_rubber_contact"])
        for row in projected
    )
    if hits > len(projected):  # defensive, keeps the H <= C contract visible
        raise LiveOracleError("actual selected-rubber hits exceed closed attempts")

    return {
        "schema_version": 2,
        "kind": INPUT_KIND,
        "diagnostic_unauthorized": True,
        "identity": identity_row,
        "bindings": binding_row,
        "training_contract_path": str(contract_path),
        "runner_preflight_facts": dict(runner_preflight_facts),
        "question_contract": question_row,
        "episodes": projected,
    }


def write_canonical_no_clobber(path: str | Path, document: Mapping[str, Any]) -> dict[str, str]:
    """Failure-atomic canonical publication; an existing target is never replaced."""

    target = Path(path)
    encoded = canonical_bytes(document) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(target), follow_symlinks=False)
        except OSError as exc:
            raise LiveOracleError("C211 live bundle no-clobber target is spent") from exc
        temporary.unlink()
        directory = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target), "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = (
    "EPISODES",
    "INPUT_KIND",
    "LIVE_EPISODE_KIND",
    "LiveOracleError",
    "canonical_bytes",
    "canonical_sha256",
    "build_achieved_analytic_evidence",
    "collect_live_oracle_bundle",
    "run_live_policy_episodes",
    "runtime_adapter_contract",
    "write_canonical_no_clobber",
)
