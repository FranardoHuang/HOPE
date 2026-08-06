"""Focused, dependency-light tests for the C211 live-oracle collector."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/action_ball_c211_live_oracle.py"
SPEC = importlib.util.spec_from_file_location("c211_live_oracle_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
PRODUCER_SOURCE = ROOT / "scripts/action_ball_c211_oracle_evidence.py"
PRODUCER_SPEC = importlib.util.spec_from_file_location(
    "c211_oracle_evidence_for_live_test", PRODUCER_SOURCE
)
assert PRODUCER_SPEC is not None and PRODUCER_SPEC.loader is not None
P = importlib.util.module_from_spec(PRODUCER_SPEC)
PRODUCER_SPEC.loader.exec_module(P)


SHA = "a" * 64


def _question():
    return {
        "target_source": "direct_ball",
        "question_source": "runtime_curriculum_sampler",
        "target_recipe": "outcome_dense_only",
        "target_validity_mask": [False, False, False],
        "target_observation_noise": False,
        "incoming_ball_fields": list(M.INCOMING_FIELDS),
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "question_rng": dict(M.QUESTION_RNG),
    }


def _incoming():
    fields = {
        "incoming_ball_contact_position_heading": [0.4, -0.2, 0.9],
        "incoming_ball_contact_velocity_heading": [-2.0, 0.1, -0.2],
        "incoming_ball_contact_spin_heading": [0.0, 4.0, 1.0],
    }
    return {
        "source": "runtime_actor_and_critic_observation_terms",
        "actor": {name: list(value) for name, value in fields.items()},
        "critic": {name: list(value) for name, value in fields.items()},
    }


def _contact(*, hit: bool, task_valid: bool = True):
    return {
        "runtime_control_step": 2,
        "task_valid": task_valid,
        "exact_strike": hit,
        "selected_face_sweep_contact": hit,
        "selected_face_bracketed": hit,
        "selected_face_edge_safe": hit,
        "selected_face_geometry_finite": True,
        "selected_face_closing_speed_positive": hit,
        "selected_face_normal_speed_consistent": hit,
        "wrong_surface_contact": False,
        "edge_or_rim_ambiguous": False,
        "between_planes_ambiguous": False,
    }


def _safety(control_steps: int):
    return {
        "hard_termination_by_reason": {name: 0 for name in M.HARD_TERMINATIONS},
        "robot_table_contact_count": 0,
        "projection_nonfinite_count": 0,
        "projection_observed_sample_count": control_steps,
        "qdes_observed_sample_count": control_steps,
        "actual_observed_sample_count": control_steps,
        "reference_guard_sample_count": control_steps,
    }


def _row(source_episode: int, *, hit: bool, wait_only: bool = False):
    control_steps = 2 if wait_only else 5
    flight, prediction = M.build_achieved_analytic_evidence(
        selected_rubber_contact=hit,
        **(
            {
                "landing_xy_m": [2.2, -0.6],
                "landing_valid": True,
                "net_crossed": True,
                "net_clear": True,
                "on_opponent_table": True,
            }
            if hit else {}
        ),
    )
    return {
        "schema_version": 2,
        "kind": M.LIVE_EPISODE_KIND,
        "source_episode": source_episode,
        "control_steps": control_steps,
        "wait_control_steps": control_steps if wait_only else 3,
        "task_valid_control_steps": 0 if wait_only else 2,
        "sampler_sample_index": None if wait_only else source_episode,
        "sampler_sample_sha256": (
            None
            if wait_only
            else hashlib.sha256(
                ("sampler-sample-%d" % source_episode).encode("ascii")
            ).hexdigest()
        ),
        "sampler_draw_start": None if wait_only else 100 + source_episode * 17,
        "sampler_draw_end": None if wait_only else 117 + source_episode * 17,
        "incoming_ball_observation": _incoming(),
        "actual_contact": _contact(hit=hit, task_valid=not wait_only),
        "achieved_analytic_flight": flight,
        "predicted_outcome": prediction,
        "attempt_closure": {
            "closed_attempt": not wait_only,
            "terminal_phase": None if wait_only else (
                "post_strike" if hit else "pre_strike_or_same_step_unknown"
            ),
            "termination_reasons": (
                ["reset_during_wait"] if wait_only
                else ["action_ball_single_stroke_complete"]
            ),
        },
        "safety": _safety(control_steps),
        "teacher_qdes": {
            "preclamp_max_abs_error_rad": 0.0,
            "teleport_used": False,
        },
    }


class FakeRunner:
    def __init__(self):
        self.calls = []

    def collect_action_ball_c211_oracle_episodes(self, *, env, episodes):
        self.calls.append((env, episodes))
        # A WAIT-only reset is real source evidence but not C.  The first
        # All 32 TASK_ACTIVE attempts close honestly as misses: H=0, C=32.
        return [
            _row(0, hit=False, wait_only=True),
            *[_row(index + 1, hit=False) for index in range(32)],
        ]


def _collect(tmp_path):
    contract = tmp_path / "training_contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    hard_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    bindings = {name: SHA for name in M.BINDING_KEYS}
    bindings["hard_contract_sha256"] = hard_sha
    runner = FakeRunner()
    env = object()
    payload = M.collect_live_oracle_bundle(
        runner,
        env,
        identity={"action_id": "take_061_unit04_bh", "action_uid": 7,
                  "motion_sha256": SHA},
        bindings=bindings,
        training_contract_path=contract,
        runner_preflight_facts={"validated": True},
        question_contract=_question(),
        episodes=32,
    )
    return runner, env, payload


def test_live_bundle_excludes_wait_and_keeps_zero_hit_closed_attempts(tmp_path):
    runner, env, payload = _collect(tmp_path)
    assert runner.calls == [(env, 32)]
    assert payload["kind"] == M.INPUT_KIND
    assert payload["diagnostic_unauthorized"] is True
    assert len(payload["episodes"]) == 32
    assert [row["episode"] for row in payload["episodes"]] == list(range(32))

    contacts = [row["observed_selected_rubber_contact"] for row in payload["episodes"]]
    hits = [row for row in contacts if row["selected_face_sweep_contact"]]
    assert len(hits) == 0
    assert len(hits) <= len(payload["episodes"])  # H <= C
    assert len(payload["episodes"]) - len(hits) == 32  # zero-hit still gives C=32
    assert all(row["task_valid"] is True for row in contacts)
    assert all(row["eligible_closed_swing"] is True for row in contacts)

    # The INPUT_KIND projection retains actual selected-contact and terminal
    # closure fields, including explicit absence of hypothetical miss flights.
    assert contacts[0]["exact_strike"] is False
    assert contacts[0]["selected_face_bracketed"] is False
    assert payload["episodes"][0]["achieved_analytic_flight"]["evaluated"] is False
    assert payload["episodes"][0]["predicted_outcome"]["evaluated"] is False
    assert payload["episodes"][0]["terminal_phase"] == (
        "pre_strike_or_same_step_unknown"
    )
    assert payload["episodes"][1]["terminal_phase"] == (
        "pre_strike_or_same_step_unknown"
    )
    assert payload["episodes"][1]["termination_reasons"] == [
        "action_ball_single_stroke_complete"
    ]

    # Prove exact compatibility at the current producer boundary, including
    # its selected-rubber row binding, rather than only comparing kind names.
    assert set(payload) == {
        "schema_version", "kind", "diagnostic_unauthorized", "identity",
        "bindings", "training_contract_path", "runner_preflight_facts",
        "question_contract", "rollout_census", "episodes",
    }
    # 人话:这一跑一共开了 33 次,其中 1 次在 WAIT 里就死了、不算一次尝试。
    # 被排除掉可以,被藏起来不行 —— 分母必须写在收据上。
    assert payload["rollout_census"] == {
        "source_episodes_consumed": 33,
        "wait_only_reset_excluded": 1,
        "closed_attempts": 32,
    }
    assert (
        payload["rollout_census"]["closed_attempts"]
        + payload["rollout_census"]["wait_only_reset_excluded"]
        == payload["rollout_census"]["source_episodes_consumed"]
    )
    P._validate_question(payload["question_contract"])
    _selected, selected_row_sha = P.build_selected_rubber(
        launch_claim_sha256=payload["bindings"]["oracle_launch_claim_sha256"],
        action_id=payload["identity"]["action_id"],
        action_uid=payload["identity"]["action_uid"],
        motion_sha256=payload["identity"]["motion_sha256"],
        observed_contacts=[
            row["observed_selected_rubber_contact"] for row in payload["episodes"]
        ],
    )
    for index, row in enumerate(payload["episodes"]):
        P._validate_episode(
            row, episode=index, selected_row_sha256=selected_row_sha[index]
        )


def test_selected_contact_requires_finite_achieved_flight_and_prediction():
    missing_flight = _row(0, hit=True)
    missing_flight["achieved_analytic_flight"] = {
        "evaluated": False, "finite": False, "landing_xy_m": None,
        "landing_valid": False, "net_crossed": False, "net_clear": False,
        "on_opponent_table": False, "source": None,
    }
    with pytest.raises(M.LiveOracleError, match="evaluation differs"):
        M._project_live_episode(missing_flight, output_episode=0)

    hypothetical_miss = _row(0, hit=False)
    hypothetical_miss["achieved_analytic_flight"]["landing_xy_m"] = [2.2, -0.6]
    with pytest.raises(M.LiveOracleError, match="hypothetical analytic flight"):
        M._project_live_episode(hypothetical_miss, output_episode=0)

    with pytest.raises(M.LiveOracleError, match="must not consume hypothetical"):
        M.build_achieved_analytic_evidence(
            selected_rubber_contact=False,
            landing_xy_m=[2.2, -0.6],
        )

    contract = M.runtime_adapter_contract()
    assert contract["physical_ball"] is False
    assert contract["fields"]["actual_selected_rubber_contact"].endswith(
        ".vb_fired[0]"
    )


def test_closed_attempt_requires_real_sampler_receipt_and_positive_rng_interval():
    row = _row(0, hit=False)
    row["sampler_draw_end"] = row["sampler_draw_start"]
    with pytest.raises(M.LiveOracleError, match="positive physical RNG draws"):
        M._project_live_episode(row, output_episode=0)

    row = _row(0, hit=False)
    row["sampler_sample_sha256"] = None
    with pytest.raises(M.LiveOracleError, match="sampler sample SHA"):
        M._project_live_episode(row, output_episode=0)

    question = _question()
    question["physical_rng_draws"] = 0
    with pytest.raises(M.LiveOracleError, match="question contract.*keys differ"):
        M._question_contract(question, bindings={})


def test_wait_contact_cannot_leak_into_task_denominator(tmp_path):
    contract = tmp_path / "training_contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    bindings = {name: SHA for name in M.BINDING_KEYS}
    bindings["hard_contract_sha256"] = hashlib.sha256(
        contract.read_bytes()
    ).hexdigest()
    question = _question()
    wait = _row(0, hit=True, wait_only=True)
    with pytest.raises(M.LiveOracleError, match="WAIT contact"):
        M.collect_live_oracle_bundle(
            object(), object(),
            identity={"action_id": "a", "action_uid": 1, "motion_sha256": SHA},
            bindings=bindings, training_contract_path=contract,
            runner_preflight_facts={}, question_contract=question, episodes=32,
            episode_source=lambda **_kwargs: [wait],
        )


def test_canonical_write_is_no_clobber(tmp_path):
    _runner, _env, payload = _collect(tmp_path)
    target = tmp_path / "c211_observed_oracle_bundle.json"
    pin = M.write_canonical_no_clobber(target, payload)
    assert target.read_bytes() == M.canonical_bytes(payload) + b"\n"
    assert pin["path"] == str(target)
    assert len(pin["sha256"]) == 64
    with pytest.raises(M.LiveOracleError, match="no-clobber"):
        M.write_canonical_no_clobber(target, payload)
    assert not list(tmp_path.glob(".*.tmp"))


def test_runtime_step_adapter_drives_real_policy_loop_for_32_episodes():
    class Base:
        device = "unit-device"
        max_episode_length = 2

    class Env:
        unwrapped = Base()

        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return "obs-0", {}

    class Runner:
        def __init__(self):
            self.policy_devices = []
            self.observations = []

        def get_inference_policy(self, *, device):
            self.policy_devices.append(device)

            def policy(observation):
                self.observations.append(observation)
                return "action-for-" + observation

            return policy

    env, runner = Env(), Runner()
    calls = []

    def step_adapter(**kwargs):
        calls.append(kwargs)
        index = kwargs["source_episode"]
        assert kwargs["actions"] == "action-for-obs-%d" % index
        row = _row(index, hit=index == 1, wait_only=index == 0)
        row["control_steps"] = 1
        row["wait_control_steps"] = int(index == 0)
        row["task_valid_control_steps"] = int(index != 0)
        row["actual_contact"]["runtime_control_step"] = 1
        row["safety"] = _safety(1)
        return {
            "next_observation": "obs-%d" % (index + 1),
            "completed_episode": row,
        }

    rows = M.run_live_policy_episodes(
        runner, env, episodes=32, runtime_step_adapter=step_adapter
    )
    assert env.reset_calls == 1
    assert runner.policy_devices == ["unit-device"]
    assert len(rows) == len(calls) == 33
    assert rows[0]["attempt_closure"]["closed_attempt"] is False
    assert rows[1]["actual_contact"]["selected_face_sweep_contact"] is True
    assert rows[2]["achieved_analytic_flight"]["evaluated"] is False
