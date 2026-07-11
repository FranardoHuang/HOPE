from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_phase1_event_timing_prereg.py"
PREREG = ROOT / "configs" / "phase1_event_timing_t0_t1_prereg_20260711.json"
VENUE = ROOT / "configs" / "phase1_venue_timing_aggregate_20260711.json"
SCHEDULE = ROOT / "configs" / "phase1_event_timing_schedule_spec_20260711.json"
SPEC = importlib.util.spec_from_file_location("validate_phase1_event_timing_prereg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_tracked_prereg_design_passes_and_binds_all_local_inputs():
    prereg, venue, schedule = VALIDATOR.validate_prereg(PREREG, VALIDATOR.sha256_file(PREREG))
    assert prereg["status"] == "preregistered_blocked_on_implementation"
    assert prereg["launch_authorized"] is False
    assert prereg["real_robot_authorized"] is False
    assert venue["source"]["sha256"] == VALIDATOR.EXPECTED_SOURCE_SHA256
    assert schedule["cell_construction"]["contains_1_903_s_target"] is False
    assert all(value is None for value in prereg["implementation_bindings"].values())
    assert set(prereg["implementation_bindings"]) == set(VALIDATOR.REQUIRED_IMPLEMENTATION_BINDINGS)
    frozen = prereg["causal_axis"]["frozen_non_timing_axes"]
    assert frozen["hold_steps_range"] == [0, 100]
    assert frozen["post_swing_start_prob"] == 0.25
    assert frozen["wrap_teleport"] is False
    assert frozen["T1_event_install_draws_extra_random_hold"] is False


def test_cli_design_check_passes_but_launch_check_fails_closed(capsys):
    digest = VALIDATOR.sha256_file(PREREG)
    assert VALIDATOR.main(
        [
            "--prereg",
            str(PREREG),
            "--expected-prereg-sha256",
            digest,
            "--mode",
            "design-check",
        ]
    ) == 0
    design_output = capsys.readouterr().out
    assert '"status": "pass_design_only"' in design_output
    assert '"event_driven_T1_supported": false' in design_output

    assert VALIDATOR.main(
        [
            "--prereg",
            str(PREREG),
            "--expected-prereg-sha256",
            digest,
            "--mode",
            "launch-check",
        ]
    ) == 1
    blocked = capsys.readouterr().err
    assert "LAUNCH BLOCKED" in blocked
    assert "event_scheduler_source" in blocked
    assert "self_hit_instrumentation" in blocked
    assert "semantics_correct_plant_contract" in blocked


def test_prereg_hash_and_baseline_git_blob_bindings_fail_closed(tmp_path: Path):
    with pytest.raises(VALIDATOR.ContractError, match="prereg SHA mismatch"):
        VALIDATOR.validate_prereg(PREREG, "0" * 64)

    payload = _load(PREREG)
    payload["baseline_source"]["git_blobs"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad-blob.json"
    _write_json(path, payload)
    with pytest.raises(VALIDATOR.ContractError, match="git blob SHA mismatch"):
        VALIDATOR.validate_prereg(path, VALIDATOR.sha256_file(path))


def test_venue_report_discloses_bias_and_forbids_using_1_903_as_target():
    venue = _load(VENUE)
    VALIDATOR.validate_venue_report(venue, root=ROOT)
    assert venue["aggregate"]["accepted_samples"] == 21
    assert venue["aggregate"]["coarse_take_category_counts"]["gaoqiu"] == 16
    assert venue["known_biases"]["overlapping_samples"] is True
    assert venue["known_biases"]["samples_independent"] is False
    assert venue["known_biases"]["effective_sample_size"] == "unknown_but_less_than_21"
    assert venue["known_biases"]["max_leg_filter_right_censors_slow_legs"] is True
    assert venue["known_biases"]["right_censor_threshold_s"] == 2.5
    assert venue["use_policy"]["median_1_903_s_is_target"] is False
    assert venue["source"]["raw_rows_or_take_ids_in_this_report"] is False

    tampered = json.loads(json.dumps(venue))
    tampered["use_policy"]["median_1_903_s_is_target"] = True
    with pytest.raises(VALIDATOR.ContractError, match="must never become"):
        VALIDATOR.validate_venue_report(tampered, root=ROOT)


def test_venue_reproduction_refuses_unbound_raw_source(tmp_path: Path):
    fake = tmp_path / "strikes.json"
    fake.write_text("[]\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.ContractError, match="restored strikes SHA mismatch"):
        VALIDATOR.reproduce_venue(_load(VENUE), fake)


def test_engineering_schedule_is_balanced_not_venue_fitted_and_q10_is_screen_only():
    schedule = _load(SCHEDULE)
    VALIDATOR.validate_schedule_spec(schedule)
    cells = schedule["timing_cells"]
    assert [
        (cell["cell_id"], cell["reveal_ticks_after_prior_strike"], cell["next_strike_ticks_after_prior_strike"])
        for cell in cells
    ] == list(VALIDATOR.EXPECTED_TIMING_CELLS)
    assert schedule["cell_construction"]["fit_to_venue_data"] is False
    assert schedule["cell_construction"]["venue_quantiles_used_as_targets_or_weights"] is False
    assert schedule["materialization"]["screen_q10"] == {
        "questions_per_side": 10,
        "scheduled_opportunities": 20,
        "sequence_length_opportunities": 10,
        "sequence_count": 2,
        "role": "screen_only",
        "may_stop_or_promote": False,
    }
    assert schedule["materialization"]["decision_q50"]["questions_per_side"] == 50
    assert schedule["immutable_engine_contract"]["engines"] == ["Isaac", "MuJoCo"]
    assert schedule["immutable_engine_contract"]["fresh_exact_only"] is True
    assert schedule["safety"]["any_self_hit_closes_cell"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["timing_cell_seconds_derived_at_50_hz"][0].__setitem__("next_strike_s", 1.903),
            "derived timing seconds|prohibited 1.903",
        ),
        (
            lambda value: value["materialization"]["screen_q10"].__setitem__("may_stop_or_promote", True),
            "q10 must remain",
        ),
        (
            lambda value: value["safety"].__setitem__("racket_or_handle_self_contact_allowed", True),
            "self-hit must be prohibited",
        ),
        (
            lambda value: value["event_semantics"].__setitem__("deadline_shift_allowed", True),
            "deadline/reset semantics",
        ),
        (
            lambda value: value["immutable_engine_contract"].__setitem__("allow_inexact_contract", True),
            "fresh exact only",
        ),
    ],
)
def test_schedule_red_team_mutations_fail(mutation, message):
    value = _load(SCHEDULE)
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        VALIDATOR.validate_schedule_spec(value)


def test_axis_rejects_topp_reward_plant_face_and_q10_governance_leaks(tmp_path: Path):
    mutations = [
        (lambda p: p["causal_axis"]["frozen_non_timing_axes"].__setitem__("motion_path_and_time_law", "TOPP_v3"), "not frozen"),
        (lambda p: p["causal_axis"]["frozen_non_timing_axes"].__setitem__("reward", "new_ready_reward"), "not frozen"),
        (lambda p: p["causal_axis"]["frozen_non_timing_axes"].__setitem__("plant", "different_per_arm"), "not frozen"),
        (lambda p: p["causal_axis"]["frozen_non_timing_axes"].__setitem__("face_pairing", "legacy_signed"), "not frozen"),
        (lambda p: p["checkpoint_and_decision_policy"].__setitem__("q10_may_stop_or_promote", True), "q10 must be screen-only"),
        (lambda p: p["lineage_and_checkpoint_selection"].__setitem__("allow_inexact_contract", True), "inexact escape"),
        (lambda p: p["evaluation_contract"].__setitem__("self_hit_hard_failure", False), "Self-hit|self-hit"),
    ]
    for index, (mutation, message) in enumerate(mutations):
        payload = _load(PREREG)
        mutation(payload)
        path = tmp_path / f"bad-axis-{index}.json"
        _write_json(path, payload)
        with pytest.raises(VALIDATOR.ContractError, match=message):
            VALIDATOR.validate_prereg(path, VALIDATOR.sha256_file(path))


def test_blocked_prereg_cannot_sneak_in_one_partial_implementation_binding(tmp_path: Path):
    payload = _load(PREREG)
    payload["implementation_bindings"]["event_scheduler_source"] = {
        "repo_path": "unreviewed.py",
        "sha256": "0" * 64,
        "bytes": 1,
    }
    path = tmp_path / "partial.json"
    _write_json(path, payload)
    with pytest.raises(VALIDATOR.ContractError, match="must keep all unimplemented bindings null"):
        VALIDATOR.validate_prereg(path, VALIDATOR.sha256_file(path))


def test_actual_capability_gap_and_reference_lineage_are_explicit():
    prereg = _load(PREREG)
    gap = prereg["observed_code_capability_gap"]
    assert gap["event_driven_T1_supported"] is False
    assert gap["post_strike_only_event_trigger"] is False
    assert gap["atomic_next_question_install"] is False
    assert gap["external_deadline_independent_of_clip_phase"] is False
    assert gap["continuous_immutable_Isaac_judge"] is False
    assert gap["continuous_immutable_MuJoCo_judge"] is False
    assert gap["racket_handle_self_hit_gate"] is False
    assert gap["existing_partial_mechanisms"]["post_swing_state_ring_buffer_for_true_resets"] is True
    assert gap["existing_partial_mechanisms"]["midswing_target_where_refinement"] is True
    assert all(item["causal_conclusion_reusable"] is False for item in prereg["local_reference_commits"])
    assert {item["commit"] for item in prereg["local_reference_commits"]} >= {
        "c951d9dcc8b73c30c6a801afb00a01716ee81baa",
        "5c346eac588bb137a90b1a646e71ee44cc805a71",
        "baf62157f3db42e4851976fbc3d7a334f47bc919",
        "287efc4cfd99b034d612c73668433d323c35f558",
        "5e975048d5086863d46095d301357d79d55a3c69",
    }
