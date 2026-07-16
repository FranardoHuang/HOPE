from __future__ import annotations

import copy
import base64
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_task_revision_supercombo_queue.py"
QUEUE = ROOT / "configs" / "phase1_task_revision_supercombo_20260716.yaml"
OLD_QUEUE = ROOT / "configs" / "phase1_rolling_timing_supercombo_20260716.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("task_revision_queue_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _raw_queue() -> dict:
    value = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, queue: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return path


def _revision(job: dict) -> dict:
    overrides = Q._compiled_overrides(job)
    argument = overrides["task.planner_revision"]
    return Q._override_hydra_mapping(
        argument, key="task.planner_revision", job_id=job["id"]
    )


def test_planner_revision_hydra_mapping_is_canonical_typed_and_rejects_json_keys():
    queue = Q.load_queue(QUEUE)
    documents = []
    for job in queue["jobs"]:
        overrides = Q._compiled_overrides(job)
        argument = overrides["task.planner_revision"]
        raw = argument.split("=", 1)[1]
        document = _revision(job)
        assert raw == Q._hydra_literal(document)
        assert type(document["enabled"]) is bool
        assert type(document["profile"]["schema_version"]) is int
        assert type(document["profile"]["early_deadline_tolerance_s"]) is float
        assert type(document["initial_tts_mixture"]["components"]) is list
        documents.append(json.dumps(document, allow_nan=False, sort_keys=True))
    assert len(set(documents)) == 6

    legacy = '++task.planner_revision={"enabled":true}'
    with pytest.raises(Q.SuccessorQueueError, match="canonical Hydra mapping"):
        Q._override_hydra_mapping(
            legacy, key="task.planner_revision", job_id="legacy_json"
        )


@pytest.mark.parametrize("value", [1.0e-6, 1.0e6, -1.0e-6, -0.0])
def test_hydra_float_literal_round_trips_as_float(value):
    rendered = Q._hydra_literal({"value": value})
    parsed = yaml.safe_load(rendered)
    assert type(parsed["value"]) is float
    assert Q._hydra_literal(parsed) == rendered


def _first_launchable(queue: dict) -> dict:
    return next(job for job in queue["jobs"] if job["scientific_launch_authorized"])


def _as_pending(queue: dict) -> dict:
    """Return a synthetic pre-activation state for fail-closed negative tests."""

    queue["launch_authorized"] = False
    queue["preregistration_status"] = Q.PENDING_STATUS
    queue["namespace_contract"]["status"] = Q.PENDING_STATUS
    blocking = queue["blocking_contract"]
    blocking["source_full_scene_probe_evidence"] = "PENDING_FULL_SCENE_PROBE"
    blocking["task_revision_full_scene_probe_evidence"] = (
        "PENDING_TASK_REVISION_PROBE"
    )
    blocking["hotstart_harness"] = "PENDING_REVIEWED_SUCCESSOR_HARNESS_BINDING"
    for job in queue["jobs"]:
        job["status"] = "blocked"
        job["blocker"] = (
            Q.TRANSPORT_BLOCKER
            if not job["scientific_launch_authorized"]
            else "PENDING synthetic activation evidence"
        )
    return queue


def _activated_value() -> dict:
    value = _raw_queue()
    commit = "a" * 40
    checkout = "/workspace/codexschema/nohope_task_revision_source_a"
    value["launch_authorized"] = True
    value["preregistration_status"] = Q.ACTIVATED_STATUS
    value["namespace_contract"]["status"] = "activated_no_clobber"
    blocking = value["blocking_contract"]
    blocking["source_checkout"] = checkout
    blocking["source_commit"] = commit
    blocking["source_full_scene_probe_evidence"] = {
        "training_runtime_status": "passed_natural_exit_rc0",
        "first_iteration_observed": True,
        "tensor_nonfinite_count": 0,
        "fatal_count": 0,
        "training_contract_lineage_exact": 1,
        "process_group_naturally_empty": True,
        "checkpoint_iteration": 2,
        "checkpoint_sha256": "b" * 64,
        "hard_contract_sha256": "c" * 64,
    }
    _runner, runner_sha = Q.continuation._runner_payload()
    blocking["hotstart_harness"] = {
        "runner_script_sha256": runner_sha,
        "reviewed_tests_passed": True,
        "reviewed_test_count": 100,
    }
    blocking["task_revision_full_scene_probe_evidence"] = {
        "status": "passed",
        "unlock_authorized": True,
        "representative_job_id": value["full_scene_probe_contract"]["representative_job_id"],
        "initial_tts_mixture_all_strata_observed": True,
        "planner_revision_attempt_accept_reject_accounting_exact": True,
        "planner_revision_accepted_observed": True,
        "planner_revision_last_precontact_accepted_observed": True,
        "planner_revision_actor_visible_observed": True,
        "planner_revision_last_precontact_actor_visible_observed": True,
        "exact_behavior_ledger_schema_complete": True,
        "receipt_path": "/workspace/probe/task_revision_probe_result.json",
        "receipt_file_sha256": "d" * 64,
        "receipt_content_sha256": "e" * 64,
    }
    for job in value["jobs"]:
        job["source"]["checkout"] = checkout
        job["source"]["commit"] = commit
        if job["scientific_launch_authorized"]:
            job["status"] = "ready"
            job["blocker"] = None
        else:
            job["status"] = "blocked"
            job["blocker"] = Q.TRANSPORT_BLOCKER
    return value


def test_checked_in_queue_is_activated_and_ready():
    queue = Q.load_queue(QUEUE)
    result = Q.cmd_validate(queue)
    assert result["activation_ready"] is True
    assert result["pending"] is False
    assert result["blockers"] == []
    assert sum(job["status"] == "ready" for job in queue["jobs"]) == 22
    assert sum(job["status"] == "blocked" for job in queue["jobs"]) == 2


def test_synthetic_pending_queue_is_valid_but_not_activation_ready():
    queue = _as_pending(Q.load_queue(QUEUE))
    result = Q.cmd_validate(queue)
    assert result["schema_valid"] is True
    assert result["successor_contract_valid"] is True
    assert result["activation_ready"] is False
    assert result["pending"] is True
    assert result["job_count"] == 24
    assert result["distinct_scientific_cells"] == 24
    assert result["launchable_job_count"] == 22
    assert result["transport_blocked_job_count"] == 2
    assert set(result["transport_blocked_job_ids"]) == Q.TRANSPORT_BLOCKED_JOB_IDS
    assert result["formal_evidence_eligible"] is False
    assert all(job["status"] == "blocked" for job in queue["jobs"])
    assert not any("source_commit is pending" in row for row in result["blockers"])
    assert any("source_full_scene_probe_evidence" in row for row in result["blockers"])
    assert any("22 jobs remain blocked" in row for row in result["blockers"])


def test_delay_pair_is_permanently_filtered_from_launch_surface(tmp_path):
    queue = Q.load_queue(QUEUE)
    filtered = Q._launchable_continuation_queue(queue)
    assert len(filtered["jobs"]) == 22
    assert not (set(job["id"] for job in filtered["jobs"]) & Q.TRANSPORT_BLOCKED_JOB_IDS)
    blocked = [job for job in queue["jobs"] if not job["scientific_launch_authorized"]]
    assert {job["id"] for job in blocked} == Q.TRANSPORT_BLOCKED_JOB_IDS
    assert {job["scientific_blocker"] for job in blocked} == {Q.TRANSPORT_BLOCKER}

    value = _raw_queue()
    next(job for job in value["jobs"] if job["id"] in Q.TRANSPORT_BLOCKED_JOB_IDS)[
        "scientific_launch_authorized"
    ] = True
    with pytest.raises(Q.SuccessorQueueError, match="NO-LAUNCH"):
        Q.load_queue(_write(tmp_path, value))


def test_no_third_cell_may_be_hidden_as_transport_blocked(tmp_path):
    value = _raw_queue()
    third = next(job for job in value["jobs"] if job["id"] not in Q.TRANSPORT_BLOCKED_JOB_IDS)
    third["scientific_launch_authorized"] = False
    third["scientific_blocker"] = Q.TRANSPORT_BLOCKER
    with pytest.raises(Q.SuccessorQueueError, match="delay-zero cell"):
        Q.load_queue(_write(tmp_path, value))


def test_activated_adapter_delegates_exactly_22_launchable_cells(tmp_path, monkeypatch):
    queue = Q.load_queue(_write(tmp_path, _activated_value()))
    validated = Q.cmd_validate(queue)
    assert validated["activation_ready"] is True
    captured = {}

    def fake_fill(delegated, *, count, execute, confirm):
        captured["job_ids"] = [job["id"] for job in delegated["jobs"]]
        captured["args"] = (count, execute, confirm)
        return {"mode": "fill", "dry_run": True, "jobs": []}

    monkeypatch.setattr(Q.continuation, "cmd_fill", fake_fill)
    Q.cmd_fill(queue, count=24, execute=False, confirm=None)
    assert len(captured["job_ids"]) == 22
    assert not (set(captured["job_ids"]) & Q.TRANSPORT_BLOCKED_JOB_IDS)
    assert captured["args"] == (24, False, None)


def test_queue_loader_rejects_duplicate_yaml_keys_before_safe_load_last_wins(tmp_path):
    raw = QUEUE.read_text(encoding="utf-8")
    assert raw.count("  max_iterations: 2001\n") == 1
    duplicate = raw.replace(
        "  max_iterations: 2001\n",
        "  max_iterations: 2001\n  max_iterations: 2001\n",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(Q.SuccessorQueueError, match="duplicate YAML key"):
        Q.load_queue(path)


def test_same_pod_parent_records_are_byte_fields_copied_from_frozen_predecessor():
    queue = _raw_queue()
    predecessor = yaml.safe_load(OLD_QUEUE.read_text(encoding="utf-8"))
    keys = (
        "original_queue_claim_path",
        "original_queue_claim_sha256",
        "original_run_binding_path",
        "original_run_binding_sha256",
        "selected_rsl_log_dir",
        "selected_checkpoint_path",
        "selected_embedded_iteration",
        "selected_checkpoint_sha256",
        "selected_hard_contract_path",
        "selected_hard_contract_sha256",
    )
    for parent in ("pod1_local_best", "pod2_quality_basin", "pod2_continuous_basin"):
        assert {key: queue["parent_selection"][parent][key] for key in keys} == {
            key: predecessor["parent_selection"][parent][key] for key in keys
        }
    assert {
        parent: sum(job["warm_start"]["parent"] == parent for job in queue["jobs"])
        for parent in ("pod1_local_best", "pod2_quality_basin", "pod2_continuous_basin")
    } == {"pod1_local_best": 12, "pod2_quality_basin": 6, "pod2_continuous_basin": 6}
    for job in queue["jobs"]:
        pod = job["resource"]["required_slot"].split("/", 1)[0]
        assert job["warm_start"]["parent"].startswith(pod + "_")


def test_four_rounds_cover_six_gpus_once_each_and_absolute_milestones_are_reachable():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    slots = list(Q.continuation.EXPECTED_SLOTS)
    assert len(plan["jobs"]) == 24
    assert [row["required_slot"] for row in plan["jobs"]] == slots * 4
    assert [row["launch_round"] for row in plan["jobs"]] == [
        launch_round for launch_round in range(1, 5) for _ in range(6)
    ]
    for row in plan["jobs"]:
        assert row["milestones"] == [
            row["parent_iteration"] + offset for offset in (200, 500, 1000, 2000)
        ]
        assert row["absolute_iteration_exclusive_bound"] == row["parent_iteration"] + 2001
        assert row["formal_evidence_eligible"] is False
        assert row["human_name"]
        assert row["scientific_question"]


def test_every_cell_has_complete_revision_profile_real_0p5_mass_and_sub_0p5_stress():
    queue = Q.load_queue(QUEUE)
    noise = set()
    mixtures = set()
    for job in queue["jobs"]:
        revision = _revision(job)
        assert set(revision) == Q.EXPECTED_REVISION_KEYS
        assert revision["enabled"] is True
        assert set(revision["profile"]) == Q.EXPECTED_PROFILE_KEYS
        assert revision["profile"]["policy_dt_s"] == 0.02
        assert revision["profile"]["min_tts_s"] == revision["profile"]["policy_dt_s"]
        assert revision["profile"]["early_deadline_tolerance_s"] == 1.0e-6
        assert revision["initial_tts_range_s"] == [0.36, 1.7]
        components = revision["initial_tts_mixture"]["components"]
        assert [(row["name"], row["range_s"]) for row in components] == [
            ("late_stress", [0.36, 0.49]),
            ("baseline_0p5", [0.5, 0.5]),
            ("fast_deploy", [0.5, 0.9]),
            ("broad_arrival", [0.9, 1.7]),
        ]
        assert sum(row["weight"] for row in components) == pytest.approx(1.0)
        assert all(row["weight"] > 0.0 for row in components)
        mixtures.add(tuple(row["weight"] for row in components))
        noise.add(
            tuple(
                revision[key]
                for key in (
                    "position_std_m",
                    "velocity_std_mps",
                    "normal_std_rad",
                    "tts_std_s",
                )
            )
        )
    assert noise == Q.ALLOWED_NOISE
    assert mixtures == {
        (0.15, 0.2, 0.3, 0.35),
        (0.2, 0.3, 0.35, 0.15),
        (0.1, 0.15, 0.25, 0.5),
    }


def test_all_cells_keep_all_revision_fields_live_through_final_precontact_tick():
    queue = Q.load_queue(QUEUE)
    execution = queue["task_revision_execution_contract"]
    assert execution["actor_visible_revision_fields"] == [
        "target_position",
        "target_velocity",
        "signed_target_normal",
        "time_to_strike",
    ]
    assert execution["update_window"] == "every_policy_tick_through_last_pre_contact_tick"
    assert execution["policy_tick_s"] == 0.02
    for job in queue["jobs"]:
        profile = _revision(job)["profile"]
        assert profile["min_tts_s"] == profile["policy_dt_s"] == 0.02


def test_all_flattened_recipes_disable_all_three_legacy_hold_clocks():
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        overrides = Q._compiled_overrides(job)
        assert {
            key: overrides[key] for key in Q.REQUIRED_SINGLE_CLOCK_OVERRIDES
        } == Q.REQUIRED_SINGLE_CLOCK_OVERRIDES


def test_all_flattened_recipes_keep_clip_switch_probability_zero():
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        overrides = Q._compiled_overrides(job)
        assert {
            key: overrides[key] for key in Q.REQUIRED_TASK_IDENTITY_OVERRIDES
        } == Q.REQUIRED_TASK_IDENTITY_OVERRIDES


def test_only_timestamp_pair_uses_two_step_delay_and_probe_representative_is_same_tick():
    queue = Q.load_queue(QUEUE)
    representative = queue["full_scene_probe_contract"]["representative_job_id"]
    for job in queue["jobs"]:
        override = Q._compiled_overrides(job)["task.racket.target_delay_steps"]
        if job.get("timestamp_role"):
            assert override == "task.racket.target_delay_steps=2"
        else:
            assert override == "task.racket.target_delay_steps=0"
        if job["id"] == representative:
            assert override == "task.racket.target_delay_steps=0"


def test_portfolio_has_24_unique_recipes_and_omits_fake_half_second_and_blocked_force():
    queue = Q.load_queue(QUEUE)
    signatures = []
    qdot = set()
    for job in queue["jobs"]:
        overrides = Q._compiled_overrides(job)
        signatures.append(tuple(sorted(overrides.items())))
        assert "task.motion.speed_scale_per_clip" not in overrides
        assert not any("lateral_perturb" in key or "external_force" in key for key in overrides)
        qdot.add(overrides["task.rewards.joint_velocity_limit_hinge_weight"])
    assert len(set(signatures)) == 24
    assert qdot == {
        "task.rewards.joint_velocity_limit_hinge_weight=-5.0",
        "task.rewards.joint_velocity_limit_hinge_weight=-2.5",
        "task.rewards.joint_velocity_limit_hinge_weight=0.0",
    }
    assert queue["pruning_contract"]["freed_slot_policy"]["stochastic_torso_push_status"] == (
        "omitted_until_E1_full_scene_launch_gate_passes"
    )


def test_timestamp_compensation_negative_differs_from_treatment_only_in_mode():
    queue = Q.load_queue(QUEUE)
    by_id = {job["id"]: job for job in queue["jobs"]}
    treatment = by_id["taskrev_p1_core_low_noise"]
    negative = by_id["taskrev_p1_uncompensated_negative"]
    left = Q._compiled_overrides(treatment)
    right = Q._compiled_overrides(negative)
    key = "task.racket.target_delay_tts_mode"
    assert left.pop(key) == "++task.racket.target_delay_tts_mode=source_timestamp_compensated"
    assert right.pop(key) == "++task.racket.target_delay_tts_mode=uncompensated"
    assert left["task.racket.target_delay_steps"] == "task.racket.target_delay_steps=2"
    assert right["task.racket.target_delay_steps"] == "task.racket.target_delay_steps=2"
    assert left == right
    assert treatment["warm_start"]["parent"] == negative["warm_start"]["parent"]


def test_pruning_uses_closeout_completion_and_exact_denominators_not_window_start_strike():
    queue = Q.load_queue(QUEUE)
    ledger = queue["exact_behavior_ledger_contract"]
    required = set(ledger["required_counters"])
    assert {"swing_outcome_count", "swing_completion_count"}.issubset(required)
    assert {
        "termination_reason_base_fell_tilt_count",
        "termination_reason_base_too_low_count",
    }.issubset(required)
    assert "swing_completion_count_is_between_zero_and_swing_outcome_count" in ledger[
        "invariants"
    ]
    assert (
        "swing_start_and_strike_opportunity_are_diagnostic_and_may_cross_window_boundaries"
        in ledger["invariants"]
    )
    pruning = queue["pruning_contract"]
    assert pruning["behavior_decision_requires_two_disjoint_complete_windows"] is True
    assert pruning["sparse_zero_without_positive_eligible_denominator_may_stop"] is False
    assert pruning["plus_500"]["required_complete_windows"] == 2
    assert pruning["plus_500"]["window_updates"] == 100
    assert pruning["plus_500"]["sparse_outcome_alone_may_stop"] is False
    assert "swing_closeout_completion_rate" in pruning["plus_1000"]["pareto_metrics"][
        "maximize"
    ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].__setitem__(
                0,
                q["jobs"][0]["recipe"]["delta"][0].replace(
                    'name: "baseline_0p5", range_s: [0.5, 0.5]',
                    'name: "baseline_0p5", range_s: [0.5, 0.6]',
                ),
            ),
            "baseline_0p5",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].append(
                "++task.motion.speed_scale_per_clip=[2.64,1.8]"
            ),
            "fake fixed-retiming",
        ),
        (
            lambda q: q["jobs"][0].update(
                scientific_question=q["jobs"][1]["scientific_question"]
            ),
            "scientific questions",
        ),
        (
            lambda q: q["exact_behavior_ledger_contract"]["required_counters"].remove(
                "swing_outcome_count"
            ),
            "swing_outcome_count",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["base"].__setitem__(
                q["jobs"][0]["recipe"]["base"].index(
                    "task.motion.hold_steps_range=[0,0]"
                ),
                "task.motion.hold_steps_range=[0,100]",
            ),
            "legacy hold clock",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].append(
                "task.motion.stand_start_min_hold=25"
            ),
            "legacy hold clock",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].append(
                "task.motion.post_swing_min_hold=25"
            ),
            "legacy hold clock",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].append(
                "task.motion.clip_switch_prob=0.1"
            ),
            "same-task side/clip identity",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].__setitem__(
                0,
                q["jobs"][0]["recipe"]["delta"][0].replace(
                    "early_deadline_tolerance_s: 1.0e-6",
                    "early_deadline_tolerance_s: 1.0e-9",
                ),
            ),
            "early_deadline_tolerance_s must be exact 1e-6",
        ),
        (
            lambda q: q["jobs"][1]["recipe"]["delta"].append(
                "task.racket.target_delay_steps=1"
            ),
            "target_delay_steps must be 0",
        ),
        (
            lambda q: q["jobs"][0]["recipe"]["delta"].__setitem__(
                1, "task.racket.target_delay_steps=1"
            ),
            "target_delay_steps must be 2",
        ),
    ],
)
def test_successor_contract_fails_closed_on_scientific_drift(tmp_path, mutate, match):
    queue = copy.deepcopy(_raw_queue())
    mutate(queue)
    with pytest.raises(Q.SuccessorQueueError, match=match):
        Q.load_queue(_write(tmp_path, queue))


def test_pending_fill_fails_before_any_ssh(monkeypatch):
    queue = _as_pending(Q.load_queue(QUEUE))
    calls = []
    monkeypatch.setattr(
        Q.continuation.lean,
        "_run_ssh",
        lambda *_args, **_kwargs: calls.append("ssh"),
    )
    with pytest.raises(Q.SuccessorQueueError, match="successor fill is blocked"):
        Q.cmd_fill(queue, count=1, execute=False, confirm=None)
    assert calls == []


def test_successor_exposes_reviewed_source_asset_prepare_without_activation(monkeypatch):
    queue = Q.load_queue(QUEUE)
    calls = []

    def fake_prepare(value, *, job_id, pod, execute, confirm):
        calls.append((value, job_id, pod, execute, confirm))
        return {"mode": "prepare-source-assets", "dry_run": not execute}

    monkeypatch.setattr(Q.continuation.lean, "cmd_prepare_source_assets", fake_prepare)
    result = Q.cmd_prepare_source_assets(
        queue,
        job_id="taskrev_p1_core_high_noise",
        pod="pod1",
        execute=False,
        confirm=None,
    )
    assert result == {"mode": "prepare-source-assets", "dry_run": True}
    assert calls == [
        (queue, "taskrev_p1_core_high_noise", "pod1", False, None)
    ]


def test_transport_blocked_job_cannot_prepare_successor_asset(monkeypatch):
    queue = Q.load_queue(QUEUE)
    monkeypatch.setattr(
        Q.continuation.lean,
        "cmd_prepare_source_assets",
        lambda *_args, **_kwargs: pytest.fail("blocked job delegated"),
    )
    with pytest.raises(Q.SuccessorQueueError, match="scientific NO-LAUNCH"):
        Q.cmd_prepare_source_assets(
            queue,
            job_id="taskrev_p1_core_low_noise",
            pod="pod1",
            execute=False,
            confirm=None,
        )


def test_generic_probe_pass_cannot_bypass_specialized_probe_blocker(monkeypatch):
    queue = Q.load_queue(QUEUE)
    queue["blocking_contract"]["task_revision_full_scene_probe_evidence"] = (
        "PENDING_TASK_REVISION_PROBE"
    )
    monkeypatch.setattr(Q.continuation, "activation_blockers", lambda _queue: [])
    blockers = Q.successor_activation_blockers(queue)
    assert blockers == [
        "task_revision_full_scene_probe_evidence must be a pass mapping"
    ]


def _exact_counters(*, outcome: int = 10, completion: int = 1) -> dict:
    physical = 2 if outcome else 0
    strike = 4 if outcome else 0
    ready = 10 if outcome else 0
    return {
        "swing_start_count": outcome,
        "swing_outcome_count": outcome,
        "swing_completion_count": completion if outcome else 0,
        "strike_opportunity_count": strike,
        "virtual_capture_count": 2 if strike else 0,
        "virtual_net_clear_count": 2 if strike else 0,
        "virtual_landing_valid_count": 1 if strike else 0,
        "virtual_legal_return_count": 1 if strike else 0,
        "physical_fall_count": physical,
        "pre_strike_physical_fall_count": physical // 2,
        "post_strike_physical_fall_count": physical - physical // 2,
        "non_physical_terminal_reset_count": 0,
        "termination_reason_base_fell_tilt_count": physical // 2,
        "termination_reason_base_too_low_count": physical - physical // 2,
        "ready_tilt_eligible_sample_count": ready,
        "ready_tilt_rad_sum": 1.0 if ready else 0.0,
        "ready_base_speed_eligible_sample_count": ready,
        "ready_base_speed_xy_mps_sum": 2.0 if ready else 0.0,
        "ready_station_offset_eligible_sample_count": ready,
        "ready_station_offset_m_sum": 1.0 if ready else 0.0,
        "ready_foot_contact_eligible_sample_count": ready,
        "ready_foot_contact_fraction_sum": 9.0 if ready else 0.0,
        "ready_foot_slip_eligible_sample_count": ready,
        "ready_foot_slip_speed_mps_sum": 1.0 if ready else 0.0,
        "planner_initial_tts_sample_count": 4,
        "planner_initial_tts_sub_0p5_count": 1,
        "planner_initial_tts_exact_0p5_count": 1,
        "planner_initial_tts_above_0p5_count": 2,
        "planner_initial_tts_component_0_count": 1,
        "planner_initial_tts_component_1_count": 1,
        "planner_initial_tts_component_2_count": 1,
        "planner_initial_tts_component_3_count": 1,
        "planner_revision_attempt_count": 10,
        "planner_revision_accepted_count": 8,
        "planner_revision_rejected_count": 2,
        "planner_revision_last_precontact_attempt_count": 1,
        "planner_revision_last_precontact_accepted_count": 1,
        "planner_revision_actor_visible_count": 8,
        "planner_revision_last_precontact_actor_visible_count": 1,
    }


def _exact_record(update: int, *, counters: dict | None = None, term="racket_target") -> dict:
    counters = copy.deepcopy(counters or _exact_counters())
    return {
        "event": "hope_exact_behavior_update",
        "schema_version": 1,
        "ppo_update": update,
        "term": term,
        "counters": counters,
        "derived": Q._derived_behavior(counters),
        "window_aggregation": "sum_counters_then_recompute_derived",
    }


def _exact_log(records: list[dict]) -> bytes:
    return ("\n".join(
        Q.EXACT_EVENT_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in records
    ) + "\n").encode()


def test_exact_behavior_two_complete_windows_can_make_only_registered_dense_collapse_stop():
    queue = Q.load_queue(QUEUE)
    required = Q._required_behavior_counters(queue)
    parsed = Q.parse_exact_behavior_log(
        _exact_log([_exact_record(update) for update in range(301, 501)]),
        required_counters=required,
    )
    result = Q.analyze_behavior_windows(
        parsed, milestone=500, milestone_offset=500, required_counters=required
    )
    assert [window["update_count"] for window in result["windows"]] == [100, 100]
    assert result["decision"] == "stop_clear_dense_collapse"
    assert result["stop_execution"] == "manual_reviewed_exact_consumer_only"


def test_zero_eligible_denominators_are_null_and_never_stop():
    queue = Q.load_queue(QUEUE)
    counters = _exact_counters(outcome=0, completion=0)
    parsed = Q.parse_exact_behavior_log(
        _exact_log([_exact_record(update, counters=counters) for update in range(301, 501)]),
        required_counters=Q._required_behavior_counters(queue),
    )
    result = Q.analyze_behavior_windows(
        parsed,
        milestone=500,
        milestone_offset=500,
        required_counters=Q._required_behavior_counters(queue),
    )
    assert result["decision"] == "continue_training_no_automatic_stop"
    assert result["windows"][0]["derived"]["swing_completion_rate"] is None
    assert result["windows"][0]["derived"]["virtual_legal_return_per_strike"] is None


def test_ready_improvement_prevents_dense_collapse_stop():
    queue = Q.load_queue(QUEUE)
    records = [_exact_record(update) for update in range(301, 501)]
    for record in records[100:]:
        record["counters"]["ready_tilt_rad_sum"] = 0.1
        record["derived"] = Q._derived_behavior(record["counters"])
    parsed = Q.parse_exact_behavior_log(
        _exact_log(records), required_counters=Q._required_behavior_counters(queue)
    )
    result = Q.analyze_behavior_windows(
        parsed,
        milestone=500,
        milestone_offset=500,
        required_counters=Q._required_behavior_counters(queue),
    )
    assert result["decision"] == "continue_training_no_automatic_stop"
    assert any("ready_metric_improved:ready_tilt_rad_mean" in row for row in result["decision_reasons"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows.append(copy.deepcopy(rows[-1])), "duplicate exact behavior update"),
        (lambda rows: rows[-1].update(term="other_provider"), "one racket_target provider"),
        (lambda rows: rows.pop(50), "checkpoint alignment"),
        (lambda rows: rows[-1].update(derived={}), "derived values drifted"),
        (
            lambda rows: rows[-1]["counters"].__setitem__(
                "planner_initial_tts_component_3_count", 0
            ),
            "mixture accounting invariant",
        ),
        (
            lambda rows: rows[-1]["counters"].__setitem__(
                "planner_revision_rejected_count", 3
            ),
            "planner-revision accounting invariant",
        ),
        (
            lambda rows: rows[-1]["counters"].pop(
                "termination_reason_base_fell_tilt_count"
            ),
            "missing counters",
        ),
    ],
)
def test_exact_behavior_consumer_fails_closed_on_drift(mutate, match):
    queue = Q.load_queue(QUEUE)
    rows = [_exact_record(update) for update in range(301, 501)]
    mutate(rows)
    raw = _exact_log(rows)
    if match == "checkpoint alignment":
        parsed = Q.parse_exact_behavior_log(
            raw, required_counters=Q._required_behavior_counters(queue)
        )
        with pytest.raises(Q.SuccessorQueueError, match=match):
            Q.analyze_behavior_windows(
                parsed,
                milestone=500,
                milestone_offset=500,
                required_counters=Q._required_behavior_counters(queue),
            )
    else:
        with pytest.raises(Q.SuccessorQueueError, match=match):
            Q.parse_exact_behavior_log(
                raw, required_counters=Q._required_behavior_counters(queue)
            )


def test_specialized_probe_requires_mixture_and_last_precontact_revision_activation():
    queue = Q.load_queue(QUEUE)
    required = Q._required_behavior_counters(queue)
    parsed = Q.parse_exact_behavior_log(
        _exact_log([_exact_record(0), _exact_record(1)]), required_counters=required
    )
    evidence = Q.validate_task_revision_probe_records(
        parsed, required_counters=required
    )
    assert evidence["exact_update_ids"] == [0, 1]
    assert evidence["aggregate_counters"]["planner_revision_accepted_count"] > 0
    assert evidence["aggregate_counters"][
        "planner_revision_last_precontact_accepted_count"
    ] > 0

    counters = _exact_counters()
    counters["planner_revision_last_precontact_attempt_count"] = 0
    counters["planner_revision_last_precontact_accepted_count"] = 0
    counters["planner_revision_last_precontact_actor_visible_count"] = 0
    parsed = Q.parse_exact_behavior_log(
        _exact_log([_exact_record(0, counters=counters), _exact_record(1, counters=counters)]),
        required_counters=required,
    )
    with pytest.raises(Q.SuccessorQueueError, match="last_precontact"):
        Q.validate_task_revision_probe_records(parsed, required_counters=required)


def test_full_scene_probe_delegates_to_reviewed_generic_harness(monkeypatch):
    queue = _raw_queue()
    representative = next(
        job
        for job in queue["jobs"]
        if job["id"] == queue["full_scene_probe_contract"]["representative_job_id"]
    )
    calls = []

    def fake(delegated_queue, **kwargs):
        calls.append((delegated_queue, kwargs))
        return {"run_dir": "/tmp/probe", "mode": "full-scene-probe"}

    monkeypatch.setattr(Q.continuation.lean, "cmd_full_scene_probe", fake)
    result = Q.cmd_full_scene_probe(
        queue,
        job_id=representative["id"],
        pod="pod1",
        gpu=1,
        attempt_id="attempt1",
        execute=True,
        confirm=Q.FULL_SCENE_PROBE_CONFIRM,
    )
    delegated_queue, delegated_kwargs = calls[0]
    assert delegated_kwargs["confirm"] == Q.continuation.lean.FULL_SCENE_PROBE_CONFIRM
    delegated_job = next(
        job for job in delegated_queue["jobs"] if job["id"] == representative["id"]
    )
    planner_argument = Q._compiled_overrides(delegated_job)["task.planner_revision"]
    assert planner_argument == Q._compiled_overrides(representative)["task.planner_revision"]
    assert '{"enabled"' not in planner_argument
    Q._override_hydra_mapping(
        planner_argument, key="task.planner_revision", job_id=representative["id"]
    )
    assert result["generic_result_alone_may_unlock_successor"] is False
    assert result["task_revision_specialized_result_path"] == (
        "/tmp/probe/task_revision_probe_result.json"
    )
    with pytest.raises(Q.SuccessorQueueError, match="preregistered representative"):
        Q.cmd_full_scene_probe(
            queue,
            job_id=queue["jobs"][0]["id"],
            pod="pod1",
            gpu=1,
            attempt_id="attempt2",
            execute=False,
            confirm=None,
        )


def test_behavior_dry_run_never_ssh_and_exposes_pending_blockers(monkeypatch):
    queue = _as_pending(Q.load_queue(QUEUE))
    calls = []
    monkeypatch.setattr(
        Q.continuation.lean,
        "_run_ssh",
        lambda *_args, **_kwargs: calls.append("ssh"),
    )
    job = _first_launchable(queue)
    milestone = job["_continuation_parent_record"]["selected_embedded_iteration"] + 500
    result = Q.cmd_behavior(
        queue,
        job_id=job["id"],
        milestone=milestone,
        execute=False,
        confirm=None,
        write_receipt=True,
    )
    assert result["automatic_stop_authorized"] is False
    assert result["activation_blockers"]
    assert calls == []


def test_exact_stop_is_explicit_dry_run_only_and_never_ssh(monkeypatch):
    queue = _as_pending(Q.load_queue(QUEUE))
    calls = []
    monkeypatch.setattr(
        Q.continuation.lean,
        "_run_ssh",
        lambda *_args, **_kwargs: calls.append("ssh"),
    )
    job = _first_launchable(queue)
    milestone = job["_continuation_parent_record"]["selected_embedded_iteration"] + 500
    result = Q.cmd_exact_stop_behavior(
        queue,
        job_id=job["id"],
        milestone=milestone,
        execute=False,
        confirm=None,
    )
    assert result["automatic_invocation"] is False
    assert result["automatic_retry_authorized"] is False
    assert result["activation_blockers"]
    assert calls == []
    with pytest.raises(Q.SuccessorQueueError, match="--execute requires"):
        Q.cmd_exact_stop_behavior(
            queue,
            job_id=job["id"],
            milestone=milestone,
            execute=True,
            confirm="wrong",
        )
    assert calls == []


def test_remote_consumer_embeds_sha_bound_queue_without_remote_control_file_write():
    queue = Q.load_queue(QUEUE)
    job = queue["jobs"][0]
    command = Q._remote_task_revision_command(
        queue,
        job,
        function="inspect_or_attest_behavior_local",
        kwargs={"job_id": job["id"], "milestone": 2100, "write_receipt": False},
    )
    assert command[1] == "-c"
    assert "--queue" not in command
    assert max(map(len, command)) < 120_000
    request = json.loads(zlib.decompress(base64.b64decode(command[-1], validate=True)))
    assert request["function"] == "inspect_or_attest_behavior_local"
    assert request["queue"]["jobs"][0]["id"] == job["id"]
    assert "run_phase1_task_revision_supercombo_queue.py" in command[2]
