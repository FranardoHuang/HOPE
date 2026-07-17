from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_task_revision_ready_successor_queue.py"
QUEUE = ROOT / "configs" / "phase1_task_revision_ready_successor_20260717.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("ready_successor_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _raw() -> dict:
    value = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _ready_counters(*, phase: int = 8, unavailable: int = 0) -> dict:
    return {
        "ready_tilt_eligible_sample_count": phase,
        "ready_tilt_rad_sum": 0.8,
        "ready_base_speed_eligible_sample_count": phase,
        "ready_base_speed_xy_mps_sum": 1.6,
        "ready_station_offset_eligible_sample_count": phase,
        "ready_station_offset_m_sum": 0.4,
        "ready_foot_contact_eligible_sample_count": phase - unavailable,
        "ready_foot_contact_fraction_sum": float(phase - unavailable),
        "ready_foot_slip_eligible_sample_count": phase - unavailable,
        "ready_foot_slip_speed_mps_sum": 0.1,
        "ready_phase_sample_count": phase,
        "ready_planner_task_entry_sample_count": phase,
        "ready_planner_legacy_hold_violation_count": 0,
        "ready_foot_sensor_unavailable_sample_count": unavailable,
        "ready_nonfinite_value_count": 0,
    }


def _probe_evidence(queue: dict, *, attempt_id: str = "unit_bound") -> dict:
    job = Q._job(queue, Q.PROBE_JOB)
    slot = Q.lean._slot_by_identity(queue, "pod2", Q.PROBE_GPU)
    claim, _argv, run_dir = Q.lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    checks = Q._validate_ready_counter_shape(_ready_counters(), "unit probe")
    content = {
        "schema_version": 1,
        "status": "passed",
        "unlock_authorized": True,
        "representative_job_id": Q.PROBE_JOB,
        "pod": "pod2",
        "gpu": Q.PROBE_GPU,
        "attempt_id": attempt_id,
        "claim_content_sha256": claim["content_sha256"],
        "generic_result_file_sha256": "a" * 64,
        "generic_result_content_sha256": "b" * 64,
        "log_prefix": {},
        "exact_update_ids": [1, 2],
        "task_revision_probe_passed": True,
        **checks,
        "aggregate_counters": _ready_counters(),
        "formal_evidence_eligible": False,
        "consumer_source": {"mode": "embedded_sha_bound", "sha256": Q._runner_sha()},
        "automatic_retry": False,
    }
    receipt = {
        "schema_version": 1,
        "content": content,
        "content_sha256": Q._canonical_sha(content),
    }
    raw = json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    evidence = {
        "schema_version": 1,
        "kind": "ready_successor_specialized_probe_receipt_v1",
        "status": "passed",
        "unlock_authorized": True,
        "producer_runner_source_sha256": Q._runner_sha(),
        "receipt_path": f"{run_dir}/ready_successor_probe_result.json",
        "receipt_file_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "receipt_file_base64": __import__("base64").b64encode(raw).decode(),
        "receipt_content_sha256": receipt["content_sha256"],
        "receipt_content": content,
    }
    evidence["evidence_content_sha256"] = Q._evidence_content_sha(evidence)
    return evidence


def test_checked_in_queue_is_plan_only_but_schema_complete():
    queue = Q.load_queue(QUEUE)
    result = Q.cmd_validate(queue)
    assert result["schema_valid"] is True
    assert result["pending"] is True
    assert result["activation_ready"] is False
    assert result["job_count"] == 4
    assert "parent_binding_evidence is not a pass mapping" not in result[
        "activation_blockers"
    ]
    assert "ready_full_scene_probe_evidence is not a pass mapping" in result[
        "activation_blockers"
    ]


def test_plan_is_one_per_gpu_then_dynamic_fourth_and_all_milestones_absolute():
    result = Q.cmd_plan(Q.load_queue(QUEUE))
    assert [row.get("planned_slot") for row in result["jobs"][:3]] == [
        "pod2/gpu0",
        "pod2/gpu1",
        "pod2/gpu2",
    ]
    fourth = result["jobs"][3]
    assert fourth["slot_policy"] == "least_occupied_pod2_at_launch"
    assert fourth["k100_on_gpu0_causes_fallback"] is True
    assert fourth["allowed_slots"] == ["pod2/gpu0", "pod2/gpu1", "pod2/gpu2"]
    assert all(row["absolute_milestones"] == [5900, 6200, 6700] for row in result["jobs"])
    assert all(row["additional_updates"] == 1001 for row in result["jobs"])


def test_matrix_is_ready_strength_by_qdot_limit_hinge_not_random_force():
    queue = Q.load_queue(QUEUE)
    cells = {(job["ready_role"], job["qdot_limit_hinge_weight"]) for job in queue["jobs"]}
    assert cells == {("baseline", -5.0), ("baseline", 0.0), ("strong", -5.0), ("strong", 0.0)}
    flattened = json.dumps(queue["jobs"], sort_keys=True)
    assert "lateral_perturb" not in flattened
    assert "external_force" not in flattened


def test_all_cells_share_exact_model5700_parent_and_fresh_run_namespace():
    queue = Q.load_queue(QUEUE)
    paths = {job["warm_start"]["checkpoint_path"] for job in queue["jobs"]}
    assert len(paths) == 1
    assert next(iter(paths)).endswith("/model_5700.pt")
    assert len({job["run_dir"] for job in queue["jobs"]}) == 4
    assert all(job["run_dir"].startswith(Q.NAMESPACE_ROOT + "/runs/") for job in queue["jobs"])
    assert all("phase1_task_revision_supercombo_20260716/runs/" not in job["run_dir"] for job in queue["jobs"])


def test_ready_ledger_requires_new_task_entry_and_sensor_honesty_counters():
    required = Q._required_counters(Q.load_queue(QUEUE))
    assert Q.READY_COUNTERS <= required
    assert {
        "ready_phase_sample_count",
        "ready_planner_task_entry_sample_count",
        "ready_planner_legacy_hold_violation_count",
        "ready_foot_sensor_unavailable_sample_count",
        "ready_nonfinite_value_count",
    } <= required


@pytest.mark.parametrize("unavailable", [0, 3, 8])
def test_ready_invariants_accept_real_foot_samples_or_explicit_unavailable(unavailable):
    checks = Q._validate_ready_counter_shape(
        _ready_counters(unavailable=unavailable), "window"
    )
    assert set(checks) == set(Q.REQUIRED_READY_INVARIANTS)
    assert all(checks.values())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("ready_phase_sample_count", 0, "ready invariants"),
        ("ready_planner_task_entry_sample_count", 7, "ready invariants"),
        ("ready_planner_legacy_hold_violation_count", 1, "ready invariants"),
        ("ready_nonfinite_value_count", 1, "ready invariants"),
        ("ready_tilt_rad_sum", float("nan"), "finite and non-negative"),
    ],
)
def test_ready_invariants_fail_closed(key, value, message):
    counters = _ready_counters()
    counters[key] = value
    with pytest.raises(Q.ReadySuccessorError, match=message):
        Q._validate_ready_counter_shape(counters, "window")


def test_foot_denominator_cannot_be_silently_dropped():
    counters = _ready_counters(phase=8, unavailable=2)
    counters["ready_foot_contact_eligible_sample_count"] = 5
    with pytest.raises(Q.ReadySuccessorError, match="ready invariants"):
        Q._validate_ready_counter_shape(counters, "window")


def test_full_scene_probe_dry_run_is_4096_by_two_in_fresh_namespace():
    queue = Q.load_queue(QUEUE)
    result = Q.cmd_full_scene_probe(
        queue, attempt_id="unit_a1", execute=False, confirm=None
    )
    assert result["dry_run"] is True
    assert result["resource"] == "pod2/gpu1"
    assert result["budget"] == {"num_envs": 4096, "max_iterations": 2, "save_interval": 1, "milestones": [1]}
    assert result["run_dir"].startswith(Q.NAMESPACE_ROOT + "/runs/_full_scene_probes/")
    assert result["ready_specialized_result_path"].endswith(
        "/ready_successor_probe_result.json"
    )
    rendered = " ".join(result["ssh_argv"])
    assert "model_5700.pt" not in rendered
    assert "phase1_task_revision_ready_successor_20260717" in rendered


def test_full_scene_execute_requires_exact_confirmation(monkeypatch):
    queue = Q.load_queue(QUEUE)
    with pytest.raises(Q.ReadySuccessorError, match=Q.PROBE_CONFIRM):
        Q.cmd_full_scene_probe(queue, attempt_id="unit_a2", execute=True, confirm="wrong")


def test_probe_finalizer_is_one_dry_run_ssh_and_new_no_clobber_receipt():
    result = Q.cmd_finalize_full_scene_probe(
        Q.load_queue(QUEUE), attempt_id="unit_a3", execute=False, confirm=None
    )
    assert result["dry_run"] is True
    assert result["one_ssh_transaction"] is True
    assert result["specialized_result_path"].endswith("ready_successor_probe_result.json")
    assert len(result["ssh_argv"]) > 2


def test_queue_rejects_old_science_namespace_reuse(tmp_path):
    value = _raw()
    value["jobs"][0]["run_dir"] = (
        "/workspace/codexschema/phase1_task_revision_supercombo_20260716/runs/p2_equal_reward"
    )
    with pytest.raises(Q.ReadySuccessorError, match="fresh namespace"):
        Q.load_queue(_write(tmp_path, value))


def test_queue_rejects_duplicate_yaml_keys_before_last_wins(tmp_path):
    raw = QUEUE.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text(raw.replace("schema_version: 1\n", "schema_version: 1\nschema_version: 1\n", 1), encoding="utf-8")
    with pytest.raises(Q.ReadySuccessorError, match="duplicate YAML key"):
        Q.load_queue(path)


def test_queue_rejects_valid_but_stale_runner_source_gate_before_remote_entry(tmp_path):
    value = _raw()
    value["blocking_contract"]["runner_source_gate"]["sha256"] = "f" * 64
    assert value["blocking_contract"]["runner_source_gate"]["sha256"] != Q._runner_sha()
    with pytest.raises(Q.ReadySuccessorError, match="executing runner bytes"):
        Q.load_queue(_write(tmp_path, value))


@pytest.mark.parametrize("entry", ["inspect", "probe", "finalize"])
def test_each_remote_entry_rechecks_runner_source_gate_after_load(entry):
    queue = Q.load_queue(QUEUE)
    queue["blocking_contract"]["runner_source_gate"]["sha256"] = "e" * 64
    with pytest.raises(Q.ReadySuccessorError, match="executing runner bytes"):
        if entry == "inspect":
            Q.cmd_inspect_parent(queue, execute=False, confirm=None)
        elif entry == "probe":
            Q.cmd_full_scene_probe(
                queue, attempt_id="runner_drift", execute=False, confirm=None
            )
        else:
            Q.cmd_finalize_full_scene_probe(
                queue, attempt_id="runner_drift", execute=False, confirm=None
            )


def test_sparse_zero_without_eligibility_can_never_prune():
    pruning = Q.load_queue(QUEUE)["pruning_contract"]
    assert pruning["sparse_zero_without_positive_eligible_denominator_may_stop"] is False
    assert pruning["behavior_decision_requires_two_disjoint_complete_windows"] is True
    assert pruning["window_updates"] == 100


def test_parent_content_digest_and_file_digests_remain_distinct_exact_fields():
    parent = Q._parent(Q.load_queue(QUEUE))
    assert parent["original_claim_content_sha256"] == (
        "e10d2c248d90daa3172ea80147a394dad64ce326eb4052889c25bfb9d3df420b"
    )
    assert parent["original_queue_claim_sha256"] == (
        "7e11016516f0ced7407a584ecbd2b73fd13f577df7f51a71da52dd147e7e9df6"
    )
    assert parent["original_run_binding_sha256"] == (
        "745b9f1d7e7cc67f4cac4b543cd8bd718bdb027c4061113dde5be33ebd85e9d0"
    )
    assert parent["selected_checkpoint_sha256"] == (
        "521d41e9ed529eaf2e740358122944bd726d5eb78403794dbc3730f18bbf984c"
    )


def test_parent_evidence_cannot_substitute_arbitrary_consistent_sha(tmp_path):
    value = _raw()
    evidence = value["blocking_contract"]["parent_integrity_evidence"]
    evidence["queue_claim"]["file_sha256"] = "5" * 64
    evidence["evidence_content_sha256"] = Q._evidence_content_sha(evidence)
    with pytest.raises(Q.ReadySuccessorError, match="exact selected parent"):
        Q.load_queue(_write(tmp_path, value))


def test_probe_evidence_binds_exact_file_bytes_source_attempt_and_claim():
    queue = Q.load_queue(QUEUE)
    evidence = _probe_evidence(queue)
    Q._validate_probe_binding_evidence(queue, evidence)
    forged = copy.deepcopy(evidence)
    forged["receipt_file_sha256"] = "f" * 64
    forged["evidence_content_sha256"] = Q._evidence_content_sha(forged)
    with pytest.raises(Q.ReadySuccessorError, match="file SHA does not bind file bytes"):
        Q._validate_probe_binding_evidence(queue, forged)
    forged = copy.deepcopy(evidence)
    forged["receipt_content"]["claim_content_sha256"] = "e" * 64
    forged["receipt_content_sha256"] = Q._canonical_sha(forged["receipt_content"])
    forged["evidence_content_sha256"] = Q._evidence_content_sha(forged)
    with pytest.raises(Q.ReadySuccessorError, match="file content differs"):
        Q._validate_probe_binding_evidence(queue, forged)


def test_probe_only_runner_rejects_forged_activated_queue(tmp_path):
    value = _raw()
    value["launch_authorized"] = True
    value["preregistration_status"] = Q.ACTIVATED_STATUS
    value["namespace_contract"]["status"] = "activated_no_clobber"
    for job in value["jobs"]:
        job["status"] = "ready"
        job["blocker"] = None
    value["blocking_contract"]["ready_full_scene_probe_evidence"] = {
        "status": "passed",
        "receipt_file_sha256": "a" * 64,
        "receipt_content_sha256": "b" * 64,
    }
    with pytest.raises(Q.ReadySuccessorError, match="probe/inspection-only"):
        Q.load_queue(_write(tmp_path, value))


def test_parent_inspector_dry_run_is_one_pod2_read_only_command():
    result = Q.cmd_inspect_parent(Q.load_queue(QUEUE), execute=False, confirm=None)
    assert result["dry_run"] is True
    assert result["pod"] == "pod2"
    assert result["read_only"] is result["no_write"] is result["no_signal"] is True
    assert result["milestone"] == 5700
    assert result["expected_claim_content_sha256"] == (
        "e10d2c248d90daa3172ea80147a394dad64ce326eb4052889c25bfb9d3df420b"
    )
    assert result["paths"]["checkpoint"].endswith("/model_5700.pt")
    assert len(result["ssh_argv"]) > 2


def test_parent_inspector_program_has_no_write_or_signal_surface():
    program = Q.PARENT_INSPECT_PROGRAM
    assert "os.O_RDONLY" in program
    assert "O_WRONLY" not in program
    assert "O_CREAT" not in program
    assert "os.kill" not in program
    assert "subprocess" not in program
    assert "nvidia-smi" not in program
    assert "torch.load" in program
    assert "nonfinite_element_count" in program


def test_parent_inspector_execute_requires_exact_confirmation():
    with pytest.raises(Q.ReadySuccessorError, match=Q.PARENT_INSPECT_CONFIRM):
        Q.cmd_inspect_parent(Q.load_queue(QUEUE), execute=True, confirm="wrong")


def test_parent_inspector_accepts_only_strict_read_only_pass(monkeypatch):
    queue = Q.load_queue(QUEUE)
    integrity = queue["blocking_contract"]["parent_integrity_evidence"]
    parent = Q._parent(queue)
    candidate = {
        "schema_version": 1,
        "status": "passed",
        "read_only": True,
        "no_write": True,
        "no_signal": True,
        "job_id": "taskrev_p2_equal_reward",
        "milestone": 5700,
        "expected_claim_content_sha256": (
            "e10d2c248d90daa3172ea80147a394dad64ce326eb4052889c25bfb9d3df420b"
        ),
        "queue_claim": copy.deepcopy(integrity["queue_claim"]),
        "run_binding": copy.deepcopy(integrity["run_binding"]),
        "milestone_receipt": {
            **copy.deepcopy(integrity["milestone_receipt"]),
            "binding_content_sha256": integrity["run_binding"]["content_sha256"],
        },
        "checkpoint": {
            **copy.deepcopy(integrity["checkpoint"]),
            "training_launch_claim_sha256": parent["original_claim_content_sha256"],
            "lineage_exact": 0,
        },
        "hard_contract": {
            **copy.deepcopy(integrity["hard_contract"]),
            "lineage_exact": 0,
        },
        "parent_selection_patch": {
            "original_queue_claim_sha256": parent["original_queue_claim_sha256"],
            "original_run_binding_sha256": parent["original_run_binding_sha256"],
            "selected_checkpoint_sha256": parent["selected_checkpoint_sha256"],
            "selected_hard_contract_path": parent["selected_hard_contract_path"],
            "selected_hard_contract_sha256": parent["selected_hard_contract_sha256"],
            "selection_is_final": True,
        },
    }
    monkeypatch.setattr(
        Q.lean,
        "_run_ssh",
        lambda *args, **kwargs: json.dumps(candidate),
    )
    result = Q.cmd_inspect_parent(
        queue,
        execute=True,
        confirm=Q.PARENT_INSPECT_CONFIRM,
    )
    assert result["inspection"] == candidate
    assert result["semantic_binding_evidence"]["status"] == "passed"
    candidate["no_signal"] = False
    with pytest.raises(Q.ReadySuccessorError, match="strict read-only pass"):
        Q.cmd_inspect_parent(
            queue,
            execute=True,
            confirm=Q.PARENT_INSPECT_CONFIRM,
        )
    candidate["no_signal"] = True
    candidate["milestone_receipt"]["binding_content_sha256"] = "6" * 64
    with pytest.raises(Q.ReadySuccessorError, match="cross-bind"):
        Q.cmd_inspect_parent(
            queue,
            execute=True,
            confirm=Q.PARENT_INSPECT_CONFIRM,
        )
    candidate["milestone_receipt"]["binding_content_sha256"] = integrity[
        "run_binding"
    ]["content_sha256"]
    candidate["checkpoint"]["training_launch_claim_sha256"] = "7" * 64
    with pytest.raises(Q.ReadySuccessorError, match="cross-bind"):
        Q.cmd_inspect_parent(
            queue,
            execute=True,
            confirm=Q.PARENT_INSPECT_CONFIRM,
        )
