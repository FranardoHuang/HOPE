from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_phase1_fresh_sz_model4000_q50_queue.py"
QUEUE_PATH = ROOT / "configs" / "phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json"
PREREG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model4000_seed_stability_q50_prereg_20260712.json"
)
MODEL2000_QUEUE_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_execution_20260711.json"
)
MODEL2000_PREREG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_prereg_20260711.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module("model4000_q50_queue_under_test", VALIDATOR_PATH)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_committed_queue_is_offline_exact_matched_and_content_bound():
    queue = _json(QUEUE_PATH)
    prereg = _json(PREREG_PATH)
    R.validate_queue(queue)
    R.validate_preregistration(prereg, queue)
    assert R.sha256_file(VALIDATOR_PATH) == queue["validator"]["sha256"]
    assert R.sha256_file(PREREG_PATH) == queue["preregistration"]["sha256"]
    assert queue["runtime_entrypoint"] is None
    assert queue["auto_start"] is False
    assert queue["barrier"]["runtime_before_pass_allowed"] is False
    assert queue["barrier"]["required_seed_coverage"] == list(R.SEED_ORDER)
    assert all(prereg["arms"][seed]["checkpoint_sha256"] is None for seed in R.SEED_ORDER)


def test_model4000_reuses_identical_paper_and_unchanged_thresholds():
    queue = _json(QUEUE_PATH)
    prereg = _json(PREREG_PATH)
    old_queue = _json(MODEL2000_QUEUE_PATH)
    old_prereg = _json(MODEL2000_PREREG_PATH)
    assert queue["paper"] == old_queue["schedule"] == prereg["paper"]["schedule"]
    assert prereg["paper"]["exam_bank"] == old_prereg["paper"]["exam_bank"]
    expected = copy.deepcopy(old_prereg["gate_rule"])
    expected["pass_action"] = "record_model_4000_seed_stability_only_continue_all_arms_unmodified"
    assert prereg["gate_rule"] == expected
    assert prereg["known_before_prereg"]["model4000_seed1"]["aggregate_return_rate"] == 0.5
    assert prereg["interpretation_rule"]["family_stable_claim_allowed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("auto_start", True),
        lambda value: value.__setitem__("runtime_entrypoint", "run-now"),
        lambda value: value["paper"].__setitem__("file_sha256", "0" * 64),
        lambda value: value["barrier"].__setitem__("runtime_before_pass_allowed", True),
        lambda value: value["barrier"].__setitem__("all_finite", False),
        lambda value: value["barrier"]["required_seed_coverage"].pop(),
        lambda value: value["formal_semantics"].__setitem__("whole_arm_stop_allowed", True),
    ),
)
def test_queue_rejects_repaper_activation_or_barrier_relaxation(mutation):
    queue = _json(QUEUE_PATH)
    mutation(queue)
    with pytest.raises(R.ContractError):
        R.validate_queue(queue)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("jobs_started", 1),
        lambda value: value["family"].__setitem__("checkpoint_iteration", 6000),
        lambda value: value["paper"]["schedule"].__setitem__("materialize_new_schedule_allowed", True),
        lambda value: value["arms"]["seed4"].__setitem__("checkpoint_sha256", "4" * 64),
        lambda value: value["arms"]["seed4"].__setitem__("lineage_exact", False),
        lambda value: value["gate_rule"].__setitem__("aggregate_rate_min_seed_min", 0.2),
        lambda value: value["interpretation_rule"]["seed4_delayed_learning_supported_only_if"].__setitem__(
            "aggregate_rate_min", 0.2
        ),
        lambda value: value["known_before_prereg"]["model4000_seed1"].__setitem__(
            "aggregate_return_rate", 1.0
        ),
    ),
)
def test_prereg_rejects_started_bound_changed_or_hindsight_mutation(mutation):
    queue = _json(QUEUE_PATH)
    prereg = _json(PREREG_PATH)
    mutation(prereg)
    with pytest.raises(R.ContractError):
        R.validate_preregistration(prereg, queue)


def _checkpoint_audit() -> dict:
    return {
        "iter": 4000,
        "training_contract_sha256": R.EXPECTED_HARD_CONTRACT_SHA256,
        "training_contract_schema_version": 3,
        "training_contract_lineage_exact": True,
        "tensor_count": 12,
        "floating_tensor_count": 10,
        "floating_elements": 1000,
        "nonfinite": 0,
    }


def _hard_contract() -> dict:
    return {
        "schema_version": 3,
        "motion_kinematics_exact": True,
        "face_command_pairing": "shared_plus_y",
        "joint_friction_coefficients": [0.0] * 31,
        "question_bank": {
            "sha256": "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700",
            "schema_version": 3,
            "split": "train",
            "source_family_sha256": R.EXPECTED_FAMILY["source_family_sha256"],
            "exact": True,
        },
    }


def test_audit_pod_requires_both_local_finite_model4000_checkpoints_and_starts_nothing(
    tmp_path: Path, monkeypatch
):
    queue = _json(QUEUE_PATH)
    prereg = _json(PREREG_PATH)
    train = tmp_path / "train"
    for index, seed in enumerate(R.POD_ARM_ORDER["pod1"], start=1):
        run = train / seed
        checkpoint = run / "model_4000.pt"
        hard = run / "params" / "training_contract.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        _write(hard, _hard_contract())
        prereg["arms"][seed]["checkpoint_path"] = str(checkpoint)
        prereg["arms"][seed]["training_contract_path"] = str(hard)
        if seed == "seed1":
            queue["seed1_reuse"]["expected_checkpoint_sha256"] = R.sha256_file(checkpoint)
    output = tmp_path / "pod1_ready.json"
    queue["pod_audit_outputs"]["pod1"] = str(output)
    schedule_path = tmp_path / "shared.schedule.json"
    schedule_path.write_bytes(b"same-paper")
    schedule = {
        "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
        "items": [{"question_id": f"{index:064x}"} for index in range(100)],
    }
    monkeypatch.setattr(
        R,
        "_validate_checkout_and_static_inputs",
        lambda *_args, **_kwargs: (train, schedule),
    )
    real_sha256_file = R.sha256_file
    monkeypatch.setattr(
        R,
        "sha256_file",
        lambda path: (
            R.EXPECTED_HARD_CONTRACT_SHA256
            if Path(path).name == "training_contract.json"
            else real_sha256_file(Path(path))
        ),
    )
    monkeypatch.setattr(R.fresh, "checkpoint_audit", lambda *_args, **_kwargs: _checkpoint_audit())
    monkeypatch.setattr(R.fresh, "validate_hard_contract", lambda *_args, **_kwargs: None)
    R.audit_pod(
        QUEUE_PATH,
        queue,
        PREREG_PATH,
        prereg,
        pod="pod1",
        schedule_path=schedule_path,
        output=output,
    )
    document = _json(output)
    assert document["content"]["actions"]["judges_started"] == 0
    assert document["content"]["actions"]["trainer_or_worker_signals"] == []
    assert set(document["content"]["arms"]) == {"seed1", "seed3"}
    checkpoint = Path(prereg["arms"]["seed3"]["checkpoint_path"])
    checkpoint.unlink()
    queue["pod_audit_outputs"]["pod1"] = str(tmp_path / "second.json")
    with pytest.raises(R.ContractError, match="not ready"):
        R.audit_pod(
            QUEUE_PATH,
            queue,
            PREREG_PATH,
            prereg,
            pod="pod1",
            schedule_path=schedule_path,
            output=tmp_path / "second.json",
        )


def _pod_document(pod: str) -> dict:
    prereg = _json(PREREG_PATH)
    arms = {}
    for seed in R.POD_ARM_ORDER[pod]:
        digit = str(int(seed[4:]))
        arm = prereg["arms"][seed]
        arms[seed] = {
            "checkpoint_path": arm["checkpoint_path"],
            "checkpoint_sha256": digit * 64,
            "training_contract_path": arm["training_contract_path"],
            "training_contract_sha256": R.EXPECTED_HARD_CONTRACT_SHA256,
            "checkpoint_audit": _checkpoint_audit(),
        }
    content = {
        "queue_id": _json(QUEUE_PATH)["queue_id"],
        "status": "pod_checkpoints_ready_judge_not_started",
        "completed_utc": "2026-07-12T01:00:00Z",
        "pod": pod,
        "queue": {"path": str(QUEUE_PATH), "sha256": R.sha256_file(QUEUE_PATH)},
        "preregistration": {"path": str(PREREG_PATH), "sha256": R.sha256_file(PREREG_PATH)},
        "validator_sha256": R.sha256_file(VALIDATOR_PATH),
        **R.EXPECTED_SEMANTICS,
        "shared_schedule": {
            "path": "/paper/shared_clean_k100.schedule.json",
            "file_sha256": R.EXPECTED_SCHEDULE["file_sha256"],
            "semantic_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
            "question_id_order_sha256": R.EXPECTED_SCHEDULE["question_id_order_sha256"],
        },
        "arm_order": list(R.POD_ARM_ORDER[pod]),
        "arms": arms,
        "actions": {
            "judges_started": 0,
            "trainer_or_worker_signals": [],
            "runtime_authorized_by_this_pod_audit": False,
            "real_robot_authorized": False,
        },
    }
    return R._content_document("phase1_fresh_sz_model4000_q50_pod_ready_audit", content)


def test_activation_is_impossible_without_two_complete_content_bound_pod_audits(tmp_path: Path):
    pod1_path = tmp_path / "pod1.json"
    pod2_path = tmp_path / "pod2.json"
    _write(pod1_path, _pod_document("pod1"))
    _write(pod2_path, _pod_document("pod2"))
    queue = _json(QUEUE_PATH)
    queue["activation_output_dir"] = str(tmp_path / "activation")
    output = R.activate(
        QUEUE_PATH,
        queue,
        PREREG_PATH,
        pod1_audit=pod1_path,
        pod1_sha=R.sha256_file(pod1_path),
        pod2_audit=pod2_path,
        pod2_sha=R.sha256_file(pod2_path),
        output_dir=tmp_path / "activation",
    )
    activation = _json(output)["content"]
    assert activation["status"] == "all_four_checkpoints_ready_judge_not_started"
    assert list(activation["arms"]) == list(R.SEED_ORDER)
    assert activation["actions"]["judges_started"] == 0
    pod2 = _json(pod2_path)
    del pod2["content"]["arms"]["seed4"]
    pod2["content"]["content_sha256"] = "not-used"
    pod2["content_sha256"] = R.canonical_sha256(pod2["content"])
    broken = tmp_path / "pod2-broken.json"
    _write(broken, pod2)
    with pytest.raises(R.ContractError):
        R.activate(
            QUEUE_PATH,
            queue,
            PREREG_PATH,
            pod1_audit=pod1_path,
            pod1_sha=R.sha256_file(pod1_path),
            pod2_audit=broken,
            pod2_sha=R.sha256_file(broken),
            output_dir=tmp_path / "activation",
        )


def test_validator_has_no_runtime_or_process_control_surface():
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "Popen(" not in source
    assert "killpg(" not in source
    assert "os.kill(" not in source
    assert "ssh " not in source
    parser_commands = {"validate-config", "audit-pod", "activate"}
    assert parser_commands == {value for value in parser_commands if value in source}
