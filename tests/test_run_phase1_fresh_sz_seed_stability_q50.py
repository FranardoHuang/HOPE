from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_fresh_sz_seed_stability_q50.py"
CONFIG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_execution_20260711.json"
)
PREREG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_prereg_20260711.json"
)
POD1_RESULT_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_pod1_result_20260711.json"
)
POD2_RESULT_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_pod2_result_20260711.json"
)
AGGREGATE_RESULT_PATH = ROOT / "configs" / (
    "phase1_fresh_SZ_model2000_seed_stability_q50_"
    "a756bf1d0e76d1016992ae241b935cf92b3c84ffd55fe503e7c199626d9c8ffd.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module("fresh_sz_seed_stability_q50_under_test", RUNNER_PATH)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_committed_paper_is_content_bound_exact_and_non_authorizing():
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    assert R.sha256_file(RUNNER_PATH) == config["tools"]["runner_sha256"]
    assert R.sha256_file(PREREG_PATH) == config["preregistration_sha256"]
    R.validate_preregistration(prereg, config)
    assert config["pod_arm_order"] == {
        "pod1": ["seed1", "seed3"],
        "pod2": ["seed2", "seed4"],
    }
    assert prereg["q10_trigger"]["observed_aggregate_return_rate"] == {
        "seed1": 0.9,
        "seed2": 1.0,
        "seed3": 1.0,
        "seed4": 0.25,
    }
    assert config["schedule"] == R.EXPECTED_SCHEDULE
    assert config["schedule"]["materialize_new_schedule_allowed"] is False
    assert config["schedule"]["allow_inexact_contract"] is False
    assert config["semantics"]["whole_arm_stop_allowed"] is False
    assert config["semantics"]["whole_arm_promote_allowed"] is False
    assert config["semantics"]["deploy_gate"] is False
    assert config["semantics"]["real_robot_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("auto_start", True),
        lambda value: value["semantics"].__setitem__("evaluation_contract_exact", False),
        lambda value: value["semantics"].__setitem__("whole_arm_stop_allowed", True),
        lambda value: value["schedule"].__setitem__("materialize_new_schedule_allowed", True),
        lambda value: value["schedule"].__setitem__("allow_inexact_contract", True),
        lambda value: value["pod_arm_order"]["pod1"].reverse(),
        lambda value: value["gate_rule"].__setitem__("aggregate_rate_min_seed_min", 0.0),
        lambda value: value["seed1_reuse"].__setitem__("fallback", "accept_anyway"),
    ),
)
def test_execution_config_rejects_relaxation_or_repapering(tmp_path: Path, mutation):
    value = _config()
    mutation(value)
    path = tmp_path / "mutated.execution.json"
    _write_json(path, value)
    with pytest.raises(R.ContractError):
        R.load_execution_config(path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("jobs_started", 1),
        lambda value: value["q10_trigger"].__setitem__("stop_or_promote_allowed", True),
        lambda value: value["paper"].__setitem__("allow_inexact_contract_required", True),
        lambda value: value["paper"]["reuse_independence"].__setitem__(
            "selected_without_seed2_seed3_seed4_policy_or_outcome", False
        ),
        lambda value: value["arms"]["seed3"].__setitem__("lineage_exact", False),
        lambda value: value["arms"]["seed4"].__setitem__("zero_joint_friction", False),
        lambda value: value["arms"]["seed4"].__setitem__(
            "training_contract_sha256", "f" * 64
        ),
        lambda value: value["formal_semantics"].__setitem__("whole_arm_promote_allowed", True),
    ),
)
def test_preregistration_rejects_started_inexact_or_changed_paper(mutation):
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    mutation(prereg)
    with pytest.raises(R.ContractError):
        R.validate_preregistration(prereg, config)


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
            "source_family_sha256": (
                "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5"
            ),
            "exact": True,
        },
    }


def test_hard_contract_requires_fresh_sz_shared_face_and_zero_plant():
    prereg = _prereg()
    R.validate_hard_contract(_hard_contract(), prereg)
    for mutation in (
        lambda value: value.__setitem__("motion_kinematics_exact", False),
        lambda value: value.__setitem__("face_command_pairing", "legacy_signed_vs_A"),
        lambda value: value["joint_friction_coefficients"].__setitem__(0, 0.01),
        lambda value: value["question_bank"].__setitem__("source_family_sha256", "e" * 64),
    ):
        candidate = _hard_contract()
        mutation(candidate)
        with pytest.raises(R.ContractError):
            R.validate_hard_contract(candidate, prereg)


def test_formal_judge_command_has_no_inexact_escape():
    command = R.fresh.build_judge_command(
        judge=Path("/eval/judge.sh"),
        arm=_prereg()["arms"]["seed3"],
        schedule_path=Path("/paper/shared.schedule.json"),
        gpu=0,
    )
    assert any(item.startswith("--exam-schedule-json ") for item in command)
    assert "--schedule-k" not in command
    assert all("--allow-inexact-contract" not in item for item in command)


def test_q10_trigger_sources_are_rehashed_and_are_never_a_decision(tmp_path: Path):
    prereg = _prereg()
    rows = []
    for seed in R.POD_ARM_ORDER["pod1"]:
        source = next(row for row in prereg["q10_trigger"]["evidence"] if row["seed"] == seed)
        row = copy.deepcopy(source)
        for name in ("state", "log"):
            path = tmp_path / f"{seed}.{name}"
            path.write_bytes(f"{seed}-{name}\n".encode())
            row[name] = {"path": str(path), "sha256": R.sha256_file(path)}
        rows.append(row)
    prereg["q10_trigger"]["evidence"] = rows + [
        row for row in prereg["q10_trigger"]["evidence"] if row["pod"] == "pod2"
    ]
    R.validate_q10_trigger_sources(prereg, pod="pod1")
    Path(rows[0]["state"]["path"]).write_bytes(b"mutated\n")
    with pytest.raises(R.ContractError, match="changed/missing"):
        R.validate_q10_trigger_sources(prereg, pod="pod1")
    assert prereg["q10_trigger"]["screen_only"] is True
    assert prereg["q10_trigger"]["stop_or_promote_allowed"] is False


def test_prepare_copies_identical_schedule_without_start_and_is_no_clobber(
    tmp_path: Path, monkeypatch
):
    config = _config()
    prereg = _prereg()
    state_dir = tmp_path / "pod1-state"
    config["runtime"]["pod_state_dirs"]["pod1"] = str(state_dir)
    config_path = tmp_path / "execution.json"
    prereg_path = tmp_path / "prereg.json"
    schedule_source = tmp_path / "frozen.schedule.json"
    schedule_source.write_bytes(b'{"frozen":true}\n')
    _write_json(config_path, config)
    _write_json(prereg_path, prereg)
    schedule = {
        "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
        "items": [{"question_id": f"{index:064x}"} for index in range(100)],
    }
    monkeypatch.setattr(
        R,
        "validate_runtime_inputs",
        lambda *_args, **_kwargs: (
            ROOT,
            ROOT,
            {},
            {"seed1": {"iteration": 2000}, "seed3": {"iteration": 2000}},
            schedule,
        ),
    )
    monkeypatch.setattr(R, "validate_schedule", lambda *_args, **_kwargs: schedule)
    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("prepare must not start a judge"),
    )
    assert R.prepare(
        config_path,
        config,
        prereg_path,
        prereg,
        pod="pod1",
        schedule_source=schedule_source,
    ) == 0
    copied = state_dir / config["runtime"]["schedule_filename"]
    assert copied.read_bytes() == schedule_source.read_bytes()
    runtime = json.loads(
        (state_dir / config["runtime"]["runtime_contract_filename"]).read_text()
    )
    assert runtime["status"] == "prepared_not_started"
    assert runtime["jobs_started"] == 0
    assert runtime["auto_start"] is False
    with pytest.raises(R.ContractError, match="no-clobber"):
        R.prepare(
            config_path,
            config,
            prereg_path,
            prereg,
            pod="pod1",
            schedule_source=schedule_source,
        )


def test_reuse_check_is_read_only_and_runs_full_chain(monkeypatch):
    config = _config()
    prereg = _prereg()
    observed = []
    monkeypatch.setattr(
        R,
        "validate_runtime_contract",
        lambda *_args, **_kwargs: {
            "shared_schedule": {"path": "/paper/shared.schedule.json"}
        },
    )
    monkeypatch.setattr(
        R,
        "validate_runtime_inputs",
        lambda *_args, **_kwargs: observed.append("runtime") or None,
    )
    monkeypatch.setattr(
        R,
        "validate_seed1_reuse",
        lambda *_args, **_kwargs: observed.append("reuse") or {
            "report": {"sha256": "a" * 64},
            "summary": {"sha256": "b" * 64},
            "attempt_ledger": {"sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("reuse-check must not start a judge"),
    )
    assert R.check_seed1_reuse(config, prereg, Path("/runtime.json"), "d" * 64) == 0
    assert observed == ["runtime", "reuse"]


def _arm_result(prereg: dict, seed: str, aggregate: int, forehand: int) -> dict:
    return {
        "checkpoint_iteration": 2000,
        "checkpoint_sha256": prereg["arms"][seed]["checkpoint_sha256"],
        "training_contract_sha256": prereg["arms"][seed]["training_contract_sha256"],
        "evaluation_contract_exact": True,
        "formal_target": True,
        "fresh_lineage": True,
        "denominators": {"aggregate": 100, "forehand": 50, "backhand": 50},
        "returned_counts": {
            "aggregate": aggregate,
            "forehand": forehand,
            "backhand": aggregate - forehand,
            "physical_falls": 0,
        },
        "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
        "question_id_order": [f"{index:064x}" for index in range(100)],
        "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
        "execution_contract_sha256": "e" * 64,
        "ready_state_sha256": "f" * 64,
    }


def _pod_result(
    tmp_path: Path, config: dict, prereg: dict, pod: str, rates: dict[str, tuple[int, int]]
) -> Path:
    content = {
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "complete",
        "completed_utc": "2026-07-11T17:00:00Z",
        "runtime_contract": {"path": "/runtime.json", "sha256": "a" * 64},
        "config_sha256": R.sha256_file(CONFIG_PATH),
        "preregistration_sha256": R.sha256_file(PREREG_PATH),
        "runner_sha256": config["tools"]["runner_sha256"],
        **R.EXPECTED_SEMANTICS,
        "shared_schedule": {},
        "arm_order": list(R.POD_ARM_ORDER[pod]),
        "execution": {},
        "arms": {
            seed: _arm_result(prereg, seed, *rates[seed])
            for seed in R.POD_ARM_ORDER[pod]
        },
        "gate_rule": R.EXPECTED_GATE_RULE,
        "actions": {
            "training": "continue_all_arms_unmodified",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    path = tmp_path / f"{pod}.json"
    R.atomic_json(path, R._content_document("phase1_fresh_sz_seed_stability_q50_pod", content))
    return path


@pytest.mark.parametrize(
    ("rates", "expected_pass"),
    (
        (
            {
                "seed1": (80, 40),
                "seed2": (82, 41),
                "seed3": (78, 39),
                "seed4": (76, 38),
            },
            True,
        ),
        (
            {
                "seed1": (90, 45),
                "seed2": (100, 50),
                "seed3": (100, 50),
                "seed4": (25, 13),
            },
            False,
        ),
    ),
)
def test_aggregate_gate_is_preregistered_and_never_stops_or_promotes(
    tmp_path: Path, rates, expected_pass: bool
):
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    pod1 = _pod_result(tmp_path, config, prereg, "pod1", rates)
    pod2 = _pod_result(tmp_path, config, prereg, "pod2", rates)
    output = R.aggregate(
        CONFIG_PATH,
        config,
        PREREG_PATH,
        prereg,
        pod1_result=pod1,
        pod1_sha=R.sha256_file(pod1),
        pod2_result=pod2,
        pod2_sha=R.sha256_file(pod2),
        output_dir=tmp_path / "aggregate",
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    content = document["content"]
    assert content["gate_pass"] is expected_pass
    assert content["actions"]["training"] == "continue_all_arms_unmodified"
    assert content["actions"]["trainer_or_worker_signals"] == []
    assert content["actions"]["stop_or_promote_authorized"] is False
    assert content["actions"]["deploy_or_real_robot_authorized"] is False
    assert content["actions"]["cross_instrument_plant_recovery_gates"] == "remain_open"


def test_runner_exposes_no_ssh_or_signal_control_surface():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in ("paramiko", "ssh ", "os.kill", "killpg", "pkill"):
        assert forbidden not in source
    assert "trainer_or_worker_signal_allowed\": False" in source
    assert "real_robot_authorized\": False" in source


def test_archived_four_seed_result_is_exact_full_denominator_and_gate_fail():
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    config_for_results = dict(config)
    config_for_results["_prereg_arms"] = prereg["arms"]
    expected_pod_shas = {
        "pod1": "0e651edae4e0e237e51ed2445f36e8dcc6903dea6444ad532f13c2819128eedc",
        "pod2": "ad1187a9a24707fe184a4751b73be05e43d46ac5596590ae3e598355f3c39bd4",
    }
    pod1 = R.validate_pod_result(
        POD1_RESULT_PATH,
        expected_pod_shas["pod1"],
        config_for_results,
        R.sha256_file(PREREG_PATH),
        pod="pod1",
    )
    pod2 = R.validate_pod_result(
        POD2_RESULT_PATH,
        expected_pod_shas["pod2"],
        config_for_results,
        R.sha256_file(PREREG_PATH),
        pod="pod2",
    )
    assert {
        seed: arm["returned_counts"]
        for seed, arm in {**pod1["arms"], **pod2["arms"]}.items()
    } == {
        "seed1": {"aggregate": 83, "forehand": 33, "backhand": 50, "physical_falls": 0},
        "seed2": {"aggregate": 100, "forehand": 50, "backhand": 50, "physical_falls": 0},
        "seed3": {"aggregate": 100, "forehand": 50, "backhand": 50, "physical_falls": 0},
        "seed4": {"aggregate": 20, "forehand": 0, "backhand": 20, "physical_falls": 0},
    }

    aggregate = json.loads(AGGREGATE_RESULT_PATH.read_text(encoding="utf-8"))
    assert aggregate["content_sha256"] == (
        "a756bf1d0e76d1016992ae241b935cf92b3c84ffd55fe503e7c199626d9c8ffd"
    )
    assert aggregate["content_sha256"] == R.canonical_sha256(aggregate["content"])
    content = aggregate["content"]
    assert content["status"] == "fail_seed_stability_checkpoint_evidence"
    assert content["aggregate_return_rates"] == {
        "seed1": 0.83,
        "seed2": 1.0,
        "seed3": 1.0,
        "seed4": 0.2,
    }
    assert content["gate_checks"]["aggregate_rate_median"]["pass"] is True
    assert content["gate_checks"]["aggregate_rate_min_seed"]["pass"] is False
    assert content["gate_checks"]["aggregate_rate_max_minus_min"]["pass"] is False
    assert content["gate_checks"]["every_seed_every_side_rate"]["pass"] is False
    assert content["gate_pass"] is False
    assert content["actions"]["training"] == "continue_all_arms_unmodified"
    assert content["actions"]["seed_stability_gate"] == "keep_open"
    assert content["actions"]["trainer_or_worker_signals"] == []
    assert content["actions"]["stop_or_promote_authorized"] is False
    assert content["actions"]["deploy_or_real_robot_authorized"] is False
