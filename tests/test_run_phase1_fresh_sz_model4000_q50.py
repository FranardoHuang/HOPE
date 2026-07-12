from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_fresh_sz_model4000_q50.py"
CONFIG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model4000_seed_stability_q50_execution_20260712.json"
)
QUEUE_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json"
)
PREREG_PATH = (
    ROOT / "configs" / "phase1_fresh_SZ_model4000_seed_stability_q50_prereg_20260712.json"
)
MODEL2000_POD1 = (
    ROOT / "configs" / "phase1_fresh_SZ_model2000_seed_stability_q50_pod1_result_20260711.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module("fresh_sz_model4000_q50_under_test", RUNNER_PATH)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config() -> dict:
    return _json(CONFIG_PATH)


def _prereg() -> dict:
    return _json(PREREG_PATH)


def _checkpoint_audit() -> dict:
    return {
        "iter": 4000,
        "training_contract_sha256": R.barrier.EXPECTED_HARD_CONTRACT_SHA256,
        "training_contract_schema_version": 3,
        "training_contract_lineage_exact": True,
        "tensor_count": 76,
        "floating_tensor_count": 74,
        "floating_elements": 1762715,
        "nonfinite": 0,
    }


def _pod_audit_document(pod: str) -> dict:
    queue = _json(QUEUE_PATH)
    prereg = _prereg()
    arms = {}
    for seed in R.POD_ARM_ORDER[pod]:
        digit = str(int(seed[4:]))
        arm = prereg["arms"][seed]
        checkpoint_sha = digit * 64
        if seed == "seed1":
            checkpoint_sha = queue["seed1_reuse"]["expected_checkpoint_sha256"]
        arms[seed] = {
            "checkpoint_path": arm["checkpoint_path"],
            "checkpoint_sha256": checkpoint_sha,
            "training_contract_path": arm["training_contract_path"],
            "training_contract_sha256": R.barrier.EXPECTED_HARD_CONTRACT_SHA256,
            "checkpoint_audit": _checkpoint_audit(),
        }
    content = {
        "queue_id": queue["queue_id"],
        "status": "pod_checkpoints_ready_judge_not_started",
        "completed_utc": "2026-07-12T01:00:00Z",
        "pod": pod,
        "queue": {"path": str(QUEUE_PATH), "sha256": R.sha256_file(QUEUE_PATH)},
        "preregistration": {
            "path": str(PREREG_PATH),
            "sha256": R.sha256_file(PREREG_PATH),
        },
        "validator_sha256": R.QUEUE_VALIDATOR_SHA256,
        **R.EXPECTED_SEMANTICS,
        "shared_schedule": {
            "path": "/paper/shared_clean_k100.schedule.json",
            "file_sha256": R.EXPECTED_SCHEDULE["file_sha256"],
            "semantic_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
            "question_id_order_sha256": R.EXPECTED_SCHEDULE[
                "question_id_order_sha256"
            ],
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
    return R.barrier._content_document(
        "phase1_fresh_sz_model4000_q50_pod_ready_audit", content
    )


def _activation(tmp_path: Path) -> tuple[Path, dict]:
    pod_paths = {}
    pod_contents = {}
    for pod in R.POD_ARM_ORDER:
        path = tmp_path / f"{pod}_audit.json"
        document = _pod_audit_document(pod)
        _write(path, document)
        pod_paths[pod] = path
        pod_contents[pod] = document["content"]
    arms = {**pod_contents["pod1"]["arms"], **pod_contents["pod2"]["arms"]}
    content = {
        "queue_id": _json(QUEUE_PATH)["queue_id"],
        "barrier_id": "fresh_SZ_model4000_all_four_ready_v1",
        "status": "all_four_checkpoints_ready_judge_not_started",
        "activated_utc": "2026-07-12T01:10:00Z",
        "queue": {"path": str(QUEUE_PATH), "sha256": R.sha256_file(QUEUE_PATH)},
        "preregistration": {
            "path": str(PREREG_PATH),
            "sha256": R.sha256_file(PREREG_PATH),
        },
        "validator_sha256": R.QUEUE_VALIDATOR_SHA256,
        "pod_audits": {
            pod: {"path": str(path), "sha256": R.sha256_file(path)}
            for pod, path in pod_paths.items()
        },
        **R.EXPECTED_SEMANTICS,
        "shared_schedule": pod_contents["pod1"]["shared_schedule"],
        "seed_order": list(R.SEED_ORDER),
        "arms": {seed: arms[seed] for seed in R.SEED_ORDER},
        "gate_rule": R.EXPECTED_GATE_RULE,
        "actions": {
            "judges_started": 0,
            "trainer_or_worker_signals": [],
            "future_q50_runner_may_prepare_only_with_this_exact_artifact": True,
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    document = R._content_document(
        "phase1_fresh_sz_model4000_q50_all_four_activation", content
    )
    path = tmp_path / f"activation_{document['content_sha256']}.json"
    _write(path, document)
    return path, document


def _validated_activation(tmp_path: Path) -> dict:
    path, _ = _activation(tmp_path)
    return R._validate_activation_document(
        path,
        R.sha256_file(path),
        QUEUE_PATH,
        _json(QUEUE_PATH),
        PREREG_PATH,
    )


def test_committed_execution_is_exact_activation_bound_and_seed1_reruns():
    config = R.load_execution_config(CONFIG_PATH)
    assert R.sha256_file(RUNNER_PATH) == config["source_bindings"]["runner"]["sha256"]
    assert config["source_bindings"]["queue"]["sha256"] == R.EXPECTED_QUEUE_SHA256
    assert config["source_bindings"]["preregistration"]["sha256"] == R.EXPECTED_PREREG_SHA256
    assert config["source_bindings"]["queue_validator"]["sha256"] == R.QUEUE_VALIDATOR_SHA256
    assert config["source_bindings"]["fresh_helper"]["sha256"] == R.FRESH_HELPER_SHA256
    assert config["activation"]["required"] is True
    assert config["runtime"]["seed1_execution"] == "fresh_rerun_same_k100_no_reuse"
    assert config["schedule"]["materialize_new_schedule_allowed"] is False
    assert config["runtime"]["kit_boot_lock"] == "/workspace/.kit_boot.lock"
    assert config["interpretation_rule"]["known_seed1_model4000"] == {
        "aggregate_return_rate": 0.5,
        "forehand_return_rate": 0.0,
        "backhand_return_rate": 1.0,
        "family_gate_pass_possible": False,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("auto_start", True),
        lambda value: value["source_bindings"]["queue"].__setitem__("sha256", "0" * 64),
        lambda value: value["source_bindings"]["runner"].__setitem__("sha256", "0" * 64),
        lambda value: value["activation"].__setitem__("required", False),
        lambda value: value["schedule"].__setitem__("materialize_new_schedule_allowed", True),
        lambda value: value["runtime"].__setitem__("seed1_execution", "reuse"),
        lambda value: value["runtime"].__setitem__("kit_boot_lock", "/tmp/private.lock"),
        lambda value: value["gate_rule"].__setitem__("aggregate_rate_min_seed_min", 0.2),
        lambda value: value["interpretation_rule"]["known_seed1_model4000"].__setitem__(
            "family_gate_pass_possible", True
        ),
    ),
)
def test_execution_config_rejects_relaxation_repaper_or_reuse(tmp_path: Path, mutation):
    config = _config()
    mutation(config)
    path = tmp_path / "mutated.json"
    _write(path, config)
    with pytest.raises(R.ContractError):
        R.load_execution_config(path)


def test_activation_revalidates_both_pod_audits_and_all_four_embedded_checkpoint_audits(
    tmp_path: Path,
):
    activation = _validated_activation(tmp_path)
    assert list(activation["content"]["arms"]) == list(R.SEED_ORDER)
    assert set(activation["pod_contents"]) == {"pod1", "pod2"}
    assert all(
        activation["content"]["arms"][seed]["checkpoint_audit"]["iter"] == 4000
        for seed in R.SEED_ORDER
    )
    assert all(
        activation["content"]["arms"][seed]["checkpoint_audit"]["nonfinite"] == 0
        for seed in R.SEED_ORDER
    )

    path, document = _activation(tmp_path / "broken")
    document["content"]["arms"]["seed4"]["checkpoint_audit"]["nonfinite"] = 1
    document["content_sha256"] = R.canonical_sha256(document["content"])
    _write(path, document)
    with pytest.raises(R.ContractError):
        R._validate_activation_document(
            path,
            R.sha256_file(path),
            QUEUE_PATH,
            _json(QUEUE_PATH),
            PREREG_PATH,
        )


def test_activation_rejects_single_pod_or_changed_audit_bytes(tmp_path: Path):
    path, document = _activation(tmp_path)
    del document["content"]["pod_audits"]["pod2"]
    document["content_sha256"] = R.canonical_sha256(document["content"])
    _write(path, document)
    with pytest.raises(R.ContractError, match="two Pod audits"):
        R._validate_activation_document(
            path,
            R.sha256_file(path),
            QUEUE_PATH,
            _json(QUEUE_PATH),
            PREREG_PATH,
        )

    path, document = _activation(tmp_path / "changed")
    pod1_path = Path(document["content"]["pod_audits"]["pod1"]["path"])
    pod1_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(R.ContractError):
        R._validate_activation_document(
            path,
            R.sha256_file(path),
            QUEUE_PATH,
            _json(QUEUE_PATH),
            PREREG_PATH,
        )


def test_contract_check_is_read_only(monkeypatch, tmp_path: Path):
    observed = []
    activation = {"content": {"shared_schedule": {"path": "/paper/schedule.json"}}}
    monkeypatch.setattr(
        R,
        "_load_bound_inputs",
        lambda _args: (
            CONFIG_PATH,
            _config(),
            QUEUE_PATH,
            _json(QUEUE_PATH),
            PREREG_PATH,
            _prereg(),
            activation,
        ),
    )
    monkeypatch.setattr(
        R,
        "_validate_runtime_inputs",
        lambda *_args, **kwargs: observed.append((kwargs["pod"], kwargs["schedule_path"])),
    )
    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("contract-check must not launch"),
    )
    schedule = tmp_path / "schedule.json"
    assert (
        R.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--expected-config-sha256",
                "0" * 64,
                "--activation",
                str(tmp_path / "activation.json"),
                "--expected-activation-sha256",
                "1" * 64,
                "contract-check",
                "--pod",
                "pod1",
                "--schedule-source",
                str(schedule),
            ]
        )
        == 0
    )
    assert observed == [("pod1", schedule.resolve())]
    assert list(tmp_path.iterdir()) == []


def test_prepare_is_no_clobber_activation_bound_and_never_materializes(
    monkeypatch, tmp_path: Path
):
    config = _config()
    config["runtime"]["pod_state_dirs"]["pod1"] = str(tmp_path / "pod1")
    config_path = tmp_path / "execution.json"
    prereg_path = tmp_path / "prereg.json"
    _write(config_path, config)
    _write(prereg_path, _prereg())
    schedule_source = tmp_path / "source.schedule.json"
    schedule_source.write_bytes(b'{"frozen":true}\n')
    schedule = {
        "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
        "items": [{"question_id": f"{index:064x}"} for index in range(100)],
    }
    activation = {
        "path": tmp_path / "activation.json",
        "sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "content": {
            "barrier_id": "fresh_SZ_model4000_all_four_ready_v1",
            "pod_audits": {"pod1": {"path": "/audit", "sha256": "c" * 64}},
            "arms": {
                seed: {"checkpoint_sha256": str(int(seed[4:])) * 64}
                for seed in R.POD_ARM_ORDER["pod1"]
            },
        },
    }
    monkeypatch.setattr(
        R,
        "_validate_runtime_inputs",
        lambda *_args, **_kwargs: (
            ROOT,
            ROOT,
            {},
            {"seed1": _checkpoint_audit(), "seed3": _checkpoint_audit()},
            schedule,
        ),
    )
    monkeypatch.setattr(R, "_validate_schedule", lambda *_args, **_kwargs: schedule)
    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("prepare must not launch"),
    )
    assert (
        R.prepare(
            config_path,
            config,
            QUEUE_PATH,
            prereg_path,
            _prereg(),
            activation,
            pod="pod1",
            schedule_source=schedule_source,
        )
        == 0
    )
    state_dir = Path(config["runtime"]["pod_state_dirs"]["pod1"])
    copied = state_dir / config["runtime"]["schedule_filename"]
    runtime_path = state_dir / config["runtime"]["runtime_contract_filename"]
    assert copied.read_bytes() == schedule_source.read_bytes()
    runtime = _json(runtime_path)
    assert runtime["activation"] == {
        "path": str(activation["path"]),
        "sha256": activation["sha256"],
        "content_sha256": activation["content_sha256"],
        "barrier_id": "fresh_SZ_model4000_all_four_ready_v1",
        "pod_audit": {"path": "/audit", "sha256": "c" * 64},
    }
    assert runtime["status"] == "prepared_not_started"
    with pytest.raises(R.ContractError, match="no-clobber"):
        R.prepare(
            config_path,
            config,
            QUEUE_PATH,
            prereg_path,
            _prereg(),
            activation,
            pod="pod1",
            schedule_source=schedule_source,
        )


class _FakeProc:
    def __init__(self, pid: int = 43210, rc: int = 0):
        self.pid = pid
        self._rc = rc

    def wait(self):
        return self._rc


def test_judge_launch_uses_shared_kit_lock_new_session_and_exact_pid_pgid(
    monkeypatch, tmp_path: Path
):
    observed = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(R.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        R.fresh,
        "build_judge_command",
        lambda **_kwargs: ["bash", "/eval/judge.sh", "--exam-schedule-json /paper"],
    )
    report = tmp_path / "run" / "judge" / "judge_report_model_4000_20260712_010203.md"
    report.parent.mkdir(parents=True)
    report.write_text("report\n", encoding="utf-8")
    monkeypatch.setattr(R.fresh.base, "find_report", lambda _text: report)
    arm = {"checkpoint_sha256": "d" * 64}
    launched = R._run_judge(
        seed="seed1",
        arm=arm,
        tools={"judge": tmp_path / "eval" / "scripts" / "judge.sh"},
        schedule_path=tmp_path / "paper.json",
        state_dir=tmp_path,
        runtime_sha="e" * 64,
        activation_sha="f" * 64,
        kit_boot_lock="/workspace/.kit_boot.lock",
        gpu=2,
    )
    assert observed["start_new_session"] is True
    assert observed["env"]["JUDGE_KIT_BOOT_LOCK"] == "/workspace/.kit_boot.lock"
    state = _json(launched["state_path"])
    assert state["pid"] == state["pgid"] == 43210
    assert state["status"] == "process_complete_unvalidated"


def _question_order() -> list[str]:
    archived = _json(MODEL2000_POD1)["content"]
    return archived["arms"]["seed1"]["question_id_order"]


def _arm_result(activation: dict, prereg: dict, seed: str, aggregate: int, forehand: int) -> dict:
    return {
        "run_name": prereg["arms"][seed]["run_name"],
        "checkpoint_iteration": 4000,
        "checkpoint_sha256": activation["content"]["arms"][seed]["checkpoint_sha256"],
        "training_contract_sha256": R.barrier.EXPECTED_HARD_CONTRACT_SHA256,
        "report": {"path": f"/pod/{seed}/report.md", "sha256": "a" * 64},
        "summary": {"path": f"/pod/{seed}/summary.json", "sha256": "b" * 64},
        "attempt_ledger": {"path": f"/pod/{seed}/attempts.csv", "sha256": "c" * 64},
        "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
        "question_id_order": _question_order(),
        "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
        "execution_contract_sha256": "d" * 64,
        "ready_state_sha256": "e" * 64,
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
        "returned_rates": {
            "aggregate": aggregate / 100,
            "forehand": forehand / 50,
            "backhand": (aggregate - forehand) / 50,
        },
        "checkpoint_audit": activation["content"]["arms"][seed]["checkpoint_audit"],
        "raw_chain_revalidated_at_pod_run": True,
    }


def _pod_result(
    tmp_path: Path,
    config: dict,
    prereg: dict,
    activation: dict,
    pod: str,
    rates: dict[str, tuple[int, int]],
) -> Path:
    content = {
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "complete",
        "completed_utc": "2026-07-12T02:00:00Z",
        "runtime_contract": {"path": "/runtime.json", "sha256": "f" * 64},
        "activation": {
            "path": str(activation["path"]),
            "sha256": activation["sha256"],
            "content_sha256": activation["content_sha256"],
            "barrier_id": activation["content"]["barrier_id"],
        },
        "config_sha256": R.sha256_file(CONFIG_PATH),
        "queue_sha256": R.EXPECTED_QUEUE_SHA256,
        "preregistration_sha256": R.EXPECTED_PREREG_SHA256,
        "queue_validator_sha256": R.QUEUE_VALIDATOR_SHA256,
        "fresh_helper_sha256": R.FRESH_HELPER_SHA256,
        "runner_sha256": config["source_bindings"]["runner"]["sha256"],
        **R.EXPECTED_SEMANTICS,
        "shared_schedule": {
            "path": "/pod/shared_clean_k100.schedule.json",
            "file_sha256": R.EXPECTED_SCHEDULE["file_sha256"],
            "schedule_sha256": R.EXPECTED_SCHEDULE["semantic_sha256"],
            "question_id_order": _question_order(),
            "question_id_order_sha256": R.EXPECTED_SCHEDULE[
                "question_id_order_sha256"
            ],
            "schedule_k": 100,
            "attempts_per_side": 50,
            "seed": 0,
            "hold_range": [0, 100],
        },
        "kit_boot_lock": "/workspace/.kit_boot.lock",
        "arm_order": list(R.POD_ARM_ORDER[pod]),
        "execution": {
            seed: {
                "mode": "fresh_identical_paper_judge",
                "seed1_reused": False,
                "state": {"path": f"/pod/{seed}/state.json", "sha256": "1" * 64},
                "runner_log": {"path": f"/pod/{seed}/runner.log", "sha256": "2" * 64},
            }
            for seed in R.POD_ARM_ORDER[pod]
        },
        "arms": {
            seed: _arm_result(activation, prereg, seed, *rates[seed])
            for seed in R.POD_ARM_ORDER[pod]
        },
        "gate_rule": R.EXPECTED_GATE_RULE,
        "interpretation_rule": R.EXPECTED_INTERPRETATION,
        "actions": {
            "training": "continue_all_arms_unmodified",
            "trainer_or_worker_signals": [],
            "stop_or_promote_authorized": False,
            "deploy_or_real_robot_authorized": False,
        },
    }
    path = tmp_path / f"{pod}.json"
    _write(path, R._content_document("phase1_fresh_sz_model4000_q50_pod", content))
    return path


@pytest.mark.parametrize(
    ("seed4", "expected"),
    (
        ((70, 30), "delayed_learning_supported_at_model4000"),
        ((60, 30), "persistent_weakness_through_model4000"),
        ((70, 20), "persistent_weakness_through_model4000"),
    ),
)
def test_aggregate_never_family_passes_and_classifies_seed4_only_on_unchanged_thresholds(
    tmp_path: Path, seed4: tuple[int, int], expected: str
):
    config = _config()
    config["runtime"]["aggregate_output_dir"] = str(tmp_path / "aggregate")
    prereg = _prereg()
    activation = _validated_activation(tmp_path / "activation")
    rates = {
        "seed1": (80, 40),
        "seed2": (82, 41),
        "seed3": (78, 39),
        "seed4": seed4,
    }
    pod1 = _pod_result(tmp_path, config, prereg, activation, "pod1", rates)
    pod2 = _pod_result(tmp_path, config, prereg, activation, "pod2", rates)
    output = R.aggregate(
        CONFIG_PATH,
        config,
        PREREG_PATH,
        prereg,
        activation,
        pod1_result=pod1,
        pod1_sha=R.sha256_file(pod1),
        pod2_result=pod2,
        pod2_sha=R.sha256_file(pod2),
        output_dir=Path(config["runtime"]["aggregate_output_dir"]),
    )
    content = _json(output)["content"]
    assert content["gate_pass"] is False
    assert content["family_stable_claim_allowed"] is False
    assert content["known_before_prereg"]["aggregate_return_rate"] == 0.5
    assert content["seed4_interpretation"]["classification"] == expected
    assert content["seed4_interpretation"]["thresholds_unchanged"] is True
    assert content["actions"]["training"] == "continue_all_arms_unmodified"
    assert content["actions"]["trainer_or_worker_signals"] == []
    assert content["actions"]["stop_or_promote_authorized"] is False
    assert content["actions"]["deploy_or_real_robot_authorized"] is False


def test_pod_result_rejects_missing_raw_chain_sha_or_seed1_reuse(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    activation = _validated_activation(tmp_path / "activation")
    rates = {"seed1": (50, 0), "seed3": (80, 40)}
    path = _pod_result(tmp_path, config, prereg, activation, "pod1", rates)
    document = _json(path)
    del document["content"]["arms"]["seed1"]["attempt_ledger"]
    document["content_sha256"] = R.canonical_sha256(document["content"])
    _write(path, document)
    with pytest.raises(R.ContractError, match="attempt_ledger"):
        R._validate_pod_result(
            path,
            R.sha256_file(path),
            config,
            prereg,
            activation,
            R.sha256_file(CONFIG_PATH),
            pod="pod1",
        )

    path = _pod_result(tmp_path / "reuse", config, prereg, activation, "pod1", rates)
    document = _json(path)
    document["content"]["execution"]["seed1"]["seed1_reused"] = True
    document["content_sha256"] = R.canonical_sha256(document["content"])
    _write(path, document)
    with pytest.raises(R.ContractError, match="freshly judged"):
        R._validate_pod_result(
            path,
            R.sha256_file(path),
            config,
            prereg,
            activation,
            R.sha256_file(CONFIG_PATH),
            pod="pod1",
        )


def test_runner_has_no_ssh_signal_or_schedule_materialization_surface():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "paramiko",
        "ssh ",
        "os.kill",
        "killpg",
        "pkill",
        "SIGTERM",
    ):
        assert forbidden not in source
    assert "tools[\"materialize_schedule\"]" not in source
    assert "start_new_session=True" in source
    assert "JUDGE_KIT_BOOT_LOCK=kit_boot_lock" in source
    assert R.EXPECTED_SEMANTICS["trainer_or_worker_signal_allowed"] is False
    assert R.EXPECTED_SEMANTICS["real_robot_authorized"] is False
