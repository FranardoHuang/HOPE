"""CPU-only fail-closed tests for the executable A225 four-arm launcher."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_action_ball_a225_four_arm_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("launch_a225_four_arm", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _write(path: Path, value) -> str:
    raw = launcher._B._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sealed(value):
    return {**value, "content_sha256": launcher.canonical_sha256(value)}


def _lineage(checkout: Path) -> dict:
    pins = {}
    for key in (
        "bundle",
        "motion",
        "immutable_tape",
        "action_manifest",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
    ):
        raw = ("a225-%s\n" % key).encode()
        path = checkout / (key + ".bin")
        path.write_bytes(raw)
        pins[key] = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
    return {
        "schema_version": 1,
        "kind": launcher.LINEAGE_KIND,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": 225,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": 318,
        "task_profile": launcher.TASK_PROFILE_ID,
        "gym_task": launcher.GYM_TASK_ID,
        "target_semantics": launcher.TARGET_SEMANTICS,
        "action_id": "take_061_unit04_bh",
        "teacher_id": "Take_061_unit04_BH",
        "seed": 0,
        **pins,
    }


def _result(path: Path, *, stage: str, materialization, oracle=None, predecessor=None) -> dict:
    unsigned = {
        "schema_version": 1,
        "kind": launcher.RESULT_KIND,
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": "1" * 64,
        "stage": stage,
        "namespace": "/tmp/a225-fixture-" + stage,
        "completion": {"terminal_kind": "clean_completion"},
        "gpu_admission": {"phase": "post_completion"},
        "output_contract": {"fixture": True},
        "arm_materialization": materialization,
        "oracle32_receipt": oracle,
        "predecessor_result": predecessor,
    }
    digest = _write(path, _sealed(unsigned))
    return {"path": str(path), "sha256": digest}


def _generated_chain(tmp_path: Path, arm_id: str, lineage_sha: str):
    arm = launcher._arm_contract(arm_id)
    materialization = launcher._planned_materialization(
        arm=arm, lineage={"lineage_sha256": lineage_sha}
    )
    materialize = _result(
        tmp_path / (arm_id + ".materialize.json"),
        stage="materialize",
        materialization=materialization,
    )
    oracle = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.ORACLE32_KIND,
            "diagnostic_unauthorized": True,
            "verdict": "PASS",
            "episodes": 32,
            "arm_id": arm_id,
            "lineage_sha256": lineage_sha,
            "arm_contract_sha256": arm["arm_contract_sha256"],
            "reward_contract_sha256": materialization["reward_contract_sha256"],
            "runtime_effective_reward_sha256": "3" * 64,
            "policy_contract_sha256": materialization["policy_contract_sha256"],
            "runtime_policy_recipe_sha256": "4" * 64,
            "actor_contract": launcher.ACTOR_CONTRACT,
            "actor_width": 225,
            "critic_contract": launcher.CRITIC_CONTRACT,
            "critic_width": 318,
            "seed": 0,
            "raw_oracle_sha256": "2" * 64,
        }
    )
    oracle_result = _result(
        tmp_path / (arm_id + ".oracle32.json"),
        stage="oracle32",
        materialization=materialization,
        oracle=oracle,
    )
    smoke_result = _result(
        tmp_path / (arm_id + ".smoke.json"),
        stage="smoke",
        materialization=materialization,
        oracle=oracle,
    )
    probe_result = _result(
        tmp_path / (arm_id + ".probe512.json"),
        stage="probe512",
        materialization=materialization,
        oracle=oracle,
        predecessor={"stage": "smoke"},
    )
    return materialize, oracle_result, smoke_result, probe_result


def _case(tmp_path: Path, *, arm_id: str, stage: str, allow_colocation: bool = False):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    lineage = _lineage(checkout)
    lineage_path = checkout / "a225_lineage.json"
    lineage_sha = _write(lineage_path, lineage)
    generated = _generated_chain(tmp_path, arm_id, lineage_sha)
    materialize, oracle_result, smoke_result, probe_result = generated
    root = tmp_path / launcher.EXPERIMENT_NAME
    root.mkdir()
    namespace = root / (arm_id + "-" + stage)
    budget = launcher.BUDGETS[stage]
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "arm_id": arm_id,
        "lineage": {"path": lineage_path.name, "sha256": lineage_sha},
        "arm_materialization": None if stage == "materialize" else materialize,
        "oracle32_receipt": oracle_result if stage in ("smoke", "probe512", "long512") else None,
        "predecessor_result": (
            smoke_result if stage == "probe512" else probe_result if stage == "long512" else None
        ),
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": 2,
            "uuid": "GPU-12345678",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
            "require_empty": not allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if allow_colocation:
        spec[launcher.COLOCATION_SPEC_KEY] = True
    spec_path = tmp_path / (arm_id + "-" + stage + ".spec.json")
    _write(spec_path, spec)
    return spec_path, spec, lineage


def _patch_plan_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda checkout, commit: {"checkout": str(checkout), "commit_sha": commit, "clean": True},
    )
    monkeypatch.setattr(launcher, "_runtime_sources", lambda checkout, commit: {})
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_environment", lambda: {"kind": "test_runtime_assets"}
    )

    def verify(checkout, commit, pin, *, name):
        path = checkout / pin["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
        return dict(pin), path

    monkeypatch.setattr(launcher._B, "_verify_tracked_file", verify)


def _flatten_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, str):
        yield value


def test_four_code_owned_arms_are_exact():
    expected = {
        launcher.ARM_IDS[0]: (-30.0, -0.5, "metrics_only", "fixed", 1e-4),
        launcher.ARM_IDS[1]: (-300.0, -5.0, "metrics_only", "fixed", 1e-4),
        launcher.ARM_IDS[2]: (-30.0, -0.5, "phase_gated", "fixed", 1e-4),
        launcher.ARM_IDS[3]: (-30.0, -0.5, "phase_gated", "adaptive", 1e-3),
    }
    assert tuple(launcher.ARMS) == launcher.ARM_IDS
    for arm_id, values in expected.items():
        arm = launcher._arm_contract(arm_id)
        assert (arm["soft_weights"]["death_penalty"], arm["soft_weights"]["qdes_limit"], arm["reference_guard_mode"], arm["ppo"]["schedule"], arm["ppo"]["learning_rate"]) == values
        assert arm["actor_hidden_dims"] == arm["critic_hidden_dims"] == [512, 256, 128]
        assert arm["init_noise_std"] == 0.02
        assert arm["entropy_coef"] == 0.01


@pytest.mark.parametrize("stage,budget", list(launcher.BUDGETS.items()))
def test_stage_budgets_are_code_owned(stage, budget):
    assert launcher.BUDGETS[stage] == budget


def test_plan_claim_is_a225_fresh_and_denies_retired_lineage(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[2], stage="long512")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["fresh_only"] is True
    assert payload["single_gpu"] is True
    assert payload["max_compute_pids_on_physical_gpu"] == 2
    assert payload["gpu_default_empty"] is True
    assert payload["vendor_v2_colocation_opt_in"] is False
    assert payload["bundle"]["lineage"]["actor_contract"] == launcher.ACTOR_CONTRACT
    assert payload["bundle"]["normalizers"] == launcher._normalizer_contract()
    assert payload["bundle"]["continuation_stop_gate"]["iter500_quantitative_threshold_status"] == "UNSET"
    assert payload["output_contract"]["speed_benchmark_eligible"] is True
    flattened = "\n".join(_flatten_strings(payload)).lower()
    for retired in ("target_recipe", "target_validity_mask", "action_ball_c225", "c225", "l194", "checkpoint"):
        assert retired not in flattened


def test_retired_vocabulary_scan_treats_hashes_as_opaque():
    launcher._assert_no_retired_contract(
        {"spec_file_sha256": "0" * 12 + "c225" + "0" * 48},
        name="opaque digest",
    )
    with pytest.raises(launcher.LaunchRefused, match="retired ABI/arm token"):
        launcher._assert_no_retired_contract(
            {"obs_mode": "action_ball_c225"}, name="semantic value"
        )


def test_training_argv_pins_a225_lineage_bootstrap_and_optimizer(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[3], stage="probe512")
    argv = launcher.build_plan(spec_path)["canonical_payload"]["training_argv"]
    for exact in (
        "task=HOPEPingPongActionBallA225VendorV2N1Learnability",
        "task.actor_obs_contract=action_ball_a225",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.algorithm.schedule=adaptive",
        "algo.algorithm.learning_rate=0.001",
        "+task.racket.reference_guard_mode=phase_gated",
        "action_ball_dynamic_ready_bootstrap=true",
    ):
        assert exact in argv
    joined = "\n".join(argv)
    assert "action_ball_policy_contract_sha256=" in joined
    assert "action_ball_manifest_sha256=" in joined
    assert "target_recipe" not in joined and "validity_mask" not in joined


def test_full_stage_chain_is_enforced(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    for stage in ("materialize", "oracle32", "smoke", "probe512", "long512"):
        spec_path, _spec, _lineage = _case(tmp_path / stage, arm_id=launcher.ARM_IDS[0], stage=stage)
        assert launcher.build_plan(spec_path)["canonical_payload"]["spec"]["stage"] == stage
    spec_path, spec, _ = _case(tmp_path / "missing", arm_id=launcher.ARM_IDS[0], stage="probe512")
    spec["predecessor_result"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="completed smoke"):
        launcher.build_plan(spec_path)


def test_cross_arm_or_oracle_content_drift_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="smoke")
    oracle_path = Path(spec["oracle32_receipt"]["path"])
    outer = json.loads(oracle_path.read_text())
    outer["oracle32_receipt"]["arm_id"] = launcher.ARM_IDS[1]
    receipt = dict(outer["oracle32_receipt"])
    receipt.pop("content_sha256")
    outer["oracle32_receipt"] = _sealed(receipt)
    outer.pop("content_sha256")
    outer = _sealed(outer)
    spec["oracle32_receipt"]["sha256"] = _write(oracle_path, outer)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="binding differs"):
        launcher.build_plan(spec_path)


def test_default_empty_gpu_and_explicit_colocation_claim_scope(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    default_path, _, _ = _case(tmp_path / "default", arm_id=launcher.ARM_IDS[0], stage="materialize")
    default = launcher.build_plan(default_path)["canonical_payload"]
    assert default["spec"]["gpu"]["require_empty"] is True
    assert default["output_contract"]["speed_benchmark_eligible"] is True

    opted_path, _, _ = _case(tmp_path / "opted", arm_id=launcher.ARM_IDS[0], stage="materialize", allow_colocation=True)
    opted = launcher.build_plan(opted_path)["canonical_payload"]
    assert opted["spec"]["gpu"]["require_empty"] is False
    assert opted["vendor_v2_colocation_opt_in"] is True
    assert opted["output_contract"]["speed_benchmark_eligible"] is False
    assert opted["output_contract"]["colocation_result_scope"] == "training_diagnostic_only"


def test_colocation_gpu_validation_is_cross_bound_and_fail_closed(tmp_path, monkeypatch):
    _, raw_spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize", allow_colocation=True)
    spec = launcher._validate_spec(raw_spec)
    query = lambda index, uuid: {
        "total_memory_mib": 24576,
        "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "processes": [{"pid": 99}],
        "nvidia_smi_path": "/usr/bin/nvidia-smi",
        "nvidia_smi_sha256": "3" * 64,
    }
    monkeypatch.setattr(launcher, "_query_gpu_processes", query)
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_gpu_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(launcher.LaunchRefused("unknown GPU co-resident")),
    )
    with pytest.raises(launcher.LaunchRefused, match="unknown GPU"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)

    monkeypatch.setattr(launcher, "_query_gpu_processes", lambda index, uuid: {**query(index, uuid), "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB - 1, "processes": []})
    with pytest.raises(launcher.LaunchRefused, match="below conservative headroom"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)


def test_scale_execute_blocks_before_any_mutation(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[1], stage="scale4096")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher, "_open_gpu_shared_lock", lambda path: pytest.fail("lock opened"))
    with pytest.raises(launcher.LaunchRefused, match="independently BLOCKED"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_confirm_digest_mismatch_blocks_before_source_lock_or_namespace(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_verify_clean_source", lambda *args: pytest.fail("source touched"))
    with pytest.raises(launcher.LaunchRefused, match="confirm-claim differs"):
        launcher.execute(plan, confirm_claim="f" * 64)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_claim_namespace_is_no_clobber(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    original = (namespace / "launch_claim.json").read_bytes()
    with pytest.raises(launcher.LaunchRefused):
        launcher._B._claim_namespace(plan)
    assert (namespace / "launch_claim.json").read_bytes() == original


def test_pre_exec_admission_race_refuses_before_execve(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(
        launcher,
        "_verify_gpu_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launcher.LaunchRefused("pre_exec race occupied GPU")
        ),
    )
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve reached"))
    lock_path = Path(plan["canonical_payload"]["spec"]["gpu"]["lock_path"])
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(launcher.LaunchRefused, match="pre_exec race"):
            launcher._internal_exec(
                namespace / "launch_claim.json", plan["launch_claim_sha256"], lock_fd
            )
    finally:
        os.close(lock_fd)
    assert not (namespace / "pre_exec_gpu_admission.json").exists()


def test_post_boot_admission_failure_routes_exact_cleanup_and_spends_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="long512")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    lock_file = tmp_path / "gpu.lock"
    monkeypatch.setattr(
        launcher, "_open_gpu_shared_lock", lambda path: os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    phases = []

    def admission(spec, *, phase, current_namespace, require_current_compute=False, **kwargs):
        phases.append((phase, require_current_compute))
        if phase == "post_boot":
            raise launcher.LaunchRefused("post_boot unknown pid")
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", admission)
    monkeypatch.setattr(
        launcher, "_reservation_document", lambda spec, digest: {"claim": digest}
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )
    cleanup_calls = []

    def cleanup(namespace, state, claim_sha, reason):
        cleanup_calls.append((namespace, state, claim_sha, reason))
        return {"cleanup": {"completed": True}, "path": namespace / "cleanup-failure.json"}

    monkeypatch.setattr(launcher, "_cleanup_post_boot_admission_failure", cleanup)
    with pytest.raises(launcher.LaunchRefused, match=r"cleanup completed.*cleanup-failure.json"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    assert namespace.is_dir()
    assert (namespace / "launch_claim.json").is_file()
    assert phases == [("pre_launch", False), ("post_boot", True)]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][0] == namespace
    assert cleanup_calls[0][2] == plan["launch_claim_sha256"]


def test_claim_revalidation_detects_code_owned_bundle_mutation(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    namespace.mkdir()
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["normalizers"]["actor"]["state"] = "donor"
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    with pytest.raises(launcher.LaunchRefused, match="drifted"):
        launcher._revalidate_claim_payload(payload)


@pytest.mark.parametrize("retired_key", ("target_recipe", "target_validity_mask", "resume_path", "checkpoint_path"))
def test_spec_rejects_retired_control_keys(tmp_path, retired_key):
    _, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    spec[retired_key] = "forbidden"
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_spec(spec)


def test_template_colocation_and_python_symlink(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\n")
    real_python.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    output = tmp_path / "template.json"
    root = tmp_path / launcher.EXPERIMENT_NAME
    root.mkdir()
    args = launcher._parser().parse_args([
        "template", "--output", str(output), "--checkout", str(checkout),
        "--commit-sha", "a" * 40, "--isaac-python", str(venv_python),
        "--arm-id", launcher.ARM_IDS[0], "--lineage-path", "a225.json",
        "--lineage-sha256", "b" * 64, "--stage", "materialize",
        "--gpu-index", "2", "--gpu-uuid", "GPU-12345678", "--owner", "Franco",
        "--namespace", str(root / "fresh"), "--allow-colocation",
    ])
    launcher._write_template(args)
    document = json.loads(output.read_text())
    assert document["source"]["isaac_python"] == str(venv_python)
    assert document[launcher.COLOCATION_SPEC_KEY] is True
    assert document["gpu"]["require_empty"] is False


def test_parser_exposes_explicit_execute_and_hidden_exec():
    parser = launcher._parser()
    subparsers = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction")
    assert set(subparsers.choices) == {"template", "plan", "execute", "_exec"}


def test_launcher_never_sets_or_repurposes_home():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"HOME"' not in source
    assert "$HOME" not in source
