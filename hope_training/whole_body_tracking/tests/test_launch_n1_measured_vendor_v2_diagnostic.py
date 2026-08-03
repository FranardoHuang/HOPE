"""CPU-only safety/argv tests for the isolated VendorV2 N1 launcher."""

from __future__ import annotations

import ast
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_n1_measured_vendor_v2_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("launch_measured_vendor_v2", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

MATERIALIZER_SCRIPT = SCRIPT.with_name("materialize_measured_action_ball_n1_bundle.py")
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_measured_vendor_v2_roundtrip", MATERIALIZER_SCRIPT
)
materializer = importlib.util.module_from_spec(MATERIALIZER_SPEC)
sys.modules[MATERIALIZER_SPEC.name] = materializer
MATERIALIZER_SPEC.loader.exec_module(materializer)


def _canonical_write(path: Path, value) -> str:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _materializations(tmp_path: Path):
    reward_payload = {"schema_version": 1, "terms": []}
    reward_sha = launcher.canonical_sha256(reward_payload)
    reward_path = tmp_path / "materialized" / "reward.json"
    reward_file_sha = _canonical_write(
        reward_path, {**reward_payload, "sha256": reward_sha}
    )
    policy_recipe = {"policy_initialization": {}}
    policy_sha = launcher.canonical_sha256(policy_recipe)
    policy_path = tmp_path / "materialized" / "policy.json"
    policy_file_sha = _canonical_write(
        policy_path,
        {
            "schema_version": 1,
            "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
            "action_count": 1,
            "action_order": [launcher.ACTION_ID],
            "policy_contract_sha256": policy_sha,
            "action_ball_ppo_runner_recipe": {
                "schema_version": 1,
                "sha256": policy_sha,
                "recipe": policy_recipe,
            },
            "policy_bootstrap": {},
        },
    )
    return {
        "reward": {"path": str(reward_path), "sha256": reward_file_sha},
        "reward_sha": reward_sha,
        "policy": {"path": str(policy_path), "sha256": policy_file_sha},
        "policy_sha": policy_sha,
    }


def _spec(
    tmp_path: Path, recipe: str = "current_lm", stage: str = "smoke"
):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    namespace_parent = tmp_path / launcher.EXPERIMENT_NAME
    namespace_parent.mkdir(parents=True, exist_ok=True)
    namespace = namespace_parent / ("n1_%s" % recipe)
    budget = launcher.BUDGETS[stage]
    materialized = _materializations(tmp_path)
    reward_pin = None if stage == "materialize" else materialized["reward"]
    policy_pin = (
        None if stage in ("materialize", "recipe") else materialized["policy"]
    )
    return {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "action_id": launcher.ACTION_ID,
        "bundle": {"path": "configs/bundle.json", "sha256": "b" * 64},
        "target_recipe": recipe,
        "target_validity_mask": list(launcher.RECIPES[recipe]),
        "reward_materialization": reward_pin,
        "policy_materialization": policy_pin,
        "policy_contract_sha256": (
            None if policy_pin is None else materialized["policy_sha"]
        ),
        "expected_effective_reward_recipe_sha256": (
            None if reward_pin is None else materialized["reward_sha"]
        ),
        "seed": 0,
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": 2,
            "uuid": "GPU-12345678",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _bundle():
    return {
        "motion": {"path": "assets/motion.npz", "sha256": "1" * 64},
        "immutable_tape": {"path": "configs/tape.npz", "sha256": "2" * 64},
        "core": {
            "manifest": {"path": "configs/manifest.json", "sha256": "3" * 64},
            "dynamic_ready": {
                "artifact": {"path": "configs/ready.json", "sha256": "4" * 64},
                "nominal_hold_receipt": {
                    "path": "configs/hold.json",
                    "sha256": "5" * 64,
                },
            },
        },
    }


def _real_dynamic_ready_pair():
    checkout = Path(__file__).resolve().parents[3]
    core_path = checkout / (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
        "take_061_unit04_bh.full.bundle.v2.ddeed84329be.json"
    )
    core = json.loads(core_path.read_text(encoding="utf-8"))
    dynamic = core["dynamic_ready"]
    artifact = json.loads(
        (checkout / dynamic["artifact"]["path"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (checkout / dynamic["nominal_hold_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return checkout, core, dynamic, artifact, receipt


def _reseal_dynamic_ready(candidate):
    candidate.pop("content_sha256", None)
    candidate["content_sha256"] = launcher._B._canonical_ascii_sha256(candidate)


def _reseal_nominal_hold(receipt):
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)


def _validate_in_memory_dynamic_pair(
    monkeypatch: pytest.MonkeyPatch, core, dynamic, candidate, receipt
):
    values = iter(
        (
            (dynamic["artifact"], candidate),
            (dynamic["nominal_hold_receipt"], receipt),
        )
    )
    monkeypatch.setattr(
        launcher._B,
        "_load_tracked_json",
        lambda *args, **kwargs: next(values),
    )
    monkeypatch.setattr(
        launcher,
        "_load_training_contract_module",
        lambda checkout: types.SimpleNamespace(
            load_action_ball_dynamic_ready_runtime_binding=lambda **kwargs: {
                "schema_version": 2,
                "kind": "action_ball_dynamic_ready_runtime_binding_v2",
                "action_order": [launcher.ACTION_ID],
                "motion_sha256_per_action": [core["motion"]["sha256"]],
            }
        ),
    )
    return launcher._validate_measured_dynamic_ready_v2(
        Path("/unused"),
        "a" * 40,
        dynamic,
        action_id=launcher.ACTION_ID,
        motion_sha256=core["motion"]["sha256"],
    )


def test_real_schema_v2_dynamic_ready_and_hold_pair_is_accepted():
    checkout, core, dynamic, _candidate, _receipt = _real_dynamic_ready_pair()
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = launcher._validate_measured_dynamic_ready_v2(
        checkout,
        commit,
        dynamic,
        action_id=launcher.ACTION_ID,
        motion_sha256=core["motion"]["sha256"],
    )
    assert result == dynamic


@pytest.mark.parametrize(
    "recipe, basename, validity",
    (
        (
            "current_lm",
            "take_061_unit04_bh.current_lm.measured_bundle.v1.a223d4c99f29.json",
            [True, True, True],
        ),
        (
            "analytic_no_velocity",
            "take_061_unit04_bh.analytic_no_velocity.measured_bundle.v1.d3c2632cbd67.json",
            [True, False, True],
        ),
        (
            "outcome_dense_only",
            "take_061_unit04_bh.outcome_dense_only.measured_bundle.v1.589db83947b7.json",
            [False, False, False],
        ),
    ),
)
def test_real_fresh_split_ready_bundles_cross_all_launch_gates(
    recipe, basename, validity
):
    checkout = Path(__file__).resolve().parents[3]
    relative = (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_final_seed0_20260803_take061_robust20n_r4_splitready/"
        + basename
    )
    path = checkout / relative
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = launcher._validate_bundle(
        checkout,
        commit,
        {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        action_id=launcher.ACTION_ID,
        recipe=recipe,
        seed=0,
    )
    assert result["target_validity"] == {
        "order": ["position", "velocity", "face"],
        "mask": validity,
    }
    assert result["runtime_contract"]["target_source"] == "immutable_tape"
    assert result["runtime_contract"]["reset_inverse_solve"] is False
    assert result["core"]["dynamic_ready"] == {
        "artifact": {
            "path": (
                "configs/action_ball_n1_measured_20260803/"
                "evidence_holdpass_robust20n_20260803/"
                "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
                "dynamic_ready.v2.json"
            ),
            "sha256": (
                "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"
            ),
        },
        "nominal_hold_receipt": {
            "path": (
                "configs/action_ball_n1_measured_20260803/"
                "evidence_holdpass_robust20n_20260803/"
                "take061.robust20n.nominal_hold.v1.json"
            ),
            "sha256": (
                "c8b92a28203cbf9b9a4f6dee784d6cc08f3f279672d8a9fc886aa6d92b5bb19b"
            ),
        },
    }


@pytest.mark.parametrize(
    "mutation, expected_error",
    (
        ("unknown_candidate_field", "keys differ"),
        ("legacy_schema", "schema-v2"),
        ("action", "schema-v2"),
        ("motion", "schema-v2"),
        ("receipt_cross_pin", "nominal-hold receipt"),
    ),
)
def test_schema_v2_dynamic_ready_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
):
    _checkout, core, dynamic, candidate, receipt = _real_dynamic_ready_pair()
    dynamic = copy.deepcopy(dynamic)
    candidate = copy.deepcopy(candidate)
    receipt = copy.deepcopy(receipt)
    if mutation == "unknown_candidate_field":
        candidate["unexpected"] = True
        _reseal_dynamic_ready(candidate)
    elif mutation == "legacy_schema":
        candidate["schema_version"] = 1
        candidate["kind"] = "agibot_a3_action_dynamic_ready_candidate_v1"
        _reseal_dynamic_ready(candidate)
    elif mutation == "action":
        candidate["action_id"] = "take_060_unit00_bh"
        _reseal_dynamic_ready(candidate)
    elif mutation == "motion":
        candidate["sources"]["stable_motion"]["sha256"] = "0" * 64
        _reseal_dynamic_ready(candidate)
    elif mutation == "receipt_cross_pin":
        receipt["artifact"]["content_sha256"] = "0" * 64
        _reseal_nominal_hold(receipt)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(launcher.LaunchRefused, match=expected_error):
        _validate_in_memory_dynamic_pair(
            monkeypatch, core, dynamic, candidate, receipt
        )


def test_spec_freezes_action_mask_budget_delay_wave_and_human_owner(tmp_path: Path):
    normalized = launcher._validate_spec(_spec(tmp_path, "teacher_pos_face_no_velocity"))
    assert normalized["action_id"] == launcher.ACTION_ID
    assert normalized["target_validity_mask"] == [True, False, True]
    assert (
        normalized["num_envs"],
        normalized["max_iterations"],
        normalized["save_interval"],
    ) == (1, 2, 1)

    wrong_mask = _spec(tmp_path, "analytic_no_velocity")
    wrong_mask["target_validity_mask"] = [True, True, True]
    with pytest.raises(launcher.LaunchRefused, match="validity"):
        launcher._validate_spec(wrong_mask)

    wrong_action = _spec(tmp_path, "outcome_dense_only")
    wrong_action["action_id"] = "take_060_unit00_bh"
    with pytest.raises(launcher.LaunchRefused, match="code-owned"):
        launcher._validate_spec(wrong_action)

    long_spec = _spec(tmp_path / "long", "teacher_pos_face_no_velocity", stage="long512")
    normalized_long = launcher._validate_spec(long_spec)
    assert (
        normalized_long["num_envs"],
        normalized_long["max_iterations"],
        normalized_long["save_interval"],
    ) == (512, 1000, 100)

    wrong_root = _spec(tmp_path / "wrong_root", "current_lm")
    namespace = Path(wrong_root["namespace"])
    bad_parent = namespace.parents[1] / "agibot_a3_action_ball_vendor_v1"
    bad_parent.mkdir(parents=True)
    wrong_root["namespace"] = str(bad_parent / namespace.name)
    wrong_root["log_path"] = str(Path(wrong_root["namespace"]) / "run.log")
    with pytest.raises(launcher.LaunchRefused, match="dedicated VendorV2"):
        launcher._validate_spec(wrong_root)


def test_cuda_launch_blocking_is_boolean_claim_owned_and_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    baseline = launcher._validate_spec(_spec(tmp_path, "current_lm"))
    assert baseline[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] is False
    assert launcher._cuda_launch_blocking_environment(baseline) == {}

    requested = _spec(tmp_path / "requested", "current_lm")
    requested[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] = True
    normalized = launcher._validate_spec(requested)
    assert normalized[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] is True
    assert launcher._cuda_launch_blocking_environment(normalized) == {
        "CUDA_LAUNCH_BLOCKING": "1"
    }
    assert launcher.canonical_sha256(normalized) != launcher.canonical_sha256(
        baseline
    )

    arbitrary = _spec(tmp_path / "arbitrary", "current_lm")
    arbitrary["environment"] = {"CUDA_LAUNCH_BLOCKING": "1"}
    with pytest.raises(launcher.LaunchRefused, match="keys differ"):
        launcher._validate_spec(arbitrary)


@pytest.mark.parametrize("value", (None, 0, 1, "1", [], {}))
def test_cuda_launch_blocking_rejects_non_boolean(tmp_path: Path, value):
    document = _spec(tmp_path, "current_lm")
    document[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] = value
    with pytest.raises(launcher.LaunchRefused, match="must be a boolean"):
        launcher._validate_spec(document)


def test_vendor_v2_colocation_is_explicit_claim_owned_and_default_empty(
    tmp_path: Path,
):
    default = launcher._validate_spec(_spec(tmp_path / "default"))
    assert default[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] is False
    assert default["gpu"]["require_empty"] is True

    opted = _spec(tmp_path / "opted")
    opted[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] = True
    opted["gpu"]["require_empty"] = False
    opted = launcher._validate_spec(opted)
    assert opted[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] is True
    assert opted["gpu"]["require_empty"] is False
    assert launcher.canonical_sha256(default) != launcher.canonical_sha256(opted)

    mismatched = _spec(tmp_path / "mismatched")
    mismatched[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] = True
    with pytest.raises(launcher.LaunchRefused, match="require_empty must be false"):
        launcher._validate_spec(mismatched)


@pytest.mark.parametrize("value", (None, 0, 1, "true", [], {}))
def test_vendor_v2_colocation_rejects_non_boolean(tmp_path: Path, value):
    document = _spec(tmp_path)
    document[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] = value
    with pytest.raises(launcher.LaunchRefused, match="must be a boolean"):
        launcher._validate_spec(document)


def _admission_spec(tmp_path: Path, *, allow: bool):
    document = _spec(tmp_path)
    if allow:
        document[launcher.VENDOR_V2_COLOCATION_SPEC_KEY] = True
        document["gpu"]["require_empty"] = False
    return launcher._validate_spec(document)


def _gpu_query(processes, *, total_memory_mib=48 * 1024, free_memory_mib=32 * 1024):
    return {
        "index": 2,
        "uuid": "GPU-12345678",
        "nvidia_smi_path": "/usr/bin/nvidia-smi",
        "nvidia_smi_sha256": "a" * 64,
        "total_memory_mib": total_memory_mib,
        "free_memory_mib": free_memory_mib,
        "processes": processes,
    }


def test_gpu_admission_default_empty_and_opt_in_max_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    empty = _admission_spec(tmp_path / "empty", allow=False)
    monkeypatch.setattr(launcher, "_query_gpu_processes", lambda *_: _gpu_query([]))
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])
    snapshot = launcher._verify_gpu_admission(
        empty, phase="pre_launch", current_namespace=None
    )
    assert snapshot["compute_process_count"] == 0
    assert snapshot["allow_vendor_v2_colocation"] is False

    process = {"pid": 123, "process_name": "python", "used_gpu_memory_mib": 4096}
    verified = {
        **process,
        "gpu_uuid": "GPU-12345678",
        "namespace": str(Path(empty["namespace"]).parent / "existing"),
        "namespace_receipt": {"path": "/receipt", "sha256": "b" * 64},
        "launch_claim_sha256": "c" * 64,
        "proc_starttime_ticks": 99,
    }
    monkeypatch.setattr(
        launcher, "_query_gpu_processes", lambda *_: _gpu_query([process])
    )
    monkeypatch.setattr(
        launcher, "_validate_runtime_gpu_process", lambda *args, **kwargs: verified
    )
    with pytest.raises(launcher.LaunchRefused, match="did not opt in"):
        launcher._verify_gpu_admission(
            empty, phase="pre_launch", current_namespace=None
        )

    opted = _admission_spec(tmp_path / "opted", allow=True)
    snapshot = launcher._verify_gpu_admission(
        opted, phase="pre_launch", current_namespace=None
    )
    assert snapshot["compute_process_count"] == 1
    assert snapshot["compute_processes"][0]["pid"] == 123
    assert snapshot["compute_processes"][0]["used_gpu_memory_mib"] == 4096
    assert snapshot["compute_processes"][0]["namespace_receipt"]["sha256"] == "b" * 64

    second = dict(process, pid=124, used_gpu_memory_mib=2048)
    monkeypatch.setattr(
        launcher,
        "_query_gpu_processes",
        lambda *_: _gpu_query([process, second]),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_gpu_process",
        lambda row, **kwargs: {
            **verified,
            "pid": row["pid"],
            "namespace": str(Path(opted["namespace"]).parent / ("n%d" % row["pid"])),
        },
    )
    with pytest.raises(launcher.LaunchRefused, match="no free compute-PID slot"):
        launcher._verify_gpu_admission(
            opted, phase="pre_launch", current_namespace=None
        )


def test_gpu_admission_unknown_co_resident_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = _admission_spec(tmp_path, allow=True)
    process = {"pid": 321, "process_name": "unknown", "used_gpu_memory_mib": 1}
    monkeypatch.setattr(
        launcher, "_query_gpu_processes", lambda *_: _gpu_query([process])
    )
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])

    def refuse(*args, **kwargs):
        raise launcher.LaunchRefused("unknown GPU co-resident pid=321")

    monkeypatch.setattr(launcher, "_validate_runtime_gpu_process", refuse)
    with pytest.raises(launcher.LaunchRefused, match="unknown GPU co-resident"):
        launcher._verify_gpu_admission(
            spec, phase="pre_launch", current_namespace=None
        )


def test_gpu_admission_rejects_47_of_48_gib_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = _admission_spec(tmp_path, allow=True)
    monkeypatch.setattr(
        launcher,
        "_query_gpu_processes",
        lambda *_: _gpu_query(
            [], total_memory_mib=48 * 1024, free_memory_mib=1024
        ),
    )
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])
    with pytest.raises(launcher.LaunchRefused, match="below conservative headroom"):
        launcher._verify_gpu_admission(
            spec, phase="pre_launch", current_namespace=None
        )


def test_pending_reservation_requires_bilateral_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = _admission_spec(tmp_path, allow=True)
    monkeypatch.setattr(launcher, "_query_gpu_processes", lambda *_: _gpu_query([]))
    monkeypatch.setattr(
        launcher,
        "_live_reservations",
        lambda *args, **kwargs: [
            {
                "owner_pid": 111,
                "owner_proc_starttime_ticks": 22,
                "namespace": str(Path(spec["namespace"]).parent / "pending"),
                "reservation_receipt": {"path": "/r", "sha256": "d" * 64},
                "allow_vendor_v2_colocation": False,
            }
        ],
    )
    with pytest.raises(launcher.LaunchRefused, match="reservation did not opt in"):
        launcher._verify_gpu_admission(
            spec, phase="pre_launch", current_namespace=None
        )


def _fake_proc_stat(pid: int, starttime: int) -> str:
    fields = ["S"] + ["0"] * 19
    fields[19] = str(starttime)
    return "%d (python worker) %s\n" % (pid, " ".join(fields))


@pytest.mark.parametrize("drift", ("commit_checkout", "gpu"))
def test_dead_historical_reservation_is_ignored_before_current_identity_checks(
    tmp_path: Path, drift: str
):
    checkout = tmp_path / "current_checkout"
    checkout.mkdir()
    root = tmp_path / launcher.EXPERIMENT_NAME
    namespace = root / "spent_old_run"
    namespace.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
        "owner_pid": 987654,
        "owner_proc_starttime_ticks": 123,
        "gpu_index": 2,
        "gpu_uuid": "GPU-12345678",
        "namespace": str(namespace),
        "checkout": str(checkout),
        "commit_sha": "a" * 40,
        "launch_claim_sha256": "b" * 64,
        "max_compute_pids": 2,
        # Historical schema deliberately predates the memory-headroom field.
        "allow_vendor_v2_colocation": True,
    }
    if drift == "commit_checkout":
        receipt["checkout"] = str(tmp_path / "old_checkout")
        receipt["commit_sha"] = "c" * 40
    else:
        receipt["gpu_index"] = 7
        receipt["gpu_uuid"] = "GPU-OLD00000"
    _canonical_write(namespace / launcher.GPU_RESERVATION_FILENAME, receipt)
    assert launcher._live_reservations(
        root,
        checkout=checkout,
        commit="a" * 40,
        gpu_index=2,
        gpu_uuid="GPU-12345678",
        proc_root=tmp_path / "empty_proc",
    ) == []


def test_live_reservation_on_another_gpu_does_not_block_current_gpu(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    root = tmp_path / launcher.EXPERIMENT_NAME
    namespace = root / "live_other_gpu"
    namespace.mkdir(parents=True)
    pid = 4567
    starttime = 987
    _canonical_write(
        namespace / launcher.GPU_RESERVATION_FILENAME,
        {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
            "owner_pid": pid,
            "owner_proc_starttime_ticks": starttime,
            "gpu_index": 7,
            "gpu_uuid": "GPU-OTHER1234",
            "namespace": str(namespace),
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "launch_claim_sha256": "b" * 64,
            "max_compute_pids": 2,
            "minimum_free_memory_mib": launcher.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": True,
        },
    )
    proc = tmp_path / "proc"
    pid_root = proc / str(pid)
    pid_root.mkdir(parents=True)
    (pid_root / "stat").write_text(
        _fake_proc_stat(pid, starttime), encoding="ascii"
    )
    assert launcher._live_reservations(
        root,
        checkout=checkout,
        commit="a" * 40,
        gpu_index=2,
        gpu_uuid="GPU-12345678",
        proc_root=proc,
    ) == []


def test_post_boot_failure_exactly_cleans_current_trainer_group_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    namespace = tmp_path / launcher.EXPERIMENT_NAME / "current_run"
    namespace.mkdir(parents=True)
    state = namespace / "run.log.launch"
    trainer_pgid = 4321
    existing_co_resident_pgid = 9876
    starttime = 654321
    leader_path = Path(str(state) + ".leader.json")
    _canonical_write(
        leader_path,
        {
            "schema_version": 1,
            "kind": "leader_identity",
            "leader": {
                "pid": trainer_pgid,
                "pgid": trainer_pgid,
                "starttime_ticks": starttime,
            },
        },
    )
    state.write_text(
        "pid=%d\npgid=%d\nleader_starttime_ticks=%d\n"
        "leader_identity_evidence=%s\nready_utc=2026-08-03T00:00:00Z\n"
        % (trainer_pgid, trainer_pgid, starttime, leader_path),
        encoding="utf-8",
    )

    class FakeExactGroup:
        def __init__(self):
            self.signals = []
            self.killed = False

        def term_group(self, proc_root, observed_leader_path, output):
            assert observed_leader_path == leader_path
            self.signals.append(("TERM", trainer_pgid))
            document = {
                "schema_version": 1,
                "kind": "pre_term_group_identity",
                "leader": {
                    "pid": trainer_pgid,
                    "pgid": trainer_pgid,
                    "starttime_ticks": starttime,
                },
                "members": [
                    {
                        "pid": trainer_pgid,
                        "pgid": trainer_pgid,
                        "starttime_ticks": starttime,
                    }
                ],
            }
            _canonical_write(output, document)
            return document

        def verify_residual(self, proc_root, term_path):
            return [] if self.killed else [types.SimpleNamespace(pid=trainer_pgid)]

        def kill_residual(self, proc_root, term_path, output):
            self.signals.append(("KILL", trainer_pgid))
            self.killed = True
            document = {
                "schema_version": 1,
                "kind": "pre_kill_group_identity",
                "leader": {
                    "pid": trainer_pgid,
                    "pgid": trainer_pgid,
                    "starttime_ticks": starttime,
                },
                "members": [],
            }
            _canonical_write(output, document)
            return document

    exact_group = FakeExactGroup()
    monkeypatch.setattr(launcher._ADMISSION, "_exact_group", exact_group)
    monkeypatch.setattr(launcher._ADMISSION, "_sleep", lambda _seconds: None)
    result = launcher._cleanup_post_boot_admission_failure(
        namespace,
        state,
        "a" * 64,
        "GPU free memory is below conservative headroom",
        proc_root=tmp_path / "proc",
    )
    assert result["cleanup"]["completed"] is True
    assert result["cleanup"]["term_member_pids"] == [trainer_pgid]
    assert exact_group.signals == [
        ("TERM", trainer_pgid),
        ("KILL", trainer_pgid),
    ]
    assert all(target != existing_co_resident_pgid for _signal, target in exact_group.signals)
    failure = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert failure["accepted"] is False
    assert failure["cleanup"]["completed"] is True
    assert failure["cleanup"]["residual_member_count"] == 0


@pytest.mark.parametrize(
    ("post_boot_error", "cleanup_expected"),
    (
        pytest.param(
            launcher.LaunchRefused("below conservative headroom"),
            True,
            id="launch_refused",
        ),
        pytest.param(
            OSError("nvidia-smi source read failed"), True, id="os_error"
        ),
        pytest.param(SystemExit(77), False, id="unexpected_base_exception"),
    ),
)
def test_launch_routes_post_boot_admission_refusal_through_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_boot_error: BaseException,
    cleanup_expected: bool,
):
    spec = _admission_spec(tmp_path / "launch", allow=True)
    namespace = Path(spec["namespace"])
    claim_sha = "a" * 64
    plan = {
        "launch_claim_sha256": claim_sha,
        "canonical_payload": {
            "spec": spec,
            "runtime_assets": {},
            "boot_marker": "Learning iteration",
        },
    }
    lock_file = tmp_path / "gpu.lock"
    lock_file.write_text("", encoding="ascii")
    cleanup_calls = []
    phases = []

    monkeypatch.setattr(launcher._B, "_verify_clean_source", lambda *args: None)
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_claim", lambda *args: None
    )

    def claim_namespace(_plan):
        namespace.mkdir()
        return namespace

    monkeypatch.setattr(launcher._B, "_claim_namespace", claim_namespace)
    monkeypatch.setattr(
        launcher,
        "_open_gpu_shared_lock",
        lambda _path: os.open(lock_file, os.O_RDWR),
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda _fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda _fd: None)
    monkeypatch.setattr(
        launcher,
        "_reservation_document",
        lambda _spec, _sha: {"schema_version": 1, "kind": "test_reservation"},
    )

    def verify(_spec, *, phase, **kwargs):
        phases.append(phase)
        if phase == "post_boot":
            raise post_boot_error
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", verify)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
    )

    def cleanup(current_namespace, state_path, observed_sha, error):
        cleanup_calls.append(
            (current_namespace, state_path, observed_sha, error)
        )
        return {
            "path": str(current_namespace / "post_boot_admission_failure.json"),
            "cleanup": {"completed": True},
        }

    monkeypatch.setattr(
        launcher, "_cleanup_post_boot_admission_failure", cleanup
    )
    if cleanup_expected:
        with pytest.raises(
            launcher.LaunchRefused, match="exact current-trainer cleanup completed"
        ):
            launcher.launch(plan, confirm_claim=claim_sha)
    else:
        with pytest.raises(SystemExit, match="77"):
            launcher.launch(plan, confirm_claim=claim_sha)
    assert phases == ["pre_launch", "post_boot"]
    if cleanup_expected:
        assert cleanup_calls == [
            (
                namespace,
                Path(spec["log_path"] + ".launch"),
                claim_sha,
                str(post_boot_error),
            )
        ]
    else:
        assert cleanup_calls == []


def _full_claim_payload(spec, bundle):
    checkout = Path(spec["source"]["checkout"])
    launcher_path = checkout / launcher.LAUNCHER_SOURCE
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text("# exact fake VendorV2 launcher\n", encoding="utf-8")
    launcher_sha = hashlib.sha256(launcher_path.read_bytes()).hexdigest()
    admission_path = checkout / launcher.ADMISSION_SOURCE
    admission_path.write_text("# exact fake VendorV2 admission\n", encoding="utf-8")
    admission_sha = hashlib.sha256(admission_path.read_bytes()).hexdigest()
    exact_group_path = checkout / launcher.EXACT_GROUP_SOURCE
    exact_group_path.write_text("# exact fake process-group helper\n", encoding="utf-8")
    exact_group_sha = hashlib.sha256(exact_group_path.read_bytes()).hexdigest()
    runtime_sources = {
        name: {
            "path": path,
            "sha256": (
                launcher_sha
                if name == "VendorV2 N1 launcher"
                else admission_sha
                if name == "VendorV2 GPU admission"
                else exact_group_sha
                if name == "exact process-group helper"
                else "e" * 64
            ),
        }
        for path, name in launcher.RUNTIME_SOURCE_PATHS
    }
    output = launcher._output_contract(spec)
    return {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "formal_evidence_prohibited": True,
        "promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "deployment_prohibited": True,
        "hardware_prohibited": True,
        "single_gpu": True,
        "max_compute_pids_on_physical_gpu": 2,
        "minimum_free_memory_mib": launcher.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "gpu_default_empty": False,
        "vendor_v2_colocation_opt_in": True,
        "fresh_only": True,
        "reward_materialization_only": False,
        "policy_recipe_materialization_only": False,
        "ppo_updates_authorized": output["ppo_update_count"],
        "control_step_action_delay": 0,
        "reset_inverse_solve": False,
        "physical_ball_semantics": launcher.PHYSICAL_BALL_SEMANTICS,
        "spec_file_sha256": "f" * 64,
        "spec": spec,
        "source": {"checkout": str(checkout), "commit_sha": "a" * 40, "clean": True},
        "runtime_sources": runtime_sources,
        "runtime_assets": {},
        "bundle": bundle,
        "materialization_inputs": {},
        "output_contract": output,
        "boot_marker": output["boot_marker"],
        "training_argv": launcher._training_argv(spec, bundle),
    }


def test_runtime_pid_crossbinds_proc_claim_checkout_memory_and_namespace_receipt(
    tmp_path: Path,
):
    spec = _admission_spec(tmp_path / "runtime", allow=True)
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / launcher._B.WBT_RELATIVE
    wbt.mkdir(parents=True)
    namespace = Path(spec["namespace"])
    namespace.mkdir()
    bundle = _bundle()
    payload = _full_claim_payload(spec, bundle)
    claim_sha = launcher.canonical_sha256(payload)
    _canonical_write(
        namespace / "launch_claim.json",
        {
            "schema_version": launcher.SCHEMA_VERSION,
            "kind": launcher.CLAIM_KIND,
            "launch_claim_sha256": claim_sha,
            "canonical_payload": payload,
        },
    )
    pid = 4321
    starttime = 98765
    receipt_path = namespace / launcher.GPU_NAMESPACE_RECEIPT_FILENAME
    receipt_sha = _canonical_write(
        receipt_path,
        {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_namespace_receipt_v1",
            "pid": pid,
            "proc_starttime_ticks": starttime,
            "gpu_index": 2,
            "gpu_uuid": "GPU-12345678",
            "namespace": str(namespace),
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "wbt_cwd": str(wbt),
            "launch_claim_sha256": claim_sha,
            "max_compute_pids": 2,
            "minimum_free_memory_mib": launcher.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": True,
        },
    )
    proc = tmp_path / "proc"
    pid_root = proc / str(pid)
    pid_root.mkdir(parents=True)
    (pid_root / "stat").write_text(_fake_proc_stat(pid, starttime), encoding="ascii")
    (pid_root / "environ").write_bytes(
        (
            "%s=%s\0%s=%s\0HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256=%s\0"
            % (
                launcher.GPU_NAMESPACE_RECEIPT_ENV,
                receipt_path,
                launcher.GPU_NAMESPACE_RECEIPT_SHA_ENV,
                receipt_sha,
                claim_sha,
            )
        ).encode()
    )
    (pid_root / "cwd").symlink_to(wbt, target_is_directory=True)
    (pid_root / "exe").symlink_to(Path(spec["source"]["isaac_python"]))
    (pid_root / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in payload["training_argv"]) + b"\0"
    )
    result = launcher._validate_runtime_gpu_process(
        {"pid": pid, "process_name": "python", "used_gpu_memory_mib": 6144},
        checkout=checkout,
        commit="a" * 40,
        gpu_index=2,
        gpu_uuid="GPU-12345678",
        current_namespace=None,
        proc_root=proc,
    )
    assert result["pid"] == pid
    assert result["used_gpu_memory_mib"] == 6144
    assert result["namespace"] == str(namespace)
    assert result["namespace_receipt"] == {
        "path": str(receipt_path),
        "sha256": receipt_sha,
    }

    (pid_root / "cmdline").write_bytes(b"python\0-c\0print(1)\0")
    with pytest.raises(launcher.LaunchRefused, match="cmdline differs"):
        launcher._validate_runtime_gpu_process(
            {"pid": pid, "process_name": "python", "used_gpu_memory_mib": 6144},
            checkout=checkout,
            commit="a" * 40,
            gpu_index=2,
            gpu_uuid="GPU-12345678",
            current_namespace=None,
            proc_root=proc,
        )
    (pid_root / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in payload["training_argv"]) + b"\0"
    )
    other_python = tmp_path / "other-python"
    other_python.write_text("#!/bin/sh\n", encoding="utf-8")
    other_python.chmod(0o755)
    (pid_root / "exe").unlink()
    (pid_root / "exe").symlink_to(other_python)
    with pytest.raises(launcher.LaunchRefused, match="executable drifted"):
        launcher._validate_runtime_gpu_process(
            {"pid": pid, "process_name": "python", "used_gpu_memory_mib": 6144},
            checkout=checkout,
            commit="a" * 40,
            gpu_index=2,
            gpu_uuid="GPU-12345678",
            current_namespace=None,
            proc_root=proc,
        )
    (pid_root / "exe").unlink()
    (pid_root / "exe").symlink_to(Path(spec["source"]["isaac_python"]))
    (checkout / launcher.LAUNCHER_SOURCE).write_text(
        "# launcher bytes drifted after claim\n", encoding="utf-8"
    )
    with pytest.raises(launcher.LaunchRefused, match="launcher bytes differ"):
        launcher._validate_runtime_gpu_process(
            {"pid": pid, "process_name": "python", "used_gpu_memory_mib": 6144},
            checkout=checkout,
            commit="a" * 40,
            gpu_index=2,
            gpu_uuid="GPU-12345678",
            current_namespace=None,
            proc_root=proc,
        )

    minimal = {"spec": spec}
    minimal_sha = launcher.canonical_sha256(minimal)
    _canonical_write(
        namespace / "launch_claim.json",
        {
            "schema_version": launcher.SCHEMA_VERSION,
            "kind": launcher.CLAIM_KIND,
            "launch_claim_sha256": minimal_sha,
            "canonical_payload": minimal,
        },
    )
    with pytest.raises(launcher.LaunchRefused, match="payload keys differ"):
        launcher._validate_namespace_claim(
            namespace,
            minimal_sha,
            checkout=checkout,
            commit="a" * 40,
            gpu_index=2,
            gpu_uuid="GPU-12345678",
            require_colocation_opt_in=True,
        )


def test_materialize_then_recipe_then_training_identity_chain_is_fail_closed(
    tmp_path: Path,
):
    materialize = launcher._validate_spec(
        _spec(tmp_path / "materialize", stage="materialize")
    )
    assert materialize["policy_contract_sha256"] is None
    assert materialize["expected_effective_reward_recipe_sha256"] is None
    assert (materialize["num_envs"], materialize["max_iterations"]) == (1, 0)

    recipe = launcher._validate_spec(
        _spec(tmp_path / "recipe", stage="recipe")
    )
    assert recipe["reward_materialization"] is not None
    assert recipe["policy_contract_sha256"] is None
    assert (recipe["num_envs"], recipe["max_iterations"]) == (1, 0)

    smoke = _spec(tmp_path / "smoke", stage="smoke")
    smoke["policy_contract_sha256"] = "f" * 64
    with pytest.raises(launcher.LaunchRefused, match="materialized recipe"):
        launcher._validate_spec(smoke)

    bad_materialize = _spec(tmp_path / "bad", stage="materialize")
    bad_materialize["expected_effective_reward_recipe_sha256"] = "a" * 64
    with pytest.raises(launcher.LaunchRefused, match="must not predeclare"):
        launcher._validate_spec(bad_materialize)

    wrong_identity_arm = _spec(
        tmp_path / "wrong_arm",
        recipe="analytic_full",
        stage="materialize",
    )
    with pytest.raises(launcher.LaunchRefused, match="current_lm identity"):
        launcher._validate_spec(wrong_identity_arm)

    wrong_seed = _spec(tmp_path / "wrong_seed", stage="smoke")
    wrong_seed["seed"] = 1
    with pytest.raises(launcher.LaunchRefused, match="seed 0"):
        launcher._validate_spec(wrong_seed)


def test_finalize_bundle_schema_roundtrips_into_launcher_successor_contract():
    assert set(launcher.BUNDLE_KEYS) == set(materializer.FINAL_BUNDLE_KEYS)
    assert "immutable_tape_build_report" in launcher.BUNDLE_KEYS
    assert "immutable_tape_receipt" not in launcher.BUNDLE_KEYS


def test_training_argv_is_fresh_delay0_fixed_tape_virtual_ball_and_same_abi(tmp_path: Path):
    spec = launcher._validate_spec(_spec(tmp_path, "outcome_dense_only"))
    bundle = _bundle()
    argv = launcher._training_argv(spec, bundle)
    joined = "\n".join(argv)
    assert "task=%s" % launcher.TASK_PROFILE_ID in argv
    assert "task.actor_obs_contract=%s" % launcher.ACTOR_CONTRACT in argv
    assert "task.racket.action_ball_target_source=immutable_tape" in argv
    assert "task.racket.action_ball_target_recipe=outcome_dense_only" in argv
    assert "task.racket.action_ball_target_validity_mask=[false,false,false]" in argv
    assert "task.actions.control_step_action_delay_min=0" in argv
    assert "task.actions.control_step_action_delay_max=0" in argv
    assert (
        "task.motion.action_ball_diagnostic_split_ready_teacher=true"
        in argv
    )
    assert "task.push.enable=false" in argv
    assert {
        value[len("~task.push.") :]
        for value in argv
        if value.startswith("~task.push.")
    } == set(launcher.DISABLED_PUSH_DORMANT_FIELDS)
    assert "task.physical_ball=false" in argv
    assert not any(
        value.lstrip("+").startswith("task.racket.physical_ball")
        for value in argv
    )
    assert "task.racket.adaptive_sigma=false" in argv
    assert argv.count(launcher.POLICY_NOISE_STD_OVERRIDE) == 1
    assert "algo.policy.noise_std_type=log" in argv
    assert "+algo.policy.noise_std_type=log" not in argv
    assert not any("checkpoint" in value or "resume" in value for value in argv)
    assert "DIAGNOSTIC_UNAUTHORIZED" in joined


def test_zero_ppo_reward_then_policy_recipe_argv_are_distinct(tmp_path: Path):
    reward_spec = launcher._validate_spec(
        _spec(tmp_path / "reward", stage="materialize")
    )
    reward_argv = launcher._training_argv(reward_spec, _bundle())
    assert "num_envs=1" in reward_argv
    assert "max_iterations=0" in reward_argv
    assert (
        "+n1_vendor_sigma_profile=" + launcher.REWARD_MATERIALIZATION_PROFILE
    ) in reward_argv
    assert any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in reward_argv
    )
    assert not any(
        value.startswith("expected_effective_reward_recipe_sha256=")
        or value.startswith("action_ball_policy_recipe_output_path=")
        for value in reward_argv
    )
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in reward_argv

    policy_spec = launcher._validate_spec(
        _spec(tmp_path / "policy", stage="recipe")
    )
    policy_argv = launcher._training_argv(policy_spec, _bundle())
    assert "max_iterations=0" in policy_argv
    assert any(
        value.startswith("expected_effective_reward_recipe_sha256=")
        for value in policy_argv
    )
    assert any(
        value.startswith("action_ball_policy_recipe_output_path=")
        for value in policy_argv
    )
    assert not any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in policy_argv
    )
    assert launcher._output_contract(reward_spec)["ppo_update_count"] == 0
    assert launcher._output_contract(policy_spec)["ppo_update_count"] == 0


def test_additive_hydra_overrides_only_name_absent_root_keys(tmp_path: Path):
    materialize = launcher._validate_spec(
        _spec(tmp_path / "materialize", stage="materialize")
    )
    recipe = launcher._validate_spec(_spec(tmp_path / "recipe", stage="recipe"))
    smoke = launcher._validate_spec(_spec(tmp_path / "smoke", stage="smoke"))
    expected_by_stage = {
        "materialize": {
        "+n1_vendor_sigma_profile",
        "+action_ball_effective_reward_recipe_output_path",
        },
        "recipe": set(),
        "smoke": set(),
    }
    for spec in (materialize, recipe, smoke):
        additions = {
            value.split("=", 1)[0]
            for value in launcher._training_argv(spec, _bundle())
            if value.startswith("+")
        }
        assert additions == expected_by_stage[spec["stage"]]
    train_yaml = (
        SCRIPT.parents[1] / "cfg" / "train.yaml"
    ).read_text(encoding="utf-8")
    assert "\nn1_vendor_sigma_profile:" not in train_yaml
    assert "\naction_ball_effective_reward_recipe_output_path:" not in train_yaml
    task_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("task", "base")
        for path in sorted((SCRIPT.parents[1] / "cfg" / folder).glob("*.yaml"))
    )
    assert "\n  physical_ball:" not in task_sources
    assert "\n  physical_ball_impulse:" not in task_sources


def test_every_training_override_matches_composed_config_ownership(tmp_path: Path):
    cfg_root = SCRIPT.parents[1] / "cfg"

    def owned_paths(path: Path, prefix: str, seen=None):
        if seen is None:
            seen = set()
        identity = (path.resolve(), prefix)
        if identity in seen:
            return set()
        seen.add(identity)
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result = set()

        def visit(value, parent):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                if key == "defaults":
                    continue
                owned = "%s.%s" % (parent, key) if parent else str(key)
                result.add(owned)
                visit(child, owned)

        visit(document, prefix)
        for default in document.get("defaults", []):
            if not isinstance(default, str) or default == "_self_":
                continue
            reference = default.split("@", 1)[0]
            inherited = (
                cfg_root / (reference[1:] + ".yaml")
                if reference.startswith("/")
                else path.parent / (reference + ".yaml")
            )
            if inherited.exists():
                result.update(owned_paths(inherited, prefix, seen))
        return result

    # Group selectors choose these exact two leaves; train.yaml's default task/algo
    # entries are intentionally not part of the selected composition.
    train_document = yaml.safe_load(
        (cfg_root / "train.yaml").read_text(encoding="utf-8")
    )
    train_document.pop("defaults")
    root_copy = tmp_path / "train_without_defaults.yaml"
    root_copy.write_text(yaml.safe_dump(train_document), encoding="utf-8")
    ownership = owned_paths(root_copy, "")
    ownership.update(
        owned_paths(
            cfg_root / "task" / (launcher.TASK_PROFILE_ID + ".yaml"), "task"
        )
    )
    ownership.update(owned_paths(cfg_root / "algo" / "ppo.yaml", "algo"))

    for stage in ("materialize", "recipe", "smoke"):
        spec = launcher._validate_spec(_spec(tmp_path / stage, stage=stage))
        argv = launcher._training_argv(spec, _bundle())
        assert argv[2:4] == ["task=%s" % launcher.TASK_PROFILE_ID, "algo=ppo"]
        for override in argv[4:]:
            additive = override.startswith("+")
            deletion = override.startswith("~")
            key = override.lstrip("+~").split("=", 1)[0]
            if deletion:
                assert "=" not in override
            assert (key not in ownership) if additive else (key in ownership), override


def test_no_push_argv_deletes_every_inherited_dormant_field(tmp_path: Path):
    cfg_root = SCRIPT.parents[1] / "cfg"

    def merge(base, overlay):
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def compose_task(path: Path):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result = {}
        for default in document.get("defaults", []):
            if not isinstance(default, str) or default == "_self_":
                continue
            reference = default.split("@", 1)[0]
            inherited = (
                cfg_root / (reference[1:] + ".yaml")
                if reference.startswith("/")
                else path.parent / (reference + ".yaml")
            )
            if inherited.exists():
                result = merge(result, compose_task(inherited))
        return merge(result, {k: v for k, v in document.items() if k != "defaults"})

    task = compose_task(
        cfg_root / "task" / (launcher.TASK_PROFILE_ID + ".yaml")
    )
    assert task["push"]["enable"] is False
    assert set(task["push"]) == {
        "enable",
        *launcher.DISABLED_PUSH_DORMANT_FIELDS,
    }
    spec = launcher._validate_spec(_spec(tmp_path, stage="materialize"))
    argv = launcher._training_argv(spec, _bundle())
    for field in launcher.DISABLED_PUSH_DORMANT_FIELDS:
        assert "~task.push.%s" % field in argv
        task["push"].pop(field)
    # This is the exact clean disable shape accepted by
    # train._apply_push_robot_task_override: no loaded value except enable=false.
    assert task["push"] == {"enable": False}


def test_physical_ball_disable_uses_only_consumed_top_level_switch(tmp_path: Path):
    train_path = SCRIPT.with_name("train.py")
    train_tree = ast.parse(train_path.read_text(encoding="utf-8"))
    racket_keys = ast.literal_eval(
        next(
            node.value
            for node in train_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_RACKET_KEYS"
                for target in node.targets
            )
        )
    )
    assert "physical_ball" not in racket_keys
    assert "physical_ball_impulse" not in racket_keys

    task_yaml = yaml.safe_load(
        (
            SCRIPT.parents[1]
            / "cfg"
            / "task"
            / (launcher.TASK_PROFILE_ID + ".yaml")
        ).read_text(encoding="utf-8")
    )
    assert task_yaml["physical_ball"] is False
    for stage in ("materialize", "recipe", "smoke"):
        spec = launcher._validate_spec(_spec(tmp_path / stage, stage=stage))
        argv = launcher._training_argv(spec, _bundle())
        assert argv.count("task.physical_ball=false") == 1
        assert not any(
            value.lstrip("+").startswith("task.racket.physical_ball")
            for value in argv
        )

    command_source = (
        SCRIPT.parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_commands.py"
    ).read_text(encoding="utf-8")
    env_source = (
        SCRIPT.parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "hope_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "physical_ball: bool = False" in command_source
    assert "\n    physical_ball_impulse:" not in command_source
    assert 'getattr(cfg, "physical_ball_impulse", False)' in command_source
    assert "physical_ball: bool = False" in env_source


def test_policy_materialization_binds_dynamic_ready_and_log_std(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding_sha = "a" * 64
    bootstrap = {
        "schema_version": 3,
        "action_count": 1,
        "action_order": [launcher.ACTION_ID],
        "ready_source": {"identity": {"binding_sha256": binding_sha}},
        "initialization": {
            "noise_std_type": "log",
            "init_noise_std": 0.02,
            "required_realized_init_noise_std": 0.02,
        },
    }
    runner = {"policy_initialization": bootstrap}
    policy_sha = launcher.canonical_sha256(runner)
    path = tmp_path / "policy.json"
    file_sha = _canonical_write(
        path,
        {
            "schema_version": 1,
            "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
            "action_count": 1,
            "action_order": [launcher.ACTION_ID],
            "policy_contract_sha256": policy_sha,
            "action_ball_ppo_runner_recipe": {
                "schema_version": 1,
                "sha256": policy_sha,
                "recipe": runner,
            },
            "policy_bootstrap": bootstrap,
        },
    )
    contract = types.SimpleNamespace(
        load_action_ball_dynamic_ready_runtime_binding=lambda **kwargs: {
            "binding_sha256": binding_sha
        },
        validate_action_ball_policy_bootstrap=lambda *args, **kwargs: None,
        action_ball_policy_bootstrap_scientific_identity=(
            lambda value, repo_root: value
        ),
    )
    monkeypatch.setattr(
        launcher, "_load_training_contract_module", lambda checkout: contract
    )
    result = launcher._validate_policy_materialization(
        {"path": str(path), "sha256": file_sha},
        checkout=tmp_path,
        bundle=_bundle(),
    )
    assert result["policy_contract_sha256"] == policy_sha
    assert result["dynamic_ready_binding_sha256"] == binding_sha
    assert result["noise_std_type"] == "log"

    contract.load_action_ball_dynamic_ready_runtime_binding = lambda **kwargs: {
        "binding_sha256": "b" * 64
    }
    with pytest.raises(launcher.LaunchRefused, match="exact log-std"):
        launcher._validate_policy_materialization(
            {"path": str(path), "sha256": file_sha},
            checkout=tmp_path,
            bundle=_bundle(),
        )


def test_every_arm_uses_same_actor_contract_and_only_mask_recipe_change(tmp_path: Path):
    contracts = set()
    for recipe in launcher.RECIPES:
        spec = launcher._validate_spec(_spec(tmp_path / recipe, recipe))
        bundle = _bundle()
        argv = launcher._training_argv(spec, bundle)
        contracts.add(next(value for value in argv if value.startswith("task.actor_obs_contract=")))
    assert contracts == {"task.actor_obs_contract=%s" % launcher.ACTOR_CONTRACT}


def test_template_and_training_argv_preserve_venv_symlink_entry(tmp_path: Path):
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real_python.chmod(0o755)
    venv_bin = tmp_path / "hope_isaac_venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(real_python)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    namespace_parent = tmp_path / launcher.EXPERIMENT_NAME
    namespace_parent.mkdir()
    output = tmp_path / "materialize.json"

    launcher._write_template(
        types.SimpleNamespace(
            stage="materialize",
            namespace=str(namespace_parent / "fresh_materialize_r1"),
            reward_materialization_path=None,
            reward_materialization_sha256=None,
            policy_materialization_path=None,
            policy_materialization_sha256=None,
            checkout=str(checkout),
            commit_sha="a" * 40,
            isaac_python=str(venv_python),
            action_id=launcher.ACTION_ID,
            bundle_path="configs/bundle.json",
            bundle_sha256="b" * 64,
            target_recipe="current_lm",
            seed=0,
            gpu_index=0,
            gpu_uuid="GPU-12345678",
            owner="Franco",
            output=str(output),
        )
    )

    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["source"]["isaac_python"] == str(venv_python)
    assert raw["source"]["isaac_python"] != str(venv_python.resolve())
    spec = launcher._validate_spec(raw)
    assert spec["source"]["isaac_python"] == str(venv_python)
    assert launcher._training_argv(spec, _bundle())[0] == str(venv_python)
