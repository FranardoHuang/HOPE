from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_balance_action_slew_queue.py"
QUEUE = ROOT / "configs" / "phase1_balance_action_slew_20260720.yaml"
PPO_CONFIG = (
    ROOT / "hope_training" / "whole_body_tracking" / "cfg" / "algo" / "ppo.yaml"
)


def _module():
    spec = importlib.util.spec_from_file_location("balance_slew_queue_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _module()
REAL_VALIDATE_ORIGIN_MAIN_AUTHORITY = Q._validate_origin_main_launch_authority
FAKE_ORIGIN_MAIN_AUTHORITY = {
    "origin_main_commit": "b" * 40,
    "now_entry_sha256": "c" * 64,
    "human_owner": "franco",
    "executor": "Codex",
    "branch": Q.AUTHORITY_BRANCH,
}


@pytest.fixture(autouse=True)
def _mock_origin_main_authority(monkeypatch):
    monkeypatch.setattr(
        Q,
        "_validate_origin_main_launch_authority",
        lambda _queue, _manifest: dict(FAKE_ORIGIN_MAIN_AUTHORITY),
    )


def _raw() -> dict:
    value = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _values(argv: list[str]) -> dict[str, str]:
    return Q._override_map(argv[2:], "test argv")


def _manifest(tmp_path: Path, queue) -> tuple[Path, str, object]:
    sha = lambda value: hashlib.sha256(value).hexdigest()
    content = {
        "schema_version": 1,
        "queue_id": queue["queue_id"],
        "queue_files": {
            "config": {"path": Q.QUEUE_CONFIG_RELATIVE, "sha256": sha(QUEUE.read_bytes())},
            "runner": {"path": Q.QUEUE_RUNNER_RELATIVE, "sha256": sha(SCRIPT.read_bytes())},
        },
        "source": {
            "checkout": queue["source"]["checkout"],
            "commit": Q.EXPECTED_REMOTE_SOURCE_COMMIT,
            "required_file_sha256": {
                relative: "1" * 64 for relative in Q._manifest_source_relative_paths(queue)
            },
        },
        "assets": {
            "a3_runtime_asset_root": {
                "path": queue["assets"]["a3_runtime_asset_root"],
                "tree_sha256": "2" * 64,
                "file_count": 10,
                "total_file_bytes": 1000,
                "symlinks_forbidden": True,
            },
            "preconverted_a3_usd": {
                "path": queue["assets"]["preconverted_a3_usd"],
                "sha256": "3" * 64,
                "bundle_root": str(PurePosixPath(queue["assets"]["preconverted_a3_usd"]).parent),
                "bundle_tree_sha256": "b" * 64,
                "file_count": 6,
                "total_file_bytes": 6000,
                "symlinks_forbidden": True,
            },
            **{
                name: {"path": queue["assets"][name], "sha256": digit * 64}
                for name, digit in (
                    ("motion_forehand", "4"),
                    ("motion_backhand", "5"),
                    ("training_question_bank", "6"),
                )
            },
        },
        "parents": {
            name: {
                "checkpoint_path": parent["checkpoint_path"],
                "checkpoint_sha256": ("7" if name == "W" else "8") * 64,
                "hard_contract_path": parent["hard_contract_path"],
                "hard_contract_sha256": ("9" if name == "W" else "a") * 64,
            }
            for name, parent in queue["parents"].items()
        },
    }
    envelope = {
        "schema_version": 1,
        "content": content,
        "content_sha256": Q._canonical_sha256(content),
    }
    path = tmp_path / "launch_manifest.json"
    path.write_bytes(Q._json_document(envelope))
    file_sha = sha(path.read_bytes())
    loaded = Q._load_launch_manifest(queue, path, file_sha)
    return path, file_sha, loaded


def _activation(mechanism: str) -> dict:
    rows = []
    for step in Q.PROBE_STEPS:
        eligible = 200
        row = {
            "step": step,
            "observed_sample_count": Q.EXPECTED_SAMPLES_PER_UPDATE,
            "previous_qdes_valid_sample_count": Q.EXPECTED_SAMPLES_PER_UPDATE - 2048,
            "previous_qdes_invalid_first_step_sample_count": 2048,
            "recovery_eligible_sample_count": eligible,
            "reward_enabled_eligible_sample_count": eligible if mechanism == "H" else 0,
            "tail_active_sample_count": 20,
            "above_margin_joint_count": 40,
            "gated_tail_value_sum": 1.25,
            "racket_swing_outcome_count": 100,
            "racket_swing_completion_count": 80,
            "racket_physical_fall_count": 10,
            "racket_pre_strike_physical_fall_count": 4,
            "racket_post_strike_physical_fall_count": 6,
            "racket_strike_opportunity_count": 80,
            "racket_virtual_legal_return_count": 40,
            "racket_ready_tilt_eligible_sample_count": 200,
            "racket_ready_tilt_rad_sum": 20.0,
            "racket_ready_nonfinite_value_count": 0,
            "qdot_observed_sample_count": Q.EXPECTED_SAMPLES_PER_UPDATE,
            "qdot_excess_sample_count": 100,
            "qdot_normalized_excess_square_sum": 50.0,
        }
        rows.append(row)
    return {
        "expected_steps": list(Q.PROBE_STEPS),
        "expected_samples_per_update": Q.EXPECTED_SAMPLES_PER_UPDATE,
        "rows": rows,
        "totals": {
            counter: sum(float(row[counter]) for row in rows)
            for counter in Q.PROBE_COUNTERS
        },
    }


def _receipt_content(queue, job, manifest) -> dict:
    claim, _ = Q._build_claim(queue, job, "probe", manifest)
    return {
        "schema_version": 1,
        "queue_id": queue["queue_id"],
        "job_id": job["id"],
        "parent": job["parent"],
        "mechanism": job["mechanism"],
        "pod": job["pod"],
        "gpu": job["gpu"],
        "status": "passed",
        "launch_manifest": {
            "file_sha256": manifest.file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "probe_claim_content_sha256": claim["content_sha256"],
        "probe_verifier_program_sha256": Q.PROBE_VERIFIER_PROGRAM_SHA256,
        "artifacts": {
            "queue_claim_file_sha256": hashlib.sha256(Q._json_document(claim)).hexdigest(),
            "run_binding_file_sha256": "c" * 64,
            "run_binding_content_sha256": "d" * 64,
            "terminal_status_file_sha256": "e" * 64,
            "terminal_status_content_sha256": "f" * 64,
            "terminal_checkpoint_path": f"/workspace/rsl/{job['id']}/model_6701.pt",
            "terminal_checkpoint_sha256": "1" * 64,
            "terminal_checkpoint_iteration": 6701,
            "hard_contract_path": f"/workspace/rsl/{job['id']}/params/training_contract.json",
            "hard_contract_sha256": "2" * 64,
        },
        "checkpoint_state_audit": {
            name: {"tensor_count": 2, "floating_elements": 20, "nonfinite_elements": 0}
            for name in (
                "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
                "privileged_obs_norm_state_dict",
            )
        },
        "activation": _activation(job["mechanism"]),
        "runtime": dict(Q.EXPECTED_PROBE_RUNTIME),
    }


def _receipt_dir(tmp_path: Path, queue, manifest) -> Path:
    root = tmp_path / "receipts"
    for job in queue["jobs"]:
        content = _receipt_content(queue, job, manifest)
        envelope = {
            "schema_version": 1,
            "content": content,
            "content_sha256": Q._canonical_sha256(content),
        }
        path = root / job["id"] / "probe_receipt.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(Q._json_document(envelope))
    return root


def test_checked_in_queue_is_valid_six_cell_no_launch_plan():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    assert plan["commands_emitted"] is False
    assert plan["launch_manifest_gate"]["status"] == "blocked_manifest_not_supplied_to_this_invocation"
    assert len(plan["jobs"]) == 6
    assert "ssh_argv" not in json.dumps(plan)


def test_exact_remote_source_commit_and_simulation_only_contract():
    queue = Q.load_queue(QUEUE)
    assert queue["source"]["identity_mode"] == "clean_detached_exact_commit"
    assert queue["source"]["commit"] == "54c9a62656f0e60e5bb41cbcfa0e5a972b793906"
    assert queue["simulation_only"] is True
    assert queue["real_robot_authorized"] is False
    assert queue["formal_exact_eligible"] is False


def test_matrix_is_wv_by_cnh_on_six_unique_gpus():
    queue = Q.load_queue(QUEUE)
    observed = {
        (job["parent"], job["mechanism"]): (job["pod"], job["gpu"])
        for job in queue["jobs"]
    }
    assert observed == {
        ("W", "C"): ("pod1", 0), ("W", "N"): ("pod1", 1), ("W", "H"): ("pod1", 2),
        ("V", "C"): ("pod2", 0), ("V", "N"): ("pod2", 1), ("V", "H"): ("pod2", 2),
    }


def test_each_parent_changes_only_registered_slew_factors():
    queue = Q.load_queue(QUEUE)
    for parent in ("W", "V"):
        invariant = []
        for job in [item for item in queue["jobs"] if item["parent"] == parent]:
            values = _values(Q._training_argv(queue, job, "train"))
            for key in Q.FACTOR_KEYS | {"run_name", "device"}:
                values.pop(key, None)
            invariant.append(values)
        assert invariant[0] == invariant[1] == invariant[2]


def test_four_new_hydra_reward_keys_use_append_override():
    queue = Q.load_queue(QUEUE)
    argv = Q._training_argv(queue, queue["jobs"][0], "probe")
    new_keys = {
        "task.rewards.processed_qdes_slew_hinge_weight",
        "task.rewards.processed_qdes_slew_hinge_margin",
        "task.rewards.processed_qdes_slew_hinge_recovery_start_s",
        "task.rewards.processed_qdes_slew_hinge_recovery_end_s",
    }
    selected = [
        item for item in argv[2:] if Q._override_key(item, "argv") in new_keys
    ]
    assert len(selected) == 4
    assert all(item.startswith("++") for item in selected)


def test_probe_budget_uses_exclusive_6702_and_terminal_6701():
    queue = Q.load_queue(QUEUE)
    assert queue["budgets"]["probe"] == {
        "num_envs": 4096,
        "num_steps_per_env": 24,
        "additional_updates": 2,
        "max_iterations": 2,
        "save_interval": 1,
        "exclusive_iteration_upper_bound": 6702,
        "terminal_checkpoint_iteration": 6701,
        "terminal_checkpoint_basename": "model_6701.pt",
    }


def test_rollout_override_targets_existing_hydra_runner_key():
    algo = yaml.safe_load(PPO_CONFIG.read_text(encoding="utf-8"))
    assert algo["runner"]["num_steps_per_env"] == 24
    queue = Q.load_queue(QUEUE)
    argv = Q._training_argv(queue, queue["jobs"][0], "probe")
    assert "algo.runner.num_steps_per_env=24" in argv
    assert not any(item.startswith("algo.num_steps_per_env=") for item in argv)


def test_training_argv_really_composes_rollout_key_when_hydra_available():
    pytest.importorskip("hydra")
    queue = Q.load_queue(QUEUE)
    argv = Q._training_argv(queue, queue["jobs"][0], "probe")
    result = subprocess.run(
        [sys.executable, str(ROOT / "hope_training/whole_body_tracking/scripts/train.py"),
         "--cfg", "job", "--resolve", *argv[2:]],
        cwd=ROOT / "hope_training/whole_body_tracking",
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = yaml.safe_load(result.stdout)
    assert resolved["algo"]["runner"]["num_steps_per_env"] == 24
    assert "num_steps_per_env" not in resolved["algo"]


def test_probe_claim_uses_absolute_terminal_milestone_and_custom_binding(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    claim, argv = Q._build_claim(queue, queue["jobs"][0], "probe", manifest)
    assert claim["content"]["budget"]["milestones"] == [6701]
    assert claim["content"]["budget"]["num_steps_per_env"] == 24
    assert "algo.runner.num_steps_per_env=24" in argv
    assert claim["content"]["purpose"] == "balance_action_slew_probe_not_science"
    assert not any("training_queue_claim_path" in item for item in argv)
    assert not any("training_run_binding_path" in item for item in argv)
    assert claim["content"]["supervisor_argv_prefix"][-1] == "--"


def test_parent_audit_requires_model_optimizer_and_both_normalizers():
    program = Q.CHECKPOINT_AUDIT_PROGRAM
    for key in (
        "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    ):
        assert key in program
    assert "torch.isfinite" in program
    assert "optimizer.get(\"state\")" in program


def test_verifier_requires_optimizer_lineage_contract_and_exact_log_markers(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    job = queue["jobs"][2]
    claim, _ = Q._build_claim(queue, job, "probe", manifest)
    spec = Q._probe_verifier_spec(queue, job, manifest, claim)
    assert spec["num_steps_per_env"] == 24
    assert spec["expected_samples_per_update"] == 98304
    assert spec["expected_processed_qdes_contract"] == {
        "schema_version": 1,
        "enabled": True,
        "weight": -0.25,
        "margin": 0.85,
        "recovery_start_s": 0.2,
        "recovery_end_s": 1.55,
        "action_name": "joint_pos",
        "command_name": "racket_target",
        "joint_count": 15,
    }
    assert spec["expected_applied_markers"] == [
        "[train.py]     rewards.action_rate_l2.weight=0.0",
        "[train.py]     rewards.processed_qdes_slew_hinge.weight=-0.25",
        "[train.py]     rewards.processed_qdes_slew_hinge.params.margin=0.85",
        "[train.py]     rewards.processed_qdes_slew_hinge.params.recovery_start_s=0.2",
        "[train.py]     rewards.processed_qdes_slew_hinge.params.recovery_end_s=1.55",
        "[train.py]     rewards.processed_qdes_slew_hinge_probe=(margin=0.85,recovery=0.2..1.55,weight=1.0)",
    ]
    verifier = Q.PROBE_VERIFIER_PROGRAM
    assert "optimizer state/param_groups are missing or empty" in verifier
    assert "training_contract_lineage_exact" in verifier and "exact lineage value 0" in verifier
    assert "expected_processed_qdes_contract" in verifier
    assert "expected_applied_markers" in verifier
    assert "0 <= eligible <= valid <= observed" in verifier
    assert "two-update probe did not observe any recovery-eligible sample" in verifier
    assert 'observed != spec["expected_samples_per_update"]' in verifier
    assert 'qdot_observed != spec["expected_samples_per_update"]' in verifier


def test_counter_validator_rejects_impossible_ledgers_and_allows_tail_underflow():
    valid = _activation("H")
    underflow = copy.deepcopy(valid)
    underflow["rows"][0]["gated_tail_value_sum"] = 0.0
    underflow["totals"]["gated_tail_value_sum"] = 1.25
    Q._validate_activation_payload(underflow, "H", "underflow")

    # A float32 environment reduction can sit a few ulps above the exact
    # J/15 bound.  Accept that scale-aware rounding, but not a material excess.
    saturated = copy.deepcopy(valid)
    saturated["rows"][0]["tail_active_sample_count"] = 200
    saturated["rows"][0]["above_margin_joint_count"] = 3000
    saturated["rows"][0]["gated_tail_value_sum"] = 200.0000142
    saturated["totals"]["tail_active_sample_count"] += 180
    saturated["totals"]["above_margin_joint_count"] += 2960
    saturated["totals"]["gated_tail_value_sum"] += 198.7500142
    Q._validate_activation_payload(saturated, "H", "float32-boundary")
    over_bound = copy.deepcopy(saturated)
    over_bound["rows"][0]["gated_tail_value_sum"] = 200.001
    over_bound["totals"]["gated_tail_value_sum"] += 0.0009858
    with pytest.raises(Q.QueueError, match="tail/value bound"):
        Q._validate_activation_payload(over_bound, "H", "over-bound")

    # The first 24-step rollout is only 0.48 s long.  With the registered
    # arrival-time mixture and a recovery gate that starts 0.20 s after strike,
    # it can legitimately contain no recovery sample; the two-update probe as a
    # whole must still exercise the gate.
    startup_zero = copy.deepcopy(valid)
    for field in (
        "recovery_eligible_sample_count",
        "reward_enabled_eligible_sample_count",
        "tail_active_sample_count",
        "above_margin_joint_count",
        "gated_tail_value_sum",
    ):
        startup_zero["totals"][field] -= startup_zero["rows"][0][field]
        startup_zero["rows"][0][field] = 0
    Q._validate_activation_payload(startup_zero, "H", "startup-zero")

    no_recovery = copy.deepcopy(startup_zero)
    for field in (
        "recovery_eligible_sample_count",
        "reward_enabled_eligible_sample_count",
        "tail_active_sample_count",
        "above_margin_joint_count",
        "gated_tail_value_sum",
    ):
        no_recovery["totals"][field] -= no_recovery["rows"][1][field]
        no_recovery["rows"][1][field] = 0
    with pytest.raises(Q.QueueError, match="any recovery-eligible sample"):
        Q._validate_activation_payload(no_recovery, "H", "no-recovery")

    attacks = []
    bad = copy.deepcopy(valid)
    bad["rows"][0]["observed_sample_count"] = Q.EXPECTED_SAMPLES_PER_UPDATE + 1
    bad["rows"][0]["previous_qdes_valid_sample_count"] += 1
    bad["totals"]["observed_sample_count"] += 1
    bad["totals"]["previous_qdes_valid_sample_count"] += 1
    attacks.append(bad)
    bad = copy.deepcopy(valid)
    bad["rows"][0]["above_margin_joint_count"] = 301
    bad["totals"]["above_margin_joint_count"] += 261
    attacks.append(bad)
    bad = copy.deepcopy(valid)
    bad["rows"][0]["gated_tail_value_sum"] = 3.0
    bad["totals"]["gated_tail_value_sum"] += 1.75
    attacks.append(bad)
    bad = copy.deepcopy(valid)
    for row in bad["rows"]:
        row["previous_qdes_invalid_first_step_sample_count"] = 1
        row["previous_qdes_valid_sample_count"] = Q.EXPECTED_SAMPLES_PER_UPDATE - 1
    bad["totals"]["previous_qdes_invalid_first_step_sample_count"] = 2.0
    bad["totals"]["previous_qdes_valid_sample_count"] = float(
        2 * (Q.EXPECTED_SAMPLES_PER_UPDATE - 1)
    )
    attacks.append(bad)
    for candidate in attacks:
        with pytest.raises(Q.QueueError):
            Q._validate_activation_payload(candidate, "H", "attack")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("racket_swing_outcome_count", 0, "completion denominator"),
        ("racket_swing_completion_count", 101, "completion denominator"),
        ("racket_physical_fall_count", 9, "physical-fall closeout"),
        ("racket_strike_opportunity_count", 0, "legal-return denominator"),
        ("racket_virtual_legal_return_count", 81, "legal-return denominator"),
        ("racket_ready_tilt_eligible_sample_count", 0, "ready-tilt denominator"),
        ("racket_ready_nonfinite_value_count", 1, "ready-tilt denominator"),
        ("qdot_observed_sample_count", 4096, "qdot observed denominator"),
        ("qdot_excess_sample_count", Q.EXPECTED_SAMPLES_PER_UPDATE + 1, "qdot excess denominator"),
        ("qdot_normalized_excess_square_sum", 0.0, "qdot excess/value"),
    ],
)
def test_scientific_metric_probe_invariants_fail_closed(field, value, message):
    activation = _activation("H")
    old = activation["rows"][0][field]
    activation["rows"][0][field] = value
    activation["totals"][field] += float(value) - float(old)
    with pytest.raises(Q.QueueError, match=message):
        Q._validate_activation_payload(activation, "H", "scientific-metric")


@pytest.mark.parametrize(
    "observed",
    [4096, Q.EXPECTED_SAMPLES_PER_UPDATE - 4096, Q.EXPECTED_SAMPLES_PER_UPDATE + 4096],
)
def test_probe_rejects_incomplete_or_extra_rollout_denominators(observed):
    activation = _activation("H")
    row = activation["rows"][0]
    old_observed = row["observed_sample_count"]
    old_valid = row["previous_qdes_valid_sample_count"]
    row["observed_sample_count"] = observed
    row["previous_qdes_valid_sample_count"] = observed - row[
        "previous_qdes_invalid_first_step_sample_count"
    ]
    activation["totals"]["observed_sample_count"] += observed - old_observed
    activation["totals"]["previous_qdes_valid_sample_count"] += (
        row["previous_qdes_valid_sample_count"] - old_valid
    )
    with pytest.raises(Q.QueueError, match="activation denominator"):
        Q._validate_activation_payload(activation, "H", "qdes-rollout")

    activation = _activation("H")
    old = activation["rows"][0]["qdot_observed_sample_count"]
    activation["rows"][0]["qdot_observed_sample_count"] = observed
    activation["totals"]["qdot_observed_sample_count"] += observed - old
    with pytest.raises(Q.QueueError, match="qdot observed denominator"):
        Q._validate_activation_payload(activation, "H", "qdot-rollout")


def test_probe_rejects_forged_expected_samples_per_update():
    activation = _activation("H")
    activation["expected_samples_per_update"] = 4096
    with pytest.raises(Q.QueueError, match="expected_samples_per_update"):
        Q._validate_activation_payload(activation, "H", "forged-rollout")


def test_probe_metric_tags_cover_slew_behavior_ready_and_qdot():
    assert Q.PROBE_COUNTER_TAGS["observed_sample_count"].startswith(
        "Live/processed_qdes_slew/"
    )
    assert Q.PROBE_COUNTER_TAGS["racket_swing_outcome_count"] == (
        "Live/racket_target/swing_outcome_count"
    )
    assert Q.PROBE_COUNTER_TAGS["qdot_observed_sample_count"] == (
        "Live/qdot/observed_sample_count"
    )
    assert Q.PROBE_FLOAT_COUNTERS == {
        "gated_tail_value_sum",
        "racket_ready_tilt_rad_sum",
        "qdot_normalized_excess_square_sum",
    }


def test_verifier_file_gpu_and_fatal_checks_are_fail_closed():
    verifier = Q.PROBE_VERIFIER_PROGRAM
    assert "O_NOFOLLOW" in verifier and "os.fstat" in verifier
    assert "nvidia-smi returned nonnumeric nonempty output" in verifier
    assert "OutOfMemoryError" in verifier
    assert "tail_tolerance = max(1.0, abs(tail_bound)) * 1.0e-6" in verifier
    assert 'for counter, tag in spec["counter_tags"].items()' in verifier
    assert "behavior physical-fall closeout inconsistency" in verifier
    assert "ready-tilt denominator/value inconsistency" in verifier
    assert "qdot excess/value inconsistency" in verifier
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count('"fatal_log_scan_clean"') == 1


def test_command_generation_fails_closed_without_reviewed_manifest():
    queue = Q.load_queue(QUEUE)
    with pytest.raises(Q.QueueError, match="--launch-manifest"):
        Q.cmd_launch_commands(queue, stage="probe")


def test_origin_main_authority_binds_same_now_entry_and_tracked_bytes(tmp_path, monkeypatch):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    commit = "d" * 40
    now = (
        Q.AUTHORITY_NOW_TITLE
        + " "
        + Q.AUTHORITY_OWNER_EXECUTOR
        + " `"
        + Q.AUTHORITY_BRANCH
        + "` `"
        + queue["queue_id"]
        + "`\n- **[12｜P1] next.**"
    ).encode()

    def fake_git(_root, *args):
        if args == ("rev-parse", "HEAD") or args == (
            "rev-parse",
            "refs/remotes/origin/main",
        ):
            return (commit + "\n").encode()
        if args[:3] == ("status", "--porcelain=v1", "--untracked-files=no"):
            return b""
        if args == ("show", "refs/remotes/origin/main:docs/NOW.md"):
            return now
        if args == (
            "show",
            f"refs/remotes/origin/main:{Q.QUEUE_CONFIG_RELATIVE}",
        ):
            return QUEUE.read_bytes()
        if args == (
            "show",
            f"refs/remotes/origin/main:{Q.QUEUE_RUNNER_RELATIVE}",
        ):
            return SCRIPT.read_bytes()
        if args == (
            "show",
            f"refs/remotes/origin/main:{Q.LAUNCH_MANIFEST_RELATIVE}",
        ):
            return manifest.path.read_bytes()
        raise AssertionError(args)

    monkeypatch.setattr(Q, "_git_read", fake_git)
    result = REAL_VALIDATE_ORIGIN_MAIN_AUTHORITY(queue, manifest)
    assert result["origin_main_commit"] == commit
    assert result["human_owner"] == "franco"
    assert result["executor"] == "Codex"
    assert result["branch"] == Q.AUTHORITY_BRANCH


def test_origin_main_authority_rejects_split_or_missing_claim_fields():
    with pytest.raises(Q.QueueError, match="not bound in one"):
        Q._authority_now_entry(
            Q.AUTHORITY_NOW_TITLE
            + " "
            + Q.AUTHORITY_OWNER_EXECUTOR
            + "\n- **[12｜P1] other.** `"
            + Q.AUTHORITY_BRANCH
            + "`"
        )


def test_authorized_render_reports_origin_main_authority(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, _ = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue,
        stage="probe",
        launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    assert result["origin_main_authority"] == FAKE_ORIGIN_MAIN_AUTHORITY


def test_manifest_wrong_file_sha_and_wrong_source_commit_are_rejected(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, _ = _manifest(tmp_path, queue)
    with pytest.raises(Q.QueueError, match="reviewed authority"):
        Q._load_launch_manifest(queue, path, "a" * 64)
    envelope = json.loads(path.read_text())
    envelope["content"]["source"]["commit"] = "b" * 40
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    path.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="remote C1"):
        Q._load_launch_manifest(queue, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_binds_complete_six_file_preconverted_usd_bundle(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    usd = manifest.content["assets"]["preconverted_a3_usd"]
    assert usd["path"].endswith("/a3_preconverted_usd/model.usd")
    assert usd["bundle_root"].endswith("/a3_preconverted_usd")
    assert usd["file_count"] == 6
    assert usd["symlinks_forbidden"] is True
    assert len(usd["bundle_tree_sha256"]) == 64
    envelope = json.loads(path.read_text())
    envelope["content"]["assets"]["preconverted_a3_usd"]["file_count"] = 1
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    path.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="six files"):
        Q._load_launch_manifest(
            queue, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_remote_preflight_hashes_usd_bundle_tree_and_forbids_symlinks():
    program = Q.REMOTE_PREFLIGHT_PROGRAM
    assert "preconverted A3 USD six-file bundle" in program
    assert 'usd_bundle["bundle_root"]' in program
    assert 'usd_bundle["bundle_tree_sha256"]' in program
    assert "os.walk" in program
    assert "O_NOFOLLOW" in program
    assert "contains a symlink" in program


def test_probe_commands_bind_claim_and_use_single_remote_bash_argument(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue, stage="probe", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    assert len(result["jobs"]) == 6
    row = result["jobs"][0]
    assert len(row["ssh_argv"]) == 9
    assert row["ssh_argv"][-1].startswith("bash -lc ")
    assert row["ssh_argv"][-2].startswith("root@")
    assert "probe_verifier_command" in row
    claim, argv = Q._build_claim(queue, queue["jobs"][0], "probe", manifest)
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"
    assert claim["content"]["source"]["commit"] == Q.EXPECTED_REMOTE_SOURCE_COMMIT
    assert claim["content"]["launch_manifest"]["file_sha256"] == file_sha
    assert claim["content"]["inputs"] == manifest.content["assets"]


def test_compose_and_launch_set_usd_path_and_unbuffered(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, _ = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue, stage="probe", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    remote_arg = result["jobs"][0]["ssh_argv"][-1]
    assert remote_arg.count("HOPE_AGIBOT_A3_USD_PATH=") >= 2
    assert remote_arg.count("PYTHONUNBUFFERED=1") >= 2
    assert "--cfg job --resolve" in remote_arg
    assert "gpu_output=$(nvidia-smi" in remote_arg
    assert "nvidia-smi returned nonnumeric nonempty output" in remote_arg
    assert "NF != 1 || $1 !~ /^[1-9][0-9]*$/" in remote_arg
    assert "count=$(nvidia-smi" not in remote_arg


def test_probe_verifier_proves_terminal_artifacts_counters_and_release(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, _ = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue, stage="probe", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    assert all("probe_verifier_command" in row for row in result["jobs"])
    verifier = Q.PROBE_VERIFIER_PROGRAM
    for token in (
        "terminal_checkpoint_basename", "terminal checkpoint launch-claim lineage mismatch",
        "EventAccumulator", "expected_steps", "reward-enabled activation mismatch",
        "probe process group still has a live member", "assigned GPU was not released",
        "obs_norm_state_dict", "privileged_obs_norm_state_dict",
    ):
        assert token in verifier


def test_train_requires_all_six_complete_receipts(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    with pytest.raises(Q.QueueError, match="--probe-receipts-dir"):
        Q.cmd_launch_commands(
            queue, stage="train", launch_manifest_path=path,
            expected_launch_manifest_sha256=file_sha,
        )
    receipts = _receipt_dir(tmp_path, queue, manifest)
    (receipts / "v_h" / "probe_receipt.json").unlink()
    with pytest.raises(Q.QueueError, match="missing"):
        Q.cmd_launch_commands(
            queue, stage="train", launch_manifest_path=path,
            expected_launch_manifest_sha256=file_sha, probe_receipts_dir=receipts,
        )


def test_six_verified_receipts_unlock_and_bind_every_train_claim(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    receipts = _receipt_dir(tmp_path, queue, manifest)
    result = Q.cmd_launch_commands(
        queue, stage="train", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha, probe_receipts_dir=receipts,
    )
    assert result["probe_receipt_count"] == 6
    assert len(result["probe_receipt_set_sha256"]) == 64
    assert len(result["jobs"]) == 6
    assert all("max_iterations=1001" in row["launch_command"] for row in result["jobs"])
    loaded = Q._load_probe_receipts(queue, manifest, receipts)
    for job in queue["jobs"]:
        claim, argv = Q._build_claim(queue, job, "train", manifest, probe_receipts=loaded)
        assert len(claim["content"]["probe_receipts"]) == 6
        assert claim["content"]["probe_receipt_set_sha256"] == result["probe_receipt_set_sha256"]
        assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"


def test_forged_receipt_digest_and_counter_inconsistency_fail_closed(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    receipts = _receipt_dir(tmp_path, queue, manifest)
    target = receipts / "w_c" / "probe_receipt.json"
    envelope = json.loads(target.read_text())
    envelope["content"]["runtime"]["gpu_released"] = False
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="canonical digest mismatch"):
        Q._load_probe_receipts(queue, manifest, receipts)
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="terminal runtime proof"):
        Q._load_probe_receipts(queue, manifest, receipts)


def test_receipt_cannot_substitute_a_different_queue_claim_file_sha(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    receipts = _receipt_dir(tmp_path, queue, manifest)
    target = receipts / "w_c" / "probe_receipt.json"
    envelope = json.loads(target.read_text())
    envelope["content"]["artifacts"]["queue_claim_file_sha256"] = "a" * 64
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="canonical expected claim"):
        Q._load_probe_receipts(queue, manifest, receipts)


@pytest.mark.parametrize("kind", ["gpu", "run_name", "run_dir", "id"])
def test_duplicate_job_identity_or_resource_is_rejected(tmp_path, kind):
    raw = _raw()
    if kind == "gpu":
        raw["jobs"][1]["gpu"] = raw["jobs"][0]["gpu"]
    elif kind == "run_name":
        raw["jobs"][1]["run_name"] = raw["jobs"][0]["run_name"]
    elif kind == "run_dir":
        raw["jobs"][1]["run_dir"] = raw["jobs"][0]["run_dir"]
    else:
        raw["jobs"][1]["id"] = raw["jobs"][0]["id"]
    with pytest.raises(Q.QueueError, match="duplicate|changed its matrix"):
        Q.load_queue(_write_yaml(tmp_path, raw))


def test_duplicate_yaml_key_and_unknown_field_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        QUEUE.read_text().replace("schema_version: 1\n", "schema_version: 1\nschema_version: 1\n", 1)
    )
    with pytest.raises(Q.QueueError, match="duplicate YAML key"):
        Q.load_queue(duplicate)
    raw = _raw()
    raw["launch_now"] = True
    with pytest.raises(Q.QueueError, match="queue keys differ"):
        Q.load_queue(_write_yaml(tmp_path, raw))


def test_default_cli_is_no_launch_and_authorized_cli_stays_blocked_without_manifest():
    default = subprocess.run(
        [sys.executable, str(SCRIPT), "--queue", str(QUEUE)],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(default.stdout)
    assert plan["commands_emitted"] is False
    assert plan["launch_manifest_gate"]["status"] == "blocked_manifest_not_supplied_to_this_invocation"
    blocked = subprocess.run(
        [sys.executable, str(SCRIPT), "--queue", str(QUEUE), "--authorize-launch"],
        check=False, capture_output=True, text=True,
    )
    assert blocked.returncode == 2
    assert "--launch-manifest" in blocked.stderr


def test_module_has_only_read_only_local_git_execution_and_no_signal_surface():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    subprocess_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(subprocess_calls) == 1
    assert subprocess_calls[0].func.attr == "run"
    assert '["git", "-C", str(repo_root), *args]' in source
    assert "os.system(" not in source
    assert "os.kill(" not in source
    assert "--execute" not in source
    assert "--probe-approved" not in source
