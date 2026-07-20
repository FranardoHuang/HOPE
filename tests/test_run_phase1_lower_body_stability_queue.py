from __future__ import annotations

import ast
import base64
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
SCRIPT = ROOT / "scripts" / "run_phase1_lower_body_stability_queue.py"
QUEUE = ROOT / "configs" / "phase1_lower_body_stability_20260720.yaml"
CHECKED_MANIFEST = ROOT / "configs" / "phase1_lower_body_stability_launch_manifest_20260720.json"


def _module():
    spec = importlib.util.spec_from_file_location("lower_body_queue_under_test", SCRIPT)
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
    "wave_a_branch": Q.AUTHORITY_WAVE_A_BRANCH,
    "wave_b_branch": Q.AUTHORITY_WAVE_B_BRANCH,
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
                "file_count": 46,
                "total_file_bytes": 15378264,
                "symlinks_forbidden": True,
            },
            "preconverted_a3_usd": {
                "path": queue["assets"]["preconverted_a3_usd"],
                "sha256": "3" * 64,
                "bundle_root": str(PurePosixPath(queue["assets"]["preconverted_a3_usd"]).parent),
                "bundle_tree_sha256": "b" * 64,
                "file_count": 6,
                "total_file_bytes": 21897893,
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


def _group(counter_names: tuple[str, ...], rows: list[dict]) -> dict:
    return {
        "rows": rows,
        "totals": {
            counter: sum(float(row[counter]) for row in rows)
            for counter in counter_names
        },
    }


def _activation(mechanism: str) -> dict:
    pose_rows = []
    bundle_rows = []
    for step in Q.PROBE_STEPS:
        eligible = 200
        pose_rows.append(
            {
                "step": step,
                "observed_sample_count": 4096 * 24,
                "support_eligible_sample_count": eligible,
                "reward_enabled_eligible_sample_count": eligible if mechanism == "B1" else 0,
                "gated_kernel_sum": 150.0,
                "gated_joint_abs_error_mean_sum": 24.0,
                "gated_reference_motion_l1_mean_sum": 12.0,
            }
        )
        bundle_rows.append(
            {
                "step": step,
                "observed_sample_count": 4096 * 24,
                "support_eligible_sample_count": eligible,
                "reward_enabled_eligible_sample_count": eligible if mechanism == "B2" else 0,
                "narrow_or_crossed_sample_count": 30,
                "gated_bundle_sum": 30.0,
                "gated_stance_tail_sum": 20.0,
                "gated_leg_velocity_tail_sum": 40.0,
                # Signed width evidence may legitimately be negative.
                "gated_signed_stance_width_m_sum": -3.5,
            }
        )
    return {
        "expected_steps": list(Q.PROBE_STEPS),
        "pose": _group(Q.POSE_PROBE_COUNTERS, pose_rows),
        "bundle": _group(Q.BUNDLE_PROBE_COUNTERS, bundle_rows),
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


def test_checked_in_manifest_binds_current_queue_and_runner_bytes():
    queue = Q.load_queue(QUEUE)
    file_sha = hashlib.sha256(CHECKED_MANIFEST.read_bytes()).hexdigest()
    manifest = Q._load_launch_manifest(queue, CHECKED_MANIFEST, file_sha)
    assert file_sha == "bae461b01382e82405202440429e6af9bf41ee8b6de2b456be98104da5f55565"
    assert manifest.content_sha256 == "feed3ed1b3a684b696e0abfe00822399a2d43d47f3c1308073ac11173434d576"


def test_exact_remote_source_commit_and_simulation_only_contract():
    queue = Q.load_queue(QUEUE)
    assert queue["source"]["identity_mode"] == "clean_detached_exact_commit"
    assert queue["source"]["commit"] == "5db7366aaa1562d592093dc0d512ec212f14e39e"
    assert queue["simulation_only"] is True
    assert queue["real_robot_authorized"] is False
    assert queue["formal_exact_eligible"] is False


def test_matrix_is_wv_by_b0b1b2_on_six_unique_gpus():
    queue = Q.load_queue(QUEUE)
    observed = {
        (job["parent"], job["mechanism"]): (job["pod"], job["gpu"])
        for job in queue["jobs"]
    }
    assert observed == {
        ("W", "B0"): ("pod1", 0), ("W", "B1"): ("pod1", 1), ("W", "B2"): ("pod1", 2),
        ("V", "B0"): ("pod2", 0), ("V", "B1"): ("pod2", 1), ("V", "B2"): ("pod2", 2),
    }


def test_every_cell_explicitly_binds_both_weights_and_fixed_support_gate():
    queue = Q.load_queue(QUEUE)
    expected = {"B0": (0.0, 0.0), "B1": (0.5, 0.0), "B2": (0.0, -0.25)}
    for name, mechanism in queue["mechanisms"].items():
        assert (
            mechanism["lower_body_pose_imitation_weight"],
            mechanism["lower_body_stability_bundle_weight"],
        ) == expected[name]
        assert mechanism["lower_body_pose_imitation_support_pre_s"] == 0.30
        assert mechanism["lower_body_pose_imitation_support_post_s"] == 0.40
        assert mechanism["lower_body_stability_support_pre_s"] == 0.30
        assert mechanism["lower_body_stability_support_post_s"] == 0.40


def test_each_parent_changes_only_registered_lower_body_factors():
    queue = Q.load_queue(QUEUE)
    for parent in ("W", "V"):
        invariant = []
        for job in [item for item in queue["jobs"] if item["parent"] == parent]:
            values = _values(Q._training_argv(queue, job, "train"))
            for key in Q.FACTOR_KEYS | {"run_name", "device"}:
                values.pop(key, None)
            invariant.append(values)
        assert invariant[0] == invariant[1] == invariant[2]


def test_action_rate_is_held_and_processed_qdes_axis_is_absent():
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        values = _values(Q._training_argv(queue, job, "train"))
        assert values["task.rewards.action_rate_weight"] == "-0.1"
        assert not (set(values) & Q.FORBIDDEN_PROCESSED_SLEW_KEYS)


def test_all_eleven_new_hydra_reward_keys_use_append_override():
    queue = Q.load_queue(QUEUE)
    argv = Q._training_argv(queue, queue["jobs"][0], "probe")
    selected = [
        item for item in argv[2:] if Q._override_key(item, "argv") in Q.FACTOR_KEYS
    ]
    assert len(selected) == 11
    assert {Q._override_key(item, "argv") for item in selected} == Q.FACTOR_KEYS
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


def test_probe_claim_uses_absolute_terminal_milestone_and_custom_binding(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    claim, argv = Q._build_claim(queue, queue["jobs"][0], "probe", manifest)
    assert claim["content"]["budget"]["milestones"] == [6701]
    assert claim["content"]["budget"]["num_steps_per_env"] == 24
    assert claim["content"]["purpose"] == "lower_body_stability_probe_not_science"
    assert "algo.runner.num_steps_per_env=24" in argv
    assert not any("training_queue_claim_path" in item for item in argv)
    assert not any("training_run_binding_path" in item for item in argv)
    assert claim["content"]["supervisor_argv_prefix"][-1] == "--"


def test_required_source_binding_covers_gate_ledger_and_contract_modules():
    queue = Q.load_queue(QUEUE)
    required = set(Q._manifest_source_relative_paths(queue))
    for suffix in (
        "cfg/algo/ppo.yaml",
        "scripts/train.py",
        "tasks/tracking/mdp/hope_commands.py",
        "tasks/tracking/mdp/hope_rewards.py",
        "tasks/tracking/config/agibot_a3/hope_env_cfg.py",
        "utils/my_on_policy_runner.py",
        "utils/training_contract.py",
    ):
        assert any(path.endswith(suffix) for path in required)


def test_verifier_contracts_bind_twelve_legs_reference_boundary_and_markers(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    b1 = next(job for job in queue["jobs"] if job["mechanism"] == "B1")
    b2 = next(job for job in queue["jobs"] if job["mechanism"] == "B2")
    b1_claim, _ = Q._build_claim(queue, b1, "probe", manifest)
    b2_claim, _ = Q._build_claim(queue, b2, "probe", manifest)
    b1_spec = Q._probe_verifier_spec(queue, b1, manifest, b1_claim)
    b2_spec = Q._probe_verifier_spec(queue, b2, manifest, b2_claim)
    assert b1_spec["expected_pose_contract"]["enabled"] is True
    assert b1_spec["expected_pose_contract"]["joint_count"] == 12
    assert b1_spec["expected_pose_contract"]["joint_names"] == Q.EXPECTED_LEG_JOINTS
    assert b2_spec["expected_bundle_contract"]["enabled"] is True
    assert b2_spec["expected_bundle_contract"]["uses_motion_reference"] is False
    assert b2_spec["expected_bundle_contract"]["foot_body_names"] == Q.EXPECTED_FOOT_BODIES
    assert b2_spec["expected_motion_sha256"] == ["4" * 64, "5" * 64]
    assert b2_spec["expected_bank_sha256"] == "6" * 64
    assert b2_spec["expected_applied_markers"][0].endswith("action_rate_l2.weight=-0.1")
    assert all("support_pre_s=0.3" in " ".join(b1_spec["expected_applied_markers"]) for _ in [0])
    assert "expected_pose_contract" in Q.PROBE_VERIFIER_PROGRAM
    assert "expected_bundle_contract" in Q.PROBE_VERIFIER_PROGRAM


@pytest.mark.parametrize("mechanism", ["B0", "B1", "B2"])
def test_counter_validator_accepts_paired_ledgers_and_signed_width(mechanism):
    Q._validate_activation_payload(
        _activation(mechanism),
        mechanism,
        mechanism,
        expected_observed_sample_count=4096 * 24,
    )


def test_counter_validator_rejects_denominator_enabled_reference_and_identity_attacks():
    attacks = []
    bad = _activation("B1")
    bad["bundle"]["rows"][0]["support_eligible_sample_count"] += 1
    bad["bundle"]["totals"]["support_eligible_sample_count"] += 1
    attacks.append((bad, "B1"))
    bad = _activation("B1")
    bad["pose"]["rows"][0]["reward_enabled_eligible_sample_count"] = 0
    bad["pose"]["totals"]["reward_enabled_eligible_sample_count"] -= 200
    attacks.append((bad, "B1"))
    bad = _activation("B0")
    for row in bad["pose"]["rows"]:
        row["gated_reference_motion_l1_mean_sum"] = 0.0
    bad["pose"]["totals"]["gated_reference_motion_l1_mean_sum"] = 0.0
    attacks.append((bad, "B0"))
    bad = _activation("B2")
    bad["bundle"]["rows"][0]["gated_bundle_sum"] = 35.0
    bad["bundle"]["totals"]["gated_bundle_sum"] += 5.0
    attacks.append((bad, "B2"))
    bad = _activation("B0")
    bad["pose"]["rows"][0]["gated_kernel_sum"] = 201.0
    bad["pose"]["totals"]["gated_kernel_sum"] += 51.0
    attacks.append((bad, "B0"))
    for activation, mechanism in attacks:
        with pytest.raises(Q.QueueError):
            Q._validate_activation_payload(
                activation,
                mechanism,
                "attack",
                expected_observed_sample_count=4096 * 24,
            )


def test_verifier_file_gpu_fatal_full_state_and_signed_counter_checks_are_fail_closed():
    verifier = Q.PROBE_VERIFIER_PROGRAM
    for token in (
        "O_NOFOLLOW", "os.fstat", "nvidia-smi returned nonnumeric nonempty output",
        "OutOfMemoryError", "Infinity", "optimizer state/param_groups are missing or empty",
        "training_contract_lineage_exact", "gated_signed_stance_width_m_sum",
        "two-update probe did not observe nontrivial lower-body motion reference",
        "hard training contract motion clip SHA mismatch",
        "hard training contract question-bank SHA mismatch",
        "unexpectedly enables processed-qdes slew",
        'observed != spec["num_envs"] * spec["num_steps_per_env"]',
    ):
        assert token in verifier
    assert 'counter not in signed_names and value < 0' in verifier
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("OutOfMemoryError") >= 2
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
        + Q.AUTHORITY_WAVE_A_BRANCH
        + "` `"
        + Q.AUTHORITY_WAVE_B_BRANCH
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
    assert result["wave_a_branch"] == Q.AUTHORITY_WAVE_A_BRANCH
    assert result["wave_b_branch"] == Q.AUTHORITY_WAVE_B_BRANCH


def test_origin_main_authority_rejects_split_or_missing_claim_fields():
    with pytest.raises(Q.QueueError, match="not bound in one"):
        Q._authority_now_entry(
            Q.AUTHORITY_NOW_TITLE
            + " "
            + Q.AUTHORITY_OWNER_EXECUTOR
            + "\n- **[12｜P1] other.** `"
            + Q.AUTHORITY_WAVE_A_BRANCH
            + "` `"
            + Q.AUTHORITY_WAVE_B_BRANCH
            + "`"
        )


def test_manifest_wrong_file_sha_and_wrong_source_commit_are_rejected(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, _, _ = _manifest(tmp_path, queue)
    with pytest.raises(Q.QueueError, match="reviewed authority"):
        Q._load_launch_manifest(queue, path, "a" * 64)
    envelope = json.loads(path.read_text())
    envelope["content"]["source"]["commit"] = "b" * 40
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    path.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="Wave-B"):
        Q._load_launch_manifest(queue, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_binds_complete_six_file_preconverted_usd_bundle(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    usd = manifest.content["assets"]["preconverted_a3_usd"]
    assert usd["path"].endswith("/a3_preconverted_usd/model.usd")
    assert usd["file_count"] == 6
    assert usd["symlinks_forbidden"] is True
    envelope = json.loads(path.read_text())
    envelope["content"]["assets"]["preconverted_a3_usd"]["file_count"] = 1
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    path.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="six files"):
        Q._load_launch_manifest(
            queue, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_remote_preflight_hashes_usd_bundle_and_forbids_symlinks():
    program = Q.REMOTE_PREFLIGHT_PROGRAM
    assert "preconverted A3 USD six-file bundle" in program
    assert 'usd_bundle["bundle_tree_sha256"]' in program
    assert "os.walk" in program
    assert "O_NOFOLLOW" in program
    assert "contains a symlink" in program
    assert '"symbolic-ref", "-q", "HEAD"' in program
    assert "source checkout is not detached" in program
    assert '"detached_head": True' in Q.PROBE_SUPERVISOR_PROGRAM


def test_remote_preflight_rejects_branch_head_even_at_exact_commit(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Queue Test"], check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("bound\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "bound"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    content = {
        "source": {"checkout": str(repo), "commit": commit, "required_file_sha256": {}},
        "assets": {},
        "parents": {},
    }
    envelope = {
        "content": content,
        "content_sha256": Q._canonical_sha256(content),
    }
    encoded = base64.b64encode(Q._canonical_bytes(envelope)).decode("ascii")
    result = subprocess.run(
        [sys.executable, "-c", Q.REMOTE_PREFLIGHT_PROGRAM, encoded],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "source checkout is not detached" in result.stderr


def test_probe_commands_bind_claim_and_use_single_remote_bash_argument(tmp_path):
    queue = Q.load_queue(QUEUE)
    path, file_sha, manifest = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue, stage="probe", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    assert result["origin_main_authority"] == FAKE_ORIGIN_MAIN_AUTHORITY
    assert len(result["jobs"]) == 6
    row = result["jobs"][0]
    assert len(row["ssh_argv"]) == 9
    assert row["ssh_argv"][-1].startswith("bash -lc ")
    assert row["ssh_argv"][-2].startswith("root@")
    assert "probe_verifier_command" in row
    claim, argv = Q._build_claim(queue, queue["jobs"][0], "probe", manifest)
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"
    assert claim["content"]["source"]["commit"] == Q.EXPECTED_REMOTE_SOURCE_COMMIT
    assert claim["content"]["inputs"] == manifest.content["assets"]


def test_compose_and_launch_set_usd_path_unbuffered_and_strict_gpu_check(tmp_path):
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
    assert "NF != 1 || $1 !~ /^[1-9][0-9]*$/" in remote_arg
    assert "OutOfMemoryError" in remote_arg


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
        "EventAccumulator", "paired lower-body denominator inconsistency",
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
    (receipts / "v_b2" / "probe_receipt.json").unlink()
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


@pytest.mark.parametrize("observed", [4096, 4096 * 23, 4096 * 25])
def test_probe_receipt_rejects_any_nonexact_rollout_sample_count(tmp_path, observed):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    receipts = _receipt_dir(tmp_path, queue, manifest)
    target = receipts / "w_b0" / "probe_receipt.json"
    envelope = json.loads(target.read_text())
    activation = envelope["content"]["activation"]
    for group_name in ("pose", "bundle"):
        group = activation[group_name]
        for row in group["rows"]:
            row["observed_sample_count"] = observed
        group["totals"]["observed_sample_count"] = observed * len(Q.PROBE_STEPS)
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="paired denominator inconsistency"):
        Q._load_probe_receipts(queue, manifest, receipts)


def test_forged_receipt_digest_and_runtime_fail_closed(tmp_path):
    queue = Q.load_queue(QUEUE)
    _, _, manifest = _manifest(tmp_path, queue)
    receipts = _receipt_dir(tmp_path, queue, manifest)
    target = receipts / "w_b0" / "probe_receipt.json"
    envelope = json.loads(target.read_text())
    envelope["content"]["runtime"]["gpu_released"] = False
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="canonical digest mismatch"):
        Q._load_probe_receipts(queue, manifest, receipts)
    envelope["content_sha256"] = Q._canonical_sha256(envelope["content"])
    target.write_bytes(Q._json_document(envelope))
    with pytest.raises(Q.QueueError, match="terminal runtime proof"):
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


def test_missing_paired_weight_and_mutually_enabled_cell_fail_closed(tmp_path):
    raw = _raw()
    del raw["mechanisms"]["B0"]["lower_body_stability_bundle_weight"]
    with pytest.raises(Q.QueueError, match="keys differ"):
        Q.load_queue(_write_yaml(tmp_path, raw))
    raw = _raw()
    raw["mechanisms"]["B1"]["lower_body_stability_bundle_weight"] = -0.25
    with pytest.raises(Q.QueueError, match="bundle=0.0|mutually exclusive"):
        Q.load_queue(_write_yaml(tmp_path, raw))


def test_m0_is_not_an_asset_cell_or_training_command(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["measurement_contract"]["moving_teacher_authorized"] is False
    assert set(queue["assets"]) == {
        "a3_runtime_asset_root", "preconverted_a3_usd", "motion_forehand",
        "motion_backhand", "training_question_bank",
    }
    assert all("m0" not in item.lower() for job in queue["jobs"] for item in Q._training_argv(queue, job, "train"))
    path, file_sha, _ = _manifest(tmp_path, queue)
    result = Q.cmd_launch_commands(
        queue, stage="probe", launch_manifest_path=path,
        expected_launch_manifest_sha256=file_sha,
    )
    for row in result["jobs"]:
        assert row["job_id"] in Q.EXPECTED_JOBS
        assert "/m0" not in row["run_dir"].lower()


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
    blocked = subprocess.run(
        [sys.executable, str(SCRIPT), "--queue", str(QUEUE), "--authorize-launch"],
        check=False, capture_output=True, text=True,
    )
    assert blocked.returncode == 2
    assert "--launch-manifest" in blocked.stderr


def test_module_only_has_read_only_git_authority_subprocess_and_no_signal_surface():
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
