import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_signed_face_d_retry_prereg_20260713.json"
LAUNCHER = ROOT / "scripts/run_phase1_signed_face_d_retry.py"
SPEC = importlib.util.spec_from_file_location("phase1_signed_face_d_retry", LAUNCHER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def manifest():
    return M.load_manifest(CONFIG)


def test_manifest_is_one_cell_versioned_retry_and_keeps_l2_judge_seed_blocked():
    data = manifest()
    assert data["manifest_id"].endswith("-v6r1")
    retry = data["retry_authority"]
    assert retry["cell_id"] == "D"
    assert retry["old_run_name"] == M.OLD_RUN_NAME
    assert retry["new_run_name"] == M.NEW_RUN_NAME
    assert retry["only_command_change_allowed"] == "run_name"
    assert retry["expected_terminal_checkpoint_iteration"] == 24
    assert data["automatic_judge_launch"] is False
    assert data["l2_training_launch_authorized"] is False
    assert data["second_seed_authorized"] is False
    assert data["direct_signals_by_retry_tool_forbidden"] is True
    assert data["locked_launcher_exact_pgid_boot_timeout_cleanup_allowed"] is True
    assert data["broad_signals_forbidden"] is True
    assert data["runtime"]["locked_launcher_sha256"] == M.LOCKED_LAUNCHER_SHA256
    assert data["runtime"]["post_contract_timeout_requires_manual_exact_state_pgid_audit"] is True
    assert data["runtime"]["training_log_root"] == str(M.TRAINING_LOG_ROOT)
    assert data["runtime"]["retry_training_run_dir_glob_suffix"] == M.NEW_RUN_GLOB_SUFFIX


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda x: x["foreign_v6"].__setitem__("config_sha256", "0" * 64),
            "foreign_v6 config_sha256 changed",
        ),
        (
            lambda x: x["retry_authority"].__setitem__("cell_id", "C"),
            "retry authority cell_id changed",
        ),
        (
            lambda x: x["retry_authority"].__setitem__("new_run_name", M.OLD_RUN_NAME),
            "retry authority new_run_name changed",
        ),
        (
            lambda x: x["retry_authority"]["old_outer_evidence"]["training_log"].__setitem__(
                "sha256", "0" * 64
            ),
            "old D training_log SHA changed",
        ),
        (
            lambda x: x["mixed_finalizer"]["required_lineage_exact"].__setitem__("D", False),
            "mixed finalizer contract changed",
        ),
        (
            lambda x: x.__setitem__(
                "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed", False
            ),
            "manifest locked_launcher_exact_pgid_boot_timeout_cleanup_allowed changed",
        ),
        (
            lambda x: x["runtime"].__setitem__(
                "training_log_root", "/tmp/unfrozen-training-log-root"
            ),
            "v6r1 runtime/write-path contract changed",
        ),
        (
            lambda x: x["runtime"].__setitem__(
                "retry_training_run_dir_glob_suffix", "_another-run"
            ),
            "v6r1 runtime/write-path contract changed",
        ),
    ],
)
def test_manifest_mutations_fail_closed(tmp_path, mutate, message):
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(M.ContractError, match=message):
        M.load_manifest(path)


def test_retry_command_changes_exactly_one_run_name_argument():
    original = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONUNBUFFERED=1",
        "/python",
        "scripts/train.py",
        "seed=3",
        f"run_name={M.OLD_RUN_NAME}",
        "checkpoint_path=null",
    ]
    retry = M.build_retry_command(original, M.OLD_RUN_NAME, M.NEW_RUN_NAME)
    assert len(retry) == len(original)
    differences = [(left, right) for left, right in zip(original, retry) if left != right]
    assert differences == [
        (f"run_name={M.OLD_RUN_NAME}", f"run_name={M.NEW_RUN_NAME}")
    ]
    assert original[6] == f"run_name={M.OLD_RUN_NAME}"


@pytest.mark.parametrize(
    "bad",
    [
        ["env", "scripts/train.py"],
        [
            "env",
            f"run_name={M.OLD_RUN_NAME}",
            f"run_name={M.OLD_RUN_NAME}",
        ],
    ],
)
def test_retry_command_rejects_missing_or_duplicate_run_name(bad):
    with pytest.raises(M.ContractError, match="exactly one"):
        M.build_retry_command(bad, M.OLD_RUN_NAME, M.NEW_RUN_NAME)


@pytest.mark.parametrize("entry_kind", ["directory", "symlink", "regular_file"])
def test_retry_training_run_name_residue_blocks_launch_readiness(
    tmp_path, monkeypatch, entry_kind
):
    data = copy.deepcopy(manifest())
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(M, "TRAINING_LOG_ROOT", log_root)
    data["runtime"]["training_log_root"] = str(log_root)
    assert M.verify_retry_training_run_absent(data)["matching_entry_count"] == 0

    residue = log_root / f"2026-07-13_23-59-59{M.NEW_RUN_GLOB_SUFFIX}"
    if entry_kind == "directory":
        residue.mkdir()
    elif entry_kind == "symlink":
        target = tmp_path / "symlink-target"
        target.mkdir()
        residue.symlink_to(target, target_is_directory=True)
    else:
        residue.write_text("manual residue", encoding="utf-8")
    with pytest.raises(M.ContractError, match="run name already exists"):
        M.verify_retry_training_run_absent(data)


def test_finalizer_accepts_only_runtime_verified_exact_run_dir(tmp_path):
    runtime = copy.deepcopy(manifest()["runtime"])
    root = tmp_path / "logs"
    root.mkdir()
    runtime["training_log_root"] = str(root)
    exact = root / f"2026-07-13_23-59-59{M.NEW_RUN_GLOB_SUFFIX}"
    exact.mkdir()
    assert M.validate_retry_training_run_dir(runtime, exact) == exact

    wrong_name = root / "2026-07-13_23-59-59_other-run"
    wrong_name.mkdir()
    with pytest.raises(M.ContractError, match="frozen root/name suffix"):
        M.validate_retry_training_run_dir(runtime, wrong_name)

    target = tmp_path / "target-run"
    target.mkdir()
    linked = root / f"2026-07-14_00-00-00{M.NEW_RUN_GLOB_SUFFIX}"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(M.ContractError, match="non-symlink directory"):
        M.validate_retry_training_run_dir(runtime, linked)


def _failed_claim_fixture(tmp_path):
    data = copy.deepcopy(manifest())
    old = data["retry_authority"]["old_outer_evidence"]
    outer_dir = tmp_path / "outer"
    outer_dir.mkdir()
    failed_training = tmp_path / "failed_training"
    failed_training.mkdir()
    data["retry_authority"]["old_failed_training_run_dir"] = str(failed_training)
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONUNBUFFERED=1",
        "/python",
        "scripts/train.py",
        f"run_name={M.OLD_RUN_NAME}",
        "checkpoint_path=null",
    ]
    launch = {
        "manifest_id": M.FOREIGN_MANIFEST_ID,
        "manifest_file_sha256": M.FOREIGN_CONFIG_SHA256,
        "launcher_file_sha256": M.FOREIGN_LAUNCHER_SHA256,
        "training_commit": M.TRAINING_COMMIT,
        "stage": "l1",
        "cell_id": "D",
        "causal_role": "fresh_guidance",
        "initialization": "fresh",
        "expected_lineage_exact": True,
        "run_name": M.OLD_RUN_NAME,
        "command": command,
    }
    files = {
        "launch_contract": outer_dir / "launch_contract.json",
        "launch_state": outer_dir / "run.log.launch",
        "training_log": outer_dir / "run.log",
        "timeout_diagnostic": outer_dir / "d_timeout_diagnostic.txt",
    }
    files["launch_contract"].write_text(json.dumps(launch), encoding="utf-8")
    files["launch_state"].write_text(
        "pid=1759428\npgid=1759428\nboot_timeout_s=900\n", encoding="utf-8"
    )
    files["training_log"].write_text(
        f"[INFO] log: {failed_training}\nboot stopped before hard contract\n",
        encoding="utf-8",
    )
    training_sha = M.sha256_file(files["training_log"])
    files["timeout_diagnostic"].write_text(
        f"{training_sha}  {files['training_log']}\nD_GROUP_BEGIN\nD_GROUP_END\n",
        encoding="utf-8",
    )
    for name, path in files.items():
        old[name] = {"path": str(path), "sha256": M.sha256_file(path)}
    foreign = SimpleNamespace(build_command=lambda *_args, **_kwargs: command)
    preflight = {"wbt": tmp_path / "wbt"}
    return data, foreign, {}, preflight, outer_dir, failed_training


def test_failed_d_claim_requires_exact_dead_pre_runtime_no_checkpoint_evidence(
    tmp_path, monkeypatch
):
    data, foreign, foreign_manifest, preflight, _outer, _failed = _failed_claim_fixture(
        tmp_path
    )
    monkeypatch.setattr(M, "process_entry_exists", lambda _pid: False)
    result = M.verify_failed_d_claim(data, foreign, foreign_manifest, preflight)
    assert result["old_pid"] == 1759428
    assert result["old_runtime_verified_absent"] is True
    assert result["old_checkpoint_count"] == 0
    assert result["command_delta"] == {
        "field": "run_name",
        "old": M.OLD_RUN_NAME,
        "new": M.NEW_RUN_NAME,
    }


@pytest.mark.parametrize("violation", ["runtime_verified", "checkpoint", "live_pid"])
def test_failed_d_claim_rejects_invalid_retry_authority(tmp_path, monkeypatch, violation):
    data, foreign, foreign_manifest, preflight, outer, failed = _failed_claim_fixture(tmp_path)
    monkeypatch.setattr(M, "process_entry_exists", lambda _pid: violation == "live_pid")
    if violation == "runtime_verified":
        (outer / "runtime_verified.json").write_text("{}", encoding="utf-8")
    if violation == "checkpoint":
        (failed / "model_0.pt").write_bytes(b"checkpoint")
    with pytest.raises(M.ContractError):
        M.verify_failed_d_claim(data, foreign, foreign_manifest, preflight)


def test_checkpoint_audit_contract_requires_finite_exact_lineage():
    audit = {
        "iter": 24,
        "training_contract_schema_version": 3,
        "training_contract_sha256": M.EXPECTED_HARD_CONTRACT_SHA256,
        "training_contract_lineage_exact": 1,
        "training_contract_provenance_location": "infos",
        "floating_tensor_count": 7,
        "nonfinite_floating_elements": 0,
    }
    M._check_audit(
        audit,
        terminal=24,
        lineage=True,
        contract_sha=M.EXPECTED_HARD_CONTRACT_SHA256,
        label="D",
    )
    bad = dict(audit, nonfinite_floating_elements=1)
    with pytest.raises(M.ContractError, match="nonfinite_floating_elements"):
        M._check_audit(
            bad,
            terminal=24,
            lineage=True,
            contract_sha=M.EXPECTED_HARD_CONTRACT_SHA256,
            label="D",
        )


def test_original_checkpoint_audit_binds_exact_abc_and_empty_d(tmp_path):
    data = copy.deepcopy(manifest())
    rows = []
    for cell_id in ("A", "B", "C"):
        cell = data["original_terminal_cells"][cell_id]
        rows.append(
            {
                "cell": cell_id,
                "checkpoint_exists": True,
                "checkpoint_path": cell["checkpoint_path"],
                "checkpoint_sha": cell["checkpoint_sha256"],
                "checkpoint_bytes": cell["checkpoint_bytes"],
                "contract_exists": True,
                "contract_sha": M.EXPECTED_HARD_CONTRACT_SHA256,
                "expected_lineage": int(cell["expected_lineage_exact"]),
                "run_dirs": [str(Path(cell["checkpoint_path"]).parent)],
                "audit": {
                    "iter": cell["expected_terminal_checkpoint_iteration"],
                    "lineage": int(cell["expected_lineage_exact"]),
                    "schema": 3,
                    "contract_sha": M.EXPECTED_HARD_CONTRACT_SHA256,
                    "nonfinite": 0,
                    "floating_tensor_count": 74,
                    "floating_elements": 1762715,
                },
            }
        )
    rows.append({"cell": "D", "expected_lineage": 1, "run_dirs": []})
    audit_path = tmp_path / "l1_checkpoint_audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    data["original_l1_checkpoint_audit"] = {
        **data["original_l1_checkpoint_audit"],
        "path": str(audit_path),
        "sha256": M.sha256_file(audit_path),
    }
    result = M.verify_original_checkpoint_audit(data)
    assert result["cells"] == ["A", "B", "C", "D"]

    rows[2]["checkpoint_sha"] = "0" * 64
    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    data["original_l1_checkpoint_audit"]["sha256"] = M.sha256_file(audit_path)
    with pytest.raises(M.ContractError, match="checkpoint_sha changed"):
        M.verify_original_checkpoint_audit(data)


def test_locked_launcher_timeout_is_exact_pgid_cleanup_not_zero_signal(
    tmp_path, monkeypatch
):
    locked = ROOT / "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
    state = tmp_path / "run.log.launch"
    state.write_text(
        "pid=2468\npgid=2468\nboot_timeout_s=900\n", encoding="utf-8"
    )

    def timeout_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(124, [str(locked)])

    monkeypatch.setattr(M.subprocess, "run", timeout_run)
    with pytest.raises(M.ContractError, match="exact-PGID TERM-then-KILL"):
        M.run_locked_launcher(
            locked,
            tmp_path / "run.log",
            state,
            ["python", "train.py"],
            cwd=tmp_path,
            environment={},
            boot_timeout=900,
        )

    state.write_text(
        "pid=2468\npgid=9999\nboot_timeout_s=900\n", encoding="utf-8"
    )
    with pytest.raises(M.ContractError, match="without complete pid=pgid cleanup evidence"):
        M.run_locked_launcher(
            locked,
            tmp_path / "run.log",
            state,
            ["python", "train.py"],
            cwd=tmp_path,
            environment={},
            boot_timeout=900,
        )


def test_mixed_activation_binds_both_controls_and_retry_lineage():
    data = manifest()
    cells = {}
    for cell_id, source, lineage in (
        ("A", "original_v6", False),
        ("B", "original_v6", False),
        ("C", "original_v6", True),
        ("D", "v6r1_single_cell_retry", True),
    ):
        cells[cell_id] = {
            "source": source,
            "expected_lineage_exact": lineage,
            "training_contract_sha256": M.EXPECTED_HARD_CONTRACT_SHA256,
            "checkpoint_sha256": cell_id.lower() * 64,
            "launch_contract_sha256": ("1" if cell_id == "D" else "2") * 64,
            "runtime_verified_sha256": ("3" if cell_id == "D" else "4") * 64,
        }
    cells["D"]["locked_launcher_exact_pgid_boot_timeout_cleanup_executed"] = False
    content = M.build_mixed_activation_content(
        data, config_sha="5" * 64, launcher_sha="6" * 64, cells=cells
    )
    assert content["foreign_manifest_file_sha256"] == M.FOREIGN_CONFIG_SHA256
    assert content["foreign_launcher_file_sha256"] == M.FOREIGN_LAUNCHER_SHA256
    assert content["retry_lineage"]["old_run_name"] == M.OLD_RUN_NAME
    assert content["retry_lineage"]["new_run_name"] == M.NEW_RUN_NAME
    assert content["retry_lineage"]["only_command_change"] == "run_name"
    assert content["l2_training_launch_authorized"] is False
    assert content["automatic_judge_launch"] is False
    assert content["second_seed_authorized"] is False
    assert content["signal_policy"] == {
        "direct_signals_sent_by_retry_tool": False,
        "locked_launcher_sha256": M.LOCKED_LAUNCHER_SHA256,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "broad_signals_forbidden": True,
        "retry_d_boot_timeout_cleanup_executed": False,
    }


def test_no_clobber_writer_refuses_existing_file(tmp_path):
    path = tmp_path / "state.json"
    M.write_json_exclusive(path, {"first": True})
    with pytest.raises(FileExistsError):
        M.write_json_exclusive(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_retry_tool_has_no_signal_or_robot_command_path():
    source = LAUNCHER.read_text(encoding="utf-8")
    for forbidden in (
        "os.kill",
        "killpg",
        "signal.",
        "pkill",
        "killall",
        "pgrep -f",
    ):
        assert forbidden not in source
    assert "subprocess.Popen" not in source
    assert "forbidden = (\"ros2 \"" in source
    assert "automatic_judge_launch" in source
    assert "l2_training_launch_authorized" in source


def test_static_cli_requires_exact_self_hashes():
    config_sha = M.sha256_file(CONFIG)
    launcher_sha = M.sha256_file(LAUNCHER)
    assert (
        M.main(
            [
                "--config",
                str(CONFIG),
                "--expected-config-sha256",
                config_sha,
                "--expected-launcher-sha256",
                launcher_sha,
                "static-validate",
            ]
        )
        == 0
    )
    with pytest.raises(M.ContractError, match="manifest file SHA mismatch"):
        M.main(
            [
                "--config",
                str(CONFIG),
                "--expected-config-sha256",
                "0" * 64,
                "--expected-launcher-sha256",
                launcher_sha,
                "static-validate",
            ]
        )
