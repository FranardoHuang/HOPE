"""Dependency-light adversarial tests for the full-scene terminal gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load("lean_queue_runtime", SCRIPTS / "lean_queue_runtime.py")
P = _load("full_scene_probe_runtime_under_test", SCRIPTS / "full_scene_probe_runtime.py")


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Mask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return _Scalar(sum(self.values))


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def is_floating_point(_value):
        return True

    @staticmethod
    def is_complex(_value):
        return False

    @staticmethod
    def isfinite(value):
        return _Mask([math.isfinite(item) for item in value.values])


def _write_proc(proc_root: Path, pid: int, pgid: int, starttime: int, argv: list[str]):
    root = proc_root / str(pid)
    root.mkdir(parents=True, exist_ok=True)
    rest = ["S", *( ["0"] * 18), str(starttime)]
    rest[2] = str(pgid)
    (root / "stat").write_text(
        f"{pid} (probe process) " + " ".join(rest) + "\n", encoding="utf-8"
    )
    (root / "cmdline").write_bytes(
        b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    )
    return pgid


def _envelope(content):
    return {
        "schema_version": 1,
        "content": content,
        "content_sha256": R.canonical_sha256(content),
    }


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_runtime_asset(root: Path) -> None:
    (root / "urdf").mkdir(parents=True)
    (root / "meshes").mkdir()
    (root / "config").mkdir()
    (root / "meshes/body.STL").write_bytes(b"exact-a3-mesh\n")
    (root / "config/joints.yaml").write_text("joints: 31\n", encoding="utf-8")
    (root / "urdf/model.urdf").write_text(
        '<robot name="a3"><link name="body"><visual><geometry>'
        '<mesh filename="../meshes/body.STL"/>'
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, live=False, wrong_supervisor_argv=False):
    root = tmp_path / "case"
    source = root / "source"
    train = source / R.TRAIN_ENTRY_RELATIVE
    train.parent.mkdir(parents=True)
    train.write_text("# exact train entry\n", encoding="utf-8")
    run_dir = root / "runs/probe"
    run_dir.mkdir(parents=True)
    claim_path = run_dir / R.PROBE_CLAIM_NAME
    binding_path = run_dir / R.PROBE_BINDING_NAME
    run_name = "full_scene_probe_not_science_case"
    log_dir = (
        source
        / R.WBT_RELATIVE
        / "logs/rsl_rl/experiment"
        / f"2026-07-14_12-34-56_{run_name}"
    )
    (log_dir / "params").mkdir(parents=True)
    assets = root / "assets"
    assets.mkdir()
    motion0 = assets / "forehand.npz"
    motion1 = assets / "backhand.npz"
    bank = assets / "train.npz"
    motion0.write_bytes(b"motion-forehand")
    motion1.write_bytes(b"motion-backhand")
    bank.write_bytes(b"bank")
    donor_checkout = root / "donor"
    donor_asset = donor_checkout / "assets/agibot_a3"
    _make_runtime_asset(donor_asset)
    target_asset = source / "runtime_assets/agibot_a3"
    target_asset.parent.mkdir(parents=True)
    shutil.copytree(donor_asset, target_asset)
    asset_inventory = P._stable_asset_inventory(donor_asset, "fixture donor asset")
    asset_urdf = P._asset_urdf_reference_closure(donor_asset, "fixture donor asset")
    asset_contract = {
        "target_relative_path": "runtime_assets/agibot_a3",
        "donor": {
            "checkout": str(donor_checkout),
            "commit": "b" * 40,
            "relative_path": "assets/agibot_a3",
        },
        **asset_inventory,
        "symlinks_forbidden": True,
        "target_must_be_gitignored": True,
    }
    source_asset_receipt = root / "source_asset_receipts/receipt.json"
    source_asset_receipt.parent.mkdir(parents=True)
    argv_without_claim = [
        "/exact/python",
        str(train),
        "task=Task",
        "algo=ppo",
        "task.actor_obs_contract=deploy_parity_face179",
        "task.plant.zero_joint_friction=true",
        "++task.physical_ball=true",
        "num_envs=4096",
        f"motion_file={motion0}",
        f"motion_file_2={motion1}",
        f"++task.racket.question_bank={bank}",
        f"run_name={run_name}",
        f"++training_queue_claim_path={claim_path}",
        f"++training_run_binding_path={binding_path}",
    ]
    supervisor_prefix = [
        "/exact/python",
        str(SCRIPTS / "full_scene_probe_runtime.py"),
        "supervise",
        "--run-dir",
        str(run_dir),
        "--log",
        str(run_dir / "run.log"),
        "--",
    ]
    content = {
        "schema_version": 1,
        "purpose": R.PROBE_PURPOSE,
        "not_science": True,
        "attestable": False,
        "promotable": False,
        "job_id": "job0",
        "pod": "pod2",
        "gpu": 1,
        "source": {
            "checkout": str(source),
            "commit": "a" * 40,
            "ignored_runtime_asset": asset_contract,
        },
        "source_asset_receipt_path": str(source_asset_receipt),
        "supervisor_argv_prefix": supervisor_prefix,
        "expected_training_contract_lineage_exact": 1,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "budget": {
            "num_envs": 4096,
            "max_iterations": 2,
            "save_interval": 1,
            "milestones": [1],
        },
        "inputs": {
            "motion": {
                "action": "paired",
                "bindings": {"motion_file": str(motion0), "motion_file_2": str(motion1)},
            },
            "bank": {"train_path": str(bank), "train_arg": "++task.racket.question_bank"},
            "exam": {"path": str(assets / "exam.npz")},
        },
        "training_argv_without_claim": argv_without_claim,
    }
    claim_digest = R.canonical_sha256(content)
    full_argv = [*argv_without_claim, f"++training_launch_claim_sha256={claim_digest}"]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": claim_digest,
        "training_argv": full_argv,
    }
    _write_json(claim_path, claim)
    source_asset_content = {
        "schema_version": 1,
        "pod": "pod2",
        "source": {"checkout": str(source), "commit": "a" * 40},
        "ignored_runtime_asset": asset_contract,
        "ignored_runtime_asset_sha256": R.canonical_sha256(asset_contract),
        "target_path": str(source / asset_contract["target_relative_path"]),
        "inventory": {
            "file_count": asset_contract["file_count"],
            "total_file_bytes": asset_contract["total_file_bytes"],
            "tree_content_sha256": asset_contract["tree_content_sha256"],
        },
        "urdf_reference_closure": asset_urdf,
        "target_gitignored": True,
        "symlinks_present": False,
    }
    _write_json(source_asset_receipt, _envelope(source_asset_content))
    proc_root = root / "proc"
    supervisor_pid = 41000
    trainer_pid = 41001
    supervisor_start = 7000
    trainer_start = 7001
    supervisor_argv = (
        ["/exact/unbound-wrapper", "probe"]
        if wrong_supervisor_argv
        else [*supervisor_prefix, *full_argv]
    )
    _write_proc(proc_root, supervisor_pid, supervisor_pid, supervisor_start, supervisor_argv)
    _write_proc(proc_root, trainer_pid, supervisor_pid, trainer_start, full_argv)
    pgids = {supervisor_pid: supervisor_pid, trainer_pid: supervisor_pid}
    binding = R.publish_run_binding(
        claim_path=claim_path,
        binding_path=binding_path,
        log_dir=log_dir,
        claim_digest=claim_digest,
        actual_argv=full_argv,
        pid=trainer_pid,
        proc_root=proc_root,
        getpgid=lambda pid: pgids[pid],
        environ={"CUDA_VISIBLE_DEVICES": "1"},
        source_verifier=lambda _path, commit: {"head": commit, "clean": True},
    )
    hard = {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"index": 0, "basename": motion0.name, "sha256": hashlib.sha256(motion0.read_bytes()).hexdigest()},
            {"index": 1, "basename": motion1.name, "sha256": hashlib.sha256(motion1.read_bytes()).hexdigest()},
        ],
        "question_bank": {"sha256": hashlib.sha256(bank.read_bytes()).hexdigest()},
    }
    hard_path = log_dir / "params/training_contract.json"
    hard_path.write_text(json.dumps(hard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hard_sha = hashlib.sha256(hard_path.read_bytes()).hexdigest()
    log_lines = [
        P.PHASE_PREFIX + json.dumps({"phase": "scene_import_start"}, separators=(",", ":")),
        P.PHASE_PREFIX + json.dumps(
            {
                "phase": "scene_import_done",
                "actual_num_envs": 4096,
                "physical_ball_enabled": True,
                "physical_scene_entities": {
                    "pb_ball": True,
                    "pb_table": True,
                    "pb_table_visual": True,
                },
            },
            separators=(",", ":"),
        ),
        P.PHASE_PREFIX + json.dumps(
            {"phase": "hard_contract_written", "sha256": hard_sha}, separators=(",", ":")
        ),
        "Learning iteration 1/2",
        P.SUPERVISOR_PHASE_PREFIX + json.dumps(
            {"phase": "first_iteration_observed"}, separators=(",", ":")
        ),
    ]
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    checkpoint_path = log_dir / "model_1.pt"
    checkpoint_path.write_bytes(b"stable model one")
    checkpoint = {
        "iter": 1,
        "model_state_dict": {"weight": FakeTensor([1.0, 2.0])},
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": hard_sha,
            "training_contract_lineage_exact": 1,
            "training_launch_claim_sha256": claim_digest,
        },
    }
    process = binding["content"]["process"]
    supervisor = binding["content"]["supervisor_process"]
    exit_content = {
        "schema_version": 1,
        "purpose": R.PROBE_PURPOSE,
        "claim_path": str(claim_path),
        "claim_content_sha256": claim_digest,
        "binding_path": str(binding_path),
        "binding_content_sha256": binding["content_sha256"],
        "log_path": str(run_dir / "run.log"),
        "supervisor_process": supervisor,
        "trainer_process": process,
        "first_iteration_observed": True,
        "termination": {"kind": "normal_exit", "exit_code": 0},
    }
    _write_json(run_dir / P.EXIT_NAME, _envelope(exit_content))
    if not live:
        shutil.rmtree(proc_root / str(supervisor_pid))
        shutil.rmtree(proc_root / str(trainer_pid))
    return {
        "run_dir": run_dir,
        "source": source,
        "proc_root": proc_root,
        "pgids": pgids,
        "supervisor_pid": supervisor_pid,
        "trainer_pid": trainer_pid,
        "supervisor_start": supervisor_start,
        "trainer_start": trainer_start,
        "supervisor_argv": supervisor_argv,
        "full_argv": full_argv,
        "checkpoint": checkpoint,
        "hard_path": hard_path,
        "log_path": run_dir / "run.log",
        "exit_path": run_dir / P.EXIT_NAME,
        "claim_digest": claim_digest,
        "source_asset_receipt": source_asset_receipt,
        "target_asset": target_asset,
        "donor_asset": donor_asset,
        "asset_inventory": asset_inventory,
        "asset_urdf": asset_urdf,
    }


def _finalize(fixture, **kwargs):
    return P.finalize(
        fixture["run_dir"],
        expected_claim_digest=fixture["claim_digest"],
        source_asset_receipt=fixture["source_asset_receipt"],
        checkpoint_loader=lambda _path: fixture["checkpoint"],
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda pid: fixture["pgids"].get(pid, pid),
        source_verifier=kwargs.pop(
            "source_verifier", lambda _path, commit: {"head": commit, "clean": True}
        ),
        hard_contract_validator=kwargs.pop(
            "hard_contract_validator", lambda _contract, _source: None
        ),
        **kwargs,
    )


def _rewrite_exit(fixture, mutate):
    value = json.loads(fixture["exit_path"].read_text(encoding="utf-8"))
    mutate(value["content"])
    value["content_sha256"] = R.canonical_sha256(value["content"])
    _write_json(fixture["exit_path"], value)


def _write_launcher_terminal(
    fixture, *, terminal_kind="pre_marker_exit", terminal_exit_code=7
):
    fixture["exit_path"].unlink()
    state_path = fixture["run_dir"] / "run.log.launch"
    leader_path = Path(str(state_path) + ".leader.json")
    leader = {
        "pid": fixture["supervisor_pid"],
        "pgid": fixture["supervisor_pid"],
        "starttime_ticks": fixture["supervisor_start"],
    }
    _write_json(
        leader_path,
        {"schema_version": 1, "kind": "leader_identity", "leader": leader},
    )
    state_path.write_text(
        "\n".join(
            [
                f"pid={leader['pid']}",
                f"pgid={leader['pgid']}",
                f"leader_starttime_ticks={leader['starttime_ticks']}",
                f"leader_identity_evidence={leader_path}",
                f"terminal_kind={terminal_kind}",
                f"terminal_exit_code={terminal_exit_code}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return state_path


def test_happy_terminal_pass_and_identical_repeat(tmp_path):
    fixture = _fixture(tmp_path)
    first = _finalize(fixture)
    content = first["result"]["content"]
    assert content["status"] == "passed"
    assert content["unlock_authorized"] is True
    assert content["not_science"] is True
    current = content["source_asset_receipt"]["current_closure"]
    assert current["target"]["inventory"] == fixture["asset_inventory"]
    assert current["donor"]["inventory"] == fixture["asset_inventory"]
    assert current["target"]["urdf_reference_closure"] == fixture["asset_urdf"]
    assert current["donor"]["urdf_reference_closure"] == fixture["asset_urdf"]
    assert first["repeated_identical"] is False
    frozen = (fixture["run_dir"] / P.RESULT_NAME).read_bytes()
    second = _finalize(fixture)
    assert second["repeated_identical"] is True
    assert (fixture["run_dir"] / P.RESULT_NAME).read_bytes() == frozen


def test_binding_rejects_unclaimed_supervisor_wrapper(tmp_path):
    with pytest.raises(R.LeanQueueRuntimeError, match="supervisor argv differs"):
        _fixture(tmp_path, wrong_supervisor_argv=True)


def test_still_live_is_not_ready_and_writes_no_result(tmp_path):
    fixture = _fixture(tmp_path, live=True)
    with pytest.raises(P.FullSceneProbeNotReady, match="still live"):
        _finalize(fixture)
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_orphan_in_original_process_group_is_not_ready(tmp_path):
    fixture = _fixture(tmp_path)
    orphan_pid = 42000
    _write_proc(
        fixture["proc_root"],
        orphan_pid,
        fixture["supervisor_pid"],
        8800,
        ["/exact/orphan-gpu-child"],
    )
    with pytest.raises(P.FullSceneProbeNotReady, match="still has live members"):
        _finalize(fixture)
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_reused_pid_proves_original_identity_absent(tmp_path):
    fixture = _fixture(tmp_path)
    _write_proc(
        fixture["proc_root"], fixture["trainer_pid"], fixture["trainer_pid"],
        fixture["trainer_start"] + 1, fixture["full_argv"],
    )
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "passed"


def test_process_vanishing_during_identity_read_is_natural_absence(tmp_path):
    proc_root = tmp_path / "proc"
    pid = 51000
    _write_proc(proc_root, pid, pid, 9000, ["/exact/probe"])

    def vanish_then_lookup(_pid):
        shutil.rmtree(proc_root / str(pid))
        raise ProcessLookupError(pid)

    P._require_naturally_absent(
        {"pid": pid, "pgid": pid, "starttime_ticks": 9000},
        "bound trainer",
        proc_root=proc_root,
        getpgid=vanish_then_lookup,
    )


def test_launcher_terminal_without_exit_receipt_freezes_failure_only(tmp_path):
    fixture = _fixture(tmp_path)
    _write_launcher_terminal(fixture)
    result = _finalize(fixture)
    content = result["result"]["content"]
    assert content["status"] == "failed"
    assert content["unlock_authorized"] is False
    assert content["failure_type"] == "LauncherTerminalFailure"
    assert content["automatic_retry_authorized"] is False


def test_nonterminal_launcher_state_remains_not_ready(tmp_path):
    fixture = _fixture(tmp_path)
    state_path = _write_launcher_terminal(fixture)
    lines = [
        line
        for line in state_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("terminal_")
    ]
    state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(P.FullSceneProbeNotReady, match="recognized terminal"):
        _finalize(fixture)
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


@pytest.mark.parametrize(
    "termination",
    [
        {"kind": "normal_exit", "exit_code": 7},
        {"kind": "signal", "signal": 9},
    ],
)
def test_nonzero_or_signal_exit_cannot_pass(tmp_path, termination):
    fixture = _fixture(tmp_path)
    _rewrite_exit(fixture, lambda content: content.__setitem__("termination", termination))
    result = _finalize(fixture)
    content = result["result"]["content"]
    assert content["status"] == "failed" and content["unlock_authorized"] is False
    assert content["automatic_retry_authorized"] is False
    assert content["terminal_evidence"]["exit_receipt"]["termination"] == termination


def test_fatal_log_marker_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write("Traceback (most recent call last):\n")
    result = _finalize(fixture)
    assert "fatal markers" in result["result"]["content"]["failure_reason"]


def test_missing_model_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    next(fixture["hard_path"].parent.parent.glob("model_1.pt")).unlink()
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "failed"
    assert "checkpoint is missing" in result["result"]["content"]["failure_reason"]


def test_nan_checkpoint_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["model_state_dict"]["weight"] = FakeTensor([float("nan")])
    result = _finalize(fixture)
    assert "non-finite" in result["result"]["content"]["failure_reason"]


def test_embedded_iteration_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["iter"] = 0
    result = _finalize(fixture)
    assert "filename iteration differs" in result["result"]["content"]["failure_reason"]


def test_boolean_embedded_iteration_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["iter"] = True
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "failed"
    assert "embedded iteration must be an integer" in result["result"]["content"]["failure_reason"]


def test_contract_sha_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["infos"]["training_contract_sha256"] = "0" * 64
    result = _finalize(fixture)
    assert "hard-contract SHA" in result["result"]["content"]["failure_reason"]


def test_actual_scene_scale_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    text = fixture["log_path"].read_text(encoding="utf-8")
    fixture["log_path"].write_text(
        text.replace('"actual_num_envs":4096', '"actual_num_envs":1'),
        encoding="utf-8",
    )
    result = _finalize(fixture)
    assert "actual scene num_envs 1 differs" in result["result"]["content"]["failure_reason"]


def test_missing_physical_scene_entity_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    text = fixture["log_path"].read_text(encoding="utf-8")
    fixture["log_path"].write_text(
        text.replace('"pb_ball":true', '"pb_ball":false'), encoding="utf-8"
    )
    result = _finalize(fixture)
    assert "physical scene entity inventory differs" in result["result"]["content"]["failure_reason"]


def test_nonzero_joint_friction_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    hard = json.loads(fixture["hard_path"].read_text(encoding="utf-8"))
    hard["joint_friction_coefficients"][3] = 0.01
    fixture["hard_path"].write_text(
        json.dumps(hard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    new_sha = hashlib.sha256(fixture["hard_path"].read_bytes()).hexdigest()
    fixture["checkpoint"]["infos"]["training_contract_sha256"] = new_sha
    text = fixture["log_path"].read_text(encoding="utf-8")
    old_sha = next(
        json.loads(line.split(P.PHASE_PREFIX, 1)[1])["sha256"]
        for line in text.splitlines()
        if P.PHASE_PREFIX in line and '"hard_contract_written"' in line
    )
    fixture["log_path"].write_text(text.replace(old_sha, new_sha), encoding="utf-8")
    result = _finalize(fixture)
    assert "31/31 zero PhysX joint friction" in result["result"]["content"]["failure_reason"]


def test_formal_schema3_validator_is_mandatory(tmp_path):
    fixture = _fixture(tmp_path)

    def reject(_contract, _source):
        raise ValueError("missing official execution field")

    result = _finalize(fixture, hard_contract_validator=reject)
    assert "failed formal schema-3 validation" in result["result"]["content"]["failure_reason"]


def test_formal_validator_direct_load_does_not_import_kit_package(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    relative = (
        R.WBT_RELATIVE
        / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
    )
    target = source / relative
    target.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py",
        target,
    )
    import builtins

    original_import = builtins.__import__

    def no_kit_package(name, *args, **kwargs):
        if name == "whole_body_tracking" or name.startswith(
            ("whole_body_tracking.", "omni.", "isaaclab.")
        ):
            raise AssertionError(f"forbidden package import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_kit_package)
    with pytest.raises(ValueError):
        P._formal_hard_contract_validator({"schema_version": 3}, source)


def test_claim_binding_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    _rewrite_exit(
        fixture,
        lambda content: content.__setitem__("claim_content_sha256", "0" * 64),
    )
    result = _finalize(fixture)
    assert "claim_content_sha256 differs" in result["result"]["content"]["failure_reason"]


def test_terminal_source_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    result = _finalize(
        fixture,
        source_verifier=lambda _path, commit: {"head": commit, "clean": False},
    )
    assert "exact clean source" in result["result"]["content"]["failure_reason"]


def test_terminal_donor_source_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    result = _finalize(
        fixture,
        source_verifier=lambda _path, commit: {
            "head": commit,
            "clean": commit != "b" * 40,
        },
    )
    content = result["result"]["content"]
    assert content["status"] == "failed"
    assert content["unlock_authorized"] is False
    assert "donor verifier did not prove exact clean source" in content["failure_reason"]


def test_ignored_source_asset_receipt_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = json.loads(fixture["source_asset_receipt"].read_text(encoding="utf-8"))
    receipt["content"]["inventory"]["file_count"] -= 1
    receipt["content_sha256"] = R.canonical_sha256(receipt["content"])
    _write_json(fixture["source_asset_receipt"], receipt)
    result = _finalize(fixture)
    assert "inventory mismatch" in result["result"]["content"]["failure_reason"]


@pytest.mark.parametrize("which", ["target_asset", "donor_asset"])
def test_direct_finalizer_rehashes_current_ignored_asset_tree(tmp_path, which):
    """Calling runtime finalize directly must not bypass terminal A3 verification."""

    fixture = _fixture(tmp_path)
    (fixture[which] / "meshes/body.STL").write_bytes(b"drift-after-hydration\n")
    result = P.finalize(
        fixture["run_dir"],
        expected_claim_digest=fixture["claim_digest"],
        # Deliberately call the terminal authority directly: no queue shell
        # wrapper or pre-finalize source-asset doctor participates.
        checkpoint_loader=lambda _path: fixture["checkpoint"],
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda pid: fixture["pgids"].get(pid, pid),
        source_verifier=lambda _path, commit: {"head": commit, "clean": True},
        hard_contract_validator=lambda _contract, _source: None,
    )
    content = result["result"]["content"]
    assert content["status"] == "failed"
    assert content["unlock_authorized"] is False
    expected = "target" if which == "target_asset" else "donor"
    assert f"current {expected} asset tree inventory drift" in content["failure_reason"]


def test_causal_lineage_cannot_unlock_fresh_probe(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["infos"]["training_contract_lineage_exact"] = 0
    result = _finalize(fixture)
    assert "lineage must equal 1" in result["result"]["content"]["failure_reason"]


def test_boolean_lineage_cannot_unlock_fresh_probe(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["infos"]["training_contract_lineage_exact"] = True
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "failed"
    assert "lineage must equal 1" in result["result"]["content"]["failure_reason"]


def test_current_queue_claim_drift_refuses_without_freezing_result(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(P.FullSceneProbeError, match="selected queue row differs"):
        P.finalize(
            fixture["run_dir"],
            expected_claim_digest="0" * 64,
            source_asset_receipt=fixture["source_asset_receipt"],
            checkpoint_loader=lambda _path: fixture["checkpoint"],
            torch_module=FakeTorch,
            proc_root=fixture["proc_root"],
            getpgid=lambda pid: fixture["pgids"].get(pid, pid),
            source_verifier=lambda _path, commit: {"head": commit, "clean": True},
        )
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_corrupt_checkpoint_loader_becomes_auditable_terminal_failure(tmp_path):
    fixture = _fixture(tmp_path)

    def corrupt(_path):
        raise RuntimeError("corrupt checkpoint archive")

    result = P.finalize(
        fixture["run_dir"],
        expected_claim_digest=fixture["claim_digest"],
        source_asset_receipt=fixture["source_asset_receipt"],
        checkpoint_loader=corrupt,
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda pid: fixture["pgids"].get(pid, pid),
        source_verifier=lambda _path, commit: {"head": commit, "clean": True},
        hard_contract_validator=lambda _contract, _source: None,
    )
    assert result["result"]["content"]["status"] == "failed"
    assert "corrupt checkpoint archive" in result["result"]["content"]["failure_reason"]


@pytest.mark.parametrize("marker", ["NaN", "Inf", "Killed"])
def test_additional_fatal_markers_cannot_pass(tmp_path, marker):
    fixture = _fixture(tmp_path)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write(marker + "\n")
    result = _finalize(fixture)
    assert "fatal markers" in result["result"]["content"]["failure_reason"]


def test_existing_result_rejects_different_recomputed_bytes(tmp_path):
    fixture = _fixture(tmp_path)
    _finalize(fixture)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write("Traceback (most recent call last):\n")
    with pytest.raises(P.FullSceneProbeError, match="different bytes"):
        _finalize(fixture)


def test_wrong_receipt_selector_does_not_burn_result(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(P.FullSceneProbeError, match="differs from immutable claim"):
        P.finalize(
            fixture["run_dir"],
            expected_claim_digest=fixture["claim_digest"],
            source_asset_receipt=tmp_path / "wrong-receipt.json",
        )
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_concurrent_identical_publication_accepts_atomic_winner(tmp_path, monkeypatch):
    target = tmp_path / "probe_result.json"
    value = {"schema_version": 1, "content": {"status": "failed"}}
    original = P.queue_runtime._atomic_publish_json

    def lose_race(path, document, label):
        original(path, document, label)
        raise P.queue_runtime.LeanQueueRuntimeError("target appeared")

    monkeypatch.setattr(P.queue_runtime, "_atomic_publish_json", lose_race)
    assert P._publish_or_accept_identical(target, value, "probe result") is True


def test_supervisor_and_finalizer_source_have_no_signal_operation():
    source = (SCRIPTS / "full_scene_probe_runtime.py").read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert ".kill(" not in source
    assert ".terminate(" not in source
    assert "automatic_retry_authorized" in source


def test_ordinary_attestor_refuses_probe_binding(tmp_path):
    fixture = _fixture(tmp_path, live=True)
    with pytest.raises(R.LeanQueueRuntimeError, match="refuses non-science"):
        R.attest_milestone(
            fixture["run_dir"] / R.PROBE_BINDING_NAME,
            1,
            checkpoint_loader=lambda _path: fixture["checkpoint"],
            torch_module=FakeTorch,
            proc_root=fixture["proc_root"],
            getpgid=lambda pid: fixture["pgids"][pid],
        )
