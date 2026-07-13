"""Dependency-light contract tests for the C2 evidence / D2-only v1r2."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/continue_phase1_signed_face_cd_l1_v1r2.py"
MANIFEST = ROOT / "configs/phase1_signed_face_cd_l1_v1r2_continuation_20260714.json"
V1_MANIFEST = ROOT / "configs/phase1_signed_face_cd_l1_prereg_20260714.json"
V1R1_SCRIPT = ROOT / "scripts/continue_phase1_signed_face_cd_l1_v1r1.py"
V1R1_MANIFEST = ROOT / "configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json"
SPEC = importlib.util.spec_from_file_location("signed_face_cd_l1_v1r2", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def continuation_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def original_manifest():
    return json.loads(V1_MANIFEST.read_text(encoding="utf-8"))


def make_external_mini_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix=".v1r2-mini-", dir=ROOT))
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    files = {
        SCRIPT: root / "scripts/continue_phase1_signed_face_cd_l1_v1r2.py",
        ROOT / "scripts/run_phase1_signed_face_cd_l1.py": root / "scripts/run_phase1_signed_face_cd_l1.py",
        V1R1_SCRIPT: root / "scripts/continue_phase1_signed_face_cd_l1_v1r1.py",
        MANIFEST: root / "configs/phase1_signed_face_cd_l1_v1r2_continuation_20260714.json",
        V1_MANIFEST: root / "configs/phase1_signed_face_cd_l1_prereg_20260714.json",
        V1R1_MANIFEST: root / "configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json",
    }
    for source, target in files.items():
        shutil.copyfile(source, target)
        target.chmod(0o555 if target.name.endswith(".py") else 0o444)
    return root


def run_external(root: Path, mode: str):
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/continue_phase1_signed_face_cd_l1_v1r2.py"),
            "--mode", mode,
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def hard_contract(original, cell_id, signs=None):
    return {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0] if signs is None else signs,
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "racket_guidance_reward": {
            "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
            "signed_face": {
                "weight": mod.v1.cells(original)[cell_id]["face_guidance_weight"],
                "command_name": "racket_target",
                "theta_max": 3.141592653589793,
            },
        },
        "joint_names": [f"joint_{index}" for index in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": original["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": original["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": {
            "sha256": original["inputs"]["schema3_train_bank"]["sha256"],
            "schema_version": 3,
            "split": "train",
            "source_family_sha256": original["inputs"]["schema3_train_bank"]["source_family_sha256"],
            "exact": True,
        },
    }


def bank_metadata(original):
    bank = original["inputs"]["schema3_train_bank"]
    return {
        "sha256": bank["sha256"],
        "source_family_sha256": bank["source_family_sha256"],
        "physics_contract_sha256": bank["physics_contract_sha256"],
    }


def test_checked_in_manifest_and_original_v1_helper_are_content_bound():
    manifest = mod.read_manifest(MANIFEST)
    checked = mod.verify_checked_in_original()
    assert manifest["continuation_id"] == mod.CONTINUATION_ID
    assert checked["source"]["commit"] == mod.v1.TRAINING_COMMIT
    assert mod.v1.sha256_file(mod.V1_SCRIPT) == mod.V1_LAUNCHER_SHA256
    assert mod.v1.sha256_file(mod.V1R1_SCRIPT) == mod.V1R1_LAUNCHER_SHA256
    assert manifest["continuation_control"]["v1_helper_sha256"] == mod.V1_LAUNCHER_SHA256
    assert checked["frozen_v1r1"]["manifest_sha256"] == mod.V1R1_MANIFEST_SHA256


def test_v1r2_manifest_rejects_duplicate_nested_json_key(tmp_path, monkeypatch):
    raw = MANIFEST.read_text(encoding="utf-8")
    raw = raw.replace(
        '"face_guidance_weight": -0.4,',
        '"face_guidance_weight": -0.4,\n    "face_guidance_weight": 0.0,',
        1,
    )
    path = tmp_path / "duplicate-manifest.json"
    path.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(mod, "CONTINUATION_MANIFEST_SHA256", mod.v1.sha256_file(path))
    with pytest.raises(mod.ContractError, match="duplicate JSON key: face_guidance_weight"):
        mod.read_manifest(path)


def test_v1r2_owned_attestation_reader_rejects_duplicate_json_key(tmp_path):
    path = tmp_path / "duplicate-attestation.json"
    path.write_text('{"artifact_kind":"first","artifact_kind":"second"}\n', encoding="utf-8")
    with pytest.raises(mod.ContractError, match="duplicate JSON key: artifact_kind"):
        mod.read_json_object_strict(path, "test attestation")


def test_external_six_file_mini_tree_static_validate_and_plan():
    root = make_external_mini_tree()
    try:
        static = run_external(root, "static-validate")
        plan = run_external(root, "plan")
        assert static.returncode == 0, static.stderr
        assert plan.returncode == 0, plan.stderr
        observed = json.loads(plan.stdout)
        assert observed["only_launchable_cell"] == "D2"
        assert observed["c2_launch_or_retry_mode_present"] is False
    finally:
        shutil.rmtree(root)


def test_runtime_receipt_binds_all_six_mini_tree_files(monkeypatch):
    root = make_external_mini_tree()
    try:
        manifest = continuation_manifest()
        manifest["continuation_control"]["root"] = str(root)
        config = root / manifest["continuation_control"]["manifest_relative_path"]
        launcher = root / manifest["continuation_control"]["launcher_relative_path"]
        helper = root / manifest["continuation_control"]["v1_helper_relative_path"]
        v1r1_helper = root / manifest["continuation_control"]["v1r1_helper_relative_path"]
        v1r1_manifest = root / manifest["continuation_control"]["v1r1_manifest_relative_path"]
        monkeypatch.setattr(mod, "ROOT", root)
        monkeypatch.setattr(mod, "V1_SCRIPT", helper)
        monkeypatch.setattr(mod, "V1R1_SCRIPT", v1r1_helper)
        monkeypatch.setattr(mod, "V1R1_MANIFEST", v1r1_manifest)
        receipt = mod.continuation_control_receipt(manifest, config, launcher)
        assert set(receipt) == {
            "manifest", "launcher", "v1_helper", "v1_manifest",
            "v1r1_helper", "v1r1_manifest",
        }
        assert receipt["manifest"]["sha256"] == mod.CONTINUATION_MANIFEST_SHA256
        assert receipt["v1_helper"]["sha256"] == mod.V1_LAUNCHER_SHA256
        assert receipt["v1_manifest"]["sha256"] == mod.V1_MANIFEST_SHA256
        assert receipt["v1r1_helper"]["sha256"] == mod.V1R1_LAUNCHER_SHA256
        assert receipt["v1r1_manifest"]["sha256"] == mod.V1R1_MANIFEST_SHA256
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize(
    "missing",
    [
        "scripts/continue_phase1_signed_face_cd_l1_v1r2.py",
        "scripts/run_phase1_signed_face_cd_l1.py",
        "scripts/continue_phase1_signed_face_cd_l1_v1r1.py",
        "configs/phase1_signed_face_cd_l1_v1r2_continuation_20260714.json",
        "configs/phase1_signed_face_cd_l1_prereg_20260714.json",
        "configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json",
    ],
)
def test_external_mini_tree_missing_any_frozen_file_fails(missing):
    root = make_external_mini_tree()
    try:
        (root / missing).unlink()
        assert run_external(root, "static-validate").returncode != 0
    finally:
        shutil.rmtree(root)


def test_flat_old_external_layout_fails_instead_of_escaping_to_repo():
    root = Path(tempfile.mkdtemp(prefix=".v1r2-flat-", dir=ROOT))
    try:
        for source in (
            SCRIPT, ROOT / "scripts/run_phase1_signed_face_cd_l1.py", V1R1_SCRIPT,
            MANIFEST, V1_MANIFEST, V1R1_MANIFEST,
        ):
            shutil.copyfile(source, root / source.name)
        completed = subprocess.run(
            [sys.executable, str(root / SCRIPT.name), "--mode", "static-validate"],
            cwd=root, text=True, capture_output=True, check=False,
        )
        assert completed.returncode != 0
        assert "must be installed under control/v1r2/scripts" in completed.stderr
    finally:
        shutil.rmtree(root)


def test_external_mini_tree_rejects_symlinked_helper():
    root = make_external_mini_tree()
    try:
        helper = root / "scripts/run_phase1_signed_face_cd_l1.py"
        helper.unlink()
        helper.symlink_to(ROOT / "scripts/run_phase1_signed_face_cd_l1.py")
        completed = run_external(root, "static-validate")
        assert completed.returncode != 0
        assert "symlink component" in completed.stderr
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("unsafe", ["", ".", "../x", "configs/../x", "/abs/x"])
def test_control_relative_paths_reject_empty_dot_traversal_and_absolute(unsafe):
    with pytest.raises(mod.ContractError):
        mod.require_safe_relative_path(unsafe, "test path")


def test_plan_has_only_d2_launch_and_no_c2_retry_surface():
    manifest = mod.read_manifest(MANIFEST)
    plan = mod.build_plan(manifest, MANIFEST, SCRIPT)
    assert plan["preserved_c2_action"] == "verify_exact_terminal_evidence_only"
    assert plan["only_launchable_cell"] == "D2"
    assert plan["c2_launch_or_retry_mode_present"] is False
    assert plan["writes_or_launches_performed"] is False
    actions = set(mod.parser()._option_string_actions["--mode"].choices)
    assert actions == {
        "plan", "static-validate", "validate-runtime", "attest-c2",
        "launch-d2", "finalize-d2", "finalize-pair",
    }
    assert "--cell" not in mod.parser()._option_string_actions
    assert not ({"launch-c2", "launch-next", "retry"} & actions)


def test_exact_float_mount_signs_accept_wire_value_and_reject_bool_or_int():
    assert mod.require_exact_float_mount_signs([1.0, -1.0]) == [1.0, -1.0]
    for invalid in ([True, -1.0], [1, -1], [1.0, -1], [1.0, 1.0]):
        with pytest.raises(mod.ContractError):
            mod.require_exact_float_mount_signs(invalid)


def test_hard_contract_accepts_emitted_floats_and_rejects_bool_int(tmp_path):
    original = original_manifest()
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps(hard_contract(original, "C2")), encoding="utf-8")
    _, observed = mod.verify_hard_contract(
        accepted, original, "C2", bank_metadata=bank_metadata(original)
    )
    assert observed["mount_normal_sign_per_clip"] == [1.0, -1.0]
    for index, signs in enumerate(([True, -1.0], [1, -1])):
        rejected = tmp_path / f"rejected-{index}.json"
        rejected.write_text(json.dumps(hard_contract(original, "C2", signs)), encoding="utf-8")
        with pytest.raises(mod.ContractError, match="exact floats"):
            mod.verify_hard_contract(
                rejected, original, "C2", bank_metadata=bank_metadata(original)
            )


def test_exact_five_key_bank_passes_v1r2_and_reproduces_v1r1_rejection(tmp_path):
    original = original_manifest()
    path = tmp_path / "training_contract.json"
    path.write_text(json.dumps(hard_contract(original, "C2")), encoding="utf-8")
    _, observed = mod.verify_hard_contract(
        path, original, "C2", bank_metadata=bank_metadata(original)
    )
    assert set(observed["question_bank"]) == {
        "sha256", "schema_version", "split", "source_family_sha256", "exact"
    }
    failure = mod.reproduce_exact_v1r1_false_rejection(path, original, "C2")
    assert failure["exact_message"] == mod.V1R1_FALSE_REJECTION
    assert failure["classification"] == "v1r1_outer_verifier_false_rejection_only"


def test_fabricated_sixth_compact_bank_leaf_fails_exact_shape(tmp_path):
    original = original_manifest()
    contract = hard_contract(original, "C2")
    contract["question_bank"]["physics_contract_sha256"] = (
        original["inputs"]["schema3_train_bank"]["physics_contract_sha256"]
    )
    path = tmp_path / "fabricated-six-key.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(mod.ContractError, match="exact trainer-emitted five-key"):
        mod.verify_hard_contract(
            path, original, "C2", bank_metadata=bank_metadata(original)
        )


def make_bank(tmp_path: Path):
    physics = "1" * 64
    family = {"artifact_kind": "test_family", "physics_contract_sha256": physics}
    family_sha = mod.v1.canonical_sha256(family)
    metadata = {
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": family_sha,
        "source_family_contract": family,
        "physics_contract_sha256": physics,
    }
    path = tmp_path / "bank.npz"
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    np.savez(path, meta_json=np.frombuffer(raw, dtype=np.uint8))
    expected = {
        "sha256": mod.v1.sha256_file(path),
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": family_sha,
        "physics_contract_sha256": physics,
        "source_family_contract_physics_must_match": True,
    }
    return path, expected


def test_bank_physics_is_independently_bound_from_exact_npz_metadata(tmp_path):
    path, expected = make_bank(tmp_path)
    receipt = mod.verify_schema3_bank_metadata(path, expected)
    assert receipt["physics_contract_sha256"] == expected["physics_contract_sha256"]
    wrong = {**expected, "physics_contract_sha256": "2" * 64}
    with pytest.raises(mod.ContractError, match="bank metadata physics_contract_sha256"):
        mod.verify_schema3_bank_metadata(path, wrong)
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(mod.ContractError, match="file SHA"):
        mod.verify_schema3_bank_metadata(path, expected)


def test_bank_metadata_rejects_duplicate_physics_key(tmp_path):
    physics = "1" * 64
    family = {"artifact_kind": "test_family", "physics_contract_sha256": physics}
    family_sha = mod.v1.canonical_sha256(family)
    raw = (
        '{"schema_version":3,"split":"train",'
        f'"source_family_sha256":"{family_sha}",'
        f'"physics_contract_sha256":"{physics}",'
        f'"physics_contract_sha256":"{"2" * 64}",'
        f'"source_family_contract":{json.dumps(family, separators=(",", ":"))}'
        '}'
    ).encode()
    path = tmp_path / "duplicate-bank.npz"
    np.savez(path, meta_json=np.frombuffer(raw, dtype=np.uint8))
    expected = {
        "sha256": mod.v1.sha256_file(path),
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": family_sha,
        "physics_contract_sha256": physics,
        "source_family_contract_physics_must_match": True,
    }
    with pytest.raises(mod.ContractError, match="duplicate JSON key: physics_contract_sha256"):
        mod.verify_schema3_bank_metadata(path, expected)


def test_manifest_pins_all_observed_c2_terminal_hashes_and_absence_boundaries():
    item = continuation_manifest()["preserved_c2"]
    assert item["training_log_sha256"] == "abffd4572578925e25b079ad6d86f6d98474e6d2dca88e0f5a2a7dd426c36dc3"
    assert item["terminal_checkpoint_sha256"] == "dbbc7a28ece4e166ba3743706827496960392fc46394286e7fddf83cddd776f6"
    assert item["hard_contract_sha256"] == "83f47ae6f0832b354112653a6f4cd66f98075181e6a546328a2b8d7a581c2772"
    assert item["training_launch_claim_sha256"] == "37fe244315bb6f3f179595ad0673cc638e6b1265f967019dff01fd518d4f86e5"
    assert item["old_runtime_verified_must_be_absent"] is True
    assert item["old_launch_failure_must_be_absent"] is True
    assert item["old_terminal_result_must_be_absent"] is True


def test_c2_attestation_is_outside_preserved_claim_namespace():
    manifest = continuation_manifest()
    original = original_manifest()
    attestation = mod.c2_attestation_path(manifest, original)
    c2_arm = mod.v1.expected_arm_paths(original, "C2")["arm"]
    assert attestation.parent == Path(manifest["outputs"]["continuation_evidence_root"])
    assert c2_arm not in attestation.parents


def test_recorded_v1r1_absence_receipt_is_content_bound():
    content = {
        "v1r1_manifest_sha256": mod.V1R1_MANIFEST_SHA256,
        "v1r1_launcher_sha256": mod.V1R1_LAUNCHER_SHA256,
        "v1r1_claim_or_runtime_artifact_adopted": False,
        "d2_claim_created_by_this_audit": False,
        "writes_or_signals_performed": False,
    }
    receipt = {"content": content, "content_sha256": mod.v1.canonical_sha256(content)}
    assert mod.validate_recorded_v1r1_absence(receipt) == receipt
    receipt["content_sha256"] = "0" * 64
    with pytest.raises(mod.ContractError, match="digest"):
        mod.validate_recorded_v1r1_absence(receipt)


def isolated_absence_inputs(tmp_path: Path):
    manifest = continuation_manifest()
    original = original_manifest()
    boundary = manifest["v1r1_absence_boundary"]
    boundary["continuation_evidence_root"] = str(tmp_path / "old-continuation")
    boundary["c2_attestation_path"] = str(tmp_path / "old-c2-attestation.json")
    boundary["paired_result_path"] = str(tmp_path / "old-pair.json")
    boundary["d2_arm_path"] = str(tmp_path / "old-d2-arm")
    original["source"]["training_checkout"] = str(tmp_path / "training-checkout")
    return manifest, original


def test_live_v1r1_absence_audit_accepts_only_empty_namespaces(tmp_path):
    manifest, original = isolated_absence_inputs(tmp_path)
    receipt = mod.audit_v1r1_absence(manifest, original)
    assert receipt["content"]["exact_d2_training_run_matches"] == []
    assert all(
        item["absent"] is True
        for item in receipt["content"]["paths"].values()
    )
    assert mod.validate_recorded_v1r1_absence(receipt) == receipt


@pytest.mark.parametrize(
    "field",
    [
        "continuation_evidence_root",
        "c2_attestation_path",
        "paired_result_path",
        "d2_arm_path",
    ],
)
def test_live_v1r1_absence_audit_rejects_each_existing_namespace(tmp_path, field):
    manifest, original = isolated_absence_inputs(tmp_path)
    path = Path(manifest["v1r1_absence_boundary"][field])
    path.mkdir(parents=True)
    with pytest.raises(mod.ContractError, match=rf"v1r1 {field} exists"):
        mod.audit_v1r1_absence(manifest, original)


def test_live_v1r1_absence_audit_rejects_symlinked_namespace(tmp_path):
    manifest, original = isolated_absence_inputs(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    path = Path(manifest["v1r1_absence_boundary"]["paired_result_path"])
    path.symlink_to(target)
    with pytest.raises(mod.ContractError, match="symlink component"):
        mod.audit_v1r1_absence(manifest, original)


def test_live_v1r1_absence_audit_rejects_exact_d2_training_run(tmp_path):
    manifest, original = isolated_absence_inputs(tmp_path)
    logs = (
        Path(original["source"]["training_checkout"])
        / original["source"]["wbt_relative_path"]
        / "logs/rsl_rl/agibot_a3_hope_virtualball"
    )
    logs.mkdir(parents=True)
    run_name = mod.v1.cells(original)[mod.CELL_ID]["run_name"]
    (logs / f"Jul14_00-00-00_{run_name}").mkdir()
    with pytest.raises(mod.ContractError, match="D2 training run exists"):
        mod.audit_v1r1_absence(manifest, original)


def test_validate_runtime_completes_absence_bank_both_gpus_and_c2_replay(
    monkeypatch, capsys,
):
    events = []
    original = original_manifest()
    absence = {
        "content": {
            "v1r1_manifest_sha256": mod.V1R1_MANIFEST_SHA256,
            "v1r1_launcher_sha256": mod.V1R1_LAUNCHER_SHA256,
            "v1r1_claim_or_runtime_artifact_adopted": False,
            "d2_claim_created_by_this_audit": False,
            "writes_or_signals_performed": False,
        },
    }
    absence["content_sha256"] = mod.v1.canonical_sha256(absence["content"])

    monkeypatch.setattr(
        mod, "continuation_control_receipt",
        lambda *args: events.append("control") or {},
    )
    monkeypatch.setattr(
        mod, "load_original_runtime",
        lambda *args: events.append("original") or (original, Path("v1.json"), Path("v1.py"), {}),
    )
    monkeypatch.setattr(
        mod, "audit_v1r1_absence",
        lambda *args: events.append("absence") or absence,
    )
    monkeypatch.setattr(
        mod, "verify_schema3_bank_metadata",
        lambda *args: events.append("bank") or bank_metadata(original),
    )

    def fake_runtime(_original, cell_id, *, require_empty_gpu):
        assert require_empty_gpu is True
        events.append(f"gpu-{cell_id}")
        return {"gpu_snapshot": {"cell": cell_id, "compute_pids": []}}

    monkeypatch.setattr(mod.v1, "verify_runtime", fake_runtime)

    def fake_c2_replay(*args, **kwargs):
        assert kwargs["stable_delay"] == 0.0
        assert kwargs["require_current_c2_gpu_empty"] is True
        assert kwargs["v1r1_absence_receipt"] == absence
        events.append("c2-terminal-v1r1-replay")
        return {
            "terminal_checkpoint_sha256": "a" * 64,
            "v1r1_false_rejection_reproduction": {
                "exact_message": mod.V1R1_FALSE_REJECTION,
            },
        }

    monkeypatch.setattr(mod, "audit_preserved_c2", fake_c2_replay)
    monkeypatch.setattr(
        mod.v1, "write_json_exclusive",
        lambda *args: pytest.fail("validate-runtime must not write v1r2 evidence"),
    )
    assert mod.main(["--manifest", str(MANIFEST), "--mode", "validate-runtime"]) == 0
    capsys.readouterr()
    assert events == [
        "control", "original", "absence", "bank", "gpu-C2", "gpu-D2",
        "c2-terminal-v1r1-replay",
    ]


def test_attest_failure_after_first_write_permanently_blocks_retry(tmp_path, monkeypatch):
    manifest = continuation_manifest()
    evidence_root = tmp_path / "v1r2-evidence"
    manifest["outputs"]["continuation_evidence_root"] = str(evidence_root)
    manifest["outputs"]["c2_attestation_path"] = str(
        evidence_root / "c2_terminal_attestation.json"
    )
    manifest["outputs"]["v1r1_absence_attestation_path"] = str(
        evidence_root / "v1r1_absence_attestation.json"
    )
    original = original_manifest()
    absence_content = {
        "v1r1_manifest_sha256": mod.V1R1_MANIFEST_SHA256,
        "v1r1_launcher_sha256": mod.V1R1_LAUNCHER_SHA256,
        "v1r1_claim_or_runtime_artifact_adopted": False,
        "d2_claim_created_by_this_audit": False,
        "writes_or_signals_performed": False,
    }
    absence = {
        "content": absence_content,
        "content_sha256": mod.v1.canonical_sha256(absence_content),
    }
    monkeypatch.setattr(mod, "continuation_control_receipt", lambda *args: {})
    monkeypatch.setattr(
        mod, "load_original_runtime",
        lambda *args: (original, Path("v1.json"), Path("v1.py"), {}),
    )
    monkeypatch.setattr(mod, "audit_v1r1_absence", lambda *args: absence)
    c2_calls = {"count": 0}

    def fail_only_after_absence_write(*args, **kwargs):
        c2_calls["count"] += 1
        if c2_calls["count"] == 1:
            return {"artifact_kind": "prewrite-c2-audit"}
        raise mod.ContractError("post-write replay drift")

    monkeypatch.setattr(mod, "audit_preserved_c2", fail_only_after_absence_write)
    with pytest.raises(mod.ContractError, match="post-write replay drift"):
        mod.write_c2_attestation(manifest, Path("manifest.json"), Path("launcher.py"))
    absence_path = evidence_root / "v1r1_absence_attestation.json"
    assert absence_path.is_file()
    assert not (evidence_root / "c2_terminal_attestation.json").exists()
    with pytest.raises(mod.ContractError, match="absence attestation already exists"):
        mod.write_c2_attestation(manifest, Path("manifest.json"), Path("launcher.py"))
    assert c2_calls["count"] == 2


def test_d2_claim_binds_v1r2_controller_original_recipe_and_c2_attestation():
    manifest = continuation_manifest()
    original = original_manifest()
    claim = mod.build_d2_claim(
        manifest,
        original,
        launcher_sha="a" * 64,
        arm_identity={"device": 1, "inode": 2},
        c2_attestation_sha="b" * 64,
    )
    assert claim["cell_id"] == "D2"
    assert claim["continuation_manifest_sha256"] == mod.CONTINUATION_MANIFEST_SHA256
    assert claim["continuation_launcher_sha256"] == "a" * 64
    assert claim["original_v1_manifest_sha256"] == mod.V1_MANIFEST_SHA256
    assert claim["frozen_v1r1_manifest_sha256"] == mod.V1R1_MANIFEST_SHA256
    assert claim["frozen_v1r1_launcher_sha256"] == mod.V1R1_LAUNCHER_SHA256
    assert claim["c2_terminal_attestation_sha256"] == "b" * 64
    assert claim["optimization_recipe"]["signed_face_guidance_weight"] == -0.4
    assert claim["c2_relaunch_authorized"] is False
    assert claim["automatic_retry_authorized"] is False


def test_mixed_outer_control_pair_normalizes_only_signed_weight():
    original = original_manifest()
    c2 = mod.v1.optimization_recipe(original, "C2")
    d2 = mod.v1.optimization_recipe(original, "D2")
    assert c2["signed_face_guidance_weight"] == 0.0
    assert d2["signed_face_guidance_weight"] == -0.4
    assert mod.normalized_recipe(c2) == mod.normalized_recipe(d2)
    drift = copy.deepcopy(d2)
    drift["num_envs"] = 513
    assert mod.normalized_recipe(c2) != mod.normalized_recipe(drift)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("c2_relaunch_forbidden", 1),
        lambda value: value["last_successful_read_only_snapshot"].__setitem__("runtime_authorization_from_snapshot", True),
        lambda value: value["preserved_c2"].__setitem__("pid_equals_pgid", True),
        lambda value: value["preserved_c2"].__setitem__("hard_contract_mount_normal_sign_per_clip", [1, -1]),
        lambda value: value["d2_only_continuation"].__setitem__("cell_id", "C2"),
        lambda value: value["decision_boundary"].__setitem__("l2", True),
        lambda value: value["continuation_control"].__setitem__("v1_helper_sha256", "0" * 64),
        lambda value: value["original_v1r1_control"].__setitem__("launcher_sha256", "0" * 64),
        lambda value: value["schema3_bank_metadata"].__setitem__("physics_contract_sha256", "0" * 64),
        lambda value: value["v1r1_absence_boundary"].__setitem__("v1r1_claim_or_runtime_artifact_adoption", True),
        lambda value: value["continuation_control"].__setitem__("launcher_relative_path", "../scripts/x.py"),
        lambda value: value["original_v1_control"].__setitem__("manifest_relative_path", "/tmp/x.json"),
    ],
)
def test_manifest_drift_fails_closed(mutator):
    value = continuation_manifest()
    mutator(value)
    with pytest.raises(mod.ContractError):
        mod.validate_manifest(value)


def test_launcher_contains_no_signal_or_robot_execution_path():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.kill(" not in source
    assert "subprocess.Popen" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "launch-c2" not in source
    assert "ros2" not in source
