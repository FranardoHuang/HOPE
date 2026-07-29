"""Host-only tests for the staged fresh-N5 launch-artifact materializer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_fresh_n5_launch_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_fresh_n5_launch_artifacts_test", SCRIPT
)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = M._json_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    return root


def test_sidecar_constructor_uses_exact_current_contracts():
    root = SCRIPT.parents[3]
    documents = M._sidecar_documents(root)
    receipt = documents["sidecar_launch_receipt.json"]
    proposal = documents["sidecar_trust_pin_proposal.json"]
    assert receipt["kind"] == "action_ball_frozen_eval_sidecar_launch"
    assert proposal["pins"][M.SIDECAR_LAUNCH_TRUST_NAME] == receipt[
        "content_sha256"
    ]
    assert proposal["pins"][M.SIDECAR_CODE_TRUST_NAME] == hashlib.sha256(
        (root / M.SIDECAR_CODE_SOURCE).read_bytes()
    ).hexdigest()
    assert proposal["authorization_granted"] is False


def test_new_directory_is_no_clobber(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "artifacts").mkdir()
    target = M._write_new_directory(
        root, "artifacts/attempt-001", {"receipt.json": {"ok": True}}
    )
    assert json.loads((target / "receipt.json").read_text()) == {"ok": True}
    with pytest.raises(M.MaterializationError, match="already exists"):
        M._write_new_directory(
            root, "artifacts/attempt-001", {"receipt.json": {"ok": False}}
        )
    assert json.loads((target / "receipt.json").read_text()) == {"ok": True}


def test_new_directory_never_exposes_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = make_repo(tmp_path)
    parent = root / "artifacts"
    parent.mkdir()
    real_json_bytes = M._json_bytes
    calls = 0

    def fail_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic serialization failure")
        return real_json_bytes(value)

    monkeypatch.setattr(M, "_json_bytes", fail_second)
    with pytest.raises(RuntimeError, match="synthetic"):
        M._write_new_directory(
            root,
            "artifacts/attempt-002",
            {"first.json": {"ok": True}, "second.json": {"ok": False}},
        )
    assert not (parent / "attempt-002").exists()
    assert list(parent.glob(".attempt-002.staging-*")) == []


def test_new_directory_deep_prepublish_failure_is_never_visible(
    tmp_path: Path,
):
    root = make_repo(tmp_path)
    parent = root / "artifacts"
    parent.mkdir()
    observed = {}

    def reject(staging: Path, target: Path):
        observed["staging"] = staging
        observed["target"] = target
        assert (staging / "receipt.json").is_file()
        assert not target.exists()
        raise M.MaterializationError("synthetic deep Gate rejection")

    with pytest.raises(M.MaterializationError, match="deep Gate"):
        M._write_new_directory(
            root,
            "artifacts/attempt-deep-gate",
            {"receipt.json": {"shape": "real validator boundary"}},
            prepublish=reject,
        )
    assert observed["target"] == parent / "attempt-deep-gate"
    assert not observed["target"].exists()
    assert not observed["staging"].exists()
    assert list(parent.glob(".attempt-deep-gate.staging-*")) == []


def test_canonical_digest_matches_launcher_utf8_contract():
    value = {"motion_path": "vendor_assets/动作.npz"}
    expected = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert M.canonical_sha256(value) == expected


def test_literal_trust_requires_exact_singleton_syntax(tmp_path: Path):
    digest = "a" * 64
    source = tmp_path / "trust.py"
    source.write_text(
        f"TRUST = frozenset(({digest!r},))\n", encoding="utf-8"
    )
    assert M._literal_trust_set(source, "TRUST") == frozenset((digest,))
    source.write_text(
        f"TRUST = frozenset(({digest!r}, {digest!r}))\n",
        encoding="utf-8",
    )
    with pytest.raises(M.MaterializationError, match="unique"):
        M._literal_trust_set(source, "TRUST")


def make_manifest(root: Path, *, order=M.ACTION_ORDER) -> tuple[dict, dict]:
    actions = []
    for index, action_id in enumerate(order):
        path = root / "motions" / f"{action_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"motion:{action_id}".encode("ascii"))
        actions.append(
            {
                "action_id": action_id,
                "action_uid": 100 + index,
                "family": (
                    "forehand"
                    if action_id in ("v12_forehand_block", "fh_loop_high")
                    else "backhand"
                ),
                "motion_path": path.relative_to(root).as_posix(),
                "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    scopes = {
        "upper": [
            {
                "motion_id": action["action_id"],
                "scope": "upper",
                "clip_index": index,
                "family": action["family"],
                "npz_sha256": action["motion_sha256"],
            }
            for index, action in enumerate(actions)
        ]
    }
    prototype = {
        "schema_version": 2,
        "scopes": scopes,
        "derived_sha256": M.canonical_sha256(scopes),
    }
    prototype_path = root / "configs" / "prototype.json"
    prototype_sha = write_json(prototype_path, prototype)
    manifest = {
        "schema_version": 3,
        "manifest_id": "fixture-fresh-n5",
        "mobility_mode": "no_move",
        "action_order": list(order),
        "prototype": {
            "path": prototype_path.relative_to(root).as_posix(),
            "sha256": prototype_sha,
            "scope": "upper",
        },
        "solver_profile_sha256": "1" * 64,
        "physics_profile_sha256": "2" * 64,
        "actions": actions,
    }
    path = root / "configs" / "manifest.json"
    digest = write_json(path, manifest)
    return manifest, {"path": "configs/manifest.json", "sha256": digest}


def test_manifest_rejects_old_or_reordered_actions(tmp_path: Path):
    root = make_repo(tmp_path)
    _, pin = make_manifest(
        root,
        order=(
            "fh_loop",
            "v12_forehand_block",
            "bh_block",
            "s0_highpress",
            "fh_loop_high",
        ),
    )
    with pytest.raises(M.MaterializationError, match="exact fresh"):
        M._load_manifest(root, pin)


def test_manifest_requires_exact_pinned_upper_prototype(tmp_path: Path):
    root = make_repo(tmp_path)
    manifest, pin = make_manifest(root)
    without_prototype = dict(manifest)
    del without_prototype["prototype"]
    missing_path = root / "configs" / "missing-prototype.json"
    missing_pin = {
        "path": missing_path.relative_to(root).as_posix(),
        "sha256": write_json(missing_path, without_prototype),
    }
    with pytest.raises(M.MaterializationError, match="manifest.prototype"):
        M._load_manifest(root, missing_pin)

    other_path = root / "configs" / "other-prototype.json"
    other_sha = write_json(other_path, {"schema_version": 2})
    with pytest.raises(M.MaterializationError, match="formal spec pin"):
        M._load_manifest(
            root,
            pin,
            expected_prototype_pin_value={
                "path": other_path.relative_to(root).as_posix(),
                "sha256": other_sha,
            },
        )


def make_physical_bundle(
    root: Path,
    manifest: dict,
    manifest_pin: dict,
) -> tuple[dict, dict]:
    batch_path = root / "inputs" / "batch.json"
    batch_sha = write_json(batch_path, {"fixture": "batch"})
    profile_path = root / "inputs" / "profile-pins.json"
    profile_sha = write_json(profile_path, {"fixture": "profiles"})
    geometry_path = root / "inputs" / "racket_contact_geometry.py"
    geometry_path.write_text("# fixture geometry\n", encoding="utf-8")
    geometry_sha = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    actions = []
    for action in manifest["actions"]:
        execution = {
            "artifact_type": "frozen_ball_to_task_solver_execution_v1",
            "execution_id": (
                f"fresh-n5:{manifest_pin['sha256']}:{action['action_id']}"
            ),
            "executed_before_gate": True,
            "solver_replayed_exact": True,
            "selector_executed": False,
            "action_identity_frozen": True,
            "action_switching_allowed": False,
            "hardware_authorized": False,
        }
        cases = [{"case_id": f"{action['action_id']}:center"}]
        receipt = {
            "schema_version": 1,
            "artifact_type": "frozen_action_ball_solver_execution_receipt_v1",
            "producer": {"fixture": True},
            "action_identity": {
                "action_id": action["action_id"],
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
            },
            "profile_identity": {"fixture": True},
            "solver_execution_identity": execution,
            "cases": cases,
        }
        receipt["receipt_payload_sha256"] = M.canonical_sha256(receipt)
        receipt_path = (
            root / "solver-receipts" / f"{action['action_id']}.json"
        )
        receipt_sha = write_json(receipt_path, receipt)
        task = {
            "schema_version": 1,
            "authority": (
                "pre_registered_frozen_action_ball_solver_receipt_v1"
            ),
            "action_id": action["action_id"],
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
            "ball_profile_sha256": "3" * 64,
            "solver_profile_sha256": manifest["solver_profile_sha256"],
            "physics_profile_sha256": manifest["physics_profile_sha256"],
            "solver_implementation_source_sha256": {
                "fixture.py": "4" * 64
            },
            "solver_execution_receipt_path": (
                receipt_path.relative_to(root).as_posix()
            ),
            "solver_execution_receipt_sha256": receipt_sha,
            "solver_execution_identity": execution,
            "solver_execution_identity_sha256": M.canonical_sha256(
                execution
            ),
            "selector_executed": False,
            "action_identity_frozen": True,
            "cases": cases,
            "cases_sha256": M.canonical_sha256(cases),
        }
        actions.append(
            {
                "action_id": action["action_id"],
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "physical_ball_launch": {"fixture": action["action_id"]},
                "physical_task_binding": task,
            }
        )
    bundle = {
        "schema_version": 1,
        "artifact_type": "fresh_n5_physical_task_bundle_v1",
        "base_manifest": {
            "path": manifest_pin["path"],
            "raw_sha256": manifest_pin["sha256"],
            "schema_version": 3,
            "strict_training_input": True,
        },
        "batch": {
            "path": batch_path.relative_to(root).as_posix(),
            "sha256": batch_sha,
        },
        "prototype": manifest["prototype"],
        "profile_pins": {
            "path": profile_path.relative_to(root).as_posix(),
            "sha256": profile_sha,
            "solver_profile_sha256": manifest["solver_profile_sha256"],
            "physics_profile_sha256": manifest["physics_profile_sha256"],
            "geometry_source_sha256": geometry_sha,
        },
        "action_order": list(M.ACTION_ORDER),
        "selector_executed": False,
        "action_identity_frozen": True,
        "action_switching_allowed": False,
        "mobility_mode": "no_move",
        "base_task_frame": "relative_about_actual_episode_spawn",
        "gate_materialization_fields": {
            "racket_geometry_contract": {
                "schema_version": 2,
                "semantics": "exact_face_contact_v2",
                "ball_target_point": "physical_ball_center_at_native_contact",
                "site_target_mapping": "site_target_from_ball_center",
                "face_velocity_mapping": (
                    "site_linear_plus_omega_cross_face_center_offset"
                ),
                "source_path": geometry_path.relative_to(root).as_posix(),
                "source_sha256": geometry_sha,
                "geometry_source_sha256": geometry_sha,
            },
            "physical_contact_contract": {
                "schema_version": 2,
                "authority": "fixture-contact-contract",
            },
        },
        "materialization_contract": {
            "training_consumer": "consume base_manifest only",
            "fitted_gate_consumer": "fixture disposable overlay",
            "current_inline_fitted_gate_support": False,
            "downstream_gap": "fixture",
            "required_external_inputs_not_synthesized": [
                "per-action compiler_candidate_pre_admission_v1 evidence",
                "formal source-receipt trust root bound to a clean commit",
                "clean committed runtime/source/data closure",
            ],
        },
        "actions": actions,
    }
    bundle["content_sha256"] = M.canonical_sha256(bundle)
    bundle_path = root / "inputs" / "physical-task-bundle.json"
    bundle_sha = write_json(bundle_path, bundle)
    return bundle, {
        "path": bundle_path.relative_to(root).as_posix(),
        "sha256": bundle_sha,
    }


def make_physical_evidence(root: Path) -> dict:
    pins = {}
    for name in ("base-build", "append-build", "base-gate", "append-gate"):
        path = root / "evidence" / f"{name}.json"
        pins[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": write_json(path, {"fixture": name}),
        }
    return pins


def physical_documents(
    root: Path,
    manifest_pin: dict,
    bundle_pin: dict,
    evidence: dict,
    *,
    output_dir: str = "artifacts/physical-001",
) -> dict:
    return M._physical_gate_documents(
        root,
        base_manifest_pin_value=manifest_pin,
        bundle_pin_value=bundle_pin,
        base_build_manifest_pin_value=evidence["base-build"],
        append_build_manifest_pin_value=evidence["append-build"],
        base_bank_gate_pin_value=evidence["base-gate"],
        append_bank_gate_pin_value=evidence["append-gate"],
        output_dir_relative=output_dir,
    )


def test_physical_gate_preserves_strict_manifest_and_materializes_overlay(
    tmp_path: Path,
):
    root = make_repo(tmp_path)
    (root / "artifacts").mkdir()
    manifest, manifest_pin = make_manifest(root)
    strict_before = (root / manifest_pin["path"]).read_bytes()
    _, bundle_pin = make_physical_bundle(root, manifest, manifest_pin)
    evidence = make_physical_evidence(root)
    documents = physical_documents(root, manifest_pin, bundle_pin, evidence)
    target = M._write_new_directory(
        root, "artifacts/physical-001", documents
    )
    assert (root / manifest_pin["path"]).read_bytes() == strict_before
    physical = json.loads(
        (target / "physical_gate_manifest.json").read_text()
    )
    assert physical["actions"][0]["physical_task_binding"][
        "action_id"
    ] == M.ACTION_ORDER[0]
    assert physical["actions"][0]["admission"][
        "training_authorized"
    ] is False
    assert physical["racket_geometry_contract"]["schema_version"] == 2
    assert physical["physical_contact_contract"]["schema_version"] == 2
    assert all(
        "physical_task_binding" not in row
        and "physical_ball_launch" not in row
        for row in manifest["actions"]
    )


def test_physical_gate_prepublish_wires_staged_bytes_to_shared_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = make_repo(tmp_path)
    (root / "artifacts").mkdir()
    manifest, manifest_pin = make_manifest(root)
    _, bundle_pin = make_physical_bundle(root, manifest, manifest_pin)
    evidence = make_physical_evidence(root)
    calls = []
    trusted = {
        "profile_id": M.ACTION_SET_PROFILE,
        "expected_n": len(M.ACTION_ORDER),
        "scope": M.SCOPE,
        "mobility_mode": M.MOBILITY_MODE,
        "ordered_action_ids": list(M.ACTION_ORDER),
        "ordered_action_uids": [
            row["action_uid"] for row in manifest["actions"]
        ],
        "manifest_path": manifest_pin["path"],
        "manifest_sha256": manifest_pin["sha256"],
    }

    class FakeActionSetContract:
        @staticmethod
        def load_contract_from_source(_source, profile):
            assert profile == M.ACTION_SET_PROFILE
            return trusted

        @staticmethod
        def verify_manifest_identity(contract, strict, strict_raw):
            assert contract is trusted
            assert strict == manifest
            assert hashlib.sha256(strict_raw).hexdigest() == manifest_pin[
                "sha256"
            ]

    class FakeGate:
        ACTION_SET_CONTRACT_SOURCE_PATH = root / manifest_pin["path"]
        action_set_contract = FakeActionSetContract

        @staticmethod
        def validate_physical_materialization_receipt(
            receipt,
            *,
            strict_manifest_pin,
            physical_manifest_pin,
            trusted_action_set,
        ):
            assert strict_manifest_pin == manifest_pin
            assert receipt["physical_gate_manifest"] == physical_manifest_pin
            assert trusted_action_set is trusted
            calls.append("receipt")

        @staticmethod
        def validate_physical_manifest(
            physical,
            *,
            trusted_action_set,
            repo_file_overrides,
        ):
            assert physical["action_order"] == list(M.ACTION_ORDER)
            assert trusted_action_set is trusted
            assert set(repo_file_overrides) == {
                row["admission"]["registry_entry_path"]
                for row in physical["actions"]
            }
            assert all(path.is_file() for path in repo_file_overrides.values())
            calls.append("manifest")

    monkeypatch.setattr(M, "_teacher_gate_module", lambda _root: FakeGate)
    target = M._write_new_directory(
        root,
        "artifacts/physical-deep-001",
        physical_documents(
            root,
            manifest_pin,
            bundle_pin,
            evidence,
            output_dir="artifacts/physical-deep-001",
        ),
        prepublish=lambda staging, final: M._validate_staged_physical_gate(
            root, staging, final
        ),
    )
    assert calls == ["receipt", "manifest"]
    assert target.is_dir()


def test_formal_physical_gate_closure_reopens_all_crossbindings(
    tmp_path: Path,
):
    root = make_repo(tmp_path)
    (root / "artifacts").mkdir()
    manifest, manifest_pin = make_manifest(root)
    _, bundle_pin = make_physical_bundle(root, manifest, manifest_pin)
    evidence = make_physical_evidence(root)
    documents = physical_documents(root, manifest_pin, bundle_pin, evidence)
    target = M._write_new_directory(
        root, "artifacts/physical-001", documents
    )
    _, _, bindings = M._load_manifest(root, manifest_pin)
    manifest_path = target / "physical_gate_manifest.json"
    receipt_path = target / "physical_gate_materialization_receipt.json"
    result = M._validate_physical_gate_closure(
        root,
        {
            "bundle": bundle_pin,
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            },
            "materialization_receipt": {
                "path": receipt_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
            },
        },
        manifest=manifest,
        manifest_pin=manifest_pin,
        bindings=bindings,
        promotion={
            "base_build_manifest": evidence["base-build"],
            "append_build_manifest": evidence["append-build"],
            "bank_gate_reports": {
                "base": {
                    "kind": "canonical_base_five_full_replay",
                    **evidence["base-gate"],
                },
                "append": {
                    "kind": "fresh_n5_append_suffix",
                    **evidence["append-gate"],
                },
            },
        },
    )
    assert result["manifest"]["sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert result["bundle"] == bundle_pin


def test_physical_gate_rejects_wrong_base_raw_sha(tmp_path: Path):
    root = make_repo(tmp_path)
    manifest, manifest_pin = make_manifest(root)
    bundle, _ = make_physical_bundle(root, manifest, manifest_pin)
    bundle["base_manifest"]["raw_sha256"] = "f" * 64
    unsigned = dict(bundle)
    del unsigned["content_sha256"]
    bundle["content_sha256"] = M.canonical_sha256(unsigned)
    bundle_path = root / "inputs" / "wrong-base-bundle.json"
    bundle_pin = {
        "path": bundle_path.relative_to(root).as_posix(),
        "sha256": write_json(bundle_path, bundle),
    }
    with pytest.raises(M.MaterializationError, match="different strict base"):
        physical_documents(
            root,
            manifest_pin,
            bundle_pin,
            make_physical_evidence(root),
        )


def test_physical_gate_rejects_strict_manifest_inline_extras(tmp_path: Path):
    root = make_repo(tmp_path)
    manifest, _ = make_manifest(root)
    manifest["actions"][0]["physical_ball_launch"] = {"forbidden": True}
    path = root / "configs" / "inline-manifest.json"
    manifest_pin = {
        "path": path.relative_to(root).as_posix(),
        "sha256": write_json(path, manifest),
    }
    with pytest.raises(M.MaterializationError, match="gate-only inline extras"):
        physical_documents(
            root,
            manifest_pin,
            {"path": "unused", "sha256": "0" * 64},
            make_physical_evidence(root),
        )


def test_physical_gate_rejects_tampered_overlay_identity(tmp_path: Path):
    root = make_repo(tmp_path)
    manifest, manifest_pin = make_manifest(root)
    bundle, _ = make_physical_bundle(root, manifest, manifest_pin)
    bundle["actions"][0]["physical_task_binding"][
        "solver_execution_identity"
    ]["execution_id"] = "fresh-n5:tampered:bh_loop_c"
    unsigned = dict(bundle)
    del unsigned["content_sha256"]
    bundle["content_sha256"] = M.canonical_sha256(unsigned)
    path = root / "inputs" / "tampered-bundle.json"
    bundle_pin = {
        "path": path.relative_to(root).as_posix(),
        "sha256": write_json(path, bundle),
    }
    with pytest.raises(M.MaterializationError, match="strict base/action"):
        physical_documents(
            root,
            manifest_pin,
            bundle_pin,
            make_physical_evidence(root),
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_contact_contract", "extra_gate_field"),
)
def test_physical_gate_rejects_inexact_gate_materialization_fields(
    tmp_path: Path,
    mutation: str,
):
    root = make_repo(tmp_path)
    manifest, manifest_pin = make_manifest(root)
    bundle, _ = make_physical_bundle(root, manifest, manifest_pin)
    fields = bundle["gate_materialization_fields"]
    if mutation == "missing_contact_contract":
        del fields["physical_contact_contract"]
    else:
        fields["unexpected"] = {"forbidden": True}
    unsigned = dict(bundle)
    del unsigned["content_sha256"]
    bundle["content_sha256"] = M.canonical_sha256(unsigned)
    path = root / "inputs" / f"{mutation}.json"
    bundle_pin = {
        "path": path.relative_to(root).as_posix(),
        "sha256": write_json(path, bundle),
    }
    with pytest.raises(
        M.MaterializationError,
        match="gate_materialization_fields keys changed",
    ):
        physical_documents(
            root,
            manifest_pin,
            bundle_pin,
            make_physical_evidence(root),
        )


@dataclass(frozen=True)
class FakeProfileKey:
    action_uid: int
    profile_sha256: str
    mobility: str

    def as_dict(self):
        return {
            "action_uid": self.action_uid,
            "profile_sha256": self.profile_sha256,
            "mobility": self.mobility,
        }


class FakeAdmission:
    class FreshN5BankPromotionBinding:
        def __init__(self, **values):
            self.__dict__.update(values)

    @staticmethod
    def _binding_document(binding):
        return {
            "purpose": binding.purpose,
            "bank_id": binding.bank_id,
            "scope": binding.scope,
            "registry_sha256": binding.registry_sha256,
            "alignment_sha256": binding.alignment_sha256,
            "motion_ids": list(binding.motion_ids),
            "npz_sha256": list(binding.npz_sha256),
            "canonical_ready_sha256": binding.canonical_ready_sha256,
            "canonical_ready_fk_sha256": binding.canonical_ready_fk_sha256,
            "build_manifest_sha256": list(binding.build_manifest_sha256),
            "evidence_receipts": [],
            "question_bank_sha256": list(binding.question_bank_sha256),
            "training_config_sha256": list(binding.training_config_sha256),
            "onnx_model_sha256": list(binding.onnx_model_sha256),
            "onnx_metadata_sha256": list(binding.onnx_metadata_sha256),
            "adoption_manifest_sha256": list(
                binding.adoption_manifest_sha256
            ),
            "base_bank_id": binding.base_bank_id,
            "bank_motion_ids": list(binding.bank_motion_ids),
            "bank_npz_sha256": list(binding.bank_npz_sha256),
            "base_build_manifest_sha256": (
                binding.base_build_manifest_sha256
            ),
            "append_build_manifest_sha256": (
                binding.append_build_manifest_sha256
            ),
            "base_bank_gate_report_sha256": (
                binding.base_bank_gate_report_sha256
            ),
            "append_bank_gate_report_sha256": (
                binding.append_bank_gate_report_sha256
            ),
            "base_swept_clearance_receipt_sha256": (
                binding.base_swept_clearance_receipt_sha256
            ),
            "append_swept_clearance_receipt_sha256": (
                binding.append_swept_clearance_receipt_sha256
            ),
            "mujoco_fitted_ball_receipt_sha256": (
                binding.mujoco_fitted_ball_receipt_sha256
            ),
            "mujoco_fitted_ball_capsule_receipt_sha256": (
                binding.mujoco_fitted_ball_capsule_receipt_sha256
            ),
            "isaac_table_filtered_smoke_receipt_sha256": (
                binding.isaac_table_filtered_smoke_receipt_sha256
            ),
        }

    @staticmethod
    def _validate_fresh_n5_bank_closure(*args, **kwargs):
        return None

    @staticmethod
    def _validate_fresh_n5_fitted_ball_receipt(*args, **kwargs):
        return {"fixture": "fitted"}

    @staticmethod
    def _validate_fresh_n5_isaac_table_smoke_receipt(*args, **kwargs):
        return None


@dataclass(frozen=True)
class FakeGeneric:
    purpose: str
    bank_id: str
    scope: str
    registry_sha256: str
    alignment_sha256: str
    motion_ids: tuple[str, ...]
    npz_sha256: tuple[str, ...]
    canonical_ready_sha256: str
    canonical_ready_fk_sha256: str
    build_manifest_sha256: tuple[str, ...]
    evidence_levels: tuple[str, ...]
    evidence_manifest_sha256: tuple[str, ...]
    evidence_certificate_sha256: tuple[tuple[str, ...], ...]
    question_bank_sha256: tuple[str, ...]
    training_config_sha256: tuple[str, ...]
    onnx_model_sha256: tuple[None, ...]
    onnx_metadata_sha256: tuple[None, ...]
    adoption_manifest_sha256: tuple[str, ...]


def fake_runtime_modules():
    curriculum = SimpleNamespace(
        ARM_CATALOG_SHA256="a" * 64,
        ActionProfileKey=FakeProfileKey,
        _canonical_sha256=M.canonical_sha256,
    )

    def drain_document(**values):
        return {
            "schema_version": 1,
            "kind": "action_ball_drain_reset_launch",
            "authority_contract_sha256": "b" * 64,
            "curriculum_contract_sha256": values[
                "curriculum_contract_sha256"
            ],
            "profile_order": [
                key.as_dict() for key in values["profile_order"]
            ],
            "arm_catalog_sha256": values["arm_catalog_sha256"],
            "scheduler_contract_sha256": values[
                "scheduler_contract_sha256"
            ],
            "sampler_sha256": values["sampler_sha256"],
            "solver_sha256": values["solver_sha256"],
            "policy_contract_sha256": values["policy_contract_sha256"],
            "runtime_source_contract_sha256": values[
                "runtime_source_contract_sha256"
            ],
            "runtime_source_path": values["runtime_source_path"],
            "runtime_source_sha256": values["runtime_source_sha256"],
            "broker_contract_sha256": values["broker_contract_sha256"],
            "attempt_pool_contract_sha256": values[
                "attempt_pool_contract_sha256"
            ],
            "task_receipt_pool_contract_sha256": values[
                "task_receipt_pool_contract_sha256"
            ],
            "env_reset_contract_sha256": values[
                "env_reset_contract_sha256"
            ],
        }

    curriculum.drain_reset_launch_receipt_document = drain_document

    def evaluator_document(**values):
        return {
            "schema_version": 4,
            "kind": "action_ball_frozen_evaluator_v4_launch",
            "authority_contract_sha256": "c" * 64,
            "curriculum_contract_sha256": values[
                "curriculum_contract_sha256"
            ],
            "profile_order": [
                key.as_dict() for key in values["profile_order"]
            ],
            "arm_catalog_sha256": values["arm_catalog_sha256"],
            "scheduler_contract_sha256": values[
                "scheduler_contract_sha256"
            ],
            "sampler_sha256": values["sampler_sha256"],
            "solver_sha256": values["solver_sha256"],
            "policy_contract_sha256": values["policy_contract_sha256"],
            "attempt_source_contract_sha256": values[
                "attempt_source_contract_sha256"
            ],
            "attempt_source_path": values["attempt_source_path"],
            "attempt_source_sha256": values["attempt_source_sha256"],
            "window_contract": {"fixture": True},
        }

    evaluation = SimpleNamespace(
        launch_receipt_document_v4=evaluator_document,
        _canonical_sha256=M.canonical_sha256,
    )
    inbox = SimpleNamespace(
        FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_CONTRACT_SHA256="d" * 64
    )
    return curriculum, evaluation, inbox


def test_formal_wires_all_hashes_without_editing_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = make_repo(tmp_path)
    manifest, manifest_pin = make_manifest(root)
    for source in (M.INBOX_SOURCE, M.HOPE_COMMANDS_SOURCE, M.SIDECAR_CODE_SOURCE):
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {source}\n", encoding="utf-8")

    evidence_pins = {}
    for name in (
        "base-build",
        "append-build",
        "base-gate",
        "append-gate",
        "base-swept",
        "append-swept",
        "fitted",
        "retained",
        "isaac",
    ):
        path = root / "evidence" / f"{name}.json"
        digest = write_json(path, {"fixture": name})
        evidence_pins[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }

    base_manifest_sha = evidence_pins["base-build"]["sha256"]
    append_manifest_sha = evidence_pins["append-build"]["sha256"]
    registry_entries = []
    for index, (action, action_id) in enumerate(
        zip(manifest["actions"], M.ACTION_ORDER)
    ):
        registry_entries.append(
            {
                "motion_id": action_id,
                "scope": "upper",
                "npz_path": action["motion_path"],
                "npz_sha256": action["motion_sha256"],
                "family": action["family"],
                "publication_class": "training_adopted",
                "training_authorized": True,
                "deployment_authorized": False,
                "hardware_authorized": False,
                "build_manifest_sha256": (
                    append_manifest_sha
                    if action_id in ("v12_forehand_block", "fh_loop_high")
                    else base_manifest_sha
                ),
            }
        )

    generic = FakeGeneric(
        purpose="training",
        bank_id="fresh-upper-n5",
        scope="upper",
        registry_sha256="e" * 64,
        alignment_sha256="f" * 64,
        motion_ids=M.ACTION_ORDER,
        npz_sha256=tuple(row["npz_sha256"] for row in registry_entries),
        canonical_ready_sha256="0" * 64,
        canonical_ready_fk_sha256="1" * 64,
        build_manifest_sha256=tuple(
            row["build_manifest_sha256"] for row in registry_entries
        ),
        evidence_levels=("E2",) * 5,
        evidence_manifest_sha256=("2" * 64,) * 5,
        evidence_certificate_sha256=(("3" * 64, "4" * 64),) * 5,
        question_bank_sha256=("5" * 64,) * 5,
        training_config_sha256=("6" * 64,) * 5,
        onnx_model_sha256=(None,) * 5,
        onnx_metadata_sha256=(None,) * 5,
        adoption_manifest_sha256=("7" * 64,) * 5,
    )
    fake_registry = SimpleNamespace(
        bank_id=generic.bank_id,
        registry_sha256=generic.registry_sha256,
    )
    fake_registry_module = SimpleNamespace(motion_admission=FakeAdmission)

    stage = root / ".fixture-stage" / "canonical_registry.json"
    stage.parent.mkdir()
    stage.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        M,
        "_stage_registry",
        lambda *args, **kwargs: (
            stage,
            fake_registry_module,
            (fake_registry, generic),
        ),
    )
    monkeypatch.setattr(M, "_runtime_modules", lambda root: fake_runtime_modules())
    physical_gate_pins = {
        "bundle": {"path": "fixture/bundle.json", "sha256": "8" * 64},
        "manifest": {
            "path": "fixture/physical-manifest.json",
            "sha256": "9" * 64,
        },
        "materialization_receipt": {
            "path": "fixture/physical-receipt.json",
            "sha256": "a" * 64,
        },
    }
    monkeypatch.setattr(
        M,
        "_validate_physical_gate_closure",
        lambda *args, **kwargs: physical_gate_pins,
    )

    spec = {
        "schema_version": 1,
        "kind": M.FORMAL_SPEC_KIND,
        "manifest": manifest_pin,
        "prototype": {
            "path": manifest["prototype"]["path"],
            "sha256": manifest["prototype"]["sha256"],
        },
        "physical_gate": {
            "bundle": physical_gate_pins["bundle"],
            "manifest": physical_gate_pins["manifest"],
            "materialization_receipt": physical_gate_pins[
                "materialization_receipt"
            ],
        },
        "registry": {
            "bank_id": generic.bank_id,
            "canonical_ready": {"path": "ready.npz", "sha256": "0" * 64},
            "canonical_ready_fk": {
                "path": "ready-fk.npz",
                "sha256": "1" * 64,
            },
            "entries": registry_entries,
        },
        "promotion": {
            "base_bank_id": "base-five",
            "bank_motion_ids": list(M.BANK_ORDER),
            "bank_npz_sha256": [
                f"{index + 20:064x}" for index in range(14)
            ],
            "base_build_manifest": evidence_pins["base-build"],
            "append_build_manifest": evidence_pins["append-build"],
            "bank_gate_reports": {
                "base": {
                    "kind": "canonical_base_five_full_replay",
                    **evidence_pins["base-gate"],
                },
                "append": {
                    "kind": "fresh_n5_append_suffix",
                    **evidence_pins["append-gate"],
                },
            },
            "continuous_swept_clearance_receipts": {
                "base": {
                    "kind": "canonical_base_five",
                    **evidence_pins["base-swept"],
                },
                "append": {
                    "kind": "fresh_n5_append_suffix",
                    **evidence_pins["append-swept"],
                },
            },
            "mujoco_fitted_ball_receipt": {
                **evidence_pins["fitted"],
                "retained_capsule_receipt": evidence_pins["retained"],
            },
            "isaac_table_filtered_smoke_receipt": evidence_pins["isaac"],
        },
        "evaluator": {
            "policy_contract_sha256": "8" * 64,
            "curriculum_contract_sha256": "9" * 64,
            "profile_order": [
                {
                    "action_uid": row["action_uid"],
                    "profile_sha256": f"{index + 40:064x}",
                    "mobility": "no_move",
                }
                for index, row in enumerate(manifest["actions"])
            ],
            "scheduler_contract_sha256": "a" * 64,
            "sampler_sha256": "b" * 64,
        },
        "drain": {
            "runtime_source_contract_sha256": "c" * 64,
            "broker_contract_sha256": "d" * 64,
            "attempt_pool_contract_sha256": "e" * 64,
            "task_receipt_pool_contract_sha256": "f" * 64,
            "env_reset_contract_sha256": "0" * 64,
        },
    }
    spec_path = root / "configs" / "formal-spec.json"
    write_json(spec_path, spec)
    sidecar = {
        "schema_version": 1,
        "kind": "action_ball_frozen_eval_sidecar_launch",
        "content": {},
        "content_sha256": "a" * 64,
    }
    sidecar_path = root / "configs" / "sidecar.json"
    sidecar_sha = write_json(sidecar_path, sidecar)
    outputs = M._formal_documents(
        root,
        spec_path,
        sidecar,
        {"path": "configs/sidecar.json", "sha256": sidecar_sha},
    )
    assert set(outputs) == {
        "canonical_registry.json",
        "promotion_certificate.json",
        "motion_admission_receipt.json",
        "evaluator_launch_receipt.json",
        "drain_reset_launch_receipt.json",
        "formal_trust_pin_proposal.json",
        "formal_materialization_receipt.json",
    }
    promotion_sha = hashlib.sha256(
        M._json_bytes(outputs["promotion_certificate.json"])
    ).hexdigest()
    assert outputs["motion_admission_receipt.json"][
        "promotion_certificate_sha256"
    ] == promotion_sha
    proposal = outputs["formal_trust_pin_proposal.json"]
    assert proposal["pins"][M.PROMOTION_TRUST_NAME] == promotion_sha
    assert proposal["manifest_bindings"][
        "disposable_physical_gate_manifest"
    ] == physical_gate_pins["manifest"]
    assert proposal["authorization_granted"] is False


def test_formal_rejects_wrong_profile_order_before_publication(
    tmp_path: Path
):
    root = make_repo(tmp_path)
    _, pin = make_manifest(root)
    manifest, _, bindings = M._load_manifest(root, pin)
    entries = [
        {
            "motion_id": binding["motion_id"],
            "scope": "upper",
            "npz_path": binding["motion_path"],
            "npz_sha256": binding["motion_sha256"],
            "family": binding["family"],
            "publication_class": "training_adopted",
            "training_authorized": True,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
        for binding in bindings
    ]
    entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(M.MaterializationError, match=r"entries\[0\]"):
        M._registry_document(
            {
                "bank_id": "fresh-upper-n5",
                "canonical_ready": {"path": "r", "sha256": "0" * 64},
                "canonical_ready_fk": {"path": "f", "sha256": "1" * 64},
                "entries": entries,
            },
            bindings,
        )
