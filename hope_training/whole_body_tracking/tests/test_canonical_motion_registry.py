"""Pure-CPU contract tests for canonical_motion_registry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "canonical_motion_registry.py"
)
_SPEC = importlib.util.spec_from_file_location("canonical_motion_registry", _SCRIPT)
registry = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = registry
_SPEC.loader.exec_module(registry)


_DUMMY_SHA = "a" * 64
_FAMILIES = ("forehand", "backhand", "forehand", "backhand", "forehand")
_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_ignores_preloaded_generic_admission_module(monkeypatch):
    fake = types.ModuleType("canonical_motion_admission")
    fake.__file__ = str(
        _SCRIPT.parent / "canonical_motion_admission.py"
    )
    monkeypatch.setitem(sys.modules, "canonical_motion_admission", fake)
    spec = importlib.util.spec_from_file_location(
        "canonical_motion_registry_injection_test", _SCRIPT
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(spec.name, None)

    assert loaded.motion_admission is not fake
    assert (
        Path(loaded.motion_admission.__file__).resolve()
        == _SCRIPT.parent / "canonical_motion_admission.py"
    )


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _schema2_arrays(*, frames: int = 5, fps: float = 50.0, seed: int = 0):
    joint_pos = np.zeros((frames, 31), dtype=np.float32)
    joint_pos[1:-1] = np.float32(seed + 1) * np.linspace(
        -0.01, 0.01, 31, dtype=np.float32
    )
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:-1] = np.float32(seed + 1) * 0.02
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_pos[1:-1, :, 2] = np.float32(0.8 + 0.01 * seed)
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    return {
        "fps": np.array([fps], dtype=np.float64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "kinematics_schema_version": np.array([2], dtype=np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
        "body_names": np.asarray(registry.RUNTIME_BODY_NAMES),
    }


def _write_npz(path: Path, *, seed: int = 0, arrays=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **(_schema2_arrays(seed=seed) if arrays is None else arrays))
    return _sha256(path)


def _write_ready(path: Path, *, joint_offset: float = 0.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joint_pos = np.zeros(31, dtype=np.float64)
    joint_pos[0] = joint_offset
    np.savez(
        path,
        joint_pos=joint_pos,
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos_w=np.zeros(3, dtype=np.float64),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        source_segment=np.array("bh_loop_c"),
        source_npz=np.array("synthetic.npz"),
        source_frame=np.array(0, dtype=np.int64),
        striking_joint_ids=np.arange(7, dtype=np.int64),
        note=np.array("registry test ready"),
    )
    return _sha256(path)


def _write_ready_fk(
    path: Path,
    *,
    canonical_ready_sha256: str,
    body_pos: np.ndarray | None = None,
    body_quat: np.ndarray | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if body_pos is None:
        body_pos = np.zeros((32, 3), dtype=np.float32)
    if body_quat is None:
        body_quat = np.zeros((32, 4), dtype=np.float32)
        body_quat[:, 0] = 1.0
    np.savez(
        path,
        canonical_ready_sha256=np.array(canonical_ready_sha256),
        body_names=np.asarray(registry.RUNTIME_BODY_NAMES),
        body_pos_w=np.asarray(body_pos, dtype=np.float32),
        body_quat_w=np.asarray(body_quat, dtype=np.float32),
        kinematics_contract_version=np.array([1], dtype=np.int64),
    )
    return _sha256(path)


def _write_evidence_bundle(
    repo: Path,
    *,
    motion_id: str,
    scope: str,
    variant: str,
    npz_sha256: str,
    level: str,
) -> tuple[Path, str]:
    level_index = registry.EVIDENCE_LEVELS.index(level)
    certificates = []
    for certificate_level in registry.EVIDENCE_LEVELS[1 : level_index + 1]:
        certificate_path = (
            repo
            / "evidence"
            / f"{motion_id}.{certificate_level.lower()}.certificate.json"
        )
        certificate_sha = _write_json(
            certificate_path,
            {
                "schema_version": 1,
                "level": certificate_level,
                "motion_id": motion_id,
                "scope": scope,
                "variant": variant,
                "npz_sha256": npz_sha256,
                "status": "pass",
            },
        )
        certificates.append(
            {
                "level": certificate_level,
                "path": certificate_path.relative_to(repo).as_posix(),
                "sha256": certificate_sha,
                "status": "pass",
            }
        )
    evidence_path = repo / "manifests" / scope / f"{motion_id}.evidence.json"
    evidence_sha = _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "motion_id": motion_id,
            "scope": scope,
            "variant": variant,
            "npz_sha256": npz_sha256,
            "highest_evidence_level": level,
            "certificates": certificates,
        },
    )
    return evidence_path, evidence_sha


def _repo_fixture(tmp_path: Path, *, scope: str = "upper"):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    ready_path = repo / "assets" / "ready" / "canonical_ready_v1.npz"
    ready_sha = _write_ready(ready_path)
    ready_fk_path = repo / "assets" / "ready" / "canonical_ready_fk_v1.npz"
    ready_fk_sha = _write_ready_fk(
        ready_fk_path, canonical_ready_sha256=ready_sha
    )
    entries = []
    for index, motion_id in enumerate(registry.CANONICAL_MOTION_IDS):
        npz_path = repo / "assets" / scope / f"{motion_id}.npz"
        npz_sha = _write_npz(npz_path, seed=index)
        source_path = repo / "manifests" / scope / f"{motion_id}.source.json"
        source_sha = _write_json(
            source_path,
            {"motion_id": motion_id, "source": f"synthetic-{index}"},
        )
        build_path = repo / "manifests" / scope / f"{motion_id}.build.json"
        build_sha = _write_json(
            build_path,
            {
                "hashes": {
                    "output_npz_sha256": npz_sha,
                    "ready_sha256": ready_sha,
                },
                "publication_class": "compiler_candidate",
                "training_authorized": False,
            },
        )
        applicability_path = (
            repo / "manifests" / scope / f"{motion_id}.applicability.json"
        )
        applicability_sha = _write_json(
            applicability_path,
            {
                "schema_version": 1,
                "motion_id": motion_id,
                "domain": "synthetic-unit-test",
                "scope": scope,
                "variant": "candidate_v1",
                "npz_sha256": npz_sha,
            },
        )
        evidence_path, evidence_sha = _write_evidence_bundle(
            repo,
            motion_id=motion_id,
            scope=scope,
            variant="candidate_v1",
            npz_sha256=npz_sha,
            level="E1",
        )
        entries.append(
            {
                "motion_id": motion_id,
                "scope": scope,
                "variant": "candidate_v1",
                "npz_path": npz_path.relative_to(repo).as_posix(),
                "npz_sha256": npz_sha,
                "frames": 5,
                "fps": 50.0,
                "family": _FAMILIES[index],
                "strike_marker_frame": 2,
                "contact_opportunity_frames": [1, 3],
                "mount_normal_sign": _SIGNS[index],
                "canonical_ready_sha256": ready_sha,
                "source_manifest_path": source_path.relative_to(repo).as_posix(),
                "source_manifest_sha256": source_sha,
                "build_manifest_path": build_path.relative_to(repo).as_posix(),
                "build_manifest_sha256": build_sha,
                "applicability_manifest_path": (
                    applicability_path.relative_to(repo).as_posix()
                ),
                "applicability_manifest_sha256": applicability_sha,
                "evidence_level": "E1",
                "evidence_manifest_path": evidence_path.relative_to(repo).as_posix(),
                "evidence_manifest_sha256": evidence_sha,
                "question_bank_path": None,
                "question_bank_sha256": None,
                "question_bank_schema_version": None,
                "training_config_path": None,
                "training_config_sha256": None,
                "training_config_schema_version": None,
                "onnx_model_path": None,
                "onnx_model_sha256": None,
                "onnx_model_schema_version": None,
                "onnx_metadata_path": None,
                "onnx_metadata_sha256": None,
                "onnx_metadata_schema_version": None,
                "adoption_manifest_path": None,
                "adoption_manifest_sha256": None,
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            }
        )
    document = {
        "schema_version": 1,
        "bank_id": f"canonical_{scope}_v1",
        "scope": scope,
        "canonical_ready_path": ready_path.relative_to(repo).as_posix(),
        "canonical_ready_sha256": ready_sha,
        "canonical_ready_fk_path": ready_fk_path.relative_to(repo).as_posix(),
        "canonical_ready_fk_sha256": ready_fk_sha,
        "entries": entries,
    }
    path = repo / "configs" / f"{scope}_bank.json"
    _write_json(path, document)
    return repo, path, document


def _rewrite_registry(path: Path, document) -> None:
    _write_json(path, document)


def _rebind_entry_npz(repo: Path, document, index: int, arrays) -> None:
    entry = document["entries"][index]
    npz_path = repo / entry["npz_path"]
    np.savez(npz_path, **arrays)
    entry["npz_sha256"] = _sha256(npz_path)
    build_path = repo / entry["build_manifest_path"]
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["hashes"]["output_npz_sha256"] = entry["npz_sha256"]
    entry["build_manifest_sha256"] = _write_json(build_path, build)


def _attach_versioned_artifact(
    repo: Path,
    document,
    entry,
    artifact_name: str,
    *,
    schema_version: int,
) -> None:
    if artifact_name == "question_bank":
        path = repo / "adoption" / f"{entry['motion_id']}.question_bank.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        motion_id = entry["motion_id"]
        meta = {
            "schema_version": 3,
            "split": "train",
            "clip_order": [motion_id],
            "clips": {
                motion_id: {
                    "motion_sha256": entry["npz_sha256"],
                    "n_frames": entry["frames"],
                    "anchor_frame": entry["strike_marker_frame"],
                }
            },
        }
        vector = np.zeros((1, 3), dtype=np.float32)
        np.savez(
            path,
            meta_json=np.frombuffer(
                json.dumps(
                    meta,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
            **{
                f"{motion_id}/contact_pos_env": np.zeros(3, dtype=np.float32),
                f"{motion_id}/incoming_vel": vector,
                f"{motion_id}/incoming_spin": vector,
                f"{motion_id}/demanded_vel": vector,
                f"{motion_id}/demanded_normal": vector,
            },
        )
    elif artifact_name == "training_config":
        path = repo / "adoption" / f"{entry['motion_id']}.training_config.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "contract": "canonical-motion-training-config-v1",
                "motion_id": entry["motion_id"],
                "scope": entry["scope"],
                "variant": entry["variant"],
                "npz_sha256": entry["npz_sha256"],
            },
        )
    elif artifact_name == "onnx_model":
        path = repo / "adoption" / f"{entry['motion_id']}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic model bytes; strict checker must reject\n")
    elif artifact_name == "onnx_metadata":
        path = repo / "adoption" / f"{entry['motion_id']}.onnx_metadata.json"
        _write_json(
            path,
            {
                "hope_metadata_schema_version": 2,
                "motion_id": entry["motion_id"],
                "scope": entry["scope"],
                "variant": entry["variant"],
                "npz_sha256": entry["npz_sha256"],
                "strike_marker_frame": entry["strike_marker_frame"],
                "contact_opportunity_frames": entry[
                    "contact_opportunity_frames"
                ],
                "mount_normal_sign": entry["mount_normal_sign"],
                "canonical_ready_sha256": entry["canonical_ready_sha256"],
                "canonical_ready_fk_sha256": document[
                    "canonical_ready_fk_sha256"
                ],
                "question_bank_sha256": entry["question_bank_sha256"],
                "training_config_sha256": entry["training_config_sha256"],
                "onnx_model_sha256": entry["onnx_model_sha256"],
            },
        )
    else:
        raise AssertionError(f"unknown artifact {artifact_name}")
    entry[f"{artifact_name}_path"] = path.relative_to(repo).as_posix()
    entry[f"{artifact_name}_sha256"] = _sha256(path)
    entry[f"{artifact_name}_schema_version"] = schema_version


def _set_entry_evidence(repo: Path, entry, level: str) -> None:
    entry["evidence_level"] = level
    evidence_path, evidence_sha = _write_evidence_bundle(
        repo,
        motion_id=entry["motion_id"],
        scope=entry["scope"],
        variant=entry["variant"],
        npz_sha256=entry["npz_sha256"],
        level=level,
    )
    entry["evidence_manifest_path"] = evidence_path.relative_to(repo).as_posix()
    entry["evidence_manifest_sha256"] = evidence_sha


def _write_entry_adoption_manifest(repo: Path, document, entry) -> None:
    artifacts = {}
    for artifact_name in (
        "question_bank",
        "training_config",
        "onnx_model",
        "onnx_metadata",
    ):
        artifact_sha = entry[f"{artifact_name}_sha256"]
        artifacts[artifact_name] = (
            None
            if artifact_sha is None
            else {
                "sha256": artifact_sha,
                "schema_version": entry[f"{artifact_name}_schema_version"],
            }
        )
    path = repo / "adoption" / f"{entry['motion_id']}.adoption.json"
    entry["adoption_manifest_path"] = path.relative_to(repo).as_posix()
    entry["adoption_manifest_sha256"] = _write_json(
        path,
        {
            "schema_version": 1,
            "motion_id": entry["motion_id"],
            "scope": entry["scope"],
            "variant": entry["variant"],
            "npz_sha256": entry["npz_sha256"],
            "strike_marker_frame": entry["strike_marker_frame"],
            "contact_opportunity_frames": entry["contact_opportunity_frames"],
            "mount_normal_sign": entry["mount_normal_sign"],
            "canonical_ready_sha256": entry["canonical_ready_sha256"],
            "canonical_ready_fk_sha256": document[
                "canonical_ready_fk_sha256"
            ],
            "artifacts": artifacts,
        },
    )


def _adopt_document_for_training(repo: Path, document) -> None:
    for entry in document["entries"]:
        entry["publication_class"] = "training_adopted"
        entry["training_authorized"] = True
        _set_entry_evidence(repo, entry, "E2")
        _attach_versioned_artifact(
            repo, document, entry, "question_bank", schema_version=3
        )
        _attach_versioned_artifact(
            repo, document, entry, "training_config", schema_version=1
        )
        _write_entry_adoption_manifest(repo, document, entry)


def _write_blob_receipt(repo: Path, relative: str, label: str):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fixture:{label}\n".encode("utf-8"))
    return {
        "path": relative,
        "sha256": _sha256(path),
    }


def _complete_bank_gate_report(
    binding,
    repo: Path,
    ready_path: Path,
):
    digest = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    manifest = _write_blob_receipt(
        repo, "trusted/bank/BUILD_MANIFEST.json", "manifest"
    )
    recipe = _write_blob_receipt(repo, "trusted/recipe.json", "recipe")
    compiler = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_compiler.py",
        "compiler",
    )
    geometry = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_geometry.py",
        "geometry",
    )
    mjcf = _write_blob_receipt(repo, "trusted/a3.xml", "mjcf")
    urdf = _write_blob_receipt(repo, "trusted/a3.urdf", "urdf")
    body_order = _write_blob_receipt(
        repo, "trusted/body_order.txt", "body-order"
    )
    gate_tool = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_bank_gate.py",
        "gate",
    )
    player_tool = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py",
        "player",
    )
    dynamics_tool = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_mujoco_dynamics_gate.py",
        "dynamics",
    )
    schema2_tool = _write_blob_receipt(
        repo,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_schema2_builder.py",
        "schema2-builder",
    )
    ready_receipt = {
        "path": ready_path.relative_to(repo).as_posix(),
        "sha256": _sha256(ready_path),
    }
    endpoint_zero = {
        "joint_start": True,
        "joint_end": True,
        "body_linear_start": True,
        "body_linear_end": True,
        "body_angular_start": True,
        "body_angular_end": True,
    }
    by_motion = dict(zip(binding.motion_ids, binding.npz_sha256))
    clips = [
        {
            "motion_id": motion_id,
            "scope": scope,
            "filename": f"{motion_id}_{scope}.npz",
            "sha256": (
                by_motion[motion_id]
                if scope == binding.scope
                else digest(f"{motion_id}:{scope}")
            ),
            "frames": 5,
            "fps": 50.0,
            "duration_s": 0.08,
            "schema2_receipts": {
                "input_sha256": digest(f"{motion_id}:{scope}:input"),
                "builder_tool_sha256": schema2_tool["sha256"],
                "manifest_sidecar": _write_blob_receipt(
                    repo,
                    f"trusted/bank/{motion_id}_{scope}.manifest.json",
                    f"{motion_id}:{scope}:manifest",
                ),
                "report_sidecar": _write_blob_receipt(
                    repo,
                    f"trusted/bank/{motion_id}_{scope}.report.json",
                    f"{motion_id}:{scope}:report",
                ),
            },
            "strict_schema2_and_ready": {
                "shared_joint_ready_exact": True,
                "shared_32_body_ready_exact": True,
                "six_velocity_classes_exact_zero": endpoint_zero,
            },
            "contact_opportunity": {
                "acceleration_allowed_through_window_end": True
            },
            "mujoco_fk": {"pass": True},
            "plant_specific_dynamics": {
                "verdict": "PASS",
                "screen_pass": True,
                "non_torque_screens_pass": True,
                "inverse_dynamics": {
                    "torque_interpretation": {"valid": True}
                },
            },
        }
        for motion_id in binding.motion_ids
        for scope in ("upper", "full")
    ]
    aggregate = {
        key: 0
        for key in registry.motion_admission._BANK_GATE_AGGREGATE_KEYS
    }
    aggregate.update(
        {
            key: 10
            for key in (
                "clip_count",
                "fk_pass_count",
                "velocity_consistency_pass_count",
                "joint_limit_pass_count",
                "geometry_pass_count",
                "non_torque_dynamics_pass_count",
                "complete_dynamics_pass_count",
                "torque_interpretation_valid_count",
            )
        }
    )
    return {
        "schema_version": 1,
        "verdict": "PASS",
        "bank_gate_pass": True,
        "candidate_integrity_pass": True,
        "grounded_trace_status": "COMPLETE_PASS",
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "library_id": binding.bank_id,
        "manifest": manifest,
        "bank_dir": "bank",
        "bound_inputs": {
            "recipe": recipe,
            "compiler": compiler,
            "geometry_tool": geometry,
            "compiler_options_sha256": digest("options"),
            "ready": ready_receipt,
            "mjcf": mjcf,
            "urdf": urdf,
            "body_order": body_order,
            "plant": {
                "mjcf_sha256": mjcf["sha256"],
                "urdf_sha256": urdf["sha256"],
                "compiled_signature_sha256": digest("signature"),
                "identity_bound": True,
                "runtime_body_order": ["pelvis_link"],
            },
            "verifier_tools": {
                "bank_gate": gate_tool,
                "mujoco_motion_player": player_tool,
                "canonical_mujoco_dynamics_gate": {
                    **dynamics_tool,
                    "report_schema_version": 1,
                },
            },
        },
        "contracts": {
            "matrix": {
                "motion_ids": list(binding.motion_ids),
                "scopes": ["upper", "full"],
                "count": 10,
            },
            "shared_ready": True,
            "six_endpoint_velocity_classes_exact_zero": True,
            "contact_opportunity_is_marker_only": True,
            "acceleration_allowed_through_window_end": True,
            "nonnegative_scalar_acceleration_through_window_end": True,
            "adv2c3_role": "comparator_only_not_default",
            "grounded_inverse_dynamics": "complete",
            "grounded_trace_status": "COMPLETE_PASS",
        },
        "aggregate": aggregate,
        "clips": clips,
        "non_claims": [],
    }


def _trusted_training_admission(
    repo: Path,
    loaded: registry.CanonicalMotionBankRegistry,
    monkeypatch,
):
    """Test-only code-root insertion for an independently bound bank receipt."""

    binding = registry.bank_promotion_binding(
        loaded, authorization_purpose="training"
    )
    gate_path = repo / "trusted" / f"{loaded.bank_id}.bank_gate.json"
    gate_sha = _write_json(
        gate_path,
        _complete_bank_gate_report(
            binding,
            repo,
            loaded.canonical_ready_path,
        ),
    )
    certificate_path = (
        repo / "trusted" / f"{loaded.bank_id}.training-promotion.json"
    )
    certificate_sha = _write_json(
        certificate_path,
        {
            "schema_version": 1,
            "certificate_type": "canonical-motion-bank-promotion-v1",
            **registry.motion_admission._binding_document(binding),
            "bank_gate_report": {
                "path": gate_path.relative_to(repo).as_posix(),
                "sha256": gate_sha,
            },
        },
    )
    monkeypatch.setattr(
        registry.motion_admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )
    return registry.verify_registry_promotion_certificate(
        loaded,
        certificate_path,
        authorization_purpose="training",
    )


def test_upper_registry_atomically_builds_four_aligned_runtime_tables(tmp_path: Path):
    repo, path, _ = _repo_fixture(tmp_path, scope="upper")
    loaded = registry.load_canonical_motion_bank_registry(path, repo_root=repo)
    tables = registry.adapt_registry_for_runtime(
        loaded, authorization_purpose=None
    )

    assert loaded.scope == tables.scope == "upper"
    assert loaded.registry_sha256 == tables.registry_sha256 == _sha256(path)
    assert tables.motion_ids == registry.CANONICAL_MOTION_IDS
    assert tables.clip_family_per_clip == _FAMILIES
    assert tables.mount_normal_sign_per_clip == _SIGNS
    assert tables.strike_phase_per_clip == (0.5,) * 5
    assert tables.contact_opportunity_frames_per_clip == ((1, 3),) * 5
    assert tables.publication_class_per_clip == ("compiler_candidate",) * 5
    assert tables.source_manifest_sha256_per_clip == tuple(
        row.source_manifest_sha256 for row in loaded.entries
    )
    assert tables.build_manifest_sha256_per_clip == tuple(
        row.build_manifest_sha256 for row in loaded.entries
    )
    assert tables.training_authorized_per_clip == (False,) * 5
    assert tables.deployment_authorized_per_clip == (False,) * 5
    assert tables.hardware_authorized_per_clip == (False,) * 5
    assert tables.npz_sha256_per_clip == tuple(
        row.npz_sha256 for row in loaded.entries
    )
    assert tables.authorization_purpose is None
    assert tables.canonical_ready_sha256 == loaded.canonical_ready_sha256
    assert tables.canonical_ready_path == str(loaded.canonical_ready_path)
    assert tables.canonical_ready_fk_sha256 == loaded.canonical_ready_fk_sha256
    assert tables.canonical_ready_fk_path == str(loaded.canonical_ready_fk_path)
    assert tables.ready_runtime_sha256 == loaded.entries[0].ready_runtime_sha256
    assert all(Path(value).is_absolute() for value in tables.motion_file)
    assert all(Path(value).is_file() for value in tables.motion_file)
    assert len(tables.alignment_sha256) == 64
    assert registry.adapt_registry_for_runtime(
        loaded, authorization_purpose=None
    ).alignment_sha256 == tables.alignment_sha256
    with pytest.raises(
        registry.MotionRegistryError, match="identity-only registry audit"
    ):
        tables.as_python_loader_tables()


def test_upper_and_full_are_two_independent_banks_with_same_id_order(tmp_path: Path):
    upper_repo, upper_path, _ = _repo_fixture(tmp_path / "upper", scope="upper")
    full_repo, full_path, _ = _repo_fixture(tmp_path / "full", scope="full")
    upper = registry.load_runtime_tables(
        upper_path, repo_root=upper_repo, authorization_purpose=None
    )
    full = registry.load_runtime_tables(
        full_path, repo_root=full_repo, authorization_purpose=None
    )

    assert upper.scope == "upper"
    assert full.scope == "full"
    assert upper.motion_ids == full.motion_ids == registry.CANONICAL_MOTION_IDS
    assert upper.registry_sha256 != full.registry_sha256
    assert upper.alignment_sha256 != full.alignment_sha256
    assert upper.motion_file != full.motion_file


def test_runtime_adoption_is_pinned_and_training_authorized_by_default(
    tmp_path: Path, monkeypatch,
):
    repo, path, document = _repo_fixture(tmp_path)
    unpinned = registry.load_canonical_motion_bank_registry(path, repo_root=repo)
    with pytest.raises(
        registry.MotionRegistryError, match="expected_registry_sha256"
    ):
        registry.adapt_registry_for_runtime(unpinned)

    registry_sha = _sha256(path)
    pinned = registry.load_canonical_motion_bank_registry(
        path,
        repo_root=repo,
        expected_registry_sha256=registry_sha,
    )
    with pytest.raises(registry.MotionRegistryError, match="trusted admission"):
        registry.adapt_registry_for_runtime(
            pinned,
            expected_alignment_sha256=registry._alignment_sha256(pinned),
            expected_canonical_ready_sha256=pinned.canonical_ready_sha256,
            expected_canonical_ready_fk_sha256=pinned.canonical_ready_fk_sha256,
        )
    candidate_admission = _trusted_training_admission(
        repo, pinned, monkeypatch
    )
    with pytest.raises(
        registry.MotionRegistryError, match="training authorization is false"
    ):
        registry.adapt_registry_for_runtime(
            pinned,
            expected_alignment_sha256=registry._alignment_sha256(pinned),
            expected_canonical_ready_sha256=pinned.canonical_ready_sha256,
            expected_canonical_ready_fk_sha256=pinned.canonical_ready_fk_sha256,
            admission=candidate_admission,
        )
    forged = replace(
        pinned,
        entries=tuple(
            replace(row, training_authorized=True) for row in pinned.entries
        ),
    )
    with pytest.raises(
        registry.MotionRegistryError, match="fresh strict parse"
    ):
        registry.adapt_registry_for_runtime(forged)

    _adopt_document_for_training(repo, document)
    _rewrite_registry(path, document)
    registry_sha = _sha256(path)
    audit = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=registry_sha,
        authorization_purpose=None,
    )
    with pytest.raises(
        registry.MotionRegistryError, match="requires expected_alignment_sha256"
    ):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256=registry_sha,
        )
    with pytest.raises(
        registry.MotionRegistryError,
        match="requires expected_canonical_ready_sha256",
    ):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256=registry_sha,
            expected_alignment_sha256=audit.alignment_sha256,
        )
    with pytest.raises(registry.MotionRegistryError, match="trusted admission"):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256=registry_sha,
            expected_alignment_sha256=audit.alignment_sha256,
            expected_canonical_ready_sha256=audit.canonical_ready_sha256,
            expected_canonical_ready_fk_sha256=audit.canonical_ready_fk_sha256,
        )
    adopted = registry.load_canonical_motion_bank_registry(
        path,
        repo_root=repo,
        expected_registry_sha256=registry_sha,
    )
    trusted_admission = _trusted_training_admission(
        repo, adopted, monkeypatch
    )
    tables = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=registry_sha,
        expected_alignment_sha256=audit.alignment_sha256,
        expected_canonical_ready_sha256=audit.canonical_ready_sha256,
        expected_canonical_ready_fk_sha256=audit.canonical_ready_fk_sha256,
        admission=trusted_admission,
    )
    assert tables.authorization_purpose == "training"
    assert tables.publication_class_per_clip == ("training_adopted",) * 5
    assert tables.training_authorized_per_clip == (True,) * 5
    assert tables.deployment_authorized_per_clip == (False,) * 5
    assert tables.evidence_level_per_clip == ("E2",) * 5
    assert tables.question_bank_schema_version_per_clip == (3,) * 5
    assert tables.training_config_schema_version_per_clip == (1,) * 5
    assert tables.onnx_model_schema_version_per_clip == (None,) * 5
    assert tables.onnx_metadata_schema_version_per_clip == (None,) * 5
    columns = tables.as_python_loader_tables()
    assert frozenset(columns) == {
        "motion_file",
        "clip_family_per_clip",
        "strike_phase_per_clip",
        "contact_opportunity_frames_per_clip",
        "mount_normal_sign_per_clip",
        "motion_ids",
        "publication_class_per_clip",
        "source_manifest_path_per_clip",
        "source_manifest_sha256_per_clip",
        "build_manifest_path_per_clip",
        "build_manifest_sha256_per_clip",
        "applicability_manifest_path_per_clip",
        "applicability_manifest_sha256_per_clip",
        "evidence_level_per_clip",
        "evidence_manifest_path_per_clip",
        "evidence_manifest_sha256_per_clip",
        "question_bank_path_per_clip",
        "question_bank_sha256_per_clip",
        "question_bank_schema_version_per_clip",
        "training_config_path_per_clip",
        "training_config_sha256_per_clip",
        "training_config_schema_version_per_clip",
        "onnx_model_path_per_clip",
        "onnx_model_sha256_per_clip",
        "onnx_model_schema_version_per_clip",
        "onnx_metadata_path_per_clip",
        "onnx_metadata_sha256_per_clip",
        "onnx_metadata_schema_version_per_clip",
        "adoption_manifest_path_per_clip",
        "adoption_manifest_sha256_per_clip",
        "training_authorized_per_clip",
        "deployment_authorized_per_clip",
        "hardware_authorized_per_clip",
        "npz_sha256_per_clip",
    }
    assert {len(value) for value in columns.values()} == {5}
    with pytest.raises(TypeError):
        columns["motion_ids"] = ()
    with pytest.raises(
        registry.MotionRegistryError, match="deployment.*trusted admission"
    ):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256=registry_sha,
            expected_alignment_sha256=audit.alignment_sha256,
            expected_canonical_ready_sha256=audit.canonical_ready_sha256,
            expected_canonical_ready_fk_sha256=audit.canonical_ready_fk_sha256,
            authorization_purpose="deployment",
        )


def test_canonical_ready_is_a_real_bound_file_and_matches_every_endpoint(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    ready_path = repo / document["canonical_ready_path"]
    ready_path.write_bytes(ready_path.read_bytes() + b"tamper")
    with pytest.raises(
        registry.MotionRegistryError, match="canonical ready SHA-256 mismatch"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "endpoint")
    arrays = _schema2_arrays(seed=0)
    arrays["joint_pos"][[0, -1], 0] = np.float32(0.125)
    _rebind_entry_npz(repo, document, 0, arrays)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="endpoint joint_pos does not exactly equal the bound canonical ready",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_all_five_npzs_share_the_full_runtime_ready_body_pose(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    arrays = _schema2_arrays(seed=1)
    arrays["body_pos_w"][[0, -1], 1, 0] = np.float32(0.125)
    _rebind_entry_npz(repo, document, 1, arrays)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="pinned canonical-ready FK truth",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_all_five_same_wrong_nonroot_fk_cannot_redefine_ready_truth(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    for index in range(5):
        arrays = _schema2_arrays(seed=index)
        arrays["body_pos_w"][[0, -1], 7, 1] = np.float32(0.375)
        _rebind_entry_npz(repo, document, index, arrays)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="pinned canonical-ready FK truth",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_ready_fk_truth_is_content_addressed_and_binds_base_ready(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    ready_fk_path = repo / document["canonical_ready_fk_path"]
    ready_fk_path.write_bytes(ready_fk_path.read_bytes() + b"tamper")
    with pytest.raises(
        registry.MotionRegistryError, match="canonical ready FK truth SHA-256 mismatch"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "wrong-ready-binding")
    ready_fk_path = repo / document["canonical_ready_fk_path"]
    document["canonical_ready_fk_sha256"] = _write_ready_fk(
        ready_fk_path,
        canonical_ready_sha256="0" * 64,
    )
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="does not bind the canonical ready SHA-256",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_expected_registry_and_alignment_digests_fail_closed(tmp_path: Path):
    repo, path, _ = _repo_fixture(tmp_path)
    actual_registry_sha = _sha256(path)
    tables = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=actual_registry_sha,
        authorization_purpose=None,
    )
    pinned = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=actual_registry_sha,
        expected_alignment_sha256=tables.alignment_sha256,
        authorization_purpose=None,
    )
    assert pinned == tables

    with pytest.raises(registry.MotionRegistryError, match="registry SHA-256 mismatch"):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256="0" * 64,
            authorization_purpose=None,
        )
    with pytest.raises(registry.MotionRegistryError, match="alignment SHA-256 mismatch"):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_alignment_sha256="0" * 64,
            authorization_purpose=None,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda doc: doc.update({"extra": True}),
            "registry keys changed",
        ),
        (
            lambda doc: doc["entries"][0].update({"extra": True}),
            r"entries\[0\] keys changed",
        ),
        (
            lambda doc: doc["entries"].__setitem__(
                0, {**doc["entries"][0], "motion_id": "bh_loop_c"}
            ),
            "preserve canonical order",
        ),
        (
            lambda doc: doc["entries"][0].__setitem__("scope", "full"),
            "differs from bank scope",
        ),
        (
            lambda doc: doc["entries"][0].__setitem__(
                "canonical_ready_sha256", "b" * 64
            ),
            "differs from the bank common ready",
        ),
        (
            lambda doc: doc["entries"][0].__setitem__(
                "contact_opportunity_frames", [3, 4]
            ),
            "contact_start <= strike_marker",
        ),
        (
            lambda doc: doc["entries"][0].__setitem__("mount_normal_sign", 1),
            "explicit JSON float",
        ),
        (
            lambda doc: doc["entries"][0].update(
                {
                    "training_authorized": False,
                    "deployment_authorized": True,
                }
            ),
            "deployment authorization requires training",
        ),
        (
            lambda doc: doc["entries"][0].update(
                {
                    "training_authorized": True,
                    "deployment_authorized": False,
                    "hardware_authorized": True,
                }
            ),
            "hardware authorization requires deployment",
        ),
        (
            lambda doc: doc["entries"][0].update(
                {"training_authorized": True}
            ),
            "publication_class='compiler_candidate' requires authorization",
        ),
        (
            lambda doc: doc["entries"][0].update(
                {"publication_class": "self_declared_release"}
            ),
            "publication_class must be one of",
        ),
    ],
)
def test_registry_cross_field_and_exact_key_errors_fail_closed(
    tmp_path: Path, mutate, message: str
):
    repo, path, document = _repo_fixture(tmp_path)
    mutate(document)
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match=message):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_publication_state_machine_requires_evidence_and_bound_artifacts(
    tmp_path: Path, monkeypatch,
):
    repo, path, document = _repo_fixture(tmp_path)
    for entry in document["entries"]:
        entry["publication_class"] = "training_adopted"
        entry["training_authorized"] = True
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="requires evidence >=E2"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    for entry in document["entries"]:
        _set_entry_evidence(repo, entry, "E2")
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError, match="lacks required versioned artifacts"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    _adopt_document_for_training(repo, document)
    _rewrite_registry(path, document)
    training_sha = _sha256(path)
    training_audit = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=training_sha,
        authorization_purpose=None,
    )
    with pytest.raises(registry.MotionRegistryError, match="trusted admission"):
        registry.load_runtime_tables(
            path,
            repo_root=repo,
            expected_registry_sha256=training_sha,
            expected_alignment_sha256=training_audit.alignment_sha256,
            expected_canonical_ready_sha256=(
                training_audit.canonical_ready_sha256
            ),
            expected_canonical_ready_fk_sha256=(
                training_audit.canonical_ready_fk_sha256
            ),
            authorization_purpose="training",
        )
    loaded = registry.load_canonical_motion_bank_registry(
        path, repo_root=repo, expected_registry_sha256=training_sha
    )
    trusted_admission = _trusted_training_admission(
        repo, loaded, monkeypatch
    )
    training = registry.load_runtime_tables(
        path,
        repo_root=repo,
        expected_registry_sha256=training_sha,
        expected_alignment_sha256=training_audit.alignment_sha256,
        expected_canonical_ready_sha256=(
            training_audit.canonical_ready_sha256
        ),
        expected_canonical_ready_fk_sha256=(
            training_audit.canonical_ready_fk_sha256
        ),
        authorization_purpose="training",
        admission=trusted_admission,
    )
    assert training.authorization_purpose == "training"

    for entry in document["entries"]:
        entry["publication_class"] = "deployment_adopted"
        entry["deployment_authorized"] = True
        _set_entry_evidence(repo, entry, "E4")
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError, match=r"lacks required.*onnx_metadata"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    for entry in document["entries"]:
        _attach_versioned_artifact(
            repo, document, entry, "onnx_model", schema_version=1
        )
        _attach_versioned_artifact(
            repo, document, entry, "onnx_metadata", schema_version=2
        )
        _write_entry_adoption_manifest(repo, document, entry)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match=(
            "strict ONNX parser is unavailable|fails strict ONNX parse/check"
        ),
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    for entry in document["entries"]:
        entry["publication_class"] = "hardware_adopted"
        entry["hardware_authorized"] = True
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="requires evidence >=E5"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    for entry in document["entries"]:
        _set_entry_evidence(repo, entry, "E5")
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match=(
            "strict ONNX parser is unavailable|fails strict ONNX parse/check"
        ),
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_adoption_artifact_bindings_are_all_or_none_and_content_addressed(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    entry = document["entries"][0]
    entry["question_bank_path"] = "adoption/missing.npz"
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="must be all-null"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "tamper")
    _adopt_document_for_training(repo, document)
    entry = document["entries"][0]
    artifact = repo / entry["question_bank_path"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError, match="question_bank SHA-256 mismatch"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_onnx_metadata_binds_the_exact_model_provenance(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    _adopt_document_for_training(repo, document)
    for entry in document["entries"]:
        entry["publication_class"] = "deployment_adopted"
        entry["deployment_authorized"] = True
        _set_entry_evidence(repo, entry, "E4")
        _attach_versioned_artifact(
            repo, document, entry, "onnx_model", schema_version=1
        )
        _attach_versioned_artifact(
            repo, document, entry, "onnx_metadata", schema_version=2
        )
        _write_entry_adoption_manifest(repo, document, entry)

    entry = document["entries"][0]
    model_path = repo / entry["onnx_model_path"]
    model_path.write_bytes(model_path.read_bytes() + b"other-model")
    entry["onnx_model_sha256"] = _sha256(model_path)
    _write_entry_adoption_manifest(repo, document, entry)
    _rewrite_registry(path, document)

    with pytest.raises(
        registry.MotionRegistryError,
        match="onnx_metadata does not exactly bind runtime identity/timing/artifacts",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_lineage_and_adoption_sidecars_prevent_cross_slot_evidence_reuse(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    first, second = document["entries"][:2]
    first["evidence_manifest_path"] = second["evidence_manifest_path"]
    first["evidence_manifest_sha256"] = second["evidence_manifest_sha256"]
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="evidence manifest does not bind exact motion lineage/level",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "artifact-swap")
    _adopt_document_for_training(repo, document)
    first, second = document["entries"][:2]
    for suffix in ("path", "sha256", "schema_version"):
        key = f"question_bank_{suffix}"
        first[key], second[key] = second[key], first[key]
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match=(
            "lacks schema-v3 motion rows|clip_order does not contain|"
            "adoption manifest does not bind question_bank"
        ),
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_unsupported_artifact_schema_version_cannot_be_self_declared(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    _adopt_document_for_training(repo, document)
    for entry in document["entries"]:
        entry["publication_class"] = "deployment_adopted"
        entry["deployment_authorized"] = True
        _set_entry_evidence(repo, entry, "E4")
        artifact = repo / "adoption" / "not_onnx_metadata.bin"
        artifact.write_bytes(b"garbage\n")
        entry["onnx_metadata_path"] = artifact.relative_to(repo).as_posix()
        entry["onnx_metadata_sha256"] = _sha256(artifact)
        entry["onnx_metadata_schema_version"] = 999
        _write_entry_adoption_manifest(repo, document, entry)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError,
        match="onnx_metadata_schema_version=999 is unsupported",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_supported_artifact_version_does_not_make_garbage_deployable(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    _adopt_document_for_training(repo, document)
    for entry in document["entries"]:
        entry["publication_class"] = "deployment_adopted"
        entry["deployment_authorized"] = True
        _set_entry_evidence(repo, entry, "E4")
        _attach_versioned_artifact(
            repo, document, entry, "onnx_model", schema_version=1
        )
        _attach_versioned_artifact(
            repo, document, entry, "onnx_metadata", schema_version=2
        )

    first = document["entries"][0]
    artifact = repo / "adoption" / "garbage_but_claimed_v2.onnx_metadata.json"
    artifact.write_bytes(b"garbage\n")
    first["onnx_metadata_path"] = artifact.relative_to(repo).as_posix()
    first["onnx_metadata_sha256"] = _sha256(artifact)
    first["onnx_metadata_schema_version"] = 2
    for entry in document["entries"]:
        _write_entry_adoption_manifest(repo, document, entry)
    _rewrite_registry(path, document)

    with pytest.raises(
        registry.MotionRegistryError,
        match="onnx_metadata is not strict UTF-8 JSON",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_evidence_level_requires_content_addressed_certificate_chain(
    tmp_path: Path,
):
    repo, path, document = _repo_fixture(tmp_path)
    entry = document["entries"][0]
    _set_entry_evidence(repo, entry, "E2")
    evidence_path = repo / entry["evidence_manifest_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["certificates"].pop()
    entry["evidence_manifest_sha256"] = _write_json(evidence_path, evidence)
    _rewrite_registry(path, document)

    with pytest.raises(
        registry.MotionRegistryError,
        match="must contain one passing certificate for every level",
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_duplicate_json_keys_and_nonfinite_json_fail_closed(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    valid_entries = json.dumps(document["entries"], allow_nan=False)
    path.write_text(
        (
            '{"schema_version":1,"schema_version":1,'
            f'"bank_id":"x","scope":"upper","canonical_ready_path":"ready.npz",'
            f'"canonical_ready_sha256":"{_DUMMY_SHA}",'
            f'"entries":{valid_entries}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(registry.MotionRegistryError, match="duplicate key"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    document["entries"][0]["fps"] = float("nan")
    path.write_text(json.dumps(document, allow_nan=True), encoding="utf-8")
    with pytest.raises(registry.MotionRegistryError, match="non-finite constant"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_path_traversal_and_symlink_escape_fail_closed(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    document["entries"][0]["npz_path"] = "../escape.npz"
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="contain '.'/'..'"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "symlink")
    outside = tmp_path / "outside.npz"
    outside.write_bytes(
        (repo / document["entries"][0]["npz_path"]).read_bytes()
    )
    link = repo / "assets" / "escaped.npz"
    link.symlink_to(outside)
    document["entries"][0]["npz_path"] = link.relative_to(repo).as_posix()
    document["entries"][0]["npz_sha256"] = _sha256(outside)
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="escapes repository root"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_npz_and_manifest_hashes_are_verified_from_bound_bytes(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    npz = repo / document["entries"][0]["npz_path"]
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(registry.MotionRegistryError, match="NPZ SHA-256 mismatch"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    repo, path, document = _repo_fixture(tmp_path / "manifest")
    source = repo / document["entries"][0]["source_manifest_path"]
    source.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(
        registry.MotionRegistryError, match="source manifest SHA-256 mismatch"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_build_manifest_must_bind_entry_npz_and_common_ready(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    build = repo / document["entries"][0]["build_manifest_path"]
    bad = {
        "hashes": {
            "output_npz_sha256": "0" * 64,
            "ready_sha256": document["canonical_ready_sha256"],
        }
    }
    document["entries"][0]["build_manifest_sha256"] = _write_json(build, bad)
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="does not bind the entry NPZ"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    bad["hashes"]["output_npz_sha256"] = document["entries"][0]["npz_sha256"]
    bad["hashes"]["ready_sha256"] = "b" * 64
    document["entries"][0]["build_manifest_sha256"] = _write_json(build, bad)
    _rewrite_registry(path, document)
    with pytest.raises(
        registry.MotionRegistryError, match="does not bind canonical ready"
    ):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda arrays: arrays.pop("body_ang_vel_w"),
            "field set changed",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "joint_pos", arrays["joint_pos"].astype(np.float64)
            ),
            "joint_pos must be exact float32",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "body_names",
                np.asarray(tuple(reversed(registry.RUNTIME_BODY_NAMES))),
            ),
            "differs from A3 runtime order",
        ),
        (
            lambda arrays: arrays["joint_pos"].__setitem__(
                -1, np.ones(31, dtype=np.float32)
            ),
            "same canonical ready",
        ),
        (
            lambda arrays: arrays["joint_vel"].__setitem__((0, 0), np.float32(1.0)),
            "first/last frames must be exactly zero",
        ),
    ],
)
def test_exact_schema2_contract_fails_closed(
    tmp_path: Path, mutate, message: str
):
    repo, path, document = _repo_fixture(tmp_path)
    entry = document["entries"][0]
    npz = repo / entry["npz_path"]
    arrays = _schema2_arrays()
    mutate(arrays)
    np.savez(npz, **arrays)
    entry["npz_sha256"] = _sha256(npz)
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match=message):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_registry_frames_and_fps_must_equal_npz(tmp_path: Path):
    repo, path, document = _repo_fixture(tmp_path)
    document["entries"][0]["frames"] = 6
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="differs from NPZ frames"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)

    document["entries"][0]["frames"] = 5
    document["entries"][0]["fps"] = 49.0
    _rewrite_registry(path, document)
    with pytest.raises(registry.MotionRegistryError, match="differs from NPZ fps"):
        registry.load_canonical_motion_bank_registry(path, repo_root=repo)


def test_json_schema_shape_matches_strict_loader_without_a_fake_registry():
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "canonical_motion_bank_registry_schema_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert frozenset(schema["required"]) == registry._TOP_LEVEL_KEYS
    entry = schema["$defs"]["entry"]
    assert entry["additionalProperties"] is False
    assert frozenset(entry["required"]) == registry._ENTRY_KEYS
    assert entry["properties"]["motion_id"]["enum"] == list(
        registry.CANONICAL_MOTION_IDS
    )
    assert schema["properties"]["entries"]["minItems"] == 5
    assert schema["properties"]["entries"]["maxItems"] == 5
    assert not any(
        path.name.endswith("_bank.json")
        for path in schema_path.parent.glob("canonical_*_bank.json")
    )
