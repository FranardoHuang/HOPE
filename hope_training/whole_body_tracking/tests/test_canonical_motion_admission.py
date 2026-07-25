"""Pure-CPU tests for the code-rooted motion admission capability."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "canonical_motion_admission.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "canonical_motion_admission_test", _SCRIPT
)
admission = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = admission
_SPEC.loader.exec_module(admission)

_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)


def _sha(char: str) -> str:
    return char * 64


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt_file(root: Path, relative: str, label: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fixture:{label}\n".encode("utf-8"))
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _binding(ready_sha256: str) -> admission.BankPromotionBinding:
    return admission.BankPromotionBinding(
        purpose="training",
        bank_id="canonical_upper_v1",
        scope="upper",
        registry_sha256=_sha("1"),
        alignment_sha256=_sha("2"),
        motion_ids=_MOTION_IDS,
        npz_sha256=tuple(_sha(str(index + 3)) for index in range(5)),
        canonical_ready_sha256=ready_sha256,
        canonical_ready_fk_sha256=_sha("9"),
        build_manifest_sha256=tuple(_sha(chr(ord("a") + index)) for index in range(5)),
        evidence_levels=("E2",) * 5,
        evidence_manifest_sha256=tuple(
            _sha(str(index)) for index in range(5)
        ),
        evidence_certificate_sha256=tuple(
            ((_sha("b"), _sha("c"))) for _ in range(5)
        ),
        question_bank_sha256=(_sha("d"),) * 5,
        training_config_sha256=(_sha("e"),) * 5,
        onnx_model_sha256=(None,) * 5,
        onnx_metadata_sha256=(None,) * 5,
        adoption_manifest_sha256=(_sha("f"),) * 5,
    )


def _certificate_document(
    binding: admission.BankPromotionBinding, report_path: str, report_sha: str
):
    expected = admission._binding_document(binding)
    return {
        "schema_version": 1,
        "certificate_type": "canonical-motion-bank-promotion-v1",
        **expected,
        "bank_gate_report": {"path": report_path, "sha256": report_sha},
    }


def _bank_gate_report(
    binding: admission.BankPromotionBinding,
    repo_root: Path,
    *,
    passed=True,
):
    manifest = _write_receipt_file(
        repo_root, "bank/BUILD_MANIFEST.json", "manifest"
    )
    recipe = _write_receipt_file(repo_root, "configs/recipe.json", "recipe")
    compiler = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_compiler.py",
        "compiler",
    )
    geometry = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_geometry.py",
        "geometry",
    )
    mjcf = _write_receipt_file(repo_root, "models/a3.xml", "mjcf")
    urdf = _write_receipt_file(repo_root, "models/a3.urdf", "urdf")
    body_order = _write_receipt_file(
        repo_root, "models/body_order.txt", "body-order"
    )
    gate_tool = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_bank_gate.py",
        "bank-gate",
    )
    player_tool = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py",
        "player",
    )
    dynamics_tool = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_mujoco_dynamics_gate.py",
        "dynamics",
    )
    schema2_tool = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_schema2_builder.py",
        "schema2-builder",
    )
    ready = {
        "path": "ready.npz",
        "sha256": binding.canonical_ready_sha256,
    }
    endpoint_zero = {
        "joint_start": True,
        "joint_end": True,
        "body_linear_start": True,
        "body_linear_end": True,
        "body_angular_start": True,
        "body_angular_end": True,
    }
    clips = []
    for motion_id, upper_sha in zip(
        binding.motion_ids, binding.npz_sha256
    ):
        for scope in ("upper", "full"):
            clip_sha = (
                upper_sha
                if scope == binding.scope
                else hashlib.sha256(
                    f"{motion_id}:{scope}".encode("utf-8")
                ).hexdigest()
            )
            clips.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": f"{motion_id}_{scope}.npz",
                    "sha256": clip_sha,
                    "frames": 10,
                    "fps": 50.0,
                    "duration_s": 0.18,
                    "schema2_receipts": {
                        "input_sha256": hashlib.sha256(
                            f"{motion_id}:{scope}:input".encode("utf-8")
                        ).hexdigest(),
                        "builder_tool_sha256": schema2_tool["sha256"],
                        "manifest_sidecar": _write_receipt_file(
                            repo_root,
                            f"bank/{motion_id}_{scope}.manifest.json",
                            f"{motion_id}:{scope}:manifest",
                        ),
                        "report_sidecar": _write_receipt_file(
                            repo_root,
                            f"bank/{motion_id}_{scope}.report.json",
                            f"{motion_id}:{scope}:report",
                        ),
                    },
                    "strict_schema2_and_ready": {
                        "shared_joint_ready_exact": True,
                        "shared_32_body_ready_exact": True,
                        "six_velocity_classes_exact_zero": endpoint_zero,
                    },
                    "contact_opportunity": {
                        "acceleration_allowed_through_window_end": True,
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
            )
    complete = bool(passed)
    return {
        "schema_version": 1,
        "verdict": "PASS" if complete else "INCOMPLETE_FAIL_CLOSED",
        "bank_gate_pass": complete,
        "candidate_integrity_pass": True,
        "grounded_trace_status": (
            "COMPLETE_PASS"
            if complete
            else "MISSING_INCOMPLETE_FAIL_CLOSED"
        ),
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
            "compiler_options_sha256": _sha("5"),
            "ready": ready,
            "mjcf": mjcf,
            "urdf": urdf,
            "body_order": body_order,
            "plant": {
                "mjcf_sha256": mjcf["sha256"],
                "urdf_sha256": urdf["sha256"],
                "compiled_signature_sha256": _sha("9"),
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
            "grounded_trace_status": (
                "COMPLETE_PASS"
                if complete
                else "MISSING_INCOMPLETE_FAIL_CLOSED"
            ),
        },
        "aggregate": {
            "clip_count": 10,
            "fk_pass_count": 10,
            "velocity_consistency_pass_count": 10,
            "joint_limit_pass_count": 10,
            "geometry_pass_count": 10,
            "non_torque_dynamics_pass_count": 10,
            "complete_dynamics_pass_count": 10,
            "incomplete_fail_closed_count": 0,
            "failed_count": 0,
            "torque_interpretation_valid_count": 10,
            "clips_with_contact_count": 0,
            "contact_frame_count": 0,
            "self_collision_violation_count": 0,
            "foot_floor_penetration_violation_count": 0,
            "nonfoot_floor_penetration_violation_count": 0,
            "other_world_penetration_violation_count": 0,
            "joint_effort_proxy_peak_utilization": 0.1,
            "actuator_force_proxy_peak_utilization": 0.1,
            "root_height_min_m": 1.0,
            "root_height_max_m": 1.0,
            "root_tilt_peak_rad": 0.0,
            "root_xy_displacement_peak_m": 0.0,
            "com_height_min_m": 0.8,
            "com_height_max_m": 0.8,
        },
        "clips": clips,
        "non_claims": [],
    }


def _trusted_fixture(tmp_path: Path, monkeypatch, *, gate_pass=True):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"canonical-ready-fixture\n")
    binding = _binding(hashlib.sha256(ready_path.read_bytes()).hexdigest())
    report_path = tmp_path / "receipts" / "bank_gate.json"
    report_sha = _write_json(
        report_path,
        _bank_gate_report(binding, tmp_path, passed=gate_pass),
    )
    certificate_path = tmp_path / "certificates" / "promotion.json"
    certificate_sha = _write_json(
        certificate_path,
        _certificate_document(
            binding,
            report_path.relative_to(tmp_path).as_posix(),
            report_sha,
        ),
    )
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )
    return binding, certificate_path, report_path


def test_trust_sets_ship_empty_and_public_constructor_rejects_callers():
    assert admission.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256 == frozenset()
    assert admission.TRUSTED_LEGACY_RAW_MOTION_SHA256 == frozenset()
    with pytest.raises(TypeError, match="opaque"):
        admission.TrustedMotionAdmission()


def test_exact_code_trusted_certificate_mints_matching_opaque_capability(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, _ = _trusted_fixture(tmp_path, monkeypatch)
    capability = admission.verify_bank_promotion_certificate(
        certificate_path, binding=binding, repo_root=tmp_path
    )

    assert type(capability) is admission.TrustedMotionAdmission
    assert capability.purpose == "training"
    admission.require_matching_admission(capability, binding)
    with pytest.raises(AttributeError, match="immutable"):
        capability._purpose = "deployment"


def test_untrusted_self_reported_certificate_is_rejected(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, _ = _trusted_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset(),
    )
    with pytest.raises(
        admission.MotionAdmissionError, match="absent from the code trust set"
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=binding, repo_root=tmp_path
        )


def test_trusted_certificate_cannot_crossbind_different_registry(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, _ = _trusted_fixture(tmp_path, monkeypatch)
    wrong = admission.BankPromotionBinding(
        **{
            **binding.__dict__,
            "registry_sha256": _sha("0"),
        }
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="does not crossbind registry_sha256",
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=wrong, repo_root=tmp_path
        )


@pytest.mark.parametrize(
    "field",
    (
        "purpose",
        "bank_id",
        "scope",
        "alignment_sha256",
        "motion_ids",
        "npz_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "build_manifest_sha256",
        "evidence_receipts",
        "question_bank_sha256",
        "training_config_sha256",
        "onnx_model_sha256",
        "onnx_metadata_sha256",
        "adoption_manifest_sha256",
    ),
)
def test_trusted_certificate_crossbinds_every_promotion_column(
    tmp_path: Path, monkeypatch, field: str
):
    binding, certificate_path, _ = _trusted_fixture(tmp_path, monkeypatch)
    document = json.loads(certificate_path.read_text(encoding="utf-8"))
    if field in ("purpose", "bank_id", "scope"):
        document[field] = f"wrong-{field}"
    elif field == "evidence_receipts":
        document[field][0]["certificate_sha256"][0] = _sha("0")
    else:
        value = document[field]
        if isinstance(value, list):
            value[0] = _sha("0")
        else:
            document[field] = _sha("0")
    certificate_sha = _write_json(certificate_path, document)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(
        admission.MotionAdmissionError, match=f"does not crossbind {field}"
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=binding, repo_root=tmp_path
        )


def test_minimal_self_reported_bank_gate_is_not_a_gate_receipt(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, report_path = _trusted_fixture(
        tmp_path, monkeypatch
    )
    report_sha = _write_json(
        report_path,
        {
            "schema_version": 1,
            "verdict": "PASS",
            "bank_gate_pass": True,
            "clips": [
                {
                    "motion_id": motion_id,
                    "scope": binding.scope,
                    "sha256": digest,
                }
                for motion_id, digest in zip(
                    binding.motion_ids, binding.npz_sha256
                )
            ],
        },
    )
    document = _certificate_document(
        binding,
        report_path.relative_to(tmp_path).as_posix(),
        report_sha,
    )
    certificate_sha = _write_json(certificate_path, document)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(
        admission.MotionAdmissionError, match="bank gate report keys changed"
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=binding, repo_root=tmp_path
        )


def test_internal_fields_do_not_replace_certificate_reverification(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, _ = _trusted_fixture(tmp_path, monkeypatch)
    capability = admission.verify_bank_promotion_certificate(
        certificate_path, binding=binding, repo_root=tmp_path
    )
    wrong = admission.BankPromotionBinding(
        **{
            **binding.__dict__,
            "registry_sha256": _sha("0"),
        }
    )
    # Even bypassing the convenience immutability with object.__setattr__
    # cannot turn one trusted certificate into authority for other bytes.
    object.__setattr__(
        capability, "_binding_sha256", admission._binding_sha256(wrong)
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="cannot revalidate its trusted certificate",
    ):
        admission.require_matching_admission(capability, wrong)


def test_incomplete_or_false_bank_gate_never_mints_capability(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, _ = _trusted_fixture(
        tmp_path, monkeypatch, gate_pass=False
    )
    with pytest.raises(
        admission.MotionAdmissionError, match="not a complete, promotable exact PASS"
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=binding, repo_root=tmp_path
        )


def test_bank_gate_report_bytes_are_rechecked_after_certificate_creation(
    tmp_path: Path, monkeypatch
):
    binding, certificate_path, report_path = _trusted_fixture(
        tmp_path, monkeypatch
    )
    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        admission.MotionAdmissionError, match="report SHA-256 differs"
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path, binding=binding, repo_root=tmp_path
        )


def test_legacy_raw_motion_needs_code_allowlist_and_is_rehashed(
    tmp_path: Path, monkeypatch
):
    clip = tmp_path / "legacy.npz"
    clip.write_bytes(b"legacy-motion")
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()

    with pytest.raises(
        admission.MotionAdmissionError, match="absent from the code trust set"
    ):
        admission.legacy_raw_motion_hashes(clip)

    monkeypatch.setattr(
        admission,
        "TRUSTED_LEGACY_RAW_MOTION_SHA256",
        frozenset({digest}),
    )
    assert admission.legacy_raw_motion_hashes(clip) == (digest,)

    clip.write_bytes(b"changed")
    with pytest.raises(
        admission.MotionAdmissionError, match="absent from the code trust set"
    ):
        admission.legacy_raw_motion_hashes(clip)
