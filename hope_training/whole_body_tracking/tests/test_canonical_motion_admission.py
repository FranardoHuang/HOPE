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


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _generic_binding(
    action_count: int,
    ready_sha256: str,
    *,
    scope: str = "upper",
) -> admission.GenericBankPromotionBinding:
    motion_ids = tuple(
        f"action_{index:03d}" for index in range(action_count)
    )
    return admission.GenericBankPromotionBinding(
        purpose="training",
        bank_id=f"generic_{scope}_n{action_count}_v2",
        scope=scope,
        registry_sha256=_digest(f"registry:{scope}:{action_count}"),
        alignment_sha256=_digest(f"alignment:{scope}:{action_count}"),
        motion_ids=motion_ids,
        npz_sha256=tuple(
            _digest(f"npz:{scope}:{motion_id}") for motion_id in motion_ids
        ),
        canonical_ready_sha256=ready_sha256,
        canonical_ready_fk_sha256=_digest("ready-fk"),
        build_manifest_sha256=tuple(
            _digest(f"build:{scope}:{motion_id}")
            for motion_id in motion_ids
        ),
        evidence_levels=("E2",) * action_count,
        evidence_manifest_sha256=tuple(
            _digest(f"evidence:{scope}:{motion_id}")
            for motion_id in motion_ids
        ),
        evidence_certificate_sha256=tuple(
            (
                _digest(f"e1:{scope}:{motion_id}"),
                _digest(f"e2:{scope}:{motion_id}"),
            )
            for motion_id in motion_ids
        ),
        question_bank_sha256=tuple(
            _digest(f"questions:{scope}:{motion_id}")
            for motion_id in motion_ids
        ),
        training_config_sha256=tuple(
            _digest(f"training:{scope}:{motion_id}")
            for motion_id in motion_ids
        ),
        onnx_model_sha256=(None,) * action_count,
        onnx_metadata_sha256=(None,) * action_count,
        adoption_manifest_sha256=tuple(
            _digest(f"adoption:{scope}:{motion_id}")
            for motion_id in motion_ids
        ),
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


def _generic_certificate_document(
    binding: admission.GenericBankPromotionBinding,
    report_path: str,
    report_sha: str,
):
    return {
        "schema_version": 2,
        "certificate_type": "canonical-motion-bank-promotion-v2",
        **admission._binding_document(binding),
        "bank_gate_report": {"path": report_path, "sha256": report_sha},
    }


def _bank_gate_report(
    binding: admission.BankPromotionBinding,
    repo_root: Path,
    *,
    passed=True,
    report_schema_version: int = 1,
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
    expected_clip_count = 2 * len(binding.motion_ids)
    report = {
        "schema_version": report_schema_version,
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
                "count": expected_clip_count,
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
            "clip_count": expected_clip_count,
            "fk_pass_count": expected_clip_count,
            "velocity_consistency_pass_count": expected_clip_count,
            "joint_limit_pass_count": expected_clip_count,
            "geometry_pass_count": expected_clip_count,
            "non_torque_dynamics_pass_count": expected_clip_count,
            "complete_dynamics_pass_count": expected_clip_count,
            "incomplete_fail_closed_count": 0,
            "failed_count": 0,
            "torque_interpretation_valid_count": expected_clip_count,
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
    if report_schema_version == 2:
        report["selected_registry_binding"] = {
            "scope": binding.scope,
            "registry_sha256": binding.registry_sha256,
            "alignment_sha256": binding.alignment_sha256,
            "canonical_ready_sha256": binding.canonical_ready_sha256,
            "canonical_ready_fk_sha256": (
                binding.canonical_ready_fk_sha256
            ),
            "motion_ids": list(binding.motion_ids),
            "npz_sha256": list(binding.npz_sha256),
            "build_manifest_sha256": list(
                binding.build_manifest_sha256
            ),
        }
    return report


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


def _trusted_generic_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    action_count: int,
    scope: str = "upper",
):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"canonical-ready-v2-fixture\n")
    binding = _generic_binding(
        action_count,
        hashlib.sha256(ready_path.read_bytes()).hexdigest(),
        scope=scope,
    )
    report_path = tmp_path / "receipts" / "bank_gate_v2.json"
    report_sha = _write_json(
        report_path,
        _bank_gate_report(
            binding,
            tmp_path,
            report_schema_version=2,
        ),
    )
    certificate_path = tmp_path / "certificates" / "promotion_v2.json"
    certificate_sha = _write_json(
        certificate_path,
        _generic_certificate_document(
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


def test_canonical_trust_set_ships_empty_and_admission_is_opaque():
    assert admission.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256 == frozenset()
    # The legacy/default motion_file path is no longer gated by this module —
    # it loads raw NPZ bytes directly — so there is no legacy raw-motion trust
    # set. Assert the gate is gone rather than re-asserting an allowlist.
    assert not hasattr(admission, "TRUSTED_LEGACY_RAW_MOTION_SHA256")
    assert not hasattr(admission, "legacy_raw_motion_hashes")
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


@pytest.mark.parametrize(
    ("action_count", "scope"),
    ((1, "upper"), (5, "full"), (6, "upper"), (93, "upper")),
)
def test_v2_generic_bank_mints_one_selected_scope_capability_for_arbitrary_n(
    tmp_path: Path,
    monkeypatch,
    action_count: int,
    scope: str,
):
    binding, certificate_path, _ = _trusted_generic_fixture(
        tmp_path,
        monkeypatch,
        action_count=action_count,
        scope=scope,
    )

    capability = admission.verify_bank_promotion_certificate(
        certificate_path,
        binding=binding,
        repo_root=tmp_path,
    )

    assert type(capability) is admission.TrustedMotionAdmission
    assert capability.purpose == "training"
    assert capability.certificate_sha256 == hashlib.sha256(
        certificate_path.read_bytes()
    ).hexdigest()
    admission.require_matching_admission(capability, binding)


@pytest.mark.parametrize("action_count", (1, 6, 93))
def test_v1_binding_remains_exactly_five_actions(
    tmp_path: Path, action_count: int
):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"ready\n")
    generic = _generic_binding(
        action_count,
        hashlib.sha256(ready_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="five unique ordered motion ids",
    ):
        admission.BankPromotionBinding(**generic.__dict__)


def test_v2_binding_rejects_empty_duplicate_and_invalid_scope(tmp_path: Path):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"ready\n")
    ready_sha = hashlib.sha256(ready_path.read_bytes()).hexdigest()
    with pytest.raises(admission.MotionAdmissionError, match="non-empty"):
        _generic_binding(0, ready_sha)

    valid = _generic_binding(2, ready_sha)
    with pytest.raises(admission.MotionAdmissionError, match="unique"):
        admission.GenericBankPromotionBinding(
            **{
                **valid.__dict__,
                "motion_ids": ("duplicate", "duplicate"),
            }
        )
    with pytest.raises(admission.MotionAdmissionError, match="scope"):
        admission.GenericBankPromotionBinding(
            **{**valid.__dict__, "scope": "mixed"}
        )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="exactly 2 receipts for E2",
    ):
        admission.GenericBankPromotionBinding(
            **{
                **valid.__dict__,
                "evidence_certificate_sha256": (
                    valid.evidence_certificate_sha256[0][:-1],
                    valid.evidence_certificate_sha256[1],
                ),
            }
        )


@pytest.mark.parametrize("mutation", ("matrix_count", "aggregate", "clip_order"))
def test_v2_generic_gate_receipt_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
):
    binding, certificate_path, report_path = _trusted_generic_fixture(
        tmp_path,
        monkeypatch,
        action_count=6,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "matrix_count":
        report["contracts"]["matrix"]["count"] += 1
    elif mutation == "aggregate":
        report["aggregate"]["complete_dynamics_pass_count"] -= 1
    else:
        report["clips"][0], report["clips"][1] = (
            report["clips"][1],
            report["clips"][0],
        )
    report_sha = _write_json(report_path, report)
    document = _generic_certificate_document(
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

    with pytest.raises(admission.MotionAdmissionError):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    ("scope", "motion_ids", "npz_sha256", "evidence_receipts"),
)
def test_v2_certificate_crossbinds_selected_scope_order_and_lineage(
    tmp_path: Path,
    monkeypatch,
    field: str,
):
    binding, certificate_path, _ = _trusted_generic_fixture(
        tmp_path,
        monkeypatch,
        action_count=6,
    )
    document = json.loads(certificate_path.read_text(encoding="utf-8"))
    if field == "scope":
        document[field] = "full"
    elif field == "evidence_receipts":
        document[field][0]["certificate_sha256"][0] = _digest("wrong-e1")
    else:
        document[field][0], document[field][1] = (
            document[field][1],
            document[field][0],
        )
    certificate_sha = _write_json(certificate_path, document)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(
        admission.MotionAdmissionError,
        match=f"does not crossbind {field}",
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


def test_v2_rejects_a_different_gate_report_with_matching_surface_identity(
    tmp_path: Path,
    monkeypatch,
):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"ready\n")
    binding = _generic_binding(
        2,
        hashlib.sha256(ready_path.read_bytes()).hexdigest(),
    )
    other_binding = admission.GenericBankPromotionBinding(
        **{
            **binding.__dict__,
            "registry_sha256": _digest("other-registry"),
            "alignment_sha256": _digest("other-alignment"),
            "canonical_ready_fk_sha256": _digest("other-ready-fk"),
            "build_manifest_sha256": tuple(
                _digest(f"other-build:{motion_id}")
                for motion_id in binding.motion_ids
            ),
        }
    )
    report_path = tmp_path / "receipts" / "other_gate_v2.json"
    report_sha = _write_json(
        report_path,
        _bank_gate_report(
            other_binding,
            tmp_path,
            report_schema_version=2,
        ),
    )
    certificate_path = tmp_path / "certificates" / "mixed_lineage_v2.json"
    certificate_sha = _write_json(
        certificate_path,
        _generic_certificate_document(
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

    with pytest.raises(
        admission.MotionAdmissionError,
        match="selected registry lineage differs",
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("certificate_version", (1, 2))
@pytest.mark.parametrize(
    ("target", "replacement"),
    (
        ("certificate_schema", None),
        ("report_schema", None),
        ("dynamics_schema", 1.0),
        ("matrix_count", None),
        ("complete_count", None),
        ("failed_count", False),
    ),
)
def test_promotion_numeric_contract_rejects_float_and_boolean_aliases(
    tmp_path: Path,
    monkeypatch,
    certificate_version: int,
    target: str,
    replacement,
):
    if certificate_version == 1:
        binding, certificate_path, report_path = _trusted_fixture(
            tmp_path,
            monkeypatch,
        )
    else:
        binding, certificate_path, report_path = _trusted_generic_fixture(
            tmp_path,
            monkeypatch,
            action_count=2,
        )
    certificate = json.loads(
        certificate_path.read_text(encoding="utf-8")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if target == "certificate_schema":
        certificate["schema_version"] = float(certificate_version)
    else:
        if target == "report_schema":
            report["schema_version"] = float(certificate_version)
        elif target == "dynamics_schema":
            report["bound_inputs"]["verifier_tools"][
                "canonical_mujoco_dynamics_gate"
            ]["report_schema_version"] = replacement
        elif target == "matrix_count":
            report["contracts"]["matrix"]["count"] = float(
                report["contracts"]["matrix"]["count"]
            )
        elif target == "complete_count":
            report["aggregate"]["complete_dynamics_pass_count"] = float(
                report["aggregate"]["complete_dynamics_pass_count"]
            )
        else:
            report["aggregate"]["failed_count"] = replacement
        certificate["bank_gate_report"]["sha256"] = _write_json(
            report_path,
            report,
        )
    certificate_sha = _write_json(certificate_path, certificate)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(admission.MotionAdmissionError):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


def test_v2_binding_cannot_ride_on_a_trusted_v1_certificate(
    tmp_path: Path, monkeypatch
):
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"ready\n")
    binding = _generic_binding(
        5, hashlib.sha256(ready_path.read_bytes()).hexdigest()
    )
    report_path = tmp_path / "receipts" / "gate.json"
    report_sha = _write_json(
        report_path,
        _bank_gate_report(
            binding,
            tmp_path,
            report_schema_version=2,
        ),
    )
    certificate_path = tmp_path / "certificates" / "wrong_version.json"
    certificate = _generic_certificate_document(
        binding,
        report_path.relative_to(tmp_path).as_posix(),
        report_sha,
    )
    certificate["schema_version"] = 1
    certificate["certificate_type"] = "canonical-motion-bank-promotion-v1"
    certificate_sha = _write_json(certificate_path, certificate)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(
        admission.MotionAdmissionError,
        match="schema/type is unsupported",
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


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
