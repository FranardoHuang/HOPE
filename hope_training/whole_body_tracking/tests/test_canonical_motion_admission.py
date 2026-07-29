"""Pure-CPU tests for the code-rooted motion admission capability."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

_GENERIC_SCRIPT = _SCRIPT.with_name("canonical_motion_generic_bank_gate.py")
_GENERIC_SPEC = importlib.util.spec_from_file_location(
    "canonical_motion_generic_bank_gate_admission_test",
    _GENERIC_SCRIPT,
)
generic_gate = importlib.util.module_from_spec(_GENERIC_SPEC)
sys.modules[_GENERIC_SPEC.name] = generic_gate
_GENERIC_SPEC.loader.exec_module(generic_gate)

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


def _time_law_marker_contract(
    *,
    window_start: float = 0.2,
    source_anchor: float = 0.1,
    window_end: float = 0.4,
):
    return {
        "marker_names": [
            "window_start",
            "source_anchor",
            "window_end",
        ],
        "path_s": {
            "window_start": window_start,
            "source_anchor": source_anchor,
            "window_end": window_end,
        },
        "time_s": {
            "window_start": window_start,
            "source_anchor": source_anchor,
            "window_end": window_end,
        },
        "source_anchor_within_solved_path": True,
        "source_anchor_independent_of_protected_window": True,
        "protected_window_order_valid": True,
        "no_early_brake_from_path_start_through_window_end": True,
        "inclusive_tick_nonempty": True,
    }


def test_final_admission_accepts_only_real_v2_time_law_identity():
    artifact_script = _SCRIPT.with_name("canonical_time_law_artifact.py")
    artifact_spec = importlib.util.spec_from_file_location(
        "canonical_time_law_artifact_identity_test", artifact_script
    )
    artifact = importlib.util.module_from_spec(artifact_spec)
    sys.modules[artifact_spec.name] = artifact
    artifact_spec.loader.exec_module(artifact)
    summary = {
        "schema_version": artifact.ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact.ARTIFACT_TYPE,
        # source_anchor intentionally precedes the protected window.
        "marker_contract": _time_law_marker_contract(),
    }

    assert (
        admission._validate_canonical_time_law_identity(
            summary, "fixture time law"
        )
        is summary
    )
    assert artifact.ARTIFACT_SCHEMA_VERSION == 2
    assert artifact.ARTIFACT_TYPE == "canonical_time_law_collocation_v2"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.update(
            {"source_anchor_independent_of_protected_window": False}
        ),
        lambda row: row["path_s"].update(
            {"window_start": 0.5, "window_end": 0.4}
        ),
        lambda row: row.update({"inclusive_tick_nonempty": False}),
    ),
)
def test_final_admission_rejects_incomplete_time_law_marker_contract(
    mutation,
):
    marker_contract = _time_law_marker_contract()
    mutation(marker_contract)
    with pytest.raises(
        admission.MotionAdmissionError, match="marker/window contract"
    ):
        admission._validate_canonical_time_law_identity(
            {
                "schema_version": 2,
                "artifact_type": "canonical_time_law_collocation_v2",
                "marker_contract": marker_contract,
            },
            "fixture time law",
        )


@pytest.mark.parametrize(
    "summary",
    (
        {
            "schema_version": 1,
            "artifact_type": "canonical_time_law_collocation_v2",
        },
        {
            "schema_version": True,
            "artifact_type": "canonical_time_law_collocation_v2",
        },
        {
            "schema_version": 2,
            "artifact_type": "canonical_time_law_collocation_v1",
        },
    ),
)
def test_final_admission_rejects_v1_or_mixed_time_law_identity(summary):
    with pytest.raises(
        admission.MotionAdmissionError, match="not exact schema-v2"
    ):
        admission._validate_canonical_time_law_identity(
            summary, "fixture time law"
        )


def test_teacher_only_fitted_receipt_schema_cannot_reenter_formal_admission():
    legacy_teacher_only = {
        key: None
        for key in admission._FRESH_N5_FITTED_BALL_KEYS
        if key
        not in {
            "ball_to_task_solver_executed_by_gate",
            "pre_registered_ball_to_task_solver_receipt_consumed",
            "solver_execution_receipt_authority",
        }
    }
    legacy_teacher_only["ball_to_task_solver_executed"] = False

    with pytest.raises(
        admission.MotionAdmissionError,
        match="keys changed.*pre_registered_ball_to_task",
    ):
        admission._exact_keys(
            legacy_teacher_only,
            admission._FRESH_N5_FITTED_BALL_KEYS,
            "legacy teacher-only fitted receipt",
        )


def test_teacher_only_action_row_without_physical_cases_is_rejected():
    legacy_action = {
        key: None
        for key in admission._FRESH_N5_FITTED_ACTION_KEYS
        if key != "physical_task_binding"
    }
    with pytest.raises(
        admission.MotionAdmissionError,
        match="keys changed.*physical_task_binding",
    ):
        admission._exact_keys(
            legacy_action,
            admission._FRESH_N5_FITTED_ACTION_KEYS,
            "legacy teacher-only action",
        )


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
    gate_script_name = (
        "canonical_motion_generic_bank_gate.py"
        if report_schema_version == 2
        else "canonical_motion_bank_gate.py"
    )
    gate_tool = _write_receipt_file(
        repo_root,
        "hope_training/whole_body_tracking/scripts/" + gate_script_name,
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


def _trusted_producer_generic_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    action_count: int,
    scope: str = "upper",
):
    """Build the admission fixture through the real generic-v2 projector."""

    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"canonical-ready-producer-v2-fixture\n")
    binding = _generic_binding(
        action_count,
        hashlib.sha256(ready_path.read_bytes()).hexdigest(),
        scope=scope,
    )
    raw_report = _bank_gate_report(
        binding,
        tmp_path,
        report_schema_version=1,
    )
    raw_report["grounded_trace_status"] = (
        "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
    )
    raw_report["contracts"]["grounded_trace_status"] = (
        "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
    )
    producer_path = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "canonical_motion_generic_bank_gate.py"
    )
    producer_path.parent.mkdir(parents=True, exist_ok=True)
    producer_path.write_bytes(_GENERIC_SCRIPT.read_bytes())
    monkeypatch.setattr(generic_gate, "__file__", str(producer_path))
    selected = {
        "scope": binding.scope,
        "registry_sha256": binding.registry_sha256,
        "alignment_sha256": binding.alignment_sha256,
        "canonical_ready_sha256": binding.canonical_ready_sha256,
        "canonical_ready_fk_sha256": binding.canonical_ready_fk_sha256,
        "motion_ids": list(binding.motion_ids),
        "npz_sha256": list(binding.npz_sha256),
        "build_manifest_sha256": list(binding.build_manifest_sha256),
    }
    produced_report = generic_gate._generic_v2_report(
        raw_report,
        selected_registry_binding=selected,
    )
    report_path = tmp_path / "receipts" / "producer_bank_gate_v2.json"
    report_sha = _write_json(report_path, dict(produced_report))
    certificate_path = (
        tmp_path / "certificates" / "producer_promotion_v2.json"
    )
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


def test_real_generic_v2_report_flows_directly_into_admission(
    tmp_path: Path,
    monkeypatch,
):
    binding, certificate_path, report_path = (
        _trusted_producer_generic_fixture(
            tmp_path,
            monkeypatch,
            action_count=3,
            scope="full",
        )
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == generic_gate.REPORT_SCHEMA_VERSION == 2
    assert report["bound_inputs"]["verifier_tools"]["bank_gate"]["path"].endswith(
        "/canonical_motion_generic_bank_gate.py"
    )
    capability = admission.verify_bank_promotion_certificate(
        certificate_path,
        binding=binding,
        repo_root=tmp_path,
    )

    admission.require_matching_admission(capability, binding)


def test_real_generic_v2_report_cannot_claim_the_legacy_gate_producer(
    tmp_path: Path,
    monkeypatch,
):
    binding, certificate_path, report_path = (
        _trusted_producer_generic_fixture(
            tmp_path,
            monkeypatch,
            action_count=3,
        )
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    legacy_path = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "canonical_motion_bank_gate.py"
    )
    report["bound_inputs"]["verifier_tools"]["bank_gate"] = {
        "path": legacy_path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
    }
    report_sha = _write_json(report_path, report)
    certificate = _generic_certificate_document(
        binding,
        report_path.relative_to(tmp_path).as_posix(),
        report_sha,
    )
    certificate_sha = _write_json(certificate_path, certificate)
    monkeypatch.setattr(
        admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset({certificate_sha}),
    )

    with pytest.raises(
        admission.MotionAdmissionError,
        match="not the repository verifier.*generic_bank_gate",
    ):
        admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=tmp_path,
        )


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


def test_isaac_table_smoke_uses_real_registered_action_ball_task_id(
    tmp_path: Path,
):
    runtime_source = (
        tmp_path
        / "hope_training/whole_body_tracking/scripts/"
        "check_table_obstacle_scene.py"
    )
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("RUNTIME = 'fixture'\n", encoding="utf-8")
    runtime_source_sha = hashlib.sha256(
        runtime_source.read_bytes()
    ).hexdigest()
    solver_source_base = (
        tmp_path
        / "hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    solver_source_base.mkdir(parents=True)
    solver_source_rows = []
    solver_source_map = {}
    for name in sorted(admission._ACTION_BALL_SOLVER_SOURCE_NAMES):
        path = solver_source_base / name
        path.write_text(f"# solver fixture {name}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        solver_source_map[name] = digest
        solver_source_rows.append(
            {
                "name": name,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": digest,
            }
        )
    contact_geometry_payload = {
        "schema_version": 2,
        "semantics": "fixture canonical exact-face geometry",
    }
    contact_geometry = {
        "payload": contact_geometry_payload,
        "sha256": hashlib.sha256(
            admission._canonical_json_bytes(contact_geometry_payload)
        ).hexdigest(),
    }
    solver_payload = {
        "kind": "fixture.solver",
        "implementation_source_sha256": solver_source_map,
        "contact_geometry": contact_geometry,
    }
    physics_payload = {
        "kind": "fixture.physics",
        "geometry_and_grading": {},
    }
    solver_profile_sha = hashlib.sha256(
        admission._canonical_json_bytes(solver_payload)
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        admission._canonical_json_bytes(physics_payload)
    ).hexdigest()
    profile_pins_path = tmp_path / "profile_pins.json"
    profile_pins_sha = _write_json(
        profile_pins_path,
        {
            "solver_payload": solver_payload,
            "physics_payload": physics_payload,
            "solver_profile_sha256": solver_profile_sha,
                "physics_profile_sha256": physics_profile_sha,
                "solver_implementation_source_sha256": solver_source_map,
                "contact_geometry": contact_geometry,
        },
    )
    geometry_source = solver_source_base / "racket_contact_geometry.py"
    geometry_contract = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": geometry_source.relative_to(tmp_path).as_posix(),
        "source_sha256": hashlib.sha256(
            geometry_source.read_bytes()
        ).hexdigest(),
        "geometry_source_sha256": contact_geometry["sha256"],
    }
    motion_ids = admission.FRESH_N5_DOWNSTREAM_MOTION_IDS
    motion_sha = tuple(
        _digest(f"isaac-motion:{motion_id}") for motion_id in motion_ids
    )
    motion_paths = []
    for motion_id, digest in zip(motion_ids, motion_sha):
        motion_path = tmp_path / "motions" / f"{motion_id}_upper.npz"
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        motion_path.write_bytes(f"motion:{motion_id}\n".encode("utf-8"))
        # The fixture binding must name the bytes actually reopened by admission.
        actual = hashlib.sha256(motion_path.read_bytes()).hexdigest()
        motion_paths.append((motion_path, actual))
    motion_sha = tuple(digest for _path, digest in motion_paths)
    action_uids = tuple(
        admission._derive_action_uid(
            motion_id,
            admission.FRESH_N5_ACTION_FAMILY[motion_id],
            digest,
        )
        for motion_id, digest in zip(motion_ids, motion_sha)
    )
    manifest_path = tmp_path / "fresh_n5_manifest.json"
    manifest_sha = _write_json(
        manifest_path,
        {
                "schema_version": 3,
                "manifest_id": "fresh_n5_fixture",
                "mobility_mode": "no_move",
                "solver_profile_sha256": solver_profile_sha,
                "physics_profile_sha256": physics_profile_sha,
                "action_order": list(motion_ids),
            "prototype": {"scope": "upper"},
            "actions": [
                {
                    "action_id": motion_id,
                        "action_uid": action_uid,
                    "family": admission.FRESH_N5_ACTION_FAMILY[motion_id],
                    "motion_path": path.relative_to(tmp_path).as_posix(),
                    "motion_sha256": digest,
                    "reference_t_hit_s": 0.2,
                    "reference_t_cycle_s": 0.8,
                    "reference_racket_site_speed_mps": 2.0,
                    "strike_phase": 0.25,
                    "reaction_margin_s": 0.1,
                    "teacher_rate_min": 0.5,
                    "teacher_rate_max": 1.0,
                    "ball_profile": {
                        "time_to_contact_center_s": 0.6,
                    },
                }
                    for motion_id, action_uid, (path, digest) in zip(
                        motion_ids, action_uids, motion_paths
                    )
                ],
            },
        )
    order_uid_digest = hashlib.sha256(
        admission._canonical_json_bytes(
            {
                "schema_version": 1,
                "ordered_actions": [
                    {
                        "index": index,
                        "action_id": action_id,
                        "action_uid": action_uid,
                    }
                    for index, (action_id, action_uid) in enumerate(
                        zip(motion_ids, action_uids)
                    )
                ],
            }
        )
    ).hexdigest()
    action_set_contract = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.action_set_contract",
        "profile_id": "fresh_upper_nomove_n5_v3",
        "expected_n": len(motion_ids),
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": list(motion_ids),
        "ordered_action_uids": list(action_uids),
        "order_uid_digest_sha256": order_uid_digest,
        "manifest_path": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "experiment_name": "fresh_n5_fixture",
        "actor_obs_contract": "action_ball_n5",
        "actor_obs_width": 186,
        "namespace_identity": f"n5-{order_uid_digest[:12]}",
    }
    action_set_contract["contract_sha256"] = hashlib.sha256(
        admission._canonical_json_bytes(action_set_contract)
    ).hexdigest()

    def make_receipt(task_id: str):
        document = {
            "schema_version": 2,
            "receipt_class": "isaac_action_ball_table_filtered_smoke_v2",
            "verdict": "PASS",
            "task_id": task_id,
            "with_table": True,
            "scope": "upper",
                "mobility_mode": "no_move",
                "action_set_contract": action_set_contract,
                "manifest": {
                    "path": manifest_path.name,
                    "sha256": manifest_sha,
                },
                "profile_contract": {
                    "profile_pins": {
                        "path": profile_pins_path.name,
                        "sha256": profile_pins_sha,
                    },
                    "solver_profile_sha256": solver_profile_sha,
                    "physics_profile_sha256": physics_profile_sha,
                    "solver_implementation_sources": solver_source_rows,
                    "racket_geometry_contract": geometry_contract,
                },
            "ordered_action_ids": list(motion_ids),
            "motion_sha256": list(motion_sha),
            "runtime_contract": {
                "source_commit_sha": "1" * 40,
                "isaac_version": "fixture-isaac",
                "python_executable": "/fixture/python",
                "runtime_source": {
                    "path": runtime_source.relative_to(tmp_path).as_posix(),
                    "sha256": runtime_source_sha,
                },
                "gpu_identity": {
                    "physical_index": 2,
                    "logical_index": 0,
                    "cuda_visible_devices": "2",
                    "gpu_uuid": "GPU-fixture",
                    "gpu_name": "Fixture GPU",
                    "driver_version": "fixture-driver",
                    "nvml_verified": True,
                },
                "physics_steps": 8,
                "real_physx_contacts": True,
                "full_action_ball_assembly": True,
                "all_32_body_pair_filters": True,
                "action_body_pair_filter_rows": 32 * len(motion_ids),
                "all_five_obstacles": True,
                "all_four_substeps": True,
                "positive_control_pass": True,
                "negative_control_pass": True,
                "zero_reset_leakage": True,
            },
            "actions": [
                {
                    "motion_id": motion_id,
                    "action_uid": action_uid,
                    "scope": "upper",
                    "body_pair_filter_count": 32,
                    "motion_sha256": digest,
                    "complete_cycle": True,
                    "isaac_filtered_contact_pass": True,
                    "table_contact_count": 0,
                    "fall_count": 0,
                    "hard_limit_count": 0,
                    "unsafe_count": 0,
                    "verdict": "PASS",
                }
                for motion_id, action_uid, digest in zip(
                    motion_ids, action_uids, motion_sha
                )
            ],
            "authorization": {
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            "non_claims": [
                "training_authorization",
                "deployment_authorization",
                "hardware_authorization",
            ],
        }
        document["receipt_payload_sha256"] = hashlib.sha256(
            admission._canonical_json_bytes(document)
        ).hexdigest()
        return document

    good_path = tmp_path / "isaac_good.json"
    good_sha = _write_json(
        good_path,
        make_receipt(admission.ACTION_BALL_ISAAC_TASK_ID),
    )
    binding = SimpleNamespace(
        isaac_table_filtered_smoke_receipt_sha256=good_sha,
        motion_ids=motion_ids,
        npz_sha256=motion_sha,
    )
    admission._validate_fresh_n5_isaac_table_smoke_receipt(
        {"path": good_path.name, "sha256": good_sha},
        binding=binding,
        repo_root=tmp_path,
    )

    def reseal(document: dict) -> dict:
        document["receipt_payload_sha256"] = hashlib.sha256(
            admission._canonical_json_bytes(
                {
                    key: value
                    for key, value in document.items()
                    if key != "receipt_payload_sha256"
                }
            )
        ).hexdigest()
        return document

    legacy_receipt = make_receipt(admission.ACTION_BALL_ISAAC_TASK_ID)
    legacy_receipt["schema_version"] = 1
    legacy_receipt["receipt_class"] = (
        "isaac_action_ball_table_filtered_smoke_v1"
    )
    legacy_path = tmp_path / "isaac_legacy_v1.json"
    legacy_sha = _write_json(legacy_path, reseal(legacy_receipt))
    with pytest.raises(
        admission.MotionAdmissionError,
        match="exact stepped fresh-N5",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {"path": legacy_path.name, "sha256": legacy_sha},
            binding=SimpleNamespace(
                isaac_table_filtered_smoke_receipt_sha256=legacy_sha,
                motion_ids=motion_ids,
                npz_sha256=motion_sha,
            ),
            repo_root=tmp_path,
        )

    short_filter_receipt = make_receipt(
        admission.ACTION_BALL_ISAAC_TASK_ID
    )
    short_filter_receipt["runtime_contract"][
        "action_body_pair_filter_rows"
    ] -= 1
    short_filter_path = tmp_path / "isaac_short_filter_rows.json"
    short_filter_sha = _write_json(
        short_filter_path, reseal(short_filter_receipt)
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="exact stepped fresh-N5",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {
                "path": short_filter_path.name,
                "sha256": short_filter_sha,
            },
            binding=SimpleNamespace(
                isaac_table_filtered_smoke_receipt_sha256=(
                    short_filter_sha
                ),
                motion_ids=motion_ids,
                npz_sha256=motion_sha,
            ),
            repo_root=tmp_path,
        )

    weak_action_filter_receipt = make_receipt(
        admission.ACTION_BALL_ISAAC_TASK_ID
    )
    weak_action_filter_receipt["actions"][0][
        "body_pair_filter_count"
    ] = 31
    weak_action_filter_path = tmp_path / "isaac_weak_action_filter.json"
    weak_action_filter_sha = _write_json(
        weak_action_filter_path, reseal(weak_action_filter_receipt)
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="partial, unsafe",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {
                "path": weak_action_filter_path.name,
                "sha256": weak_action_filter_sha,
            },
            binding=SimpleNamespace(
                isaac_table_filtered_smoke_receipt_sha256=(
                    weak_action_filter_sha
                ),
                motion_ids=motion_ids,
                npz_sha256=motion_sha,
            ),
            repo_root=tmp_path,
        )

    old_fake_id = "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"
    bad_path = tmp_path / "isaac_old_fake_task.json"
    bad_sha = _write_json(bad_path, make_receipt(old_fake_id))
    bad_binding = SimpleNamespace(
        isaac_table_filtered_smoke_receipt_sha256=bad_sha,
        motion_ids=motion_ids,
        npz_sha256=motion_sha,
    )
    with pytest.raises(
        admission.MotionAdmissionError,
        match="exact stepped fresh-N5",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {"path": bad_path.name, "sha256": bad_sha},
            binding=bad_binding,
            repo_root=tmp_path,
        )

    stale_uid_receipt = make_receipt(
        admission.ACTION_BALL_ISAAC_TASK_ID
    )
    stale_uid_receipt["actions"][0]["action_uid"] += 1
    stale_uid_receipt["receipt_payload_sha256"] = hashlib.sha256(
        admission._canonical_json_bytes(
            {
                key: value
                for key, value in stale_uid_receipt.items()
                if key != "receipt_payload_sha256"
            }
        )
    ).hexdigest()
    stale_uid_path = tmp_path / "isaac_stale_uid.json"
    stale_uid_sha = _write_json(stale_uid_path, stale_uid_receipt)
    with pytest.raises(
        admission.MotionAdmissionError,
        match="partial, unsafe",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {"path": stale_uid_path.name, "sha256": stale_uid_sha},
            binding=SimpleNamespace(
                isaac_table_filtered_smoke_receipt_sha256=stale_uid_sha,
                motion_ids=motion_ids,
                npz_sha256=motion_sha,
            ),
            repo_root=tmp_path,
        )

    with pytest.raises(
        admission.MotionAdmissionError,
        match="do not bind the same manifest",
    ):
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            {"path": good_path.name, "sha256": good_sha},
            binding=binding,
            repo_root=tmp_path,
            expected_identity=admission._FreshN5EvidenceIdentity(
                manifest_sha256=manifest_sha,
                manifest_id="fresh_n5_fixture",
                action_set_contract_sha256=action_set_contract[
                    "contract_sha256"
                ],
                action_ids=tuple(motion_ids),
                action_uids=action_uids,
                profile_pins_sha256=profile_pins_sha,
                solver_profile_sha256=solver_profile_sha,
                physics_profile_sha256=physics_profile_sha,
                geometry_source_sha256=geometry_contract[
                    "geometry_source_sha256"
                ],
                code_commit="2" * 40,
            ),
        )
