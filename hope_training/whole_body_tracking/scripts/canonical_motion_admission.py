#!/usr/bin/env python3
"""Code-rooted admission capabilities for motion-training consumers.

Registry JSON, evidence labels, and adoption manifests are provenance records,
not authority.  This module is the only place that turns a bank-promotion
certificate into an opaque runtime capability.  A certificate is trusted only
when the SHA-256 of the exact bytes parsed by this verifier is present in the
code-owned trust set below.  Neither trust set is configurable through Hydra.

Both sets intentionally ship empty.  Adding a digest is a reviewed source-code
change.  Until that happens, new canonical-bank and legacy raw-motion launches
fail closed.

This is an operational configuration/provenance boundary, not a sandbox
against arbitrary Python already executing in the process.  In-process code is
part of the trusted computing base and can rebind module globals; an adversarial
plugin threat model would require an external signed ledger or launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256: frozenset[str] = frozenset()
TRUSTED_LEGACY_RAW_MOTION_SHA256: frozenset[str] = frozenset()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MINT_TOKEN = object()
_PURPOSES = ("training", "deployment", "hardware")
_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "certificate_type",
        "purpose",
        "bank_id",
        "scope",
        "registry_sha256",
        "alignment_sha256",
        "motion_ids",
        "npz_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "build_manifest_sha256",
        "bank_gate_report",
        "evidence_receipts",
        "question_bank_sha256",
        "training_config_sha256",
        "onnx_model_sha256",
        "onnx_metadata_sha256",
        "adoption_manifest_sha256",
    }
)
_BANK_GATE_BINDING_KEYS = frozenset({"path", "sha256"})
_BANK_GATE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "verdict",
        "bank_gate_pass",
        "candidate_integrity_pass",
        "grounded_trace_status",
        "publication_class",
        "training_authorized",
        "hardware_authorized",
        "library_id",
        "manifest",
        "bank_dir",
        "bound_inputs",
        "contracts",
        "aggregate",
        "clips",
        "non_claims",
    }
)
_BANK_GATE_BOUND_INPUT_KEYS = frozenset(
    {
        "recipe",
        "compiler",
        "geometry_tool",
        "compiler_options_sha256",
        "ready",
        "mjcf",
        "urdf",
        "body_order",
        "plant",
        "verifier_tools",
    }
)
_BANK_GATE_CONTRACT_KEYS = frozenset(
    {
        "matrix",
        "shared_ready",
        "six_endpoint_velocity_classes_exact_zero",
        "contact_opportunity_is_marker_only",
        "acceleration_allowed_through_window_end",
        "nonnegative_scalar_acceleration_through_window_end",
        "adv2c3_role",
        "grounded_inverse_dynamics",
        "grounded_trace_status",
    }
)
_BANK_GATE_AGGREGATE_KEYS = frozenset(
    {
        "clip_count",
        "fk_pass_count",
        "velocity_consistency_pass_count",
        "joint_limit_pass_count",
        "geometry_pass_count",
        "non_torque_dynamics_pass_count",
        "complete_dynamics_pass_count",
        "incomplete_fail_closed_count",
        "failed_count",
        "torque_interpretation_valid_count",
        "clips_with_contact_count",
        "contact_frame_count",
        "self_collision_violation_count",
        "foot_floor_penetration_violation_count",
        "nonfoot_floor_penetration_violation_count",
        "other_world_penetration_violation_count",
        "joint_effort_proxy_peak_utilization",
        "actuator_force_proxy_peak_utilization",
        "root_height_min_m",
        "root_height_max_m",
        "root_tilt_peak_rad",
        "root_xy_displacement_peak_m",
        "com_height_min_m",
        "com_height_max_m",
    }
)
_BANK_GATE_CLIP_KEYS = frozenset(
    {
        "motion_id",
        "scope",
        "filename",
        "sha256",
        "frames",
        "fps",
        "duration_s",
        "schema2_receipts",
        "strict_schema2_and_ready",
        "contact_opportunity",
        "mujoco_fk",
        "plant_specific_dynamics",
    }
)
_EVIDENCE_RECEIPT_KEYS = frozenset(
    {
        "motion_id",
        "evidence_level",
        "evidence_manifest_sha256",
        "certificate_sha256",
    }
)


class MotionAdmissionError(ValueError):
    """No trusted capability can be minted for the requested motion bytes."""


@dataclass(frozen=True)
class BankPromotionBinding:
    """Exact registry-derived values a trusted certificate must authorize."""

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
    question_bank_sha256: tuple[str | None, ...]
    training_config_sha256: tuple[str | None, ...]
    onnx_model_sha256: tuple[str | None, ...]
    onnx_metadata_sha256: tuple[str | None, ...]
    adoption_manifest_sha256: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.purpose not in _PURPOSES:
            raise MotionAdmissionError(
                f"purpose must be one of {_PURPOSES}, got {self.purpose!r}"
            )
        count = len(self.motion_ids)
        if count != 5 or len(set(self.motion_ids)) != count:
            raise MotionAdmissionError(
                "bank promotion binding requires five unique ordered motion ids"
            )
        columns = (
            self.npz_sha256,
            self.build_manifest_sha256,
            self.evidence_levels,
            self.evidence_manifest_sha256,
            self.evidence_certificate_sha256,
            self.question_bank_sha256,
            self.training_config_sha256,
            self.onnx_model_sha256,
            self.onnx_metadata_sha256,
            self.adoption_manifest_sha256,
        )
        if any(len(column) != count for column in columns):
            raise MotionAdmissionError(
                "bank promotion binding columns must all have length five"
            )
        for label, digest in (
            ("registry_sha256", self.registry_sha256),
            ("alignment_sha256", self.alignment_sha256),
            ("canonical_ready_sha256", self.canonical_ready_sha256),
            ("canonical_ready_fk_sha256", self.canonical_ready_fk_sha256),
        ):
            _digest(digest, label)
        for label, values in (
            ("npz_sha256", self.npz_sha256),
            ("build_manifest_sha256", self.build_manifest_sha256),
            ("evidence_manifest_sha256", self.evidence_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                _digest(digest, f"{label}[{index}]")
        for row, values in enumerate(self.evidence_certificate_sha256):
            for column, digest in enumerate(values):
                _digest(
                    digest,
                    f"evidence_certificate_sha256[{row}][{column}]",
                )
        for label, values in (
            ("question_bank_sha256", self.question_bank_sha256),
            ("training_config_sha256", self.training_config_sha256),
            ("onnx_model_sha256", self.onnx_model_sha256),
            ("onnx_metadata_sha256", self.onnx_metadata_sha256),
            ("adoption_manifest_sha256", self.adoption_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                if digest is not None:
                    _digest(digest, f"{label}[{index}]")


class TrustedMotionAdmission:
    """Config-opaque token minted after code-rooted certificate verification."""

    __slots__ = (
        "_certificate_sha256",
        "_binding_sha256",
        "_purpose",
        "_bank_id",
        "_scope",
        "_certificate_path",
        "_repo_root",
        "_sealed",
    )

    def __init__(
        self,
        *,
        _token: object | None = None,
        certificate_sha256: str = "",
        binding_sha256: str = "",
        purpose: str = "",
        bank_id: str = "",
        scope: str = "",
        certificate_path: str = "",
        repo_root: str = "",
    ) -> None:
        if _token is not _MINT_TOKEN:
            raise TypeError(
                "TrustedMotionAdmission is opaque; use "
                "verify_bank_promotion_certificate"
            )
        object.__setattr__(self, "_certificate_sha256", certificate_sha256)
        object.__setattr__(self, "_binding_sha256", binding_sha256)
        object.__setattr__(self, "_purpose", purpose)
        object.__setattr__(self, "_bank_id", bank_id)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_certificate_path", certificate_path)
        object.__setattr__(self, "_repo_root", repo_root)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TrustedMotionAdmission is immutable")
        object.__setattr__(self, name, value)

    @property
    def certificate_sha256(self) -> str:
        return self._certificate_sha256

    @property
    def purpose(self) -> str:
        return self._purpose


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotionAdmissionError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise MotionAdmissionError(f"JSON contains forbidden constant {value}")


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionAdmissionError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MotionAdmissionError(f"{label} must contain one JSON object")
    return value


def _exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionAdmissionError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise MotionAdmissionError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MotionAdmissionError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _snapshot(path: Path, label: str) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MotionAdmissionError(f"cannot read {label} {path}: {exc}") from exc
    return payload, hashlib.sha256(payload).hexdigest()


def _repo_file(value: Any, repo_root: Path, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise MotionAdmissionError(
            f"{label} must be one normalized repository-relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise MotionAdmissionError(f"{label} may not contain '.' or '..'")
    try:
        root = repo_root.resolve(strict=True)
        path = root.joinpath(*pure.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MotionAdmissionError(
            f"{label} does not resolve inside repository root"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError(f"{label} is not a regular file")
    return path


def _receipt_file(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
    expected_repo_path: str | None = None,
) -> Path:
    """Reopen one content-addressed receipt inside the repository boundary."""

    path_value = receipt["path"]
    if not isinstance(path_value, str) or not path_value:
        raise MotionAdmissionError(f"{label}.path must be non-empty")
    candidate = Path(path_value).expanduser()
    try:
        root = repo_root.resolve(strict=True)
        if candidate.is_absolute():
            path = candidate.resolve(strict=True)
            path.relative_to(root)
        else:
            path = _repo_file(path_value, root, f"{label}.path")
    except (OSError, ValueError) as exc:
        raise MotionAdmissionError(
            f"{label}.path does not resolve inside repository root"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError(f"{label}.path is not a regular file")
    if expected_repo_path is not None:
        expected = root.joinpath(*PurePosixPath(expected_repo_path).parts).resolve(
            strict=True
        )
        if path != expected:
            raise MotionAdmissionError(
                f"{label}.path is not the repository verifier {expected_repo_path}"
            )
    expected_sha = _digest(receipt["sha256"], f"{label}.sha256")
    _, actual_sha = _snapshot(path, label)
    if actual_sha != expected_sha:
        raise MotionAdmissionError(f"{label} bytes differ from its receipt")
    return path


def _binding_document(binding: BankPromotionBinding) -> Mapping[str, Any]:
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
        "evidence_receipts": [
            {
                "motion_id": motion_id,
                "evidence_level": level,
                "evidence_manifest_sha256": manifest_sha,
                "certificate_sha256": list(certificate_sha),
            }
            for motion_id, level, manifest_sha, certificate_sha in zip(
                binding.motion_ids,
                binding.evidence_levels,
                binding.evidence_manifest_sha256,
                binding.evidence_certificate_sha256,
            )
        ],
        "question_bank_sha256": list(binding.question_bank_sha256),
        "training_config_sha256": list(binding.training_config_sha256),
        "onnx_model_sha256": list(binding.onnx_model_sha256),
        "onnx_metadata_sha256": list(binding.onnx_metadata_sha256),
        "adoption_manifest_sha256": list(binding.adoption_manifest_sha256),
    }


def _binding_sha256(binding: BankPromotionBinding) -> str:
    payload = json.dumps(
        _binding_document(binding),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bank_gate_report(
    binding_row: Mapping[str, Any],
    *,
    binding: BankPromotionBinding,
    repo_root: Path,
) -> None:
    row = _exact_keys(
        binding_row, _BANK_GATE_BINDING_KEYS, "bank_gate_report"
    )
    path = _repo_file(row["path"], repo_root, "bank_gate_report.path")
    expected_sha = _digest(row["sha256"], "bank_gate_report.sha256")
    payload, actual_sha = _snapshot(path, "bank gate report")
    if actual_sha != expected_sha:
        raise MotionAdmissionError(
            "bank gate report SHA-256 differs from the trusted certificate"
        )
    report = _exact_keys(
        _strict_json_bytes(payload, "bank gate report"),
        _BANK_GATE_REPORT_KEYS,
        "bank gate report",
    )
    if (
        report["schema_version"] != 1
        or report["verdict"] != "PASS"
        or report["bank_gate_pass"] is not True
        or report["candidate_integrity_pass"] is not True
        or report["grounded_trace_status"] not in ("PASS", "COMPLETE_PASS")
        # The bank gate is an evidence producer, never its own adopter.  A
        # trusted promotion certificate supplies the separate purpose-specific
        # authorization after this diagnostic-only PASS is reviewed.
        or report["publication_class"] != "post_build_diagnostic_only"
        or report["training_authorized"] is not False
        or report["hardware_authorized"] is not False
    ):
        raise MotionAdmissionError(
            "bank gate report is not a complete, promotable exact PASS"
        )
    if report["library_id"] != binding.bank_id:
        raise MotionAdmissionError(
            "bank gate library_id differs from the promoted bank_id"
        )
    grounded_claim = report.get("contracts", {}).get(
        "grounded_inverse_dynamics"
    )
    if (
        not isinstance(grounded_claim, str)
        or not grounded_claim
        or "incomplete" in grounded_claim.lower()
        or "missing" in grounded_claim.lower()
    ):
        raise MotionAdmissionError(
            "bank gate grounded inverse-dynamics claim is not complete"
        )

    manifest = _exact_keys(
        report["manifest"], frozenset({"path", "sha256"}), "bank gate manifest"
    )
    _receipt_file(
        manifest,
        repo_root=repo_root,
        label="bank gate manifest",
    )
    if not isinstance(report["bank_dir"], str) or not report["bank_dir"]:
        raise MotionAdmissionError("bank gate bank_dir must be non-empty")

    bound = _exact_keys(
        report["bound_inputs"],
        _BANK_GATE_BOUND_INPUT_KEYS,
        "bank gate bound_inputs",
    )
    bound_paths: dict[str, Path] = {}
    for name in (
        "recipe",
        "compiler",
        "geometry_tool",
        "ready",
        "mjcf",
        "urdf",
        "body_order",
    ):
        receipt = _exact_keys(
            bound[name],
            frozenset({"path", "sha256"}),
            f"bank gate bound_inputs.{name}",
        )
        expected_path = {
            "compiler": (
                "hope_training/whole_body_tracking/scripts/"
                "canonical_motion_compiler.py"
            ),
            "geometry_tool": (
                "hope_training/whole_body_tracking/scripts/"
                "canonical_motion_geometry.py"
            ),
        }.get(name)
        bound_paths[name] = _receipt_file(
            receipt,
            repo_root=repo_root,
            label=f"bank gate bound_inputs.{name}",
            expected_repo_path=expected_path,
        )
    if bound["ready"]["sha256"] != binding.canonical_ready_sha256:
        raise MotionAdmissionError(
            "bank gate ready digest differs from the promoted canonical ready"
        )
    _digest(
        bound["compiler_options_sha256"],
        "bank gate bound_inputs.compiler_options_sha256",
    )
    plant = _exact_keys(
        bound["plant"],
        frozenset(
            {
                "mjcf_sha256",
                "urdf_sha256",
                "compiled_signature_sha256",
                "identity_bound",
                "runtime_body_order",
            }
        ),
        "bank gate bound_inputs.plant",
    )
    for key in ("mjcf_sha256", "urdf_sha256", "compiled_signature_sha256"):
        _digest(plant[key], f"bank gate bound_inputs.plant.{key}")
    if (
        plant["mjcf_sha256"] != bound["mjcf"]["sha256"]
        or plant["urdf_sha256"] != bound["urdf"]["sha256"]
        or plant["identity_bound"] is not True
        or not isinstance(plant["runtime_body_order"], list)
        or not plant["runtime_body_order"]
    ):
        raise MotionAdmissionError("bank gate plant identity is not bound")
    tools = _exact_keys(
        bound["verifier_tools"],
        frozenset(
            {
                "bank_gate",
                "mujoco_motion_player",
                "canonical_mujoco_dynamics_gate",
            }
        ),
        "bank gate bound_inputs.verifier_tools",
    )
    for name, receipt_keys in (
        ("bank_gate", frozenset({"path", "sha256"})),
        ("mujoco_motion_player", frozenset({"path", "sha256"})),
        (
            "canonical_mujoco_dynamics_gate",
            frozenset({"path", "sha256", "report_schema_version"}),
        ),
    ):
        receipt = _exact_keys(
            tools[name],
            receipt_keys,
            f"bank gate verifier_tools.{name}",
        )
        _receipt_file(
            receipt,
            repo_root=repo_root,
            label=f"bank gate verifier_tools.{name}",
            expected_repo_path={
                "bank_gate": (
                    "hope_training/whole_body_tracking/scripts/"
                    "canonical_motion_bank_gate.py"
                ),
                "mujoco_motion_player": (
                    "hope_training/whole_body_tracking/scripts/"
                    "mujoco_motion_player.py"
                ),
                "canonical_mujoco_dynamics_gate": (
                    "hope_training/whole_body_tracking/scripts/"
                    "canonical_mujoco_dynamics_gate.py"
                ),
            }[name],
        )
    if (
        tools["canonical_mujoco_dynamics_gate"]["report_schema_version"] != 1
    ):
        raise MotionAdmissionError(
            "bank gate dynamics verifier report schema changed"
        )

    contracts = _exact_keys(
        report["contracts"],
        _BANK_GATE_CONTRACT_KEYS,
        "bank gate contracts",
    )
    matrix = _exact_keys(
        contracts["matrix"],
        frozenset({"motion_ids", "scopes", "count"}),
        "bank gate contracts.matrix",
    )
    if (
        matrix["motion_ids"] != list(binding.motion_ids)
        or matrix["scopes"] != ["upper", "full"]
        or matrix["count"] != 10
        or contracts["shared_ready"] is not True
        or contracts["six_endpoint_velocity_classes_exact_zero"] is not True
        or contracts["contact_opportunity_is_marker_only"] is not True
        or contracts["acceleration_allowed_through_window_end"] is not True
        or contracts["nonnegative_scalar_acceleration_through_window_end"]
        is not True
        or contracts["adv2c3_role"] != "comparator_only_not_default"
        or contracts["grounded_trace_status"]
        != report["grounded_trace_status"]
    ):
        raise MotionAdmissionError(
            "bank gate contracts do not describe the complete canonical matrix"
        )

    aggregate = _exact_keys(
        report["aggregate"],
        _BANK_GATE_AGGREGATE_KEYS,
        "bank gate aggregate",
    )
    exact_ten = (
        "clip_count",
        "fk_pass_count",
        "velocity_consistency_pass_count",
        "joint_limit_pass_count",
        "geometry_pass_count",
        "non_torque_dynamics_pass_count",
        "complete_dynamics_pass_count",
        "torque_interpretation_valid_count",
    )
    if (
        any(aggregate[key] != 10 for key in exact_ten)
        or aggregate["incomplete_fail_closed_count"] != 0
        or aggregate["failed_count"] != 0
    ):
        raise MotionAdmissionError(
            "bank gate aggregate is not a complete ten-clip dynamics PASS"
        )

    clips = report.get("clips")
    if not isinstance(clips, list) or len(clips) != 10:
        raise MotionAdmissionError(
            "bank gate report must contain the complete ten-clip matrix"
        )
    expected_matrix = tuple(
        (motion_id, scope)
        for motion_id in binding.motion_ids
        for scope in ("upper", "full")
    )
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(clips):
        clip = _exact_keys(
            raw, _BANK_GATE_CLIP_KEYS, f"bank gate clips[{index}]"
        )
        _digest(clip["sha256"], f"bank gate clips[{index}].sha256")
        schema2 = _exact_keys(
            clip["schema2_receipts"],
            frozenset(
                {
                    "input_sha256",
                    "builder_tool_sha256",
                    "manifest_sidecar",
                    "report_sidecar",
                }
            ),
            f"bank gate clips[{index}].schema2_receipts",
        )
        _digest(
            schema2["input_sha256"],
            f"bank gate clips[{index}].schema2_receipts.input_sha256",
        )
        _digest(
            schema2["builder_tool_sha256"],
            f"bank gate clips[{index}].schema2_receipts.builder_tool_sha256",
        )
        for sidecar in ("manifest_sidecar", "report_sidecar"):
            receipt = _exact_keys(
                schema2[sidecar],
                frozenset({"path", "sha256"}),
                f"bank gate clips[{index}].schema2_receipts.{sidecar}",
            )
            _receipt_file(
                receipt,
                repo_root=repo_root,
                label=(
                    f"bank gate clips[{index}]."
                    f"schema2_receipts.{sidecar}"
                ),
            )
        builder_path = repo_root.joinpath(
            "hope_training",
            "whole_body_tracking",
            "scripts",
            "canonical_schema2_builder.py",
        )
        _, builder_sha = _snapshot(builder_path, "canonical schema2 builder")
        if schema2["builder_tool_sha256"] != builder_sha:
            raise MotionAdmissionError(
                f"bank gate clips[{index}] schema2 builder digest changed"
            )
        ready = clip["strict_schema2_and_ready"]
        dynamics = clip["plant_specific_dynamics"]
        inverse = (
            dynamics.get("inverse_dynamics", {})
            if isinstance(dynamics, Mapping)
            else {}
        )
        torque = (
            inverse.get("torque_interpretation", {})
            if isinstance(inverse, Mapping)
            else {}
        )
        endpoint_zero = (
            ready.get("six_velocity_classes_exact_zero", {})
            if isinstance(ready, Mapping)
            else {}
        )
        if (
            not isinstance(ready, Mapping)
            or ready.get("shared_joint_ready_exact") is not True
            or ready.get("shared_32_body_ready_exact") is not True
            or not isinstance(endpoint_zero, Mapping)
            or frozenset(endpoint_zero)
            != frozenset(
                {
                    "joint_start",
                    "joint_end",
                    "body_linear_start",
                    "body_linear_end",
                    "body_angular_start",
                    "body_angular_end",
                }
            )
            or not all(value is True for value in endpoint_zero.values())
            or not isinstance(clip["contact_opportunity"], Mapping)
            or clip["contact_opportunity"].get(
                "acceleration_allowed_through_window_end"
            )
            is not True
            or not isinstance(clip["mujoco_fk"], Mapping)
            or clip["mujoco_fk"].get("pass") is not True
            or not isinstance(dynamics, Mapping)
            or dynamics.get("verdict") != "PASS"
            or dynamics.get("screen_pass") is not True
            or dynamics.get("non_torque_screens_pass") is not True
            or not isinstance(torque, Mapping)
            or torque.get("valid") is not True
        ):
            raise MotionAdmissionError(
                f"bank gate clips[{index}] is not an exact ready/FK/dynamics PASS"
            )
        rows.append(clip)
    observed_matrix = tuple(
        (row_["motion_id"], row_["scope"]) for row_ in rows
    )
    if observed_matrix != expected_matrix:
        raise MotionAdmissionError(
            "bank gate report clip matrix order or identity changed"
        )
    scoped = [row_ for row_ in rows if row_["scope"] == binding.scope]
    observed = tuple(
        (row_["motion_id"], row_["sha256"]) for row_ in scoped
    )
    expected = tuple(zip(binding.motion_ids, binding.npz_sha256))
    if observed != expected:
        raise MotionAdmissionError(
            "bank gate report does not bind the ordered five scope NPZ hashes"
        )


def _verify_bank_promotion_certificate_snapshot(
    certificate_path: os.PathLike[str] | str,
    *,
    binding: BankPromotionBinding,
    repo_root: os.PathLike[str] | str,
) -> tuple[Path, Path, str]:
    """Reopen and verify the complete certificate/report closure."""

    path = Path(certificate_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise MotionAdmissionError(
            f"cannot resolve promotion certificate {path}: {exc}"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError("promotion certificate is not a regular file")
    payload, certificate_sha = _snapshot(path, "promotion certificate")
    if certificate_sha not in TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256:
        raise MotionAdmissionError(
            "promotion certificate SHA-256 is absent from the code trust set"
        )
    certificate = _exact_keys(
        _strict_json_bytes(payload, "promotion certificate"),
        _CERTIFICATE_KEYS,
        "promotion certificate",
    )
    if (
        certificate["schema_version"] != 1
        or certificate["certificate_type"]
        != "canonical-motion-bank-promotion-v1"
    ):
        raise MotionAdmissionError(
            "promotion certificate schema/type is unsupported"
        )
    expected = _binding_document(binding)
    for key, value in expected.items():
        if certificate[key] != value:
            raise MotionAdmissionError(
                f"promotion certificate does not crossbind {key}"
            )
    evidence = certificate["evidence_receipts"]
    if not isinstance(evidence, list):
        raise MotionAdmissionError(
            "promotion certificate evidence_receipts must be a list"
        )
    for index, row in enumerate(evidence):
        _exact_keys(
            row,
            _EVIDENCE_RECEIPT_KEYS,
            f"promotion certificate evidence_receipts[{index}]",
        )
    _validate_bank_gate_report(
        certificate["bank_gate_report"],
        binding=binding,
        repo_root=Path(repo_root),
    )
    try:
        resolved_root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise MotionAdmissionError(
            f"cannot resolve repository root {repo_root}: {exc}"
        ) from exc
    return path, resolved_root, certificate_sha


def verify_bank_promotion_certificate(
    certificate_path: os.PathLike[str] | str,
    *,
    binding: BankPromotionBinding,
    repo_root: os.PathLike[str] | str,
) -> TrustedMotionAdmission:
    """Verify exact trusted bytes and mint one opaque bank capability."""

    path, resolved_root, certificate_sha = (
        _verify_bank_promotion_certificate_snapshot(
            certificate_path,
            binding=binding,
            repo_root=repo_root,
        )
    )
    return TrustedMotionAdmission(
        _token=_MINT_TOKEN,
        certificate_sha256=certificate_sha,
        binding_sha256=_binding_sha256(binding),
        purpose=binding.purpose,
        bank_id=binding.bank_id,
        scope=binding.scope,
        certificate_path=str(path),
        repo_root=str(resolved_root),
    )


def require_matching_admission(
    admission: Any, binding: BankPromotionBinding
) -> None:
    """Reject missing, fabricated, stale, or wrong-purpose capabilities."""

    if type(admission) is not TrustedMotionAdmission:
        raise MotionAdmissionError(
            "runtime purpose requires one opaque TrustedMotionAdmission"
        )
    try:
        path, resolved_root, certificate_sha = (
            _verify_bank_promotion_certificate_snapshot(
                admission._certificate_path,
                binding=binding,
                repo_root=admission._repo_root,
            )
        )
        required = (
            admission._binding_sha256 == _binding_sha256(binding)
            and admission._purpose == binding.purpose
            and admission._bank_id == binding.bank_id
            and admission._scope == binding.scope
            and admission._certificate_sha256 == certificate_sha
            and admission._certificate_path == str(path)
            and admission._repo_root == str(resolved_root)
        )
    except (AttributeError, MotionAdmissionError) as exc:
        raise MotionAdmissionError(
            "TrustedMotionAdmission cannot revalidate its trusted certificate"
        ) from exc
    if not required:
        raise MotionAdmissionError(
            "TrustedMotionAdmission does not match this exact registry binding"
        )


def legacy_raw_motion_hashes(
    motion_files: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> tuple[str, ...]:
    """Hash exact raw-motion bytes and require the code-owned legacy allowlist."""

    _, digests = legacy_raw_motion_snapshots(motion_files)
    return digests


def legacy_raw_motion_snapshots(
    motion_files: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> tuple[tuple[bytes, ...], tuple[str, ...]]:
    """Return the exact admitted bytes so a consumer never reopens the paths."""

    if isinstance(motion_files, (str, os.PathLike)):
        values = (motion_files,)
    else:
        try:
            values = tuple(motion_files)
        except TypeError as exc:
            raise MotionAdmissionError(
                "legacy motion files must be one path or a path sequence"
            ) from exc
    if not values:
        raise MotionAdmissionError("legacy motion file sequence may not be empty")
    snapshots: list[bytes] = []
    digests: list[str] = []
    for index, value in enumerate(values):
        path = Path(value).expanduser()
        try:
            path = path.resolve(strict=True)
        except OSError as exc:
            raise MotionAdmissionError(
                f"cannot resolve legacy motion file[{index}]: {exc}"
            ) from exc
        if not path.is_file():
            raise MotionAdmissionError(
                f"legacy motion file[{index}] is not a regular file"
            )
        payload, digest = _snapshot(path, f"legacy motion file[{index}]")
        if digest not in TRUSTED_LEGACY_RAW_MOTION_SHA256:
            raise MotionAdmissionError(
                f"legacy motion file[{index}] SHA-256 is absent from the "
                "code trust set"
            )
        snapshots.append(payload)
        digests.append(digest)
    return tuple(snapshots), tuple(digests)


__all__ = [
    "BankPromotionBinding",
    "MotionAdmissionError",
    "TrustedMotionAdmission",
    "legacy_raw_motion_hashes",
    "legacy_raw_motion_snapshots",
    "require_matching_admission",
    "verify_bank_promotion_certificate",
]
