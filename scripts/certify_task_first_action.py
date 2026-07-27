#!/usr/bin/env python3
"""Fail-closed certification for one task-first action across upper/full scopes.

This module is deliberately a *composition* layer.  It does not replace the
canonical compiler, bank verifier, MuJoCo player, dynamics gate, or analytic
return scorer:

* ``mujoco_motion_player`` is the authority for the physical ``right_racket``
  site twist.  Its site velocity is recovered from generalized velocity via
  ``mj_jacSite``/``mj_objectVelocity`` and therefore includes the
  ``omega x offset`` contribution.  Wrist-COM speed is never accepted here.
* ``canonical_mujoco_dynamics_gate`` supplies the exact vendor-model binding
  and its contact classification.  The ``scan-collisions`` subcommand only
  adds the missing stationary-X comparison and dense pose interpolation.
* ``reference_return_gate`` remains the legal-return scorer.

The certifier consumes a SHA-256-bound plan and SHA-256-bound evidence.  It
compares the required whole-action/task-center translations
``[[0, 0], [-0.05, 0], [-0.10, 0]]`` metres (negative X is farther from the
table), but it never chooses one.  Reference evidence can populate diagnostics,
but this command always emits a blocked diagnostic: only the code-rooted generic
registry/admission path may mint a runtime capability.  A simulator smoke still
requires an in-process producer chain, and formal training additionally requires
the canonical verifier's exact grounded collocation trace and a trusted promotion
certificate.

No command in this file calls ``mj_step``, trains a policy, deploys, or issues
hardware commands.  Outputs are no-clobber JSON receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 4
PLAN_KIND = "task_first_action_certification_plan_v4"
COLLISION_REPORT_KIND = "task_first_station_center_collision_v4"
CERTIFICATE_KIND = "task_first_action_prerun_diagnostic_v4"
BEHAVIOR_CONTACT_EVIDENCE_KIND = (
    "task_first_action_behavior_contact_evidence_v1"
)
STATION_SELECTION_APPROVAL_KIND = (
    "task_first_station_selection_approval_v1"
)
SCOPES = ("upper", "full")
STATION_CENTER_SHIFT_CANDIDATES_XY_M = (
    (0.0, 0.0),
    (-0.05, 0.0),
    (-0.10, 0.0),
)
RACKET_SITE_NAME = "right_racket"
RACKET_SITE_BODY = "right_wrist_yaw_Link"
RACKET_SITE_OFFSET_WRIST_M = (0.21021, 0.032078, 0.032036)
RACKET_VELOCITY_SOURCE = (
    "schema root/joint qvel -> MuJoCo root Jacobian -> "
    "mj_jacSite and mj_objectVelocity; no pose finite differences"
)
RACKET_COLLISION_GEOMS = ("right_racket_collision", "right_racket_handle_collision")
TABLE_TOP = "motion_table_top"
NET = "motion_net"
NET_POSTS = ("motion_net_post_left", "motion_net_post_right")
OBSTACLES = (TABLE_TOP, NET) + NET_POSTS
REQUIRED_GATES = (
    "canonical_candidate_integrity",
    "grounded_dynamics",
    "post_retime_t_hit",
    "post_retime_t_cycle",
    "physical_blade_site_speed",
    "dense_collision",
    "shared_ready_return",
    "reference_returnability",
)
DIAGNOSTIC_REFERENCE_GATES = (
    "canonical_candidate_integrity",
    "compiler_anchor_in_preregistered_range",
    "post_retime_t_hit",
    "post_retime_t_cycle",
    "physical_blade_site_speed",
    "dense_collision",
    "shared_ready_return",
    "reference_returnability",
)
STATION_COMPARISON_GATES = tuple(
    gate for gate in DIAGNOSTIC_REFERENCE_GATES if gate != "post_retime_t_hit"
)

# These are code-reviewed *admissibility* limits for a diagnostic plan, not
# action-specific acceptance values.  The plan may choose a stricter interval
# inside this envelope, but it may not make a weak reference look green by
# choosing zero, effectively unbounded, or undersampled gates.  Action-specific
# values remain content-bound in the plan and must be reviewed before use.
SOURCE_ANCHOR_TIME_LIMITS_S = (0.01, 0.5)
SOURCE_ANCHOR_MAX_WINDOW_S = 0.25
# Franco's 2026-07-25 decision made 0.5 s a comparison reference, not a
# universal acceptance threshold.  A formal t_hit gate must come from the
# action-specific, code-rooted behavior/contact authority used by motion
# admission.  This diagnostic certifier deliberately has no such authority.
T_HIT_REFERENCE_S = 0.5
T_CYCLE_LIMITS_S = (0.05, 3.0)
T_CYCLE_MAX_WINDOW_S = 1.0
BLADE_SITE_SPEED_LIMITS_M_S = (1.0, 7.2)
BLADE_SITE_SPEED_MAX_WINDOW_M_S = 6.2
SHARED_READY_POSE_TOLERANCE_MAX = 1.0e-6
DENSE_COLLISION_MIN_HZ = 400.0
MINIMUM_TABLE_NET_CLEARANCE_M = 0.005
REFERENCE_RETURN_MIN_SAMPLES = 256
REFERENCE_RETURN_MIN_FRACTION = 0.5
REFERENCE_RETURN_MAX_CAPTURE_RADIUS_M = 0.095
REFERENCE_RETURN_MIN_APPROACH_SPEED_M_S = 0.3
REFERENCE_RETURN_MIN_AXIS_SPAN_M_S = 0.01
VENUE_BALL_SPEED_RANGE_M_S = (1.0, 7.0)
VENUE_SPIN_MAGNITUDE_MAX_RAD_S = 15.0 * 2.0 * math.pi
CODE_ROOTED_BALL_PHYSICS = Path("configs/ball_physics_venue.yaml")
CODE_ROOTED_VENUE_PROFILE = Path(
    "configs/venue_profiles/franco_rig_20260725.json"
)

# These are code-owned authorization roots, not Hydra/plan inputs.  They ship
# empty deliberately.  Adding one exact receipt digest requires source review;
# until then a plan cannot self-report behavior contact or approve a station
# shift after seeing comparison results.
TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256: frozenset[str] = frozenset()
TRUSTED_STATION_SELECTION_APPROVAL_SHA256: frozenset[str] = frozenset()

_BEHAVIOR_CONTACT_AUTHORITY_CONTRACT = {
    "schema_version": 1,
    "kind": BEHAVIOR_CONTACT_EVIDENCE_KIND,
    "action_specific_t_hit_range": True,
    "ordered_scopes": list(SCOPES),
    "motion_bytes_bound_per_scope": True,
    "evidence_artifact_bytes_bound": True,
    "plan_or_report_may_self_authorize": False,
}
BEHAVIOR_CONTACT_AUTHORITY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _BEHAVIOR_CONTACT_AUTHORITY_CONTRACT,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()

_BEHAVIOR_CONTACT_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "authority_contract_sha256",
        "action_id",
        "accepted_t_hit_range_s",
        "measurements",
        "evidence_artifact",
        "non_claims",
    }
)
_BEHAVIOR_CONTACT_MEASUREMENT_KEYS = frozenset(
    {"scope", "motion_sha256", "t_hit_s"}
)
_STATION_SELECTION_APPROVAL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "action_id",
        "selected_station_center_shift_xy_m",
        "comparison_input_sha256",
        "approval_policy",
        "non_claims",
    }
)

_BANK_REPORT_KEYS = frozenset(
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
_BANK_BOUND_INPUT_KEYS = frozenset(
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
_BANK_CLIP_KEYS = frozenset(
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
_BANK_CONTRACT_KEYS = frozenset(
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
_BANK_AGGREGATE_KEYS = frozenset(
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
_PLAYBACK_REPORT_KEYS = frozenset(
    {
        "verdict",
        "evidence_boundary",
        "authorization",
        "contract",
        "motion",
        "gates",
        "racket",
        "per_body_max",
        "artifacts",
    }
)
_COLLISION_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "report_kind",
        "action_id",
        "scope",
        "station_center_shift_xy_m",
        "verdict",
        "artifacts",
        "sampling",
        "model",
        "checks",
        "clearance",
        "authorization",
        "non_claims",
    }
)


class CertificationError(ValueError):
    """A malformed, unbound, contradictory, or incomplete certification input."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    data: bytes
    sha256: str

    def binding(self) -> Dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise CertificationError(f"{label} must be one lowercase SHA-256")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CertificationError(f"{label} must be a non-empty trimmed string")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CertificationError(f"{label} must be a JSON boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CertificationError(f"{label} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CertificationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise CertificationError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CertificationError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CertificationError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise CertificationError(f"{label} must be a JSON array")
    return value


def _exact_keys(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    expected = frozenset(keys)
    if frozenset(row) != expected:
        raise CertificationError(
            f"{label} keys changed: expected={sorted(expected)} actual={sorted(row)}"
        )
    return row


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CertificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise CertificationError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CertificationError(f"cannot parse strict {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CertificationError(f"{label} must be a JSON object")
    return value


def _resolve_bound_path(raw: str, base_dir: Path) -> Path:
    path = Path(_nonempty(raw, "binding.path")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Path(os.path.abspath(os.fspath(path)))


def read_bound_file(binding: Any, base_dir: Path, label: str) -> Snapshot:
    row = _exact_keys(binding, ("path", "sha256"), label)
    path = _resolve_bound_path(row["path"], base_dir)
    expected = _digest(row["sha256"], f"{label}.sha256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CertificationError(f"cannot open exact {label}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CertificationError(f"{label} must be a regular non-symlink file")
        chunks: List[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise CertificationError(f"{label} changed during immutable read")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise CertificationError(f"{label} short/long read")
        actual = _sha256_bytes(data)
        if actual != expected:
            raise CertificationError(
                f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
            )
        return Snapshot(path=path, data=data, sha256=actual)
    finally:
        os.close(fd)


def read_recipe_bound_file(
    binding: Any,
    *,
    plan_base_dir: Path,
    label: str,
) -> Snapshot:
    """Resolve recipe-owned paths without silently accepting path drift."""

    row = _exact_keys(binding, ("path", "sha256"), label)
    raw = _nonempty(row["path"], f"{label}.path")
    path = Path(raw).expanduser()
    if path.is_absolute():
        bases = (Path("/"),)
    else:
        bases = (plan_base_dir, Path(__file__).resolve().parents[1])
    matches: List[Path] = []
    for base in bases:
        candidate = path if path.is_absolute() else base / path
        candidate = Path(os.path.abspath(os.fspath(candidate)))
        if candidate.is_file() and not candidate.is_symlink():
            matches.append(candidate)
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise CertificationError(
            f"{label} path must resolve to exactly one regular file, got "
            f"{[str(item) for item in unique]}"
        )
    return read_bound_file(
        {"path": str(unique[0]), "sha256": row["sha256"]},
        Path("/"),
        label,
    )


def _require_code_rooted_bytes(
    snapshot: Snapshot,
    *,
    repository_relative_path: Path,
    label: str,
) -> None:
    """Require bytes identical to the reviewed repository physics/profile truth."""

    expected_path = Path(__file__).resolve().parents[1] / repository_relative_path
    if (
        not expected_path.is_file()
        or expected_path.is_symlink()
        or snapshot.sha256 != _sha256_file(expected_path)
    ):
        raise CertificationError(
            f"{label} must bind exact code-rooted {repository_relative_path.as_posix()}"
        )


def _same_float(lhs: float, rhs: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(lhs) - float(rhs)) <= tolerance


def _station_index(value: Any, label: str) -> int:
    vector = _sequence(value, label)
    if len(vector) != 2:
        raise CertificationError(f"{label} must be one [x, y] pair")
    xy = (_finite(vector[0], f"{label}[0]"), _finite(vector[1], f"{label}[1]"))
    matches = [
        index
        for index, candidate in enumerate(STATION_CENTER_SHIFT_CANDIDATES_XY_M)
        if all(
            _same_float(actual, expected, tolerance=1.0e-12)
            for actual, expected in zip(xy, candidate)
        )
    ]
    if len(matches) != 1:
        raise CertificationError(
            f"{label} must be one of "
            f"{[list(row) for row in STATION_CENTER_SHIFT_CANDIDATES_XY_M]}, got {xy}"
        )
    return matches[0]


def _canonical_document_sha256(value: Any, label: str) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CertificationError(
            f"{label} must contain only finite canonical JSON data"
        ) from exc
    return _sha256_bytes(payload)


def _validate_code_trust_set(value: Any, label: str) -> frozenset[str]:
    if (
        type(value) is not frozenset
        or any(
            type(digest) is not str
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
            for digest in value
        )
    ):
        raise CertificationError(f"{label} code trust set is malformed")
    return value


def _station_comparison_input_sha256(plan: Mapping[str, Any]) -> str:
    """Bind every comparison input without circularly binding its approval."""

    excluded = {
        "selected_station_center_shift_xy_m",
        "station_selection_approval",
    }
    comparison = {
        key: value for key, value in plan.items() if key not in excluded
    }
    return _canonical_document_sha256(
        comparison, "station comparison input"
    )


def _load_trusted_behavior_contact_evidence(
    binding: Any,
    *,
    base_dir: Path,
    action_id: str,
) -> Optional[Mapping[str, Any]]:
    """Load action-specific t_hit truth only through a code-pinned receipt."""

    if binding is None:
        return None
    snapshot = read_bound_file(
        binding, base_dir, "behavior/contact evidence receipt"
    )
    trusted = _validate_code_trust_set(
        TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256,
        "behavior/contact evidence",
    )
    if not trusted:
        raise CertificationError(
            "behavior/contact evidence code trust set is empty"
        )
    if snapshot.sha256 not in trusted:
        raise CertificationError(
            "behavior/contact evidence receipt is not code-pinned"
        )
    receipt = _exact_keys(
        _parse_json(snapshot.data, "behavior/contact evidence receipt"),
        _BEHAVIOR_CONTACT_EVIDENCE_KEYS,
        "behavior/contact evidence receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != BEHAVIOR_CONTACT_EVIDENCE_KIND
        or receipt["authority_contract_sha256"]
        != BEHAVIOR_CONTACT_AUTHORITY_CONTRACT_SHA256
        or receipt["action_id"] != action_id
    ):
        raise CertificationError(
            "behavior/contact evidence identity/authority drifted"
        )
    accepted = _sequence(
        receipt["accepted_t_hit_range_s"],
        "behavior/contact accepted t_hit range",
    )
    if len(accepted) != 2:
        raise CertificationError(
            "behavior/contact accepted t_hit range must have two bounds"
        )
    accepted_lo = _finite(
        accepted[0], "behavior/contact accepted t_hit lower"
    )
    accepted_hi = _finite(
        accepted[1], "behavior/contact accepted t_hit upper"
    )
    if accepted_lo < 0.0 or accepted_lo > accepted_hi:
        raise CertificationError(
            "behavior/contact accepted t_hit range is malformed"
        )
    raw_measurements = _sequence(
        receipt["measurements"], "behavior/contact measurements"
    )
    if len(raw_measurements) != len(SCOPES):
        raise CertificationError(
            "behavior/contact evidence must contain upper/full measurements"
        )
    measurements: Dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_measurements):
        row = _exact_keys(
            raw,
            _BEHAVIOR_CONTACT_MEASUREMENT_KEYS,
            f"behavior/contact measurements[{index}]",
        )
        scope = row["scope"]
        if scope != SCOPES[index]:
            raise CertificationError(
                "behavior/contact measurement scope order drifted"
            )
        motion_sha256 = _digest(
            row["motion_sha256"],
            f"behavior/contact {scope} motion SHA-256",
        )
        t_hit_s = _finite(
            row["t_hit_s"], f"behavior/contact {scope} t_hit_s"
        )
        if not accepted_lo <= t_hit_s <= accepted_hi:
            raise CertificationError(
                f"behavior/contact {scope} t_hit lies outside its "
                "code-pinned action-specific range"
            )
        measurements[scope] = {
            "scope": scope,
            "motion_sha256": motion_sha256,
            "t_hit_s": t_hit_s,
        }
    artifact = read_bound_file(
        receipt["evidence_artifact"],
        snapshot.path.parent,
        "behavior/contact evidence artifact",
    )
    non_claims = _sequence(
        receipt["non_claims"], "behavior/contact evidence non_claims"
    )
    if (
        not non_claims
        or any(type(item) is not str or not item for item in non_claims)
    ):
        raise CertificationError(
            "behavior/contact evidence non_claims must be non-empty strings"
        )
    return {
        "receipt": snapshot.binding(),
        "receipt_sha256": snapshot.sha256,
        "authority_contract_sha256": (
            BEHAVIOR_CONTACT_AUTHORITY_CONTRACT_SHA256
        ),
        "accepted_t_hit_range_s": [accepted_lo, accepted_hi],
        "measurements": measurements,
        "evidence_artifact": artifact.binding(),
    }


def _load_trusted_station_selection_approval(
    binding: Any,
    *,
    base_dir: Path,
    action_id: str,
    selected: Optional[Tuple[float, float]],
    comparison_input_sha256: str,
) -> Optional[Mapping[str, Any]]:
    """Require a separate code-reviewed receipt for any selected shift."""

    if selected is None:
        if binding is not None:
            raise CertificationError(
                "station selection approval is forbidden when no shift is "
                "selected"
            )
        return None
    if binding is None:
        raise CertificationError(
            "selected station center shift requires an independent "
            "code-pinned approval receipt"
        )
    snapshot = read_bound_file(
        binding, base_dir, "station selection approval receipt"
    )
    trusted = _validate_code_trust_set(
        TRUSTED_STATION_SELECTION_APPROVAL_SHA256,
        "station selection approval",
    )
    if not trusted:
        raise CertificationError(
            "station selection approval code trust set is empty"
        )
    if snapshot.sha256 not in trusted:
        raise CertificationError(
            "station selection approval receipt is not code-pinned"
        )
    receipt = _exact_keys(
        _parse_json(snapshot.data, "station selection approval receipt"),
        _STATION_SELECTION_APPROVAL_KEYS,
        "station selection approval receipt",
    )
    approved_index = _station_index(
        receipt["selected_station_center_shift_xy_m"],
        "approved station center shift",
    )
    approved = STATION_CENTER_SHIFT_CANDIDATES_XY_M[approved_index]
    non_claims = _sequence(
        receipt["non_claims"], "station selection approval non_claims"
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != STATION_SELECTION_APPROVAL_KIND
        or receipt["action_id"] != action_id
        or approved != selected
        or receipt["comparison_input_sha256"]
        != comparison_input_sha256
        or receipt["approval_policy"]
        != "independent_code_reviewed_station_selection_v1"
        or not non_claims
        or any(type(item) is not str or not item for item in non_claims)
    ):
        raise CertificationError(
            "station selection approval identity/comparison binding drifted"
        )
    return {
        "receipt": snapshot.binding(),
        "receipt_sha256": snapshot.sha256,
        "comparison_input_sha256": comparison_input_sha256,
        "selected_station_center_shift_xy_m": list(selected),
    }


def _load_npz(snapshot: Snapshot) -> Mapping[str, np.ndarray]:
    import io

    try:
        with np.load(io.BytesIO(snapshot.data), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise CertificationError(f"cannot parse motion NPZ {snapshot.path}: {exc}") from exc


def _scalar_array(value: Any, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise CertificationError(f"{label} must contain one scalar")
    return _finite(array[0], label)


def _text_scalar_array(value: Any, label: str) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise CertificationError(f"{label} must contain one scalar string")
    raw = array[0]
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeError as exc:
            raise CertificationError(f"{label} is not UTF-8") from exc
    return _nonempty(str(raw), label)


def _quat_angle_rad(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    left = np.asarray(lhs, np.float64)
    right = np.asarray(rhs, np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _motion_receipt(
    payload: Mapping[str, np.ndarray],
    *,
    pose_tolerance: float,
) -> Mapping[str, Any]:
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "kinematics_schema_version",
        "body_pos_point",
        "body_lin_vel_point",
        "body_names",
    }
    if frozenset(payload) not in (frozenset(required), frozenset(required | {
        "kinematics_migration_source_sha256",
        "kinematics_migration_source_point",
        "kinematics_migration_tool",
    })):
        raise CertificationError("compiled motion must use the exact schema-2 11/14-field set")
    fps = _scalar_array(payload["fps"], "motion.fps")
    if fps <= 0.0:
        raise CertificationError("motion.fps must be positive")
    if _scalar_array(payload["kinematics_schema_version"], "motion schema") != 2.0:
        raise CertificationError("compiled motion must use kinematics_schema_version=2")
    if _text_scalar_array(payload["body_pos_point"], "body_pos_point") != "link_origin":
        raise CertificationError("compiled motion body_pos_point must be link_origin")
    if (
        _text_scalar_array(payload["body_lin_vel_point"], "body_lin_vel_point")
        != "center_of_mass"
    ):
        raise CertificationError(
            "compiled motion body_lin_vel_point must be center_of_mass"
        )
    body_names = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(payload["body_names"]).reshape(-1).tolist()
    )
    body_order_path = (
        Path(__file__).resolve().parents[1] / "configs/a3_runtime_body_order.txt"
    )
    try:
        expected_body_names = tuple(
            line.strip()
            for line in body_order_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise CertificationError(f"cannot read runtime body order: {exc}") from exc
    if body_names != expected_body_names:
        raise CertificationError("compiled motion body_names differ from runtime order")
    joint_pos = np.asarray(payload["joint_pos"], np.float64)
    joint_vel = np.asarray(payload["joint_vel"], np.float64)
    body_pos = np.asarray(payload["body_pos_w"], np.float64)
    body_quat = np.asarray(payload["body_quat_w"], np.float64)
    body_lin = np.asarray(payload["body_lin_vel_w"], np.float64)
    body_ang = np.asarray(payload["body_ang_vel_w"], np.float64)
    if (
        joint_pos.ndim != 2
        or joint_pos.shape[1] != 31
        or joint_vel.shape != joint_pos.shape
        or body_pos.shape != (joint_pos.shape[0], 32, 3)
        or body_quat.shape != (joint_pos.shape[0], 32, 4)
        or body_lin.shape != body_pos.shape
        or body_ang.shape != body_pos.shape
        or joint_pos.shape[0] < 2
    ):
        raise CertificationError("compiled motion schema-2 shapes disagree")
    arrays = (joint_pos, joint_vel, body_pos, body_quat, body_lin, body_ang)
    if not all(np.isfinite(array).all() for array in arrays):
        raise CertificationError("compiled motion contains NaN/Inf")
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0))
    )
    if quaternion_norm_error > 2.0e-3:
        raise CertificationError("compiled motion contains non-unit body quaternions")
    frames = int(joint_pos.shape[0])
    joint_delta = float(np.max(np.abs(joint_pos[-1] - joint_pos[0])))
    body_pos_delta = float(np.max(np.linalg.norm(body_pos[-1] - body_pos[0], axis=1)))
    body_quat_delta = float(np.max(_quat_angle_rad(body_quat[-1], body_quat[0])))
    endpoint_nonzero = int(
        np.count_nonzero(joint_vel[[0, -1]])
        + np.count_nonzero(body_lin[[0, -1]])
        + np.count_nonzero(body_ang[[0, -1]])
    )
    return {
        "frames": frames,
        "fps": fps,
        "duration_s": (frames - 1) / fps,
        "shared_ready_return": {
            "pass": bool(
                joint_delta <= pose_tolerance
                and body_pos_delta <= pose_tolerance
                and body_quat_delta <= pose_tolerance
                and endpoint_nonzero == 0
            ),
            "joint_position_max_abs_delta_rad": joint_delta,
            "body_position_max_delta_m": body_pos_delta,
            "body_orientation_max_delta_rad": body_quat_delta,
            "endpoint_velocity_nonzero_value_count": endpoint_nonzero,
            "pose_tolerance": pose_tolerance,
            "velocity_policy": "exact_zero_joint_body_linear_body_angular_at_both_endpoints",
        },
    }


def _canonical_ready_state(snapshot: Snapshot) -> Mapping[str, np.ndarray]:
    payload = _load_npz(snapshot)
    required = {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "source_segment",
        "source_npz",
        "source_frame",
        "striking_joint_ids",
        "note",
    }
    if frozenset(payload) != frozenset(required):
        raise CertificationError("canonical ready NPZ field set changed")
    joint_pos = np.asarray(payload["joint_pos"], np.float64)
    joint_vel = np.asarray(payload["joint_vel"], np.float64)
    root_pos = np.asarray(payload["root_pos_w"], np.float64)
    root_quat = np.asarray(payload["root_quat_w"], np.float64)
    if (
        joint_pos.shape != (31,)
        or joint_vel.shape != (31,)
        or root_pos.shape != (3,)
        or root_quat.shape != (4,)
        or not all(
            np.isfinite(array).all()
            for array in (joint_pos, joint_vel, root_pos, root_quat)
        )
        or abs(float(np.linalg.norm(root_quat)) - 1.0) > 2.0e-3
        or np.count_nonzero(joint_vel) != 0
    ):
        raise CertificationError("canonical ready pose/velocity contract is malformed")
    return {
        "joint_pos": joint_pos,
        "root_pos": root_pos,
        "root_quat": root_quat / np.linalg.norm(root_quat),
    }


def _canonical_ready_fk_state(
    snapshot: Snapshot,
    *,
    canonical_ready_sha256: str,
    canonical_ready: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    """Validate the content-addressed 32-body FK truth bound to canonical ready."""

    payload = _load_npz(snapshot)
    required = {
        "canonical_ready_sha256",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "kinematics_contract_version",
    }
    if frozenset(payload) != frozenset(required):
        raise CertificationError("canonical ready-FK NPZ field set changed")
    if (
        _text_scalar_array(
            payload["canonical_ready_sha256"],
            "ready-FK canonical_ready_sha256",
        )
        != canonical_ready_sha256
    ):
        raise CertificationError(
            "canonical ready-FK does not bind the exact canonical-ready digest"
        )
    version = np.asarray(payload["kinematics_contract_version"])
    if (
        version.dtype != np.dtype(np.int64)
        or version.shape != (1,)
        or int(version[0]) != 1
    ):
        raise CertificationError(
            "canonical ready-FK kinematics_contract_version must be exact int64 [1]"
        )
    body_order_path = (
        Path(__file__).resolve().parents[1] / "configs/a3_runtime_body_order.txt"
    )
    expected_names = tuple(
        line.strip()
        for line in body_order_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    raw_names = np.asarray(payload["body_names"])
    names = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_names.reshape(-1).tolist()
    )
    body_pos = np.asarray(payload["body_pos_w"])
    body_quat = np.asarray(payload["body_quat_w"])
    if (
        raw_names.shape != (32,)
        or raw_names.dtype.kind not in ("U", "S")
        or names != expected_names
        or body_pos.dtype != np.dtype(np.float32)
        or body_pos.shape != (32, 3)
        or body_quat.dtype != np.dtype(np.float32)
        or body_quat.shape != (32, 4)
        or not np.isfinite(body_pos).all()
        or not np.isfinite(body_quat).all()
        or float(np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)))
        > 2.0e-3
    ):
        raise CertificationError("canonical ready-FK body truth is malformed")
    if (
        not np.array_equal(
            body_pos[0],
            np.asarray(canonical_ready["root_pos"], dtype=np.float32),
        )
        or not np.array_equal(
            body_quat[0],
            np.asarray(canonical_ready["root_quat"], dtype=np.float32),
        )
    ):
        raise CertificationError(
            "canonical ready-FK root differs from the exact canonical-ready pose"
        )
    return {
        "body_pos": body_pos.astype(np.float64),
        "body_quat": body_quat.astype(np.float64),
    }


def _motion_ready_truth_gate(
    payload: Mapping[str, np.ndarray],
    ready: Mapping[str, np.ndarray],
    ready_fk: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> Mapping[str, Any]:
    joints = np.asarray(payload["joint_pos"], np.float64)
    body_pos = np.asarray(payload["body_pos_w"], np.float64)
    body_quat = np.asarray(payload["body_quat_w"], np.float64)
    joint_error = float(
        max(
            np.max(np.abs(joints[0] - ready["joint_pos"])),
            np.max(np.abs(joints[-1] - ready["joint_pos"])),
        )
    )
    root_position_error = float(
        max(
            np.linalg.norm(body_pos[0, 0] - ready["root_pos"]),
            np.linalg.norm(body_pos[-1, 0] - ready["root_pos"]),
        )
    )
    root_orientation_error = float(
        max(
            _quat_angle_rad(body_quat[0, 0], ready["root_quat"]),
            _quat_angle_rad(body_quat[-1, 0], ready["root_quat"]),
        )
    )
    body_position_error = float(
        max(
            np.max(
                np.linalg.norm(
                    body_pos[0] - np.asarray(ready_fk["body_pos"], np.float64),
                    axis=1,
                )
            ),
            np.max(
                np.linalg.norm(
                    body_pos[-1] - np.asarray(ready_fk["body_pos"], np.float64),
                    axis=1,
                )
            ),
        )
    )
    body_orientation_error = float(
        max(
            np.max(
                _quat_angle_rad(
                    body_quat[0],
                    np.asarray(ready_fk["body_quat"], np.float64),
                )
            ),
            np.max(
                _quat_angle_rad(
                    body_quat[-1],
                    np.asarray(ready_fk["body_quat"], np.float64),
                )
            ),
        )
    )
    passed = bool(
        joint_error <= tolerance
        and root_position_error <= tolerance
        and root_orientation_error <= tolerance
        and body_position_error <= tolerance
        and body_orientation_error <= tolerance
    )
    return {
        "pass": passed,
        "joint_position_max_abs_error_rad": joint_error,
        "root_position_max_error_m": root_position_error,
        "root_orientation_max_error_rad": root_orientation_error,
        "body_position_max_error_m": body_position_error,
        "body_orientation_max_error_rad": body_orientation_error,
        "tolerance": tolerance,
        "truth_source": (
            f"{snapshot_binding_label('canonical_ready')}+"
            f"{snapshot_binding_label('canonical_ready_fk')}"
        ),
    }


def snapshot_binding_label(name: str) -> str:
    """Small JSON-stable label used inside nested receipts."""

    return f"content_bound_{name}_npz"


def _manifest_output(
    manifest: Mapping[str, Any], action_id: str, scope: str
) -> Mapping[str, Any]:
    rows = _sequence(manifest.get("outputs"), "build_manifest.outputs")
    matches = [
        _mapping(row, "build_manifest output")
        for row in rows
        if isinstance(row, dict)
        and row.get("motion_id") == action_id
        and row.get("scope") == scope
    ]
    if len(matches) != 1:
        raise CertificationError(
            f"build manifest must contain exactly one {action_id}/{scope} output"
        )
    return matches[0]


def _bank_clip(
    report: Mapping[str, Any], action_id: str, scope: str
) -> Mapping[str, Any]:
    rows = _sequence(report.get("clips"), "canonical verifier clips")
    matches = [
        _exact_keys(row, _BANK_CLIP_KEYS, "canonical verifier clip")
        for row in rows
        if isinstance(row, dict)
        and row.get("motion_id") == action_id
        and row.get("scope") == scope
    ]
    if len(matches) != 1:
        raise CertificationError(
            f"canonical verifier must contain exactly one {action_id}/{scope} clip"
        )
    return matches[0]


def _load_marker_row(authority: Snapshot, action_id: str) -> Any:
    scripts_dir = (
        Path(__file__).resolve().parents[1]
        / "hope_training/whole_body_tracking/scripts"
    )
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import canonical_motion_markers  # type: ignore

        semantics = canonical_motion_markers.load_canonical_motion_markers(
            authority.path,
            expected_authority_sha256=authority.sha256,
            repo_root=Path(__file__).resolve().parents[1],
            profile="v3",
        )
        return semantics.row(action_id)
    except Exception as exc:
        raise CertificationError(
            f"canonical marker v3 authority validation failed: {exc}"
        ) from exc


def _validate_bank_contract(
    report: Mapping[str, Any],
    *,
    manifest_sha256: str,
    recipe_sha256: str,
    mjcf_sha256: str,
    urdf_sha256: str,
    ready_sha256: str,
) -> Mapping[str, bool]:
    report = _exact_keys(report, _BANK_REPORT_KEYS, "canonical verifier report")
    if (
        report.get("schema_version") != 1
        or report.get("publication_class") != "post_build_diagnostic_only"
        or type(report.get("candidate_integrity_pass")) is not bool
        or type(report.get("bank_gate_pass")) is not bool
        or report.get("training_authorized") is not False
        or report.get("hardware_authorized") is not False
    ):
        raise CertificationError("canonical verifier top-level contract is malformed")
    _nonempty(report["library_id"], "canonical verifier library_id")
    _nonempty(report["bank_dir"], "canonical verifier bank_dir")
    non_claims = _sequence(report["non_claims"], "canonical verifier non_claims")
    if any(not isinstance(item, str) or not item for item in non_claims):
        raise CertificationError("canonical verifier non_claims must be non-empty strings")
    verdict = report.get("verdict")
    gate_pass = report["bank_gate_pass"]
    grounded_trace_status = report.get("grounded_trace_status")
    # The currently pinned canonical verifier explicitly publishes no positive
    # grounded result: schema-2 has no exact qacc/time-law collocation trace.
    # Treating a hand-written PASS JSON as provenance would be a forged
    # certificate.  Promotion stays fail-closed until the verifier contract
    # itself changes and this composition layer is deliberately reviewed.
    if (
        gate_pass is not False
        or verdict not in ("FAIL", "INCOMPLETE_FAIL_CLOSED")
        or grounded_trace_status != "MISSING_INCOMPLETE_FAIL_CLOSED"
    ):
        raise CertificationError("canonical verifier verdict/bank_gate_pass contradict")
    manifest = _exact_keys(
        report.get("manifest"), ("path", "sha256"), "canonical verifier manifest"
    )
    _nonempty(manifest["path"], "canonical verifier manifest.path")
    if manifest.get("sha256") != manifest_sha256:
        raise CertificationError("canonical verifier manifest SHA differs from plan")
    bound_inputs = _exact_keys(
        report.get("bound_inputs"),
        _BANK_BOUND_INPUT_KEYS,
        "canonical verifier bound_inputs",
    )
    for name, expected_sha in (
        ("recipe", recipe_sha256),
        ("ready", ready_sha256),
        ("mjcf", mjcf_sha256),
        ("urdf", urdf_sha256),
    ):
        receipt = _exact_keys(
            bound_inputs[name],
            ("path", "sha256"),
            f"canonical verifier {name}",
        )
        _nonempty(receipt["path"], f"canonical verifier {name}.path")
        if receipt["sha256"] != expected_sha:
            raise CertificationError(
                f"canonical verifier {name} binding differs from plan"
            )
    _digest(
        bound_inputs["compiler_options_sha256"],
        "canonical verifier compiler_options_sha256",
    )
    code_rooted_inputs = {
        "compiler": Path(
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_compiler.py"
        ),
        "geometry_tool": Path(
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_geometry.py"
        ),
        "body_order": Path("configs/a3_runtime_body_order.txt"),
    }
    for name, relative_path in code_rooted_inputs.items():
        receipt = _exact_keys(
            bound_inputs[name],
            ("path", "sha256"),
            f"canonical verifier {name}",
        )
        expected = Path(__file__).resolve().parents[1] / relative_path
        if (
            Path(_nonempty(receipt["path"], f"canonical verifier {name}.path")).name
            != expected.name
            or not expected.is_file()
            or receipt["sha256"] != _sha256_file(expected)
        ):
            raise CertificationError(
                f"canonical verifier {name} is not exact code-rooted input"
            )
    if (
        _mapping(bound_inputs.get("recipe"), "canonical verifier recipe").get("sha256")
        != recipe_sha256
        or _mapping(bound_inputs.get("mjcf"), "canonical verifier mjcf").get("sha256")
        != mjcf_sha256
    ):
        raise CertificationError("canonical verifier recipe/MJCF binding differs from plan")
    plant = _exact_keys(
        bound_inputs["plant"],
        (
            "mjcf_sha256",
            "urdf_sha256",
            "compiled_signature_sha256",
            "identity_bound",
            "runtime_body_order",
        ),
        "canonical verifier plant",
    )
    if (
        plant["mjcf_sha256"] != mjcf_sha256
        or plant["urdf_sha256"] != urdf_sha256
        or plant["identity_bound"] is not True
        or not isinstance(plant["runtime_body_order"], list)
        or len(plant["runtime_body_order"]) != 32
    ):
        raise CertificationError("canonical verifier plant identity is malformed")
    _digest(
        plant["compiled_signature_sha256"],
        "canonical verifier compiled signature",
    )
    tools = _exact_keys(
        bound_inputs.get("verifier_tools"),
        (
            "bank_gate",
            "mujoco_motion_player",
            "canonical_mujoco_dynamics_gate",
        ),
        "canonical verifier tools",
    )
    tool_paths = {
        "bank_gate": (
            "canonical bank gate",
            Path(__file__).resolve().parents[1]
            / "hope_training/whole_body_tracking/scripts/canonical_motion_bank_gate.py",
        ),
        "mujoco_motion_player": (
            "MuJoCo motion player",
            Path(__file__).resolve().parents[1]
            / "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py",
        ),
        "canonical_mujoco_dynamics_gate": (
            "canonical MuJoCo dynamics gate",
            Path(__file__).resolve().parents[1]
            / "hope_training/whole_body_tracking/scripts/canonical_mujoco_dynamics_gate.py",
        ),
    }
    for key, (label, local_tool) in tool_paths.items():
        expected_keys = (
            ("path", "sha256", "report_schema_version")
            if key == "canonical_mujoco_dynamics_gate"
            else ("path", "sha256")
        )
        receipt = _exact_keys(
            tools.get(key), expected_keys, f"{label} tool binding"
        )
        if (
            not local_tool.is_file()
            or Path(receipt["path"]).name != local_tool.name
            or receipt.get("sha256") != _sha256_file(local_tool)
            or (
                key == "canonical_mujoco_dynamics_gate"
                and receipt["report_schema_version"] != 1
            )
        ):
            raise CertificationError(
                f"canonical verifier does not bind the exact local {label} source"
            )
    contracts = _exact_keys(
        report["contracts"], _BANK_CONTRACT_KEYS, "canonical verifier contracts"
    )
    if (
        contracts.get("grounded_trace_status") != grounded_trace_status
        or not isinstance(contracts.get("grounded_inverse_dynamics"), str)
        or "incomplete" not in contracts["grounded_inverse_dynamics"].lower()
    ):
        raise CertificationError(
            "canonical verifier grounded contract is not formally fail-closed"
        )
    matrix = _exact_keys(
        contracts["matrix"],
        ("motion_ids", "scopes", "count"),
        "canonical verifier contracts.matrix",
    )
    if (
        not isinstance(matrix["motion_ids"], list)
        or not matrix["motion_ids"]
        or matrix["scopes"] != list(SCOPES)
        or type(matrix["count"]) is not int
        or matrix["count"] < 2
        or contracts["shared_ready"] is not True
        or contracts["six_endpoint_velocity_classes_exact_zero"] is not True
        or contracts["contact_opportunity_is_marker_only"] is not True
        or contracts["acceleration_allowed_through_window_end"] is not True
        or contracts["nonnegative_scalar_acceleration_through_window_end"]
        is not True
        or contracts["adv2c3_role"] != "comparator_only_not_default"
    ):
        raise CertificationError("canonical verifier contracts are malformed")
    aggregate = _exact_keys(
        report["aggregate"], _BANK_AGGREGATE_KEYS, "canonical verifier aggregate"
    )
    clips = _sequence(report["clips"], "canonical verifier clips")
    if not clips:
        raise CertificationError("canonical verifier clips must be non-empty")
    for index, clip in enumerate(clips):
        checked_clip = _exact_keys(
            clip, _BANK_CLIP_KEYS, f"canonical verifier clips[{index}]"
        )
        _nonempty(checked_clip["motion_id"], f"canonical verifier clip {index} motion_id")
        if checked_clip["scope"] not in SCOPES:
            raise CertificationError("canonical verifier clip scope is invalid")
    if matrix["count"] != len(clips):
        raise CertificationError("canonical verifier matrix/clip count contradicts")
    count_keys = tuple(
        key
        for key in _BANK_AGGREGATE_KEYS
        if key == "clip_count" or key.endswith("_count")
    )
    if (
        aggregate["clip_count"] != len(clips)
        or any(type(aggregate[key]) is not int or aggregate[key] < 0 for key in count_keys)
    ):
        raise CertificationError("canonical verifier aggregate counts are malformed")
    return {
        "candidate_integrity_pass": bool(report["candidate_integrity_pass"]),
        "bank_gate_pass": False,
    }


def _playback_state_at(
    report: Mapping[str, Any],
    time_s: float,
    fps: float,
    frames: int,
    output_frame: int,
) -> Mapping[str, Any]:
    report = _exact_keys(report, _PLAYBACK_REPORT_KEYS, "MuJoCo playback report")
    if report.get("verdict") != "PASS":
        raise CertificationError("MuJoCo player report must have verdict PASS")
    artifacts = _exact_keys(
        report.get("artifacts"),
        (
            "motion_sha256",
            "mjcf_path",
            "mjcf_sha256",
            "racket_reference_path",
            "racket_reference_sha256",
        ),
        "playback.artifacts",
    )
    _nonempty(artifacts["mjcf_path"], "playback.artifacts.mjcf_path")
    _digest(artifacts["motion_sha256"], "playback motion SHA-256")
    _digest(artifacts["mjcf_sha256"], "playback MJCF SHA-256")
    if (
        artifacts["racket_reference_path"] is not None
        or artifacts["racket_reference_sha256"] is not None
    ):
        raise CertificationError(
            "playback external racket reference must remain disabled"
        )
    contract = _exact_keys(
        report.get("contract"),
        (
            "schema",
            "joint_columns",
            "body_columns",
            "joint_order",
            "body_order",
            "joint_mapping",
            "body_mapping",
            "racket_site",
            "racket_site_body",
            "racket_site_local_position_m",
            "racket_normal_convention",
        ),
        "playback.contract",
    )
    boundary = _exact_keys(
        report.get("evidence_boundary"),
        (
            "level",
            "mj_forward_calls",
            "mj_step_calls",
            "dynamic_certificate",
            "training_certificate",
            "deployment_certificate",
            "hardware_certificate",
            "real_robot_certificate",
            "racket_velocity_source",
            "statement",
        ),
        "playback.evidence_boundary",
    )
    if (
        contract.get("schema") != "exact schema-2 11/14 fields"
        or contract.get("joint_columns") != 31
        or contract.get("body_columns") != 32
        or not isinstance(contract.get("joint_order"), list)
        or len(contract["joint_order"]) != 31
        or len(set(contract["joint_order"])) != 31
        or not isinstance(contract.get("body_order"), list)
        or tuple(contract["body_order"])
        != tuple(
            line.strip()
            for line in (
                Path(__file__).resolve().parents[1]
                / "configs/a3_runtime_body_order.txt"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        or contract.get("joint_mapping") != "name_to_mjcf_qpos_address"
        or contract.get("body_mapping") != "name_to_mjcf_body_id"
        or contract.get("racket_site") != RACKET_SITE_NAME
        or contract.get("racket_site_body") != RACKET_SITE_BODY
        or tuple(contract.get("racket_site_local_position_m", ()))
        != RACKET_SITE_OFFSET_WRIST_M
        or boundary.get("racket_velocity_source") != RACKET_VELOCITY_SOURCE
    ):
        raise CertificationError(
            "playback does not bind the physical blade site and Jacobian point velocity contract"
        )
    if (
        boundary.get("level") != "kinematic_playback_only"
        or boundary.get("mj_forward_calls") != 2 * frames
        or boundary.get("mj_step_calls") != 0
        or boundary.get("dynamic_certificate") is not False
        or boundary.get("training_certificate") is not False
        or boundary.get("deployment_certificate") is not False
        or boundary.get("hardware_certificate") is not False
        or boundary.get("real_robot_certificate") is not False
        or not isinstance(boundary.get("statement"), str)
        or not boundary["statement"]
        or _mapping(report.get("authorization"), "playback.authorization")
        != {"training": False, "deployment": False, "hardware": False}
    ):
        raise CertificationError("playback evidence boundary/authorization is malformed")
    motion = _exact_keys(
        report["motion"],
        (
            "path",
            "frames",
            "fps",
            "duration_s",
            "migration_provenance",
            "body_lin_vel_point",
        ),
        "playback.motion",
    )
    if (
        not isinstance(motion["path"], str)
        or not motion["path"]
        or motion["frames"] != frames
        or not _same_float(_finite(motion["fps"], "playback.motion.fps"), fps)
        or not _same_float(
            _finite(motion["duration_s"], "playback.motion.duration_s"),
            (frames - 1) / fps,
        )
        or type(motion["migration_provenance"]) is not bool
        or motion["body_lin_vel_point"] != "center_of_mass"
    ):
        raise CertificationError("playback motion identity/timing is malformed")
    gates = _exact_keys(
        report.get("gates"),
        (
            "position",
            "orientation",
            "racket_site_position_vs_schema",
            "racket_site_normal_vs_schema",
            "racket_site_linear_velocity_vs_schema",
            "racket_site_angular_velocity_vs_schema",
            "racket_site_jacobian_vs_object_velocity",
            "table_contact",
            "racket_external_reference",
        ),
        "playback.gates",
    )
    required_pass_gates = (
        "position",
        "orientation",
        "racket_site_position_vs_schema",
        "racket_site_normal_vs_schema",
        "racket_site_linear_velocity_vs_schema",
        "racket_site_angular_velocity_vs_schema",
        "racket_site_jacobian_vs_object_velocity",
    )
    if any(
        _mapping(gates.get(name), f"playback gate {name}").get("pass") is not True
        for name in required_pass_gates
    ):
        raise CertificationError("playback has a failed FK/site/Jacobian gate")
    expected_gate_keys = {
        "position": {
            "pass",
            "threshold_m",
            "max_error_m",
            "worst_frame",
            "worst_body",
        },
        "orientation": {
            "pass",
            "threshold_rad",
            "max_error_rad",
            "worst_frame",
            "worst_body",
        },
        "racket_site_position_vs_schema": {
            "pass",
            "threshold",
            "max_error_m",
            "worst_frame",
        },
        "racket_site_normal_vs_schema": {
            "pass",
            "threshold",
            "max_error_rad",
            "worst_frame",
        },
        "racket_site_linear_velocity_vs_schema": {
            "pass",
            "threshold",
            "max_error_m_s",
            "worst_frame",
        },
        "racket_site_angular_velocity_vs_schema": {
            "pass",
            "threshold",
            "max_error_rad_s",
            "worst_frame",
        },
        "racket_site_jacobian_vs_object_velocity": {
            "pass",
            "threshold_max_abs",
            "linear_max_abs_error",
            "linear_worst_frame",
            "angular_max_abs_error",
            "angular_worst_frame",
            "root_twist_max_abs_error",
        },
    }
    for gate_name, keys in expected_gate_keys.items():
        _exact_keys(gates[gate_name], keys, f"playback gate {gate_name}")
    tolerance_fields = (
        ("position", "threshold_m", 1.0e-4),
        ("orientation", "threshold_rad", 1.0e-4),
        ("racket_site_position_vs_schema", "threshold", 1.0e-4),
        ("racket_site_normal_vs_schema", "threshold", 1.0e-4),
        ("racket_site_linear_velocity_vs_schema", "threshold", 1.0e-3),
        ("racket_site_angular_velocity_vs_schema", "threshold", 1.0e-3),
        (
            "racket_site_jacobian_vs_object_velocity",
            "threshold_max_abs",
            1.0e-9,
        ),
    )
    for gate_name, threshold_key, maximum in tolerance_fields:
        threshold = _finite(
            _mapping(gates[gate_name], f"playback gate {gate_name}").get(
                threshold_key
            ),
            f"playback gate {gate_name}.{threshold_key}",
        )
        if threshold < 0.0 or threshold > maximum:
            raise CertificationError(
                f"playback gate {gate_name} uses an oversized tolerance"
            )
    table_gate = _exact_keys(
        gates.get("table_contact"),
        (
            "enabled",
            "pass",
            "obstacle_names",
            "isaac_equivalent_obstacles",
            "table_pose",
            "augmented_mjcf_sha256",
            "strikes_table",
            "contact_frames",
            "max_penetration_m",
            "worst",
            "per_obstacle",
        ),
        "playback table contact",
    )
    if (
        table_gate.get("enabled") is not True
        or table_gate.get("pass") is not True
        or not isinstance(table_gate.get("augmented_mjcf_sha256"), str)
        or len(table_gate["augmented_mjcf_sha256"]) != 64
    ):
        raise CertificationError("playback must be generated with the table enabled")
    external_gate = _exact_keys(
        gates["racket_external_reference"],
        ("enabled", "pass", "path", "max_errors"),
        "playback racket external reference",
    )
    if (
        external_gate.get("enabled") is not False
        or external_gate.get("pass") is not True
        or external_gate.get("path") is not None
        or external_gate.get("max_errors") is not None
    ):
        raise CertificationError("playback external racket reference boundary changed")
    jacobian = _mapping(
        gates.get("racket_site_jacobian_vs_object_velocity"),
        "playback racket Jacobian gate",
    )
    if jacobian.get("pass") is not True:
        raise CertificationError("playback racket Jacobian cross-check did not pass")
    racket = _exact_keys(
        report.get("racket"),
        ("array_receipts", "trajectory_sha256", "peaks", "per_frame"),
        "playback.racket",
    )
    peaks = _exact_keys(
        racket["peaks"],
        (
            "site_position_norm_max_m",
            "site_normal_norm_error_max",
            "site_linear_speed_max_m_s",
            "site_linear_speed_peak_frame",
            "site_angular_speed_max_rad_s",
            "site_angular_speed_peak_frame",
        ),
        "playback.racket.peaks",
    )
    for key in (
        "site_position_norm_max_m",
        "site_normal_norm_error_max",
        "site_linear_speed_max_m_s",
        "site_angular_speed_max_rad_s",
    ):
        if _finite(peaks[key], f"playback racket peak {key}") < 0.0:
            raise CertificationError("playback racket peaks must be non-negative")
    for key in (
        "site_linear_speed_peak_frame",
        "site_angular_speed_peak_frame",
    ):
        frame = _integer(peaks[key], f"playback racket peak {key}")
        if frame >= frames:
            raise CertificationError("playback racket peak frame lies outside motion")
    per_body = _mapping(report["per_body_max"], "playback.per_body_max")
    if frozenset(per_body) != frozenset(contract["body_order"]):
        raise CertificationError("playback per-body receipt names differ from contract")
    for name, raw in per_body.items():
        row = _exact_keys(
            raw, ("position_m", "orientation_rad"), f"playback per-body {name}"
        )
        if (
            _finite(row["position_m"], f"playback {name} position") < 0.0
            or _finite(row["orientation_rad"], f"playback {name} orientation")
            < 0.0
        ):
            raise CertificationError("playback per-body errors must be non-negative")
    per_frame = _sequence(
        racket.get("per_frame"),
        "playback.racket.per_frame",
    )
    if len(per_frame) != frames:
        raise CertificationError("playback racket trajectory frame count differs from motion")
    trajectory_arrays: Dict[str, np.ndarray] = {}
    vector_keys = (
        ("site_pos_w", "site_pos_w_m"),
        ("site_normal_w", "site_local_plus_y_normal_w"),
        ("site_lin_vel_w", "site_lin_vel_w_m_s"),
        ("site_ang_vel_w", "site_ang_vel_w_rad_s"),
    )
    for receipt_name, row_key in vector_keys:
        values = np.asarray(
            [
                _exact_keys(
                    per_frame[index],
                    (
                        "frame",
                        "time_s",
                        "site_pos_w_m",
                        "site_local_plus_y_normal_w",
                        "site_lin_vel_w_m_s",
                        "site_ang_vel_w_rad_s",
                    ),
                    f"playback frame {index}",
                ).get(row_key)
                for index in range(frames)
            ],
            dtype="<f8",
        )
        if values.shape != (frames, 3) or not np.isfinite(values).all():
            raise CertificationError(f"playback {row_key} trajectory is malformed")
        trajectory_arrays[receipt_name] = values
    receipts = _mapping(racket.get("array_receipts"), "playback racket array receipts")
    if frozenset(receipts) != frozenset(name for name, _key in vector_keys):
        raise CertificationError("playback racket array receipt set changed")

    def array_digest(array: np.ndarray) -> str:
        value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest = hashlib.sha256()
        digest.update(b"numpy-array-v1\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    combined = hashlib.sha256()
    combined.update(b"right-racket-trajectory-v1\0")
    for name, values in trajectory_arrays.items():
        digest = array_digest(values)
        receipt = _mapping(receipts.get(name), f"playback receipt {name}")
        if (
            receipt.get("sha256") != digest
            or receipt.get("dtype") != "<f8"
            or receipt.get("shape") != [frames, 3]
        ):
            raise CertificationError(f"playback trajectory receipt {name} contradicts rows")
        combined.update(name.encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
    if (
        racket.get("trajectory_sha256")
        != combined.hexdigest()
    ):
        raise CertificationError("playback combined racket trajectory digest contradicts rows")
    fractional = time_s * fps
    lo = int(math.floor(fractional))
    hi = int(math.ceil(fractional))
    if lo < 0 or hi >= frames:
        raise CertificationError(
            "compiler source anchor lies outside the compiled motion"
        )
    expected_output_frame = int(np.clip(np.rint(fractional), 0, frames - 1))
    if output_frame != expected_output_frame:
        raise CertificationError(
            "compiler source-anchor output_frame is not the nearest runtime observation tick"
        )
    alpha = fractional - lo

    def vector(frame: int, key: str) -> np.ndarray:
        row = _mapping(per_frame[frame], f"playback frame {frame}")
        value = np.asarray(row.get(key), np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise CertificationError(f"playback frame {frame} {key} is not a finite vec3")
        if row.get("frame") != frame:
            raise CertificationError("playback per-frame indices are not exact and ordered")
        if not _same_float(
            _finite(row.get("time_s"), f"playback frame {frame} time_s"),
            frame / fps,
        ):
            raise CertificationError("playback per-frame timestamps are not exact")
        return value

    v0 = vector(lo, "site_lin_vel_w_m_s")
    v1 = vector(hi, "site_lin_vel_w_m_s")
    # Runtime consumes the nearest 50 Hz observation tick, not a synthetic
    # wrist-COM interpolation.  The bracketing interpolation is retained only
    # as a diagnostic; the gate below uses this exact site/Jacobian tick.
    position = vector(output_frame, "site_pos_w_m")
    velocity = vector(output_frame, "site_lin_vel_w_m_s")
    normal = vector(output_frame, "site_local_plus_y_normal_w")
    interpolated_velocity = (1.0 - alpha) * v0 + alpha * v1
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise CertificationError("interpolated racket normal is zero")
    normal /= norm
    bracket_speeds = (float(np.linalg.norm(v0)), float(np.linalg.norm(v1)))
    return {
        "artifacts": artifacts,
        "position_w_m": position,
        "velocity_w_m_s": velocity,
        "normal_w": normal,
        "linear_speed_m_s": float(np.linalg.norm(velocity)),
        "diagnostic_interpolated_speed_m_s": float(
            np.linalg.norm(interpolated_velocity)
        ),
        "bracket_speed_min_m_s": min(bracket_speeds),
        "bracket_speed_max_m_s": max(bracket_speeds),
        "bracket_frames": [lo, hi],
        "fractional_frame": fractional,
        "runtime_output_frame": output_frame,
        "runtime_time_s": output_frame / fps,
        "includes_omega_cross_offset": True,
        "wrist_com_speed_used": False,
    }


def _validate_collision_report(
    report: Mapping[str, Any],
    *,
    action_id: str,
    scope: str,
    station_center_shift_xy_m: Tuple[float, float],
    motion_sha: str,
    mjcf_sha: str,
    urdf_sha: str,
    compiled_signature: str,
    frames: int,
    source_fps: float,
    minimum_hz: float,
    minimum_clearance_m: float,
) -> Mapping[str, Any]:
    report = _exact_keys(report, _COLLISION_REPORT_KEYS, "collision report")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("report_kind") != COLLISION_REPORT_KIND
        or report.get("action_id") != action_id
        or report.get("scope") != scope
        or STATION_CENTER_SHIFT_CANDIDATES_XY_M[
            _station_index(
                report.get("station_center_shift_xy_m"),
                "collision station center shift",
            )
        ]
        != station_center_shift_xy_m
    ):
        raise CertificationError("collision report identity/scope/station shift mismatch")
    artifacts = _exact_keys(
        report.get("artifacts"),
        ("motion", "mjcf", "urdf", "compiled_model_signature_sha256", "tool"),
        "collision.artifacts",
    )
    for name in ("motion", "mjcf", "urdf"):
        receipt = _exact_keys(
            artifacts[name], ("path", "sha256"), f"collision {name}"
        )
        _nonempty(receipt["path"], f"collision {name}.path")
        _digest(receipt["sha256"], f"collision {name}.sha256")
    if (
        _mapping(artifacts.get("motion"), "collision motion").get("sha256")
        != motion_sha
        or _mapping(artifacts.get("mjcf"), "collision mjcf").get("sha256")
        != mjcf_sha
        or _mapping(artifacts.get("urdf"), "collision urdf").get("sha256")
        != urdf_sha
        or artifacts.get("compiled_model_signature_sha256")
        != compiled_signature
    ):
        raise CertificationError("collision report artifact SHA differs from plan")
    tool = _exact_keys(
        artifacts.get("tool"), ("path", "sha256"), "collision tool binding"
    )
    if (
        Path(_nonempty(tool["path"], "collision tool path")).name
        != Path(__file__).name
        or tool.get("sha256") != _sha256_file(Path(__file__).resolve())
    ):
        raise CertificationError(
            "collision report was not produced by this exact certification source"
        )
    sampling = _exact_keys(
        report.get("sampling"),
        (
            "source_fps",
            "substeps_per_source_interval",
            "sample_hz",
            "sample_count",
            "entire_cycle",
            "interpolation",
            "mj_forward_calls",
            "mj_step_calls",
        ),
        "collision.sampling",
    )
    sample_hz = _finite(sampling.get("sample_hz"), "collision sample_hz")
    substeps = _integer(
        sampling.get("substeps_per_source_interval"),
        "collision substeps_per_source_interval",
        1,
    )
    expected_count = (frames - 1) * substeps + 1
    if (
        sampling.get("entire_cycle") is not True
        or not _same_float(
            _finite(sampling.get("source_fps"), "collision source_fps"),
            source_fps,
        )
        or not _same_float(sample_hz, source_fps * substeps)
        or sampling.get("sample_count") != expected_count
        or sampling.get("interpolation")
        != "root_xyz_and_joint_linear_plus_shortest_arc_root_quaternion_slerp"
        or sampling.get("mj_forward_calls") != 2 * expected_count
        or sampling.get("mj_step_calls") != 0
    ):
        raise CertificationError("collision report did not scan the entire cycle")
    model = _exact_keys(
        report.get("model"),
        (
            "robot_collision_geom_count",
            "racket_collision_geoms_included",
            "obstacle_names",
            "table_legs_present",
        ),
        "collision.model",
    )
    if (
        _integer(model.get("robot_collision_geom_count"), "robot collision geom count", 1)
        < len(RACKET_COLLISION_GEOMS)
        or tuple(model.get("racket_collision_geoms_included", ()))
        != RACKET_COLLISION_GEOMS
        or tuple(model.get("obstacle_names", ())) != OBSTACLES
        or model.get("table_legs_present") is not False
    ):
        raise CertificationError("collision report omitted racket or table/net geometry")
    checks = _exact_keys(
        report.get("checks"),
        (
            "self_collision",
            "foot_ground_penetration",
            "nonfoot_ground_collision",
            "table_top_collision",
            "net_collision",
            "net_post_collision",
            "aggregate",
        ),
        "collision.checks",
    )
    expected_checks = (
        "self_collision",
        "foot_ground_penetration",
        "nonfoot_ground_collision",
        "table_top_collision",
        "net_collision",
        "net_post_collision",
        "aggregate",
    )
    if frozenset(checks) != frozenset(expected_checks):
        raise CertificationError("collision report check set changed")
    component_rows: Dict[str, Mapping[str, Any]] = {}
    for name in expected_checks[:-1]:
        component = _exact_keys(
            checks[name],
            (
                "pass",
                "violation_sample_count",
                "violation_contact_count",
                "maximum_penetration_m",
                "tolerance_m",
            ),
            f"collision {name}",
        )
        violation_samples = _integer(
            component["violation_sample_count"],
            f"collision {name}.violation_sample_count",
        )
        violation_contacts = _integer(
            component["violation_contact_count"],
            f"collision {name}.violation_contact_count",
        )
        maximum_penetration = _finite(
            component["maximum_penetration_m"],
            f"collision {name}.maximum_penetration_m",
        )
        tolerance = _finite(
            component["tolerance_m"], f"collision {name}.tolerance_m"
        )
        if (
            type(component["pass"]) is not bool
            or component["pass"] is not (violation_contacts == 0)
            or violation_samples > violation_contacts
            or maximum_penetration < 0.0
            or tolerance < 0.0
        ):
            raise CertificationError(
                f"collision {name} evidence contradicts its pass flag"
            )
        component_rows[name] = component
    component_pass = all(
        component_rows[name]["pass"] is True for name in expected_checks[:-1]
    )
    aggregate = _exact_keys(
        checks["aggregate"], ("pass",), "collision aggregate"
    )
    expected_verdict = "PASS" if component_pass else "FAIL"
    if (
        aggregate.get("pass") is not component_pass
        or report.get("verdict") != expected_verdict
    ):
        raise CertificationError("collision aggregate contradicts component gates")
    authorization = _mapping(report.get("authorization"), "collision.authorization")
    if authorization != {
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }:
        raise CertificationError("collision report authorization boundary changed")
    clearance = _exact_keys(
        report.get("clearance"),
        ("minimum_table_net_clearance_m", "distance_query_cap_m", "minimum"),
        "collision.clearance",
    )
    minimum_observed = _finite(
        clearance.get("minimum_table_net_clearance_m"),
        "minimum table/net clearance",
    )
    if _finite(
        clearance["distance_query_cap_m"], "collision distance query cap"
    ) < minimum_observed:
        raise CertificationError("collision clearance exceeds its distance-query cap")
    minimum_row = _exact_keys(
        clearance["minimum"],
        ("sample", "time_s", "robot_geom", "obstacle", "distance_m"),
        "collision clearance minimum",
    )
    if (
        _integer(minimum_row["sample"], "collision minimum.sample")
        >= expected_count
        or _finite(minimum_row["time_s"], "collision minimum.time_s") < 0.0
        or not _nonempty(
            minimum_row["robot_geom"], "collision minimum.robot_geom"
        )
        or minimum_row["obstacle"] not in OBSTACLES
        or not _same_float(
            _finite(minimum_row["distance_m"], "collision minimum.distance_m"),
            minimum_observed,
        )
    ):
        raise CertificationError("collision minimum witness contradicts clearance")
    non_claims = _sequence(report["non_claims"], "collision.non_claims")
    if any(not isinstance(item, str) or not item for item in non_claims):
        raise CertificationError("collision non_claims must be non-empty strings")
    passed = bool(
        component_pass
        and sample_hz >= minimum_hz
        and minimum_observed >= minimum_clearance_m
    )
    return {
        "pass": passed,
        "component_pass": component_pass,
        "sample_hz": sample_hz,
        "required_sample_hz": minimum_hz,
        "minimum_table_net_clearance_m": minimum_observed,
        "required_minimum_clearance_m": minimum_clearance_m,
        "checks": checks,
    }


def _reference_return_fraction(
    *,
    state: Mapping[str, Any],
    station_center_shift_xy_m: Tuple[float, float],
    task: Mapping[str, Any],
) -> float:
    scripts_dir = Path(__file__).resolve().parents[1] / "hope_training/whole_body_tracking/scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import reference_return_gate  # type: ignore
    except Exception as exc:
        raise CertificationError(f"cannot import exact reference return gate: {exc}") from exc
    position = np.asarray(state["position_w_m"], np.float64).copy()
    position[:2] += np.asarray(station_center_shift_xy_m, np.float64)
    try:
        return float(
            reference_return_gate.score_reference_returns(
                p_contact_w=position,
                v_racket_w=state["velocity_w_m_s"],
                n_racket_w=state["normal_w"],
                vel_box=task["incoming_velocity_box_m_s"],
                spin_abs_max=float(task["spin_abs_max_rad_s"]),
                n_samples=int(task["samples"]),
                seed=int(task["seed"]),
                face_sign=float(task["face_sign"]),
                venue_yaml=str(task["venue_yaml_path"]),
                capture_radius=float(task["capture_radius_m"]),
                min_approach_speed=float(task["minimum_approach_speed_m_s"]),
            )
        )
    except Exception as exc:
        raise CertificationError(f"reference return scorer failed: {exc}") from exc


def _validate_task_distribution(raw: Any) -> Mapping[str, Any]:
    row = _exact_keys(
        raw,
        (
            "incoming_velocity_box_m_s",
            "spin_abs_max_rad_s",
            "samples",
            "seed",
            "face_sign",
            "capture_radius_m",
            "minimum_approach_speed_m_s",
            "minimum_legal_return_fraction",
        ),
        "task_distribution",
    )
    box = _sequence(row["incoming_velocity_box_m_s"], "incoming velocity box")
    if len(box) != 3:
        raise CertificationError("incoming velocity box must have x/y/z intervals")
    checked_box: List[List[float]] = []
    for axis, interval in enumerate(box):
        bounds = _sequence(interval, f"incoming velocity interval {axis}")
        if len(bounds) != 2:
            raise CertificationError("each incoming velocity interval must have two bounds")
        lo = _finite(bounds[0], f"velocity axis {axis} lower")
        hi = _finite(bounds[1], f"velocity axis {axis} upper")
        if lo > hi:
            raise CertificationError("incoming velocity interval lower exceeds upper")
        if hi - lo < REFERENCE_RETURN_MIN_AXIS_SPAN_M_S:
            raise CertificationError(
                "incoming velocity box cannot be a zero/single-point custom domain"
            )
        checked_box.append([lo, hi])
    spin = _finite(row["spin_abs_max_rad_s"], "spin_abs_max_rad_s")
    samples = _integer(row["samples"], "task_distribution.samples", 1)
    seed = _integer(row["seed"], "task_distribution.seed", 0)
    face_sign = _finite(row["face_sign"], "task_distribution.face_sign")
    if face_sign not in (-1.0, 1.0):
        raise CertificationError("face_sign must be exactly -1 or +1")
    capture = _finite(row["capture_radius_m"], "capture_radius_m")
    approach = _finite(row["minimum_approach_speed_m_s"], "minimum approach speed")
    minimum = _finite(row["minimum_legal_return_fraction"], "minimum return fraction")
    corner_speeds = [
        math.sqrt(x * x + y * y + z * z)
        for x in checked_box[0]
        for y in checked_box[1]
        for z in checked_box[2]
    ]
    minimum_speed = math.sqrt(
        sum(
            (
                0.0
                if lo <= 0.0 <= hi
                else min(abs(lo), abs(hi))
            )
            ** 2
            for lo, hi in checked_box
        )
    )
    if (
        spin <= 0.0
        or math.sqrt(3.0) * spin > VENUE_SPIN_MAGNITUDE_MAX_RAD_S
        or checked_box[0][1] >= 0.0
        or minimum_speed < VENUE_BALL_SPEED_RANGE_M_S[0]
        or max(corner_speeds) > VENUE_BALL_SPEED_RANGE_M_S[1]
        or capture <= 0.0
        or capture > REFERENCE_RETURN_MAX_CAPTURE_RADIUS_M
        or approach < REFERENCE_RETURN_MIN_APPROACH_SPEED_M_S
        or samples < REFERENCE_RETURN_MIN_SAMPLES
        or not REFERENCE_RETURN_MIN_FRACTION <= minimum <= 1.0
    ):
        raise CertificationError("task distribution thresholds leave their physical domains")
    return {
        "incoming_velocity_box_m_s": checked_box,
        "spin_abs_max_rad_s": spin,
        "samples": samples,
        "seed": seed,
        "face_sign": face_sign,
        "capture_radius_m": capture,
        "minimum_approach_speed_m_s": approach,
        "minimum_legal_return_fraction": minimum,
    }


def _validate_code_rooted_venue_profile(snapshot: Snapshot) -> Mapping[str, Any]:
    _require_code_rooted_bytes(
        snapshot,
        repository_relative_path=CODE_ROOTED_VENUE_PROFILE,
        label="venue profile",
    )
    profile = _parse_json(snapshot.data, "venue profile")
    if profile.get("schema_version") != "venue_profile_v1":
        raise CertificationError("venue profile schema_version changed")
    physics = _mapping(profile.get("physics"), "venue profile.physics")
    required_ranges = (
        "static_friction_range",
        "dynamic_friction_range",
        "restitution_range",
        "mass_distribution_params",
    )
    for key in required_ranges:
        interval = _sequence(physics.get(key), f"venue profile.physics.{key}")
        if len(interval) != 2:
            raise CertificationError(f"venue profile.physics.{key} must be a range")
        lo = _finite(interval[0], f"venue profile.physics.{key}[0]")
        hi = _finite(interval[1], f"venue profile.physics.{key}[1]")
        if lo < 0.0 or hi <= lo:
            raise CertificationError(
                f"venue profile.physics.{key} cannot be zero or single-point"
            )
    return profile


def _validate_thresholds(raw: Any) -> Mapping[str, float]:
    row = _exact_keys(
        raw,
        (
            "source_anchor_time_min_s",
            "source_anchor_time_max_s",
            "t_hit_reference_s",
            "t_cycle_min_s",
            "t_cycle_max_s",
            "blade_site_speed_min_m_s",
            "blade_site_speed_max_m_s",
            "shared_ready_pose_tolerance",
            "dense_collision_min_hz",
            "minimum_table_net_clearance_m",
        ),
        "thresholds",
    )
    checked = {key: _finite(value, f"thresholds.{key}") for key, value in row.items()}
    source_min = checked["source_anchor_time_min_s"]
    source_max = checked["source_anchor_time_max_s"]
    cycle_min = checked["t_cycle_min_s"]
    cycle_max = checked["t_cycle_max_s"]
    speed_min = checked["blade_site_speed_min_m_s"]
    speed_max = checked["blade_site_speed_max_m_s"]
    if (
        not SOURCE_ANCHOR_TIME_LIMITS_S[0]
        <= source_min
        <= source_max
        <= SOURCE_ANCHOR_TIME_LIMITS_S[1]
        or checked["t_hit_reference_s"] != T_HIT_REFERENCE_S
        or source_max - source_min > SOURCE_ANCHOR_MAX_WINDOW_S
        or not T_CYCLE_LIMITS_S[0]
        <= cycle_min
        <= cycle_max
        <= T_CYCLE_LIMITS_S[1]
        or cycle_max - cycle_min > T_CYCLE_MAX_WINDOW_S
        or not BLADE_SITE_SPEED_LIMITS_M_S[0]
        <= speed_min
        <= speed_max
        <= BLADE_SITE_SPEED_LIMITS_M_S[1]
        or speed_max - speed_min > BLADE_SITE_SPEED_MAX_WINDOW_M_S
        or not 0.0
        <= checked["shared_ready_pose_tolerance"]
        <= SHARED_READY_POSE_TOLERANCE_MAX
        or checked["dense_collision_min_hz"] < DENSE_COLLISION_MIN_HZ
        or checked["minimum_table_net_clearance_m"]
        < MINIMUM_TABLE_NET_CLEARANCE_M
    ):
        raise CertificationError(
            "certification thresholds leave the code-reviewed admissibility envelope"
        )
    return checked


def certify_plan(plan: Mapping[str, Any], *, base_dir: Path) -> Mapping[str, Any]:
    """Verify one already content-bound certification plan."""

    plan = _exact_keys(
        plan,
        (
            "schema_version",
            "plan_kind",
            "action_id",
            "bindings",
            "required_scopes",
            "station_center_shift_candidates_xy_m",
            "selected_station_center_shift_xy_m",
            "station_selection_approval",
            "behavior_contact_evidence",
            "thresholds",
            "task_distribution",
            "scopes",
            "authorization_intent",
        ),
        "certification plan",
    )
    if plan["schema_version"] != SCHEMA_VERSION or plan["plan_kind"] != PLAN_KIND:
        raise CertificationError("certification plan schema/kind mismatch")
    action_id = _nonempty(plan["action_id"], "action_id")
    if tuple(plan["required_scopes"]) != SCOPES:
        raise CertificationError(f"required_scopes must be exactly {list(SCOPES)}")
    candidate_rows = _sequence(
        plan["station_center_shift_candidates_xy_m"], "station center candidates"
    )
    candidates = tuple(
        STATION_CENTER_SHIFT_CANDIDATES_XY_M[
            _station_index(value, f"station center candidate {index}")
        ]
        for index, value in enumerate(candidate_rows)
    )
    if candidates != STATION_CENTER_SHIFT_CANDIDATES_XY_M:
        raise CertificationError(
            "station center candidates must be exactly "
            f"{[list(row) for row in STATION_CENTER_SHIFT_CANDIDATES_XY_M]}"
        )
    selected: Optional[Tuple[float, float]]
    if plan["selected_station_center_shift_xy_m"] is None:
        selected = None
    else:
        selected = STATION_CENTER_SHIFT_CANDIDATES_XY_M[
            _station_index(
                plan["selected_station_center_shift_xy_m"],
                "selected station center shift",
            )
        ]
    behavior_contact_evidence = (
        _load_trusted_behavior_contact_evidence(
            plan["behavior_contact_evidence"],
            base_dir=base_dir,
            action_id=action_id,
        )
    )
    comparison_input_sha256 = _station_comparison_input_sha256(plan)
    station_selection_approval = (
        _load_trusted_station_selection_approval(
            plan["station_selection_approval"],
            base_dir=base_dir,
            action_id=action_id,
            selected=selected,
            comparison_input_sha256=comparison_input_sha256,
        )
    )
    if plan["authorization_intent"] != (
        "task_first_training_only_no_deployment_no_hardware"
    ):
        raise CertificationError("authorization intent must remain training-only")
    thresholds = _validate_thresholds(plan["thresholds"])
    task = _validate_task_distribution(plan["task_distribution"])

    binding_rows = _exact_keys(
        plan["bindings"],
        (
            "source",
            "recipe",
            "build_manifest",
            "canonical_verifier_report",
            "mjcf",
            "venue_yaml",
            "venue_profile",
        ),
        "plan.bindings",
    )
    snapshots = {
        name: read_bound_file(binding_rows[name], base_dir, name)
        for name in binding_rows
    }
    _require_code_rooted_bytes(
        snapshots["venue_yaml"],
        repository_relative_path=CODE_ROOTED_BALL_PHYSICS,
        label="ball physics",
    )
    _validate_code_rooted_venue_profile(snapshots["venue_profile"])
    task = {**dict(task), "venue_yaml_path": str(snapshots["venue_yaml"].path)}
    recipe = _parse_json(snapshots["recipe"].data, "recipe")
    manifest = _parse_json(snapshots["build_manifest"].data, "build manifest")
    bank = _parse_json(
        snapshots["canonical_verifier_report"].data, "canonical verifier report"
    )
    ready_row = _mapping(recipe.get("canonical_ready"), "recipe.canonical_ready")
    ready_snapshot = read_recipe_bound_file(
        {
            "path": ready_row.get("path"),
            "sha256": ready_row.get("sha256"),
        },
        plan_base_dir=base_dir,
        label="canonical ready",
    )
    snapshots["canonical_ready"] = ready_snapshot
    canonical_ready = _canonical_ready_state(ready_snapshot)
    ready_fk_row = _mapping(
        recipe.get("canonical_ready_fk"), "recipe.canonical_ready_fk"
    )
    ready_fk_snapshot = read_recipe_bound_file(
        {
            "path": ready_fk_row.get("path"),
            "sha256": ready_fk_row.get("sha256"),
        },
        plan_base_dir=base_dir,
        label="canonical ready-FK",
    )
    snapshots["canonical_ready_fk"] = ready_fk_snapshot
    canonical_ready_fk = _canonical_ready_fk_state(
        ready_fk_snapshot,
        canonical_ready_sha256=ready_snapshot.sha256,
        canonical_ready=canonical_ready,
    )
    marker_ref = _mapping(recipe.get("marker_authority"), "recipe.marker_authority")
    marker_snapshot = read_recipe_bound_file(
        {
            "path": marker_ref.get("path"),
            "sha256": marker_ref.get("sha256"),
        },
        plan_base_dir=base_dir,
        label="canonical marker authority",
    )
    snapshots["marker_authority"] = marker_snapshot
    marker_row = _load_marker_row(marker_snapshot, action_id)
    source_specs = [
        row
        for row in _sequence(recipe.get("motion_specs"), "recipe.motion_specs")
        if isinstance(row, dict) and row.get("motion_id") == action_id
    ]
    if len(source_specs) != 1:
        raise CertificationError("recipe must contain exactly one selected action source")
    source_spec = source_specs[0]
    if source_spec.get("source_sha256") != snapshots["source"].sha256:
        raise CertificationError("recipe action source SHA differs from plan source")
    if (
        marker_row.bound_recipe_source_sha256 != snapshots["source"].sha256
        or marker_row.post_retime_behavior_gate_status
        != "PENDING_POST_RETIME_BEHAVIOR_RESCAN"
    ):
        raise CertificationError(
            "marker authority source binding/status is not the expected pending-v3 contract"
        )
    model_contract = _mapping(recipe.get("model_contract"), "recipe.model_contract")
    if model_contract.get("mjcf_sha256") != snapshots["mjcf"].sha256:
        raise CertificationError("recipe MJCF SHA differs from plan")
    urdf_snapshot = read_recipe_bound_file(
        {
            "path": model_contract.get("urdf_path"),
            "sha256": model_contract.get("urdf_sha256"),
        },
        plan_base_dir=base_dir,
        label="vendor URDF",
    )
    snapshots["urdf"] = urdf_snapshot
    manifest_recipe = _mapping(manifest.get("recipe"), "build_manifest.recipe")
    if manifest_recipe.get("sha256") != snapshots["recipe"].sha256:
        raise CertificationError("build manifest recipe SHA differs from plan")
    bank_contract = _validate_bank_contract(
        bank,
        manifest_sha256=snapshots["build_manifest"].sha256,
        recipe_sha256=snapshots["recipe"].sha256,
        mjcf_sha256=snapshots["mjcf"].sha256,
        urdf_sha256=urdf_snapshot.sha256,
        ready_sha256=ready_snapshot.sha256,
    )
    candidate_integrity = bank_contract["candidate_integrity_pass"]
    bank_gate_pass = bank_contract["bank_gate_pass"]
    bank_plant = _mapping(
        _mapping(bank.get("bound_inputs"), "canonical verifier bound_inputs").get(
            "plant"
        ),
        "canonical verifier plant",
    )
    compiled_signature = _digest(
        bank_plant.get("compiled_signature_sha256"),
        "canonical verifier compiled signature",
    )
    if (
        bank_plant.get("mjcf_sha256") != snapshots["mjcf"].sha256
        or bank_plant.get("urdf_sha256") != urdf_snapshot.sha256
        or bank_plant.get("identity_bound") is not True
    ):
        raise CertificationError("canonical verifier plant identity differs from recipe")

    scope_plans = _exact_keys(plan["scopes"], SCOPES, "plan.scopes")
    scope_results: Dict[str, Any] = {}
    cross_scope_ready: Dict[str, Mapping[str, np.ndarray]] = {}
    scope_motion_shas: Dict[str, str] = {}
    for scope in SCOPES:
        scope_plan = _exact_keys(
            scope_plans[scope],
            ("motion", "playback_report", "collision_reports"),
            f"plan.scopes.{scope}",
        )
        motion_snapshot = read_bound_file(
            scope_plan["motion"], base_dir, f"{scope} compiled motion"
        )
        playback_snapshot = read_bound_file(
            scope_plan["playback_report"], base_dir, f"{scope} playback report"
        )
        motion = _load_npz(motion_snapshot)
        if motion_snapshot.sha256 in scope_motion_shas.values():
            raise CertificationError("upper/full compiled motions must be distinct bytes")
        scope_motion_shas[scope] = motion_snapshot.sha256
        motion_receipt = _motion_receipt(
            motion, pose_tolerance=thresholds["shared_ready_pose_tolerance"]
        )
        ready_truth = _motion_ready_truth_gate(
            motion,
            canonical_ready,
            canonical_ready_fk,
            tolerance=thresholds["shared_ready_pose_tolerance"],
        )
        cross_scope_ready[scope] = {
            "joint_pos": np.asarray(motion["joint_pos"], np.float64)[0],
            "body_pos": np.asarray(motion["body_pos_w"], np.float64)[0],
            "body_quat": np.asarray(motion["body_quat_w"], np.float64)[0],
        }
        output = _manifest_output(manifest, action_id, scope)
        if (
            output.get("output_npz_sha256") != motion_snapshot.sha256
            or output.get("filename") != motion_snapshot.path.name
        ):
            raise CertificationError(f"{scope} manifest output binding differs from motion")
        preprocessing = _mapping(
            output.get("scope_preprocessing"), f"{scope} scope_preprocessing"
        )
        if output.get("scope") != scope or not isinstance(
            preprocessing.get("algorithm"), str
        ):
            raise CertificationError(f"{scope} manifest scope preprocessing is missing")
        opportunity = _mapping(
            _mapping(output.get("search"), f"{scope} search").get(
                "contact_opportunity"
            ),
            f"{scope} contact opportunity",
        )
        authority_anchor, _anchor_basis = marker_row.contact_anchor()
        authority_window, _window_basis = marker_row.search_window()
        if (
            opportunity.get("source_anchor_frame") != authority_anchor
            or tuple(opportunity.get("source_span_inclusive", ()))
            != authority_window
            or opportunity.get("marker_only") is not True
            or opportunity.get("pose_locked") is not False
            or opportunity.get("velocity_locked") is not False
        ):
            raise CertificationError(
                f"{scope} contact opportunity does not match marker authority"
            )
        source_anchor_time = _finite(
            output.get("source_anchor_time_s"),
            f"{scope} source_anchor_time_s",
        )
        if source_anchor_time < 0.0:
            raise CertificationError(f"{scope} source_anchor_time_s must be non-negative")
        retiming = _mapping(output.get("retiming"), f"{scope} retiming")
        marker = _mapping(
            _mapping(retiming.get("markers"), f"{scope} retiming markers").get(
                "source_anchor"
            ),
            f"{scope} source_anchor marker",
        )
        if not _same_float(
            _finite(marker.get("time_s"), f"{scope} marker source anchor"),
            source_anchor_time,
        ):
            raise CertificationError(
                f"{scope} manifest source-anchor fields contradict"
            )
        output_frame = _integer(
            marker.get("output_frame"), f"{scope} marker output_frame"
        )
        fractional_frame = _finite(
            marker.get("output_fractional_frame"),
            f"{scope} marker output_fractional_frame",
        )
        if not _same_float(
            fractional_frame,
            source_anchor_time * float(motion_receipt["fps"]),
            tolerance=1.0e-9,
        ):
            raise CertificationError(
                f"{scope} marker fractional frame/time contradict"
            )
        manifest_cycle = _finite(output.get("duration_s"), f"{scope} duration_s")
        cycle = _finite(motion_receipt["duration_s"], f"{scope} motion duration")
        if not _same_float(manifest_cycle, cycle, tolerance=1.0e-9):
            raise CertificationError(f"{scope} manifest/motion t_cycle mismatch")

        bank_clip = _bank_clip(bank, action_id, scope)
        if (
            bank_clip.get("sha256") != motion_snapshot.sha256
            or bank_clip.get("mujoco_fk", {}).get("pass") is not True
        ):
            raise CertificationError(f"{scope} canonical verifier clip binding/FK failed")
        # A generic ``screen_pass`` is not a content-addressed grounded
        # collocation trace.  This diagnostic has no trusted promotion
        # certificate and therefore cannot turn it into grounded evidence.
        _mapping(
            bank_clip.get("plant_specific_dynamics"),
            f"{scope} plant-specific dynamics",
        )
        grounded = False

        playback = _parse_json(playback_snapshot.data, f"{scope} playback report")
        anchor_state = _playback_state_at(
            playback,
            source_anchor_time,
            float(motion_receipt["fps"]),
            int(motion_receipt["frames"]),
            output_frame,
        )
        behavior_measurement = (
            None
            if behavior_contact_evidence is None
            else behavior_contact_evidence["measurements"][scope]
        )
        if behavior_measurement is None:
            behavior_t_hit_s = None
            timing_hit_gate = False
            state = anchor_state
        else:
            if (
                behavior_measurement["motion_sha256"]
                != motion_snapshot.sha256
            ):
                raise CertificationError(
                    f"{scope} behavior/contact evidence binds different "
                    "motion bytes"
                )
            behavior_t_hit_s = _finite(
                behavior_measurement["t_hit_s"],
                f"{scope} trusted behavior/contact t_hit_s",
            )
            if behavior_t_hit_s > cycle:
                raise CertificationError(
                    f"{scope} trusted behavior/contact t_hit lies outside "
                    "the compiled cycle"
                )
            behavior_output_frame = int(
                np.clip(
                    np.rint(
                        behavior_t_hit_s
                        * float(motion_receipt["fps"])
                    ),
                    0,
                    int(motion_receipt["frames"]) - 1,
                )
            )
            state = _playback_state_at(
                playback,
                behavior_t_hit_s,
                float(motion_receipt["fps"]),
                int(motion_receipt["frames"]),
                behavior_output_frame,
            )
            accepted_t_hit = behavior_contact_evidence[
                "accepted_t_hit_range_s"
            ]
            timing_hit_gate = bool(
                accepted_t_hit[0]
                <= behavior_t_hit_s
                <= accepted_t_hit[1]
            )
        playback_artifacts = _mapping(state["artifacts"], f"{scope} playback artifacts")
        if (
            playback_artifacts.get("motion_sha256") != motion_snapshot.sha256
            or playback_artifacts.get("mjcf_sha256") != snapshots["mjcf"].sha256
        ):
            raise CertificationError(f"{scope} playback artifact binding differs from plan")
        speed = _finite(state["linear_speed_m_s"], f"{scope} blade speed")
        bracket_min = _finite(
            state["bracket_speed_min_m_s"], f"{scope} blade bracket min"
        )
        bracket_max = _finite(
            state["bracket_speed_max_m_s"], f"{scope} blade bracket max"
        )
        speed_gate = bool(
            thresholds["blade_site_speed_min_m_s"]
            <= speed
            <= thresholds["blade_site_speed_max_m_s"]
        )

        collision_bindings = _sequence(
            scope_plan["collision_reports"], f"{scope} collision reports"
        )
        if len(collision_bindings) != len(
            STATION_CENTER_SHIFT_CANDIDATES_XY_M
        ):
            raise CertificationError(
                f"{scope} must bind exactly three station collision reports"
            )
        collision_by_shift: Dict[Tuple[float, float], Mapping[str, Any]] = {}
        collision_binding_receipts: Dict[Tuple[float, float], Mapping[str, str]] = {}
        for index, binding in enumerate(collision_bindings):
            collision_snapshot = read_bound_file(
                binding, base_dir, f"{scope} collision report {index}"
            )
            collision_raw = _parse_json(
                collision_snapshot.data, f"{scope} collision report {index}"
            )
            shift = STATION_CENTER_SHIFT_CANDIDATES_XY_M[
                _station_index(
                    collision_raw.get("station_center_shift_xy_m"),
                    "collision station center shift",
                )
            ]
            if shift in collision_by_shift:
                raise CertificationError(f"{scope} duplicates collision shift {shift}")
            collision_by_shift[shift] = _validate_collision_report(
                collision_raw,
                action_id=action_id,
                scope=scope,
                station_center_shift_xy_m=shift,
                motion_sha=motion_snapshot.sha256,
                mjcf_sha=snapshots["mjcf"].sha256,
                urdf_sha=urdf_snapshot.sha256,
                compiled_signature=compiled_signature,
                frames=int(motion_receipt["frames"]),
                source_fps=float(motion_receipt["fps"]),
                minimum_hz=thresholds["dense_collision_min_hz"],
                minimum_clearance_m=thresholds["minimum_table_net_clearance_m"],
            )
            collision_binding_receipts[shift] = collision_snapshot.binding()
        if tuple(collision_by_shift) != STATION_CENTER_SHIFT_CANDIDATES_XY_M:
            raise CertificationError(f"{scope} collision reports omit a required station shift")

        # `source_anchor_time_s` remains a compiler marker.  It can never
        # substitute for t_hit; only the separately code-pinned,
        # action-specific behavior/contact receipt above can open this gate.
        anchor_time_gate = bool(
            thresholds["source_anchor_time_min_s"]
            <= source_anchor_time
            <= thresholds["source_anchor_time_max_s"]
            and thresholds["source_anchor_time_min_s"]
            <= anchor_state["runtime_time_s"]
            <= thresholds["source_anchor_time_max_s"]
        )
        timing_cycle_gate = (
            thresholds["t_cycle_min_s"] <= cycle <= thresholds["t_cycle_max_s"]
        )
        ready_gate = bool(
            motion_receipt["shared_ready_return"]["pass"] is True
            and ready_truth["pass"] is True
        )
        station_results: Dict[str, Any] = {}
        for shift in STATION_CENTER_SHIFT_CANDIDATES_XY_M:
            return_fraction = _finite(
                _reference_return_fraction(
                    state=state,
                    station_center_shift_xy_m=shift,
                    task=task,
                ),
                f"{scope} {shift} reference return fraction",
            )
            if not 0.0 <= return_fraction <= 1.0:
                raise CertificationError(
                    f"{scope} {shift} reference return fraction must be in [0, 1]"
                )
            gates = {
                "canonical_candidate_integrity": bool(candidate_integrity),
                "compiler_anchor_in_preregistered_range": anchor_time_gate,
                "grounded_dynamics": bool(grounded),
                "post_retime_t_hit": bool(timing_hit_gate),
                "post_retime_t_cycle": bool(timing_cycle_gate),
                "physical_blade_site_speed": bool(speed_gate),
                "dense_collision": bool(collision_by_shift[shift]["pass"]),
                "shared_ready_return": bool(ready_gate),
                "reference_returnability": bool(
                    return_fraction >= task["minimum_legal_return_fraction"]
                ),
            }
            station_results[f"{shift[0]:.2f},{shift[1]:.2f}"] = {
                "station_center_shift_xy_m": list(shift),
                "all_required_gates_pass": all(gates.values()),
                "gates": gates,
                "reference_return_fraction": return_fraction,
                "collision": collision_by_shift[shift],
                "collision_report": collision_binding_receipts[shift],
            }
        scope_results[scope] = {
            "motion": motion_snapshot.binding(),
            "playback_report": playback_snapshot.binding(),
            "timing": {
                "compiler_source_anchor_time_s": source_anchor_time,
                "compiler_anchor_nearest_tick_time_s": anchor_state[
                    "runtime_time_s"
                ],
                "compiler_anchor_output_frame": anchor_state[
                    "runtime_output_frame"
                ],
                "preregistered_compiler_anchor_range_s": [
                    thresholds["source_anchor_time_min_s"],
                    thresholds["source_anchor_time_max_s"],
                ],
                "comparison_t_hit_reference_s": T_HIT_REFERENCE_S,
                "t_hit_acceptance_authority": (
                    "code_pinned_action_specific_behavior_contact_evidence"
                ),
                "accepted_action_specific_t_hit_range_s": (
                    None
                    if behavior_contact_evidence is None
                    else behavior_contact_evidence[
                        "accepted_t_hit_range_s"
                    ]
                ),
                "post_retime_behavior_t_hit_s": behavior_t_hit_s,
                "behavior_contact_evidence_receipt": (
                    None
                    if behavior_contact_evidence is None
                    else behavior_contact_evidence["receipt"]
                ),
                "behavior_contact_evidence_artifact": (
                    None
                    if behavior_contact_evidence is None
                    else behavior_contact_evidence["evidence_artifact"]
                ),
                "t_hit_gate_result": bool(timing_hit_gate),
                "compiler_anchor_substituted_for_t_hit": False,
                "post_retime_behavior_t_hit_measured": bool(
                    behavior_measurement is not None
                ),
                "behavior_contact_output_frame": (
                    None
                    if behavior_measurement is None
                    else state["runtime_output_frame"]
                ),
                "post_retime_behavior_gate_status": (
                    marker_row.post_retime_behavior_gate_status
                ),
                "t_cycle_s": cycle,
                "t_cycle_range_s": [
                    thresholds["t_cycle_min_s"],
                    thresholds["t_cycle_max_s"],
                ],
            },
            "physical_blade_site": {
                "site_name": RACKET_SITE_NAME,
                "site_body": RACKET_SITE_BODY,
                "local_offset_m": list(RACKET_SITE_OFFSET_WRIST_M),
                "velocity_source": RACKET_VELOCITY_SOURCE,
                "includes_omega_cross_offset": True,
                "wrist_com_speed_used": False,
                "linear_speed_m_s": speed,
                "diagnostic_fractional_linear_speed_m_s": state[
                    "diagnostic_interpolated_speed_m_s"
                ],
                "bracket_speed_min_m_s": bracket_min,
                "bracket_speed_max_m_s": bracket_max,
                "bracket_frames": state["bracket_frames"],
                "fractional_frame": state["fractional_frame"],
            },
            "shared_ready_return": motion_receipt["shared_ready_return"],
            "canonical_ready_truth": ready_truth,
            "stations": station_results,
        }

    cross_scope_joint_delta = float(
        np.max(
            np.abs(
                cross_scope_ready["upper"]["joint_pos"]
                - cross_scope_ready["full"]["joint_pos"]
            )
        )
    )
    cross_scope_body_pos_delta = float(
        np.max(
            np.linalg.norm(
                cross_scope_ready["upper"]["body_pos"]
                - cross_scope_ready["full"]["body_pos"],
                axis=1,
            )
        )
    )
    cross_scope_body_quat_delta = float(
        np.max(
            _quat_angle_rad(
                cross_scope_ready["upper"]["body_quat"],
                cross_scope_ready["full"]["body_quat"],
            )
        )
    )
    cross_scope_ready_max = max(
        cross_scope_joint_delta,
        cross_scope_body_pos_delta,
        cross_scope_body_quat_delta,
    )
    cross_scope_ready_pass = (
        cross_scope_ready_max <= thresholds["shared_ready_pose_tolerance"]
    )
    if not cross_scope_ready_pass:
        for scope in SCOPES:
            for row in scope_results[scope]["stations"].values():
                row["gates"]["shared_ready_return"] = False
                row["all_required_gates_pass"] = False

    diagnostic_reference_blockers: List[str] = []
    if selected is None:
        diagnostic_reference_blockers.append(
            "station_center_shift_not_selected: comparison evidence never auto-adopts a stance"
        )
    else:
        common_pass_candidates = []
        for candidate in STATION_CENTER_SHIFT_CANDIDATES_XY_M:
            candidate_key = f"{candidate[0]:.2f},{candidate[1]:.2f}"
            common_pass_candidates.append(
                all(
                    scope_results[scope]["stations"][candidate_key]["gates"][gate]
                    is True
                    for scope in SCOPES
                    for gate in STATION_COMPARISON_GATES
                )
            )
        passing_indices = [
            index
            for index, passed in enumerate(common_pass_candidates)
            if passed
        ]
        if passing_indices:
            nearest_passing = STATION_CENTER_SHIFT_CANDIDATES_XY_M[
                passing_indices[0]
            ]
            if selected != nearest_passing:
                raise CertificationError(
                    "selected station center is not the nearest upper/full "
                    "common-pass candidate: "
                    f"selected={selected} nearest={nearest_passing}"
                )
        key = f"{selected[0]:.2f},{selected[1]:.2f}"
        for scope in SCOPES:
            gates = scope_results[scope]["stations"][key]["gates"]
            for gate in DIAGNOSTIC_REFERENCE_GATES:
                if gates[gate] is not True:
                    diagnostic_reference_blockers.append(f"{scope}/{gate}")
    if not cross_scope_ready_pass:
        diagnostic_reference_blockers.append(
            "upper/full shared-ready endpoints differ"
        )
    diagnostic_reference_checks_pass = len(diagnostic_reference_blockers) == 0

    # The composition path consumes externally generated JSON.  A public tool
    # hash proves code identity, not that the named producer actually emitted
    # the receipt.  Keep even the diagnostic smoke authorization false until a
    # single no-clobber command recomputes playback and all six collision scans
    # from the bound bytes in-process.
    diagnostic_smoke_authorized = False
    diagnostic_smoke_blockers = list(diagnostic_reference_blockers)
    diagnostic_smoke_blockers.append(
        "external_playback_and_collision_json_are_untrusted_diagnostics_not_producer_provenance"
    )
    if behavior_contact_evidence is None:
        diagnostic_smoke_blockers.append(
            "post_retime_behavior_t_hit_rescan_pending"
        )

    # The current canonical verifier has no content-addressed exact
    # collocation trace and therefore cannot prove grounded dynamics.  Keep
    # formal task-first training closed even when the reference checks pass.
    training_blockers = list(diagnostic_smoke_blockers)
    training_blockers.append(
        "grounded_collocation_trace_missing: canonical bank gate is "
        "MISSING_INCOMPLETE_FAIL_CLOSED"
    )
    training_blockers.append(
        "downstream_task_manifest_not_bound_to_selected_station_center_shift_xy_m"
    )
    training_blockers.append(
        "training_adopted_registry_E2_evidence_and_four_pin_runtime_binding_missing"
    )
    training_blockers.append(
        "trusted_generic_bank_promotion_capability_missing"
    )
    training_authorized = False

    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": CERTIFICATE_KIND,
        "action_id": action_id,
        "verdict": "BLOCKED",
        "publication_class": "diagnostic_only_blocked",
        "admission_capability_minted": False,
        "admission_authority": (
            "code_rooted_canonical_motion_admission_only; "
            "this diagnostic never mints a runtime capability"
        ),
        "diagnostic_reference_checks_pass": diagnostic_reference_checks_pass,
        "diagnostic_smoke_authorized": diagnostic_smoke_authorized,
        "training_authorized": training_authorized,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "selected_station_center_shift_xy_m": (
            None if selected is None else list(selected)
        ),
        "station_comparison_input_sha256": comparison_input_sha256,
        "station_selection_approval": station_selection_approval,
        "station_selection_policy": (
            "independent_code_pinned_approval_of_exact_comparison_input; "
            "the plan author cannot self-approve after observing results; "
            "translation applies to the whole action and task center, never "
            "only to a base reward"
        ),
        "behavior_contact_authority": behavior_contact_evidence,
        "required_scopes": list(SCOPES),
        "required_gates": list(REQUIRED_GATES),
        "diagnostic_reference_gates": list(DIAGNOSTIC_REFERENCE_GATES),
        "bindings": {name: snapshot.binding() for name, snapshot in snapshots.items()},
        "cross_scope_shared_ready": {
            "pass": cross_scope_ready_pass,
            "joint_position_max_abs_delta_rad": cross_scope_joint_delta,
            "body_position_max_delta_m": cross_scope_body_pos_delta,
            "body_orientation_max_delta_rad": cross_scope_body_quat_delta,
            "worst_mixed_unit_delta": cross_scope_ready_max,
            "tolerance": thresholds["shared_ready_pose_tolerance"],
        },
        "scopes": scope_results,
        "diagnostic_reference_blockers": diagnostic_reference_blockers,
        "diagnostic_smoke_blockers": diagnostic_smoke_blockers,
        "training_blockers": training_blockers,
        "non_claims": [
            "dense collision interpolation is finite sampling, not a mathematical swept-volume proof",
            "reference returnability is an analytic necessary screen, not observed policy success",
            "the reported source anchor and nearest runtime tick are not t_hit or observed ball contact",
            "shared-ready endpoint equality is kinematic reference evidence, not policy recovery",
            "reference certification is not deployment or hardware authorization",
            "this report never authorizes a diagnostic simulator smoke or training",
            "external JSON receipts are untrusted diagnostics until playback and collision producers are recomputed in-process",
            "station_center_shift_xy_m translates the whole action/task center; it is not a base-only reward offset",
            "a trained policy may still hit the table even when the kinematic reference clears it; Isaac smoke remains mandatory",
            "table legs are absent because the shared Isaac/MuJoCo table contract has no legs",
            "no real robot command, simulator step, training run, or asset mutation occurred",
        ],
    }


def _slerp_wxyz(lhs: np.ndarray, rhs: np.ndarray, fraction: float) -> np.ndarray:
    left = np.asarray(lhs, np.float64)
    right = np.asarray(rhs, np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        out = (1.0 - fraction) * left + fraction * right
        return out / np.linalg.norm(out)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sine * left
        + math.sin(fraction * angle) / sine * right
    )


def _dense_qpos(
    clip: Any,
    plant: Any,
    *,
    station_center_shift_xy_m: Tuple[float, float],
    substeps: int,
) -> np.ndarray:
    frames = int(clip.n_frames)
    count = (frames - 1) * substeps + 1
    out = np.repeat(np.asarray(plant.contact_model.qpos0)[None, :], count, axis=0)
    cursor = 0
    root_pos = np.asarray(clip.body_pos_w[:, 0], np.float64)
    root_quat = np.asarray(clip.body_quat_w[:, 0], np.float64)
    joints = np.asarray(clip.joint_pos, np.float64)
    for frame in range(frames - 1):
        for step in range(substeps):
            alpha = step / float(substeps)
            position = (1.0 - alpha) * root_pos[frame] + alpha * root_pos[frame + 1]
            position = position.copy()
            position[:2] += np.asarray(station_center_shift_xy_m, np.float64)
            out[cursor, 0:3] = position
            out[cursor, 3:7] = _slerp_wxyz(
                root_quat[frame], root_quat[frame + 1], alpha
            )
            out[cursor, plant.joint_qposadr] = (
                (1.0 - alpha) * joints[frame] + alpha * joints[frame + 1]
            )
            cursor += 1
    position = root_pos[-1].copy()
    position[:2] += np.asarray(station_center_shift_xy_m, np.float64)
    out[cursor, 0:3] = position
    out[cursor, 3:7] = root_quat[-1] / np.linalg.norm(root_quat[-1])
    out[cursor, plant.joint_qposadr] = joints[-1]
    return out


def scan_collisions(
    *,
    action_id: str,
    scope: str,
    station_center_shift_xy_m: Sequence[float],
    motion_path: Path,
    expected_motion_sha256: str,
    mjcf_path: Path,
    expected_mjcf_sha256: str,
    urdf_path: Path,
    expected_urdf_sha256: str,
    expected_compiled_signature: str,
    substeps: int,
) -> Mapping[str, Any]:
    """Dense finite collision scan using the existing canonical MuJoCo plant."""

    action_id = _nonempty(action_id, "action_id")
    if scope not in SCOPES:
        raise CertificationError(f"scope must be one of {list(SCOPES)}")
    shift = STATION_CENTER_SHIFT_CANDIDATES_XY_M[
        _station_index(
            list(station_center_shift_xy_m),
            "station_center_shift_xy_m",
        )
    ]
    _integer(substeps, "substeps", 1)
    for path, digest, label in (
        (motion_path, expected_motion_sha256, "motion"),
        (mjcf_path, expected_mjcf_sha256, "mjcf"),
        (urdf_path, expected_urdf_sha256, "urdf"),
    ):
        if path.is_symlink() or not path.is_file():
            raise CertificationError(f"{label} must be a regular non-symlink file")
        if _sha256_file(path) != _digest(digest, f"{label} sha256"):
            raise CertificationError(f"{label} SHA-256 mismatch")

    motion_scripts = (
        Path(__file__).resolve().parents[1]
        / "hope_training/whole_body_tracking/scripts"
    )
    repo_scripts = Path(__file__).resolve().parent
    for path in (motion_scripts, repo_scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import canonical_mujoco_dynamics_gate as dynamics  # type: ignore
        import mujoco_motion_player as player  # type: ignore
        import mujoco_table_scene  # type: ignore
        import mujoco  # type: ignore
    except Exception as exc:
        raise CertificationError(f"MuJoCo collision dependencies unavailable: {exc}") from exc

    clip = player.load_motion(motion_path)
    plant = dynamics.load_plant(
        mjcf_path,
        expected_mjcf_sha256=expected_mjcf_sha256,
        expected_compiled_signature=_digest(
            expected_compiled_signature, "compiled model signature"
        ),
        urdf_path=urdf_path,
        with_table=True,
    )
    if plant.urdf_sha256 != expected_urdf_sha256:
        raise CertificationError("loaded plant URDF SHA differs from pinned input")
    qpos = _dense_qpos(
        clip,
        plant,
        station_center_shift_xy_m=shift,
        substeps=substeps,
    )
    qvel = np.zeros((qpos.shape[0], int(plant.nv)), np.float64)
    _frames, contacts, _com, _tilt, _contact_count = dynamics._scan_pose_geometry(
        plant, qpos, qvel
    )
    thresholds = dynamics.GateThresholds()
    self_bad = [
        row
        for row in contacts["self_contacts"]
        if float(row["depth_m"]) > thresholds.self_penetration_tolerance_m
    ]
    foot_bad = [
        row
        for row in contacts["floor_contacts"]
        if row["robot_body_is_foot"]
        and float(row["depth_m"]) > thresholds.foot_floor_penetration_tolerance_m
    ]
    nonfoot_bad = [
        row
        for row in contacts["floor_contacts"]
        if not row["robot_body_is_foot"]
        and float(row["depth_m"]) > thresholds.nonfoot_floor_penetration_tolerance_m
    ]

    obstacle_bad: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in OBSTACLES}
    for row in contacts["other_world_contacts"]:
        pair = tuple(row.get("geom_pair", ()))
        obstacle = next((name for name in OBSTACLES if name in pair), None)
        if obstacle is None:
            raise CertificationError(
                f"unexpected non-floor world contact in table model: {pair}"
            )
        if float(row["depth_m"]) > thresholds.nonfoot_floor_penetration_tolerance_m:
            obstacle_bad[obstacle].append(row)

    model = plant.contact_model
    obstacle_ids = _mapping(
        plant.table_scene.obstacle_geom_ids, "table obstacle geom ids"
    )
    robot_geom_ids = [
        gid
        for gid in range(int(model.ngeom))
        if int(model.geom_bodyid[gid]) != 0
        and (
            int(model.geom_contype[gid]) != 0
            or int(model.geom_conaffinity[gid]) != 0
        )
    ]
    robot_geom_names = [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}")
        for gid in robot_geom_ids
    ]
    for name in RACKET_COLLISION_GEOMS:
        if robot_geom_names.count(name) != 1:
            raise CertificationError(f"collision model lacks exact racket geom {name}")

    distance_max = 1.0
    minimum = distance_max
    minimum_row: Optional[Mapping[str, Any]] = None
    data = mujoco.MjData(model)
    fromto = np.zeros(6, np.float64)
    for sample in range(qpos.shape[0]):
        data.qpos[:] = qpos[sample]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for robot_id, robot_name in zip(robot_geom_ids, robot_geom_names):
            for obstacle, obstacle_id in obstacle_ids.items():
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        int(robot_id),
                        int(obstacle_id),
                        distance_max,
                        fromto,
                    )
                )
                if distance < minimum:
                    minimum = distance
                    minimum_row = {
                        "sample": sample,
                        "time_s": sample / (float(clip.fps) * substeps),
                        "robot_geom": robot_name,
                        "obstacle": obstacle,
                        "distance_m": distance,
                    }

    def check(rows: Sequence[Mapping[str, Any]], tolerance: float) -> Mapping[str, Any]:
        peak = max((float(row["depth_m"]) for row in rows), default=0.0)
        return {
            "pass": len(rows) == 0,
            "violation_sample_count": len({int(row["frame"]) for row in rows}),
            "violation_contact_count": len(rows),
            "maximum_penetration_m": peak,
            "tolerance_m": float(tolerance),
        }

    table_check = check(
        obstacle_bad[TABLE_TOP], thresholds.nonfoot_floor_penetration_tolerance_m
    )
    net_check = check(
        obstacle_bad[NET], thresholds.nonfoot_floor_penetration_tolerance_m
    )
    posts = obstacle_bad[NET_POSTS[0]] + obstacle_bad[NET_POSTS[1]]
    post_check = check(posts, thresholds.nonfoot_floor_penetration_tolerance_m)
    checks = {
        "self_collision": check(
            self_bad, thresholds.self_penetration_tolerance_m
        ),
        "foot_ground_penetration": check(
            foot_bad, thresholds.foot_floor_penetration_tolerance_m
        ),
        "nonfoot_ground_collision": check(
            nonfoot_bad, thresholds.nonfoot_floor_penetration_tolerance_m
        ),
        "table_top_collision": table_check,
        "net_collision": net_check,
        "net_post_collision": post_check,
    }
    aggregate_pass = all(row["pass"] is True for row in checks.values())
    checks["aggregate"] = {"pass": aggregate_pass}
    tool_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": COLLISION_REPORT_KIND,
        "action_id": action_id,
        "scope": scope,
        "station_center_shift_xy_m": list(shift),
        "verdict": "PASS" if aggregate_pass else "FAIL",
        "artifacts": {
            "motion": {
                "path": str(motion_path.resolve()),
                "sha256": expected_motion_sha256,
            },
            "mjcf": {
                "path": str(mjcf_path.resolve()),
                "sha256": expected_mjcf_sha256,
            },
            "urdf": {
                "path": str(urdf_path.resolve()),
                "sha256": expected_urdf_sha256,
            },
            "compiled_model_signature_sha256": expected_compiled_signature,
            "tool": {"path": str(tool_path), "sha256": _sha256_file(tool_path)},
        },
        "sampling": {
            "source_fps": float(clip.fps),
            "substeps_per_source_interval": substeps,
            "sample_hz": float(clip.fps) * substeps,
            "sample_count": int(qpos.shape[0]),
            "entire_cycle": True,
            "interpolation": (
                "root_xyz_and_joint_linear_plus_shortest_arc_root_quaternion_slerp"
            ),
            "mj_forward_calls": int(2 * qpos.shape[0]),
            "mj_step_calls": 0,
        },
        "model": {
            "robot_collision_geom_count": len(robot_geom_ids),
            "racket_collision_geoms_included": list(RACKET_COLLISION_GEOMS),
            "obstacle_names": list(OBSTACLES),
            "table_legs_present": False,
        },
        "checks": checks,
        "clearance": {
            "minimum_table_net_clearance_m": float(minimum),
            "distance_query_cap_m": distance_max,
            "minimum": minimum_row,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "finite dense interpolation is not mathematical continuous swept-volume proof",
            "kinematic pose collision is not grounded dynamics, balance, or controller tracking",
            "table legs are absent from the shared simulator contract",
        ],
    }


def template_plan(action_id: str, source_path: str, source_sha256: str) -> Mapping[str, Any]:
    """Emit a visibly incomplete preregistration skeleton; it can never certify as-is."""

    return {
        "schema_version": SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "action_id": _nonempty(action_id, "action_id"),
        "bindings": {
            "source": {
                "path": _nonempty(source_path, "source_path"),
                "sha256": _digest(source_sha256, "source_sha256"),
            },
            "recipe": {"path": "FILL_ME", "sha256": "FILL_ME"},
            "build_manifest": {"path": "FILL_ME", "sha256": "FILL_ME"},
            "canonical_verifier_report": {
                "path": "FILL_ME",
                "sha256": "FILL_ME",
            },
            "mjcf": {"path": "FILL_ME", "sha256": "FILL_ME"},
            "venue_yaml": {"path": "FILL_ME", "sha256": "FILL_ME"},
            "venue_profile": {"path": "FILL_ME", "sha256": "FILL_ME"},
        },
        "required_scopes": list(SCOPES),
        "station_center_shift_candidates_xy_m": [
            list(row) for row in STATION_CENTER_SHIFT_CANDIDATES_XY_M
        ],
        "selected_station_center_shift_xy_m": None,
        "station_selection_approval": None,
        "behavior_contact_evidence": None,
        "thresholds": {
            "source_anchor_time_min_s": None,
            "source_anchor_time_max_s": None,
            "t_hit_reference_s": T_HIT_REFERENCE_S,
            "t_cycle_min_s": None,
            "t_cycle_max_s": None,
            "blade_site_speed_min_m_s": None,
            "blade_site_speed_max_m_s": None,
            "shared_ready_pose_tolerance": 1.0e-6,
            "dense_collision_min_hz": 400.0,
            "minimum_table_net_clearance_m": None,
        },
        "task_distribution": {
            "incoming_velocity_box_m_s": None,
            "spin_abs_max_rad_s": None,
            "samples": None,
            "seed": 0,
            "face_sign": 1.0,
            "capture_radius_m": 0.095,
            "minimum_approach_speed_m_s": 0.3,
            "minimum_legal_return_fraction": None,
        },
        "scopes": {
            scope: {
                "motion": {"path": "FILL_ME", "sha256": "FILL_ME"},
                "playback_report": {"path": "FILL_ME", "sha256": "FILL_ME"},
                "collision_reports": [
                    {"path": "FILL_ME", "sha256": "FILL_ME"}
                    for _ in STATION_CENTER_SHIFT_CANDIDATES_XY_M
                ],
            }
            for scope in SCOPES
        },
        "authorization_intent": "task_first_training_only_no_deployment_no_hardware",
    }


def _write_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="print an incomplete fail-closed plan")
    template.add_argument("--action-id", required=True)
    template.add_argument("--source-path", required=True)
    template.add_argument("--source-sha256", required=True)

    scan = subparsers.add_parser(
        "scan-collisions",
        help="dense read-only MuJoCo collision scan at one preregistered station shift",
    )
    scan.add_argument("--action-id", required=True)
    scan.add_argument("--scope", choices=SCOPES, required=True)
    scan.add_argument(
        "--station-center-shift-xy-m",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        required=True,
        help="whole-action/task-center translation; negative X is farther from table",
    )
    scan.add_argument("--motion", type=Path, required=True)
    scan.add_argument("--expected-motion-sha256", required=True)
    scan.add_argument("--mjcf", type=Path, required=True)
    scan.add_argument("--expected-mjcf-sha256", required=True)
    scan.add_argument("--urdf", type=Path, required=True)
    scan.add_argument("--expected-urdf-sha256", required=True)
    scan.add_argument("--expected-compiled-signature", required=True)
    scan.add_argument("--substeps", type=int, default=8)
    scan.add_argument("--out", type=Path, required=True)

    certify = subparsers.add_parser(
        "certify", help="verify a content-addressed upper/full certification plan"
    )
    certify.add_argument("--plan", type=Path, required=True)
    certify.add_argument("--expected-plan-sha256", required=True)
    certify.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "template":
        print(
            json.dumps(
                template_plan(args.action_id, args.source_path, args.source_sha256),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "scan-collisions":
        try:
            report = scan_collisions(
                action_id=args.action_id,
                scope=args.scope,
                station_center_shift_xy_m=args.station_center_shift_xy_m,
                motion_path=args.motion.resolve(),
                expected_motion_sha256=args.expected_motion_sha256,
                mjcf_path=args.mjcf.resolve(),
                expected_mjcf_sha256=args.expected_mjcf_sha256,
                urdf_path=args.urdf.resolve(),
                expected_urdf_sha256=args.expected_urdf_sha256,
                expected_compiled_signature=args.expected_compiled_signature,
                substeps=args.substeps,
            )
            _write_no_clobber(args.out, report)
            return 0 if report["verdict"] == "PASS" else 2
        except Exception as exc:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "report_kind": COLLISION_REPORT_KIND,
                "verdict": "FAIL",
                "diagnostic_smoke_authorized": False,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_no_clobber(args.out, failure)
            return 2
    if args.command == "certify":
        plan_path = args.plan.resolve()
        try:
            if plan_path.is_symlink() or not plan_path.is_file():
                raise CertificationError("plan must be a regular non-symlink file")
            expected = _digest(args.expected_plan_sha256, "expected plan SHA-256")
            data = plan_path.read_bytes()
            actual = _sha256_bytes(data)
            if actual != expected:
                raise CertificationError(
                    f"plan SHA-256 mismatch: expected={expected} actual={actual}"
                )
            plan = _parse_json(data, "certification plan")
            report = certify_plan(plan, base_dir=plan_path.parent)
            report = dict(report)
            report["plan"] = {"path": str(plan_path), "sha256": actual}
            _write_no_clobber(args.out, report)
            return 0 if report["diagnostic_smoke_authorized"] is True else 2
        except Exception as exc:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "report_kind": CERTIFICATE_KIND,
                "verdict": "FAIL",
                "publication_class": "diagnostic_only_blocked",
                "admission_capability_minted": False,
                "diagnostic_smoke_authorized": False,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_no_clobber(args.out, failure)
            return 2
    raise AssertionError(f"unreachable command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
