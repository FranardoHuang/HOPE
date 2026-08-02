#!/usr/bin/env python3
"""Fail-closed teacher-motion + native MuJoCo ball-return diagnostic A/B.

This tool answers one narrow physical question for every action in an exact
action-conditioned ball manifest:

    Does the action's own schema-2 teacher motion, replayed continuously in the
    exact vendor A3 model with a dynamic thin-shell ball and collidable
    table/net, make a *native MuJoCo* racket contact, clear the net, and make
    its first post-hit table contact on the opponent half without robot-table,
    self-contact, fall, or joint-limit violations under one explicitly
    calibrated native-MuJoCo material recipe?

It deliberately does not load a policy, run the action selector, invoke the
ball->task solver, or call ``virtual_return_scorer``.  The solver and physics
profile are byte-bound only so the center incoming ball has the same identity
as training.  Contact force comes exclusively from ``mj_contactForce`` and
landing comes exclusively from a native ball/table contact.

Evidence boundary
-----------------
The robot trajectory is prescribed at every MuJoCo physics substep.  The ball
and all contacts are dynamic; the robot is not.  A PASS is therefore a
``diagnostic_native_ab`` result, not the formal venue-physics return Gate, a
policy/PD-plant certificate, Isaac/MuJoCo parity, deployment, or hardware
authorization.

The formal referee remains BLOCKED in this file.  Its required authority is a
dynamic ball with frozen venue-fitted aero/Magnus and paddle/table impulses,
where the paddle impulse is triggered only by an actual swept selected-face
intersection (including face-center offset, ball radius, and omega-cross-r
velocity) and native ball/racket impulse is disabled to prevent double
counting.  This diagnostic intentionally does not emulate that authority or
claim its PASS.

The repository currently has no reviewed mapping from the venue restitution
fit to MuJoCo ``solref/solimp/friction``.  Diagnostic execution consequently
requires an external, pre-registered native-material certificate and its
expected SHA-256.  No guessed material is supplied by this script.
``--preflight-only`` emits all blockers without importing MuJoCo.

The current ``action_ball_n5_*_20260728.json`` files contain only four actions.
The default ``--expected-actions 5`` rejects them even though their filename
says N5.

Exit codes
~~~~~~~~~~
0
    All actions passed the diagnostic A/B.  This never authorizes the formal
    venue-physics Gate.
2
    Malformed/untrusted input or runtime infrastructure failure.
3
    Complete receipt was produced, but preflight is blocked or one/more
    physical actions failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
WBT_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
ROOT_SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT_SCRIPTS))

import mujoco_motion_player as motion_player  # noqa: E402
import mujoco_table_scene as table_scene  # noqa: E402
import racket_geometry_contract as racket_geometry  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_SEMANTICS = "diagnostic_kinematic_teacher_native_mujoco_contact_ab"
PROFILE_PINS_SCHEMA_VERSION = 1
PROFILE_PINS_KIND = "whole_body_tracking.action_ball.profile_pins"
PROFILE_PINS_SOURCE_NAMES = frozenset(
    {
        "continuous_questions.py",
        "hope_commands.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    }
)
PROFILE_PINS_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_authority",
        "cfg",
        "geometry",
        "planes",
        "venue_yaml",
        "venue_yaml_sha256",
        "solver_implementation_source_sha256",
        "contact_geometry",
        "physics_profile_sha256",
        "solver_profile_sha256",
        "physics_payload",
        "solver_payload",
    }
)
BALL_BODY_NAME = "teacher_ball_body"
BALL_JOINT_NAME = "teacher_ball_freejoint"
BALL_GEOM_NAME = "teacher_ball_geom"
RACKET_GEOM_NAME = "right_racket_collision"
RACKET_HANDLE_GEOM_NAME = "right_racket_handle_collision"
TABLE_GEOM_NAME = "motion_table_top"
NET_GEOM_NAMES = (
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)
DEFAULT_MANIFEST = REPO_ROOT / "configs/action_ball_n5_nomove_f20_20260728.json"
DEFAULT_PROFILE_PINS = REPO_ROOT / "configs/action_ball_profile_pins_20260728.json"
DEFAULT_MJCF = table_scene.CANONICAL_MJCF
DEFAULT_IDENTITY_MANIFEST = REPO_ROOT / "configs/a3_mujoco_identity_v2_20260803.json"
DEFAULT_IDENTITY_MANIFEST_SHA256 = (
    "b8fc5deaaff8d213c2d077a0e7892b30d7f5a6c77c3d06dc029e3a2616d54d91"
)
RACKET_GEOMETRY_PRODUCTION_SOURCE = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
)
RACKET_GEOMETRY_SOURCE_SHA256 = str(
    racket_geometry.GEOMETRY_SOURCE_SHA256
)
ROOT_Z_FALL_M = 0.55
ROOT_TILT_FALL_RAD = 0.70
EPS = 1.0e-12


class GateError(ValueError):
    """Malformed, untrusted, or internally inconsistent gate input."""


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise GateError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _reject_nonfinite_constant(value: str) -> None:
    raise GateError(f"non-finite JSON constant {value!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_json_exact(
    path: os.PathLike[str] | str,
    label: str,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file() or p.is_symlink():
        raise GateError(f"{label} must be a regular non-symlink file: {p}")
    raw = p.read_bytes()
    digest = sha256_bytes(raw)
    if expected_sha256 is not None:
        _require_sha(expected_sha256, f"expected {label} SHA-256")
        if digest != expected_sha256:
            raise GateError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value, {
        "path": str(p),
        "sha256": digest,
        "size_bytes": len(raw),
    }


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GateError(f"{label} must be a lowercase 64-digit SHA-256")
    return value


def _number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be a JSON number")
    out = float(value)
    if not math.isfinite(out):
        raise GateError(f"{label} must be finite")
    if positive and out <= 0.0:
        raise GateError(f"{label} must be > 0")
    if nonnegative and out < 0.0:
        raise GateError(f"{label} must be >= 0")
    return out


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be an integer")
    if positive and value <= 0:
        raise GateError(f"{label} must be > 0")
    return int(value)


def _vector(value: Any, n: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise GateError(f"{label} must contain exactly {n} JSON numbers")
    return np.asarray([_number(v, f"{label}[{i}]") for i, v in enumerate(value)], np.float64)


def _unit_vector(value: Any, label: str, tolerance: float = 2.0e-5) -> np.ndarray:
    out = _vector(value, 3, label)
    norm = float(np.linalg.norm(out))
    if abs(norm - 1.0) > tolerance:
        raise GateError(f"{label} must be unit length, got {norm:.9g}")
    return out / norm


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{label} must be a non-empty string")
    return value


def _resolve_repo_file(
    relative: Any,
    label: str,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    raw = _nonempty_string(relative, label)
    root = (REPO_ROOT if repo_root is None else Path(repo_root)).resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise GateError(f"{label} escapes the repository root: {raw!r}") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise GateError(f"{label} is not a regular non-symlink file: {candidate}")
    return candidate


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    action_uid: int
    motion_path: Path
    motion_sha256: str
    strike_phase: float
    t_hit_s: float
    t_cycle_s: float
    racket_speed_mps: float
    reaction_margin_s: float
    mount_normal_sign: int
    ball_profile: Mapping[str, Any]


@dataclass(frozen=True)
class ManifestContract:
    manifest_id: str
    mobility_mode: str
    action_order: Tuple[str, ...]
    actions: Tuple[ActionSpec, ...]
    solver_profile_sha256: str
    physics_profile_sha256: str
    racket_geometry_contract: Mapping[str, Any]


def validate_racket_geometry_binding(raw: Any) -> Dict[str, Any]:
    """Require an explicit migration away from legacy site==ball co-location."""

    row = _mapping(raw, "manifest.racket_geometry_contract")
    if _integer(row.get("schema_version"), "racket geometry schema") != 2:
        raise GateError("racket geometry schema_version must equal 2")
    if row.get("semantics") != "exact_face_contact_v2":
        raise GateError(
            "manifest lacks physical face-center + ball-radius target semantics"
        )
    if row.get("ball_target_point") != "physical_ball_center_at_native_contact":
        raise GateError("manifest ball target must be the physical ball center")
    if row.get("site_target_mapping") != "site_target_from_ball_center":
        raise GateError("manifest does not bind the site<-ball-center offset mapping")
    if (
        row.get("face_velocity_mapping")
        != "site_linear_plus_omega_cross_face_center_offset"
    ):
        raise GateError("manifest does not bind omega-cross-r face-center velocity")
    source_path = _resolve_repo_file(
        row.get("source_path"), "racket geometry source path"
    )
    source_sha = _require_sha(
        row.get("source_sha256"), "racket geometry source SHA"
    )
    actual_sha = sha256_file(source_path)
    geometry_payload_sha = _require_sha(
        row.get("geometry_source_sha256"),
        "racket geometry canonical payload SHA",
    )
    if (
        source_path != RACKET_GEOMETRY_PRODUCTION_SOURCE.resolve()
        or actual_sha != source_sha
        or geometry_payload_sha != RACKET_GEOMETRY_SOURCE_SHA256
    ):
        raise GateError(
            "versioned physical racket geometry source does not match the gate pin"
        )
    return {
        "schema_version": 2,
        "semantics": row["semantics"],
        "ball_target_point": row["ball_target_point"],
        "site_target_mapping": row["site_target_mapping"],
        "face_velocity_mapping": row["face_velocity_mapping"],
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "geometry_source_sha256": geometry_payload_sha,
        "official_ball_radius_m": float(racket_geometry.BALL_RADIUS_M),
        "red_site_to_ball_center_m": racket_geometry.ball_center_from_site_local(
            +1
        ).tolist(),
        "black_site_to_ball_center_m": racket_geometry.ball_center_from_site_local(
            -1
        ).tolist(),
        "red_legacy_colocation_error_m": racket_geometry.legacy_colocation_error_m(
            +1
        ),
        "black_legacy_colocation_error_m": racket_geometry.legacy_colocation_error_m(
            -1
        ),
    }


def validate_manifest(
    raw: Mapping[str, Any],
    *,
    expected_actions: int,
) -> ManifestContract:
    if _integer(raw.get("schema_version"), "manifest.schema_version") != 3:
        raise GateError("manifest.schema_version must equal 3")
    manifest_id = _nonempty_string(raw.get("manifest_id"), "manifest.manifest_id")
    mobility_mode = _nonempty_string(raw.get("mobility_mode"), "manifest.mobility_mode")
    order_raw = raw.get("action_order")
    if not isinstance(order_raw, list) or any(not isinstance(v, str) or not v for v in order_raw):
        raise GateError("manifest.action_order must be a non-empty string list")
    order = tuple(order_raw)
    if len(order) != expected_actions:
        raise GateError(
            f"action_count_mismatch: expected exact N={expected_actions}, "
            f"manifest.action_order has {len(order)}"
        )
    if len(set(order)) != len(order):
        raise GateError("manifest.action_order contains duplicates")
    actions_raw = raw.get("actions")
    if not isinstance(actions_raw, list) or len(actions_raw) != expected_actions:
        actual = len(actions_raw) if isinstance(actions_raw, list) else "non-list"
        raise GateError(
            f"action_count_mismatch: expected exact N={expected_actions}, "
            f"manifest.actions has {actual}"
        )
    geometry_binding = validate_racket_geometry_binding(
        raw.get("racket_geometry_contract")
    )

    actions: List[ActionSpec] = []
    seen_uids: set[int] = set()
    for index, (expected_id, row_value) in enumerate(zip(order, actions_raw)):
        row = _mapping(row_value, f"manifest.actions[{index}]")
        action_id = _nonempty_string(row.get("action_id"), f"actions[{index}].action_id")
        if action_id != expected_id:
            raise GateError(
                f"action order drift at dense slot {index}: order={expected_id!r}, row={action_id!r}"
            )
        uid = _integer(row.get("action_uid"), f"{action_id}.action_uid", positive=True)
        if uid in seen_uids:
            raise GateError(f"duplicate action_uid {uid}")
        seen_uids.add(uid)
        motion_path = _resolve_repo_file(row.get("motion_path"), f"{action_id}.motion_path")
        motion_sha = _require_sha(row.get("motion_sha256"), f"{action_id}.motion_sha256")
        actual_motion_sha = sha256_file(motion_path)
        if actual_motion_sha != motion_sha:
            raise GateError(
                f"{action_id} motion SHA mismatch: expected {motion_sha}, got {actual_motion_sha}"
            )
        strike_phase = _number(row.get("strike_phase"), f"{action_id}.strike_phase")
        if not 0.0 <= strike_phase <= 1.0:
            raise GateError(f"{action_id}.strike_phase must be in [0,1]")
        t_hit = _number(row.get("reference_t_hit_s"), f"{action_id}.reference_t_hit_s", positive=True)
        t_cycle = _number(
            row.get("reference_t_cycle_s"), f"{action_id}.reference_t_cycle_s", positive=True
        )
        if t_hit >= t_cycle:
            raise GateError(f"{action_id}: t_hit must be before t_cycle")
        if abs(strike_phase * t_cycle - t_hit) > 1.0e-5:
            raise GateError(
                f"{action_id}: strike_phase, t_hit, and t_cycle are "
                "not the same frozen timing law"
            )
        speed = _number(
            row.get("reference_racket_site_speed_mps"),
            f"{action_id}.reference_racket_site_speed_mps",
            positive=True,
        )
        reaction = _number(
            row.get("reaction_margin_s"), f"{action_id}.reaction_margin_s", nonnegative=True
        )
        mount_normal_sign = _integer(
            row.get("mount_normal_sign"), f"{action_id}.mount_normal_sign"
        )
        if mount_normal_sign not in (-1, 1):
            raise GateError(f"{action_id}.mount_normal_sign must be -1 or +1")
        profile = _mapping(row.get("ball_profile"), f"{action_id}.ball_profile")
        ttc = _number(
            profile.get("time_to_contact_center_s"),
            f"{action_id}.ball_profile.time_to_contact_center_s",
            positive=True,
        )
        if ttc + EPS < t_hit + reaction:
            raise GateError(
                f"{action_id}: center time-to-contact {ttc} is below "
                f"t_hit+reaction_margin {t_hit + reaction}"
            )
        wait = ttc - t_hit
        if wait < -EPS or wait > 1.0 + EPS:
            raise GateError(
                f"{action_id}: pre-swing wait {wait} is outside the frozen [0,1] s bound"
            )
        _vector(
            profile.get("contact_offset_center_b_yaw_m"),
            3,
            f"{action_id}.contact_offset_center_b_yaw_m",
        )
        direction = _unit_vector(
            profile.get("incoming_direction_center_b_yaw"),
            f"{action_id}.incoming_direction_center_b_yaw",
        )
        inbound = _unit_vector(
            profile.get("incoming_inbound_axis_b_yaw"),
            f"{action_id}.incoming_inbound_axis_b_yaw",
        )
        inbound_cos = _number(
            profile.get("incoming_inbound_min_cosine"),
            f"{action_id}.incoming_inbound_min_cosine",
        )
        if not -1.0 <= inbound_cos <= 1.0:
            raise GateError(f"{action_id}: inbound cosine must be in [-1,1]")
        if float(direction @ inbound) + 1.0e-9 < inbound_cos:
            raise GateError(f"{action_id}: center incoming direction is outside its inbound cone")
        _number(
            profile.get("incoming_speed_center_mps"),
            f"{action_id}.incoming_speed_center_mps",
            positive=True,
        )
        _unit_vector(
            profile.get("spin_direction_center_b_yaw"),
            f"{action_id}.spin_direction_center_b_yaw",
        )
        _number(
            profile.get("spin_magnitude_center_radps"),
            f"{action_id}.spin_magnitude_center_radps",
            nonnegative=True,
        )
        _vector(
            profile.get("base_spawn_center_w_xy_m"),
            2,
            f"{action_id}.base_spawn_center_w_xy_m",
        )
        travel = _vector(
            profile.get("base_travel_center_b_yaw_xy_m"),
            2,
            f"{action_id}.base_travel_center_b_yaw_xy_m",
        )
        if mobility_mode == "no_move" and float(np.max(np.abs(travel))) > EPS:
            raise GateError(f"{action_id}: no_move requires zero base travel center")
        actions.append(
            ActionSpec(
                action_id=action_id,
                action_uid=uid,
                motion_path=motion_path,
                motion_sha256=motion_sha,
                strike_phase=strike_phase,
                t_hit_s=t_hit,
                t_cycle_s=t_cycle,
                racket_speed_mps=speed,
                reaction_margin_s=reaction,
                mount_normal_sign=mount_normal_sign,
                ball_profile=profile,
            )
        )

    return ManifestContract(
        manifest_id=manifest_id,
        mobility_mode=mobility_mode,
        action_order=order,
        actions=tuple(actions),
        solver_profile_sha256=_require_sha(
            raw.get("solver_profile_sha256"), "manifest.solver_profile_sha256"
        ),
        physics_profile_sha256=_require_sha(
            raw.get("physics_profile_sha256"), "manifest.physics_profile_sha256"
        ),
        racket_geometry_contract=geometry_binding,
    )


def validate_profile_pins(
    raw: Mapping[str, Any],
    manifest: Optional[ManifestContract],
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate the one portable formal profile-pins document.

    The exact executable source map is the code identity inside this
    repository document.  Its commit is intentionally supplied by the
    immutable launch capsule and checked separately; embedding ``source_rev``
    here would be self-referential once the pins file itself is committed.
    """

    root = (REPO_ROOT if repo_root is None else Path(repo_root)).resolve()
    if "source_rev" in raw:
        raise GateError(
            "profile_pins.source_rev is a repo-contained commit self-reference "
            "and is forbidden; use external exact-commit subset-blob authority"
        )
    if raw.get("schema_version") != PROFILE_PINS_SCHEMA_VERSION:
        raise GateError(
            f"profile_pins.schema_version must equal "
            f"{PROFILE_PINS_SCHEMA_VERSION}"
        )
    if raw.get("kind") != PROFILE_PINS_KIND:
        raise GateError(f"profile_pins.kind must equal {PROFILE_PINS_KIND!r}")
    if set(raw) != PROFILE_PINS_TOP_LEVEL_KEYS:
        missing = sorted(PROFILE_PINS_TOP_LEVEL_KEYS - set(raw))
        extra = sorted(set(raw) - PROFILE_PINS_TOP_LEVEL_KEYS)
        raise GateError(
            "profile_pins top-level schema is not exact: "
            f"missing={missing}, extra={extra}"
        )

    solver_payload = _mapping(raw.get("solver_payload"), "profile_pins.solver_payload")
    physics_payload = _mapping(raw.get("physics_payload"), "profile_pins.physics_payload")
    if (
        solver_payload.get("schema_version") != 2
        or solver_payload.get("kind")
        != "whole_body_tracking.continuous_questions.solve_proposals"
    ):
        raise GateError(
            "solver payload must be the exact schema-2 fixed-action solver"
        )
    if (
        physics_payload.get("schema_version") != 1
        or physics_payload.get("kind")
        != "whole_body_tracking.action_ball.physics_and_scorer"
    ):
        raise GateError(
            "physics payload must be the exact schema-1 physics/scorer profile"
        )
    solver_digest = sha256_bytes(canonical_json_bytes(solver_payload))
    physics_digest = sha256_bytes(canonical_json_bytes(physics_payload))
    for label, computed, listed, manifest_value in (
        (
            "solver",
            solver_digest,
            _require_sha(raw.get("solver_profile_sha256"), "profile solver SHA"),
            None if manifest is None else manifest.solver_profile_sha256,
        ),
        (
            "physics",
            physics_digest,
            _require_sha(raw.get("physics_profile_sha256"), "profile physics SHA"),
            None if manifest is None else manifest.physics_profile_sha256,
        ),
    ):
        if computed != listed or (
            manifest_value is not None and computed != manifest_value
        ):
            raise GateError(
                f"{label} profile identity mismatch: computed={computed}, "
                f"pins={listed}, manifest={manifest_value}"
            )
    if solver_payload.get("physics_profile_sha256") != physics_digest:
        raise GateError(
            "solver payload does not bind the exact physics profile SHA"
        )

    source_map = _mapping(
        raw.get("solver_implementation_source_sha256"),
        "profile_pins.solver_implementation_source_sha256",
    )
    payload_source = _mapping(
        solver_payload.get("implementation_source_sha256"),
        "solver_payload.implementation_source_sha256",
    )
    if dict(source_map) != dict(payload_source):
        raise GateError("solver implementation source map drifted from solver payload")
    for name, digest in source_map.items():
        _nonempty_string(name, "solver source name")
        _require_sha(digest, f"solver source {name}")
    solver_source_dir = (
        root
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    if set(source_map) != PROFILE_PINS_SOURCE_NAMES:
        raise GateError(
            "solver implementation source map must bind the exact five solver files"
        )
    source_authority = _mapping(
        raw.get("source_authority"), "profile_pins.source_authority"
    )
    source_blob_map_sha256 = sha256_bytes(
        canonical_json_bytes(dict(source_map))
    )
    if (
        set(source_authority)
        != {
            "schema_version",
            "authority",
            "commit_binding",
            "embedded_commit",
            "source_blob_map_sha256",
        }
        or source_authority.get("schema_version") != 1
        or source_authority.get("authority")
        != "external_exact_commit_subset_blob_map_v1"
        or source_authority.get("commit_binding")
        != "external_preexec_immutable_launch_capsule_v1"
        or source_authority.get("embedded_commit") is not False
        or source_authority.get("source_blob_map_sha256")
        != source_blob_map_sha256
    ):
        raise GateError(
            "profile source authority must bind the exact five-file blob map "
            "without embedding a self-referential commit"
        )
    for name, expected_digest in source_map.items():
        source_path = solver_source_dir / name
        if sha256_file(source_path) != expected_digest:
            raise GateError(
                f"solver implementation source drift: {name} no longer matches its pin"
            )

    contact_geometry = _mapping(
        raw.get("contact_geometry"), "profile_pins.contact_geometry"
    )
    solver_contact_geometry = _mapping(
        solver_payload.get("contact_geometry"),
        "solver_payload.contact_geometry",
    )
    if (
        set(contact_geometry) != {"payload", "sha256"}
        or dict(contact_geometry) != dict(solver_contact_geometry)
    ):
        raise GateError(
            "profile and solver payload must bind one exact contact geometry"
        )
    contact_geometry_payload = _mapping(
        contact_geometry.get("payload"), "contact_geometry.payload"
    )
    contact_geometry_sha = _require_sha(
        contact_geometry.get("sha256"), "contact geometry payload SHA"
    )
    if (
        sha256_bytes(canonical_json_bytes(contact_geometry_payload))
        != contact_geometry_sha
        or contact_geometry_sha != RACKET_GEOMETRY_SOURCE_SHA256
    ):
        raise GateError(
            "contact geometry payload seal differs from executable geometry"
        )

    venue_source = _mapping(
        _mapping(physics_payload.get("venue_source"), "physics_payload.venue_source"),
        "physics_payload.venue_source",
    )
    venue_path = _resolve_repo_file(
        venue_source.get("path"),
        "physics venue path",
        repo_root=root,
    )
    venue_sha = _require_sha(venue_source.get("file_sha256"), "physics venue SHA")
    if sha256_file(venue_path) != venue_sha:
        raise GateError("physics venue YAML bytes do not match the pinned physics payload")
    if (
        raw.get("venue_yaml") != venue_source.get("path")
        or raw.get("venue_yaml_sha256") != venue_sha
    ):
        raise GateError(
            "portable venue path/SHA drifted from the physics payload"
        )
    params = _mapping(
        physics_payload.get("virtual_ball_params"),
        "physics_payload.virtual_ball_params",
    )
    grading = _mapping(
        physics_payload.get("geometry_and_grading"),
        "physics_payload.geometry_and_grading",
    )
    return {
        "solver_profile_sha256": solver_digest,
        "physics_profile_sha256": physics_digest,
        "solver_implementation_source_sha256": dict(source_map),
        "source_authority": dict(source_authority),
        "source_blob_map_sha256": source_blob_map_sha256,
        "contact_geometry_sha256": contact_geometry_sha,
        "venue_yaml": {
            "path": str(venue_path),
            "sha256": venue_sha,
        },
        "ball_radius_m": _number(params.get("ball_radius"), "physics ball radius", positive=True),
        "gravity_mps2": _number(params.get("g"), "physics gravity", positive=True),
        "flight_k_d_per_m": _number(params.get("k_d"), "physics k_d", nonnegative=True),
        "flight_k_m": _number(params.get("k_m"), "physics k_m", nonnegative=True),
        "table_surface_z_m": _number(
            grading.get("table_surface_z_m"), "grading table surface"
        ),
        "ball_center_net_top_z_m": _number(
            grading.get("ball_center_net_top_z_m"), "grading net top"
        ),
        "net_x_m": _number(grading.get("net_x_m"), "grading net x"),
        "opponent_near_x_m": _number(
            grading.get("opponent_near_x_m"), "grading opponent near x"
        ),
        "opponent_far_x_m": _number(
            grading.get("opponent_far_x_m"), "grading opponent far x"
        ),
        "minimum_landing_depth_m": _number(
            grading.get("minimum_landing_depth_m"),
            "grading minimum landing depth",
            nonnegative=True,
        ),
        "table_half_width_m": _number(
            grading.get("table_half_width_m"), "grading table half width", positive=True
        ),
    }


def _contact_parameters(value: Any, label: str) -> Dict[str, Any]:
    row = _mapping(value, label)
    friction = _vector(row.get("friction"), 5, f"{label}.friction")
    solref = _vector(row.get("solref"), 2, f"{label}.solref")
    solimp = _vector(row.get("solimp"), 5, f"{label}.solimp")
    if np.any(friction < 0.0):
        raise GateError(f"{label}.friction must be nonnegative")
    condim = _integer(row.get("condim"), f"{label}.condim", positive=True)
    if condim not in (1, 3, 4, 6):
        raise GateError(f"{label}.condim must be one of 1,3,4,6")
    return {
        "friction": friction.tolist(),
        "solref": solref.tolist(),
        "solimp": solimp.tolist(),
        "condim": condim,
        "margin_m": _number(row.get("margin_m", 0.0), f"{label}.margin_m", nonnegative=True),
        "gap_m": _number(row.get("gap_m", 0.0), f"{label}.gap_m", nonnegative=True),
    }


def validate_material_certificate(
    raw: Mapping[str, Any],
    *,
    expected_ball_radius_m: float,
) -> Dict[str, Any]:
    if _integer(raw.get("schema_version"), "material.schema_version") != 1:
        raise GateError("material.schema_version must equal 1")
    if raw.get("certificate_type") != "native_mujoco_pingpong_contact_material_v1":
        raise GateError("material.certificate_type is not the native MuJoCo contact contract")
    authorization = _mapping(raw.get("authorization"), "material.authorization")
    if authorization.get("diagnostic_native_ab") is not True:
        raise GateError("material certificate does not authorize diagnostic_native_ab")
    reviewer = _nonempty_string(authorization.get("reviewed_by"), "material.reviewed_by")
    simulation = _mapping(raw.get("simulation"), "material.simulation")
    timestep = _number(simulation.get("timestep_s"), "material timestep", positive=True)
    integrator = _nonempty_string(simulation.get("integrator"), "material integrator")
    ball = _mapping(raw.get("ball"), "material.ball")
    radius = _number(ball.get("radius_m"), "material ball radius", positive=True)
    if abs(radius - expected_ball_radius_m) > 1.0e-12:
        raise GateError(
            f"material ball radius {radius} != physics profile {expected_ball_radius_m}"
        )
    mass = _number(ball.get("mass_kg"), "material ball mass", positive=True)
    inertia_coeff = _number(
        ball.get("inertia_coeff"), "material ball inertia coefficient", positive=True
    )
    pairs_raw = _mapping(raw.get("contact_pairs"), "material.contact_pairs")
    required_pairs = ("ball_racket", "ball_table", "ball_net")
    if set(pairs_raw) != set(required_pairs):
        raise GateError(
            f"material.contact_pairs must be exact {list(required_pairs)}, got {sorted(pairs_raw)}"
        )
    pairs = {name: _contact_parameters(pairs_raw[name], f"material.{name}") for name in required_pairs}
    calibration = _mapping(raw.get("calibration"), "material.calibration")
    artifact_sha = _require_sha(
        calibration.get("artifact_sha256"), "material calibration artifact SHA"
    )
    if calibration.get("native_ball_racket_contact_verified") is not True:
        raise GateError("material lacks native ball-racket calibration verification")
    if calibration.get("native_ball_table_contact_verified") is not True:
        raise GateError("material lacks native ball-table calibration verification")
    return {
        "certificate_id": _nonempty_string(raw.get("certificate_id"), "material.certificate_id"),
        "reviewed_by": reviewer,
        "mujoco_version": _nonempty_string(
            simulation.get("mujoco_version"), "material MuJoCo version"
        ),
        "timestep_s": timestep,
        "integrator": integrator,
        "ball": {
            "radius_m": radius,
            "mass_kg": mass,
            "inertia_coeff": inertia_coeff,
            "diagonal_inertia_kg_m2": inertia_coeff * mass * radius * radius,
        },
        "contact_pairs": pairs,
        "calibration_artifact_sha256": artifact_sha,
    }


def quaternion_yaw_wxyz(quaternion: Sequence[float]) -> float:
    q = np.asarray(quaternion, np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise GateError("root quaternion must be four finite values")
    norm = float(np.linalg.norm(q))
    if norm < EPS:
        raise GateError("root quaternion is zero")
    w, x, y, z = q / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_yaw(vector: Sequence[float], yaw: float) -> np.ndarray:
    v = np.asarray(vector, np.float64)
    if v.shape != (3,):
        raise GateError("yaw rotation needs a 3-vector")
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray((c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]), np.float64)


def center_ball_state(action: ActionSpec, clip: motion_player.MotionClip) -> Dict[str, Any]:
    yaw = quaternion_yaw_wxyz(clip.body_quat_w[0, 0])
    profile = action.ball_profile
    base = np.r_[
        _vector(profile["base_spawn_center_w_xy_m"], 2, "base_spawn_center"),
        0.0,
    ]
    root_xy = clip.body_pos_w[0, 0, :2]
    if float(np.max(np.abs(root_xy - base[:2]))) > 2.0e-4:
        raise GateError(
            f"{action.action_id}: manifest base spawn {base[:2].tolist()} "
            f"does not match teacher root {root_xy.tolist()}"
        )
    contact = base + rotate_yaw(
        _vector(profile["contact_offset_center_b_yaw_m"], 3, "contact offset"), yaw
    )
    direction = rotate_yaw(
        _unit_vector(profile["incoming_direction_center_b_yaw"], "incoming direction"), yaw
    )
    speed = _number(profile["incoming_speed_center_mps"], "incoming speed", positive=True)
    velocity = direction * speed
    if velocity[0] >= -1.0e-6:
        raise GateError(
            f"{action.action_id}: center ball is not incoming from the opponent (world vx={velocity[0]})"
        )
    spin_axis = rotate_yaw(
        _unit_vector(profile["spin_direction_center_b_yaw"], "spin direction"), yaw
    )
    spin = spin_axis * _number(
        profile["spin_magnitude_center_radps"], "spin magnitude", nonnegative=True
    )
    return {
        "base_yaw_rad": yaw,
        "contact_position_w_m": contact,
        "incoming_velocity_w_mps": velocity,
        "spin_w_radps": spin,
        "time_to_contact_s": _number(
            profile["time_to_contact_center_s"], "time to contact", positive=True
        ),
        "pre_swing_wait_s": _number(
            profile["time_to_contact_center_s"], "time to contact", positive=True
        )
        - action.t_hit_s,
    }


def _format_numbers(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def assemble_physical_scene_xml(
    canonical_xml: bytes,
    *,
    obstacle_rows: Mapping[str, Any],
    material: Mapping[str, Any],
) -> Tuple[bytes, Dict[str, Any]]:
    """Append table/net + thin-shell ball + explicit native contact pairs."""

    table_xml = table_scene.augment_mjcf_xml(
        canonical_xml, obstacle_rows, collidable=True
    )
    try:
        root = ET.fromstring(table_xml)
    except ET.ParseError as exc:
        raise GateError(f"cannot parse table-augmented MJCF: {exc}") from exc
    option = root.find("./option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", format(float(material["timestep_s"]), ".17g"))
    option.set("integrator", str(material["integrator"]))

    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise GateError("vendor MJCF lacks worldbody")
    if root.find(f".//body[@name='{BALL_BODY_NAME}']") is not None:
        raise GateError(f"vendor MJCF already defines {BALL_BODY_NAME}")
    ball = material["ball"]
    body = ET.SubElement(worldbody, "body", {"name": BALL_BODY_NAME, "pos": "0 0 100"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": format(float(ball["mass_kg"]), ".17g"),
            "diaginertia": _format_numbers([ball["diagonal_inertia_kg_m2"]] * 3),
        },
    )
    ET.SubElement(body, "freejoint", {"name": BALL_JOINT_NAME})
    ET.SubElement(
        body,
        "geom",
        {
            "name": BALL_GEOM_NAME,
            "type": "sphere",
            "size": format(float(ball["radius_m"]), ".17g"),
            "rgba": "1 0.5 0 1",
            "contype": "1",
            "conaffinity": "7",
            "condim": "3",
        },
    )
    contact = root.find("./contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    pair_bindings: List[Dict[str, Any]] = []

    def add_pair(pair_name: str, geom1: str, geom2: str, parameters: Mapping[str, Any]) -> None:
        if root.find(f".//geom[@name='{geom2}']") is None:
            raise GateError(f"contact pair {pair_name} references missing geom {geom2!r}")
        attributes = {
            "name": pair_name,
            "geom1": geom1,
            "geom2": geom2,
            "condim": str(int(parameters["condim"])),
            "friction": _format_numbers(parameters["friction"]),
            "solref": _format_numbers(parameters["solref"]),
            "solimp": _format_numbers(parameters["solimp"]),
            "margin": format(float(parameters["margin_m"]), ".17g"),
            "gap": format(float(parameters["gap_m"]), ".17g"),
        }
        ET.SubElement(contact, "pair", attributes)
        pair_bindings.append({"name": pair_name, **attributes})

    add_pair("teacher_ball_racket", BALL_GEOM_NAME, RACKET_GEOM_NAME, material["contact_pairs"]["ball_racket"])
    add_pair("teacher_ball_table", BALL_GEOM_NAME, TABLE_GEOM_NAME, material["contact_pairs"]["ball_table"])
    for net_name in NET_GEOM_NAMES:
        add_pair(
            f"teacher_ball_{net_name}",
            BALL_GEOM_NAME,
            net_name,
            material["contact_pairs"]["ball_net"],
        )
    final_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return final_xml, {
        "canonical_xml_sha256": sha256_bytes(canonical_xml),
        "table_augmented_xml_sha256": sha256_bytes(table_xml),
        "physical_scene_xml_sha256": sha256_bytes(final_xml),
        "obstacle_geometry_sha256": sha256_bytes(canonical_json_bytes(obstacle_rows)),
        "ball": dict(ball),
        "contact_pairs": pair_bindings,
    }


def reverse_free_flight(
    contact_position: Sequence[float],
    contact_velocity: Sequence[float],
    spin: Sequence[float],
    *,
    duration_s: float,
    k_d: float,
    k_m: float,
    gravity: float,
    max_step_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse the final collision-free incoming segment with RK4.

    This is initial-state construction only.  It never predicts or grades the
    post-racket return.  Any native table contact during this segment is
    retained and reported.
    """

    p = np.asarray(contact_position, np.float64).copy()
    v = np.asarray(contact_velocity, np.float64).copy()
    omega = np.asarray(spin, np.float64)
    duration = _number(duration_s, "precontact flight duration", nonnegative=True)
    if duration == 0.0:
        return p, v
    n_steps = max(1, int(math.ceil(duration / max_step_s)))
    h = -duration / n_steps

    def derivative(pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        del pos
        acceleration = np.asarray((0.0, 0.0, -gravity), np.float64)
        acceleration = acceleration - k_d * float(np.linalg.norm(vel)) * vel
        acceleration = acceleration + k_m * np.cross(omega, vel)
        return vel, acceleration

    for _ in range(n_steps):
        k1p, k1v = derivative(p, v)
        k2p, k2v = derivative(p + 0.5 * h * k1p, v + 0.5 * h * k1v)
        k3p, k3v = derivative(p + 0.5 * h * k2p, v + 0.5 * h * k2v)
        k4p, k4v = derivative(p + h * k3p, v + h * k3v)
        p = p + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    if not (np.isfinite(p).all() and np.isfinite(v).all()):
        raise GateError("reverse incoming-flight integration produced NaN/Inf")
    return p, v


def slerp_wxyz(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    a = np.asarray(q0, np.float64)
    b = np.asarray(q1, np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + alpha * (b - a)
        return out / np.linalg.norm(out)
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sine = math.sin(angle)
    return (
        math.sin((1.0 - alpha) * angle) / sine * a
        + math.sin(alpha * angle) / sine * b
    )


def interpolate_teacher(
    clip: motion_player.MotionClip,
    motion_time_s: float,
) -> Dict[str, Any]:
    time = float(np.clip(motion_time_s, 0.0, (clip.n_frames - 1) / clip.fps))
    position = time * clip.fps
    lo = int(math.floor(position))
    hi = min(lo + 1, clip.n_frames - 1)
    alpha = position - lo

    def lerp(values: np.ndarray) -> np.ndarray:
        return (1.0 - alpha) * values[lo] + alpha * values[hi]

    return {
        "root_pos": lerp(clip.body_pos_w[:, 0]),
        "root_quat": slerp_wxyz(clip.body_quat_w[lo, 0], clip.body_quat_w[hi, 0], alpha),
        "root_lin_vel": lerp(clip.body_lin_vel_w[:, 0]),
        "root_ang_vel": lerp(clip.body_ang_vel_w[:, 0]),
        "joint_pos": lerp(clip.joint_pos),
        "joint_vel": lerp(clip.joint_vel),
        "body_lin_vel_point": clip.body_lin_vel_point,
    }


def _set_teacher_state(
    mujoco: Any,
    model: Any,
    data: Any,
    binding: motion_player.ModelBinding,
    state: Mapping[str, Any],
) -> None:
    root = binding.root_qpos_adr
    data.qpos[root : root + 3] = state["root_pos"]
    data.qpos[root + 3 : root + 7] = state["root_quat"]
    data.qpos[binding.joint_qpos_adrs] = state["joint_pos"]
    # Only overwrite robot dofs; the dynamic ball free-joint velocity is state.
    root_slice = slice(binding.root_dof_adr, binding.root_dof_adr + 6)
    data.qvel[root_slice] = 0.0
    data.qvel[binding.joint_dof_adrs] = state["joint_vel"]
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, int(model.nv)), np.float64)
    jacr = np.zeros((3, int(model.nv)), np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, int(binding.body_ids[0]))
    matrix = np.vstack((jacp[:, root_slice], jacr[:, root_slice]))
    nonroot = np.concatenate((jacp @ data.qvel, jacr @ data.qvel))
    root_linear = np.asarray(state["root_lin_vel"], np.float64).copy()
    root_angular = np.asarray(state["root_ang_vel"], np.float64)
    if state.get("body_lin_vel_point") == motion_player.BODY_LIN_VEL_POINT:
        root_offset_w = (
            np.asarray(data.xipos[binding.body_ids[0]], np.float64)
            - np.asarray(data.xpos[binding.body_ids[0]], np.float64)
        )
        root_linear = root_linear - np.cross(root_angular, root_offset_w)
    target = np.concatenate((root_linear, root_angular))
    data.qvel[root_slice] = np.linalg.solve(matrix, target - nonroot)
    mujoco.mj_forward(model, data)


def _root_tilt_rad(quaternion_wxyz: Sequence[float]) -> float:
    rotation = motion_player.quaternion_wxyz_to_matrix(np.asarray(quaternion_wxyz, np.float64))
    return math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))


def _site_twist(
    mujoco: Any,
    model: Any,
    data: Any,
    site_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, int(model.nv)), np.float64)
    jacr = np.zeros((3, int(model.nv)), np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, int(site_id))
    qvel = np.asarray(data.qvel, np.float64)
    return jacp @ qvel, jacr @ qvel


@dataclass
class NativeEvents:
    racket_contact_time_s: Optional[float] = None
    racket_contact_position_m: Optional[List[float]] = None
    racket_speed_mps: Optional[float] = None
    physical_face_center_speed_mps: Optional[float] = None
    physical_face_center_position_m: Optional[List[float]] = None
    incoming_ball_position_m: Optional[List[float]] = None
    incoming_ball_velocity_mps: Optional[List[float]] = None
    contact_impulse_ns: float = 0.0
    contact_peak_force_n: float = 0.0
    racket_contact_steps: int = 0
    net_crossing: Optional[Dict[str, float]] = None
    first_landing: Optional[Dict[str, Any]] = None
    incoming_table_contacts: int = 0
    ball_net_contacts: int = 0
    ball_other_robot_contacts: List[Dict[str, Any]] = field(default_factory=list)
    robot_obstacle_contacts: List[Dict[str, Any]] = field(default_factory=list)
    self_contacts: List[Dict[str, Any]] = field(default_factory=list)
    joint_limit_violations: List[Dict[str, Any]] = field(default_factory=list)
    fall: Optional[Dict[str, Any]] = None


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return str(value) if value else f"geom_{int(geom_id)}"


def evaluate_failure_reasons(
    *,
    events: NativeEvents,
    target_contact_position: Sequence[float],
    target_incoming_velocity: Sequence[float],
    expected_global_contact_time_s: float,
    contact_time_tolerance_s: float,
    contact_position_tolerance_m: float,
    incoming_velocity_tolerance_mps: float,
    opponent_near_x_m: float,
    opponent_far_x_m: float,
    table_half_width_m: float,
) -> List[str]:
    reasons: List[str] = []
    if events.racket_contact_time_s is None:
        reasons.append("no_native_ball_racket_contact")
    else:
        if abs(events.racket_contact_time_s - expected_global_contact_time_s) > contact_time_tolerance_s:
            reasons.append("native_contact_time_mismatch")
        if events.racket_contact_position_m is None or float(
            np.linalg.norm(
                np.asarray(events.racket_contact_position_m) - np.asarray(target_contact_position)
            )
        ) > contact_position_tolerance_m:
            reasons.append("native_contact_position_mismatch")
        if events.incoming_ball_velocity_mps is None or float(
            np.linalg.norm(
                np.asarray(events.incoming_ball_velocity_mps)
                - np.asarray(target_incoming_velocity)
            )
        ) > incoming_velocity_tolerance_mps:
            reasons.append("native_incoming_velocity_mismatch")
        if not math.isfinite(events.contact_impulse_ns) or events.contact_impulse_ns <= 0.0:
            reasons.append("native_contact_impulse_nonpositive")
    if events.net_crossing is None:
        reasons.append("no_post_hit_net_crossing")
    elif not bool(events.net_crossing.get("cleared", False)):
        reasons.append("net_not_cleared")
    if events.ball_net_contacts:
        reasons.append("ball_hit_net_or_post")
    if events.first_landing is None:
        reasons.append("no_native_first_table_landing")
    else:
        x = float(events.first_landing["ball_center_m"][0])
        y = float(events.first_landing["ball_center_m"][1])
        if not (
            opponent_near_x_m <= x <= opponent_far_x_m
            and abs(y) <= table_half_width_m
        ):
            reasons.append("first_landing_outside_opponent_table")
        if not bool(events.first_landing.get("descending", False)):
            reasons.append("first_landing_not_descending")
    if events.ball_other_robot_contacts:
        reasons.append("ball_hit_non_racket_robot_geom")
    if events.robot_obstacle_contacts:
        reasons.append("robot_hit_table_edge_or_net")
    if events.self_contacts:
        reasons.append("robot_self_contact")
    if events.joint_limit_violations:
        reasons.append("joint_limit_violation")
    if events.fall is not None:
        reasons.append("fall")
    return reasons


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _render_video(
    frames: Sequence[np.ndarray],
    path: Path,
    fps: int,
) -> Dict[str, Any]:
    if not frames:
        return {"status": "NOT_REQUESTED_OR_NO_FRAMES", "path": None}
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        return {
            "status": "FAILED",
            "error": "video output must use the .mp4 suffix",
            "path": None,
        }
    if not path.parent.is_dir() or path.parent.is_symlink():
        return {
            "status": "FAILED",
            "error": "video parent must be an existing plain directory",
            "path": None,
        }
    if os.path.lexists(path):
        return {
            "status": "FAILED",
            "error": "refusing to overwrite an existing video path",
            "path": None,
        }
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        return {"status": "UNAVAILABLE", "error": f"imageio import failed: {exc}", "path": None}
    try:
        imageio.mimwrite(
            str(path),
            list(frames),
            fps=int(fps),
            codec="libx264",
            macro_block_size=1,
        )
    except Exception as exc:  # pragma: no cover - depends on ffmpeg runtime
        return {"status": "FAILED", "error": str(exc), "path": None}
    metadata = os.lstat(path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size <= 0
    ):
        return {
            "status": "FAILED",
            "error": "video encoder did not produce a nonempty regular file",
            "path": None,
        }
    return {
        "status": "WRITTEN",
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "frames": len(frames),
        "fps": int(fps),
    }


def run_one_action(
    *,
    mujoco: Any,
    model: Any,
    scene_contract: Mapping[str, Any],
    action: ActionSpec,
    clip: motion_player.MotionClip,
    center: Mapping[str, Any],
    profile: Mapping[str, Any],
    material: Mapping[str, Any],
    post_contact_s: float,
    precontact_flight_s: float,
    contact_time_tolerance_s: float,
    contact_position_tolerance_m: float,
    incoming_velocity_tolerance_mps: float,
    render_path: Optional[Path],
    render_fps: int,
) -> Dict[str, Any]:
    binding = motion_player.bind_model(mujoco, model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    ball_joint_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT_NAME)
    )
    ball_geom_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME)
    )
    ball_body_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY_NAME)
    )
    if min(ball_joint_id, ball_geom_id, ball_body_id) < 0:
        raise GateError("compiled physical scene is missing the dynamic ball")
    ball_qpos = int(model.jnt_qposadr[ball_joint_id])
    ball_dof = int(model.jnt_dofadr[ball_joint_id])
    table_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)): name
        for name in table_scene.OBSTACLE_NAMES
    }
    if any(value < 0 for value in table_ids):
        raise GateError("compiled physical scene is missing table/net geometry")
    racket_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RACKET_GEOM_NAME)
    )
    if racket_id < 0:
        raise GateError("compiled physical scene is missing the racket collision geom")
    floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    robot_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) != 0
        and geom_id != ball_geom_id
    }
    robot_body_ids = {int(model.geom_bodyid[g]) for g in robot_geom_ids}
    dt = float(model.opt.timestep)
    if abs(dt - float(material["timestep_s"])) > 1.0e-15:
        raise GateError("compiled model timestep drifted from material certificate")
    if str(getattr(mujoco, "__version__", "")) != str(material["mujoco_version"]):
        raise GateError(
            f"MuJoCo version {getattr(mujoco, '__version__', '')!r} "
            f"!= material certificate {material['mujoco_version']!r}"
        )
    clip_duration = (clip.n_frames - 1) / clip.fps
    if abs(clip_duration - action.t_cycle_s) > 1.0 / clip.fps + 1.0e-9:
        raise GateError(
            f"{action.action_id}: motion duration {clip_duration} does not bind "
            f"manifest t_cycle {action.t_cycle_s}"
        )
    expected_contact = float(center["time_to_contact_s"])
    wait = float(center["pre_swing_wait_s"])
    total_time = max(wait + clip_duration, expected_contact + post_contact_s)
    flight_duration = min(precontact_flight_s, expected_contact)
    activation_time = expected_contact - flight_duration
    birth_pos, birth_vel = reverse_free_flight(
        center["contact_position_w_m"],
        center["incoming_velocity_w_mps"],
        center["spin_w_radps"],
        duration_s=flight_duration,
        k_d=float(profile["flight_k_d_per_m"]),
        k_m=float(profile["flight_k_m"]),
        gravity=float(profile["gravity_mps2"]),
        max_step_s=min(dt, 5.0e-4),
    )
    data.qpos[ball_qpos : ball_qpos + 3] = np.asarray((0.0, 0.0, 100.0))
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = np.asarray((1.0, 0.0, 0.0, 0.0))
    data.qvel[ball_dof : ball_dof + 6] = 0.0

    events = NativeEvents()
    trajectory: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    renderer = None
    render_stride = max(1, int(round(1.0 / (render_fps * dt))))
    if render_path is not None:
        try:
            renderer = mujoco.Renderer(model, height=720, width=960)
        except Exception as exc:  # pragma: no cover - GL availability is host-specific
            renderer = None
            render_error = str(exc)
        else:
            render_error = None
    else:
        render_error = None

    active = False
    previous_ball_pos: Optional[np.ndarray] = None
    previous_ball_vel: Optional[np.ndarray] = None
    post_hit = False
    max_steps = int(math.ceil(total_time / dt)) + 2
    for step in range(max_steps):
        sim_time = step * dt
        motion_time = sim_time - wait
        teacher = interpolate_teacher(clip, motion_time)
        _set_teacher_state(mujoco, model, data, binding, teacher)
        if not active and sim_time + 0.5 * dt >= activation_time:
            data.qpos[ball_qpos : ball_qpos + 3] = birth_pos
            data.qpos[ball_qpos + 3 : ball_qpos + 7] = np.asarray((1.0, 0.0, 0.0, 0.0))
            data.qvel[ball_dof : ball_dof + 3] = birth_vel
            data.qvel[ball_dof + 3 : ball_dof + 6] = center["spin_w_radps"]
            active = True
            mujoco.mj_forward(model, data)

        # Reference safety is evaluated before dynamics can perturb the prescribed pose.
        root_z = float(data.qpos[binding.root_qpos_adr + 2])
        root_tilt = _root_tilt_rad(data.qpos[binding.root_qpos_adr + 3 : binding.root_qpos_adr + 7])
        if events.fall is None and (root_z < ROOT_Z_FALL_M or root_tilt > ROOT_TILT_FALL_RAD):
            events.fall = {
                "time_s": sim_time,
                "root_z_m": root_z,
                "root_tilt_rad": root_tilt,
            }
        q_values = np.asarray(data.qpos)[binding.joint_qpos_adrs]
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in motion_player.RUNTIME_JOINT_NAMES
            ],
            np.int64,
        )
        limited = np.asarray(model.jnt_limited)[joint_ids].astype(bool)
        ranges = np.asarray(model.jnt_range)[joint_ids]
        bad = np.flatnonzero(
            limited
            & ((q_values < ranges[:, 0] - 1.0e-7) | (q_values > ranges[:, 1] + 1.0e-7))
        )
        if bad.size and not events.joint_limit_violations:
            events.joint_limit_violations.append(
                {
                    "time_s": sim_time,
                    "joints": [motion_player.RUNTIME_JOINT_NAMES[int(i)] for i in bad],
                }
            )

        if active:
            ball_vel = np.asarray(data.qvel[ball_dof : ball_dof + 3], np.float64)
            omega = np.asarray(data.qvel[ball_dof + 3 : ball_dof + 6], np.float64)
            acceleration = (
                -float(profile["flight_k_d_per_m"]) * float(np.linalg.norm(ball_vel)) * ball_vel
                + float(profile["flight_k_m"]) * np.cross(omega, ball_vel)
            )
            data.xfrc_applied[ball_body_id, :3] = float(material["ball"]["mass_kg"]) * acceleration
        else:
            data.xfrc_applied[ball_body_id, :] = 0.0

        if active:
            previous_ball_pos = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], np.float64).copy()
            previous_ball_vel = np.asarray(data.qvel[ball_dof : ball_dof + 3], np.float64).copy()
        mujoco.mj_step(model, data)
        after_time = float(data.time)
        ball_pos = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], np.float64).copy()
        ball_vel = np.asarray(data.qvel[ball_dof : ball_dof + 3], np.float64).copy()

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            pair = {g1, g2}
            names = (_geom_name(mujoco, model, g1), _geom_name(mujoco, model, g2))
            if ball_geom_id in pair:
                other = g2 if g1 == ball_geom_id else g1
                if other == racket_id:
                    force = np.zeros(6, np.float64)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    normal_force = abs(float(force[0]))
                    events.contact_impulse_ns += normal_force * dt
                    events.contact_peak_force_n = max(events.contact_peak_force_n, normal_force)
                    events.racket_contact_steps += 1
                    if events.racket_contact_time_s is None:
                        events.racket_contact_time_s = after_time
                        events.racket_contact_position_m = ball_pos.tolist()
                        events.incoming_ball_position_m = (
                            previous_ball_pos.tolist() if previous_ball_pos is not None else None
                        )
                        events.incoming_ball_velocity_mps = (
                            previous_ball_vel.tolist() if previous_ball_vel is not None else None
                        )
                        site_linear, site_angular = _site_twist(
                            mujoco, model, data, binding.racket_site_id
                        )
                        events.racket_speed_mps = float(np.linalg.norm(site_linear))
                        site_rotation = np.asarray(
                            data.site_xmat[binding.racket_site_id], np.float64
                        ).reshape(3, 3)
                        face_offset_w = site_rotation @ (
                            racket_geometry.face_center_from_site_local(
                                action.mount_normal_sign
                            )
                        )
                        face_linear = racket_geometry.rigid_point_velocity(
                            site_linear, site_angular, face_offset_w
                        )
                        events.physical_face_center_speed_mps = float(
                            np.linalg.norm(face_linear)
                        )
                        events.physical_face_center_position_m = (
                            np.asarray(
                                data.site_xpos[binding.racket_site_id],
                                np.float64,
                            )
                            + face_offset_w
                        ).tolist()
                        post_hit = True
                elif other in table_ids:
                    obstacle = table_ids[other]
                    if obstacle == TABLE_GEOM_NAME:
                        if not post_hit:
                            events.incoming_table_contacts += 1
                        elif events.first_landing is None:
                            events.first_landing = {
                                "time_s": after_time,
                                "ball_center_m": ball_pos.tolist(),
                                "ball_velocity_mps": ball_vel.tolist(),
                                "descending": bool(
                                    previous_ball_vel is not None and previous_ball_vel[2] < 0.0
                                ),
                                "native_contact": True,
                            }
                    else:
                        events.ball_net_contacts += 1
                elif other in robot_geom_ids:
                    events.ball_other_robot_contacts.append(
                        {"time_s": after_time, "other_geom": _geom_name(mujoco, model, other)}
                    )
            else:
                world_obstacle = (
                    g1 in table_ids or g2 in table_ids
                )
                if world_obstacle:
                    other = g2 if g1 in table_ids else g1
                    if other in robot_geom_ids:
                        record = {
                            "time_s": after_time,
                            "robot_geom": _geom_name(mujoco, model, other),
                            "obstacle": table_ids[g1] if g1 in table_ids else table_ids[g2],
                        }
                        if not events.robot_obstacle_contacts or events.robot_obstacle_contacts[-1] != record:
                            events.robot_obstacle_contacts.append(record)
                elif g1 in robot_geom_ids and g2 in robot_geom_ids:
                    b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                    if b1 != b2:
                        record = {"time_s": after_time, "geoms": list(names)}
                        if not events.self_contacts or events.self_contacts[-1] != record:
                            events.self_contacts.append(record)

        if (
            post_hit
            and events.net_crossing is None
            and previous_ball_pos is not None
            and previous_ball_pos[0] <= float(profile["net_x_m"]) < ball_pos[0]
        ):
            alpha = (
                float(profile["net_x_m"]) - previous_ball_pos[0]
            ) / max(ball_pos[0] - previous_ball_pos[0], EPS)
            crossing = previous_ball_pos + alpha * (ball_pos - previous_ball_pos)
            events.net_crossing = {
                "time_s": after_time - dt + alpha * dt,
                "x_m": float(profile["net_x_m"]),
                "y_m": float(crossing[1]),
                "ball_center_z_m": float(crossing[2]),
                "required_ball_center_z_m": float(profile["ball_center_net_top_z_m"]),
                "clearance_margin_m": float(
                    crossing[2] - float(profile["ball_center_net_top_z_m"])
                ),
                "cleared": bool(
                    crossing[2] > float(profile["ball_center_net_top_z_m"])
                    and abs(crossing[1]) <= float(profile["table_half_width_m"])
                ),
            }

        if active and (step % max(1, int(round(0.01 / dt))) == 0):
            trajectory.append(
                {
                    "time_s": after_time,
                    "ball_position_m": ball_pos.tolist(),
                    "ball_velocity_mps": ball_vel.tolist(),
                    "racket_position_m": np.asarray(
                        data.site_xpos[binding.racket_site_id], np.float64
                    ).tolist(),
                    "root_z_m": root_z,
                    "root_tilt_rad": root_tilt,
                }
            )
        if renderer is not None and step % render_stride == 0:
            try:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())
            except Exception as exc:  # pragma: no cover - GL availability is host-specific
                render_error = str(exc)
                renderer = None
        if after_time >= total_time - 0.5 * dt:
            break

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    video = (
        _render_video(frames, render_path, render_fps)
        if render_path is not None and render_error is None
        else {
            "status": "UNAVAILABLE" if render_path is not None else "NOT_REQUESTED",
            "error": render_error,
            "path": None,
        }
    )
    reasons = evaluate_failure_reasons(
        events=events,
        target_contact_position=center["contact_position_w_m"],
        target_incoming_velocity=center["incoming_velocity_w_mps"],
        expected_global_contact_time_s=expected_contact,
        contact_time_tolerance_s=contact_time_tolerance_s,
        contact_position_tolerance_m=contact_position_tolerance_m,
        incoming_velocity_tolerance_mps=incoming_velocity_tolerance_mps,
        opponent_near_x_m=float(profile["opponent_near_x_m"]),
        opponent_far_x_m=float(profile["opponent_far_x_m"]),
        table_half_width_m=float(profile["table_half_width_m"]),
    )
    contact_time = events.racket_contact_time_s
    return {
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion": {
            "path": str(action.motion_path),
            "sha256": action.motion_sha256,
            "frames": clip.n_frames,
            "fps": clip.fps,
        },
        "center_ball": {
            "contact_position_w_m": np.asarray(center["contact_position_w_m"]).tolist(),
            "incoming_velocity_w_mps": np.asarray(center["incoming_velocity_w_mps"]).tolist(),
            "spin_w_radps": np.asarray(center["spin_w_radps"]).tolist(),
            "time_to_contact_s": expected_contact,
            "pre_swing_wait_s": wait,
            "activation_time_s": activation_time,
            "native_precontact_flight_s": flight_duration,
            "birth_position_w_m": birth_pos.tolist(),
            "birth_velocity_w_mps": birth_vel.tolist(),
            "prebounce_reconstruction_scope": (
                "final_collision_free_segment_only; no recorded/native pre-bounce birth artifact"
            ),
        },
        "timing": {
            "reference_t_hit_s": action.t_hit_s,
            "reference_t_cycle_s": action.t_cycle_s,
            "motion_duration_s": clip_duration,
            "native_contact_global_time_s": contact_time,
            "native_contact_motion_time_s": (
                None if contact_time is None else contact_time - wait
            ),
        },
        "racket": {
            "mount_normal_sign": action.mount_normal_sign,
            "reference_site_speed_mps": action.racket_speed_mps,
            "physical_site_speed_at_contact_mps": events.racket_speed_mps,
            "physical_face_center_speed_at_contact_mps": (
                events.physical_face_center_speed_mps
            ),
            "physical_face_center_position_at_contact_m": (
                events.physical_face_center_position_m
            ),
            "site_to_ball_center_local_m": (
                racket_geometry.ball_center_from_site_local(
                    action.mount_normal_sign
                ).tolist()
            ),
            "legacy_site_ball_colocation_error_m": (
                racket_geometry.legacy_colocation_error_m(
                    action.mount_normal_sign
                )
            ),
            "native_contact_steps": events.racket_contact_steps,
            "native_contact_peak_force_n": events.contact_peak_force_n,
            "native_contact_impulse_ns": events.contact_impulse_ns,
        },
        "net_crossing": events.net_crossing,
        "first_landing": events.first_landing,
        "safety": {
            "incoming_ball_table_contacts": events.incoming_table_contacts,
            "ball_net_or_post_contacts": events.ball_net_contacts,
            "ball_other_robot_contacts": events.ball_other_robot_contacts[:20],
            "robot_table_edge_net_contacts": events.robot_obstacle_contacts[:20],
            "self_contacts": events.self_contacts[:20],
            "joint_limit_violations": events.joint_limit_violations[:20],
            "fall": events.fall,
        },
        "trajectory_100hz": trajectory,
        "video": video,
        "scene_contract_sha256": scene_contract["physical_scene_xml_sha256"],
        "verdict": "DIAGNOSTIC_PASS" if not reasons else "DIAGNOSTIC_FAIL",
        "failure_reasons": reasons,
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }


def _preflight(
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any], Optional[ManifestContract], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    blockers: List[str] = []
    evidence: Dict[str, Any] = {}
    manifest: Optional[ManifestContract] = None
    profile: Optional[Dict[str, Any]] = None
    material: Optional[Dict[str, Any]] = None
    manifest_raw: Dict[str, Any] = {}
    if not args.manifest_sha256:
        blockers.append("missing_expected_manifest_sha256")
    else:
        try:
            manifest_raw, manifest_receipt = read_json_exact(
                args.manifest,
                "action-ball manifest",
                expected_sha256=args.manifest_sha256,
            )
            manifest = validate_manifest(
                manifest_raw, expected_actions=args.expected_actions
            )
            evidence["manifest"] = manifest_receipt
        except GateError as exc:
            blockers.append(f"manifest:{exc}")
            if isinstance(manifest_raw, dict):
                if "racket_geometry_contract" not in manifest_raw:
                    blockers.append(
                        "manifest:missing_versioned_physical_racket_geometry_contract"
                    )
                order = manifest_raw.get("action_order")
                if isinstance(order, list) and "fh_loop" in order:
                    blockers.append(
                        "manifest:contains_retired_old_forehand_fh_loop"
                    )
    if not args.profile_pins_sha256:
        blockers.append("missing_expected_profile_pins_sha256")
    else:
        try:
            pins_raw, pins_receipt = read_json_exact(
                args.profile_pins,
                "action-ball profile pins",
                expected_sha256=args.profile_pins_sha256,
            )
            profile = validate_profile_pins(pins_raw, manifest)
            evidence["profile_pins"] = pins_receipt
        except GateError as exc:
            blockers.append(f"profile_pins:{exc}")
    try:
        identity_raw, identity_receipt = read_json_exact(
            args.identity_manifest,
            "MuJoCo identity manifest",
            expected_sha256=args.identity_manifest_sha256,
        )
        expected = _mapping(identity_raw.get("expected"), "identity.expected")
        actual_mjcf_sha = sha256_file(args.mjcf)
        if actual_mjcf_sha != _require_sha(
            expected.get("root_mjcf_sha256"), "identity root MJCF SHA"
        ):
            raise GateError("vendor MJCF bytes do not match the identity manifest")
        evidence["mujoco_identity_manifest"] = identity_receipt
        evidence["vendor_mjcf"] = {
            "path": str(Path(args.mjcf).expanduser().resolve()),
            "sha256": actual_mjcf_sha,
            "portable_identity_sha256": expected.get("portable_identity_sha256"),
            "source_closure_sha256": expected.get("source_closure_sha256"),
        }
    except (GateError, OSError) as exc:
        blockers.append(f"mujoco_identity:{exc}")
    if not args.material_certificate or not args.material_certificate_sha256:
        blockers.append("missing_pre_registered_native_mujoco_material_certificate")
    elif profile is not None:
        try:
            material_raw, material_receipt = read_json_exact(
                args.material_certificate,
                "native MuJoCo material certificate",
                expected_sha256=args.material_certificate_sha256,
            )
            material = validate_material_certificate(
                material_raw,
                expected_ball_radius_m=float(profile["ball_radius_m"]),
            )
            evidence["material_certificate"] = material_receipt
        except GateError as exc:
            blockers.append(f"material_certificate:{exc}")
    return blockers, evidence, manifest, profile, material


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--profile-pins", type=Path, default=DEFAULT_PROFILE_PINS)
    parser.add_argument("--profile-pins-sha256", default="")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--identity-manifest", type=Path, default=DEFAULT_IDENTITY_MANIFEST)
    parser.add_argument(
        "--identity-manifest-sha256",
        default=DEFAULT_IDENTITY_MANIFEST_SHA256,
    )
    parser.add_argument("--material-certificate", type=Path)
    parser.add_argument("--material-certificate-sha256", default="")
    parser.add_argument("--expected-actions", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--post-contact-s", type=float, default=1.5)
    parser.add_argument(
        "--precontact-flight-s",
        type=float,
        default=0.25,
        help="native final incoming segment; full pre-bounce birth needs a separate artifact",
    )
    parser.add_argument("--contact-time-tolerance-s", type=float, default=0.04)
    parser.add_argument("--contact-position-tolerance-m", type=float, default=0.04)
    parser.add_argument("--incoming-velocity-tolerance-mps", type=float, default=0.35)
    parser.add_argument(
        "--shared-ready-joint-linf-rad",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--recovery-joint-linf-rad",
        type=float,
        default=1.0e-6,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.expected_actions <= 0:
        print("[teacher-ball-gate][FATAL] --expected-actions must be positive", file=sys.stderr)
        return 2
    numeric_args = {
        "post-contact-s": args.post_contact_s,
        "precontact-flight-s": args.precontact_flight_s,
        "contact-time-tolerance-s": args.contact_time_tolerance_s,
        "contact-position-tolerance-m": args.contact_position_tolerance_m,
        "incoming-velocity-tolerance-mps": args.incoming_velocity_tolerance_mps,
        "shared-ready-joint-linf-rad": args.shared_ready_joint_linf_rad,
        "recovery-joint-linf-rad": args.recovery_joint_linf_rad,
    }
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in numeric_args.values()
    ):
        print(
            "[teacher-ball-gate][FATAL] durations and tolerances must be finite and nonnegative",
            file=sys.stderr,
        )
        return 2
    if args.render_fps <= 0:
        print("[teacher-ball-gate][FATAL] --render-fps must be positive", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"[teacher-ball-gate][FATAL] refusing to overwrite {args.out}", file=sys.stderr)
        return 2
    blockers, evidence, manifest, profile, material = _preflight(args)
    base_receipt: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "mujoco_teacher_motion_native_ball_diagnostic",
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "contact_authority": "native_mujoco_mj_step_mj_contactForce",
        "landing_authority": "native_ball_motion_table_top_contact",
        "selector_executed": False,
        "ball_to_task_solver_executed": False,
        "analytic_or_counterfactual_return_scorer_executed": False,
        "formal_venue_physics_gate": {
            "status": "BLOCKED",
            "blocker": (
                "frozen venue-fitted swept-selected-face contact referee is "
                "not implemented here; native MuJoCo material response is "
                "diagnostic A/B only"
            ),
            "native_ball_racket_impulse_must_be_disabled_in_formal_referee": True,
        },
        "expected_actions": args.expected_actions,
        "preflight": {
            "status": "BLOCKED" if blockers else "PASS",
            "blockers": blockers,
            "evidence": evidence,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }
    if args.preflight_only or blockers:
        base_receipt["status"] = "BLOCKED" if blockers else "DIAGNOSTIC_PREFLIGHT_PASS"
        base_receipt["verdict"] = "BLOCKED" if blockers else "NOT_RUN"
        _write_json_no_clobber(args.out, base_receipt)
        print(
            f"[teacher-ball-gate] {base_receipt['status']} receipt={args.out} "
            f"blockers={len(blockers)}"
        )
        return 3 if blockers else 0
    assert manifest is not None and profile is not None and material is not None
    if args.render_dir is not None:
        render_dir = args.render_dir.expanduser().resolve()
        try:
            render_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(
                f"[teacher-ball-gate][FATAL] refusing existing render dir {render_dir}",
                file=sys.stderr,
            )
            return 2
    else:
        render_dir = None
    try:
        import mujoco
        from canonical_mujoco_identity import verify_exact_mujoco_identity

        verified = verify_exact_mujoco_identity(
            mjcf_path=args.mjcf,
            expected_manifest_path=args.identity_manifest,
            trusted_expected_manifest_sha256=args.identity_manifest_sha256,
        )
        canonical_xml = Path(args.mjcf).read_bytes()
        obstacles = table_scene.obstacle_geometry()
        physical_xml, scene_contract = assemble_physical_scene_xml(
            canonical_xml,
            obstacle_rows=obstacles,
            material=material,
        )
        assets = table_scene._mesh_assets(canonical_xml, Path(args.mjcf).resolve().parent)
        model = mujoco.MjModel.from_xml_string(physical_xml.decode("utf-8"), assets=assets)
        if abs(float(model.opt.gravity[2]) + float(profile["gravity_mps2"])) > 1.0e-12:
            raise GateError("compiled scene gravity drifted from the physics profile")
        scene_contract = {
            **scene_contract,
            "base_portable_identity_sha256": verified.portable_identity_sha256,
            "mujoco_version": str(mujoco.__version__),
            "timestep_s": float(model.opt.timestep),
            "gravity_mps2": np.asarray(model.opt.gravity, np.float64).tolist(),
            "racket_geometry": {
                "geom_name": RACKET_GEOM_NAME,
                "geom_id": int(
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, RACKET_GEOM_NAME
                    )
                ),
                "type": "mesh",
                "mesh_name": "collision_right_racket_face",
                "mesh_asset_sha256": sha256_bytes(
                    assets["meshes/collision_optimized/right_racket_face_collision.STL"]
                ),
            },
        }
        clips = {
            action.action_id: motion_player.load_motion(action.motion_path)
            for action in manifest.actions
        }
        starts = np.stack([clips[a.action_id].joint_pos[0] for a in manifest.actions])
        shared_ready_linf = float(np.max(np.ptp(starts, axis=0)))
        shared_ready_pass = (
            shared_ready_linf <= float(args.shared_ready_joint_linf_rad)
        )
        actions_out: List[Dict[str, Any]] = []
        for action in manifest.actions:
            clip = clips[action.action_id]
            center = center_ball_state(action, clip)
            render_path = (
                None
                if render_dir is None
                else render_dir / f"{action.action_id}_teacher_native_ball.mp4"
            )
            result = run_one_action(
                mujoco=mujoco,
                model=model,
                scene_contract=scene_contract,
                action=action,
                clip=clip,
                center=center,
                profile=profile,
                material=material,
                post_contact_s=args.post_contact_s,
                precontact_flight_s=args.precontact_flight_s,
                contact_time_tolerance_s=args.contact_time_tolerance_s,
                contact_position_tolerance_m=args.contact_position_tolerance_m,
                incoming_velocity_tolerance_mps=args.incoming_velocity_tolerance_mps,
                render_path=render_path,
                render_fps=args.render_fps,
            )
            recovery = float(np.max(np.abs(clip.joint_pos[-1] - clip.joint_pos[0])))
            root_position_recovery = float(
                np.max(
                    np.abs(
                        clip.body_pos_w[-1, 0] - clip.body_pos_w[0, 0]
                    )
                )
            )
            root_orientation_recovery = float(
                np.max(
                    np.abs(
                        clip.body_quat_w[-1, 0]
                        - clip.body_quat_w[0, 0]
                    )
                )
            )
            recovery_pass = (
                recovery <= float(args.recovery_joint_linf_rad)
                and root_position_recovery <= 1.0e-6
                and root_orientation_recovery <= 1.0e-6
            )
            result["shared_ready_recovery"] = {
                "bank_start_joint_linf_rad": shared_ready_linf,
                "action_end_to_start_joint_linf_rad": recovery,
                "action_root_end_to_start_position_linf_m": (
                    root_position_recovery
                ),
                "action_root_end_to_start_quaternion_linf": (
                    root_orientation_recovery
                ),
                "shared_ready_threshold_rad": float(
                    args.shared_ready_joint_linf_rad
                ),
                "recovery_threshold_rad": float(
                    args.recovery_joint_linf_rad
                ),
                "shared_ready_pass": shared_ready_pass,
                "recovery_pass": recovery_pass,
            }
            if not shared_ready_pass:
                result["failure_reasons"].append(
                    "shared_ready_bank_mismatch"
                )
            if not recovery_pass:
                result["failure_reasons"].append(
                    "teacher_does_not_recover_to_ready"
                )
            if result["failure_reasons"]:
                result["verdict"] = "DIAGNOSTIC_FAIL"
            actions_out.append(result)
        overall_pass = all(row["verdict"] == "DIAGNOSTIC_PASS" for row in actions_out)
        base_receipt.update(
            {
                "status": "DIAGNOSTIC_PASS" if overall_pass else "DIAGNOSTIC_FAIL",
                "verdict": "DIAGNOSTIC_PASS" if overall_pass else "DIAGNOSTIC_FAIL",
                "manifest_id": manifest.manifest_id,
                "action_order": list(manifest.action_order),
                "scene_contract": scene_contract,
                "scene_contract_sha256": sha256_bytes(canonical_json_bytes(scene_contract)),
                "material_contract": material,
                "profile_contract": profile,
                "racket_geometry_contract": (
                    manifest.racket_geometry_contract
                ),
                "shared_ready_joint_linf_rad": shared_ready_linf,
                "actions": actions_out,
            }
        )
        _write_json_no_clobber(args.out, base_receipt)
        print(
            f"[teacher-ball-gate] {base_receipt['status']} "
            f"actions={len(actions_out)} receipt={args.out}"
        )
        return 0 if overall_pass else 3
    except Exception as exc:  # noqa: BLE001 - always leave a fail-closed receipt
        base_receipt.update(
            {
                "status": "INFRASTRUCTURE_FAIL",
                "verdict": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        try:
            _write_json_no_clobber(args.out, base_receipt)
        except Exception as write_exc:
            print(
                f"[teacher-ball-gate][FATAL] {exc}; additionally cannot write receipt: {write_exc}",
                file=sys.stderr,
            )
            return 2
        print(f"[teacher-ball-gate][FATAL] {exc}; receipt={args.out}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
