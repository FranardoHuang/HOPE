#!/usr/bin/env python3
"""Build a commit-required split-ready lineage for one A211 question.

The physical reset and hidden-WAIT teacher are the tracked safe-ready state.
At task reveal the teacher switches to the exact measured frame 0 and the
public teacher-start clock leaves time for the policy to learn that bridge.
The 60-policy-step nominal receipt authorizes only the at-most-25-step hidden
WAIT.  It is deliberately not a four-second passive-soak or a frame0 birth
certificate.

This producer is deliberately not a launcher.  The launcher remains the final
authority: until this output and every pin inside it are committed, its
tracked-file gate rejects them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a211_four_arm_diagnostic.py"
SHA256_HEX = frozenset("0123456789abcdef")


class MaterializationError(RuntimeError):
    """An A211 lineage input or publication boundary was invalid."""


def _load_launcher():
    spec = importlib.util.spec_from_file_location("_a211_lineage_launcher", LAUNCHER_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import A211 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_L = _load_launcher()


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializationError("value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in SHA256_HEX for c in value):
        raise MaterializationError("%s must be one lowercase SHA-256" % name)
    return value


def _commit(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 40 or any(c not in SHA256_HEX for c in value):
        raise MaterializationError("%s must be a 40-character lowercase Git commit" % name)
    return value


def _relative(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise MaterializationError("%s must be a non-empty POSIX relative path" % name)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise MaterializationError("%s must be a normalized relative path" % name)
    return path.as_posix()


def _regular(path: Path, *, name: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MaterializationError("cannot inspect %s: %s" % (name, exc)) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MaterializationError("%s must be a regular non-symlink file" % name)


def _strict_json(
    raw: bytes, *, name: str, require_canonical_bytes: bool = True
) -> dict[str, Any]:
    def unique(rows):
        output = {}
        for key, value in rows:
            if key in output:
                raise MaterializationError("%s contains duplicate key %r" % (name, key))
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MaterializationError("%s contains non-finite %s" % (name, token))
            ),
        )
    except MaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("%s is not strict UTF-8 JSON" % name) from exc
    if type(value) is not dict:
        raise MaterializationError("%s root must be an object" % name)
    if require_canonical_bytes and raw != canonical_bytes(value) + b"\n":
        raise MaterializationError("%s must be canonical JSON plus newline" % name)
    return value


def _repo_root(value: str) -> Path:
    root = Path(value).resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise MaterializationError("--repo-root must be the Git worktree root")
    return root


def _git(root: Path, args: Sequence[str], *, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False,
        capture_output=True, text=text,
    )


def _pin(path: str, digest: str, *, name: str) -> dict[str, str]:
    return {"path": _relative(path, name=name + ".path"), "sha256": _sha(digest, name=name + ".sha256")}


def _input(
    root: Path, *, path: str, digest: str, name: str, explicit: bool,
    source_commit: str, require_canonical_bytes: bool = True,
) -> tuple[dict[str, str], dict[str, Any], str]:
    pin = _pin(path, digest, name=name)
    candidate = root / pin["path"]
    _regular(candidate, name=name)
    observed = sha256_file(candidate)
    if observed != pin["sha256"]:
        raise MaterializationError("%s file SHA differs" % name)
    if not explicit:
        tracked = _git(root, ("ls-files", "--stage", "--", pin["path"]))
        if tracked.returncode or not tracked.stdout.strip():
            raise MaterializationError("%s is not tracked; use its explicit input flag" % name)
        committed = _git(root, ("show", source_commit + ":" + pin["path"]), text=False)
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != pin["sha256"]:
            raise MaterializationError("%s differs from --source-commit" % name)
    document = _strict_json(
        candidate.read_bytes(),
        name=name,
        require_canonical_bytes=require_canonical_bytes,
    )
    return pin, document, canonical_sha256(document)


def _tracked_input(root: Path, *, path: str, digest: str, name: str, source_commit: str) -> tuple[dict[str, str], dict[str, Any], str]:
    return _input(
        root, path=path, digest=digest, name=name, explicit=False,
        source_commit=source_commit, require_canonical_bytes=False,
    )


def _tracked_file(root: Path, *, path: str, digest: str, name: str, source_commit: str) -> dict[str, str]:
    pin = _pin(path, digest, name=name)
    candidate = root / pin["path"]
    _regular(candidate, name=name)
    if sha256_file(candidate) != pin["sha256"]:
        raise MaterializationError("%s file SHA differs" % name)
    tracked = _git(root, ("ls-files", "--stage", "--", pin["path"]))
    if tracked.returncode or not tracked.stdout.strip():
        raise MaterializationError("%s is not tracked" % name)
    committed = _git(root, ("show", source_commit + ":" + pin["path"]), text=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != pin["sha256"]:
        raise MaterializationError("%s differs from --source-commit" % name)
    return pin


def _require_seal(document: Mapping[str, Any], key: str, *, name: str) -> str:
    seal = _sha(document.get(key), name=name + "." + key)
    unsigned = dict(document)
    unsigned.pop(key)
    if canonical_sha256(unsigned) != seal:
        raise MaterializationError("%s %s is not reproducible" % (name, key))
    return seal


def _manifest_semantics(manifest: Mapping[str, Any], *, action_id: str, action_uid: int, motion: Mapping[str, str]) -> tuple[str, str]:
    if manifest.get("schema_version") != 3 or manifest.get("action_order") != [action_id] or manifest.get("mobility_mode") != "no_move":
        raise MaterializationError("action manifest fixed-N1 identity differs")
    actions = manifest.get("actions")
    if type(actions) is not list or len(actions) != 1 or type(actions[0]) is not dict:
        raise MaterializationError("action manifest must contain exactly one action")
    action = actions[0]
    if action.get("action_id") != action_id or action.get("action_uid") != action_uid or action.get("motion_path") != motion["path"] or action.get("motion_sha256") != motion["sha256"]:
        raise MaterializationError("action manifest action pin differs")
    return (
        _sha(manifest.get("solver_profile_sha256"), name="manifest solver_profile_sha256"),
        _sha(manifest.get("physics_profile_sha256"), name="manifest physics_profile_sha256"),
    )


def _dynamic_semantics(document: Mapping[str, Any], *, action_id: str, motion: Mapping[str, str], nominal: bool) -> str:
    if document.get("action_id") != action_id or document.get("motion_sha256", motion["sha256"]) != motion["sha256"]:
        raise MaterializationError("dynamic-ready input action/motion differs")
    if nominal:
        if document.get("verdict") != "PASS":
            raise MaterializationError("nominal-hold receipt verdict differs")
    else:
        if document.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2":
            raise MaterializationError("dynamic-ready artifact kind differs")
        teacher = document.get("teacher_reference")
        if type(teacher) is not dict or teacher.get("motion_sha256") != motion["sha256"]:
            raise MaterializationError("dynamic-ready teacher motion differs")
    return _require_seal(document, "content_sha256", name="dynamic-ready input")


_FRAME0_HANDOFF_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "selection_semantics",
        "state_sha256_semantics",
        "physical_ready_state_sha256",
        "teacher_frame0_state_sha256",
        "mjcf_audit_state_sha256",
        "stored_root_quaternion_norm",
        "mjcf_audit_root_quat_wxyz",
        "mjcf_audit_quaternion_semantics",
        "stored_teacher_and_physical_quaternion_unchanged",
        "endpoints_bitwise_equal",
        "physical_ready_joint_velocity_exact_zero",
        "teacher_static_endpoint_joint_velocity_exact_zero",
        "measured_motion_velocity_channels_consumed",
        "not_a_motion_velocity_continuity_claim",
        "certified_transition_s",
        "required_min_wait_s",
        "torque_speed_curve_required",
        "torque_speed_non_requirement_reason",
        "runtime_transition_reference_required",
        "required_followup_hold_gate",
        "required_followup_policy_steps",
        "required_followup_physics_steps",
        "diagnostic_unauthorized",
        "training_authorized",
    )
)


def _finite_vector(value: Any, *, width: int, name: str) -> list[float]:
    if (
        type(value) is not list
        or len(value) != width
        or any(
            type(item) not in (int, float)
            or type(item) is bool
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise MaterializationError("%s must be %d finite numbers" % (name, width))
    return [float(item) for item in value]


def _whole_body_state_sha256(
    joint_pos: Sequence[float],
    root_pos: Sequence[float],
    root_quat: Sequence[float],
) -> str:
    """Reproduce canonical_grounded_ready.state_digest without MuJoCo."""

    digest = hashlib.sha256()
    for label, values in (
        ("joint_pos", joint_pos),
        ("root_pos_w", root_pos),
        ("root_quat_wxyz", root_quat),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(label.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _exact_zero_handoff_semantics(
    artifact: Mapping[str, Any], *, motion_sha256: str
) -> bool:
    """Validate the only authority under which physical==teacher is legal."""

    physical = artifact.get("physical_ready")
    teacher = artifact.get("teacher_reference")
    composition = artifact.get("physical_birth_composition")
    static = artifact.get("physical_birth_static_evidence")
    top_handoff = artifact.get("frame0_handoff")
    if top_handoff is None:
        return False
    if not all(
        type(value) is dict
        for value in (physical, teacher, composition, static, top_handoff)
    ):
        raise MaterializationError("exact frame0 handoff authority is incomplete")
    composition_handoff = composition.get("frame0_handoff")
    static_handoff = static.get("frame0_handoff")
    if (
        type(composition_handoff) is not dict
        or type(static_handoff) is not dict
        or set(top_handoff) != _FRAME0_HANDOFF_KEYS
        or top_handoff != composition_handoff
        or top_handoff != static_handoff
    ):
        raise MaterializationError("exact frame0 handoff copies differ or are malformed")

    physical_q = _finite_vector(
        physical.get("joint_pos_rad"), width=31, name="physical_ready.joint_pos_rad"
    )
    physical_dq = _finite_vector(
        physical.get("joint_vel_radps"), width=31, name="physical_ready.joint_vel_radps"
    )
    physical_root = _finite_vector(
        physical.get("root_pos_w_m"), width=3, name="physical_ready.root_pos_w_m"
    )
    physical_quat = _finite_vector(
        physical.get("root_quat_wxyz"), width=4, name="physical_ready.root_quat_wxyz"
    )
    teacher_q = _finite_vector(
        teacher.get("joint_pos_rad"), width=31, name="teacher_reference.joint_pos_rad"
    )
    teacher_root = _finite_vector(
        teacher.get("root_pos_w_m"), width=3, name="teacher_reference.root_pos_w_m"
    )
    teacher_quat = _finite_vector(
        teacher.get("root_quat_wxyz"), width=4, name="teacher_reference.root_quat_wxyz"
    )
    teacher_static_dq = _finite_vector(
        teacher.get("static_handoff_joint_vel_radps"),
        width=31,
        name="teacher_reference.static_handoff_joint_vel_radps",
    )
    state_sha = _whole_body_state_sha256(physical_q, physical_root, physical_quat)
    quaternion_norm = float(np.linalg.norm(np.asarray(physical_quat, np.float64)))
    if (
        not math.isfinite(quaternion_norm)
        or quaternion_norm <= 0.0
        or abs(quaternion_norm - 1.0) > 2.0e-6
    ):
        raise MaterializationError("stored exact-frame0 quaternion is invalid")
    audit_quat = _finite_vector(
        top_handoff.get("mjcf_audit_root_quat_wxyz"),
        width=4,
        name="frame0_handoff.mjcf_audit_root_quat_wxyz",
    )
    expected_audit_quat = (
        np.asarray(physical_quat, np.float64) / quaternion_norm
    ).tolist()
    audit_state_sha = _whole_body_state_sha256(
        physical_q, physical_root, audit_quat
    )
    composition_teacher_quat = _finite_vector(
        composition.get("teacher_root_quat_wxyz"),
        width=4,
        name="physical_birth_composition.teacher_root_quat_wxyz",
    )
    composition_physical_quat = _finite_vector(
        composition.get("physical_root_quat_wxyz"),
        width=4,
        name="physical_birth_composition.physical_root_quat_wxyz",
    )
    composition_stored_quat = _finite_vector(
        composition.get("stored_physical_root_quat_wxyz"),
        width=4,
        name="physical_birth_composition.stored_physical_root_quat_wxyz",
    )
    composition_audit_quat = _finite_vector(
        composition.get("mjcf_audit_root_quat_wxyz"),
        width=4,
        name="physical_birth_composition.mjcf_audit_root_quat_wxyz",
    )
    static_stored_quat = _finite_vector(
        static.get("stored_root_quat_wxyz"),
        width=4,
        name="physical_birth_static_evidence.stored_root_quat_wxyz",
    )
    static_audit_quat = _finite_vector(
        static.get("mjcf_audit_root_quat_wxyz"),
        width=4,
        name="physical_birth_static_evidence.mjcf_audit_root_quat_wxyz",
    )

    robust = static.get("direct_frame0_robust_minimum_slacks")
    witness = static.get("evaluator_evidence")
    racket = static.get("independent_measured_racket_frame0")
    if (
        physical_q != teacher_q
        or physical_root != teacher_root
        or physical_quat != teacher_quat
        or physical_dq != [0.0] * 31
        or teacher_static_dq != [0.0] * 31
        or teacher.get("semantics") != "exact_motion_bytes_frame0_reference"
        or teacher.get("motion_sha256") != motion_sha256
        or teacher.get("frame_index") != 0
        or teacher.get("static_handoff_velocity_semantics")
        != "constructed_zero_joint_velocity_endpoint_not_measured_motion_velocity"
        or composition.get("semantics")
        != "measured_frame0_direct_if_safe_else_lexicographic_whole_body_safe_ready"
        or composition.get("teacher_reference_unchanged") is not True
        or composition.get("historical_physical_birth_seed_consumed") is not False
        or composition.get("selection_priority")
        != [
            "exact_measured_frame0_if_all_safety_gates_pass",
            "lexicographic_whole_body_safe_ready_only_if_frame0_unsafe",
        ]
        or composition.get("exact_measured_frame0_selected") is not True
        or composition.get("teacher_and_physical_birth_differ") is not False
        or composition.get("changed_joint_mask") != [False] * 31
        or composition.get("changed_joint_indices") != []
        or composition.get("changed_joint_names") != []
        or composition.get("physical_minus_teacher_joint_pos_rad") != [0.0] * 31
        or composition.get("physical_minus_teacher_root_pos_m") != [0.0] * 3
        or composition.get("physical_minus_teacher_root_rotation_vector_rad")
        != [0.0] * 3
        or composition_teacher_quat != teacher_quat
        or composition_physical_quat != physical_quat
        or composition_stored_quat != physical_quat
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(composition_audit_quat, expected_audit_quat)
        )
        or static.get("fresh_direct_robust_gate_passed") is not True
        or static.get("authority")
        != "fresh_current_exact_mjcf_whole_body_lexicographic_search"
        or static.get("selected_hold_witness_authority")
        != "new_backend_new_solver_final_state_cache_miss"
        or static.get("exact_contact_lp_reused") is not False
        or static.get("all_safety_slacks_meet_original_and_locked_gate") is not True
        or static.get("geometry_passed") is not True
        or static.get("ground_dynamics_passed") is not True
        or static.get("stored_endpoint_state_sha256") != state_sha
        or static.get("mjcf_audit_state_sha256") != audit_state_sha
        or static_stored_quat != physical_quat
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(static_audit_quat, expected_audit_quat)
        )
        or not math.isclose(
            float(static.get("stored_root_quaternion_norm", math.nan)),
            quaternion_norm,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or type(robust) is not dict
        or not robust
        or any(
            type(value) not in (int, float)
            or type(value) is bool
            or not math.isfinite(float(value))
            for value in robust.values()
        )
        or static.get("direct_frame0_robust_gate_sha256")
        != canonical_sha256(robust)
        or type(witness) is not dict
        or witness.get("lp_feasible") is not True
        or witness.get("exact_state_lp_cache_hit") is not False
        or witness.get("evaluated_state_sha256") != audit_state_sha
        or witness.get("required_minimum_normal_force_per_contact_n") != 0.1
        or witness.get("required_minimum_normal_force_per_foot_n") != 1.0
        or type(racket) is not dict
        or racket.get("authority")
        != "independent_schema_v4_measured_racket_channel"
        or racket.get("motion_sha256") != motion_sha256
        or racket.get("frame_index") != 0
        or top_handoff.get("schema_version") != 1
        or top_handoff.get("kind") != "exact_frame0_zero_duration_handoff_v1"
        or top_handoff.get("selection_semantics")
        != "threshold_first_exact_frame0_direct"
        or top_handoff.get("state_sha256_semantics")
        != "float64_array_bytes_without_quaternion_normalization_v1"
        or top_handoff.get("physical_ready_state_sha256") != state_sha
        or top_handoff.get("teacher_frame0_state_sha256") != state_sha
        or top_handoff.get("mjcf_audit_state_sha256") != audit_state_sha
        or not math.isclose(
            float(top_handoff.get("stored_root_quaternion_norm", math.nan)),
            quaternion_norm,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(audit_quat, expected_audit_quat)
        )
        or top_handoff.get("mjcf_audit_quaternion_semantics")
        != "stored_root_quat_unit_normalized_for_numerical_backend_only"
        or top_handoff.get("stored_teacher_and_physical_quaternion_unchanged")
        is not True
        or top_handoff.get("endpoints_bitwise_equal") is not True
        or top_handoff.get("physical_ready_joint_velocity_exact_zero") is not True
        or top_handoff.get("teacher_static_endpoint_joint_velocity_exact_zero")
        is not True
        or top_handoff.get("measured_motion_velocity_channels_consumed") is not False
        or top_handoff.get("not_a_motion_velocity_continuity_claim") is not True
        or top_handoff.get("certified_transition_s") != 0.0
        or top_handoff.get("required_min_wait_s") != 0.0
        or top_handoff.get("torque_speed_curve_required") is not False
        or top_handoff.get("torque_speed_non_requirement_reason")
        != (
            "identical_stored_configuration_and_constructed_zero_joint_"
            "velocity_endpoints"
        )
        or top_handoff.get("runtime_transition_reference_required") is not False
        or top_handoff.get("required_followup_hold_gate")
        != _L.FRAME0_LIVE_RECEIPT_KIND
        or top_handoff.get("required_followup_policy_steps")
        != _L.PHYSICAL_READY_HOLD_POLICY_STEPS
        or top_handoff.get("required_followup_physics_steps")
        != _L.PHYSICAL_READY_HOLD_PHYSICS_STEPS
        or top_handoff.get("diagnostic_unauthorized") is not True
        or top_handoff.get("training_authorized") is not False
    ):
        raise MaterializationError("exact frame0 zero-handoff authority is invalid")
    return True


# Removed 2026-08-05 (safety-gate dead-code cleanup): the frame0-exact cluster
# that used to live here -- _physical_ready_long_hold_semantics, _FRAME0_KEYS,
# _motion_frame0, _exact_float_vector, _validate_frame0_payload and
# _frame0_exact_semantics (the only reader of the retired immutable tape).
# None of them was reachable from materialize(); the live v5 lineage path
# validates measured frame 0 through _L._validate_teacher_frame0_artifact and
# _L._split_ready_reset_wait_semantics in
# launch_action_ball_a211_four_arm_diagnostic.py.  Recover from git: this file
# at commit 80fe5f6f, lines 550-937.


def _write_new(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    relative = _relative(relative, name="output")
    check = _git(root, ("check-ignore", "-q", "--no-index", "--", relative))
    if check.returncode == 0:
        raise MaterializationError("output must not be Git-ignored")
    if check.returncode not in (0, 1):
        raise MaterializationError("cannot inspect output ignore policy")
    output = root / relative
    if output.exists() or output.is_symlink():
        raise MaterializationError("no-clobber output already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise MaterializationError("output parent escaped repo root") from exc
    if parent != output.parent:
        raise MaterializationError("output parent must not traverse a symlink")
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".a211-lineage-", dir=str(parent))
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:  # pragma: no cover - race protection
            raise MaterializationError("no-clobber output already exists: %s" % output) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    _regular(output, name="output")
    return {"path": relative, "sha256": sha256_file(output)}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root(args.repo_root)
    source_commit = _commit(args.source_commit, name="source_commit")
    if _git(root, ("rev-parse", "--verify", source_commit + "^{commit}")).returncode:
        raise MaterializationError("source_commit is not a Git commit")
    manifest_pin, manifest, manifest_content_sha = _input(
        root, path=args.action_manifest_path, digest=args.expected_action_manifest_sha256,
        name="action manifest", explicit=args.action_manifest_explicit, source_commit=source_commit,
    )
    motion = _tracked_file(
        root, path=args.motion_path, digest=args.expected_motion_sha256,
        name="motion", source_commit=source_commit,
    )
    dynamic_ready, dynamic_doc, _dynamic_doc_content = _tracked_input(
        root, path=args.dynamic_ready_artifact_path, digest=args.expected_dynamic_ready_artifact_sha256,
        name="dynamic-ready artifact", source_commit=source_commit,
    )
    nominal_hold, hold_doc, _hold_doc_content = _tracked_input(
        root, path=args.dynamic_ready_nominal_receipt_path, digest=args.expected_dynamic_ready_nominal_receipt_sha256,
        name="dynamic-ready nominal-hold receipt", source_commit=source_commit,
    )
    teacher_frame0_artifact, frame0_artifact_doc, _frame0_artifact_content = _tracked_input(
        root, path=args.teacher_frame0_artifact_path,
        digest=args.expected_teacher_frame0_artifact_sha256,
        name="teacher-frame0 artifact", source_commit=source_commit,
    )
    initial_center_receipt, initial_center_receipt_doc, _ = _tracked_input(
        root,
        path=args.initial_center_task_receipt_path,
        digest=args.expected_initial_center_task_receipt_sha256,
        name="initial-center task receipt",
        source_commit=source_commit,
    )
    action_id = _L.ACTION_ID
    action_uid = _L.ACTION_UID
    teacher_id = _L.TEACHER_ID
    solver_sha, physics_sha = _manifest_semantics(manifest, action_id=action_id, action_uid=action_uid, motion=motion)
    dynamic_content_sha = _dynamic_semantics(dynamic_doc, action_id=action_id, motion=motion, nominal=False)
    hold_content_sha = _dynamic_semantics(hold_doc, action_id=action_id, motion=motion, nominal=True)
    if (
        dynamic_ready["sha256"] != _L.SPLIT_READY_DYNAMIC_ARTIFACT_SHA256
        or nominal_hold["sha256"] != _L.SPLIT_READY_NOMINAL_HOLD_SHA256
        or teacher_frame0_artifact["sha256"]
        != _L.SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256
    ):
        raise MaterializationError("split-ready authority bytes differ")
    try:
        teacher_frame0 = _L._validate_teacher_frame0_artifact(
            frame0_artifact_doc,
            motion_path=root / motion["path"],
            motion_sha256=motion["sha256"],
        )
        initial_center_timing = _L._initial_center_timing_authority(
            receipt=initial_center_receipt_doc,
            receipt_pin=initial_center_receipt,
            action_manifest=manifest,
            action_manifest_pin=manifest_pin,
            motion_sha256=motion["sha256"],
        )
        reset_wait = _L._split_ready_reset_wait_semantics(
            dynamic=dynamic_doc,
            nominal=hold_doc,
            dynamic_pin=dynamic_ready,
            nominal_pin=nominal_hold,
            teacher_frame0=teacher_frame0["frame0"],
            motion_sha256=motion["sha256"],
            initial_center_timing_authority=initial_center_timing,
        )
    except (_L.LaunchRefused, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MaterializationError):
            raise
        raise MaterializationError(str(exc)) from exc
    try:
        dr_l0_manifest = _L._dr_l0_manifest_binding(
            root,
            source_commit,
            family="A",
            task_profile=_L.TASK_PROFILE_ID,
        )
    except _L.LaunchRefused as exc:
        raise MaterializationError(str(exc)) from exc
    lineage = {
        "schema_version": 5, "kind": _L.LINEAGE_KIND,
        "actor_contract": _L.ACTOR_CONTRACT, "actor_width": _L.ACTOR_WIDTH,
        "critic_contract": _L.CRITIC_CONTRACT, "critic_width": _L.CRITIC_WIDTH,
        "trainability_contract": _L.TRAINABILITY_CONTRACT,
        "actor_layout_identity": _L._actor_layout_identity(),
        "task_profile": _L.TASK_PROFILE_ID, "gym_task": _L.GYM_TASK_ID,
        "target_semantics": _L.TARGET_SEMANTICS,
        "runtime_target_contract": _L._runtime_target_contract(),
        "action_id": action_id,
        "curriculum_scope": _L._curriculum_scope_contract(),
        "teacher_id": teacher_id, "seed": 0,
        "motion": motion,
        "action_manifest": manifest_pin,
        "initial_center_task_receipt": initial_center_receipt,
        "dynamic_ready_artifact": dynamic_ready,
        "dynamic_ready_nominal_receipt": nominal_hold,
        "teacher_frame0_artifact": teacher_frame0_artifact,
        "dr_l0_manifest": dr_l0_manifest,
    }
    lineage_pin = _write_new(root, args.output, canonical_bytes(lineage) + b"\n")
    semantic = {
        "actor_layout_content_sha256": _L._actor_layout_identity()[
            "content_sha256"
        ],
        "manifest_content_sha256": manifest_content_sha,
        "solver_profile_sha256": solver_sha, "physics_profile_sha256": physics_sha,
        "dynamic_ready_content_sha256": dynamic_content_sha,
        "nominal_hold_content_sha256": hold_content_sha,
        "teacher_frame0_artifact_content_sha256": teacher_frame0[
            "content_sha256"
        ],
        "split_ready_reset_wait_claim_sha256": reset_wait["claim_sha256"],
        "initial_center_timing_claim_sha256": initial_center_timing[
            "claim_sha256"
        ],
        "dr_l0_contract_sha256": dr_l0_manifest["contract_sha256"],
    }
    return {
        "status": "MATERIALIZED_COMMIT_REQUIRED", "diagnostic_unauthorized": True,
        "launch_authorized": False, "lineage": lineage_pin,
        "lineage_content_sha256": canonical_sha256(lineage),
        "semantic_sha256": canonical_sha256(semantic), "source_commit": source_commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    for name in ("action-manifest",):
        parser.add_argument("--%s-path" % name, required=True)
        parser.add_argument("--expected-%s-sha256" % name, required=True)
        parser.add_argument("--%s-explicit" % name, action="store_true")
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--initial-center-task-receipt-path", required=True)
    parser.add_argument(
        "--expected-initial-center-task-receipt-sha256", required=True
    )
    parser.add_argument("--dynamic-ready-artifact-path", required=True)
    parser.add_argument("--expected-dynamic-ready-artifact-sha256", required=True)
    parser.add_argument("--dynamic-ready-nominal-receipt-path", required=True)
    parser.add_argument("--expected-dynamic-ready-nominal-receipt-sha256", required=True)
    parser.add_argument("--teacher-frame0-artifact-path", required=True)
    parser.add_argument("--expected-teacher-frame0-artifact-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize(args)
    except MaterializationError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
