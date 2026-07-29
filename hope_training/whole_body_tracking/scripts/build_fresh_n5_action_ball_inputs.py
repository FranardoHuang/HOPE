#!/usr/bin/env python3
"""Build the exact fresh-N5 ball centres and frozen solver-case overlay.

This producer is deliberately split at the schema boundary:

* ``manifest.json`` is the strict ActionBall schema-v3 training input.  It has
  no gate-only ``physical_*`` keys.
* ``fresh_n5_physical_task_bundle.json`` is a content-bound overlay.  A
  downstream materializer may copy the pinned racket/contact contracts plus
  ``physical_ball_launch`` and ``physical_task_binding`` into a disposable
  fitted-gate manifest, while the training runtime continues to consume the
  strict base manifest.

The action identity is always frozen before a ball is considered.  The
teacher-centre inverse screen samples incoming balls for that one action and
keeps the full proposal/rejection denominator.  The task solve then calls the
same fixed-action ``continuous_questions.solve_proposals`` entry point used by
the runtime; no selector or action switching exists in this script.

All public outputs are no-clobber.  ``build-all`` creates a brand-new output
directory, so a failed retry cannot silently replace prior evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_DIR_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)

FRESH_N5_ACTION_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FAMILIES = {
    "bh_loop_c": "backhand",
    "v12_forehand_block": "forehand",
    "bh_block": "backhand",
    "s0_highpress": "backhand",
    "fh_loop_high": "forehand",
}
FRESH_N5_FORBIDDEN = frozenset(("fh_loop", "fh_block_syn"))
FRESH_BANK_ORDER = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
    "v12_forehand_block",
)
CASE_ROLES = (
    "center_positive_seed_0",
    "center_positive_seed_1",
    "support_positive",
    "negative_t_hit_offset",
    "negative_face_sign",
    "negative_ball_state_mismatch",
)
POSITIVE_CASE_ROLES = frozenset(CASE_ROLES[:3])
NEGATIVE_REASONS = {
    "negative_t_hit_offset": "teacher_contact_time_mismatch",
    "negative_face_sign": "selected_face_sign_mismatch",
    "negative_ball_state_mismatch": "bound_ball_state_mismatch",
}
SOLVER_SOURCE_NAMES = (
    "continuous_questions.py",
    "hope_commands.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
)
UNITS = {
    "position": "m",
    "velocity": "m/s",
    "spin": "rad/s",
    "time": "s",
}


class FreshN5BuildError(RuntimeError):
    """Fail-closed producer error."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreshN5BuildError("%s is not readable exact JSON: %s" % (label, exc))
    if not isinstance(value, dict):
        raise FreshN5BuildError("%s must be one JSON object" % label)
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshN5BuildError("%s must be one lowercase SHA-256" % label)
    return value


def _finite_vec(value: Any, length: int, label: str) -> List[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != length
    ):
        raise FreshN5BuildError("%s must be a length-%d vector" % (label, length))
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise FreshN5BuildError("%s contains NaN/Inf" % label)
    return result


def _repo_relative(path: Path, repo_root: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise FreshN5BuildError(
            "%s must stay under repo root: %s" % (label, path)
        ) from exc


def _write_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _copy_exclusive(source: Path, target: Path) -> None:
    raw = source.read_bytes()
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, str(path))
    if specification is None or specification.loader is None:
        raise FreshN5BuildError("cannot import %s from %s" % (name, path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_mdp_package(repo_root: Path) -> Dict[str, Any]:
    """Load dependency-light MDP siblings under one synthetic package.

    The production modules use relative imports.  Importing them as unrelated
    top-level files works for a few helpers but fails once the real fixed-action
    solver imports its siblings.  A private package with an explicit ``__path__``
    preserves the production relative-import graph without importing the
    Isaac-heavy public package ``__init__``.
    """

    mdp_dir = repo_root / MDP_DIR_REL
    package_name = "_fresh_n5_mdp"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(mdp_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    modules = {}
    for name in (
        "racket_contact_geometry",
        "virtual_ball",
        "stroke_prototypes_torch",
        "continuous_questions",
    ):
        modules[name] = _load_module(
            "%s.%s" % (package_name, name),
            mdp_dir / ("%s.py" % name),
        )
    return modules


def _quat_rotation(q_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise FreshN5BuildError("quaternion must be one finite wxyz vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise FreshN5BuildError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _yaw_degrees(q_wxyz: Sequence[float]) -> float:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return math.degrees(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _runtime_site_state(
    clip_path: Path,
    strike_frame: int,
    geometry_module: Any,
) -> Dict[str, Any]:
    with np.load(str(clip_path), allow_pickle=False) as clip:
        names = [str(item) for item in clip["body_names"]]
        try:
            wrist = names.index("right_wrist_yaw_Link")
            pelvis = names.index("pelvis_link")
        except ValueError as exc:
            raise FreshN5BuildError("%s lacks wrist/pelvis bodies" % clip_path) from exc
        positions = np.asarray(clip["body_pos_w"], dtype=np.float64)
        quaternions = np.asarray(clip["body_quat_w"], dtype=np.float64)
        if "body_ang_vel_w" not in clip.files:
            raise FreshN5BuildError("%s lacks body_ang_vel_w" % clip_path)
        angular = np.asarray(clip["body_ang_vel_w"], dtype=np.float64)
        fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
        total = int(positions.shape[0])
    if abs(fps - 50.0) > 1.0e-9:
        raise FreshN5BuildError("%s fps %.12g != required 50" % (clip_path, fps))
    if not 0 < strike_frame < total - 1:
        raise FreshN5BuildError(
            "%s contact frame %d is not interior to T=%d"
            % (clip_path, strike_frame, total)
        )
    offset = np.asarray(geometry_module.RACKET_SITE_OFFSET_WRIST_M, dtype=np.float64)
    rotations = np.stack([_quat_rotation(row) for row in quaternions[:, wrist]], axis=0)
    site = positions[:, wrist] + np.einsum("tij,j->ti", rotations, offset)
    lower = max(0, strike_frame - 2)
    upper = min(total - 1, strike_frame + 2)
    # Runtime keeps the 4-frame denominator even when clamped.
    site_velocity = (site[upper] - site[lower]) / (4.0 / fps)
    return {
        "fps": fps,
        "T": total,
        "wrist_index": wrist,
        "pelvis_index": pelvis,
        "site_position_w_m": site[strike_frame],
        "site_velocity_w_mps": site_velocity,
        "racket_quat_wxyz": quaternions[strike_frame, wrist],
        "racket_angular_velocity_w_radps": angular[strike_frame, wrist],
        "root_position_frame0_w_m": positions[0, pelvis],
        "root_quat_frame0_wxyz": quaternions[0, pelvis],
    }


def _validate_bank_and_prototype(
    *,
    bank_manifest_path: Path,
    prototype_path: Path,
    repo_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    bank = _read_json(bank_manifest_path, "complete 7x2 bank manifest")
    matrix = bank.get("output_matrix")
    expected_matrix = {
        "motion_ids": list(FRESH_BANK_ORDER),
        "scopes": ["upper", "full"],
        "candidate_count": 14,
    }
    if matrix != expected_matrix:
        raise FreshN5BuildError(
            "bank output_matrix is not the exact complete ordered 7x2 matrix"
        )
    outputs = bank.get("outputs")
    if not isinstance(outputs, list):
        raise FreshN5BuildError("bank outputs must be a list")
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for index, row in enumerate(outputs):
        if not isinstance(row, dict):
            raise FreshN5BuildError("bank output[%d] is not an object" % index)
        key = (str(row.get("motion_id")), str(row.get("scope")))
        if key in indexed:
            raise FreshN5BuildError("bank duplicates %r" % (key,))
        indexed[key] = row
    expected_keys = {
        (motion_id, scope)
        for motion_id in FRESH_BANK_ORDER
        for scope in ("upper", "full")
    }
    if set(indexed) != expected_keys:
        raise FreshN5BuildError(
            "bank does not contain exactly all 14 motion/scope rows"
        )
    bank_root = bank_manifest_path.parent
    for motion_id in FRESH_BANK_ORDER:
        for scope in ("upper", "full"):
            row = indexed[(motion_id, scope)]
            expected_filename = "%s_%s_canonical_v2.npz" % (
                motion_id,
                scope,
            )
            if row.get("filename") != expected_filename:
                raise FreshN5BuildError(
                    "%s/%s filename is not %s" % (motion_id, scope, expected_filename)
                )
            expected_sha = _require_sha(
                row.get("output_npz_sha256"),
                "%s/%s output_npz_sha256" % (motion_id, scope),
            )
            candidate = (bank_root / expected_filename).resolve()
            if not candidate.is_file():
                raise FreshN5BuildError("complete 7x2 bank is missing %s" % candidate)
            actual_sha = _sha256_file(candidate)
            if actual_sha != expected_sha:
                raise FreshN5BuildError(
                    "%s/%s clip hash drifted: expected %s, got %s"
                    % (motion_id, scope, expected_sha, actual_sha)
                )

    prototype = _read_json(prototype_path, "fresh upper prototype")
    if prototype.get("schema_version") != 2:
        raise FreshN5BuildError("fresh upper prototype must use schema_version 2")
    scopes = prototype.get("scopes")
    if not isinstance(scopes, dict) or tuple(scopes) != ("upper",):
        raise FreshN5BuildError("fresh prototype must contain only the upper scope")
    rows = scopes["upper"]
    if (
        not isinstance(rows, list)
        or tuple(row.get("motion_id") for row in rows if isinstance(row, dict))
        != FRESH_N5_ACTION_ORDER
    ):
        raise FreshN5BuildError("fresh upper prototype action order drifted")
    derived = _canonical_sha256(scopes)
    if prototype.get("derived_sha256") != derived:
        raise FreshN5BuildError("prototype derived_sha256 does not seal its scopes")

    selected = []
    proto_by_action: Dict[str, Dict[str, Any]] = {}
    for clip_index, action_id in enumerate(FRESH_N5_ACTION_ORDER):
        if action_id in FRESH_N5_FORBIDDEN:
            raise FreshN5BuildError("forbidden legacy action entered fresh N5")
        output = indexed[(action_id, "upper")]
        expected_filename = "%s_upper_canonical_v2.npz" % action_id
        if output.get("filename") != expected_filename:
            raise FreshN5BuildError(
                "%s upper filename is not %s" % (action_id, expected_filename)
            )
        expected_sha = _require_sha(
            output.get("output_npz_sha256"),
            "%s output_npz_sha256" % action_id,
        )
        clip_path = (bank_root / expected_filename).resolve()
        if not clip_path.is_file():
            raise FreshN5BuildError("missing fresh upper clip %s" % clip_path)
        actual_sha = _sha256_file(clip_path)
        if actual_sha != expected_sha:
            raise FreshN5BuildError(
                "%s clip hash drifted: expected %s, got %s"
                % (action_id, expected_sha, actual_sha)
            )
        proto = rows[clip_index]
        if (
            proto.get("scope") != "upper"
            or proto.get("clip_index") != clip_index
            or proto.get("family") != FRESH_N5_FAMILIES[action_id]
            or proto.get("npz_sha256") != actual_sha
        ):
            raise FreshN5BuildError(
                "%s prototype identity/hash/frame row drifted" % action_id
            )
        selected.append(
            {
                "action_id": action_id,
                "family": FRESH_N5_FAMILIES[action_id],
                "clip_path": clip_path,
                "clip_repo_path": _repo_relative(
                    clip_path, repo_root, "%s clip" % action_id
                ),
                "motion_sha256": actual_sha,
                "output": output,
            }
        )
        proto_by_action[action_id] = proto
    return selected, proto_by_action


def _validate_profile_pins(
    path: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    pins = _read_json(path, "profile pins")
    solver_sha = _require_sha(pins.get("solver_profile_sha256"), "solver profile")
    physics_sha = _require_sha(pins.get("physics_profile_sha256"), "physics profile")
    source_map = pins.get("solver_implementation_source_sha256")
    if not isinstance(source_map, dict) or set(source_map) != set(SOLVER_SOURCE_NAMES):
        raise FreshN5BuildError(
            "profile pins must bind the exact five solver source files"
        )
    mdp_dir = repo_root / MDP_DIR_REL
    for name in SOLVER_SOURCE_NAMES:
        expected = _require_sha(source_map.get(name), "solver source %s" % name)
        actual = _sha256_file(mdp_dir / name)
        if actual != expected:
            raise FreshN5BuildError(
                "solver source %s drifted from profile pins: expected %s, got %s"
                % (name, expected, actual)
            )
    contact_geometry = pins.get("contact_geometry")
    if not isinstance(contact_geometry, dict) or not isinstance(
        contact_geometry.get("payload"), dict
    ):
        raise FreshN5BuildError("profile pins lack the exact contact_geometry payload")
    geometry_sha = _require_sha(contact_geometry.get("sha256"), "contact geometry")
    if _canonical_sha256(contact_geometry["payload"]) != geometry_sha:
        raise FreshN5BuildError("contact geometry payload SHA mismatch")
    physics_payload = pins.get("physics_payload")
    solver_payload = pins.get("solver_payload")
    if _canonical_sha256(physics_payload) != physics_sha:
        raise FreshN5BuildError("physics payload SHA mismatch")
    if _canonical_sha256(solver_payload) != solver_sha:
        raise FreshN5BuildError("solver payload SHA mismatch")
    return {
        "raw": pins,
        "file_sha256": _sha256_file(path),
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "source_map": dict(sorted(source_map.items())),
        "geometry_source_sha256": geometry_sha,
    }


def _load_incoming_distribution(
    path: Path,
    expected_sha256: str,
) -> Dict[str, Any]:
    expected = _require_sha(expected_sha256, "incoming distribution")
    actual = _sha256_file(path)
    if actual != expected:
        raise FreshN5BuildError(
            "incoming distribution SHA mismatch: expected %s, got %s"
            % (expected, actual)
        )
    document = _read_json(path, "incoming distribution")
    sampling = document.get("sampling_spec")
    if not isinstance(sampling, dict):
        raise FreshN5BuildError("incoming distribution lacks sampling_spec")
    pool = sampling.get("pooled_matchlike")
    if not isinstance(pool, dict):
        raise FreshN5BuildError("incoming distribution lacks pooled_matchlike")
    gaussian = pool.get("trunc_gaussian")
    if not isinstance(gaussian, dict):
        raise FreshN5BuildError("incoming distribution lacks truncated Gaussian")
    if gaussian.get("variables") != [
        "vx",
        "vy",
        "vz",
        "z_above_surface",
        "w_norm",
    ]:
        raise FreshN5BuildError("incoming distribution variable order drifted")
    mean = np.asarray(gaussian.get("mean"), dtype=np.float64)
    covariance = np.asarray(gaussian.get("cov"), dtype=np.float64)
    lower = np.asarray(gaussian.get("clip_lo"), dtype=np.float64)
    upper = np.asarray(gaussian.get("clip_hi"), dtype=np.float64)
    if (
        mean.shape != (5,)
        or covariance.shape != (5, 5)
        or lower.shape != (5,)
        or upper.shape != (5,)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not bool((lower < upper).all())
    ):
        raise FreshN5BuildError("incoming truncated Gaussian is malformed")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-8:
        raise FreshN5BuildError("incoming covariance is not positive semidefinite")
    return {
        "path": path,
        "sha256": actual,
        "document": document,
        "mean": mean,
        "covariance": covariance,
        "lower": lower,
        "upper": upper,
    }


def _draw_incoming_prior(
    *,
    distribution: Mapping[str, Any],
    rng: np.random.Generator,
    count: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    accepted: List[np.ndarray] = []
    raw_draws = 0
    outside_hull = 0
    not_incoming = 0
    while sum(len(block) for block in accepted) < count:
        remaining = count - sum(len(block) for block in accepted)
        draw_count = max(64, 2 * remaining)
        draw = rng.multivariate_normal(
            distribution["mean"],
            distribution["covariance"],
            size=draw_count,
            check_valid="raise",
        )
        raw_draws += draw_count
        inside = np.logical_and(
            draw >= distribution["lower"],
            draw <= distribution["upper"],
        ).all(axis=1)
        outside_hull += int((~inside).sum())
        incoming = draw[:, 0] < -1.0e-6
        not_incoming += int((inside & ~incoming).sum())
        kept = draw[inside & incoming]
        if kept.size:
            accepted.append(kept[:remaining])
        if raw_draws > max(10000, 1000 * count):
            raise FreshN5BuildError(
                "incoming truncated Gaussian could not supply enough incoming rows"
            )
    rows = np.concatenate(accepted, axis=0)[:count]
    return rows, {
        "raw_draw_count": raw_draws,
        "accepted_proposal_count": count,
        "truncation_rejection_counts": {
            "outside_observed_component_hull": outside_hull,
            "not_incoming_from_opponent": not_incoming,
        },
        "pool": "pooled_matchlike",
        "variables": [
            "vx",
            "vy",
            "vz",
            "z_above_surface",
            "w_norm",
        ],
        "center_spin_policy": (
            "sample empirical w_norm and an energy-fraction-weighted "
            "top/side/rifle axis; the selected per-action vector is written "
            "into the strict manifest spin centre"
        ),
    }


def _default_inverse_screen(
    *,
    action_id: str,
    contact_w_m: Sequence[float],
    face_velocity_w_mps: Sequence[float],
    physical_normal_w: Sequence[float],
    seed: int,
    proposal_count: int,
    venue_yaml: Path,
    profile_pins: Mapping[str, Any],
    repo_root: Path,
    incoming_distribution: Mapping[str, Any],
) -> Dict[str, Any]:
    del action_id
    if proposal_count < 64:
        raise FreshN5BuildError("inverse screen needs at least 64 proposals per action")
    import torch

    mdp_dir = repo_root / MDP_DIR_REL
    virtual_ball = _load_module("fresh_n5_virtual_ball", mdp_dir / "virtual_ball.py")
    prm = virtual_ball.load_venue_params(str(venue_yaml))
    rng = np.random.default_rng(seed)
    prior_rows, prior_receipt = _draw_incoming_prior(
        distribution=incoming_distribution,
        rng=rng,
        count=proposal_count,
    )
    velocity = prior_rows[:, :3]
    pool = incoming_distribution["document"]["pooled_matchlike"]["at_strike"]
    energy = np.asarray(
        (
            float(pool["frac_top_energy"]),
            float(pool["frac_side_energy"]),
            float(pool["frac_rifle_energy"]),
        ),
        np.float64,
    )
    if (
        energy.shape != (3,)
        or not np.isfinite(energy).all()
        or bool((energy < 0.0).any())
        or float(energy.sum()) <= 0.0
    ):
        raise FreshN5BuildError("incoming spin energy fractions are malformed")
    energy /= float(energy.sum())
    horizontal = velocity.copy()
    horizontal[:, 2] = 0.0
    horizontal_norm = np.linalg.norm(horizontal, axis=1, keepdims=True)
    if bool((horizontal_norm <= 1.0e-9).any()):
        raise FreshN5BuildError("incoming prior produced a zero horizontal direction")
    d_hat = horizontal / horizontal_norm
    z_hat = np.zeros_like(d_hat)
    z_hat[:, 2] = 1.0
    t_hat = np.cross(z_hat, d_hat)
    coefficients = rng.normal(size=(proposal_count, 3)) * np.sqrt(energy.reshape(1, 3))
    spin_axis = (
        coefficients[:, 0:1] * t_hat
        + coefficients[:, 1:2] * z_hat
        + coefficients[:, 2:3] * d_hat
    )
    spin_axis /= np.maximum(
        np.linalg.norm(spin_axis, axis=1, keepdims=True),
        1.0e-12,
    )
    spin = spin_axis * prior_rows[:, 4:5]
    dtype = torch.float64
    v_in = torch.as_tensor(velocity, dtype=dtype)
    w_in = torch.as_tensor(spin, dtype=dtype)
    v_face = torch.as_tensor(face_velocity_w_mps, dtype=dtype).reshape(1, 3)
    v_face = v_face.expand(proposal_count, 3).clone()
    normal = torch.as_tensor(physical_normal_w, dtype=dtype).reshape(1, 3)
    normal = normal.expand(proposal_count, 3).clone()
    v_out, w_out = virtual_ball.predict_paddle_contact(v_in, v_face, normal, w_in, prm)
    contact = torch.as_tensor(contact_w_m, dtype=dtype).reshape(1, 3)
    contact = contact.expand(proposal_count, 3).clone()
    planes = profile_pins["raw"]["planes"]
    physics_geometry = profile_pins["raw"]["physics_payload"]["geometry_and_grading"]
    rollout = virtual_ball.coarse_landing(
        contact,
        v_out,
        w_out,
        prm,
        surface_z=float(planes["surface_z"]),
        net_x=float(planes["net_x"]),
        h=float(profile_pins["raw"]["cfg"]["vb_rollout_h"]),
        n_steps=int(profile_pins["raw"]["cfg"]["vb_rollout_steps"]),
    )
    land_valid = rollout["land_valid"].detach().cpu().numpy().astype(bool)
    net_valid = rollout["net_valid"].detach().cpu().numpy().astype(bool)
    net_z = rollout["net_z"].detach().cpu().numpy()
    land_xy = rollout["land_xy"].detach().cpu().numpy()
    finite = (
        np.isfinite(velocity).all(axis=1)
        & np.isfinite(v_out.detach().cpu().numpy()).all(axis=1)
        & np.isfinite(w_out.detach().cpu().numpy()).all(axis=1)
        & np.isfinite(land_xy).all(axis=1)
        & np.isfinite(net_z)
    )
    net_clear = net_valid & (net_z > float(planes["net_top_z"]))
    in_x = (land_xy[:, 0] > float(physics_geometry["net_x_m"])) & (
        land_xy[:, 0] <= float(physics_geometry["opponent_far_x_m"])
    )
    in_y = np.abs(land_xy[:, 1]) <= float(physics_geometry["table_half_width_m"])
    legal = finite & land_valid & net_clear & in_x & in_y
    reasons = {
        "nonfinite": int((~finite).sum()),
        "no_landing": int((finite & ~land_valid).sum()),
        "net_not_cleared": int((finite & land_valid & ~net_clear).sum()),
        "landing_x_outside_opponent_table": int(
            (finite & land_valid & net_clear & ~in_x).sum()
        ),
        "landing_y_outside_table": int(
            (finite & land_valid & net_clear & in_x & ~in_y).sum()
        ),
    }
    reasons = {key: value for key, value in reasons.items() if value}
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        return {
            "proposal_count": proposal_count,
            "legal_count": 0,
            "rejection_counts": reasons,
            "status": "NO_LEGAL_CENTER",
            "incoming_prior": prior_receipt,
        }
    legal_velocity = velocity[legal_indices]
    centroid = legal_velocity.mean(axis=0)
    # A sampled medoid is itself already proven legal; unlike an arithmetic
    # centroid it cannot fall through a non-convex legal set.
    medoid_offset = legal_velocity - centroid.reshape(1, 3)
    chosen_index = int(
        legal_indices[int(np.argmin(np.sum(medoid_offset * medoid_offset, axis=1)))]
    )
    chosen_v_out = v_out[chosen_index].detach().cpu().tolist()
    chosen_w_out = w_out[chosen_index].detach().cpu().tolist()
    return {
        "proposal_count": proposal_count,
        "legal_count": int(legal_indices.size),
        "rejection_counts": reasons,
        "status": "PASS",
        "selected_proposal_index": chosen_index,
        "incoming_velocity_w_mps": velocity[chosen_index].tolist(),
        "incoming_spin_w_radps": spin[chosen_index].tolist(),
        "outgoing_velocity_w_mps": chosen_v_out,
        "outgoing_spin_w_radps": chosen_w_out,
        "legal_landing_w_xy_m": land_xy[chosen_index].tolist(),
        "legal_net_z_m": float(net_z[chosen_index]),
        "incoming_prior": prior_receipt,
    }


def build_batch_document(
    *,
    bank_manifest_path: Path,
    prototype_path: Path,
    profile_pins_path: Path,
    venue_yaml: Path,
    repo_root: Path,
    seed: int,
    proposal_count: int,
    incoming_dist_path: Path,
    incoming_dist_sha256: str,
    inverse_screen: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    selected, prototypes = _validate_bank_and_prototype(
        bank_manifest_path=bank_manifest_path,
        prototype_path=prototype_path,
        repo_root=repo_root,
    )
    pins = _validate_profile_pins(profile_pins_path, repo_root)
    incoming_distribution = _load_incoming_distribution(
        incoming_dist_path,
        incoming_dist_sha256,
    )
    if _sha256_file(venue_yaml) != pins["raw"]["venue_yaml_sha256"]:
        raise FreshN5BuildError("venue YAML bytes drifted from profile pins")
    geometry = _load_module(
        "fresh_n5_racket_contact_geometry",
        repo_root / MDP_DIR_REL / "racket_contact_geometry.py",
    )
    if geometry.GEOMETRY_SOURCE_SHA256 != pins["geometry_source_sha256"]:
        raise FreshN5BuildError("runtime geometry source differs from profile pins")
    screen = inverse_screen or _default_inverse_screen
    units = []
    screens = []
    for slot, selected_row in enumerate(selected):
        action_id = selected_row["action_id"]
        proto = prototypes[action_id]
        strike_frame = int(proto["contact_frame"])
        state = _runtime_site_state(selected_row["clip_path"], strike_frame, geometry)
        face_sign = int(proto["face_sign"])
        ball_local = np.asarray(
            geometry.ball_center_from_site_local(face_sign),
            dtype=np.float64,
        )
        rotation = _quat_rotation(state["racket_quat_wxyz"])
        contact_w = state["site_position_w_m"] + rotation.dot(ball_local)
        face_velocity = np.asarray(
            geometry.face_center_velocity_from_site(
                state["site_velocity_w_mps"].tolist(),
                state["racket_angular_velocity_w_radps"].tolist(),
                state["racket_quat_wxyz"].tolist(),
                face_sign,
            ),
            dtype=np.float64,
        )
        raw_normal = rotation.dot(np.asarray((0.0, 1.0, 0.0)))
        physical_normal = raw_normal * float(face_sign)
        result = screen(
            action_id=action_id,
            contact_w_m=contact_w.tolist(),
            face_velocity_w_mps=face_velocity.tolist(),
            physical_normal_w=physical_normal.tolist(),
            seed=int(seed) + 1009 * slot,
            proposal_count=int(proposal_count),
            venue_yaml=venue_yaml,
            profile_pins=pins,
            repo_root=repo_root,
            incoming_distribution=incoming_distribution,
        )
        if (
            not isinstance(result, dict)
            or result.get("status") != "PASS"
            or int(result.get("legal_count", 0)) <= 0
        ):
            raise FreshN5BuildError(
                "%s inverse screen found no legal centre; receipt=%r"
                % (action_id, result)
            )
        v_in = _finite_vec(
            result.get("incoming_velocity_w_mps"),
            3,
            "%s incoming centre" % action_id,
        )
        if v_in[0] >= -1.0e-6:
            raise FreshN5BuildError("%s inverse centre is not incoming" % action_id)
        root0 = state["root_position_frame0_w_m"]
        yaw = _yaw_degrees(state["root_quat_frame0_wxyz"])
        unit = {
            "uid": action_id,
            "family": "FH" if selected_row["family"] == "forehand" else "BH",
            "npz": selected_row["clip_repo_path"],
            "npz_sha256": selected_row["motion_sha256"],
            "T": int(state["T"]),
            "fps": 50,
            "hit_frame_50": strike_frame,
            "strike_phase": float(proto["strike_phase"]),
            "yaw_before_deg": yaw,
            "station_xy_hope_m": [
                float(root0[0]) - 0.5,
                float(root0[1]) - 0.7625,
            ],
            "ball_pos_hit_hope_m": [
                float(contact_w[0]) - 0.5,
                float(contact_w[1]) - 0.7625,
                float(contact_w[2]) - 0.76,
            ],
            "v_in_fit_hope_ms": v_in,
            "w_in_fit_hope_radps": _finite_vec(
                result.get("incoming_spin_w_radps", [0.0, 0.0, 0.0]),
                3,
                "%s incoming spin centre" % action_id,
            ),
            "v_out_fit_hope_ms": _finite_vec(
                result.get("outgoing_velocity_w_mps"),
                3,
                "%s outgoing centre" % action_id,
            ),
            "w_out_nominal_radps": _finite_vec(
                result.get("outgoing_spin_w_radps"),
                3,
                "%s outgoing spin" % action_id,
            ),
            "world_z0": "floor",
            "contact_point_semantics": "physical_ball_center_at_exact_teacher_strike",
            "base_semantics": "relative_contact_about_actual_spawn;no_move_goal_equals_spawn",
            "inverse_screen_receipt_sha256": _canonical_sha256(result),
        }
        if round(unit["strike_phase"] * (unit["T"] - 1)) != strike_frame:
            raise FreshN5BuildError(
                "%s prototype strike phase targets a different runtime frame"
                % action_id
            )
        units.append(unit)
        screens.append(
            {
                "action_id": action_id,
                "action_slot": slot,
                "action_frozen_before_sampling": True,
                "seed": int(seed) + 1009 * slot,
                **result,
                "receipt_payload_sha256": _canonical_sha256(result),
            }
        )
    if tuple(unit["uid"] for unit in units) != FRESH_N5_ACTION_ORDER:
        raise FreshN5BuildError("fresh N5 batch order changed during construction")
    return {
        "schema_version": 1,
        "artifact_type": "fresh_n5_action_ball_batch_v1",
        "action_order": list(FRESH_N5_ACTION_ORDER),
        "selector_executed": False,
        "action_identity_frozen_before_ball_sampling": True,
        "mobility_mode": "no_move",
        "base_task_frame": "relative_about_actual_episode_spawn",
        "source_bank": {
            "path": _repo_relative(bank_manifest_path, repo_root, "bank manifest"),
            "sha256": _sha256_file(bank_manifest_path),
            "required_matrix": "ordered_complete_7x2",
        },
        "prototype": {
            "path": _repo_relative(prototype_path, repo_root, "prototype"),
            "sha256": _sha256_file(prototype_path),
            "scope": "upper",
        },
        "profile_pins": {
            "path": _repo_relative(profile_pins_path, repo_root, "profile pins"),
            "sha256": pins["file_sha256"],
            "solver_profile_sha256": pins["solver_profile_sha256"],
            "physics_profile_sha256": pins["physics_profile_sha256"],
            "geometry_source_sha256": pins["geometry_source_sha256"],
        },
        "venue": {
            "path": _repo_relative(venue_yaml, repo_root, "venue YAML"),
            "sha256": _sha256_file(venue_yaml),
        },
        "incoming_distribution": {
            "path": _repo_relative(
                incoming_dist_path,
                repo_root,
                "incoming distribution",
            ),
            "sha256": incoming_distribution["sha256"],
            "pool": "pooled_matchlike",
            "usage": (
                "action-specific teacher inverse-screen proposal prior; "
                "never a direct global-centre substitution"
            ),
        },
        "inverse_screen": {
            "method": (
                "explicit_seed_venue_incoming_distribution_then_exact_teacher_"
                "face_contact_and_legal_return_screen"
            ),
            "proposal_count_per_action": int(proposal_count),
            "proposal_denominator_preserved": True,
            "rejection_reasons_preserved": True,
            "screens": screens,
        },
        "units": units,
    }


def _build_strict_manifest(
    *,
    batch_path: Path,
    prototype_path: Path,
    profile_pins: Mapping[str, Any],
    repo_root: Path,
    manifest_id: str,
    out_path: Path,
) -> None:
    builder = _load_module(
        "fresh_n5_build_action_ball_manifest",
        SCRIPTS_DIR / "build_action_ball_manifest.py",
    )
    motion_prefix = batch_path.parent
    batch = _read_json(batch_path, "fresh N5 batch")
    if not batch.get("units"):
        raise FreshN5BuildError("fresh N5 batch has no units")
    # Unit paths already name exact repo-relative clips.  The legacy builder
    # requires a prefix in fresh mode and then keeps only the basename.
    first_motion = repo_root / batch["units"][0]["npz"]
    clip_dir = first_motion.parent
    if any((repo_root / row["npz"]).parent != clip_dir for row in batch["units"]):
        raise FreshN5BuildError("fresh N5 clips must share one bank directory")
    with tempfile.TemporaryDirectory(prefix="fresh_n5_manifest_") as temporary:
        temp_out = Path(temporary) / "manifest.json"
        argv = [
            "build",
            "--batch-manifest",
            str(batch_path),
            "--batch-root",
            str(repo_root),
            "--repo-root",
            str(repo_root),
            "--out",
            str(temp_out),
            "--manifest-id",
            manifest_id,
            "--expect-units",
            "5",
            "--motion-path-prefix",
            _repo_relative(clip_dir, repo_root, "fresh bank clip directory"),
            "--prototype-path",
            _repo_relative(prototype_path, repo_root, "prototype"),
            "--prototype-scope",
            "upper",
            "--fresh-n5-upper",
            "--expected-geometry-source-sha256",
            profile_pins["geometry_source_sha256"],
            "--solver-profile-sha256",
            profile_pins["solver_profile_sha256"],
            "--physics-profile-sha256",
            profile_pins["physics_profile_sha256"],
            "--inbound-axis-mode",
            "env_neg_x_in_b_yaw",
            "--holdout-split-id",
            "heldout_ball_fresh_n5_v1",
        ]
        try:
            return_code = builder.main(argv)
        except SystemExit as exc:
            raise FreshN5BuildError(
                "strict manifest builder rejected fresh N5: %s" % exc
            ) from exc
        if return_code != 0:
            raise FreshN5BuildError("strict manifest builder returned %s" % return_code)
        # The shared builder predates per-action measured incoming-spin
        # centres and therefore exposes only one CLI scalar.  Fresh N5 keeps
        # the strict schema but replaces those fields from the exact batch
        # units before running the strict loader again below.
        manifest_document = _read_json(temp_out, "temporary strict manifest")
        unit_by_id = {row["uid"]: row for row in batch["units"]}
        for action in manifest_document["actions"]:
            unit = unit_by_id[action["action_id"]]
            spin_w = np.asarray(
                unit.get("w_in_fit_hope_radps", (0.0, 0.0, 0.0)),
                np.float64,
            )
            magnitude = float(np.linalg.norm(spin_w))
            if magnitude <= 1.0e-9:
                continue
            yaw = math.radians(float(unit["yaw_before_deg"]))
            cosine, sine = math.cos(-yaw), math.sin(-yaw)
            spin_b = np.asarray(
                (
                    cosine * spin_w[0] - sine * spin_w[1],
                    sine * spin_w[0] + cosine * spin_w[1],
                    spin_w[2],
                ),
                np.float64,
            )
            direction = spin_b / magnitude
            tangent_u, tangent_v = builder._tangent_frame(
                tuple(float(value) for value in direction)
            )
            profile = action["ball_profile"]
            profile["spin_direction_center_b_yaw"] = direction.tolist()
            profile["spin_direction_tangent_u_b_yaw"] = list(tangent_u)
            profile["spin_direction_tangent_v_b_yaw"] = list(tangent_v)
            profile["spin_magnitude_center_radps"] = magnitude
            profile["spin_magnitude_min_radps"] = 0.0
            profile["spin_magnitude_max_radps"] = max(60.0, 1.5 * magnitude)
            profile["spin_magnitude_std_lower_initial_radps"] = min(
                5.0,
                magnitude,
            )
            profile["spin_magnitude_std_lower_max_radps"] = magnitude
            upper_room = float(profile["spin_magnitude_max_radps"]) - magnitude
            profile["spin_magnitude_std_upper_initial_radps"] = min(
                5.0,
                upper_room,
            )
            profile["spin_magnitude_std_upper_max_radps"] = upper_room
        temp_out.write_text(
            json.dumps(
                manifest_document,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_mod, _, _, _ = builder._mdp_modules(repo_root)
        manifest_mod.load_action_ball_manifest(
            temp_out,
            verify_referenced_assets=True,
            repo_root=repo_root,
        )
        _copy_exclusive(temp_out, out_path)


def _load_formal_fitted_gate(repo_root: Path):
    gate_path = (
        repo_root / "hope_training/whole_body_tracking/scripts/"
        "mujoco_teacher_motion_fitted_ball_gate.py"
    )
    return _load_module("_fresh_n5_formal_fitted_gate", gate_path)


def _build_fitted_gate_contracts(
    *,
    gate: Any,
    repo_root: Path,
    profile_pins: Mapping[str, Any],
) -> Dict[str, Any]:
    def pinned_map(paths: Mapping[str, Path]) -> Dict[str, str]:
        result = {}
        for name, path in paths.items():
            resolved = Path(path).resolve()
            _repo_relative(resolved, repo_root, "fitted Gate dependency %s" % name)
            if not resolved.is_file():
                raise FreshN5BuildError(
                    "fitted Gate dependency is missing: %s" % resolved
                )
            result[str(name)] = _sha256_file(resolved)
        return dict(sorted(result.items()))

    contact_model_path = Path(gate.CONTACT_MODEL_PATH).resolve()
    contact_model_sha = _sha256_file(contact_model_path)
    if contact_model_sha != gate.CONTACT_MODEL_SHA256:
        raise FreshN5BuildError(
            "formal fitted contact-model bytes drifted from Gate constant"
        )
    runtime_sources = pinned_map(gate.RUNTIME_SOURCE_PATHS)
    execution_sources = pinned_map(gate.RUNTIME_EXECUTION_SOURCE_PATHS)
    execution_data = pinned_map(gate.RUNTIME_EXECUTION_DATA_PATHS)
    scene_paths = {
        "scripts/mujoco_table_scene.py": (repo_root / "scripts/mujoco_table_scene.py"),
        "scripts/audit_motion_schema2_table_net_clearance.py": (
            repo_root / "scripts/audit_motion_schema2_table_net_clearance.py"
        ),
        "table_tennis/geometry.py": (
            repo_root / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/geometry.py"
        ),
        "table_tennis/table_frame.py": (
            repo_root / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/table_tennis/table_frame.py"
        ),
    }
    face_mesh = {
        str(name): _sha256_file(Path(gate.FACE_MESH_PATHS[sign]))
        for name, sign in gate.FACE_MESH_PIN_KEYS.items()
    }
    geometry_path = repo_root / MDP_DIR_REL / "racket_contact_geometry.py"
    geometry_sha = profile_pins["geometry_source_sha256"]
    return {
        "racket_geometry_contract": {
            "schema_version": 2,
            "semantics": "exact_face_contact_v2",
            "ball_target_point": "physical_ball_center_at_native_contact",
            "site_target_mapping": "site_target_from_ball_center",
            "face_velocity_mapping": (
                "site_linear_plus_omega_cross_face_center_offset"
            ),
            "source_path": _repo_relative(
                geometry_path,
                repo_root,
                "racket geometry source",
            ),
            "source_sha256": _sha256_file(geometry_path),
            "geometry_source_sha256": geometry_sha,
        },
        "physical_contact_contract": {
            "schema_version": int(gate.CONTACT_CONTRACT_VERSION),
            "authority": str(gate.CONTACT_AUTHORITY),
            "native_ball_contact_disabled": True,
            "contact_model_path": _repo_relative(
                contact_model_path,
                repo_root,
                "formal contact model",
            ),
            "contact_model_sha256": contact_model_sha,
            "runtime_source_sha256": runtime_sources,
            "runtime_execution_source_sha256": execution_sources,
            "runtime_execution_data_sha256": execution_data,
            "convergence_timestep_s": list(gate.DEFAULT_DT_S),
            "venue_yaml_sha256": profile_pins["raw"]["venue_yaml_sha256"],
            "scene_source_sha256": pinned_map(scene_paths),
            "selected_face_mesh_sha256": dict(sorted(face_mesh.items())),
        },
    }


def _reverse_fitted_flight_step(
    *,
    gate: Any,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    spin_radps: np.ndarray,
    duration_s: float,
    venue: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """One negative-time RK4 step using the formal Gate acceleration.

    The Gate's public ``advance_fitted_flight`` intentionally treats negative
    durations as identity.  Search therefore integrates its exact acceleration
    field with a negative RK4 step, and every accepted result is independently
    replayed forward by the Gate's own positive-time primitive at two dt values.
    """

    p = np.asarray(position_m, np.float64)
    v = np.asarray(velocity_mps, np.float64)
    w = np.asarray(spin_radps, np.float64)
    h = -abs(float(duration_s))

    def derivative(velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return velocity, gate.aero_acceleration(velocity, w, venue)

    k1p, k1v = derivative(v)
    k2p, k2v = derivative(v + 0.5 * h * k1v)
    k3p, k3v = derivative(v + 0.5 * h * k2v)
    k4p, k4v = derivative(v + h * k3v)
    return (
        p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def _reverse_to_last_table_bounce(
    *,
    gate: Any,
    contact_position_m: Sequence[float],
    contact_velocity_mps: Sequence[float],
    contact_spin_radps: Sequence[float],
    maximum_duration_s: float,
    center_surface_z_m: float,
    venue: Any,
    step_s: float = 0.0005,
) -> Dict[str, Any]:
    p = np.asarray(contact_position_m, np.float64)
    v = np.asarray(contact_velocity_mps, np.float64)
    w = np.asarray(contact_spin_radps, np.float64)
    if p[2] <= center_surface_z_m:
        raise FreshN5BuildError(
            "teacher contact is not above the ball-centre table plane"
        )
    elapsed = 0.0
    while elapsed + 1.0e-12 < maximum_duration_s:
        duration = min(step_s, maximum_duration_s - elapsed)
        p_next, v_next = _reverse_fitted_flight_step(
            gate=gate,
            position_m=p,
            velocity_mps=v,
            spin_radps=w,
            duration_s=duration,
            venue=venue,
        )
        if p[2] > center_surface_z_m >= p_next[2]:
            alpha = float((p[2] - center_surface_z_m) / max(p[2] - p_next[2], 1.0e-12))
            bounce_p = p + alpha * (p_next - p)
            bounce_v = v + alpha * (v_next - v)
            return {
                "bounce_to_contact_s": elapsed + alpha * duration,
                "position_m": bounce_p,
                "velocity_plus_mps": bounce_v,
                "spin_plus_radps": w.copy(),
                "reverse_step_s": step_s,
            }
        p, v = p_next, v_next
        elapsed += duration
    raise FreshN5BuildError(
        "reverse fitted flight did not encounter the last table bounce "
        "inside the registered contact horizon"
    )


def _invert_fitted_table_contact(
    *,
    gate: Any,
    velocity_plus_mps: Sequence[float],
    spin_plus_radps: Sequence[float],
    venue: Any,
) -> Dict[str, Any]:
    """Analytic inverse of the registered table model for ``b_t == 0``.

    The fitted table normal is +Z.  Eliminating pre-impact tangential velocity
    and spin reduces the tangential impulse to one scalar.  Both the raw-grip
    and Coulomb-cap branches are evaluated by the formal ``fitted_contact``
    primitive; the inverse is accepted only when its replay is numerically
    exact.
    """

    if abs(float(venue.table_b_t)) > 1.0e-15:
        raise FreshN5BuildError(
            "analytic one-bounce inverse is registered only for table_b_t == 0"
        )
    v_plus = np.asarray(velocity_plus_mps, np.float64)
    w_plus = np.asarray(spin_plus_radps, np.float64)
    if v_plus.shape != (3,) or w_plus.shape != (3,):
        raise FreshN5BuildError("table inverse target state has wrong shape")
    if v_plus[2] <= 1.0e-6:
        raise FreshN5BuildError(
            "last-bounce post-impact vertical velocity is not positive"
        )
    radius = float(venue.ball_radius)
    inertia_coeff = float(venue.inertia_coeff)
    restitution = float(venue.table_e)
    if restitution <= 0.0:
        raise FreshN5BuildError("table restitution must be positive")
    base = np.asarray(
        (
            v_plus[0] - radius * w_plus[1],
            v_plus[1] + radius * w_plus[0],
        ),
        np.float64,
    )
    base_norm = float(np.linalg.norm(base))
    if base_norm <= 1.0e-12:
        direction = np.asarray((1.0, 0.0), np.float64)
        q_raw = 0.0
    else:
        direction = base / base_norm
        multiplier = 1.0 + 1.0 / inertia_coeff
        denominator = 1.0 - float(venue.table_a_t) * multiplier
        if denominator <= 1.0e-8:
            raise FreshN5BuildError(
                "registered table tangential inverse is singular/nonpositive"
            )
        q_raw = float(venue.table_a_t) * base_norm / denominator
    v_minus_z = -v_plus[2] / restitution
    q_cap = float(venue.table_mu) * (1.0 + restitution) * abs(v_minus_z)
    q = min(q_raw, q_cap)
    delta_v_t = -q * direction
    v_minus = np.asarray(
        (
            v_plus[0] - delta_v_t[0],
            v_plus[1] - delta_v_t[1],
            v_minus_z,
        ),
        np.float64,
    )
    w_minus = np.asarray(
        (
            w_plus[0] - delta_v_t[1] / (inertia_coeff * radius),
            w_plus[1] + delta_v_t[0] / (inertia_coeff * radius),
            w_plus[2],
        ),
        np.float64,
    )
    replay = gate.fitted_contact(
        v_minus,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        w_minus,
        e_eff=venue.table_e,
        a_t=venue.table_a_t,
        b_t=venue.table_b_t,
        mu=venue.table_mu,
    )
    velocity_error = float(np.linalg.norm(replay["velocity_plus_mps"] - v_plus))
    spin_error = float(np.linalg.norm(replay["spin_plus_radps"] - w_plus))
    if velocity_error > 1.0e-8 or spin_error > 1.0e-6:
        raise FreshN5BuildError(
            "formal table-contact inverse replay mismatch "
            "(velocity %.3g, spin %.3g)" % (velocity_error, spin_error)
        )
    return {
        "velocity_minus_mps": v_minus,
        "spin_minus_radps": w_minus,
        "velocity_plus_mps": v_plus,
        "spin_plus_radps": w_plus,
        "tangential_impulse_speed_mps": q,
        "raw_branch_impulse_speed_mps": q_raw,
        "cap_branch_limit_mps": q_cap,
        "cap_binds": bool(q_cap < q_raw),
        "replay_velocity_error_mps": velocity_error,
        "replay_spin_error_radps": spin_error,
    }


def _replay_one_bounce_launch(
    *,
    gate: Any,
    launch_position_m: Sequence[float],
    launch_velocity_mps: Sequence[float],
    launch_spin_radps: Sequence[float],
    duration_s: float,
    venue: Any,
    table_profile: Mapping[str, float],
    step_s: float,
) -> Dict[str, Any]:
    p = np.asarray(launch_position_m, np.float64).copy()
    v = np.asarray(launch_velocity_mps, np.float64).copy()
    w = np.asarray(launch_spin_radps, np.float64).copy()
    elapsed = 0.0
    bounce_count = 0
    bounce_state = None
    net_crossing = None
    while elapsed + 1.0e-12 < duration_s:
        dt = min(float(step_s), duration_s - elapsed)
        p_next, v_next = gate.advance_fitted_flight(p, v, w, dt, venue)
        if (
            net_crossing is None
            and p[0] > float(table_profile["net_x_m"])
            and p_next[0] <= float(table_profile["net_x_m"])
        ):
            alpha_net = float(
                (p[0] - float(table_profile["net_x_m"]))
                / max(p[0] - p_next[0], 1.0e-12)
            )
            net_crossing = {
                "time_s": elapsed + alpha_net * dt,
                "ball_center_z_m": float(p[2] + alpha_net * (p_next[2] - p[2])),
            }
        hit = gate.swept_table_crossing(
            p,
            p_next,
            v,
            center_surface_z_m=float(table_profile["center_surface_z_m"]),
            near_x_m=float(table_profile["eroded_near_x_m"]),
            far_x_m=float(table_profile["eroded_far_x_m"]),
            half_width_m=float(table_profile["eroded_half_width_m"]),
        )
        if hit is None:
            p, v = p_next, v_next
            elapsed += dt
            continue
        alpha, point = hit
        bounce_count += 1
        event_time = elapsed + float(alpha) * dt
        event_v = v + float(alpha) * (v_next - v)
        fitted = gate.fitted_contact(
            event_v,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            w,
            e_eff=venue.table_e,
            a_t=venue.table_a_t,
            b_t=venue.table_b_t,
            mu=venue.table_mu,
        )
        bounce_state = {
            "time_s": event_time,
            "position_m": np.asarray(point, np.float64).tolist(),
            "velocity_minus_mps": event_v.tolist(),
            "spin_minus_radps": w.tolist(),
            "velocity_plus_mps": fitted["velocity_plus_mps"].tolist(),
            "spin_plus_radps": fitted["spin_plus_radps"].tolist(),
        }
        remaining = (1.0 - float(alpha)) * dt
        p = np.asarray(point, np.float64) + np.asarray((0.0, 0.0, 1.0e-7))
        v = np.asarray(fitted["velocity_plus_mps"], np.float64)
        w = np.asarray(fitted["spin_plus_radps"], np.float64)
        if remaining > 0.0:
            p, v = gate.advance_fitted_flight(p, v, w, remaining, venue)
        elapsed += dt
    return {
        "arrival_position_m": p.tolist(),
        "arrival_velocity_mps": v.tolist(),
        "arrival_spin_radps": w.tolist(),
        "bounce_count": bounce_count,
        "bounce_state": bounce_state,
        "net_crossing": net_crossing,
        "step_s": float(step_s),
    }


def solve_one_bounce_launch(
    *,
    gate: Any,
    target_position_m: Sequence[float],
    target_velocity_mps: Sequence[float],
    target_spin_radps: Sequence[float],
    time_to_contact_s: float,
    venue: Any,
    table_profile: Mapping[str, float],
    seed: int,
    proposal_count: int = 16,
) -> Dict[str, Any]:
    if proposal_count < 2:
        raise FreshN5BuildError("one-bounce shooting needs at least two proposals")
    ttc = float(time_to_contact_s)
    if not math.isfinite(ttc) or ttc <= 0.10:
        raise FreshN5BuildError("one-bounce shooting TTC must exceed 0.10 s")
    target_p = np.asarray(target_position_m, np.float64)
    target_v = np.asarray(target_velocity_mps, np.float64)
    target_w = np.asarray(target_spin_radps, np.float64)
    reverse = _reverse_to_last_table_bounce(
        gate=gate,
        contact_position_m=target_p,
        contact_velocity_mps=target_v,
        contact_spin_radps=target_w,
        maximum_duration_s=ttc - 0.10,
        center_surface_z_m=float(table_profile["center_surface_z_m"]),
        venue=venue,
    )
    bounce_p = np.asarray(reverse["position_m"], np.float64)
    if not (
        float(table_profile["eroded_near_x_m"])
        <= bounce_p[0]
        <= float(table_profile["eroded_far_x_m"])
        and abs(bounce_p[1]) <= float(table_profile["eroded_half_width_m"])
    ):
        raise FreshN5BuildError(
            "last-bounce inverse crosses outside the formal eroded table footprint"
        )
    table_inverse = _invert_fitted_table_contact(
        gate=gate,
        velocity_plus_mps=reverse["velocity_plus_mps"],
        spin_plus_radps=reverse["spin_plus_radps"],
        venue=venue,
    )
    pre_bounce_duration = ttc - float(reverse["bounce_to_contact_s"])
    launch_p = bounce_p + np.asarray((0.0, 0.0, 1.0e-7))
    launch_v = np.asarray(table_inverse["velocity_minus_mps"], np.float64)
    launch_w = np.asarray(table_inverse["spin_minus_radps"], np.float64)
    elapsed = 0.0
    while elapsed + 1.0e-12 < pre_bounce_duration:
        dt = min(0.0005, pre_bounce_duration - elapsed)
        launch_p, launch_v = _reverse_fitted_flight_step(
            gate=gate,
            position_m=launch_p,
            velocity_mps=launch_v,
            spin_radps=launch_w,
            duration_s=dt,
            venue=venue,
        )
        elapsed += dt
    if (
        launch_v[0] >= -1.0e-6
        or launch_p[2] <= float(table_profile["center_surface_z_m"]) + 0.005
        or launch_p[0] <= float(table_profile["net_x_m"]) + 0.05
    ):
        raise FreshN5BuildError(
            "analytic one-bounce launch violates birth-side geometry"
        )
    launch_speed = float(np.linalg.norm(launch_v))
    launch_spin = float(np.linalg.norm(launch_w))
    if launch_speed > 20.0 or launch_spin > 1200.0:
        raise FreshN5BuildError(
            "analytic one-bounce launch exceeds fixed search domain "
            "(speed %.3f m/s, spin %.3f rad/s)" % (launch_speed, launch_spin)
        )

    rng = np.random.default_rng(seed)
    candidate_rows = []
    for index in range(proposal_count):
        if index == 0:
            candidate_p = launch_p.copy()
            candidate_v = launch_v.copy()
            candidate_w = launch_w.copy()
        else:
            candidate_p = launch_p + rng.normal(0.0, 0.002, size=3)
            candidate_v = launch_v + rng.normal(0.0, 0.01, size=3)
            candidate_w = launch_w + rng.normal(0.0, 0.10, size=3)
        coarse = _replay_one_bounce_launch(
            gate=gate,
            launch_position_m=candidate_p,
            launch_velocity_mps=candidate_v,
            launch_spin_radps=candidate_w,
            duration_s=ttc,
            venue=venue,
            table_profile=table_profile,
            step_s=0.001,
        )
        fine = _replay_one_bounce_launch(
            gate=gate,
            launch_position_m=candidate_p,
            launch_velocity_mps=candidate_v,
            launch_spin_radps=candidate_w,
            duration_s=ttc,
            venue=venue,
            table_profile=table_profile,
            step_s=0.0005,
        )
        fine_p = np.asarray(fine["arrival_position_m"], np.float64)
        fine_v = np.asarray(fine["arrival_velocity_mps"], np.float64)
        fine_w = np.asarray(fine["arrival_spin_radps"], np.float64)
        coarse_p = np.asarray(coarse["arrival_position_m"], np.float64)
        coarse_v = np.asarray(coarse["arrival_velocity_mps"], np.float64)
        coarse_w = np.asarray(coarse["arrival_spin_radps"], np.float64)
        residuals = {
            "arrival_position_m": float(np.linalg.norm(fine_p - target_p)),
            "arrival_velocity_mps": float(np.linalg.norm(fine_v - target_v)),
            "arrival_spin_radps": float(np.linalg.norm(fine_w - target_w)),
            "dual_dt_position_m": float(np.linalg.norm(fine_p - coarse_p)),
            "dual_dt_velocity_mps": float(np.linalg.norm(fine_v - coarse_v)),
            "dual_dt_spin_radps": float(np.linalg.norm(fine_w - coarse_w)),
        }
        reasons = []
        if fine["bounce_count"] != 1 or coarse["bounce_count"] != 1:
            reasons.append("incoming_bounce_count_not_one")
        if fine["net_crossing"] is None or coarse["net_crossing"] is None:
            reasons.append("incoming_net_not_crossed")
        else:
            if float(fine["net_crossing"]["ball_center_z_m"]) <= float(
                table_profile["net_top_z_m"]
            ) or float(coarse["net_crossing"]["ball_center_z_m"]) <= float(
                table_profile["net_top_z_m"]
            ):
                reasons.append("incoming_net_not_cleared")
        tolerances = {
            "arrival_position_m": 0.002,
            "arrival_velocity_mps": 0.02,
            "arrival_spin_radps": 0.05,
            "dual_dt_position_m": 0.001,
            "dual_dt_velocity_mps": 0.01,
            "dual_dt_spin_radps": 0.02,
        }
        for name, tolerance in tolerances.items():
            if residuals[name] > tolerance:
                reasons.append("%s_out_of_tolerance" % name)
        candidate_rows.append(
            {
                "proposal_index": index,
                "launch_position_w_m": candidate_p.tolist(),
                "launch_velocity_w_mps": candidate_v.tolist(),
                "launch_spin_w_radps": candidate_w.tolist(),
                "coarse_replay": coarse,
                "fine_replay": fine,
                "residuals": residuals,
                "rejection_reasons": reasons,
                "admitted": not reasons,
            }
        )
    admitted = [row for row in candidate_rows if row["admitted"]]
    reason_counts: Dict[str, int] = {}
    for row in candidate_rows:
        for reason in row["rejection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if not admitted:
        raise FreshN5BuildError(
            "one-bounce shooting found no dual-dt legal launch; "
            "proposal_count=%d reasons=%r" % (proposal_count, reason_counts)
        )
    selected = min(
        admitted,
        key=lambda row: (
            row["residuals"]["arrival_position_m"],
            row["residuals"]["arrival_velocity_mps"],
            row["proposal_index"],
        ),
    )
    launch_payload = {
        "activation_time_s": 0.0,
        "position_w_m": selected["launch_position_w_m"],
        "velocity_w_mps": selected["launch_velocity_w_mps"],
        "spin_w_radps": selected["launch_spin_w_radps"],
        "required_incoming_table_bounces": 1,
    }
    return {
        "status": "PASS",
        "seed": int(seed),
        "proposal_count": int(proposal_count),
        "admitted_count": len(admitted),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "selected_proposal_index": int(selected["proposal_index"]),
        "fixed_search_domain": {
            "launch_speed_max_mps": 20.0,
            "launch_spin_max_radps": 1200.0,
            "birth_x_min_m": float(table_profile["net_x_m"]) + 0.05,
            "birth_z_min_m": float(table_profile["center_surface_z_m"]) + 0.005,
        },
        "formal_forward_primitives": [
            "advance_fitted_flight",
            "swept_table_crossing",
            "fitted_contact",
        ],
        "reverse_last_bounce": {
            **reverse,
            "position_m": np.asarray(reverse["position_m"]).tolist(),
            "velocity_plus_mps": np.asarray(reverse["velocity_plus_mps"]).tolist(),
            "spin_plus_radps": np.asarray(reverse["spin_plus_radps"]).tolist(),
        },
        "table_contact_inverse": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in table_inverse.items()
        },
        "launch_state": launch_payload,
        "selected_coarse_replay": selected["coarse_replay"],
        "selected_fine_replay": selected["fine_replay"],
        "selected_residuals": selected["residuals"],
        "tolerances": tolerances,
        "proposals": candidate_rows,
    }


def _shoot_physical_launch_map(
    *,
    base_manifest_path: Path,
    batch_path: Path,
    profile_pins_path: Path,
    incoming_dist_path: Path,
    incoming_dist_sha256: str,
    repo_root: Path,
    output_root: Path,
    seed: int,
    proposals_per_case: int,
) -> Path:
    base = _read_json(base_manifest_path, "strict base manifest")
    batch = _read_json(batch_path, "fresh N5 batch")
    if tuple(base.get("action_order", ())) != FRESH_N5_ACTION_ORDER:
        raise FreshN5BuildError("shooting base manifest order drifted")
    _load_incoming_distribution(incoming_dist_path, incoming_dist_sha256)
    pins = _validate_profile_pins(profile_pins_path, repo_root)
    gate = _load_formal_fitted_gate(repo_root)
    venue_path = repo_root / pins["raw"]["physics_payload"]["venue_source"]["path"]
    venue = gate.load_venue_yaml(
        venue_path,
        pins["raw"]["physics_payload"]["venue_source"]["file_sha256"],
    )
    geometry = pins["raw"]["physics_payload"]["geometry_and_grading"]
    eroded_margin = float(venue.ball_radius) + float(
        gate.FORMAL_SHADOW_CLEARANCE_GUARD_M
    )
    table_profile = {
        "center_surface_z_m": float(geometry["ball_center_surface_z_m"]),
        "eroded_near_x_m": float(geometry["opponent_near_x_m"]) + eroded_margin,
        "eroded_far_x_m": float(geometry["opponent_far_x_m"]) - eroded_margin,
        "eroded_half_width_m": float(geometry["table_half_width_m"]) - eroded_margin,
        "net_x_m": float(geometry["net_x_m"]),
        "net_top_z_m": float(geometry["ball_center_net_top_z_m"]),
    }
    actions = {row["action_id"]: row for row in base["actions"]}
    units = {row["uid"]: row for row in batch["units"]}
    raw_dir = output_root / "launch_raw_inputs"
    upstream_dir = output_root / "launch_upstream"
    artifact_dir = output_root / "launch_artifacts"
    raw_dir.mkdir()
    upstream_dir.mkdir()
    artifact_dir.mkdir()
    rows = []
    for slot, action_id in enumerate(FRESH_N5_ACTION_ORDER):
        action = actions[action_id]
        unit = units[action_id]
        center_contact = np.asarray(
            (
                float(unit["ball_pos_hit_hope_m"][0]) + 0.5,
                float(unit["ball_pos_hit_hope_m"][1]) + 0.7625,
                float(unit["ball_pos_hit_hope_m"][2]) + 0.76,
            ),
            np.float64,
        )
        incoming_velocity = np.asarray(unit["v_in_fit_hope_ms"], np.float64)
        incoming_spin = np.asarray(
            unit.get("w_in_fit_hope_radps", (0.0, 0.0, 0.0)),
            np.float64,
        )
        profile = action["ball_profile"]
        support_contact = center_contact.copy()
        support_delta = min(
            0.01,
            max(
                1.0e-4,
                0.5 * float(profile["contact_offset_std_upper_initial_m"][1]),
            ),
        )
        base_yaw = math.radians(float(unit["yaw_before_deg"]))
        support_contact[0] += -math.sin(base_yaw) * support_delta
        support_contact[1] += math.cos(base_yaw) * support_delta
        center_ttc = float(profile["time_to_contact_center_s"])
        support_ttc = min(
            float(profile["time_to_contact_max_s"]),
            center_ttc
            + max(
                0.02,
                0.5 * float(profile["time_to_contact_std_upper_initial_s"]),
            ),
        )
        center_receipt = solve_one_bounce_launch(
            gate=gate,
            target_position_m=center_contact,
            target_velocity_mps=incoming_velocity,
            target_spin_radps=incoming_spin,
            time_to_contact_s=center_ttc,
            venue=venue,
            table_profile=table_profile,
            seed=int(seed) + 2003 * slot,
            proposal_count=proposals_per_case,
        )
        support_receipt = solve_one_bounce_launch(
            gate=gate,
            target_position_m=support_contact,
            target_velocity_mps=incoming_velocity,
            target_spin_radps=incoming_spin,
            time_to_contact_s=support_ttc,
            venue=venue,
            table_profile=table_profile,
            seed=int(seed) + 2003 * slot + 1,
            proposal_count=proposals_per_case,
        )
        case_launches = {}
        for name, receipt in (
            ("center", center_receipt),
            ("support", support_receipt),
        ):
            payload = dict(receipt["launch_state"])
            case_launches[name] = {
                **payload,
                "state_sha256": _canonical_sha256(payload),
            }
        launch_state = {
            "source": "pre_registered_native_shooting_receipt_v1",
            **center_receipt["launch_state"],
        }
        raw_input = {
            "schema_version": 1,
            "artifact_type": "native_shooting_solver_input_v1",
            "pre_registered": True,
            "action_id": action_id,
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
            "coordinate_frame": "mujoco_world",
            "units": UNITS,
            "launch_state": launch_state,
            "target_contact_state": {
                "time_to_contact_s": center_ttc,
                "position_w_m": center_contact.tolist(),
                "velocity_w_mps": incoming_velocity.tolist(),
                "spin_w_radps": incoming_spin.tolist(),
            },
            "incoming_distribution": {
                "path": _repo_relative(
                    incoming_dist_path,
                    repo_root,
                    "incoming distribution",
                ),
                "sha256": incoming_dist_sha256,
                "pool": "pooled_matchlike",
            },
            "search_contract": {
                "seed": center_receipt["seed"],
                "proposal_count": center_receipt["proposal_count"],
                "formal_forward_primitives": center_receipt[
                    "formal_forward_primitives"
                ],
                "dual_dt_s": [0.001, 0.0005],
                "fixed_search_domain": center_receipt["fixed_search_domain"],
            },
        }
        raw_path = raw_dir / ("%s.json" % action_id)
        _write_exclusive(raw_path, raw_input)
        raw_sha = _sha256_file(raw_path)
        upstream = {
            "schema_version": 1,
            "artifact_type": "native_shooting_solver_receipt_v1",
            "action_id": action_id,
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
            "coordinate_frame": "mujoco_world",
            "units": UNITS,
            "authorization": {
                "physical_gate_input_authorized": True,
                "hardware_authorized": False,
            },
            "status": "PASS",
            "pre_registered": True,
            "solver_input_sha256": raw_sha,
            "launch_state": launch_state,
            "center_shooting_receipt": center_receipt,
            "support_shooting_receipt": support_receipt,
            "producer": {
                "source_path": _repo_relative(
                    Path(__file__),
                    repo_root,
                    "fresh N5 producer",
                ),
                "source_sha256": _sha256_file(Path(__file__)),
                "formal_gate_source_path": _repo_relative(
                    Path(gate.__file__),
                    repo_root,
                    "formal fitted gate",
                ),
                "formal_gate_source_sha256": _sha256_file(Path(gate.__file__)),
            },
        }
        upstream["receipt_payload_sha256"] = _canonical_sha256(upstream)
        upstream_path = upstream_dir / ("%s.json" % action_id)
        _write_exclusive(upstream_path, upstream)
        upstream_sha = _sha256_file(upstream_path)
        source_artifact = {
            "schema_version": 1,
            "artifact_type": "pre_registered_native_shooting_receipt_v1",
            "action_id": action_id,
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
            "coordinate_frame": "mujoco_world",
            "units": UNITS,
            "authorization": {
                "physical_gate_input_authorized": True,
                "hardware_authorized": False,
            },
            "launch_state": launch_state,
            "upstream_evidence_path": _repo_relative(
                upstream_path,
                repo_root,
                "%s shooting upstream" % action_id,
            ),
            "upstream_evidence_sha256": upstream_sha,
            "shooting_solver_input_sha256": raw_sha,
        }
        artifact_path = artifact_dir / ("%s.json" % action_id)
        _write_exclusive(artifact_path, source_artifact)
        artifact_sha = _sha256_file(artifact_path)
        physical_payload = {
            **launch_state,
            "source_artifact_path": _repo_relative(
                artifact_path,
                repo_root,
                "%s shooting source artifact" % action_id,
            ),
            "source_artifact_sha256": artifact_sha,
        }
        physical_launch = {
            **physical_payload,
            "state_sha256": _canonical_sha256(physical_payload),
        }
        rows.append(
            {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "physical_ball_launch": physical_launch,
                "case_launches": case_launches,
                "shooting_evidence": {
                    "raw_input_path": _repo_relative(
                        raw_path,
                        repo_root,
                        "%s shooting raw input" % action_id,
                    ),
                    "raw_input_sha256": raw_sha,
                    "upstream_receipt_path": _repo_relative(
                        upstream_path,
                        repo_root,
                        "%s shooting upstream" % action_id,
                    ),
                    "upstream_receipt_sha256": upstream_sha,
                    "source_artifact_path": physical_payload["source_artifact_path"],
                    "source_artifact_sha256": artifact_sha,
                },
            }
        )
    document = {
        "schema_version": 1,
        "artifact_type": "fresh_n5_physical_launch_map_v1",
        "action_order": list(FRESH_N5_ACTION_ORDER),
        "base_manifest_raw_sha256": _sha256_file(base_manifest_path),
        "incoming_distribution": {
            "path": _repo_relative(
                incoming_dist_path,
                repo_root,
                "incoming distribution",
            ),
            "sha256": incoming_dist_sha256,
            "pool": "pooled_matchlike",
        },
        "formal_forward_source": {
            "path": _repo_relative(
                Path(gate.__file__),
                repo_root,
                "formal fitted gate",
            ),
            "sha256": _sha256_file(Path(gate.__file__)),
        },
        "selector_executed": False,
        "action_identity_frozen": True,
        "actions": rows,
    }
    document["content_sha256"] = _canonical_sha256(document)
    map_path = output_root / "fresh_n5_physical_launch_map.json"
    _write_exclusive(map_path, document)
    return map_path


def _validate_launch_row(raw: Any, label: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise FreshN5BuildError("%s must be an object" % label)
    expected = {
        "activation_time_s",
        "position_w_m",
        "velocity_w_mps",
        "spin_w_radps",
        "required_incoming_table_bounces",
        "state_sha256",
    }
    if set(raw) != expected:
        raise FreshN5BuildError("%s key set is not exact" % label)
    payload = {
        key: raw[key]
        for key in (
            "activation_time_s",
            "position_w_m",
            "velocity_w_mps",
            "spin_w_radps",
            "required_incoming_table_bounces",
        )
    }
    activation = float(payload["activation_time_s"])
    if not math.isfinite(activation) or activation < 0.0:
        raise FreshN5BuildError("%s activation time is invalid" % label)
    _finite_vec(payload["position_w_m"], 3, "%s position" % label)
    velocity = _finite_vec(payload["velocity_w_mps"], 3, "%s velocity" % label)
    _finite_vec(payload["spin_w_radps"], 3, "%s spin" % label)
    if velocity[0] >= -1.0e-6:
        raise FreshN5BuildError("%s velocity is not incoming" % label)
    if payload["required_incoming_table_bounces"] != 1:
        raise FreshN5BuildError(
            "%s must require exactly one incoming table bounce" % label
        )
    if _canonical_sha256(payload) != raw["state_sha256"]:
        raise FreshN5BuildError("%s state SHA mismatch" % label)
    return dict(raw)


def _load_physical_launches(
    path: Path,
    repo_root: Path,
    strict_actions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    document = _read_json(path, "physical launch map")
    if (
        document.get("schema_version") != 1
        or document.get("artifact_type") != "fresh_n5_physical_launch_map_v1"
        or tuple(document.get("action_order", ())) != FRESH_N5_ACTION_ORDER
    ):
        raise FreshN5BuildError("physical launch map identity/order is invalid")
    rows = document.get("actions")
    if not isinstance(rows, list) or len(rows) != len(FRESH_N5_ACTION_ORDER):
        raise FreshN5BuildError("physical launch map must contain five actions")
    result = {}
    for expected_action, row in zip(FRESH_N5_ACTION_ORDER, rows):
        if not isinstance(row, dict) or row.get("action_id") != expected_action:
            raise FreshN5BuildError("physical launch map action order drifted")
        strict = strict_actions[expected_action]
        if (
            row.get("action_uid") != strict["action_uid"]
            or row.get("motion_sha256") != strict["motion_sha256"]
        ):
            raise FreshN5BuildError(
                "%s physical launch identity drifted" % expected_action
            )
        physical_launch = row.get("physical_ball_launch")
        if not isinstance(physical_launch, dict):
            raise FreshN5BuildError("%s lacks physical_ball_launch" % expected_action)
        expected_launch_keys = {
            "source",
            "activation_time_s",
            "position_w_m",
            "velocity_w_mps",
            "spin_w_radps",
            "source_artifact_path",
            "source_artifact_sha256",
            "required_incoming_table_bounces",
            "state_sha256",
        }
        if set(physical_launch) != expected_launch_keys:
            raise FreshN5BuildError(
                "%s physical_ball_launch key set is not exact" % expected_action
            )
        if physical_launch["source"] not in (
            "recorded_pre_hit_state_v1",
            "pre_registered_native_shooting_receipt_v1",
        ):
            raise FreshN5BuildError(
                "%s physical launch source is not formal" % expected_action
            )
        _finite_vec(
            physical_launch["position_w_m"],
            3,
            "%s physical launch position" % expected_action,
        )
        top_velocity = _finite_vec(
            physical_launch["velocity_w_mps"],
            3,
            "%s physical launch velocity" % expected_action,
        )
        _finite_vec(
            physical_launch["spin_w_radps"],
            3,
            "%s physical launch spin" % expected_action,
        )
        if top_velocity[0] >= -1.0e-6:
            raise FreshN5BuildError(
                "%s physical launch is not incoming" % expected_action
            )
        if physical_launch["required_incoming_table_bounces"] != 1:
            raise FreshN5BuildError(
                "%s physical launch must require exactly one bounce" % expected_action
            )
        source_path = repo_root / str(physical_launch.get("source_artifact_path", ""))
        expected_source_sha = _require_sha(
            physical_launch.get("source_artifact_sha256"),
            "%s launch source artifact" % expected_action,
        )
        if (
            not source_path.is_file()
            or _sha256_file(source_path) != expected_source_sha
        ):
            raise FreshN5BuildError(
                "%s physical launch source artifact drifted" % expected_action
            )
        launch_payload = {
            key: physical_launch[key]
            for key in (
                "source",
                "activation_time_s",
                "position_w_m",
                "velocity_w_mps",
                "spin_w_radps",
                "source_artifact_path",
                "source_artifact_sha256",
                "required_incoming_table_bounces",
            )
        }
        if _canonical_sha256(launch_payload) != physical_launch.get("state_sha256"):
            raise FreshN5BuildError(
                "%s physical_ball_launch state SHA mismatch" % expected_action
            )
        cases = row.get("case_launches")
        if not isinstance(cases, dict) or set(cases) != {"center", "support"}:
            raise FreshN5BuildError(
                "%s case_launches must contain exact center/support" % expected_action
            )
        result[expected_action] = {
            "physical_ball_launch": dict(physical_launch),
            "center": _validate_launch_row(
                cases["center"], "%s center launch" % expected_action
            ),
            "support": _validate_launch_row(
                cases["support"], "%s support launch" % expected_action
            ),
        }
    return result


def _tensor_list(value: Any) -> List[float]:
    return [float(component) for component in value.detach().cpu().tolist()]


def _build_solver_bundle(
    *,
    base_manifest_path: Path,
    batch_path: Path,
    prototype_path: Path,
    profile_pins_path: Path,
    physical_launches_path: Path,
    repo_root: Path,
    receipt_dir: Path,
) -> Dict[str, Any]:
    import torch

    base_raw_sha = _sha256_file(base_manifest_path)
    base = _read_json(base_manifest_path, "strict base manifest")
    if (
        base.get("schema_version") != 3
        or tuple(base.get("action_order", ())) != FRESH_N5_ACTION_ORDER
        or base.get("mobility_mode") != "no_move"
    ):
        raise FreshN5BuildError(
            "base manifest is not exact strict fresh-N5/no_move schema-v3"
        )
    actions = base.get("actions")
    if not isinstance(actions, list) or len(actions) != 5:
        raise FreshN5BuildError("strict base manifest must contain five actions")
    if any(
        "physical_ball_launch" in row or "physical_task_binding" in row
        for row in actions
    ):
        raise FreshN5BuildError(
            "strict training manifest contains forbidden gate-only extras"
        )
    strict_by_id = {row["action_id"]: row for row in actions}
    batch = _read_json(batch_path, "fresh N5 batch")
    batch_by_id = {row["uid"]: row for row in batch["units"]}
    pins = _validate_profile_pins(profile_pins_path, repo_root)
    formal_gate = _load_formal_fitted_gate(repo_root)
    gate_materialization_fields = _build_fitted_gate_contracts(
        gate=formal_gate,
        repo_root=repo_root,
        profile_pins=pins,
    )
    launches = _load_physical_launches(
        physical_launches_path,
        repo_root,
        strict_by_id,
    )

    mdp_modules = _load_mdp_package(repo_root)
    geometry = mdp_modules["racket_contact_geometry"]
    virtual_ball = mdp_modules["virtual_ball"]
    prototypes_module = mdp_modules["stroke_prototypes_torch"]
    continuous = mdp_modules["continuous_questions"]
    expected_motion_sha = [
        strict_by_id[action]["motion_sha256"] for action in FRESH_N5_ACTION_ORDER
    ]
    prototypes = prototypes_module.load_stroke_prototype_tensors(
        str(prototype_path),
        scope="upper",
        device="cpu",
        expected_sha256=_sha256_file(prototype_path),
        expected_motion_ids=FRESH_N5_ACTION_ORDER,
        expected_motion_sha256=expected_motion_sha,
    )
    venue_path = repo_root / pins["raw"]["physics_payload"]["venue_source"]["path"]
    prm = virtual_ball.load_venue_params(str(venue_path))
    cfg_raw = pins["raw"]["cfg"]
    solver_cfg = continuous.ContinuousQuestionCfg(
        fixed_direction=True,
        n_iters=int(cfg_raw["cq_n_iters"]),
        tol_m=float(cfg_raw["cq_tol_m"]),
        speed_budget=float(cfg_raw["cq_speed_budget"]),
    )
    planes = pins["raw"]["planes"]
    geometry_sha = pins["geometry_source_sha256"]

    overlay_actions = []
    for slot, action_id in enumerate(FRESH_N5_ACTION_ORDER):
        action = strict_by_id[action_id]
        unit = batch_by_id[action_id]
        motion_path = repo_root / action["motion_path"]
        strike = round(float(action["strike_phase"]) * (int(unit["T"]) - 1))
        state = _runtime_site_state(motion_path, strike, geometry)
        base_quat = torch.as_tensor(
            np.stack((state["root_quat_frame0_wxyz"],) * 2),
            dtype=torch.float32,
        )
        center_contact = np.asarray(
            (
                float(unit["ball_pos_hit_hope_m"][0]) + 0.5,
                float(unit["ball_pos_hit_hope_m"][1]) + 0.7625,
                float(unit["ball_pos_hit_hope_m"][2]) + 0.76,
            ),
            dtype=np.float32,
        )
        center_velocity = np.asarray(unit["v_in_fit_hope_ms"], dtype=np.float32)
        profile = action["ball_profile"]
        support_contact = center_contact.copy()
        support_delta = min(
            0.01,
            0.5 * float(profile["contact_offset_std_upper_initial_m"][1]),
        )
        if support_delta <= 1.0e-6:
            support_delta = min(
                0.01,
                0.5
                * (
                    float(profile["contact_offset_max_b_yaw_m"][1])
                    - float(profile["contact_offset_center_b_yaw_m"][1])
                ),
            )
        if support_delta <= 1.0e-6:
            raise FreshN5BuildError(
                "%s profile has no usable support perturbation" % action_id
            )
        base_yaw = math.radians(float(unit["yaw_before_deg"]))
        support_contact[0] += -math.sin(base_yaw) * support_delta
        support_contact[1] += math.cos(base_yaw) * support_delta
        contacts = torch.as_tensor(
            np.stack((center_contact, support_contact)),
            dtype=torch.float32,
        )
        velocities = torch.as_tensor(
            np.stack((center_velocity, center_velocity)),
            dtype=torch.float32,
        )
        incoming_spin = np.asarray(
            unit.get("w_in_fit_hope_radps", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        spins = torch.as_tensor(
            np.stack((incoming_spin, incoming_spin)),
            dtype=torch.float32,
        )
        aim = torch.as_tensor(
            np.stack((base["landing_aim"]["center_w_xy_m"],) * 2),
            dtype=torch.float32,
        )
        reference_rotation = _quat_rotation(state["racket_quat_wxyz"])
        reference_normal = reference_rotation.dot(np.asarray((0.0, 1.0, 0.0)))
        ref_normals = torch.as_tensor(
            np.stack((reference_normal, reference_normal)),
            dtype=torch.float32,
        )
        clip_ids = torch.as_tensor((slot, slot), dtype=torch.long)
        solved = continuous.solve_proposals(
            clip_ids,
            contacts,
            velocities,
            spins,
            aim,
            ref_normals,
            protos=prototypes,
            base_quat=base_quat,
            prm=prm,
            surface_z=float(planes["surface_z"]),
            net_x=float(planes["net_x"]),
            net_top_z=float(planes["net_top_z"]),
            cfg=solver_cfg,
            h=float(cfg_raw["vb_rollout_h"]),
            n_steps=int(cfg_raw["vb_rollout_steps"]),
        )
        if solved.ok.tolist() != [True, True]:
            raise FreshN5BuildError(
                "%s center/support fixed-action solver rejected: %r"
                % (action_id, solved.reason_counts)
            )
        root0 = state["root_position_frame0_w_m"]
        base_spawn = [float(root0[0]), float(root0[1]), float(root0[2])]
        ttc_center = float(profile["time_to_contact_center_s"])
        ttc_support = min(
            float(profile["time_to_contact_max_s"]),
            ttc_center
            + max(0.02, float(profile["time_to_contact_std_upper_initial_s"]) * 0.5),
        )
        if ttc_support <= ttc_center:
            ttc_support = ttc_center
        case_source = {
            "center": {
                "contact": contacts[0],
                "velocity": velocities[0],
                "spin": spins[0],
                "aim": aim[0],
                "v_racket": solved.v_racket[0],
                "normal": solved.n_racket[0],
                "residual": float(solved.resid_m[0]),
                "ttc": ttc_center,
                "launch": launches[action_id]["center"],
            },
            "support": {
                "contact": contacts[1],
                "velocity": velocities[1],
                "spin": spins[1],
                "aim": aim[1],
                "v_racket": solved.v_racket[1],
                "normal": solved.n_racket[1],
                "residual": float(solved.resid_m[1]),
                "ttc": ttc_support,
                "launch": launches[action_id]["support"],
            },
        }
        execution_identity = {
            "artifact_type": "frozen_ball_to_task_solver_execution_v1",
            "execution_id": "fresh-n5:%s:%s" % (base_raw_sha, action_id),
            "executed_before_gate": True,
            "solver_replayed_exact": True,
            "selector_executed": False,
            "action_identity_frozen": True,
            "action_switching_allowed": False,
            "hardware_authorized": False,
        }
        execution_sha = _canonical_sha256(execution_identity)

        cases = []
        for index, role in enumerate(CASE_ROLES):
            source_key = "support" if role == "support_positive" else "center"
            source = case_source[source_key]
            sample_seed = 2026072900 + 100 * slot + index
            launch = dict(source["launch"])
            if float(launch["activation_time_s"]) >= float(source["ttc"]):
                raise FreshN5BuildError(
                    "%s %s launch has no pre-contact lead" % (action_id, role)
                )
            proposal = {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "sample_seed": sample_seed,
                "sample_index": index,
                "ball_contact_w_m": _tensor_list(source["contact"]),
                "time_to_contact_s": float(source["ttc"]),
                "incoming_velocity_w_mps": _tensor_list(source["velocity"]),
                "incoming_spin_w_radps": _tensor_list(source["spin"]),
                "base_spawn_w_m": base_spawn,
                "base_goal_w_m": list(base_spawn),
                "landing_aim_w_xy_m": _tensor_list(source["aim"]),
                "launch": launch,
            }
            proposal_sha = _canonical_sha256(proposal)
            exact_geometry = geometry.solve_exact_face_contact(
                ball_contact_w_m=proposal["ball_contact_w_m"],
                racket_face_center_velocity_w_mps=_tensor_list(source["v_racket"]),
                solved_raw_a_normal_w=_tensor_list(source["normal"]),
                mount_normal_sign=int(action["mount_normal_sign"]),
                reference_racket_quat_wxyz=state["racket_quat_wxyz"].tolist(),
                reference_racket_angular_velocity_w_radps=(
                    state["racket_angular_velocity_w_radps"].tolist()
                ),
                reference_racket_site_speed_mps=float(
                    action["reference_racket_site_speed_mps"]
                ),
                teacher_rate_min=float(action["teacher_rate_min"]),
                teacher_rate_max=float(action["teacher_rate_max"]),
            )
            teacher_rate = float(exact_geometry.teacher_rate)
            scaled_t_hit = float(action["reference_t_hit_s"]) / teacher_rate
            scaled_t_cycle = float(action["reference_t_cycle_s"]) / teacher_rate
            task = {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "ball_proposal_sha256": proposal_sha,
                "mount_normal_sign": int(action["mount_normal_sign"]),
                "ball_contact_w_m": proposal["ball_contact_w_m"],
                "racket_site_target_w_m": list(exact_geometry.racket_site_target_w_m),
                "racket_normal_w": _tensor_list(source["normal"]),
                "reference_racket_quat_wxyz": state["racket_quat_wxyz"].tolist(),
                "reference_racket_angular_velocity_w_radps": (
                    state["racket_angular_velocity_w_radps"].tolist()
                ),
                "racket_command_quat_wxyz": list(
                    exact_geometry.racket_command_quat_wxyz
                ),
                "racket_face_center_velocity_w_mps": list(
                    exact_geometry.racket_face_center_velocity_w_mps
                ),
                "racket_site_velocity_w_mps": list(
                    exact_geometry.racket_site_velocity_w_mps
                ),
                "racket_command_angular_velocity_w_radps": list(
                    exact_geometry.racket_command_angular_velocity_w_radps
                ),
                "geometry_source_sha256": geometry_sha,
                "reference_t_hit_s": float(action["reference_t_hit_s"]),
                "reference_t_cycle_s": float(action["reference_t_cycle_s"]),
                "reference_racket_site_speed_mps": float(
                    action["reference_racket_site_speed_mps"]
                ),
                "required_racket_site_speed_mps": float(
                    np.linalg.norm(exact_geometry.racket_site_velocity_w_mps)
                ),
                "reaction_margin_s": float(action["reaction_margin_s"]),
                "teacher_rate_min": float(action["teacher_rate_min"]),
                "teacher_rate_max": float(action["teacher_rate_max"]),
                "teacher_rate": teacher_rate,
                "scaled_t_hit_s": scaled_t_hit,
                "scaled_t_cycle_s": scaled_t_cycle,
                "pre_swing_wait_s": float(source["ttc"]) - scaled_t_hit,
                "solver_residual_m": float(source["residual"]),
                "landing_aim_w_xy_m": proposal["landing_aim_w_xy_m"],
                "solver_profile_sha256": pins["solver_profile_sha256"],
                "physics_profile_sha256": pins["physics_profile_sha256"],
            }
            task_sha = _canonical_sha256(task)
            if role in POSITIVE_CASE_ROLES:
                fault = {"kind": "none"}
                verdict = "PASS"
                reason = None
            elif role == "negative_t_hit_offset":
                fault = {"kind": "teacher_t_hit_offset", "offset_s": 0.05}
                verdict = "FAIL"
                reason = NEGATIVE_REASONS[role]
            elif role == "negative_face_sign":
                fault = {"kind": "selected_face_sign_flip"}
                verdict = "FAIL"
                reason = NEGATIVE_REASONS[role]
            else:
                fault = {
                    "kind": "launch_velocity_delta",
                    "launch_velocity_delta_w_mps": [0.0, 0.3, 0.0],
                }
                verdict = "FAIL"
                reason = NEGATIVE_REASONS[role]
            case_id = "%s:%s" % (action_id, role)
            binding_payload = {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "case_id": case_id,
                "case_role": role,
                "sample_seed": sample_seed,
                "ball_proposal_sha256": proposal_sha,
                "task_payload_sha256": task_sha,
                "solver_execution_identity_sha256": execution_sha,
                "fault_injection": fault,
                "expected_physical_verdict": verdict,
                "expected_failure_reason": reason,
            }
            cases.append(
                {
                    "case_id": case_id,
                    "case_role": role,
                    "sample_seed": sample_seed,
                    "expected_physical_verdict": verdict,
                    "expected_failure_reason": reason,
                    "ball_proposal": proposal,
                    "ball_proposal_sha256": proposal_sha,
                    "task_payload": task,
                    "task_payload_sha256": task_sha,
                    "fault_injection": fault,
                    "case_binding_sha256": _canonical_sha256(binding_payload),
                }
            )

        ball_profile_sha = _canonical_sha256(action["ball_profile"])
        receipt = {
            "schema_version": 1,
            "artifact_type": "frozen_action_ball_solver_execution_receipt_v1",
            "producer": {
                "source_path": MDP_DIR_REL + "/hope_commands.py",
                "source_sha256": pins["source_map"]["hope_commands.py"],
                "runtime_receipt_type": "ActionBallTaskReceipt",
                "exact_solver_replay_required": True,
                "selector_executed": False,
                "hardware_authorized": False,
            },
            "action_identity": {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
            },
            "profile_identity": {
                "ball_profile_sha256": ball_profile_sha,
                "solver_profile_sha256": pins["solver_profile_sha256"],
                "physics_profile_sha256": pins["physics_profile_sha256"],
                "solver_implementation_source_sha256": pins["source_map"],
                "geometry_source_sha256": geometry_sha,
            },
            "solver_execution_identity": execution_identity,
            "cases": cases,
        }
        receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
        receipt_path = receipt_dir / ("%s.json" % action_id)
        _write_exclusive(receipt_path, receipt)
        receipt_sha = _sha256_file(receipt_path)
        binding = {
            "schema_version": 1,
            "authority": "pre_registered_frozen_action_ball_solver_receipt_v1",
            "action_id": action_id,
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
            "ball_profile_sha256": ball_profile_sha,
            "solver_profile_sha256": pins["solver_profile_sha256"],
            "physics_profile_sha256": pins["physics_profile_sha256"],
            "solver_implementation_source_sha256": pins["source_map"],
            "solver_execution_receipt_path": _repo_relative(
                receipt_path,
                repo_root,
                "%s solver receipt" % action_id,
            ),
            "solver_execution_receipt_sha256": receipt_sha,
            "solver_execution_identity": execution_identity,
            "solver_execution_identity_sha256": execution_sha,
            "selector_executed": False,
            "action_identity_frozen": True,
            "cases": cases,
            "cases_sha256": _canonical_sha256(cases),
        }
        overlay_actions.append(
            {
                "action_id": action_id,
                "action_uid": action["action_uid"],
                "motion_sha256": action["motion_sha256"],
                "physical_ball_launch": launches[action_id]["physical_ball_launch"],
                "physical_task_binding": binding,
            }
        )

    bundle = {
        "schema_version": 1,
        "artifact_type": "fresh_n5_physical_task_bundle_v1",
        "base_manifest": {
            "path": _repo_relative(
                base_manifest_path, repo_root, "strict base manifest"
            ),
            "raw_sha256": base_raw_sha,
            "schema_version": 3,
            "strict_training_input": True,
        },
        "batch": {
            "path": _repo_relative(batch_path, repo_root, "fresh batch"),
            "sha256": _sha256_file(batch_path),
        },
        "prototype": {
            "path": _repo_relative(prototype_path, repo_root, "fresh prototype"),
            "sha256": _sha256_file(prototype_path),
            "scope": "upper",
        },
        "profile_pins": {
            "path": _repo_relative(profile_pins_path, repo_root, "profile pins"),
            "sha256": pins["file_sha256"],
            "solver_profile_sha256": pins["solver_profile_sha256"],
            "physics_profile_sha256": pins["physics_profile_sha256"],
            "geometry_source_sha256": geometry_sha,
        },
        "action_order": list(FRESH_N5_ACTION_ORDER),
        "selector_executed": False,
        "action_identity_frozen": True,
        "action_switching_allowed": False,
        "mobility_mode": "no_move",
        "base_task_frame": "relative_about_actual_episode_spawn",
        "gate_materialization_fields": gate_materialization_fields,
        "materialization_contract": {
            "training_consumer": "consume base_manifest only",
            "fitted_gate_consumer": (
                "first cross-check base raw_sha256; then copy "
                "gate_materialization_fields.racket_geometry_contract and "
                "gate_materialization_fields.physical_contact_contract to "
                "the disposable manifest top level, and copy each overlay "
                "action physical_ball_launch and physical_task_binding to "
                "the identity-matched action row"
            ),
            "current_inline_fitted_gate_support": False,
            "downstream_gap": (
                "mujoco_teacher_motion_fitted_ball_gate currently reads "
                "physical fields inline and needs an overlay materializer/input"
            ),
            "required_external_inputs_not_synthesized": [
                "per-action compiler_candidate_pre_admission_v1 evidence",
                "formal source-receipt trust root bound to a clean commit",
                "clean committed runtime/source/data closure",
            ],
        },
        "actions": overlay_actions,
    }
    bundle["content_sha256"] = _canonical_sha256(bundle)
    return bundle


def _prepare_new_directory(path: Path, repo_root: Path) -> None:
    _repo_relative(path, repo_root, "output directory")
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FreshN5BuildError(
            "refusing to overwrite existing output directory %s" % path
        ) from exc


def _cmd_build_batch(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    document = build_batch_document(
        bank_manifest_path=Path(args.bank_manifest).resolve(),
        prototype_path=Path(args.prototype).resolve(),
        profile_pins_path=Path(args.profile_pins).resolve(),
        venue_yaml=Path(args.venue_yaml).resolve(),
        repo_root=repo_root,
        seed=args.seed,
        proposal_count=args.proposals_per_action,
        incoming_dist_path=Path(args.incoming_dist).resolve(),
        incoming_dist_sha256=args.incoming_dist_sha256,
    )
    output = Path(args.out).resolve()
    _repo_relative(output, repo_root, "batch output")
    _write_exclusive(output, document)
    print("WROTE %s" % output)
    return 0


def _cmd_build_overlay(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.out).resolve()
    receipt_dir = output.parent / (output.stem + "_solver_receipts")
    _repo_relative(output, repo_root, "overlay output")
    if os.path.lexists(str(output)) or os.path.lexists(str(receipt_dir)):
        raise FreshN5BuildError(
            "refusing to overwrite existing overlay or receipt directory"
        )
    receipt_dir.mkdir(parents=True, exist_ok=False)
    bundle = _build_solver_bundle(
        base_manifest_path=Path(args.base_manifest).resolve(),
        batch_path=Path(args.batch).resolve(),
        prototype_path=Path(args.prototype).resolve(),
        profile_pins_path=Path(args.profile_pins).resolve(),
        physical_launches_path=Path(args.physical_launches).resolve(),
        repo_root=repo_root,
        receipt_dir=receipt_dir,
    )
    _write_exclusive(output, bundle)
    print("WROTE %s" % output)
    return 0


def _cmd_build_all(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    _prepare_new_directory(output_dir, repo_root)
    batch_path = output_dir / "fresh_n5_batch.json"
    manifest_path = output_dir / "manifest.json"
    overlay_path = output_dir / "fresh_n5_physical_task_bundle.json"
    receipt_dir = output_dir / "solver_receipts"
    try:
        batch = build_batch_document(
            bank_manifest_path=Path(args.bank_manifest).resolve(),
            prototype_path=Path(args.prototype).resolve(),
            profile_pins_path=Path(args.profile_pins).resolve(),
            venue_yaml=Path(args.venue_yaml).resolve(),
            repo_root=repo_root,
            seed=args.seed,
            proposal_count=args.proposals_per_action,
            incoming_dist_path=Path(args.incoming_dist).resolve(),
            incoming_dist_sha256=args.incoming_dist_sha256,
        )
        _write_exclusive(batch_path, batch)
        pins = _validate_profile_pins(Path(args.profile_pins).resolve(), repo_root)
        _build_strict_manifest(
            batch_path=batch_path,
            prototype_path=Path(args.prototype).resolve(),
            profile_pins=pins,
            repo_root=repo_root,
            manifest_id=args.manifest_id,
            out_path=manifest_path,
        )
        if args.physical_launches:
            physical_launches_path = Path(args.physical_launches).resolve()
        else:
            physical_launches_path = _shoot_physical_launch_map(
                base_manifest_path=manifest_path,
                batch_path=batch_path,
                profile_pins_path=Path(args.profile_pins).resolve(),
                incoming_dist_path=Path(args.incoming_dist).resolve(),
                incoming_dist_sha256=args.incoming_dist_sha256,
                repo_root=repo_root,
                output_root=output_dir,
                seed=args.seed,
                proposals_per_case=args.shooting_proposals_per_case,
            )
        receipt_dir.mkdir()
        bundle = _build_solver_bundle(
            base_manifest_path=manifest_path,
            batch_path=batch_path,
            prototype_path=Path(args.prototype).resolve(),
            profile_pins_path=Path(args.profile_pins).resolve(),
            physical_launches_path=physical_launches_path,
            repo_root=repo_root,
            receipt_dir=receipt_dir,
        )
        _write_exclusive(overlay_path, bundle)
    except Exception:
        # Preserve the no-clobber namespace and partial evidence for diagnosis.
        # Never delete or retry into it.
        raise
    print("WROTE %s" % output_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("build-batch")
    batch.add_argument("--bank-manifest", required=True)
    batch.add_argument("--prototype", required=True)
    batch.add_argument("--profile-pins", required=True)
    batch.add_argument("--venue-yaml", required=True)
    batch.add_argument("--incoming-dist", required=True)
    batch.add_argument("--incoming-dist-sha256", required=True)
    batch.add_argument("--seed", type=int, default=20260729)
    batch.add_argument("--proposals-per-action", type=int, default=8000)
    batch.add_argument("--out", required=True)
    batch.set_defaults(func=_cmd_build_batch)

    overlay = subparsers.add_parser("build-overlay")
    overlay.add_argument("--base-manifest", required=True)
    overlay.add_argument("--batch", required=True)
    overlay.add_argument("--prototype", required=True)
    overlay.add_argument("--profile-pins", required=True)
    overlay.add_argument("--physical-launches", required=True)
    overlay.add_argument("--out", required=True)
    overlay.set_defaults(func=_cmd_build_overlay)

    all_parser = subparsers.add_parser("build-all")
    all_parser.add_argument("--bank-manifest", required=True)
    all_parser.add_argument("--prototype", required=True)
    all_parser.add_argument("--profile-pins", required=True)
    all_parser.add_argument("--venue-yaml", required=True)
    all_parser.add_argument("--incoming-dist", required=True)
    all_parser.add_argument("--incoming-dist-sha256", required=True)
    all_parser.add_argument(
        "--physical-launches",
        default=None,
        help=(
            "optional pre-existing formal launch map (for recorded launches); "
            "default builds deterministic one-bounce shooting receipts"
        ),
    )
    all_parser.add_argument(
        "--shooting-proposals-per-case",
        type=int,
        default=16,
    )
    all_parser.add_argument("--manifest-id", required=True)
    all_parser.add_argument("--seed", type=int, default=20260729)
    all_parser.add_argument("--proposals-per-action", type=int, default=8000)
    all_parser.add_argument("--out-dir", required=True)
    all_parser.set_defaults(func=_cmd_build_all)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FreshN5BuildError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
