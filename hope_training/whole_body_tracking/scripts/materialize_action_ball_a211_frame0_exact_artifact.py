#!/usr/bin/env python3
"""Materialize one canonical A211 frame-0 exact-state candidate artifact.

This host-only producer copies the measured motion's exact frame-0 root pose
and 31 joint positions, while deliberately replacing root/joint velocities
with zeros for the nominal reset candidate.  It does not run the Pod nominal
hold and cannot produce or impersonate the separate PASS receipt required by
the A211 lineage/launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a211_four_arm_diagnostic.py"
ROOT_BODY = "pelvis_link"
ACTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MaterializationError(RuntimeError):
    """The measured frame-0 input or candidate publication was invalid."""


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "_a211_frame0_artifact_launcher", LAUNCHER_FILE
    )
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
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
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


def require_content_sha(document: Mapping[str, Any]) -> str:
    """Verify and return a candidate artifact's reproducible semantic seal."""

    seal = document.get("content_sha256")
    if type(seal) is not str or SHA256_RE.fullmatch(seal) is None:
        raise MaterializationError("content_sha256 must be one lowercase SHA-256")
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    if canonical_sha256(unsigned) != seal:
        raise MaterializationError("content_sha256 is not reproducible")
    return seal


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


def _input_path(root: Path, relative: str) -> Path:
    relative = _relative(relative, name="motion_path")
    path = root / relative
    _regular(path, name="motion")
    if path.resolve(strict=True) != path:
        raise MaterializationError("motion path must not traverse a symlink")
    return path


def _timing_receipt(
    root: Path,
    relative: str,
    expected_file_sha256: str,
    *,
    motion_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the sealed task receipt that owns the birth timing."""

    relative = _relative(relative, name="timing_receipt_path")
    expected_file_sha256 = str(expected_file_sha256)
    if SHA256_RE.fullmatch(expected_file_sha256) is None:
        raise MaterializationError(
            "expected_timing_receipt_sha256 must be one lowercase SHA-256"
        )
    path = root / relative
    _regular(path, name="timing receipt")
    if path.resolve(strict=True) != path:
        raise MaterializationError("timing receipt path must not traverse a symlink")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise MaterializationError("timing receipt file SHA differs")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("timing receipt is not strict JSON") from exc
    if type(receipt) is not dict or raw != canonical_bytes(receipt) + b"\n":
        raise MaterializationError("timing receipt must be canonical JSON plus newline")
    seal = receipt.get("canonical_sha256")
    unsigned = dict(receipt)
    unsigned.pop("canonical_sha256", None)
    if (
        type(seal) is not str
        or SHA256_RE.fullmatch(seal) is None
        or canonical_sha256(unsigned) != seal
        or receipt.get("schema_version") != 5
        or receipt.get("motion_sha256") != motion_sha256
    ):
        raise MaterializationError("timing receipt seal/action binding differs")
    return {"path": relative, "sha256": expected_file_sha256}, receipt


def _derive_birth_horizon(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Derive reset-to-teacher-start coverage; never reuse the 200-tick soak."""

    dt = receipt.get("contact_time_step_s")
    pre_wait = receipt.get("pre_swing_wait_s")
    if (
        type(dt) not in (int, float)
        or type(dt) is bool
        or type(pre_wait) not in (int, float)
        or type(pre_wait) is bool
        or not math.isfinite(float(dt))
        or not math.isfinite(float(pre_wait))
        or float(dt) <= 0.0
        or float(pre_wait) < 0.0
        or not math.isclose(float(dt), _L.POLICY_DT_S, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise MaterializationError("timing receipt policy/pre-swing timing differs")
    pre_wait_ticks = int(math.ceil(float(pre_wait) / float(dt)))
    post_reset_coverage_ticks = 1
    policy_ticks = (
        post_reset_coverage_ticks
        + int(_L.WAIT_SCHEDULE["max_wait_ticks"])
        + pre_wait_ticks
    )
    if policy_ticks < 1:
        raise MaterializationError("birth horizon derivation differs")
    return {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_horizon_v1",
        "derivation": (
            "post_reset_coverage_plus_max_reset_wait_plus_ceil_pre_swing_wait"
        ),
        "timing_receipt_canonical_sha256": receipt["canonical_sha256"],
        "policy_dt_s": float(dt),
        "post_reset_coverage_policy_ticks": post_reset_coverage_ticks,
        "max_reset_wait_policy_ticks": int(_L.WAIT_SCHEDULE["max_wait_ticks"]),
        "pre_swing_wait_s": float(pre_wait),
        "pre_swing_wait_policy_ticks_ceil": pre_wait_ticks,
        "required_policy_ticks": policy_ticks,
    }


def _scalar(array: np.ndarray, *, name: str) -> object:
    values = np.asarray(array).reshape(-1)
    if values.size != 1:
        raise MaterializationError("motion %s must be scalar" % name)
    return values[0]


def _finite_array(value: np.ndarray, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fc" or not np.all(np.isfinite(array)):
        raise MaterializationError(
            "motion %s must be a finite numeric array shaped %s" % (name, shape)
        )
    return array


def _load_frame0(motion_path: Path) -> dict[str, list[float]]:
    required = {
        "joint_pos",
        "joint_vel",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "kinematics_schema_version",
        "body_pos_point",
        "body_lin_vel_point",
        "measured_racket_schema_version",
    }
    try:
        with np.load(motion_path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise MaterializationError(
                    "measured motion lacks fields %s" % sorted(missing)
                )
            schema = _scalar(
                archive["kinematics_schema_version"],
                name="kinematics_schema_version",
            )
            measured_schema = _scalar(
                archive["measured_racket_schema_version"],
                name="measured_racket_schema_version",
            )
            pos_point = str(_scalar(archive["body_pos_point"], name="body_pos_point"))
            vel_point = str(
                _scalar(archive["body_lin_vel_point"], name="body_lin_vel_point")
            )
            names_raw = np.asarray(archive["body_names"])
            if names_raw.ndim != 1:
                raise MaterializationError("motion body_names must be one-dimensional")
            body_names = tuple(str(value) for value in names_raw.tolist())
            joint_pos = np.asarray(archive["joint_pos"])
            joint_vel = np.asarray(archive["joint_vel"])
            body_pos = np.asarray(archive["body_pos_w"])
            body_quat = np.asarray(archive["body_quat_w"])
            body_lin_vel = np.asarray(archive["body_lin_vel_w"])
            body_ang_vel = np.asarray(archive["body_ang_vel_w"])
    except MaterializationError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise MaterializationError("cannot load measured motion NPZ") from exc

    if (
        type(schema.item() if isinstance(schema, np.generic) else schema) not in (int,)
        or int(schema) != 2
        or type(measured_schema.item() if isinstance(measured_schema, np.generic) else measured_schema)
        not in (int,)
        or int(measured_schema) != 4
        or pos_point != "link_origin"
        or vel_point != "center_of_mass"
    ):
        raise MaterializationError("measured motion kinematics/schema point semantics differ")
    if (
        not body_names
        or len(set(body_names)) != len(body_names)
        or body_names.count(ROOT_BODY) != 1
    ):
        raise MaterializationError("motion body-name identity is invalid")
    if joint_pos.ndim != 2 or joint_pos.shape[0] < 1 or joint_pos.shape[1] != 31:
        raise MaterializationError("motion joint_pos must have shape (T,31), T>=1")
    frame_count = int(joint_pos.shape[0])
    body_count = len(body_names)
    joint_pos = _finite_array(
        joint_pos, name="joint_pos", shape=(frame_count, 31)
    )
    _finite_array(joint_vel, name="joint_vel", shape=(frame_count, 31))
    body_pos = _finite_array(
        body_pos, name="body_pos_w", shape=(frame_count, body_count, 3)
    )
    body_quat = _finite_array(
        body_quat, name="body_quat_w", shape=(frame_count, body_count, 4)
    )
    _finite_array(
        body_lin_vel,
        name="body_lin_vel_w",
        shape=(frame_count, body_count, 3),
    )
    _finite_array(
        body_ang_vel,
        name="body_ang_vel_w",
        shape=(frame_count, body_count, 3),
    )
    root_index = body_names.index(ROOT_BODY)
    root_quat = body_quat[0, root_index]
    norm = float(np.linalg.norm(root_quat.astype(np.float64, copy=False)))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise MaterializationError("motion frame-0 root quaternion is not unit length")

    # tolist() performs no normalization or coordinate conversion.  In
    # particular, float32 source values become exactly representable Python
    # floats and survive canonical JSON round-trip without numeric drift.
    return {
        "root_pos_w_m": body_pos[0, root_index].tolist(),
        "root_quat_wxyz": root_quat.tolist(),
        "root_lin_vel_w_mps": [0.0, 0.0, 0.0],
        "root_ang_vel_w_radps": [0.0, 0.0, 0.0],
        "joint_pos_rad": joint_pos[0].tolist(),
        "joint_vel_radps": [0.0] * 31,
    }


def _write_new(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    relative = _relative(relative, name="output")
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
        descriptor, temporary = tempfile.mkstemp(
            prefix=".a211-frame0-artifact-", dir=str(parent)
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:  # pragma: no cover - race protection
            raise MaterializationError(
                "no-clobber output already exists: %s" % output
            ) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    _regular(output, name="output")
    return {"path": relative, "sha256": sha256_file(output)}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve(strict=True)
    if not root.is_dir():
        raise MaterializationError("repo_root must be a directory")
    action_id = args.action_id
    if type(action_id) is not str or ACTION_ID_RE.fullmatch(action_id) is None:
        raise MaterializationError("action_id is malformed")
    expected_motion_sha = args.expected_motion_sha256
    if type(expected_motion_sha) is not str or SHA256_RE.fullmatch(expected_motion_sha) is None:
        raise MaterializationError("expected_motion_sha256 must be one lowercase SHA-256")
    motion_path = _input_path(root, args.motion_path)
    if sha256_file(motion_path) != expected_motion_sha:
        raise MaterializationError("motion file SHA differs")
    timing_pin, timing_receipt = _timing_receipt(
        root,
        args.timing_receipt_path,
        args.expected_timing_receipt_sha256,
        motion_sha256=expected_motion_sha,
    )
    birth_horizon = _derive_birth_horizon(timing_receipt)

    unsigned = {
        "schema_version": 2,
        "kind": _L.FRAME0_EXACT_ARTIFACT_KIND,
        "diagnostic_unauthorized": True,
        "source_kind": _L.FRAME0_EXACT_SOURCE_KIND,
        "action_id": action_id,
        "motion_sha256": expected_motion_sha,
        "policy_dt_s": _L.POLICY_DT_S,
        "wait_schedule_canonical_sha256": _L.WAIT_SCHEDULE["canonical_sha256"],
        "timing_receipt": timing_pin,
        "birth_horizon": birth_horizon,
        "frame0": _load_frame0(motion_path),
    }
    artifact = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
    require_content_sha(artifact)
    pin = _write_new(root, args.output, canonical_bytes(artifact) + b"\n")
    return {
        "status": "MATERIALIZED_POD_NOMINAL_HOLD_REQUIRED",
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "nominal_hold_receipt_created": False,
        "artifact": pin,
        "artifact_content_sha256": artifact["content_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--timing-receipt-path", required=True)
    parser.add_argument("--expected-timing-receipt-sha256", required=True)
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
