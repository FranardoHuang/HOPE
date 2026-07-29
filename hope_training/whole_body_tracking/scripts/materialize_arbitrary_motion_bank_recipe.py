#!/usr/bin/env python3
"""Materialize a no-authority arbitrary-N compiler recipe from one capsule.

The source capsule owns ordered action identity and source bytes.  This tool
adds only content-bound compiler inputs, the shared canonical-ready sidecar,
the diagnostic source-hold check, marker/recovery policy, and the explicit
episode-base-local placement contract.  It validates the completed recipe
through :mod:`canonical_motion_arbitrary_bank` before publishing it with
no-clobber semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import canonical_motion_arbitrary_bank as arbitrary  # noqa: E402


class ArbitraryRecipeMaterializationError(RuntimeError):
    """The requested recipe cannot be published without weakening identity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _repo_file(root: Path, value: os.PathLike[str] | str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ArbitraryRecipeMaterializationError(
            f"{label} must be one existing repository file"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ArbitraryRecipeMaterializationError(
            f"{label} must be a regular non-symlink file"
        )
    return resolved


def _binding(root: Path, path: Path) -> Mapping[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = arbitrary._strict_json_bytes(path.read_bytes(), label)
    except (OSError, arbitrary.ArbitraryBankError) as exc:
        raise ArbitraryRecipeMaterializationError(str(exc)) from exc
    return value


def _vector(value: Sequence[float], length: int, label: str) -> list[float]:
    try:
        result = arbitrary._vector(value, length, label)
    except arbitrary.ArbitraryBankError as exc:
        raise ArbitraryRecipeMaterializationError(str(exc)) from exc
    return [float(item) for item in result]


def materialize_recipe(
    *,
    repo_root: os.PathLike[str] | str,
    bank_id: str,
    source_capsule_path: os.PathLike[str] | str,
    expected_source_capsule_sha256: str,
    compiler_template_path: os.PathLike[str] | str,
    expected_compiler_template_sha256: str,
    source_hold_motion_path: os.PathLike[str] | str,
    expected_source_hold_motion_sha256: str,
    source_hold_frame: int,
    hold_tolerances: Mapping[str, float],
    acceleration_receipt_path: os.PathLike[str] | str,
    expected_acceleration_receipt_sha256: str,
    marker_half_width_frames: int,
    minimum_source_preparation_frames: int,
    minimum_source_recovery_frames: int,
    minimum_compiled_recovery_s: float,
    full_root_position_lower: Sequence[float],
    full_root_position_upper: Sequence[float],
    full_root_velocity: Sequence[float],
    full_root_acceleration: Sequence[float],
    samples_per_scaled_unit: float,
    min_connector_intervals: int,
    min_core_intervals: int,
    grid_subdivisions: int,
    search_workers: int,
    search_parallel_backend: str,
    output_path: os.PathLike[str] | str,
) -> tuple[Path, Mapping[str, Any], str]:
    """Validate and exclusively publish one strict arbitrary-N recipe."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ArbitraryRecipeMaterializationError(
            "repo_root must be one real directory"
        )
    capsule_path = _repo_file(
        root, source_capsule_path, "source capsule"
    )
    template_path = _repo_file(
        root, compiler_template_path, "compiler template"
    )
    hold_path = _repo_file(
        root, source_hold_motion_path, "source hold motion"
    )
    acceleration_path = _repo_file(
        root, acceleration_receipt_path, "acceleration receipt"
    )
    expected_pairs = (
        (
            capsule_path,
            expected_source_capsule_sha256,
            "source capsule",
        ),
        (
            template_path,
            expected_compiler_template_sha256,
            "compiler template",
        ),
        (
            hold_path,
            expected_source_hold_motion_sha256,
            "source hold motion",
        ),
        (
            acceleration_path,
            expected_acceleration_receipt_sha256,
            "acceleration receipt",
        ),
    )
    for path, expected, label in expected_pairs:
        try:
            digest = arbitrary._digest(expected, f"expected {label} SHA-256")
        except arbitrary.ArbitraryBankError as exc:
            raise ArbitraryRecipeMaterializationError(str(exc)) from exc
        actual = _sha256(path)
        if actual != digest:
            raise ArbitraryRecipeMaterializationError(
                f"{label} SHA-256 mismatch: expected {digest}, got {actual}"
            )
    capsule = _json(capsule_path, "source capsule")
    if (
        capsule.get("schema_version") != 1
        or capsule.get("consumer_interface")
        != arbitrary.SOURCE_CAPSULE_INTERFACE
        or capsule.get("verdict") != "PASS_SOURCE_INVENTORY_ONLY"
    ):
        raise ArbitraryRecipeMaterializationError(
            "source capsule does not expose the arbitrary-N consumer interface"
        )
    raw_actions = capsule.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ArbitraryRecipeMaterializationError(
            "source capsule actions must be non-empty"
        )
    motion_ids: list[str] = []
    for index, row in enumerate(raw_actions):
        if not isinstance(row, Mapping):
            raise ArbitraryRecipeMaterializationError(
                f"source capsule actions[{index}] must be an object"
            )
        try:
            motion_id = arbitrary._slug(
                row.get("action_id"),
                f"source capsule actions[{index}].action_id",
            )
        except arbitrary.ArbitraryBankError as exc:
            raise ArbitraryRecipeMaterializationError(str(exc)) from exc
        motion_ids.append(motion_id)
    if len(set(motion_ids)) != len(motion_ids):
        raise ArbitraryRecipeMaterializationError(
            "source capsule ordered action IDs contain duplicates"
        )
    try:
        normalized_bank_id = arbitrary._slug(bank_id, "bank_id")
        template = arbitrary.load_canonical_motion_recipe(
            template_path, repo_root=root
        )
    except Exception as exc:
        raise ArbitraryRecipeMaterializationError(
            f"compiler template load failed: {type(exc).__name__}: {exc}"
        ) from exc
    tolerance_keys = arbitrary._READY_TOLERANCE_KEYS
    if frozenset(hold_tolerances) != tolerance_keys:
        raise ArbitraryRecipeMaterializationError(
            "hold_tolerances must contain the exact six ready-hold metrics"
        )
    document = {
        "schema_version": 1,
        "recipe_type": arbitrary.RECIPE_TYPE,
        "bank_id": normalized_bank_id,
        "publication_class": arbitrary.PUBLICATION_CLASS,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "producer": _binding(root, Path(arbitrary.__file__).resolve()),
        "source_capsule": _binding(root, capsule_path),
        "ordered_motion_ids": motion_ids,
        "shared_ready": {
            "canonical_ready": _binding(
                root, Path(template.ready.path).resolve()
            ),
            "source_motion_path": hold_path.relative_to(root).as_posix(),
            "source_motion_sha256": _sha256(hold_path),
            "source_frame": int(source_hold_frame),
            "hold_tolerances": {
                key: float(hold_tolerances[key])
                for key in sorted(hold_tolerances)
            },
            "evidence_status": (
                "SOURCE_HOLD_ONLY_NOT_GROUNDED_CERTIFICATE"
            ),
        },
        "marker_policy": {
            "mode": "source_hit_centered_marker_only_v1",
            "half_width_frames": int(marker_half_width_frames),
            "minimum_source_preparation_frames": int(
                minimum_source_preparation_frames
            ),
            "minimum_source_recovery_frames": int(
                minimum_source_recovery_frames
            ),
            "minimum_compiled_recovery_s": float(
                minimum_compiled_recovery_s
            ),
        },
        "compiler_template": _binding(root, template_path),
        "compiler_options": {
            "joint_acceleration_receipt": _binding(
                root, acceleration_path
            ),
            "full_root_position_lower": _vector(
                full_root_position_lower, 6, "full_root_position_lower"
            ),
            "full_root_position_upper": _vector(
                full_root_position_upper, 6, "full_root_position_upper"
            ),
            "full_root_velocity": _vector(
                full_root_velocity, 6, "full_root_velocity"
            ),
            "full_root_acceleration": _vector(
                full_root_acceleration, 6, "full_root_acceleration"
            ),
            "samples_per_scaled_unit": float(samples_per_scaled_unit),
            "min_connector_intervals": int(min_connector_intervals),
            "min_core_intervals": int(min_core_intervals),
            "grid_subdivisions": int(grid_subdivisions),
            "search_workers": int(search_workers),
            "search_parallel_backend": search_parallel_backend,
        },
        "required_output_matrix": {
            "motion_ids": motion_ids,
            "scopes": list(arbitrary.SCOPES),
            "candidate_count": 2 * len(motion_ids),
        },
        "placement_contract": dict(arbitrary._PLACEMENT_CONTRACT),
        "non_claims": [
            "grounded_ready_certificate",
            "dynamics_or_balance",
            "table_or_collision_safety",
            "physical_ball_return",
            "training_authorization",
            "hardware_authorization",
        ],
    }
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = Path(os.path.abspath(os.fspath(destination)))
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ArbitraryRecipeMaterializationError(
            "output_path must stay inside repo_root"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite existing recipe {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        arbitrary.load_arbitrary_bank_recipe(
            temporary,
            repo_root=root,
        )
        # Hard-link publication is the atomic no-clobber operation.  Do not
        # unlink ``destination`` on failure: a concurrent publisher may have
        # won the race after the lexists precheck, and those bytes are not ours
        # to remove.
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination, document, hashlib.sha256(payload).hexdigest()


def _six(value: str) -> list[float]:
    try:
        result = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected six comma-separated numbers"
        ) from exc
    if len(result) != 6:
        raise argparse.ArgumentTypeError(
            "expected six comma-separated numbers"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--source-capsule", required=True)
    parser.add_argument("--expected-source-capsule-sha256", required=True)
    parser.add_argument("--compiler-template", required=True)
    parser.add_argument("--expected-compiler-template-sha256", required=True)
    parser.add_argument("--source-hold-motion", required=True)
    parser.add_argument("--expected-source-hold-motion-sha256", required=True)
    parser.add_argument("--source-hold-frame", type=int, default=0)
    parser.add_argument("--acceleration-receipt", required=True)
    parser.add_argument("--expected-acceleration-receipt-sha256", required=True)
    parser.add_argument("--marker-half-width-frames", type=int, default=2)
    parser.add_argument(
        "--minimum-source-preparation-frames", type=int, default=10
    )
    parser.add_argument(
        "--minimum-source-recovery-frames", type=int, default=8
    )
    parser.add_argument(
        "--minimum-compiled-recovery-s", type=float, default=0.2
    )
    parser.add_argument(
        "--full-root-position-lower",
        type=_six,
        default=_six("0,-0.4,0.85,-1,-1,-1"),
    )
    parser.add_argument(
        "--full-root-position-upper",
        type=_six,
        default=_six("0.4,0.1,1.05,1,1,1"),
    )
    parser.add_argument(
        "--full-root-velocity",
        type=_six,
        default=_six("1,1,0.5,2,2,2"),
    )
    parser.add_argument(
        "--full-root-acceleration",
        type=_six,
        default=_six("10,10,5,20,20,20"),
    )
    parser.add_argument("--samples-per-scaled-unit", type=float, default=6.0)
    parser.add_argument("--min-connector-intervals", type=int, default=5)
    parser.add_argument("--min-core-intervals", type=int, default=5)
    parser.add_argument("--grid-subdivisions", type=int, default=4)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument(
        "--search-parallel-backend",
        choices=("thread", "process"),
        default="thread",
    )
    for name, default in (
        ("joint-position-rad", 1.0e-4),
        ("root-position-m", 1.0e-6),
        ("root-orientation-rad", 1.0e-4),
        ("joint-velocity-rad-s", 2.0e-3),
        ("body-linear-velocity-m-s", 5.0e-4),
        ("body-angular-velocity-rad-s", 2.0e-3),
    ):
        parser.add_argument(
            f"--hold-tolerance-{name}",
            type=float,
            default=default,
        )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tolerances = {
        "joint_position_rad": args.hold_tolerance_joint_position_rad,
        "root_position_m": args.hold_tolerance_root_position_m,
        "root_orientation_rad": args.hold_tolerance_root_orientation_rad,
        "joint_velocity_rad_s": args.hold_tolerance_joint_velocity_rad_s,
        "body_linear_velocity_m_s": (
            args.hold_tolerance_body_linear_velocity_m_s
        ),
        "body_angular_velocity_rad_s": (
            args.hold_tolerance_body_angular_velocity_rad_s
        ),
    }
    try:
        path, document, digest = materialize_recipe(
            repo_root=args.repo_root,
            bank_id=args.bank_id,
            source_capsule_path=args.source_capsule,
            expected_source_capsule_sha256=(
                args.expected_source_capsule_sha256
            ),
            compiler_template_path=args.compiler_template,
            expected_compiler_template_sha256=(
                args.expected_compiler_template_sha256
            ),
            source_hold_motion_path=args.source_hold_motion,
            expected_source_hold_motion_sha256=(
                args.expected_source_hold_motion_sha256
            ),
            source_hold_frame=args.source_hold_frame,
            hold_tolerances=tolerances,
            acceleration_receipt_path=args.acceleration_receipt,
            expected_acceleration_receipt_sha256=(
                args.expected_acceleration_receipt_sha256
            ),
            marker_half_width_frames=args.marker_half_width_frames,
            minimum_source_preparation_frames=(
                args.minimum_source_preparation_frames
            ),
            minimum_source_recovery_frames=(
                args.minimum_source_recovery_frames
            ),
            minimum_compiled_recovery_s=(
                args.minimum_compiled_recovery_s
            ),
            full_root_position_lower=args.full_root_position_lower,
            full_root_position_upper=args.full_root_position_upper,
            full_root_velocity=args.full_root_velocity,
            full_root_acceleration=args.full_root_acceleration,
            samples_per_scaled_unit=args.samples_per_scaled_unit,
            min_connector_intervals=args.min_connector_intervals,
            min_core_intervals=args.min_core_intervals,
            grid_subdivisions=args.grid_subdivisions,
            search_workers=args.search_workers,
            search_parallel_backend=args.search_parallel_backend,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": "PASS_RECIPE_INPUTS_ONLY",
                "path": str(path),
                "sha256": digest,
                "motion_count": len(document["ordered_motion_ids"]),
                "candidate_count": document["required_output_matrix"][
                    "candidate_count"
                ],
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
