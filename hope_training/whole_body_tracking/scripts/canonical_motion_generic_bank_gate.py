#!/usr/bin/env python3
"""Verify one standalone arbitrary-N canonical bank without a five-ID prefix.

This is a discriminated adapter over the independent canonical bank verifier.
It exists because the historical verifier's default profile intentionally
means the canonical five, while a ChingMu-style bank is an independent ordered
set rather than an append to that five.

The adapter first reopens the arbitrary-N recipe, source capsule, exact N x 2
compiler manifest, outputs, sidecars, and a schema-v2 generic registry.  It
then runs the existing independent FK, plant-dynamics, persisted-time-law,
grounded left/midpoint/right, and continuous swept-clearance verifier against
that exact matrix.  The historical verifier is executed with a process-local
matrix view; its canonical-five public defaults and files are not modified.

The emitted report uses the schema-v2 ``generic_v2`` report contract, binds
the selected registry projection, and names this adapter as the bank-gate
tool.  Admission must reopen these exact producer bytes as well as every
underlying safety and dynamics receipt represented by the report.  This tool
never mints a training, deployment, or hardware capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import canonical_motion_arbitrary_bank as arbitrary  # noqa: E402
import canonical_motion_bank_gate as bank_gate  # noqa: E402
import canonical_motion_registry as registry_module  # noqa: E402
from canonical_motion_recipe import CanonicalMotionRecipe  # noqa: E402


REPORT_PROFILE = "generic_v2"
REPORT_SCHEMA_VERSION = 2
SCOPES = ("upper", "full")
_MATRIX_LOCK = threading.RLock()
_SHA256 = frozenset("0123456789abcdef")
_SELECTED_KEYS = frozenset(
    {
        "scope",
        "registry_sha256",
        "alignment_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "motion_ids",
        "npz_sha256",
        "build_manifest_sha256",
    }
)
_GENERIC_AGGREGATE_DROP = frozenset(
    {
        "swept_clearance_pass_count",
        "swept_clearance_minimum_certified_lower_bound_m",
        "time_law_artifact_count",
        "grounded_lmr_pass_count",
        "grounded_lmr_incomplete_count",
    }
)


class GenericBankGateError(RuntimeError):
    """The arbitrary-N bank is incomplete or its identity has drifted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise GenericBankGateError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GenericBankGateError(f"cannot read {label}: {exc}") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GenericBankGateError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GenericBankGateError(
            f"{label} contains non-finite constant {value}"
        )

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except GenericBankGateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GenericBankGateError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(result, Mapping):
        raise GenericBankGateError(f"{label} must contain one JSON object")
    return result


def _absolute(value: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _manifest_and_matrix(
    loaded: arbitrary.LoadedArbitraryRecipe,
    manifest_path: Path,
    bank_directory: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[tuple[str, str], tuple[Mapping[str, Any], Path]],
]:
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not bank_directory.is_dir()
        or bank_directory.is_symlink()
    ):
        raise GenericBankGateError(
            "manifest must be a regular file and bank must be a real directory"
        )
    if manifest_path != bank_directory / bank_gate.MANIFEST_NAME:
        raise GenericBankGateError(
            "generic bank manifest must be the BUILD_MANIFEST.json inside "
            "the exact bank directory"
        )
    manifest = _strict_json(manifest_path, "arbitrary-N build manifest")
    try:
        arbitrary.validate_arbitrary_build_manifest(manifest, loaded)
    except arbitrary.ArbitraryBankError as exc:
        raise GenericBankGateError(str(exc)) from exc
    recipe_binding = manifest.get("recipe")
    if (
        not isinstance(recipe_binding, Mapping)
        or recipe_binding.get("sha256") != loaded.sha256
        or Path(str(recipe_binding.get("path", ""))).resolve()
        != loaded.path.resolve()
    ):
        raise GenericBankGateError(
            "build manifest recipe binding differs from the loaded "
            "arbitrary-N recipe"
        )
    expected_matrix = tuple(
        (motion_id, scope)
        for motion_id in loaded.motion_ids
        for scope in SCOPES
    )
    expected_filenames = tuple(
        f"{motion_id}_{scope}_canonical_v2.npz"
        for motion_id, scope in expected_matrix
    )
    try:
        matrix = bank_gate._validate_bank_file_set(
            bank_directory,
            manifest["outputs"],
            expected_matrix=expected_matrix,
            expected_filenames=expected_filenames,
            label="standalone arbitrary-N bank",
        )
    except bank_gate.CanonicalMotionBankGateError as exc:
        raise GenericBankGateError(str(exc)) from exc
    minimum_recovery = float(
        loaded.raw["marker_policy"]["minimum_compiled_recovery_s"]
    )
    for motion_id, scope in expected_matrix:
        row, output_path = matrix[(motion_id, scope)]
        expected_sha = _digest(
            row["output_npz_sha256"],
            f"{motion_id}/{scope} output sha256",
        )
        actual_sha = _sha256_file(output_path)
        if actual_sha != expected_sha:
            raise GenericBankGateError(
                f"{motion_id}/{scope} output SHA-256 mismatch"
            )
        t_hit = row.get("source_anchor_time_s")
        t_cycle = row.get("duration_s")
        if (
            isinstance(t_hit, bool)
            or isinstance(t_cycle, bool)
            or not isinstance(t_hit, (int, float))
            or not isinstance(t_cycle, (int, float))
            or not math.isfinite(float(t_hit))
            or not math.isfinite(float(t_cycle))
            or not 0.0 < float(t_hit) < float(t_cycle)
            or float(t_cycle) - float(t_hit)
            < minimum_recovery - 1.0e-12
        ):
            raise GenericBankGateError(
                f"{motion_id}/{scope} compiled t_hit/t_cycle/recovery is invalid"
            )
    return manifest, matrix


def _registry_binding(
    *,
    registry_path: Path,
    expected_registry_sha256: str,
    repo_root: Path,
    loaded: arbitrary.LoadedArbitraryRecipe,
    manifest: Mapping[str, Any],
    matrix: Mapping[
        tuple[str, str], tuple[Mapping[str, Any], Path]
    ],
) -> tuple[Mapping[str, Any], Any]:
    try:
        registry = registry_module.load_canonical_motion_bank_registry(
            registry_path,
            repo_root=repo_root,
            expected_registry_sha256=expected_registry_sha256,
        )
    except Exception as exc:
        raise GenericBankGateError(
            f"strict generic registry load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if registry.schema_version != registry_module.GENERIC_REGISTRY_SCHEMA_VERSION:
        raise GenericBankGateError(
            "standalone arbitrary-N gate requires registry schema_version=2"
        )
    if (
        registry.bank_id != loaded.raw["bank_id"]
        or registry.motion_ids != loaded.motion_ids
        or registry.canonical_ready_sha256
        != loaded.canonical_recipe.ready.sha256
    ):
        raise GenericBankGateError(
            "generic registry bank/order/ready differs from compiler identity"
        )
    manifest_sha = _sha256_file(Path(manifest["recipe"]["path"]).resolve())
    if manifest_sha != loaded.sha256:
        raise GenericBankGateError(
            "recipe bytes changed after build-manifest validation"
        )
    build_manifest_sha = _sha256_file(
        Path(matrix[(loaded.motion_ids[0], "upper")][1]).parent
        / bank_gate.MANIFEST_NAME
    )
    for entry in registry.entries:
        row, output_path = matrix[(entry.motion_id, registry.scope)]
        if (
            Path(entry.npz_path).resolve() != output_path.resolve()
            or entry.npz_sha256 != row["output_npz_sha256"]
            or entry.build_manifest_sha256 != build_manifest_sha
        ):
            raise GenericBankGateError(
                f"generic registry row {entry.motion_id!r} does not bind "
                "the selected compiler output and build manifest"
            )
    try:
        binding = registry_module.bank_promotion_binding(
            registry,
            authorization_purpose="training",
        )
    except Exception as exc:
        raise GenericBankGateError(
            f"cannot derive generic registry binding: {type(exc).__name__}: {exc}"
        ) from exc
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
    if frozenset(selected) != _SELECTED_KEYS:
        raise GenericBankGateError(
            "derived selected registry binding schema changed"
        )
    return MappingProxyType(selected), binding


def _engine_recipe(
    loaded: arbitrary.LoadedArbitraryRecipe,
) -> CanonicalMotionRecipe:
    """Supply the legacy engine's irrelevant numeric s0 option sentinel.

    The standalone producer forbids the ``s0_highpress`` identity and compiles
    with an exact zero grounding transform.  The reused option validator still
    asks for the historical s0 ceiling, so this verifier-only raw view adds a
    zero-ceiling sentinel.  It does not add a source, output, or motion ID and
    cannot trigger the compiler's s0 transform.
    """

    raw = dict(loaded.canonical_recipe.raw)
    specs = list(raw.get("motion_specs", ()))
    if any(
        isinstance(row, Mapping) and row.get("motion_id") == "s0_highpress"
        for row in specs
    ):
        raise GenericBankGateError(
            "standalone arbitrary-N recipe unexpectedly contains s0_highpress"
        )
    specs.append(
        {
            "motion_id": "s0_highpress",
            "scope_overrides": {
                "full": {"maximum_grounding_offset_m": 0.0}
            },
            "verifier_only_option_sentinel": True,
        }
    )
    raw["motion_specs"] = specs
    return replace(
        loaded.canonical_recipe,
        raw=MappingProxyType(raw),
    )


def _run_independent_engine(
    *,
    loaded: arbitrary.LoadedArbitraryRecipe,
    manifest_path: Path,
    bank_directory: Path,
    mjcf_path: Path,
    urdf_path: Path,
    body_order_path: Path,
    expected_compiled_signature: str,
    swept_clearance_receipt_path: Path,
    expected_swept_clearance_receipt_sha256: str,
    engine_recipe: CanonicalMotionRecipe,
    plant_loader: Any = None,
    player_runner: Any = None,
    dynamics_runner: Any = None,
    grounded_lmr_runner: Any = None,
) -> Mapping[str, Any]:
    """Run the historical verifier under an isolated arbitrary matrix view."""

    motion_ids = loaded.motion_ids
    expected_matrix = tuple(
        (motion_id, scope)
        for motion_id in motion_ids
        for scope in SCOPES
    )
    expected_filenames = tuple(
        f"{motion_id}_{scope}_canonical_v2.npz"
        for motion_id, scope in expected_matrix
    )

    def recipe_loader(path: Path) -> CanonicalMotionRecipe:
        if Path(path).resolve() != loaded.path.resolve():
            raise GenericBankGateError(
                "independent engine requested an unexpected recipe"
            )
        return engine_recipe

    kwargs: dict[str, Any] = {
        "mjcf_path": mjcf_path,
        "urdf_path": urdf_path,
        "body_order_path": body_order_path,
        "expected_compiled_signature": expected_compiled_signature,
        "swept_clearance_receipt_path": swept_clearance_receipt_path,
        "expected_swept_clearance_receipt_sha256": (
            expected_swept_clearance_receipt_sha256
        ),
        "recipe_loader": recipe_loader,
    }
    for key, value in (
        ("plant_loader", plant_loader),
        ("player_runner", player_runner),
        ("dynamics_runner", dynamics_runner),
        ("grounded_lmr_runner", grounded_lmr_runner),
    ):
        if value is not None:
            kwargs[key] = value

    # The underlying verifier has immutable historical defaults represented as
    # module constants.  Limit the alternate view to this locked call and
    # restore it under BaseException so the canonical-five API is unchanged.
    with _MATRIX_LOCK:
        old_ids = bank_gate.MOTION_IDS
        old_matrix = bank_gate.EXPECTED_MATRIX
        old_filenames = bank_gate.EXPECTED_FILENAMES
        bank_gate.MOTION_IDS = motion_ids
        bank_gate.EXPECTED_MATRIX = expected_matrix
        bank_gate.EXPECTED_FILENAMES = expected_filenames
        try:
            return bank_gate.verify_canonical_motion_bank(
                manifest_path,
                bank_directory,
                **kwargs,
            )
        finally:
            bank_gate.MOTION_IDS = old_ids
            bank_gate.EXPECTED_MATRIX = old_matrix
            bank_gate.EXPECTED_FILENAMES = old_filenames


def _generic_v2_report(
    raw_report: Mapping[str, Any],
    *,
    selected_registry_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        type(raw_report.get("schema_version")) is not int
        or raw_report.get("schema_version") != 1
        or raw_report.get("verdict") != "PASS"
        or raw_report.get("bank_gate_pass") is not True
        or raw_report.get("candidate_integrity_pass") is not True
        or raw_report.get("grounded_trace_status")
        != "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
    ):
        raise GenericBankGateError(
            "independent engine did not return a complete grounded PASS"
        )
    if frozenset(selected_registry_binding) != _SELECTED_KEYS:
        raise GenericBankGateError(
            "selected generic registry binding schema changed"
        )
    report = dict(raw_report)
    # This adapter is the schema boundary: the reused independent engine emits
    # the historical schema-v1 report, while the selected arbitrary-N registry
    # binding below is the discriminating v2 field.  Never inherit the legacy
    # version number into a generic report.
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["grounded_trace_status"] = "COMPLETE_PASS"
    bound = dict(report["bound_inputs"])
    bound.pop("swept_clearance_receipt", None)
    tools = dict(bound["verifier_tools"])
    tools["bank_gate"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }
    bound["verifier_tools"] = tools
    report["bound_inputs"] = bound
    contracts = dict(report["contracts"])
    contracts.pop("swept_clearance", None)
    contracts["grounded_trace_status"] = "COMPLETE_PASS"
    report["contracts"] = contracts
    aggregate = {
        key: value
        for key, value in report["aggregate"].items()
        if key not in _GENERIC_AGGREGATE_DROP
    }
    report["aggregate"] = aggregate
    report["clips"] = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "canonical_time_law",
                "grounded_left_midpoint_right",
            }
        }
        for row in report["clips"]
    ]
    report["selected_registry_binding"] = dict(
        selected_registry_binding
    )
    report["non_claims"] = list(report["non_claims"]) + [
        "generic report projection omits but was gated by the exact swept-clearance and persisted-time-law receipts",
        "admission must reopen the exact generic bank-gate producer path and SHA before promotion",
        "bank-gate PASS is not training deployment or hardware authorization",
    ]
    return MappingProxyType(report)


def verify_generic_motion_bank(
    manifest_path: os.PathLike[str] | str,
    bank_directory: os.PathLike[str] | str,
    *,
    recipe_path: os.PathLike[str] | str,
    repo_root: os.PathLike[str] | str,
    registry_path: os.PathLike[str] | str,
    expected_registry_sha256: str,
    mjcf_path: os.PathLike[str] | str,
    urdf_path: os.PathLike[str] | str,
    body_order_path: os.PathLike[str] | str,
    expected_compiled_signature: str,
    swept_clearance_receipt_path: os.PathLike[str] | str,
    expected_swept_clearance_receipt_sha256: str,
    plant_loader: Any = None,
    player_runner: Any = None,
    dynamics_runner: Any = None,
    grounded_lmr_runner: Any = None,
) -> Mapping[str, Any]:
    """Return a diagnostic generic-v2 report only after the complete gate."""

    root = _absolute(repo_root).resolve(strict=True)
    loaded = arbitrary.load_arbitrary_bank_recipe(
        recipe_path,
        repo_root=root,
    )
    manifest_file = _absolute(manifest_path)
    bank = _absolute(bank_directory)
    manifest, matrix = _manifest_and_matrix(
        loaded, manifest_file, bank
    )
    selected, _binding = _registry_binding(
        registry_path=_absolute(registry_path),
        expected_registry_sha256=_digest(
            expected_registry_sha256,
            "expected generic registry SHA-256",
        ),
        repo_root=root,
        loaded=loaded,
        manifest=manifest,
        matrix=matrix,
    )
    report = _run_independent_engine(
        loaded=loaded,
        manifest_path=manifest_file,
        bank_directory=bank,
        mjcf_path=_absolute(mjcf_path),
        urdf_path=_absolute(urdf_path),
        body_order_path=_absolute(body_order_path),
        expected_compiled_signature=_digest(
            expected_compiled_signature,
            "expected compiled model signature",
        ),
        swept_clearance_receipt_path=_absolute(
            swept_clearance_receipt_path
        ),
        expected_swept_clearance_receipt_sha256=_digest(
            expected_swept_clearance_receipt_sha256,
            "expected swept-clearance receipt SHA-256",
        ),
        engine_recipe=_engine_recipe(loaded),
        plant_loader=plant_loader,
        player_runner=player_runner,
        dynamics_runner=dynamics_runner,
        grounded_lmr_runner=grounded_lmr_runner,
    )
    return _generic_v2_report(
        report,
        selected_registry_binding=selected,
    )


def _failure_report(exc: Exception) -> Mapping[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "verdict": "FAIL",
        "bank_gate_pass": False,
        "candidate_integrity_pass": False,
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "error": f"{type(exc).__name__}: {exc}",
        "non_claims": [
            "failure is fail-closed",
            "no training deployment or hardware capability was minted",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--body-order", required=True)
    parser.add_argument("--expected-compiled-signature", required=True)
    parser.add_argument("--swept-clearance-receipt", required=True)
    parser.add_argument(
        "--expected-swept-clearance-receipt-sha256",
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_generic_motion_bank(
            args.manifest,
            args.bank_dir,
            recipe_path=args.recipe,
            repo_root=args.repo_root,
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
            mjcf_path=args.mjcf,
            urdf_path=args.urdf,
            body_order_path=args.body_order,
            expected_compiled_signature=args.expected_compiled_signature,
            swept_clearance_receipt_path=args.swept_clearance_receipt,
            expected_swept_clearance_receipt_sha256=(
                args.expected_swept_clearance_receipt_sha256
            ),
        )
    except Exception as exc:
        report = _failure_report(exc)
    try:
        bank_gate.write_bank_report_no_clobber(report, args.output)
    except (OSError, FileExistsError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("bank_gate_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
