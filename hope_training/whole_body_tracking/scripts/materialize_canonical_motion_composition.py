#!/usr/bin/env python3
"""Materialize the exact fresh-N5 7x2 compiler bank without rebuilding bytes.

The canonical compiler deliberately publishes the historical five-motion base
and the two-motion append suffix as two independently verifiable banks.  The
fresh-N5 prototype consumer, however, requires one complete ordered 7x2
``BUILD_MANIFEST.json`` and all referenced files under one directory.  This
tool closes only that packaging gap:

* reopen the exact base and append manifests by caller-supplied SHA-256;
* require both inputs to satisfy the canonical bank-gate manifest schema;
* require an unchanged five-motion prefix and zero station shift;
* strictly reopen every schema-2 and time-law artifact;
* copy the already-compiled bytes into a new no-clobber directory;
* publish a composed manifest plus a content-bound composition receipt.

It never compiles, retimes, shifts, authorizes, or edits a motion artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_stroke_prototypes as prototype_consumer  # noqa: E402
import canonical_motion_bank_gate as bank_gate  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402


BASE_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
APPEND_MOTION_IDS = ("fh_loop_high", "v12_forehand_block")
COMPOSED_MOTION_IDS = BASE_MOTION_IDS + APPEND_MOTION_IDS
SCOPES = ("upper", "full")
MANIFEST_NAME = "BUILD_MANIFEST.json"
RECEIPT_NAME = "COMPOSITION_RECEIPT.json"
RECEIPT_CLASS = "fresh_n5_canonical_motion_composition_v1"
_DIGEST_LENGTH = 64
_TIME_LAW_KEYS = frozenset(
    {
        "npz_filename",
        "npz_sha256",
        "manifest_filename",
        "manifest_sha256",
        "bundle_sha256",
        "schema_version",
        "artifact_type",
    }
)
_COMPOSITION_KEYS = frozenset(
    {
        "mode",
        "base_outputs_rebuilt",
        "base_recipe",
        "base_build_manifest",
        "base_output_matrix",
        "base_outputs",
        "appended_motion_ids",
        "appended_scopes",
        "station_center_shift_xy_m",
        "composed_candidate_count",
    }
)


class CanonicalMotionCompositionError(RuntimeError):
    """The two compiler banks cannot form the exact fresh-N5 composition."""


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CanonicalMotionCompositionError(
            f"{label} must be a lowercase SHA-256 string"
        )
    if (
        len(value) != _DIGEST_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CanonicalMotionCompositionError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _real_directory(path: os.PathLike[str] | str, label: str) -> Path:
    result = _absolute(path)
    try:
        metadata = result.lstat()
    except OSError as exc:
        raise CanonicalMotionCompositionError(
            f"cannot inspect {label} {result}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or result.is_symlink():
        raise CanonicalMotionCompositionError(
            f"{label} must be a real non-symlink directory: {result}"
        )
    return result


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise CanonicalMotionCompositionError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except CanonicalMotionCompositionError:
        raise
    except OSError as exc:
        raise CanonicalMotionCompositionError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise CanonicalMotionCompositionError(
            f"{label} changed during stable read: {path}"
        )
    return b"".join(chunks)


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalMotionCompositionError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CanonicalMotionCompositionError(
            f"{label} contains non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except CanonicalMotionCompositionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalMotionCompositionError(
            f"cannot parse {label}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CanonicalMotionCompositionError(
            f"{label} must contain one JSON object"
        )
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalMotionCompositionError(
            f"composition JSON is not strict: {exc}"
        ) from exc


def _matrix(motion_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (motion_id, scope)
        for motion_id in motion_ids
        for scope in SCOPES
    )


def _validate_manifest_shell(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    label: str,
) -> None:
    try:
        bank_gate._validate_top_contract(
            manifest,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        raise CanonicalMotionCompositionError(
            f"{label} does not satisfy the canonical bank-gate schema: {exc}"
        ) from exc


def _validate_artifact_row(
    row: Mapping[str, Any],
    *,
    index: int,
    expected_key: tuple[str, str],
    bank_dir: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    motion_id, scope = expected_key
    if frozenset(row) != bank_gate._OUTPUT_KEYS_WITH_TIME_LAW:
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] keys differ from the exact "
            "time-law bank-gate row schema"
        )
    if (
        row.get("motion_id") != motion_id
        or row.get("scope") != scope
    ):
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] changed order or identity; expected "
            f"{expected_key!r}"
        )
    filename = f"{motion_id}_{scope}_canonical_v2.npz"
    if row.get("filename") != filename:
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] filename must be {filename!r}"
        )
    output_sha = _digest(
        row.get("output_npz_sha256"),
        f"{label} outputs[{index}].output_npz_sha256",
    )
    payloads: dict[str, bytes] = {}
    npz_path = bank_dir / filename
    payloads[filename] = _read_regular(
        npz_path, f"{label} outputs[{index}] NPZ"
    )
    if _sha256(payloads[filename]) != output_sha:
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] NPZ hash drifted"
        )

    for suffix, row_key in (
        (".manifest.json", "schema2_manifest"),
        (".report.json", "schema2_report"),
    ):
        sidecar_name = filename + suffix
        payload = _read_regular(
            bank_dir / sidecar_name,
            f"{label} outputs[{index}] {row_key} sidecar",
        )
        sidecar = _strict_json_bytes(
            payload, f"{label} outputs[{index}] {row_key} sidecar"
        )
        if sidecar != row.get(row_key):
            raise CanonicalMotionCompositionError(
                f"{label} outputs[{index}] {row_key} sidecar disagrees "
                "with BUILD_MANIFEST"
            )
        payloads[sidecar_name] = payload

    time_law = row.get("time_law_artifact")
    if (
        not isinstance(time_law, Mapping)
        or frozenset(time_law) != _TIME_LAW_KEYS
    ):
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] lacks the exact time-law artifact row"
        )
    artifact_name = filename[: -len(".npz")] + ".time_law.npz"
    artifact_manifest_name = artifact_name + ".manifest.json"
    if (
        time_law.get("npz_filename") != artifact_name
        or time_law.get("manifest_filename") != artifact_manifest_name
        or time_law.get("schema_version")
        != compiler.time_law_artifact.ARTIFACT_SCHEMA_VERSION
        or time_law.get("artifact_type")
        != compiler.time_law_artifact.ARTIFACT_TYPE
    ):
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] time-law filename/schema changed"
        )
    artifact_path = bank_dir / artifact_name
    artifact_manifest_path = bank_dir / artifact_manifest_name
    try:
        artifact = compiler.time_law_artifact.read_time_law_artifact(
            artifact_path,
            manifest_path=artifact_manifest_path,
        )
    except Exception as exc:
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] time-law strict reopen failed: {exc}"
        ) from exc
    if (
        artifact.npz_sha256
        != _digest(
            time_law.get("npz_sha256"),
            f"{label} outputs[{index}] time-law npz_sha256",
        )
        or artifact.manifest_sha256
        != _digest(
            time_law.get("manifest_sha256"),
            f"{label} outputs[{index}] time-law manifest_sha256",
        )
        or artifact.bundle_sha256
        != _digest(
            time_law.get("bundle_sha256"),
            f"{label} outputs[{index}] time-law bundle_sha256",
        )
    ):
        raise CanonicalMotionCompositionError(
            f"{label} outputs[{index}] time-law hashes disagree with "
            "BUILD_MANIFEST"
        )
    payloads[artifact_name] = artifact.npz_bytes
    payloads[artifact_manifest_name] = artifact.manifest_bytes
    receipt = {
        "motion_id": motion_id,
        "scope": scope,
        "filename": filename,
        "sha256": output_sha,
        "schema2_manifest_sha256": _sha256(
            payloads[filename + ".manifest.json"]
        ),
        "schema2_report_sha256": _sha256(
            payloads[filename + ".report.json"]
        ),
        "time_law_npz_filename": artifact_name,
        "time_law_npz_sha256": artifact.npz_sha256,
        "time_law_manifest_filename": artifact_manifest_name,
        "time_law_manifest_sha256": artifact.manifest_sha256,
        "time_law_bundle_sha256": artifact.bundle_sha256,
    }
    return receipt, payloads


def _load_bank(
    *,
    manifest_path_raw: os.PathLike[str] | str,
    expected_manifest_sha256: str,
    bank_dir_raw: os.PathLike[str] | str,
    expected_motion_ids: tuple[str, ...],
    label: str,
) -> tuple[Path, Mapping[str, Any], str, list[dict[str, Any]], dict[str, bytes]]:
    bank_dir = _real_directory(bank_dir_raw, f"{label} bank directory")
    manifest_path = _absolute(manifest_path_raw)
    if manifest_path.parent != bank_dir or manifest_path.name != MANIFEST_NAME:
        raise CanonicalMotionCompositionError(
            f"{label} manifest must be {bank_dir / MANIFEST_NAME}"
        )
    manifest_bytes = _read_regular(
        manifest_path, f"{label} BUILD_MANIFEST"
    )
    manifest_sha = _sha256(manifest_bytes)
    if manifest_sha != _digest(
        expected_manifest_sha256,
        f"expected {label} BUILD_MANIFEST SHA-256",
    ):
        raise CanonicalMotionCompositionError(
            f"{label} BUILD_MANIFEST SHA-256 mismatch: expected "
            f"{expected_manifest_sha256}, got {manifest_sha}"
        )
    manifest = _strict_json_bytes(
        manifest_bytes, f"{label} BUILD_MANIFEST"
    )
    _validate_manifest_shell(manifest, manifest_path, label)
    expected_matrix = _matrix(expected_motion_ids)
    if manifest.get("output_matrix") != {
        "motion_ids": list(expected_motion_ids),
        "scopes": list(SCOPES),
        "candidate_count": len(expected_matrix),
    }:
        raise CanonicalMotionCompositionError(
            f"{label} output_matrix must be the exact ordered "
            f"{len(expected_motion_ids)}x2 matrix"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_matrix):
        raise CanonicalMotionCompositionError(
            f"{label} outputs must contain exactly {len(expected_matrix)} rows"
        )
    receipts: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for index, (raw_row, expected_key) in enumerate(
        zip(outputs, expected_matrix)
    ):
        if not isinstance(raw_row, Mapping):
            raise CanonicalMotionCompositionError(
                f"{label} outputs[{index}] must be one JSON object"
            )
        receipt, row_payloads = _validate_artifact_row(
            raw_row,
            index=index,
            expected_key=expected_key,
            bank_dir=bank_dir,
            label=label,
        )
        overlap = set(payloads).intersection(row_payloads)
        if overlap:
            raise CanonicalMotionCompositionError(
                f"{label} output filenames are duplicated: "
                f"{sorted(overlap)}"
            )
        receipts.append(receipt)
        payloads.update(row_payloads)
    return manifest_path, manifest, manifest_sha, receipts, payloads


def _resolve_composition_path(
    value: Any,
    *,
    append_manifest_path: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise CanonicalMotionCompositionError(
            f"{label} must be a non-empty path"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = append_manifest_path.parent / path
    return _absolute(path)


def _validate_append_binding(
    *,
    base_manifest_path: Path,
    base_manifest: Mapping[str, Any],
    base_manifest_sha: str,
    base_receipts: Sequence[Mapping[str, Any]],
    append_manifest_path: Path,
    append_manifest: Mapping[str, Any],
) -> None:
    composition = append_manifest.get("append_only_composition")
    if (
        not isinstance(composition, Mapping)
        or frozenset(composition) != _COMPOSITION_KEYS
    ):
        raise CanonicalMotionCompositionError(
            "append BUILD_MANIFEST lacks the exact append-only composition"
        )
    if (
        composition.get("mode")
        != "reuse_exact_base_outputs_compile_appended_only"
        or composition.get("base_outputs_rebuilt") is not False
        or composition.get("base_output_matrix")
        != base_manifest.get("output_matrix")
        or composition.get("appended_motion_ids")
        != list(APPEND_MOTION_IDS)
        or composition.get("appended_scopes") != list(SCOPES)
        or composition.get("station_center_shift_xy_m") != [0.0, 0.0]
        or append_manifest.get("station_center_shift_xy_m") != [0.0, 0.0]
        or composition.get("composed_candidate_count") != 14
    ):
        raise CanonicalMotionCompositionError(
            "append composition rebuilt, reordered, or station-shifted the "
            "fresh-N5 bank"
        )
    base_manifest_binding = composition.get("base_build_manifest")
    if not isinstance(base_manifest_binding, Mapping):
        raise CanonicalMotionCompositionError(
            "append composition base_build_manifest must be an object"
        )
    if (
        _resolve_composition_path(
            base_manifest_binding.get("path"),
            append_manifest_path=append_manifest_path,
            label="append composition base BUILD_MANIFEST",
        )
        != base_manifest_path
        or _digest(
            base_manifest_binding.get("sha256"),
            "append composition base BUILD_MANIFEST SHA-256",
        )
        != base_manifest_sha
    ):
        raise CanonicalMotionCompositionError(
            "append composition binds a different base BUILD_MANIFEST"
        )
    base_outputs = composition.get("base_outputs")
    if not isinstance(base_outputs, list) or len(base_outputs) != 10:
        raise CanonicalMotionCompositionError(
            "append composition must bind all ten base outputs"
        )
    for index, (raw, expected) in enumerate(
        zip(base_outputs, base_receipts)
    ):
        if not isinstance(raw, Mapping):
            raise CanonicalMotionCompositionError(
                f"append composition base_outputs[{index}] must be an object"
            )
        expected_path = base_manifest_path.parent / str(expected["filename"])
        if (
            frozenset(raw)
            != frozenset({"motion_id", "scope", "path", "sha256"})
            or raw.get("motion_id") != expected["motion_id"]
            or raw.get("scope") != expected["scope"]
            or raw.get("sha256") != expected["sha256"]
            or _resolve_composition_path(
                raw.get("path"),
                append_manifest_path=append_manifest_path,
                label=f"append composition base_outputs[{index}]",
            )
            != expected_path
        ):
            raise CanonicalMotionCompositionError(
                "append composition base output identity/order drifted"
            )
    shared_keys = (
        "compiler",
        "geometry_tool",
        "weighted_arc_tool",
        "compiler_options",
        "ready",
        "search_contract",
        "contact_opportunity_contract",
        "time_law_claim",
        "post_build_gates",
        "non_claims",
    )
    for key in shared_keys:
        if append_manifest.get(key) != base_manifest.get(key):
            raise CanonicalMotionCompositionError(
                f"base and append shared compiler contract {key!r} differs"
            )


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish(
    output: Path,
    payloads: Mapping[str, bytes],
) -> None:
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=str(output.parent),
        )
    )
    try:
        for name in sorted(payloads):
            if Path(name).name != name:
                raise CanonicalMotionCompositionError(
                    f"output filename escaped publication directory: {name!r}"
                )
            _write_exclusive(staging / name, payloads[name])
            if _read_regular(
                staging / name, f"staged composition output {name}"
            ) != payloads[name]:
                raise CanonicalMotionCompositionError(
                    f"staged composition output bytes changed: {name}"
                )
        directory_fd = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        compiler._rename_directory_noreplace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def materialize(
    *,
    base_manifest_path: os.PathLike[str] | str,
    expected_base_manifest_sha256: str,
    base_bank_dir: os.PathLike[str] | str,
    append_manifest_path: os.PathLike[str] | str,
    expected_append_manifest_sha256: str,
    append_bank_dir: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
) -> Mapping[str, Any]:
    """Materialize one exact, byte-preserving fresh-N5 7x2 composition."""

    output = _absolute(output_directory)
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )
    base_dir = _real_directory(base_bank_dir, "base bank directory")
    append_dir = _real_directory(append_bank_dir, "append bank directory")
    for source in (base_dir, append_dir):
        try:
            output.relative_to(source)
        except ValueError:
            pass
        else:
            raise CanonicalMotionCompositionError(
                "output directory may not be nested inside an input bank"
            )

    (
        base_path,
        base_manifest,
        base_manifest_sha,
        base_receipts,
        base_payloads,
    ) = _load_bank(
        manifest_path_raw=base_manifest_path,
        expected_manifest_sha256=expected_base_manifest_sha256,
        bank_dir_raw=base_dir,
        expected_motion_ids=BASE_MOTION_IDS,
        label="base",
    )
    (
        append_path,
        append_manifest,
        append_manifest_sha,
        append_receipts,
        append_payloads,
    ) = _load_bank(
        manifest_path_raw=append_manifest_path,
        expected_manifest_sha256=expected_append_manifest_sha256,
        bank_dir_raw=append_dir,
        expected_motion_ids=APPEND_MOTION_IDS,
        label="append",
    )
    _validate_append_binding(
        base_manifest_path=base_path,
        base_manifest=base_manifest,
        base_manifest_sha=base_manifest_sha,
        base_receipts=base_receipts,
        append_manifest_path=append_path,
        append_manifest=append_manifest,
    )
    overlap = set(base_payloads).intersection(append_payloads)
    if overlap:
        raise CanonicalMotionCompositionError(
            f"base and append artifact filenames overlap: {sorted(overlap)}"
        )

    composed_manifest = copy.deepcopy(dict(append_manifest))
    composed_manifest["library_id"] = (
        str(append_manifest["library_id"])
        + "__materialized_fresh_n5_7x2"
    )
    composed_manifest["output_matrix"] = {
        "motion_ids": list(COMPOSED_MOTION_IDS),
        "scopes": list(SCOPES),
        "candidate_count": 14,
    }
    composed_manifest["outputs"] = [
        *copy.deepcopy(list(base_manifest["outputs"])),
        *copy.deepcopy(list(append_manifest["outputs"])),
    ]
    try:
        selected = prototype_consumer._fresh_n5_upper_outputs(
            composed_manifest
        )
    except (Exception, SystemExit) as exc:
        raise CanonicalMotionCompositionError(
            f"composed manifest fails --fresh-n5-upper consumer: {exc}"
        ) from exc
    if tuple(row["motion_id"] for row in selected) != (
        "bh_loop_c",
        "v12_forehand_block",
        "bh_block",
        "s0_highpress",
        "fh_loop_high",
    ):
        raise CanonicalMotionCompositionError(
            "fresh-N5 upper consumer selected the wrong action order"
        )

    manifest_payload = _json_bytes(composed_manifest)
    all_receipts = [
        *(
            {**row, "source_bank": "base"}
            for row in base_receipts
        ),
        *(
            {**row, "source_bank": "append"}
            for row in append_receipts
        ),
    ]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_class": RECEIPT_CLASS,
        "verdict": "PASS_BYTE_PRESERVING_MATERIALIZATION",
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "hardware_authorized": False,
        "contract": {
            "motion_ids": list(COMPOSED_MOTION_IDS),
            "scopes": list(SCOPES),
            "candidate_count": 14,
            "base_outputs_rebuilt": False,
            "motion_bytes_modified": False,
            "retimed": False,
            "station_center_shift_xy_m": [0.0, 0.0],
            "prototype_fresh_n5_upper_order": [
                row["motion_id"] for row in selected
            ],
        },
        "source_banks": {
            "base": {
                "bank_directory": str(base_dir),
                "manifest_path": str(base_path),
                "manifest_sha256": base_manifest_sha,
            },
            "append": {
                "bank_directory": str(append_dir),
                "manifest_path": str(append_path),
                "manifest_sha256": append_manifest_sha,
            },
        },
        "build_manifest": {
            "path": MANIFEST_NAME,
            "sha256": _sha256(manifest_payload),
        },
        "outputs": all_receipts,
        "artifact_file_count": len(base_payloads) + len(append_payloads),
    }
    canonical_receipt = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    receipt["receipt_payload_sha256"] = _sha256(canonical_receipt)
    receipt_payload = _json_bytes(receipt)

    payloads = dict(base_payloads)
    payloads.update(append_payloads)
    payloads[MANIFEST_NAME] = manifest_payload
    payloads[RECEIPT_NAME] = receipt_payload
    _publish(output, payloads)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--base-bank-dir", required=True)
    parser.add_argument("--append-manifest", required=True)
    parser.add_argument("--append-manifest-sha256", required=True)
    parser.add_argument("--append-bank-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    return materialize(
        base_manifest_path=args.base_manifest,
        expected_base_manifest_sha256=args.base_manifest_sha256,
        base_bank_dir=args.base_bank_dir,
        append_manifest_path=args.append_manifest,
        expected_append_manifest_sha256=args.append_manifest_sha256,
        append_bank_dir=args.append_bank_dir,
        output_directory=args.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = run(argv)
    except (CanonicalMotionCompositionError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
