#!/usr/bin/env python3
"""Materialize one diagnostic body-shape vector into a GVHMR result cohort.

The input cohort is content-addressed by a preregistration JSON.  The canonical
vector is the coordinate-wise median of per-video coordinate-wise medians, so
longer clips do not receive more weight.  Only
``smpl_params_global.betas`` may change.  Every other supported payload leaf is
hashed before and after a save/reload round trip and must remain bit-exact.

This is deliberately a diagnostic body-shape normalization, not a measured
human-height calibration and not an A3/formal motion-library acceptance.  The
tool is no-clobber: the output root must not exist, and publication uses
exclusive creation/hard links so an existing artifact is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


TARGET_PATH = ("smpl_params_global", "betas")
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
AGGREGATION_METHOD = "coordinatewise_median_of_per_video_coordinatewise_medians"


class MaterializationError(ValueError):
    """The canonical-betas contract cannot be satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise MaterializationError(f"refusing to overwrite existing file: {path}") from None


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise MaterializationError(f"{label} root must be a mapping")
    return payload


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise MaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MaterializationError(f"torch is required to load GVHMR PT files: {exc}") from None
    return torch


def validate_execution_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan.get("execution_contract")
    if not isinstance(contract, dict):
        raise MaterializationError("plan.execution_contract must be a mapping")
    if contract.get("cpu_only") is not True:
        raise MaterializationError("execution_contract.cpu_only must be true")
    if contract.get("CUDA_VISIBLE_DEVICES") != "":
        raise MaterializationError(
            "execution_contract.CUDA_VISIBLE_DEVICES must be the empty string"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise MaterializationError("CUDA_VISIBLE_DEVICES must be the empty string")
    expected_executable = contract.get("python_executable")
    if not isinstance(expected_executable, str) or not expected_executable:
        raise MaterializationError("execution_contract.python_executable must be a string")
    if Path(sys.executable).resolve() != Path(expected_executable).resolve():
        raise MaterializationError(
            f"python executable {sys.executable} != expected {expected_executable}"
        )
    version = subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True, check=False
    )
    actual_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or actual_version != contract.get("python_version"):
        raise MaterializationError(
            f"python version {actual_version!r} != expected {contract.get('python_version')!r}"
        )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise MaterializationError(f"pip freeze failed: {freeze.stderr.strip()}")
    normalized = "\n".join(
        sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
    )
    actual_freeze_sha = hashlib.sha256((normalized + "\n").encode()).hexdigest()
    expected_freeze_sha = _require_sha(
        contract.get("pip_freeze_sha256"), "execution_contract.pip_freeze_sha256"
    )
    if actual_freeze_sha != expected_freeze_sha:
        raise MaterializationError(
            f"pip freeze sha256 {actual_freeze_sha} != expected {expected_freeze_sha}"
        )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": actual_version,
        "pip_freeze_sha256": actual_freeze_sha,
        "CUDA_VISIBLE_DEVICES": "",
        "cpu_only": True,
    }


def load_torch_result(path: Path, torch: Any) -> Any:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise MaterializationError(f"cannot load GVHMR result {path}: {exc}") from None


def validate_plan(
    plan_path: Path,
    expected_plan_sha256: str,
    source_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _require_sha(expected_plan_sha256, "--expected-plan-sha256")
    actual_plan_sha = sha256_file(plan_path)
    if actual_plan_sha != expected_plan_sha256:
        raise MaterializationError(
            f"plan sha256 {actual_plan_sha} != expected {expected_plan_sha256}"
        )
    plan = _read_json(plan_path, "canonical-betas plan")
    if plan.get("schema_version") != 1:
        raise MaterializationError("canonical-betas plan schema_version must be 1")
    if plan.get("status") != "preregistered_not_executed":
        raise MaterializationError("canonical-betas plan must remain preregistered_not_executed")
    if plan.get("formal_eligible") is not False:
        raise MaterializationError("canonical-betas plan must explicitly remain formal_eligible=false")
    if plan.get("a3_calibrated") is not False or plan.get("measured_height_m") is not None:
        raise MaterializationError(
            "this lane requires a3_calibrated=false and measured_height_m=null"
        )
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise MaterializationError(f"body_shape_contract must be {BODY_SHAPE_CONTRACT}")
    aggregation = plan.get("aggregation")
    if not isinstance(aggregation, dict):
        raise MaterializationError("plan.aggregation must be a mapping")
    if aggregation.get("method") != AGGREGATION_METHOD:
        raise MaterializationError(f"aggregation.method must be {AGGREGATION_METHOD}")
    if aggregation.get("same_performer_asserted") is not True:
        raise MaterializationError("same_performer_asserted=true is required")
    if not isinstance(aggregation.get("performer_group_id"), str):
        raise MaterializationError("aggregation.performer_group_id must be a string")
    if aggregation.get("expected_beta_components") != 10:
        raise MaterializationError("expected_beta_components must be 10")
    if aggregation.get("changed_field_allowlist") != ["smpl_params_global.betas"]:
        raise MaterializationError(
            "changed_field_allowlist must contain only smpl_params_global.betas"
        )
    blockers = plan.get("formal_blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or not all(isinstance(item, str) and item.strip() for item in blockers)
    ):
        raise MaterializationError("formal_blockers must be a non-empty string list")
    output_contract = plan.get("output_contract")
    if not isinstance(output_contract, dict):
        raise MaterializationError("plan.output_contract must be a mapping")
    required_output_contract = {
        "result_suffix": ".diagnostic_cohort_median_betas.pt",
        "canonical_betas_filename": "canonical_betas.json",
        "completion_manifest_filename": "materialization_manifest.json",
        "output_root_must_not_exist": True,
        "no_clobber": True,
        "completion_manifest_published_last": True,
    }
    for field, expected in required_output_contract.items():
        if output_contract.get(field) != expected:
            raise MaterializationError(f"output_contract.{field} must be {expected!r}")

    source_binding = plan.get("source_results_manifest")
    if not isinstance(source_binding, dict):
        raise MaterializationError("plan.source_results_manifest must be a mapping")
    expected_source_sha = _require_sha(
        source_binding.get("sha256"), "source_results_manifest.sha256"
    )
    actual_source_sha = sha256_file(source_manifest_path)
    if actual_source_sha != expected_source_sha:
        raise MaterializationError(
            f"source manifest sha256 {actual_source_sha} != expected {expected_source_sha}"
        )
    source_manifest = _read_json(source_manifest_path, "GVHMR result manifest")
    if source_manifest.get("status") != "complete":
        raise MaterializationError("GVHMR result manifest must be complete")
    if source_manifest.get("formal_eligible") is not False:
        raise MaterializationError("GVHMR result manifest must remain formal_eligible=false")
    source_rows = source_manifest.get("results")
    plan_rows = plan.get("inputs")
    if not isinstance(source_rows, list) or not isinstance(plan_rows, list) or not plan_rows:
        raise MaterializationError("source and plan results must be non-empty lists")
    expected_count = aggregation.get("expected_inputs")
    if not isinstance(expected_count, int) or expected_count < 2 or len(plan_rows) != expected_count:
        raise MaterializationError("aggregation.expected_inputs must match at least two plan inputs")
    source_by_id = {
        row.get("asset_id"): row for row in source_rows if isinstance(row, dict)
    }
    if len(source_by_id) != len(source_rows):
        raise MaterializationError("GVHMR result manifest has duplicate or malformed asset ids")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    bound_fields = ("result_path", "result_sha256", "result_bytes", "frames")
    for index, row in enumerate(plan_rows):
        if not isinstance(row, dict):
            raise MaterializationError(f"inputs[{index}] must be a mapping")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise MaterializationError(f"inputs[{index}].asset_id is unsafe: {asset_id!r}")
        if asset_id in seen_ids:
            raise MaterializationError(f"duplicate plan asset_id: {asset_id}")
        seen_ids.add(asset_id)
        source_row = source_by_id.get(asset_id)
        if not isinstance(source_row, dict):
            raise MaterializationError(f"plan asset is absent from source manifest: {asset_id}")
        for field in bound_fields:
            if row.get(field) != source_row.get(field):
                raise MaterializationError(
                    f"{asset_id}.{field} does not match the bound GVHMR result manifest"
                )
        _require_sha(row.get("result_sha256"), f"{asset_id}.result_sha256")
        if not isinstance(row.get("result_bytes"), int) or row["result_bytes"] <= 0:
            raise MaterializationError(f"{asset_id}.result_bytes must be positive")
        if not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise MaterializationError(f"{asset_id}.frames must be > 1")
        source_path = str(row.get("result_path"))
        if source_path in seen_paths:
            raise MaterializationError(f"duplicate GVHMR result path: {source_path}")
        seen_paths.add(source_path)
    if set(seen_ids) != set(source_by_id):
        raise MaterializationError(
            "plan must bind the complete GVHMR result cohort; subset materialization is forbidden"
        )
    return plan, source_manifest, actual_source_sha


def verify_source_file(row: Mapping[str, Any]) -> Path:
    path = Path(str(row["result_path"])).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise MaterializationError(f"{row['asset_id']}: missing/empty source {path}")
    if path.stat().st_size != row["result_bytes"]:
        raise MaterializationError(
            f"{row['asset_id']}: bytes {path.stat().st_size} != {row['result_bytes']}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != row["result_sha256"]:
        raise MaterializationError(
            f"{row['asset_id']}: sha256 {actual_sha} != {row['result_sha256']}"
        )
    return path


def _get_beta_value(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise MaterializationError("GVHMR result root must be a mapping")
    params = payload.get("smpl_params_global")
    if not isinstance(params, dict) or "betas" not in params:
        raise MaterializationError("missing smpl_params_global.betas")
    return params["betas"]


def beta_array_and_metadata(
    payload: Any,
    expected_frames: int,
    torch: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = _get_beta_value(payload)
    if torch is not None and torch.is_tensor(value):
        if value.layout != torch.strided:
            raise MaterializationError("betas tensor must use torch.strided layout")
        try:
            array = value.detach().cpu().numpy()
        except Exception as exc:
            raise MaterializationError(f"cannot convert betas tensor to numpy: {exc}") from None
        metadata = {
            "storage_kind": "torch_tensor",
            "dtype": str(value.dtype),
            "device_after_cpu_load": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "shape": list(value.shape),
            "stride": list(value.stride()),
        }
    elif isinstance(value, np.ndarray):
        array = value
        metadata = {
            "storage_kind": "numpy_ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "strides_bytes": list(value.strides),
        }
    else:
        raise MaterializationError(
            f"betas must be a torch tensor or numpy ndarray, got {type(value).__name__}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise MaterializationError(f"betas must use a real floating dtype, got {array.dtype}")
    if array.shape == (10,):
        matrix = array.reshape(1, 10)
        shape_contract = "vector_10"
    elif array.shape == (1, 10):
        matrix = array
        shape_contract = "singleton_by_10"
    elif array.shape == (expected_frames, 10):
        matrix = array
        shape_contract = "frames_by_10"
    else:
        raise MaterializationError(
            f"betas shape {array.shape}, expected (10,), (1, 10), or ({expected_frames}, 10)"
        )
    if not np.isfinite(matrix).all():
        raise MaterializationError("betas contain non-finite values")
    metadata["shape_contract"] = shape_contract
    metadata["numpy_dtype"] = array.dtype.str
    return np.asarray(matrix), metadata


def compute_canonical_vector(
    beta_matrices: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    if len(beta_matrices) < 2:
        raise MaterializationError("canonical aggregation requires at least two videos")
    per_video: list[np.ndarray] = []
    max_deviations: list[float] = []
    for matrix in beta_matrices:
        if matrix.ndim != 2 or matrix.shape[1] != 10:
            raise MaterializationError(f"internal beta matrix shape is invalid: {matrix.shape}")
        work = matrix.astype(np.float64, copy=False)
        median = np.median(work, axis=0)
        if not np.isfinite(median).all():
            raise MaterializationError("per-video median contains non-finite values")
        per_video.append(median)
        max_deviations.append(float(np.max(np.abs(work - median.reshape(1, 10)))))
    canonical = np.median(np.stack(per_video, axis=0), axis=0)
    if canonical.shape != (10,) or not np.isfinite(canonical).all():
        raise MaterializationError("cohort median is not a finite 10-vector")
    return canonical, per_video, max_deviations


def quantize_canonical_vector(vector: np.ndarray, numpy_dtype: str) -> np.ndarray:
    dtype = np.dtype(numpy_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise MaterializationError(f"cannot quantize canonical betas to {dtype}")
    quantized = np.asarray(vector, dtype=dtype)
    if quantized.shape != (10,) or not np.isfinite(quantized).all():
        raise MaterializationError("quantized canonical betas are invalid")
    return quantized


def replace_betas(payload: Any, canonical: np.ndarray, torch: Any | None) -> Any:
    output = copy.deepcopy(payload)
    params = output["smpl_params_global"]
    original = params["betas"]
    if torch is not None and torch.is_tensor(original):
        base = torch.as_tensor(canonical, dtype=original.dtype, device=original.device)
        replacement = torch.empty_like(original, memory_format=torch.preserve_format)
        if tuple(original.shape) == (10,):
            replacement.copy_(base)
        else:
            replacement.copy_(base.reshape(1, 10).expand(tuple(original.shape)))
        if original.requires_grad:
            replacement.requires_grad_(True)
    elif isinstance(original, np.ndarray):
        replacement = np.empty_like(original, order="K")
        if original.shape == (10,):
            replacement[...] = canonical
        else:
            replacement[...] = canonical.reshape(1, 10)
    else:  # beta_array_and_metadata already rejects this; keep fail-closed locally.
        raise MaterializationError("unsupported betas storage kind")
    params["betas"] = replacement
    return output


def _digest_token(digest: Any, token: str | bytes) -> None:
    data = token.encode("utf-8") if isinstance(token, str) else token
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def semantic_digest(
    payload: Any,
    torch: Any | None,
    excluded_path: tuple[str, ...] = TARGET_PATH,
) -> tuple[str, int]:
    """Hash supported payload leaves exactly, excluding only one named path."""

    digest = hashlib.sha256()
    leaf_count = 0

    def walk(value: Any, path: tuple[str, ...]) -> None:
        nonlocal leaf_count
        if path == excluded_path:
            _digest_token(digest, "EXCLUDED:" + ".".join(path))
            return
        if isinstance(value, Mapping):
            _digest_token(digest, f"mapping:{type(value).__module__}.{type(value).__qualname__}")
            _digest_token(digest, str(len(value)))
            for key, child in value.items():
                if not isinstance(key, str):
                    raise MaterializationError(
                        f"unsupported non-string mapping key at {'.'.join(path) or '<root>'}"
                    )
                _digest_token(digest, "key:" + key)
                walk(child, path + (key,))
            return
        if isinstance(value, list):
            _digest_token(digest, f"list:{len(value)}")
            for index, child in enumerate(value):
                walk(child, path + (f"[{index}]",))
            return
        if isinstance(value, tuple):
            _digest_token(digest, f"tuple:{len(value)}")
            for index, child in enumerate(value):
                walk(child, path + (f"[{index}]",))
            return
        if torch is not None and torch.is_tensor(value):
            if value.layout != torch.strided:
                raise MaterializationError(
                    f"unsupported tensor layout at {'.'.join(path)}: {value.layout}"
                )
            tensor = value.detach().cpu()
            try:
                raw = tensor.contiguous().numpy().tobytes(order="C")
            except Exception as exc:
                raise MaterializationError(
                    f"cannot byte-audit tensor at {'.'.join(path)}: {exc}"
                ) from None
            _digest_token(digest, "torch_tensor")
            _digest_token(digest, str(value.dtype))
            _digest_token(digest, json.dumps(list(value.shape)))
            _digest_token(digest, json.dumps(list(value.stride())))
            _digest_token(digest, str(int(value.storage_offset())))
            _digest_token(digest, str(bool(value.requires_grad)))
            _digest_token(digest, raw)
            leaf_count += 1
            return
        if isinstance(value, np.ndarray):
            _digest_token(digest, "numpy_ndarray")
            _digest_token(digest, value.dtype.str)
            _digest_token(digest, json.dumps(list(value.shape)))
            _digest_token(digest, json.dumps(list(value.strides)))
            _digest_token(digest, value.tobytes(order="A"))
            leaf_count += 1
            return
        if value is None:
            _digest_token(digest, "none")
        elif isinstance(value, bool):
            _digest_token(digest, b"bool:\x01" if value else b"bool:\x00")
        elif isinstance(value, int):
            _digest_token(digest, "int:" + str(value))
        elif isinstance(value, float):
            _digest_token(digest, b"float:" + struct.pack(">d", value))
        elif isinstance(value, str):
            _digest_token(digest, "str:" + value)
        elif isinstance(value, bytes):
            _digest_token(digest, b"bytes:" + value)
        elif isinstance(value, np.generic):
            _digest_token(digest, "numpy_scalar:" + value.dtype.str)
            _digest_token(digest, value.tobytes())
        else:
            raise MaterializationError(
                f"unsupported payload leaf at {'.'.join(path)}: {type(value).__name__}"
            )
        leaf_count += 1

    walk(payload, ())
    return digest.hexdigest(), leaf_count


def verify_materialized_betas(
    payload: Any,
    canonical: np.ndarray,
    expected_frames: int,
    expected_metadata: Mapping[str, Any],
    torch: Any | None,
) -> dict[str, Any]:
    matrix, metadata = beta_array_and_metadata(payload, expected_frames, torch)
    for field in ("storage_kind", "dtype", "shape", "shape_contract", "numpy_dtype"):
        if metadata.get(field) != expected_metadata.get(field):
            raise MaterializationError(f"materialized betas changed {field}")
    expected = np.broadcast_to(canonical.reshape(1, 10), matrix.shape)
    if matrix.dtype != canonical.dtype or matrix.tobytes(order="C") != expected.tobytes(order="C"):
        raise MaterializationError("materialized betas are not bit-exact canonical repetitions")
    return metadata


def save_and_reload_verified(
    output_payload: Any,
    input_non_beta_sha: str,
    canonical: np.ndarray,
    expected_frames: int,
    expected_beta_metadata: Mapping[str, Any],
    temporary_path: Path,
    torch: Any,
) -> tuple[str, int, str, int]:
    with temporary_path.open("xb") as handle:
        torch.save(output_payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    reloaded = load_torch_result(temporary_path, torch)
    output_non_beta_sha, output_leaf_count = semantic_digest(reloaded, torch)
    if output_non_beta_sha != input_non_beta_sha:
        raise MaterializationError(
            "save/reload changed payload content outside smpl_params_global.betas"
        )
    verify_materialized_betas(
        reloaded, canonical, expected_frames, expected_beta_metadata, torch
    )
    return (
        output_non_beta_sha,
        output_leaf_count,
        sha256_file(temporary_path),
        temporary_path.stat().st_size,
    )


def _vector_sha256(vector: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(vector.dtype.str.encode("ascii"))
    digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


def _publish_staged(staging: Path, output_root: Path, manifest_name: str) -> None:
    if output_root.exists():
        raise MaterializationError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError:
        raise MaterializationError(f"output root appeared during publication: {output_root}") from None
    names = sorted(path.name for path in staging.iterdir())
    if manifest_name not in names:
        raise MaterializationError("staging is missing the completion manifest")
    names.remove(manifest_name)
    names.append(manifest_name)  # Completion marker is always published last.
    for name in names:
        source = staging / name
        destination = output_root / name
        try:
            os.link(source, destination)
        except FileExistsError:
            raise MaterializationError(
                f"refusing to overwrite artifact that appeared during publication: {destination}"
            ) from None
    for source in staging.iterdir():
        source.unlink()
    staging.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load, hash and aggregate every source but write no output",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    expected_tool_sha = _require_sha(args.expected_tool_sha256, "--expected-tool-sha256")
    actual_tool_sha = sha256_file(script_path)
    if actual_tool_sha != expected_tool_sha:
        raise MaterializationError(
            f"tool sha256 {actual_tool_sha} != expected {expected_tool_sha}"
        )
    plan_path = args.plan.resolve()
    source_manifest_path = args.source_manifest.resolve()
    plan, _, source_manifest_sha = validate_plan(
        plan_path,
        args.expected_plan_sha256,
        source_manifest_path,
    )
    execution_fingerprint = validate_execution_contract(plan)
    output_root = args.output_root.resolve()
    planned_output_root = Path(str(plan["output_contract"].get("output_root", ""))).resolve()
    if output_root != planned_output_root:
        raise MaterializationError(
            f"output root {output_root} != preregistered {planned_output_root}"
        )
    if output_root.exists():
        raise MaterializationError(f"output root already exists: {output_root}")

    torch = _load_torch()
    loaded: list[dict[str, Any]] = []
    storage_contract: tuple[str, str, str] | None = None
    for row in plan["inputs"]:
        source = verify_source_file(row)
        payload = load_torch_result(source, torch)
        matrix, metadata = beta_array_and_metadata(payload, row["frames"], torch)
        contract = (
            str(metadata["storage_kind"]),
            str(metadata["dtype"]),
            str(metadata["numpy_dtype"]),
        )
        if storage_contract is None:
            storage_contract = contract
        elif contract != storage_contract:
            raise MaterializationError(
                f"mixed beta storage/dtype contracts are forbidden: {contract} != {storage_contract}"
            )
        non_beta_sha, non_beta_leaf_count = semantic_digest(payload, torch)
        # Detect a source mutation racing the load.
        if sha256_file(source) != row["result_sha256"]:
            raise MaterializationError(f"{row['asset_id']}: source mutated during load")
        loaded.append(
            {
                "row": row,
                "source": source,
                "payload": payload,
                "matrix": matrix,
                "beta_metadata": metadata,
                "input_non_beta_sha256": non_beta_sha,
                "input_non_beta_leaf_count": non_beta_leaf_count,
            }
        )
    assert storage_contract is not None
    raw_canonical, per_video, max_deviations = compute_canonical_vector(
        [item["matrix"] for item in loaded]
    )
    canonical = quantize_canonical_vector(raw_canonical, storage_contract[2])
    vector_sha = _vector_sha256(canonical)
    heuristic_height = 1.66 + 0.1 * float(canonical[0])
    if not math.isfinite(heuristic_height):
        raise MaterializationError("GMR heuristic height is non-finite")

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validation_pass",
                    "inputs": len(loaded),
                    "body_shape_contract": BODY_SHAPE_CONTRACT,
                    "canonical_vector_sha256": vector_sha,
                    "canonical_numpy_dtype": canonical.dtype.str,
                    "measured_height_m": None,
                    "gmr_loader_heuristic_height_m_not_calibration": heuristic_height,
                    "formal_eligible": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    try:
        staging.mkdir()
    except FileExistsError:
        raise MaterializationError(f"staging path already exists: {staging}") from None

    manifest_name = "materialization_manifest.json"
    canonical_name = "canonical_betas.json"
    result_rows: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(loaded):
            row = item["row"]
            output_name = f"{row['asset_id']}.diagnostic_cohort_median_betas.pt"
            temporary = staging / output_name
            output_payload = replace_betas(item["payload"], canonical, torch)
            # The in-memory source must also remain unchanged after deepcopy/replacement.
            source_sha_after, source_leaves_after = semantic_digest(item["payload"], torch)
            if (
                source_sha_after != item["input_non_beta_sha256"]
                or source_leaves_after != item["input_non_beta_leaf_count"]
            ):
                raise MaterializationError(f"{row['asset_id']}: source payload mutated in memory")
            verify_materialized_betas(
                output_payload,
                canonical,
                row["frames"],
                item["beta_metadata"],
                torch,
            )
            output_non_beta_sha, output_leaf_count, output_sha, output_bytes = (
                save_and_reload_verified(
                    output_payload,
                    item["input_non_beta_sha256"],
                    canonical,
                    row["frames"],
                    item["beta_metadata"],
                    temporary,
                    torch,
                )
            )
            result_rows.append(
                {
                    "asset_id": row["asset_id"],
                    "source_path": str(item["source"]),
                    "source_sha256": row["result_sha256"],
                    "source_bytes": row["result_bytes"],
                    "frames": row["frames"],
                    "source_beta_contract": item["beta_metadata"],
                    "source_beta_values_sha256": hashlib.sha256(
                        item["matrix"].tobytes(order="C")
                    ).hexdigest(),
                    "per_video_coordinatewise_median": [float(v) for v in per_video[index]],
                    "frame_beta_max_abs_deviation_from_video_median": max_deviations[index],
                    "output_path": str(output_root / output_name),
                    "output_sha256": output_sha,
                    "output_bytes": output_bytes,
                    "output_beta_contract": item["beta_metadata"],
                    "output_canonical_vector_sha256": vector_sha,
                    "non_beta_bit_exact": True,
                    "source_non_beta_semantic_sha256": item["input_non_beta_sha256"],
                    "output_non_beta_semantic_sha256": output_non_beta_sha,
                    "source_non_beta_leaf_count": item["input_non_beta_leaf_count"],
                    "output_non_beta_leaf_count": output_leaf_count,
                }
            )

        canonical_payload = {
            "schema_version": 1,
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "aggregation_method": AGGREGATION_METHOD,
            "performer_group_id": plan["aggregation"]["performer_group_id"],
            "same_performer_asserted": True,
            "equal_weight_per_video": True,
            "input_video_count": len(loaded),
            "dtype": storage_contract[1],
            "numpy_dtype": canonical.dtype.str,
            "components": [float(value) for value in canonical],
            "vector_sha256": vector_sha,
            "measured_height_m": None,
            "a3_calibrated": False,
            "gmr_loader_heuristic_height_m_not_calibration": heuristic_height,
            "formal_eligible": False,
        }
        _write_exclusive(staging / canonical_name, _json_bytes(canonical_payload))
        canonical_artifact_sha = sha256_file(staging / canonical_name)
        result_manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": utc_now(),
            "scope": (
                "diagnostic same-performer GVHMR beta normalization only; no measured-height, "
                "A3, GMR, collision, clearance, schema-2, returnability, training, or real-robot acceptance"
            ),
            "plan_path": str(plan_path),
            "plan_sha256": args.expected_plan_sha256,
            "tool_path": str(script_path),
            "tool_sha256": actual_tool_sha,
            "execution_fingerprint": execution_fingerprint,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha,
            "output_root": str(output_root),
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "aggregation": plan["aggregation"],
            "canonical_betas_artifact": {
                "path": str(output_root / canonical_name),
                "sha256": canonical_artifact_sha,
                "vector_sha256": vector_sha,
                "dtype": storage_contract[1],
                "numpy_dtype": canonical.dtype.str,
                "components": [float(value) for value in canonical],
            },
            "measured_height_m": None,
            "a3_calibrated": False,
            "gmr_loader_heuristic_height_m_not_calibration": heuristic_height,
            "formal_eligible": False,
            "formal_blockers": plan["formal_blockers"],
            "results": result_rows,
        }
        _write_exclusive(staging / manifest_name, _json_bytes(result_manifest))
        _publish_staged(staging, output_root, manifest_name)
    except Exception:
        # Staging is private to this invocation and never an accepted result.
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        f"[canonical-betas] PASS inputs={len(result_rows)} "
        f"vector_sha256={vector_sha} output_root={output_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MaterializationError as exc:
        print(f"[canonical-betas] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
