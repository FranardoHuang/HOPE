#!/usr/bin/env python3
"""Validate the structural/finite contract of one GVHMR result.

This is deliberately narrower than visual motion acceptance.  Passing means
the output is a loadable, finite SMPL parameter sequence with the expected
source-frame count; it does not prove pose quality, mirror correctness,
contact phase, A3 safety, or returnability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class ResultError(ValueError):
    """A GVHMR output violates the expected structural contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and hasattr(value, "cpu"):
        value = value.float().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ResultError(f"{name} must be numeric, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        bad = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ResultError(f"{name} contains {bad} non-finite value(s)")
    return array


def validate_payload(payload: Any, expected_frames: int) -> dict[str, Any]:
    if not isinstance(expected_frames, int) or expected_frames <= 1:
        raise ResultError("expected_frames must be an integer > 1")
    if not isinstance(payload, dict):
        raise ResultError("GVHMR result root must be a mapping")
    params = payload.get("smpl_params_global")
    if not isinstance(params, dict):
        raise ResultError("missing mapping smpl_params_global")

    required = ("body_pose", "betas", "global_orient", "transl")
    missing = [name for name in required if name not in params]
    if missing:
        raise ResultError(f"smpl_params_global missing {missing}")
    arrays = {name: _array(params[name], f"smpl_params_global.{name}") for name in required}

    n = expected_frames
    expected_shapes = {
        "body_pose": (n, 63),
        "global_orient": (n, 3),
        "transl": (n, 3),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ResultError(f"{name} shape {arrays[name].shape}, expected {shape}")
    betas_shape = arrays["betas"].shape
    if betas_shape not in {(10,), (1, 10), (n, 10)}:
        raise ResultError(
            f"betas shape {betas_shape}, expected (10,), (1, 10), or ({n}, 10)"
        )

    return {
        "actual_frames": n,
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "finite_elements": int(sum(array.size for array in arrays.values())),
    }


def load_torch_result(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ResultError(f"torch is required to load {path}: {exc}") from None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = args.result.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "result_path": str(result),
        "expected_frames": args.expected_frames,
    }
    try:
        if not result.is_file() or result.stat().st_size <= 0:
            raise ResultError(f"missing or empty GVHMR result: {result}")
        report["result_bytes"] = result.stat().st_size
        report["result_sha256"] = sha256_file(result)
        report.update(validate_payload(load_torch_result(result), args.expected_frames))
        report["status"] = "pass"
        atomic_json(args.json_out.resolve(), report)
        print(
            f"[audit-gvhmr-result] PASS frames={report['actual_frames']} "
            f"finite={report['finite_elements']} sha256={report['result_sha256'][:12]}..."
        )
        return 0
    except (OSError, ResultError) as exc:
        report.update(status="fail", error=str(exc))
        atomic_json(args.json_out.resolve(), report)
        print(f"[audit-gvhmr-result] FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
