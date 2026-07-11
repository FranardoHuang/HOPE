#!/usr/bin/env python3
"""Audit one diagnostic GVHMR-to-A3 GMR result and its warm-up evidence.

Passing this audit proves only a finite 30 Hz, 31-DoF structural result with
the expected number of frames and a logged frame-zero IK warm-up convergence.
The source still uses video-estimated body betas, so this result is never a
formal motion-library or robot-safety acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


class ResultError(ValueError):
    """A GMR output or warm-up log violates the diagnostic contract."""


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
WARMUP_TOKEN = re.compile(r"warm(?:[\s_-]*up)?", re.IGNORECASE)
ROUND_PATTERNS = (
    re.compile(r"(?:pass|round|iter(?:ation)?)\s*[:=#]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*(?:passes|rounds|iterations)\b", re.IGNORECASE),
    re.compile(r"(?:converged|finished)\s+(?:in|after)\s+(\d+)\s*"
               r"(?:passes|rounds|iterations)", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*/\s*\d+\b"),
)
MAX_DQ_PATTERNS = (
    re.compile(
        rf"max\s*\|\s*dq\s*\|\s*[:=<>]?\s*({NUMBER_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"max\s*\|\s*(?:\N{{GREEK CAPITAL LETTER DELTA}}|delta)\s*q\s*\|"
        rf"\s*[:=<>]?\s*({NUMBER_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:final[\s_-]*)?max(?:imum)?[\s_-]*"
        rf"(?:abs(?:olute)?[\s_-]*)?(?:(?:delta|\N{{GREEK CAPITAL LETTER DELTA}})"
        rf"[\s_-]*)?d?q\s*[:=<>]?\s*({NUMBER_PATTERN})",
        re.IGNORECASE,
    ),
)


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


def _numeric_array(value: Any, name: str) -> np.ndarray:
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
        raise ResultError("GMR result root must be a mapping")
    missing = [name for name in ("fps", "root_pos", "root_rot", "dof_pos") if name not in payload]
    if missing:
        raise ResultError(f"GMR result missing {missing}")

    fps_array = _numeric_array(payload["fps"], "fps")
    if fps_array.size != 1:
        raise ResultError(f"fps must be scalar, got shape={fps_array.shape}")
    fps = float(fps_array.reshape(-1)[0])
    if abs(fps - 30.0) > 1e-9:
        raise ResultError(f"fps={fps}, expected exactly 30")

    arrays = {
        name: _numeric_array(payload[name], name)
        for name in ("root_pos", "root_rot", "dof_pos")
    }
    expected_shapes = {
        "root_pos": (expected_frames, 3),
        "root_rot": (expected_frames, 4),
        "dof_pos": (expected_frames, 31),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ResultError(f"{name} shape {arrays[name].shape}, expected {shape}")
    quat_norm_error = np.abs(np.linalg.norm(arrays["root_rot"], axis=1) - 1.0)
    max_quat_norm_error = float(np.max(quat_norm_error))
    if max_quat_norm_error > 1e-3:
        raise ResultError(
            f"root_rot quaternion max norm error={max_quat_norm_error}, required <= 0.001"
        )

    return {
        "actual_frames": expected_frames,
        "fps": fps,
        "root_rotation_convention": "xyzw",
        "root_rotation_convention_evidence": (
            "bound GMR output schema; component ordering is not inferable from values alone"
        ),
        "root_rotation_max_norm_error": max_quat_norm_error,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "finite_elements": int(sum(value.size for value in arrays.values()) + 1),
    }


def _default_warmup_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not WARMUP_TOKEN.search(line):
            continue
        round_match = None
        for pattern in ROUND_PATTERNS:
            round_match = pattern.search(line)
            if round_match is not None:
                break
        dq_match = None
        for pattern in MAX_DQ_PATTERNS:
            dq_match = pattern.search(line)
            if dq_match is not None:
                break
        if round_match is None or dq_match is None:
            continue
        events.append(
            {
                "rounds": int(round_match.group(1)),
                "max_dq": float(dq_match.group(1)),
                "line_number": line_number,
                "matched_line": line.strip()[:500],
            }
        )
    return events


def parse_warmup_evidence(
    text: str,
    *,
    threshold: float = 1e-4,
    max_rounds: int = 200,
    custom_regex: str | None = None,
) -> dict[str, Any]:
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ResultError("warm-up threshold must be finite and positive")
    if not isinstance(max_rounds, int) or max_rounds <= 0:
        raise ResultError("warm-up max_rounds must be a positive integer")
    if custom_regex is None:
        events = _default_warmup_events(text)
        parser = "built-in"
    else:
        try:
            pattern = re.compile(custom_regex, re.MULTILINE)
        except re.error as exc:
            raise ResultError(f"invalid warm-up regex: {exc}") from None
        if not {"rounds", "max_dq"}.issubset(pattern.groupindex):
            raise ResultError("custom warm-up regex requires named groups 'rounds' and 'max_dq'")
        events = []
        for match in pattern.finditer(text):
            try:
                rounds = int(match.group("rounds"))
                max_dq = float(match.group("max_dq"))
            except (TypeError, ValueError):
                raise ResultError("custom warm-up regex captured an invalid rounds/max_dq value") from None
            events.append(
                {
                    "rounds": rounds,
                    "max_dq": max_dq,
                    "line_number": text.count("\n", 0, match.start()) + 1,
                    "matched_line": match.group(0).strip()[:500],
                }
            )
        parser = "custom_regex"

    if not events:
        raise ResultError(
            "no parseable frame-zero warm-up convergence evidence; provide the exact "
            "log format with --warmup-regex"
        )
    final = events[-1]
    if final["rounds"] <= 0 or final["rounds"] > max_rounds:
        raise ResultError(
            f"final warm-up rounds={final['rounds']}, required 1..{max_rounds}"
        )
    if not np.isfinite(final["max_dq"]) or final["max_dq"] >= threshold:
        raise ResultError(
            f"final warm-up max_dq={final['max_dq']}, required < {threshold}"
        )
    return {
        "parser": parser,
        "matched_events": len(events),
        "rounds": final["rounds"],
        "max_dq": final["max_dq"],
        "threshold_strict_lt": threshold,
        "max_rounds": max_rounds,
        "line_number": final["line_number"],
        "matched_line": final["matched_line"],
    }


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # A generated GMR pickle may reference numpy/torch classes.
        raise ResultError(f"cannot load GMR pickle {path}: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--warmup-threshold", type=float, default=1e-4)
    parser.add_argument("--warmup-max-rounds", type=int, default=200)
    parser.add_argument("--warmup-regex")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = args.result.resolve()
    run_log = args.run_log.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "result_path": str(result),
        "run_log_path": str(run_log),
        "expected_frames": args.expected_frames,
        "body_shape_contract": "diagnostic_video_betas",
        "formal_eligible": False,
    }
    try:
        if not result.is_file() or result.stat().st_size <= 0:
            raise ResultError(f"missing or empty GMR result: {result}")
        if not run_log.is_file() or run_log.stat().st_size <= 0:
            raise ResultError(f"missing or empty GMR run log: {run_log}")
        report["result_bytes"] = result.stat().st_size
        report["result_sha256"] = sha256_file(result)
        report["run_log_bytes"] = run_log.stat().st_size
        report["run_log_sha256"] = sha256_file(run_log)
        report.update(validate_payload(load_pickle(result), args.expected_frames))
        report["warmup"] = parse_warmup_evidence(
            run_log.read_text(encoding="utf-8", errors="replace"),
            threshold=args.warmup_threshold,
            max_rounds=args.warmup_max_rounds,
            custom_regex=args.warmup_regex,
        )
        report["status"] = "pass"
        atomic_json(args.json_out.resolve(), report)
        print(
            f"[audit-gmr-result] PASS frames={report['actual_frames']} "
            f"warmup_rounds={report['warmup']['rounds']} "
            f"max_dq={report['warmup']['max_dq']:.6g} "
            f"sha256={report['result_sha256'][:12]}..."
        )
        return 0
    except (OSError, ResultError) as exc:
        report.update(status="fail", error=str(exc))
        atomic_json(args.json_out.resolve(), report)
        print(f"[audit-gmr-result] FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
