#!/usr/bin/env python3
"""Negative production-runner preflight for the formal face179 model contract.

This never starts a backend. It first requires one publishable envelope-bearing 179-D ONNX, then
creates temporary metadata-mutated copies and proves each is rejected before backend Init.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


BACKEND_MARKERS = (
    "backend cfg",
    "A3AimrtBackend initialised",
    "A3AimrtBackend started",
)
REQUIRED_ACCEPT_MARKERS = (
    "backend_not_initialized=true",
    "publishable_model_contract=true",
    "training_contract_exact=1",
    "obs_dim=179",
)
VARIANTS = (
    "metadata_stripped",
    "missing_envelope",
    "training_contract_inexact",
    "actor_leg_ref_mask_unsupported",
)


def mutate_metadata(metadata: dict[str, str], variant: str) -> dict[str, str]:
    """Return one deliberate contract violation without mutating the caller's mapping."""

    result = copy.deepcopy(metadata)
    if variant == "metadata_stripped":
        result.clear()
    elif variant == "missing_envelope":
        result.pop("stage1_normal_envelope_payload_sha256", None)
    elif variant == "training_contract_inexact":
        result["training_contract_exact"] = "0"
    elif variant == "actor_leg_ref_mask_unsupported":
        result["actor_leg_ref_mask"] = "1"
    else:
        raise ValueError(f"unknown preflight mutation variant: {variant}")
    return result


def preflight_command(
    runner: str | os.PathLike[str],
    runtime_cfg: str | os.PathLike[str],
    model: str | os.PathLike[str],
) -> list[str]:
    return [
        str(runner),
        "--runtime-cfg", str(runtime_cfg),
        "--model-path", str(model),
        "--planner", "--no-publish", "--model-preflight-only",
    ]


def check_result(result: subprocess.CompletedProcess[str], *, should_accept: bool) -> None:
    combined = f"{result.stdout}\n{result.stderr}"
    if any(marker in combined for marker in BACKEND_MARKERS):
        raise RuntimeError("preflight touched or announced a backend")
    if should_accept:
        if result.returncode != 0:
            raise RuntimeError(f"publishable baseline preflight failed rc={result.returncode}: {combined}")
        missing = [marker for marker in REQUIRED_ACCEPT_MARKERS if marker not in combined]
        if missing:
            raise RuntimeError(f"accepted preflight omitted parsed-contract markers: {missing}")
    elif result.returncode == 0:
        raise RuntimeError(f"invalid model unexpectedly passed preflight: {combined}")


def _run(command: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )


def run_suite(
    runner: Path,
    runtime_cfg: Path,
    model_path: Path,
    *,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    import onnx  # Optional integration dependency; helper unit tests stay dependency-light.

    for path, label in ((runner, "runner"), (runtime_cfg, "runtime config"), (model_path, "model")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    model = onnx.load(model_path)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    obs = next((value for value in model.graph.input if value.name == "obs"), None)
    if obs is None or obs.type.tensor_type.shape.dim[1].dim_value != 179:
        raise RuntimeError("integration baseline must be a 179-D ONNX")
    if metadata.get("training_contract_exact") != "1":
        raise RuntimeError("integration baseline must declare training_contract_exact=1")
    if not metadata.get("stage1_normal_envelope_payload_sha256"):
        raise RuntimeError("integration baseline must carry the formal normal envelope")

    baseline = _run(
        preflight_command(runner, runtime_cfg, model_path), timeout_s=timeout_s
    )
    check_result(baseline, should_accept=True)
    results: dict[str, Any] = {
        "baseline": {"returncode": baseline.returncode},
        "variants": {},
    }
    with tempfile.TemporaryDirectory(prefix="face179_preflight_") as directory:
        root = Path(directory)
        for variant in VARIANTS:
            mutated = copy.deepcopy(model)
            values = mutate_metadata(metadata, variant)
            del mutated.metadata_props[:]
            for key, value in sorted(values.items()):
                entry = mutated.metadata_props.add()
                entry.key, entry.value = key, value
            variant_path = root / f"{variant}.onnx"
            onnx.save(mutated, variant_path)
            result = _run(
                preflight_command(runner, runtime_cfg, variant_path), timeout_s=timeout_s
            )
            check_result(result, should_accept=False)
            results["variants"][variant] = {"returncode": result.returncode}

    forbidden = _run(
        preflight_command(runner, runtime_cfg, model_path)
        + ["--allow-legacy-model-diagnostic"],
        timeout_s=timeout_s,
    )
    check_result(forbidden, should_accept=False)
    if forbidden.returncode != 2:
        raise RuntimeError(
            "legacy-diagnostic + preflight must fail command validation with rc=2, got "
            f"{forbidden.returncode}"
        )
    results["legacy_flag_with_preflight"] = {"returncode": forbidden.returncode}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runtime-cfg", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()
    result = run_suite(
        args.runner.resolve(),
        args.runtime_cfg.resolve(),
        args.model.resolve(),
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
