#!/usr/bin/env python3
"""Bind, verify, or prepare a semantics-correct plant-contract v1.

This tool is offline and launch-free.  It never edits training checkouts,
starts a simulator, or authorizes hardware commands.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/plant_contract.py"
)
SPEC = importlib.util.spec_from_file_location("plant_contract_v1", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load plant contract module at {MODULE_PATH}")
PLANT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANT)
PlantContractError = PLANT.PlantContractError
bind_contract_sha256 = PLANT.bind_contract_sha256
prepare_runtime_adapter = PLANT.prepare_runtime_adapter
validate_plant_contract = PLANT.validate_plant_contract


def _read_json(path: Path, where: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlantContractError(f"cannot read {where} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise PlantContractError(f"{where} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser(
        "bind", help="bind the canonical contract SHA and validate the result"
    )
    bind.add_argument("draft", type=Path)
    bind.add_argument("output", type=Path)

    verify = subparsers.add_parser("verify", help="validate a bound contract")
    verify.add_argument("contract", type=Path)

    prepare = subparsers.add_parser(
        "prepare", help="compile one engine adapter for an in-support requested envelope"
    )
    prepare.add_argument("contract", type=Path)
    prepare.add_argument("--engine", choices=("physx", "mujoco"), required=True)
    prepare.add_argument("--requested-support", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "bind":
            bound = bind_contract_sha256(_read_json(args.draft, "draft contract"))
            normalized = validate_plant_contract(bound)
            _write_json(args.output, bound)
            print("PLANT_CONTRACT_BIND_OK")
            print(f"contract_sha256={normalized['contract_sha256']}")
            print("hardware_commands_authorized=false")
            return 0

        contract = _read_json(args.contract, "plant contract")
        if args.command == "verify":
            normalized = validate_plant_contract(contract)
            print("PLANT_CONTRACT_VERIFY_OK")
            print(f"contract_sha256={normalized['contract_sha256']}")
            print("hardware_commands_authorized=false")
            return 0

        support = _read_json(args.requested_support, "requested support")
        runtime = prepare_runtime_adapter(
            contract,
            engine=args.engine,
            requested_support=support,
        )
        _write_json(args.output, runtime)
        print("PLANT_RUNTIME_ADAPTER_PREPARE_OK")
        print(f"engine={args.engine}")
        print(f"runtime_adapter_sha256={runtime['runtime_adapter_sha256']}")
        print("hardware_commands_authorized=false")
        return 0
    except PlantContractError as exc:
        print(f"PLANT_CONTRACT_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
