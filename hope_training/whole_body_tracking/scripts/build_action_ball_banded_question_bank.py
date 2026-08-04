#!/usr/bin/env python3
"""Build one deterministic, no-clobber banded bank from solved receipts.

This is an offline publisher, not a runtime solver.  Every input and the
reachable-domain coverage manifest are addressed as ``PATH=SHA256``.  Input
files carry admitted receipts plus proposal/rejection denominators and the
offline producer source/input roots.  Publication rejects both missing and
excess action/domain-level blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


MDP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
_SPEC = importlib.util.spec_from_file_location(
    "_offline_banded_question_bank", MDP_ROOT / "action_ball_banded_question_bank.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load the banded question-bank module")
_BANK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BANK
_SPEC.loader.exec_module(_BANK)
BandedQuestionBank = _BANK.BandedQuestionBank
BandedQuestionBlock = _BANK.BandedQuestionBlock
ActionBallTaskReceipt = _BANK._runtime.ActionBallTaskReceipt
DomainLevels = _BANK._runtime.ActionDomainLevels
_block_key_for_receipt = _BANK._block_key_for_receipt
_sha256_json = _BANK._sha256_json
load_banded_question_bank = _BANK.load_banded_question_bank
question_lineage_for_blocks = _BANK.question_lineage_for_blocks


class BuildError(RuntimeError):
    pass


def _parse_input(value: str) -> tuple[Path, str]:
    path_text, separator, expected = value.rpartition("=")
    if not separator or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise BuildError("--input must be PATH=lowercase_sha256")
    if not path_text:
        raise BuildError("--input path must be non-empty")
    return Path(path_text).expanduser().resolve(strict=True), expected


def _load_receipts(path: Path, expected_sha256: str):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise BuildError(
            f"input file SHA mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"input is not UTF-8 JSON: {path}") from exc
    expected_keys = {
        "schema_version",
        "kind",
        "source_id",
        "solver_mode",
        "offline_producer_source_sha256",
        "offline_input_root_sha256",
        "solve_ledger",
        "receipts",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or document["schema_version"] != 1
        or document["kind"] != "action_ball_offline_solved_receipts"
    ):
        raise BuildError(f"input does not use the exact offline solve schema: {path}")
    if type(document["source_id"]) is not str or not document["source_id"]:
        raise BuildError(f"input source_id must be non-empty: {path}")
    if document["solver_mode"] != "current_lm_only":
        raise BuildError(f"input solver_mode must be current_lm_only: {path}")
    for field in ("offline_producer_source_sha256", "offline_input_root_sha256"):
        value = document[field]
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise BuildError(f"input {field} must be lowercase SHA256: {path}")
    rows = document["receipts"]
    if not isinstance(rows, list) or not rows:
        raise BuildError(f"input contains no admitted task receipts: {path}")
    try:
        receipts = tuple(ActionBallTaskReceipt.from_dict(row) for row in rows)
    except Exception as exc:
        raise BuildError(f"invalid ActionBall task receipt in {path}: {exc}") from exc
    block_keys = {
        _sha256_json(_block_key_for_receipt(receipt)) for receipt in receipts
    }
    if len(block_keys) != 1:
        raise BuildError(
            f"each offline solve input must contain exactly one action/domain block: {path}"
        )
    ledger = document["solve_ledger"]
    if not isinstance(ledger, dict) or set(ledger) != {
        "proposed_count", "admitted_count", "rejections"
    }:
        raise BuildError(f"input solve_ledger schema mismatch: {path}")
    proposed = ledger["proposed_count"]
    admitted = ledger["admitted_count"]
    rejections = ledger["rejections"]
    if (
        type(proposed) is not int
        or proposed < 1
        or type(admitted) is not int
        or admitted != len(receipts)
        or admitted > proposed
        or not isinstance(rejections, list)
    ):
        raise BuildError(f"input solve denominators are invalid: {path}")
    reasons = []
    for item in rejections:
        if (
            not isinstance(item, dict)
            or set(item) != {"reason", "count"}
            or type(item["reason"]) is not str
            or not item["reason"]
            or type(item["count"]) is not int
            or item["count"] < 1
        ):
            raise BuildError(f"input rejection row is invalid: {path}")
        reasons.append({"reason": item["reason"], "count": item["count"]})
    if (
        len({item["reason"] for item in reasons}) != len(reasons)
        or sum(item["count"] for item in reasons) != proposed - admitted
    ):
        raise BuildError(f"input solve ledger does not conserve P=A+R: {path}")
    return actual, receipts, {
        "source_id": document["source_id"],
        "solver_mode": "current_lm_only",
        "block_key_sha256": next(iter(block_keys)),
        "offline_producer_source_sha256": document[
            "offline_producer_source_sha256"
        ],
        "offline_input_root_sha256": document["offline_input_root_sha256"],
        "proposed_count": proposed,
        "admitted_count": admitted,
        "rejections": sorted(reasons, key=lambda item: item["reason"]),
    }


def _load_coverage(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise BuildError(
            f"coverage file SHA mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"coverage is not UTF-8 JSON: {path}") from exc
    expected_keys = {
        "schema_version",
        "kind",
        "arm_catalog_sha256",
        "arm_keys",
        "expected_action_uids",
        "reachable_arm_keys_by_action",
        "reachable_blocks",
        "canonical_sha256",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or document["schema_version"] != 1
        or document["kind"] != "action_ball_reachable_domain_level_blocks"
    ):
        raise BuildError("coverage manifest schema/kind mismatch")
    unsigned = dict(document)
    declared = unsigned.pop("canonical_sha256")
    if declared != _sha256_json(unsigned):
        raise BuildError("coverage manifest canonical SHA mismatch")
    return {
        **unsigned,
        "source_file_sha256": actual,
        "source_canonical_sha256": declared,
    }


def build_bank(
    *, inputs: tuple[tuple[Path, str], ...], coverage: tuple[Path, str], split_seed: int
) -> BandedQuestionBank:
    if type(split_seed) is not int or split_seed < 0:
        raise BuildError("split_seed must be a non-negative integer")
    grouped: dict[str, list[object]] = {}
    keys: dict[str, dict] = {}
    lineage_inputs = []
    for path, expected in inputs:
        file_sha, receipts, input_lineage = _load_receipts(path, expected)
        ordered = tuple(sorted(receipts, key=lambda row: row.canonical_sha256))
        lineage_inputs.append(
            {
                **input_lineage,
                "file_sha256": file_sha,
                "receipt_canonical_sha256": [
                    row.canonical_sha256 for row in ordered
                ],
            }
        )
        for receipt in ordered:
            key = _block_key_for_receipt(receipt)
            key_sha = _sha256_json(key)
            keys[key_sha] = key
            grouped.setdefault(key_sha, []).append(receipt)
    blocks = tuple(
        BandedQuestionBlock(
            key=keys[key_sha],
            rows=tuple(
                sorted(grouped[key_sha], key=lambda row: row.canonical_sha256)
            ),
        )
        for key_sha in sorted(grouped)
    )
    coverage_document = _load_coverage(*coverage)
    lineage_inputs.sort(key=lambda item: (item["file_sha256"], item["source_id"]))
    return BandedQuestionBank(
        split_seed=split_seed,
        blocks=blocks,
        coverage=coverage_document,
        question_lineage=question_lineage_for_blocks(blocks),
        producer_lineage={
            "schema_version": 1,
            "kind": "action_ball_banded_question_bank.offline_solved_receipts",
            "row_order": "canonical_receipt_sha256",
            "producer_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "bank_module_source_sha256": hashlib.sha256(
                (MDP_ROOT / "action_ball_banded_question_bank.py").read_bytes()
            ).hexdigest(),
            "inputs": lineage_inputs,
        },
    )


def _publish_new(path: Path, raw: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.resolve(strict=True)
    if os.path.lexists(path):
        raise BuildError(f"no-clobber output already exists: {path}")
    fd, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BuildError(f"no-clobber output already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="solved receipt JSON addressed as PATH=lowercase_sha256",
    )
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument(
        "--coverage",
        required=True,
        help="complete reachable-domain manifest addressed as PATH=lowercase_sha256",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = tuple(_parse_input(value) for value in args.input)
    coverage = _parse_input(args.coverage)
    bank = build_bank(inputs=inputs, coverage=coverage, split_seed=args.split_seed)
    raw = (
        json.dumps(
            bank.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _publish_new(args.output, raw)
    file_sha = hashlib.sha256(raw).hexdigest()
    loaded = load_banded_question_bank(
        args.output, expected_file_sha256=file_sha
    )
    if loaded.to_dict() != bank.to_dict():
        raise BuildError("published bank failed strict roundtrip verification")
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "file_sha256": file_sha,
                "canonical_sha256": bank.canonical_sha256,
                "block_count": len(bank.blocks),
                "expected_action_count": len(
                    bank.coverage["expected_action_uids"]
                ),
                "offline_solve_ledger": bank.canonical_payload[
                    "offline_solve_ledger"
                ],
                "base_question_root_sha256": bank.question_lineage[
                    "base_question_root_sha256"
                ],
                "target_recipe": bank.canonical_payload["target_recipe"],
                "online_solver_calls": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
