#!/usr/bin/env python3
"""Fail-closed A3 GMR ``dof_pos`` -> runtime ``joint_pos`` order contract.

The two column domains intentionally differ.  This module validates their
content-bound tables, validates the complete bijection, checks the two legacy
source-order mirrors, and optionally checks a complete ONNX metadata mapping.
It is a source gate only: passing it does not create or certify schema-2 motion.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_CONTRACT = "configs/a3_joint_order_bijection_v1.json"
_JOINT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_joint$")


class JointOrderContractError(ValueError):
    """The joint-order contract is incomplete, ambiguous, or inconsistent."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JointOrderContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(text: str, label: str):
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise JointOrderContractError(f"invalid {label} JSON: {exc}") from exc


@dataclass(frozen=True)
class JointOrderContract:
    contract_id: str
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    target_from_source_indices: tuple[int, ...]
    source_from_target_indices: tuple[int, ...]
    contract_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping, expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise JointOrderContractError(f"{label} keys changed: missing={missing}, extra={extra}")


def _repo_path(repo_root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise JointOrderContractError(f"{label} must be a non-empty repo-relative path")
    root = repo_root.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise JointOrderContractError(f"{label} escapes repository root: {raw}") from exc
    if not path.is_file():
        raise JointOrderContractError(f"{label} is missing: {raw}")
    return path


def normalize_names(values: Sequence[object], *, expected_count: int, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise JointOrderContractError(f"{label} must be a sequence of joint names, not text")
    names = tuple(str(value) for value in values)
    if len(names) != expected_count:
        raise JointOrderContractError(
            f"{label} length {len(names)} != expected {expected_count}"
        )
    if any(not name or name.strip() != name or not _JOINT_RE.fullmatch(name) for name in names):
        raise JointOrderContractError(f"{label} contains an empty or malformed exact joint name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise JointOrderContractError(f"{label} contains duplicate names: {duplicates}")
    return names


def read_order_file(path: Path, *, expected_count: int, label: str) -> tuple[str, ...]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return normalize_names(names, expected_count=expected_count, label=label)


def validate_bijection(
    source_values: Sequence[object],
    target_values: Sequence[object],
    *,
    expected_count: int = 31,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate two complete named domains and return both exact permutations."""

    source = normalize_names(
        source_values, expected_count=expected_count, label="source_order"
    )
    target = normalize_names(
        target_values, expected_count=expected_count, label="target_order"
    )
    missing = sorted(set(source) - set(target))
    extra = sorted(set(target) - set(source))
    if missing or extra:
        raise JointOrderContractError(
            f"source/target joint sets differ: missing_from_target={missing}, extra_in_target={extra}"
        )
    if source == target:
        raise JointOrderContractError("source and target orders unexpectedly became identical")
    target_from_source = tuple(source.index(name) for name in target)
    source_from_target = tuple(target.index(name) for name in source)
    if sorted(target_from_source) != list(range(expected_count)) or sorted(source_from_target) != list(
        range(expected_count)
    ):
        raise JointOrderContractError("joint-order mapping is not a complete bijection")
    return source, target, target_from_source, source_from_target


def _read_yaml_list(path: Path, field: str) -> list[str]:
    values: list[str] = []
    in_field = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")):
            in_field = stripped == f"{field}:"
            continue
        if in_field:
            if not stripped.startswith("- "):
                raise JointOrderContractError(f"legacy YAML {field} is not a flat list")
            values.append(stripped[2:].strip())
    if not in_field:
        raise JointOrderContractError(f"legacy YAML field is missing: {field}")
    return values


def _read_python_literal_list(path: Path, field: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == field for target in targets):
                matches.append(node.value)
    if len(matches) != 1:
        raise JointOrderContractError(
            f"Python literal mirror {field} must have exactly one top-level assignment"
        )
    try:
        value = ast.literal_eval(matches[0])
    except (ValueError, SyntaxError) as exc:
        raise JointOrderContractError(f"Python mirror {field} is not a literal list") from exc
    if not isinstance(value, list):
        raise JointOrderContractError(f"Python mirror {field} is not a literal list")
    return value


def _parse_csv_names(value: object, *, expected_count: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise JointOrderContractError(f"{label} metadata must be one CSV string")
    # Empty cells are rejected rather than silently removed.
    fields = value.split(",")
    if any(not field.strip() for field in fields):
        raise JointOrderContractError(f"{label} metadata contains an empty CSV field")
    return normalize_names(
        [field.strip() for field in fields], expected_count=expected_count, label=label
    )


def _parse_csv_ints(value: object, *, expected_count: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise JointOrderContractError(f"{label} metadata must be one CSV string")
    fields = value.split(",")
    if len(fields) != expected_count or any(not field.strip() for field in fields):
        raise JointOrderContractError(f"{label} metadata length is not {expected_count}")
    try:
        return tuple(int(field.strip()) for field in fields)
    except ValueError as exc:
        raise JointOrderContractError(f"{label} metadata contains a non-integer") from exc


def validate_runtime_metadata(metadata: Mapping[str, object], contract: JointOrderContract) -> None:
    """Require the complete exporter metadata that proves identity action/articulation order."""

    required = {"joint_names", "articulation_joint_names", "action_joint_ids"}
    missing = sorted(required - set(metadata))
    if missing:
        raise JointOrderContractError(f"runtime metadata is partial; missing={missing}")
    expected_count = len(contract.target_names)
    joint_names = _parse_csv_names(
        metadata["joint_names"], expected_count=expected_count, label="joint_names"
    )
    articulation_names = _parse_csv_names(
        metadata["articulation_joint_names"],
        expected_count=expected_count,
        label="articulation_joint_names",
    )
    action_ids = _parse_csv_ints(
        metadata["action_joint_ids"], expected_count=expected_count, label="action_joint_ids"
    )
    if joint_names != contract.target_names:
        raise JointOrderContractError("joint_names metadata does not equal runtime target order")
    if articulation_names != contract.target_names:
        raise JointOrderContractError(
            "articulation_joint_names metadata does not equal runtime target order"
        )
    if action_ids != tuple(range(expected_count)):
        raise JointOrderContractError("action_joint_ids metadata is not identity 0..30")


def reorder_source_to_target(values, contract: JointOrderContract):
    """Return ``values[..., target]`` from GMR-source columns, rejecting wrong shape/finite."""

    import numpy as np

    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != len(contract.source_names):
        raise JointOrderContractError(
            f"GMR dof_pos last dimension must be {len(contract.source_names)}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise JointOrderContractError("GMR dof_pos must be numeric and finite")
    return array[..., list(contract.target_from_source_indices)]


def load_contract(contract_path: str | Path = DEFAULT_CONTRACT, *, repo_root: str | Path | None = None) -> JointOrderContract:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    path = _repo_path(root, str(contract_path), "joint-order contract")
    raw = _load_json(path.read_text(encoding="utf-8"), "joint-order contract")
    if not isinstance(raw, dict):
        raise JointOrderContractError("joint-order contract root must be an object")
    _exact_keys(
        raw,
        {
            "schema_version", "contract_id", "expected_joint_count", "source_order",
            "target_order", "target_from_source_indices", "source_from_target_indices",
            "legacy_mirrors", "runtime_metadata_contract", "status",
        },
        "joint-order contract",
    )
    if raw["schema_version"] != 1 or raw["contract_id"] != "a3-gmr-dof-pos-to-runtime-articulation-v1":
        raise JointOrderContractError("unsupported joint-order contract identity")
    count = raw["expected_joint_count"]
    if type(count) is not int or count != 31:
        raise JointOrderContractError("expected_joint_count must remain integer 31")

    def load_side(value: object, *, expected_name: str, label: str) -> tuple[str, ...]:
        if not isinstance(value, dict):
            raise JointOrderContractError(f"{label} must be an object")
        _exact_keys(value, {"name", "semantics", "path", "file_sha256", "names_sha256"}, label)
        if value["name"] != expected_name or not isinstance(value["semantics"], str):
            raise JointOrderContractError(f"{label} identity/semantics changed")
        side_path = _repo_path(root, value["path"], f"{label}.path")
        if _sha256(side_path) != value["file_sha256"]:
            raise JointOrderContractError(f"{label} file SHA mismatch")
        names = read_order_file(side_path, expected_count=count, label=label)
        if _names_sha256(names) != value["names_sha256"]:
            raise JointOrderContractError(f"{label} canonical names SHA mismatch")
        return names

    source = load_side(raw["source_order"], expected_name="gmr_dof_pos", label="source_order")
    target = load_side(
        raw["target_order"],
        expected_name="runtime_articulation_joint_pos",
        label="target_order",
    )
    source, target, target_from_source, source_from_target = validate_bijection(
        source, target, expected_count=count
    )
    if list(target_from_source) != raw["target_from_source_indices"]:
        raise JointOrderContractError("target_from_source_indices does not match the named tables")
    if list(source_from_target) != raw["source_from_target_indices"]:
        raise JointOrderContractError("source_from_target_indices does not match the named tables")
    mirrors = raw["legacy_mirrors"]
    if not isinstance(mirrors, list) or len(mirrors) != 3:
        raise JointOrderContractError(
            "legacy_mirrors must bind the GMR YAML/Python and byte-frozen L0 runtime mirrors"
        )
    for index, mirror in enumerate(mirrors):
        if not isinstance(mirror, dict):
            raise JointOrderContractError(f"legacy_mirrors[{index}] must be an object")
        _exact_keys(mirror, {"kind", "path", "field", "must_equal"}, f"legacy_mirrors[{index}]")
        expected_mirror = {
            "gmr_dof_pos": source,
            "runtime_articulation_joint_pos": target,
        }.get(mirror["must_equal"])
        if expected_mirror is None:
            raise JointOrderContractError("legacy mirror names an unknown order domain")
        mirror_path = _repo_path(root, mirror["path"], f"legacy_mirrors[{index}].path")
        if mirror["kind"] == "yaml_list":
            mirror_names = _read_yaml_list(mirror_path, mirror["field"])
        elif mirror["kind"] == "python_literal_list":
            mirror_names = _read_python_literal_list(mirror_path, mirror["field"])
        else:
            raise JointOrderContractError(f"unsupported legacy mirror kind: {mirror['kind']}")
        mirror_normalized = normalize_names(
            mirror_names, expected_count=count, label=f"legacy_mirrors[{index}]"
        )
        if mirror_normalized != expected_mirror:
            raise JointOrderContractError(
                f"legacy_mirrors[{index}] no longer equals {mirror['must_equal']}"
            )

    metadata_contract = raw["runtime_metadata_contract"]
    if metadata_contract != {
        "required_keys": ["joint_names", "articulation_joint_names", "action_joint_ids"],
        "joint_names_must_equal": "runtime_articulation_joint_pos",
        "articulation_joint_names_must_equal": "runtime_articulation_joint_pos",
        "action_joint_ids_must_equal": "identity_0_to_30",
    }:
        raise JointOrderContractError("runtime_metadata_contract changed")
    status = raw["status"]
    required_false = {
        "source_gate_pass_can_authorize_schema2_materialization", "schema2_materialization_run",
        "fk_run", "l0_run", "simulator_run", "training_run", "hardware_run",
    }
    if not isinstance(status, dict) or set(status) != required_false | {"next_gate"}:
        raise JointOrderContractError("status keys changed")
    if any(status[key] is not False for key in required_false) or not isinstance(status["next_gate"], str):
        raise JointOrderContractError("source-only status may not claim a runtime gate")

    return JointOrderContract(
        contract_id=raw["contract_id"],
        source_names=source,
        target_names=target,
        target_from_source_indices=target_from_source,
        source_from_target_indices=source_from_target,
        contract_path=path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--metadata-json",
        help="optional JSON mapping of ONNX custom metadata; all three order keys are required",
    )
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    metadata_checked = False
    metadata_sha256 = None
    metadata_bytes = None
    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        metadata = _load_json(metadata_path.read_text(encoding="utf-8"), "runtime metadata")
        if not isinstance(metadata, dict):
            raise JointOrderContractError("metadata JSON root must be an object")
        validate_runtime_metadata(metadata, contract)
        metadata_checked = True
        metadata_sha256 = _sha256(metadata_path)
        metadata_bytes = metadata_path.stat().st_size
    print(
        json.dumps(
            {
                "contract_id": contract.contract_id,
                "joint_count": len(contract.source_names),
                "contract_sha256": _sha256(contract.contract_path),
                "source_equals_target": contract.source_names == contract.target_names,
                "bijection_valid": True,
                "runtime_metadata_checked": metadata_checked,
                "runtime_metadata_sha256": metadata_sha256,
                "runtime_metadata_bytes": metadata_bytes,
                "schema2_materialization_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
