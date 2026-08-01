#!/usr/bin/env python3
"""Fail-closed offline audit of one ActionBall training run's Reward evidence.

This tool validates artifacts that a real run is expected to emit.  It does not
run Isaac and never upgrades host fixtures or synthetic JSON into Isaac runtime
evidence.  A PASS means only that the supplied run artifacts are internally
consistent and contain the fields needed to audit:

* the composed effective Reward recipe and its SHA-256;
* per-update RewardManager activation closure;
* the same activation totals split by frozen action identity;
* ActionBall curriculum/outcome accounting;
* the joint-safety update sidecar receipts; and
* event-bound soft/hard/table/fall/death evidence proving that generic death is
  charged exactly once and no reason-specific terminal penalty is stacked.

The current runtime may not emit every record above yet.  That is intentional:
missing evidence is reported as ``FAIL_CLOSED`` rather than inferred from a
TensorBoard mean, a unit test, or a zero counter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path


_REWARD_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils"
    / "effective_reward_recipe.py"
)
_REWARD_TAXONOMY_SPEC = importlib.util.spec_from_file_location(
    "hope_reward_taxonomy_for_audit", _REWARD_TAXONOMY_PATH
)
if (
    _REWARD_TAXONOMY_SPEC is None
    or _REWARD_TAXONOMY_SPEC.loader is None
):
    raise RuntimeError("cannot load authoritative Reward taxonomy")
REWARD_TAXONOMY = importlib.util.module_from_spec(_REWARD_TAXONOMY_SPEC)
_REWARD_TAXONOMY_SPEC.loader.exec_module(REWARD_TAXONOMY)


RECIPE_SCHEMA_VERSION = 1
ACTIVATION_SCHEMA_VERSION = 1
PER_ACTION_SCHEMA_VERSION = 2
SAFETY_TRANSITION_SCHEMA_VERSION = 2
JOINT_SAFETY_SCHEMA_VERSION = 2
ACTION_BALL_LEDGER_SCHEMA_VERSION = 1
EPISODE_SEGMENTED_CLOSURE_SCHEMA_VERSION = 1

ACTIVATION_EVENT = "hope_effective_reward_activation_update"
PER_ACTION_EVENT = "hope_effective_reward_activation_by_action_update"
SAFETY_TRANSITION_EVENT = "hope_reward_safety_transition_update"
EPISODE_SEGMENTED_CLOSURE_EVENT = (
    "hope_reward_episode_segmented_closure_update"
)
JOINT_SAFETY_EVENT = "hope_joint_safety_update"
ACTION_BALL_LEDGER_EVENT = "action_ball_training_ledger"
EXACT_BEHAVIOR_EVENT = "hope_exact_behavior_update"

EVENT_PREFIXES = {
    "HOPE_EFFECTIVE_REWARD_ACTIVATION_UPDATE_JSON=": ACTIVATION_EVENT,
    "HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON=": PER_ACTION_EVENT,
    "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=": SAFETY_TRANSITION_EVENT,
    "HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_UPDATE_JSON=": (
        EPISODE_SEGMENTED_CLOSURE_EVENT
    ),
    "HOPE_JOINT_SAFETY_UPDATE_JSON=": JOINT_SAFETY_EVENT,
    "HOPE_EXACT_BEHAVIOR_UPDATE_JSON=": EXACT_BEHAVIOR_EVENT,
}

SHA256_HEX = frozenset("0123456789abcdef")
LEDGER_COUNTERS = (
    "P",
    "A",
    "I",
    "S",
    "C",
    "L",
    "F",
    "U_table",
    "U_fall",
    "U_collision",
    "U_joint_qdes",
    "U_joint_actual",
    "X",
)
JOINT_COUNTERS = (
    "qdes_joint_count",
    "policy_crossing_joint_count",
    "substep_hard_crossing_joint_count",
    "actual_hard_edge_joint_count",
)
TERMINATION_REASON_CLASSES = frozenset(
    (
        "hard_limit",
        "table_hit",
        "fall",
        "reference_envelope",
        "other_termination",
    )
)
SOFT_LIMIT_TERM_NAMES = ("joint_limit", "qdes_limit_barrier")
SOFT_LIMIT_CALLABLES = {
    "joint_limit": "actual_joint_limit_barrier_v2",
    "qdes_limit_barrier": "qdes_limit_barrier_v2",
}
ADOPTED_POLICY_DT_S = 0.02
ADOPTED_DEATH_WEIGHT = -300.0
ADOPTED_DEATH_PER_EVENT = -6.0
ADOPTED_SOFT_LIMIT_WEIGHT = -5.0
ADOPTED_SOFT_LIMIT_MARGIN_FRAC = 0.08
ADOPTED_SOFT_LIMIT_PENALTY_FLOOR = 0.25
ADOPTED_SOFT_LIMIT_MAX_JOINTS = 31
REQUIRED_TERMINATION_TERM_ORDER = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
HARD_SAFETY_TERMINATION_TERMS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
REFERENCE_ENVELOPE_TERMINATION_TERMS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
REWARD_GROUP_FIELDS = frozenset(
    (
        "group",
        "objective_term_names",
        "diagnostic_probe_term_names",
        "eligibility",
        "eligible_sample_count",
        "nonzero_sample_count",
        "weighted_sum",
        "weighted_p5",
        "weighted_p50",
        "weighted_p95",
        "positive_weighted_sum",
        "negative_weighted_sum",
        "positive_return_fraction",
        "negative_return_fraction",
    )
)


class AuditInputError(ValueError):
    """An input file cannot be read without guessing its semantics."""


class Audit:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.checks = {}

    def fail(self, code, message):
        self.failures.append({"code": str(code), "message": str(message)})

    def warn(self, code, message):
        self.warnings.append({"code": str(code), "message": str(message)})

    def check(self, name, passed, detail):
        self.checks[str(name)] = {
            "status": "PASS" if passed else "FAIL_CLOSED",
            "detail": str(detail),
        }
        if not passed:
            self.fail(name, detail)


def _canonical_json(value):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AuditInputError("value is not canonical finite JSON") from exc


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe_term_sha256(term):
    return _sha256_bytes(_canonical_json(term).encode("utf-8"))


def _is_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in SHA256_HEX for character in value)
    )


def _finite_number(value):
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _close(left, right, *, rel_tol=1.0e-5, abs_tol=1.0e-7):
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol
    )


def _guarded_call(audit, check_name, function, *args, **kwargs):
    """Turn malformed evidence into a structured fail-closed result.

    The verifier is itself part of the evidence boundary.  A bad nested shape
    must not escape as a traceback that leaves the caller guessing whether the
    gate passed.
    """

    try:
        return function(*args, **kwargs)
    except Exception as exc:
        audit.check(
            check_name,
            False,
            "malformed evidence rejected by {}: {}: {}".format(
                function.__name__, type(exc).__name__, exc
            ),
        )
        return None


def _torch_sidecar_loader(path):
    """Safely decode a torch sidecar without permitting arbitrary pickle code."""

    try:
        import torch
    except ImportError as exc:
        raise AuditInputError(
            "torch is required to decode joint-safety sidecar contents"
        ) from exc
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise AuditInputError(
            "installed torch lacks the required weights_only sidecar decoder"
        ) from exc
    except Exception as exc:
        raise AuditInputError(
            "joint-safety sidecar is not safely decodable: {}".format(path)
        ) from exc


def _plain_value(value):
    """Convert a CPU/tensor-like value to built-in containers for inspection."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise AuditInputError("tensor-like sidecar value has no tolist()")
        value = tolist()
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    return value


def _tensor_metadata(value, *, label):
    """Return the exact dtype/shape/bytes used by runner-side hashes."""

    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise AuditInputError("{} must be a decoded tensor".format(label))
    tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    contiguous = getattr(tensor, "contiguous", None)
    if callable(contiguous):
        tensor = contiguous()
    dtype = str(getattr(tensor, "dtype", ""))
    shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
    numpy = getattr(tensor, "numpy", None)
    if not dtype or not callable(numpy):
        raise AuditInputError("{} lacks tensor dtype/bytes".format(label))
    array = numpy()
    tobytes = getattr(array, "tobytes", None)
    if not callable(tobytes):
        raise AuditInputError("{} cannot expose contiguous bytes".format(label))
    return dtype, shape, tobytes(order="C")


def _int64_vector_bytes(values):
    if any(type(value) is not int for value in values):
        raise AuditInputError("identity hash input contains a non-integer")
    return struct.pack("<{}q".format(len(values)), *values)


def _runner_identity_sha256(row, *, action_ball_enabled):
    digest = hashlib.sha256()
    digest.update(b"action_ball=1" if action_ball_enabled else b"action_ball=0")
    for field in (
        "action_episode_sequence",
        "episode_length",
        "action_uid",
        "birth_generation",
        "swing_generation",
    ):
        values = row[field]
        digest.update(field.encode("utf-8"))
        digest.update(b"torch.int64")
        digest.update(json.dumps([len(values)]).encode("ascii"))
        digest.update(_int64_vector_bytes(values))
    digest.update(
        json.dumps(
            list(row["birth_receipt_sha256"]),
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _joint_payload_bytes(value):
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach()
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if not callable(numel) or not callable(element_size):
            raise AuditInputError("tensor lacks numel/element_size budget metadata")
        return int(numel()) * int(element_size())
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, dict):
        return sum(_joint_payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_joint_payload_bytes(item) for item in value)
    return 0


def _plain_vector(value, *, length, label, integer=False, minimum=0):
    value = _plain_value(value)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AuditInputError(
            "{} must contain exactly {} values".format(label, length)
        )
    result = list(value)
    if integer:
        if any(
            type(item) is not int or item < minimum
            for item in result
        ):
            raise AuditInputError(
                "{} must contain plain integers >= {}".format(label, minimum)
            )
    elif any(not _finite_number(item) for item in result):
        raise AuditInputError("{} must contain finite numbers".format(label))
    return result


def _plain_matrix(
    value, *, rows, columns, label, integer=False, minimum=0
):
    value = _plain_value(value)
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise AuditInputError(
            "{} must contain exactly {} rows".format(label, rows)
        )
    return [
        _plain_vector(
            row,
            length=columns,
            label="{}[{}]".format(label, index),
            integer=integer,
            minimum=minimum,
        )
        for index, row in enumerate(value)
    ]


def _plain_bool_vector(value, *, length, label):
    value = _plain_value(value)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(type(item) is not bool for item in value)
    ):
        raise AuditInputError(
            "{} must contain exactly {} booleans".format(label, length)
        )
    return list(value)


def _plain_bool_matrix(value, *, rows, columns, label):
    value = _plain_value(value)
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise AuditInputError(
            "{} must contain exactly {} boolean rows".format(label, rows)
        )
    return [
        _plain_bool_vector(
            row,
            length=columns,
            label="{}[{}]".format(label, index),
        )
        for index, row in enumerate(value)
    ]


def _sparse_counter_map(value):
    indices = _plain_value(value["index"])
    values = _plain_value(value["value"])
    return {
        (int(coordinates[0]), int(coordinates[1])): int(count)
        for coordinates, count in zip(indices, values)
    }


def _validate_sparse_counter(
    value, *, num_envs, joint_count, label
):
    if not isinstance(value, dict) or set(value) != {
        "index",
        "value",
        "nonzero_cells",
        "event_count",
    }:
        raise AuditInputError("{} has an invalid sparse-counter schema".format(label))
    indices = _plain_value(value["index"])
    values = _plain_value(value["value"])
    if not isinstance(indices, (list, tuple)) or not isinstance(
        values, (list, tuple)
    ):
        raise AuditInputError("{} sparse index/value must be sequences".format(label))
    if (
        not _nonnegative_int(value["nonzero_cells"])
        or not _nonnegative_int(value["event_count"])
        or len(indices) != value["nonzero_cells"]
        or len(values) != value["nonzero_cells"]
    ):
        raise AuditInputError("{} sparse counts do not match payload".format(label))
    seen = set()
    for index, coordinates in enumerate(indices):
        if (
            not isinstance(coordinates, (list, tuple))
            or len(coordinates) != 2
            or not _nonnegative_int(coordinates[0])
            or not _nonnegative_int(coordinates[1])
            or coordinates[0] >= num_envs
            or coordinates[1] >= joint_count
            or tuple(coordinates) in seen
        ):
            raise AuditInputError(
                "{} sparse coordinate {} is invalid/duplicate".format(label, index)
            )
        seen.add(tuple(coordinates))
    if any(not _nonnegative_int(item) for item in values):
        raise AuditInputError("{} sparse values must be nonnegative integers".format(label))
    if sum(values) != value["event_count"]:
        raise AuditInputError("{} sparse event_count does not close".format(label))
    return int(value["event_count"])


def _load_json_object(path, label):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditInputError("{} is not readable JSON: {}".format(label, path)) from exc
    if not isinstance(value, dict):
        raise AuditInputError("{} must contain one JSON object".format(label))
    _canonical_json(value)
    return value


def _event_lines(paths, audit):
    events = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError as exc:
            audit.fail("event_file_unreadable", "{}: {}".format(path, exc))
            continue
        with stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = None
                expected_event = None
                for prefix, event_name in EVENT_PREFIXES.items():
                    if line.startswith(prefix):
                        payload = line[len(prefix) :]
                        expected_event = event_name
                        break
                if payload is None:
                    if not line.startswith("{"):
                        continue
                    payload = line
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError as exc:
                    audit.fail(
                        "event_json_invalid",
                        "{}:{} contains an invalid JSON event: {}".format(
                            path, line_number, exc
                        ),
                    )
                    continue
                if not isinstance(record, dict):
                    audit.fail(
                        "event_not_object",
                        "{}:{} event must be a JSON object".format(path, line_number),
                    )
                    continue
                try:
                    _canonical_json(record)
                except AuditInputError as exc:
                    audit.fail(
                        "event_not_finite",
                        "{}:{} {}".format(path, line_number, exc),
                    )
                    continue
                if expected_event is not None and record.get("event") != expected_event:
                    audit.fail(
                        "event_prefix_mismatch",
                        "{}:{} prefix claims {!r}, payload claims {!r}".format(
                            path,
                            line_number,
                            expected_event,
                            record.get("event"),
                        ),
                    )
                    continue
                record = dict(record)
                record["_audit_source_path"] = str(path)
                record["_audit_source_line"] = line_number
                events.append(record)
    return events


def _load_recipe(path, audit):
    try:
        receipt = _load_json_object(path, "effective Reward recipe")
    except AuditInputError as exc:
        audit.check("recipe_integrity", False, exc)
        return None
    if set(receipt) != {"schema_version", "terms", "sha256"}:
        audit.check(
            "recipe_integrity",
            False,
            "recipe must contain exactly schema_version, terms and sha256",
        )
        return None
    if receipt.get("schema_version") != RECIPE_SCHEMA_VERSION:
        audit.check(
            "recipe_integrity",
            False,
            "unsupported recipe schema_version={!r}".format(
                receipt.get("schema_version")
            ),
        )
        return None
    terms = receipt.get("terms")
    if not isinstance(terms, list) or not terms:
        audit.check("recipe_integrity", False, "recipe terms must be a non-empty list")
        return None
    normalized = {}
    names = []
    for index, term in enumerate(terms):
        if not isinstance(term, dict) or set(term) != {
            "name",
            "callable",
            "weight",
            "params",
        }:
            audit.check(
                "recipe_integrity",
                False,
                "recipe term {} has an invalid field set".format(index),
            )
            return None
        name = term.get("name")
        callable_name = term.get("callable")
        weight = term.get("weight")
        if (
            type(name) is not str
            or not name
            or name.strip() != name
            or type(callable_name) is not str
            or not callable_name
            or not _finite_number(weight)
            or float(weight) == 0.0
        ):
            audit.check(
                "recipe_integrity",
                False,
                "recipe term {} has invalid name/callable/non-zero finite weight".format(
                    index
                ),
            )
            return None
        if name in normalized:
            audit.check(
                "recipe_integrity",
                False,
                "recipe has duplicate term {!r}".format(name),
            )
            return None
        name_probe = name.endswith("_probe")
        callable_probe = callable_name.rsplit(".", 1)[-1].endswith("_probe")
        if name_probe != callable_probe:
            audit.check(
                "recipe_integrity",
                False,
                "recipe term {!r} has ambiguous diagnostic-probe identity".format(name),
            )
            return None
        _canonical_json(term.get("params"))
        normalized[name] = term
        names.append(name)
    if names != sorted(names):
        audit.check(
            "recipe_integrity",
            False,
            "recipe term list is not sorted by name",
        )
        return None
    payload = {
        "schema_version": receipt["schema_version"],
        "terms": receipt["terms"],
    }
    digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    if not _is_sha256(receipt.get("sha256")) or receipt["sha256"] != digest:
        audit.check(
            "recipe_integrity",
            False,
            "recipe SHA-256 does not match its canonical payload",
        )
        return None
    audit.check(
        "recipe_integrity",
        True,
        "{} active terms; canonical sha256={}".format(len(terms), digest),
    )
    return {
        "receipt": receipt,
        "terms": normalized,
        "sha256": digest,
    }


def _load_manifest(path, audit):
    if path is None:
        audit.check(
            "manifest_action_identity",
            False,
            "an exact action manifest is required for per-action completeness",
        )
        return None
    try:
        manifest = _load_json_object(path, "action manifest")
    except AuditInputError as exc:
        audit.check("manifest_action_identity", False, exc)
        return None
    actions = manifest.get("actions")
    order = manifest.get("action_order")
    if (
        not isinstance(actions, list)
        or not actions
        or not isinstance(order, list)
        or len(order) != len(actions)
    ):
        audit.check(
            "manifest_action_identity",
            False,
            "manifest needs non-empty actions and equal-length action_order",
        )
        return None
    rows = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            audit.check(
                "manifest_action_identity",
                False,
                "manifest action {} is not an object".format(index),
            )
            return None
        action_id = action.get("action_id")
        action_uid = action.get("action_uid")
        if (
            type(action_id) is not str
            or not action_id
            or type(action_uid) is not int
            or action_uid < 0
            or order[index] != action_id
            or action_id in rows
        ):
            audit.check(
                "manifest_action_identity",
                False,
                "manifest action {0} has invalid id/uid/order binding".format(index),
            )
            return None
        rows[action_id] = action_uid
    if len(set(rows.values())) != len(rows):
        audit.check(
            "manifest_action_identity",
            False,
            "manifest action_uid values are not unique",
        )
        return None
    digest = _sha256_file(path)
    audit.check(
        "manifest_action_identity",
        True,
        "{} ordered actions; file sha256={}".format(len(order), digest),
    )
    return {
        "document": manifest,
        "order": tuple(order),
        "uids": rows,
        "file_sha256": digest,
    }


def _validate_run_recipe_binding(recipe_path, run_dir, recipe, audit):
    if run_dir is None:
        audit.check(
            "run_recipe_binding",
            False,
            "--run-dir is required to bind the recipe to its training contract",
        )
        return
    root = Path(run_dir).resolve()
    expected_recipe = root / "params" / "effective_reward_recipe.json"
    supplied_recipe = Path(recipe_path)
    actual_recipe = supplied_recipe.parent.resolve() / supplied_recipe.name
    if (
        actual_recipe != expected_recipe
        or not actual_recipe.is_file()
        or actual_recipe.is_symlink()
    ):
        audit.check(
            "run_recipe_binding",
            False,
            "recipe must be the regular run-local params/effective_reward_recipe.json",
        )
        return
    contract_path = root / "params" / "training_contract.json"
    try:
        contract = _load_json_object(contract_path, "training contract")
    except AuditInputError as exc:
        audit.check("run_recipe_binding", False, exc)
        return
    if contract.get("effective_reward_recipe") != recipe["receipt"]:
        audit.check(
            "run_recipe_binding",
            False,
            "training_contract.json does not embed the exact effective Reward receipt",
        )
        return
    audit.check(
        "run_recipe_binding",
        True,
        "run-local recipe equals the receipt embedded in training_contract.json",
    )


def _group_event(events, event_name):
    return [record for record in events if record.get("event") == event_name]


def _validate_update_sequence(records, *, field, label, audit):
    values = []
    seen = set()
    for record in records:
        value = record.get(field)
        if not _nonnegative_int(value):
            audit.fail(
                "{}_update_invalid".format(label),
                "{} has invalid {}={!r}".format(label, field, value),
            )
            continue
        if value in seen:
            audit.fail(
                "{}_update_duplicate".format(label),
                "{} repeats update {}".format(label, value),
            )
        seen.add(value)
        values.append(value)
    ordered = sorted(values)
    if values != ordered:
        audit.fail(
            "{}_update_order".format(label),
            "{} records are not in increasing update order".format(label),
        )
    if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
        audit.fail(
            "{}_update_gap".format(label),
            "{} update sequence is not contiguous: {}".format(label, ordered),
        )
    return tuple(ordered)


def _validate_activation(events, recipe, audit):
    records = _group_event(events, ACTIVATION_EVENT)
    if not records:
        audit.check(
            "runtime_activation_integrity",
            False,
            "no {} records were supplied".format(ACTIVATION_EVENT),
        )
        return None
    before = len(audit.failures)
    updates = _validate_update_sequence(
        records, field="ppo_update", label="reward_activation", audit=audit
    )
    by_update = {}
    step_dt = None
    recipe_terms = recipe["terms"]
    expected_names = tuple(sorted(recipe_terms))
    expected_objectives = tuple(
        name for name in expected_names if not name.endswith("_probe")
    )
    expected_probes = tuple(
        name for name in expected_names if name.endswith("_probe")
    )
    totals = {
        name: {
            "observed_sample_count": 0,
            "nonzero_sample_count": 0,
            "raw_sum": 0.0,
            "weighted_sum": 0.0,
        }
        for name in expected_names
    }
    for record in records:
        update = record.get("ppo_update")
        if record.get("schema_version") != ACTIVATION_SCHEMA_VERSION:
            audit.fail(
                "activation_schema",
                "activation update {} has unsupported schema".format(update),
            )
            continue
        if record.get("recipe_sha256") != recipe["sha256"]:
            audit.fail(
                "activation_recipe_sha256",
                "activation update {} is not bound to the exact recipe SHA".format(
                    update
                ),
            )
        if record.get("task_kind") != "action_ball":
            audit.fail(
                "activation_task_kind",
                "activation update {} is not task_kind=action_ball".format(update),
            )
        dt = record.get("step_dt_s")
        if not _finite_number(dt) or float(dt) <= 0.0:
            audit.fail(
                "activation_dt",
                "activation update {} has invalid step_dt_s".format(update),
            )
            continue
        dt = float(dt)
        if step_dt is None:
            step_dt = dt
        elif dt != step_dt:
            audit.fail(
                "activation_dt_drift",
                "activation step_dt_s changed from {} to {}".format(step_dt, dt),
            )
        environment_steps = record.get("environment_step_count")
        expected_steps = record.get("expected_environment_step_count")
        num_envs = record.get("num_envs")
        observed = record.get("observed_sample_count")
        if (
            not _nonnegative_int(environment_steps)
            or environment_steps <= 0
            or environment_steps != expected_steps
            or not _nonnegative_int(num_envs)
            or num_envs <= 0
            or observed != environment_steps * num_envs
        ):
            audit.fail(
                "activation_sample_conservation",
                "activation update {} has inconsistent step/env/sample counts".format(
                    update
                ),
            )
        common_start = record.get("common_step_counter_start")
        common_end = record.get("common_step_counter_end")
        if (
            not _nonnegative_int(common_start)
            or not _nonnegative_int(common_end)
            or not _nonnegative_int(environment_steps)
            or common_end != common_start + environment_steps
        ):
            audit.fail(
                "activation_common_step_counter",
                "activation update {} common_step_counter does not close".format(
                    update
                ),
            )
        contract = record.get("reward_cache_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("source")
            != "isaaclab_reward_manager_private_step_cache"
            or contract.get("step_cache_semantics") != "raw_times_weight"
            or contract.get("weighted_semantics")
            != "raw_times_weight_times_step_dt"
            or contract.get("total_reward_closure") != "validated"
            or not _finite_number(contract.get("max_abs_error"))
        ):
            audit.fail(
                "activation_cache_contract",
                "activation update {} lacks the reviewed RewardManager cache contract".format(
                    update
                ),
            )
        if tuple(record.get("objective_term_names", ())) != expected_objectives:
            audit.fail(
                "activation_objective_names",
                "activation update {} objective term list differs from recipe".format(
                    update
                ),
            )
        if tuple(record.get("diagnostic_probe_term_names", ())) != expected_probes:
            audit.fail(
                "activation_probe_names",
                "activation update {} diagnostic probe list differs from recipe".format(
                    update
                ),
            )
        raw_terms = record.get("terms")
        if not isinstance(raw_terms, list):
            audit.fail(
                "activation_terms_missing",
                "activation update {} terms is not a list".format(update),
            )
            continue
        term_rows = {}
        for row in raw_terms:
            if not isinstance(row, dict) or type(row.get("name")) is not str:
                audit.fail(
                    "activation_term_invalid",
                    "activation update {} contains a malformed term row".format(
                        update
                    ),
                )
                continue
            if row["name"] in term_rows:
                audit.fail(
                    "activation_term_duplicate",
                    "activation update {} duplicates term {!r}".format(
                        update, row["name"]
                    ),
                )
            term_rows[row["name"]] = row
        if tuple(sorted(term_rows)) != expected_names:
            audit.fail(
                "activation_term_set",
                "activation update {} term names differ from recipe".format(update),
            )
            continue
        weighted_total = 0.0
        for name in expected_names:
            row = term_rows[name]
            configured = recipe_terms[name]
            expected_role = "diagnostic_probe" if name.endswith("_probe") else "objective"
            if (
                row.get("callable") != configured["callable"]
                or not _close(row.get("weight"), configured["weight"], rel_tol=0.0)
                or row.get("role") != expected_role
                or row.get("recipe_term_sha256")
                != _recipe_term_sha256(configured)
            ):
                audit.fail(
                    "activation_recipe_binding",
                    "activation update {} term {!r} callable/weight/role drift".format(
                        update, name
                    ),
                )
            term_observed = row.get("observed_sample_count")
            nonzero = row.get("nonzero_sample_count")
            raw_sum = row.get("raw_sum")
            weighted_sum = row.get("weighted_sum")
            if (
                row.get("observed_environment_step_count") != environment_steps
                or term_observed != observed
                or not _nonnegative_int(term_observed)
                or not _nonnegative_int(nonzero)
                or nonzero > observed
                or not _finite_number(raw_sum)
                or not _finite_number(weighted_sum)
                or row.get("raw_recovery")
                != "validated_weighted_eq_raw_times_weight_times_step_dt"
                or not _finite_number(row.get("raw_recomposition_max_abs_error"))
                or row.get("eligibility") != "unknown"
            ):
                audit.fail(
                    "activation_term_fields",
                    "activation update {} term {!r} has incomplete/inconsistent fields".format(
                        update, name
                    ),
                )
                continue
            expected_weighted = float(raw_sum) * float(configured["weight"]) * dt
            if not _close(weighted_sum, expected_weighted):
                audit.fail(
                    "activation_raw_weight_dt",
                    "activation update {} term {!r} violates raw*weight*dt: {} != {}".format(
                        update, name, weighted_sum, expected_weighted
                    ),
                )
            if expected_role == "diagnostic_probe" and (
                nonzero != 0
                or not _close(raw_sum, 0.0)
                or not _close(weighted_sum, 0.0)
            ):
                audit.fail(
                    "activation_probe_nonzero",
                    "activation diagnostic probe {!r} contributed to Reward".format(name),
                )
            weighted_total += float(weighted_sum)
            totals[name]["observed_sample_count"] += int(term_observed)
            totals[name]["nonzero_sample_count"] += int(nonzero)
            totals[name]["raw_sum"] += float(raw_sum)
            totals[name]["weighted_sum"] += float(weighted_sum)
        if not _close(record.get("total_weighted_reward_sum"), weighted_total):
            audit.fail(
                "activation_total_closure",
                "activation update {} total does not equal sum(term weighted_sum)".format(
                    update
                ),
            )
        by_update[update] = {
            "record": record,
            "terms": term_rows,
        }
    for name in expected_objectives:
        if totals[name]["nonzero_sample_count"] <= 0:
            audit.fail(
                "objective_activation_unproven",
                "active objective term {!r} never produced a nonzero weighted "
                "sample; composed presence alone is not causal activation proof".format(
                    name
                ),
            )
    passed = len(audit.failures) == before and len(by_update) == len(records)
    audit.check(
        "runtime_activation_integrity",
        passed,
        "{} contiguous update(s), dt={} s; runtime cache claims checked".format(
            len(updates), step_dt
        )
        if passed
        else "one or more runtime activation records failed conservation/binding",
    )
    return {
        "records": by_update,
        "updates": updates,
        "step_dt_s": step_dt,
        "totals": totals,
    }


def _validate_episode_segmented_closure(
    events, recipe, activation, audit
):
    records = _group_event(events, EPISODE_SEGMENTED_CLOSURE_EVENT)
    if not records:
        audit.check(
            "episode_segmented_reward_closure",
            False,
            "no {} PASS receipts were supplied".format(
                EPISODE_SEGMENTED_CLOSURE_EVENT
            ),
        )
        return None
    before = len(audit.failures)
    updates = _validate_update_sequence(
        records,
        field="ppo_update",
        label="episode_segmented_reward_closure",
        audit=audit,
    )
    if activation is None:
        audit.check(
            "episode_segmented_reward_closure",
            False,
            "runtime activation is unavailable; closure cannot be bound",
        )
        return None
    if updates != activation["updates"]:
        audit.fail(
            "episode_closure_update_coverage",
            "closure updates {} do not exactly match activation updates {}".format(
                list(updates), list(activation["updates"])
            ),
        )
    recipe_names = set(recipe["terms"])
    seen_segment_keys = set()
    live_e2_updates = []
    all_sources_live = True
    by_update = {}
    required_check_names = (
        "all_step_reward_buf_equals_all_term_sums",
        "all_episode_sums_equal_captured_term_sums",
        "all_reset_episode_sums_cleared",
        "exact_environment_step_coverage",
    )
    numeric_check_names = (
        "environment_step_count",
        "reset_batch_count",
        "completed_episode_count",
        "manager_episode_sum_comparison_count",
        "dashboard_term_comparison_count",
        "reward_buf_term_sum_comparison_count",
        "manager_clear_comparison_count",
    )
    error_check_names = (
        "max_abs_manager_episode_sum_error",
        "max_abs_dashboard_normalization_error",
        "max_abs_reward_buf_vs_term_sum_error",
        "max_abs_manager_clear_error",
    )
    for record in records:
        update = record.get("ppo_update")
        activation_row = activation["records"].get(update)
        if activation_row is None:
            audit.fail(
                "episode_closure_activation_binding",
                "closure update {} has no activation record".format(update),
            )
            continue
        activation_record = activation_row["record"]
        if (
            record.get("schema_version")
            != EPISODE_SEGMENTED_CLOSURE_SCHEMA_VERSION
            or record.get("status") != "PASS"
            or record.get("task_kind") != "action_ball"
            or record.get("capture_mode")
            != "reward_manager_reset_pre_clear_hook"
        ):
            audit.fail(
                "episode_closure_status",
                "closure update {} is not a reset-hook PASS receipt".format(
                    update
                ),
            )
        source = record.get("evidence_source")
        if source != "live_isaac_reward_manager":
            all_sources_live = False
            if source != "synthetic_test_fixture":
                audit.fail(
                    "episode_closure_source",
                    "closure update {} has unknown evidence_source {!r}".format(
                        update, source
                    ),
                )
        if (
            record.get("recipe_sha256") != recipe["sha256"]
            or record.get("step_dt_s") != activation_record.get("step_dt_s")
            or record.get("num_envs") != activation_record.get("num_envs")
        ):
            audit.fail(
                "episode_closure_recipe_activation_binding",
                "closure update {} is not bound to the exact recipe/dt/env count".format(
                    update
                ),
            )
        num_envs = record.get("num_envs")
        max_episode_length_s = record.get("max_episode_length_s")
        term_names = record.get("all_reward_manager_term_names")
        if (
            not _nonnegative_int(num_envs)
            or num_envs <= 0
            or not _finite_number(max_episode_length_s)
            or float(max_episode_length_s) <= 0.0
            or not isinstance(term_names, list)
            or not term_names
            or any(type(name) is not str or not name for name in term_names)
            or len(term_names) != len(set(term_names))
            or not recipe_names.issubset(set(term_names))
            or record.get("segment_key_fields")
            != ["env_id", "reset_generation"]
        ):
            audit.fail(
                "episode_closure_contract",
                "closure update {} has an invalid env/term/segment contract".format(
                    update
                ),
            )
            continue
        checks = record.get("checks")
        if (
            not isinstance(checks, dict)
            or checks.get("status") != "PASS"
            or any(
                checks.get(name) != "PASS"
                for name in required_check_names
            )
            or any(
                not _nonnegative_int(checks.get(name))
                for name in numeric_check_names
            )
            or any(
                not _finite_number(checks.get(name))
                or float(checks.get(name)) < 0.0
                for name in error_check_names
            )
            or checks.get("environment_step_count")
            != activation_record.get("environment_step_count")
        ):
            audit.fail(
                "episode_closure_checks",
                "closure update {} lacks finite PASS conservation checks".format(
                    update
                ),
            )
        completed = record.get("completed_episode_segments")
        open_segments = record.get("open_episode_segments")
        reset_batches = record.get("reset_batches")
        if (
            not isinstance(completed, list)
            or record.get("completed_episode_count") != len(completed)
            or not isinstance(open_segments, list)
            or record.get("open_episode_count") != len(open_segments)
            or len(open_segments) != num_envs
            or not isinstance(reset_batches, list)
        ):
            audit.fail(
                "episode_closure_rows",
                "closure update {} has inconsistent segment/reset row counts".format(
                    update
                ),
            )
            continue
        eligible_completed = False
        for kind, rows in (
            ("completed", completed),
            ("open", open_segments),
        ):
            update_keys = set()
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    audit.fail(
                        "episode_closure_segment_invalid",
                        "{} segment {} in update {} is not an object".format(
                            kind, index, update
                        ),
                    )
                    continue
                env_id = row.get("env_id")
                generation = row.get("reset_generation")
                step_count = row.get("step_count")
                key = (env_id, generation)
                local = row.get("local_term_sums")
                if (
                    not _nonnegative_int(env_id)
                    or env_id >= num_envs
                    or not _nonnegative_int(generation)
                    or not _nonnegative_int(step_count)
                    or key in update_keys
                    or not isinstance(local, list)
                    or len(local) != len(term_names)
                    or any(not _finite_number(value) for value in local)
                ):
                    audit.fail(
                        "episode_closure_segment_contract",
                        "{} segment {} in update {} is malformed/duplicate".format(
                            kind, index, update
                        ),
                    )
                    continue
                update_keys.add(key)
                local_total = sum(float(value) for value in local)
                if (
                    not _close(row.get("all_term_sum"), local_total)
                    or not _close(
                        row.get("reward_buf_sum"), local_total
                    )
                    or not _finite_number(
                        row.get("reward_buf_vs_all_terms_abs_error")
                    )
                ):
                    audit.fail(
                        "episode_closure_reward_buf_terms",
                        "{} segment {} in update {} does not close reward_buf/all terms".format(
                            kind, index, update
                        ),
                    )
                if kind == "completed":
                    manager = row.get("reward_manager_episode_sums")
                    if (
                        row.get("segment_key") != [env_id, generation]
                        or key in seen_segment_keys
                        or not isinstance(manager, list)
                        or len(manager) != len(term_names)
                        or any(
                            not _finite_number(value) for value in manager
                        )
                        or any(
                            not _close(left, right)
                            for left, right in zip(local, manager)
                        )
                    ):
                        audit.fail(
                            "episode_closure_manager_sums",
                            "completed segment {} in update {} does not bind "
                            "the unique key/local sums/_episode_sums".format(
                                index, update
                            ),
                        )
                    seen_segment_keys.add(key)
                    if (
                        step_count > 0
                        and row.get("administrative_reset") is False
                    ):
                        eligible_completed = True
                elif row.get("status") != "OPEN_NOT_E2":
                    audit.fail(
                        "episode_closure_open_status",
                        "open segment {} in update {} is not explicitly non-E2".format(
                            index, update
                        ),
                    )
        for batch_index, batch in enumerate(reset_batches):
            if not isinstance(batch, dict):
                audit.fail(
                    "episode_closure_dashboard_batch",
                    "dashboard batch {} in update {} is not an object".format(
                        batch_index, update
                    ),
                )
                continue
            env_ids = batch.get("env_ids")
            generations = batch.get("reset_generations")
            rows = batch.get("terms")
            if (
                batch.get("status") != "PASS"
                or batch.get("normalization_divisor_s")
                != max_episode_length_s
                or not isinstance(env_ids, list)
                or not env_ids
                or len(env_ids) != len(set(env_ids))
                or any(
                    not _nonnegative_int(env_id) or env_id >= num_envs
                    for env_id in env_ids
                )
                or not isinstance(generations, list)
                or len(generations) != len(env_ids)
                or any(
                    not _nonnegative_int(value) for value in generations
                )
                or not isinstance(rows, list)
                or [row.get("name") for row in rows if isinstance(row, dict)]
                != term_names
            ):
                audit.fail(
                    "episode_closure_dashboard_batch",
                    "dashboard batch {} in update {} has an invalid contract".format(
                        batch_index, update
                    ),
                )
                continue
            for row in rows:
                manager_mean = row.get(
                    "reward_manager_episode_sum_mean"
                )
                expected = row.get("expected_dashboard_value")
                actual = row.get("actual_dashboard_value")
                if (
                    not _finite_number(manager_mean)
                    or not _finite_number(expected)
                    or not _finite_number(actual)
                    or not _close(
                        expected,
                        float(manager_mean)
                        / float(max_episode_length_s),
                    )
                    or not _close(actual, expected)
                    or not _finite_number(row.get("abs_error"))
                ):
                    audit.fail(
                        "episode_closure_dashboard_normalization",
                        "dashboard term {!r} in update {} does not normalize "
                        "_episode_sums by max_episode_length_s".format(
                            row.get("name"), update
                        ),
                    )
        dashboard = record.get("dashboard_normalization")
        expected_dashboard_status = (
            "PASS" if reset_batches else "NOT_OBSERVED_NO_RESET"
        )
        if (
            not isinstance(dashboard, dict)
            or dashboard.get("status") != expected_dashboard_status
            or dashboard.get("reset_batch_count") != len(reset_batches)
            or checks.get("reset_batch_count") != len(reset_batches)
            or checks.get("completed_episode_count") != len(completed)
        ):
            audit.fail(
                "episode_closure_dashboard_coverage",
                "closure update {} dashboard coverage/counts are inconsistent".format(
                    update
                ),
            )
        declared_e2 = record.get("e2_eligible")
        expected_e2 = (
            source == "live_isaac_reward_manager"
            and eligible_completed
            and bool(reset_batches)
        )
        if declared_e2 is not expected_e2:
            audit.fail(
                "episode_closure_e2_claim",
                "closure update {} has an unsupported e2_eligible claim".format(
                    update
                ),
            )
        if expected_e2:
            live_e2_updates.append(update)
        by_update[update] = record
    passed = len(audit.failures) == before and len(by_update) == len(records)
    audit.check(
        "episode_segmented_reward_closure",
        passed,
        "{} reset-hook PASS receipt(s); {} update(s) contain completed "
        "live Isaac episode evidence".format(
            len(records), len(live_e2_updates)
        )
        if passed
        else "missing/FAIL/malformed episode-segmented closure evidence",
    )
    return {
        "updates": updates,
        "by_update": by_update,
        "live_e2_updates": tuple(live_e2_updates),
        "all_sources_live": all_sources_live,
    }


def _validate_action_ledger(events, manifest, activation, audit):
    records = _group_event(events, ACTION_BALL_LEDGER_EVENT)
    if not records:
        audit.check(
            "curriculum_episode_accounting",
            False,
            "no action_ball_training_ledger records were supplied",
        )
        return None
    before = len(audit.failures)
    updates = _validate_update_sequence(
        records, field="step", label="action_ball_ledger", audit=audit
    )
    expected_updates = () if activation is None else activation["updates"]
    if updates != expected_updates:
        audit.fail(
            "action_ledger_update_coverage",
            "action-ball ledger updates {} differ from activation updates {}".format(
                updates, expected_updates
            ),
        )
    previous = None
    last = None
    by_update = {}
    for record in records:
        step = record.get("step")
        if record.get("schema_version") != ACTION_BALL_LEDGER_SCHEMA_VERSION:
            audit.fail(
                "action_ledger_schema",
                "action-ball ledger update {} has unsupported schema".format(step),
            )
        if record.get("status") != "report_only_requires_frozen_checkpoint_evidence":
            audit.fail(
                "action_ledger_status",
                "action-ball ledger update {} has an unknown status".format(step),
            )
        if record.get("diagnostic_unauthorized") is True:
            audit.fail(
                "action_ledger_diagnostic",
                "diagnostic_unauthorized ledger cannot satisfy a formal Reward gate",
            )
        if (
            manifest is None
            or tuple(record.get("action_order", ())) != manifest["order"]
            or record.get("manifest_sha256") != manifest["file_sha256"]
        ):
            audit.fail(
                "action_ledger_manifest",
                "action-ball ledger update {} is not bound to the exact manifest".format(
                    step
                ),
            )
        ledger = record.get("ledger")
        if not isinstance(ledger, dict) or (
            manifest is not None and set(ledger) != set(manifest["order"])
        ):
            audit.fail(
                "action_ledger_actions",
                "action-ball ledger update {} has an incomplete action set".format(
                    step
                ),
            )
            continue
        normalized = {}
        action_iteration = (
            tuple(ledger)
            if manifest is None
            else manifest["order"]
        )
        for action_id in action_iteration:
            raw_counters = ledger[action_id]
            if not isinstance(raw_counters, dict) or set(raw_counters) != set(
                LEDGER_COUNTERS
            ):
                audit.fail(
                    "action_ledger_counter_set",
                    "action {!r} has an invalid curriculum counter set".format(
                        action_id
                    ),
                )
                continue
            if any(not _nonnegative_int(value) for value in raw_counters.values()):
                audit.fail(
                    "action_ledger_counter_type",
                    "action {!r} has non-integer/negative curriculum counters".format(
                        action_id
                    ),
                )
                continue
            counters = dict(raw_counters)
            safe_closed = counters["L"] + counters["F"]
            unsafe_closed = counters["C"] - safe_closed
            raw_unsafe_counts = tuple(
                counters[name]
                for name in (
                    "U_table",
                    "U_fall",
                    "U_collision",
                    "U_joint_qdes",
                    "U_joint_actual",
                )
            )
            # ``C`` closes attempts once.  The U_* fields are independent raw
            # episode-sticky safety signals and may overlap on one closure.
            # Exact union conservation is proved later against the terminal
            # transcript; these are the strongest transcript-free bounds.
            if (
                unsafe_closed < 0
                or unsafe_closed < max(raw_unsafe_counts, default=0)
                or unsafe_closed > sum(raw_unsafe_counts)
            ):
                audit.fail(
                    "action_ledger_outcome_conservation",
                    "action {!r} violates "
                    "max(raw U_*) <= C-L-F <= sum(raw U_*); U_* "
                    "signals may overlap and cannot be summed as closure "
                    "outcomes".format(action_id),
                )
            if not (
                counters["P"] >= counters["A"] >= counters["I"]
                and counters["I"] == counters["S"]
                and counters["S"] >= counters["C"]
            ):
                audit.fail(
                    "action_ledger_pool_conservation",
                    "action {!r} violates P>=A>=I=S>=C".format(action_id),
                )
            if previous is not None:
                for name in LEDGER_COUNTERS:
                    if (
                        action_id not in previous
                        or counters[name] < previous[action_id][name]
                    ):
                        audit.fail(
                            "action_ledger_not_cumulative",
                            "action {!r} counter {} lost its prior cumulative row or decreased".format(
                                action_id, name
                            ),
                        )
            normalized[action_id] = counters
        expected_uid_keys = (
            set()
            if manifest is None
            else {str(manifest["uids"][name]) for name in manifest["order"]}
        )
        rejections = record.get("solver_rejections")
        pool = record.get("pool")
        if (
            not isinstance(rejections, dict)
            or set(rejections) != expected_uid_keys
            or not isinstance(pool, dict)
            or set(pool) != expected_uid_keys
        ):
            audit.fail(
                "action_ledger_solver_pool_keys",
                "solver_rejections/pool do not contain exactly one row per manifest UID",
            )
        elif manifest is not None:
            pool_fields = {
                "requests",
                "refill_calls",
                "proposed",
                "admitted",
                "issued",
                "discarded",
                "pending",
            }
            for action_id in manifest["order"]:
                uid = str(manifest["uids"][action_id])
                rejected_row = rejections[uid]
                pool_row = pool[uid]
                if (
                    not isinstance(rejected_row, dict)
                    or any(
                        type(reason) is not str
                        or not reason
                        or not _nonnegative_int(count)
                        for reason, count in rejected_row.items()
                    )
                    or not isinstance(pool_row, dict)
                    or set(pool_row) != pool_fields
                    or any(
                        not _nonnegative_int(value) for value in pool_row.values()
                    )
                ):
                    audit.fail(
                        "action_ledger_solver_pool_row",
                        "action {!r} has malformed solver rejection/pool rows".format(
                            action_id
                        ),
                    )
                    continue
                counters = normalized.get(action_id)
                if counters is None:
                    continue
                rejected = sum(rejected_row.values())
                if (
                    rejected != counters["P"] - counters["A"]
                    or pool_row["requests"] != counters["I"]
                    or pool_row["proposed"] != counters["P"]
                    or pool_row["admitted"] != counters["A"]
                    or pool_row["issued"] != counters["I"]
                    or pool_row["pending"]
                    != pool_row["admitted"]
                    - pool_row["issued"]
                    - pool_row["discarded"]
                    or pool_row["admitted"]
                    < pool_row["issued"] + pool_row["discarded"]
                ):
                    audit.fail(
                        "action_ledger_solver_pool_conservation",
                        "action {!r} rejection/pool rows do not close to P/A/I".format(
                            action_id
                        ),
                    )
        curriculum = record.get("curriculum")
        if not isinstance(curriculum, dict) or (
            manifest is not None and set(curriculum) != expected_uid_keys
        ):
            audit.fail(
                "curriculum_fields_missing",
                "action-ball ledger update {} lacks one curriculum row per action".format(
                    step
                ),
            )
        else:
            for uid, row in curriculum.items():
                if (
                    type(uid) is not str
                    or not isinstance(row, dict)
                    or type(row.get("phase")) is not str
                    or not isinstance(row.get("frontiers"), dict)
                    or not isinstance(row.get("expected_domains"), list)
                ):
                    audit.fail(
                        "curriculum_row_invalid",
                        "action-ball curriculum row {!r} is incomplete".format(uid),
                    )
        previous = normalized
        last = normalized
        by_update[step] = normalized
    passed = len(audit.failures) == before and last is not None
    audit.check(
        "curriculum_episode_accounting",
        passed,
        "{} cumulative per-action ledger update(s) conserve solver and outcomes".format(
            len(records)
        )
        if passed
        else "curriculum/episode ledgers are missing or fail conservation",
    )
    return {
        "updates": updates,
        "by_update": by_update,
        "last": last,
    }


def _safe_artifact_path(run_dir, relative):
    if type(relative) is not str or not relative:
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    root = Path(run_dir).resolve()
    unresolved = root / path
    candidate = unresolved.parent.resolve() / unresolved.name
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _decode_compact_identity(identity, *, policy_steps, manifest):
    expected_keys = {
        "encoding",
        "num_envs",
        "action_ball_enabled",
        "initial",
        "episode_length_int32",
        "swing_generation_int32",
        "changes",
        "initial_birth_receipt_sha256",
        "birth_receipt_changes",
        "per_step_identity_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != expected_keys:
        raise AuditInputError("joint-safety compact identity schema drift")
    num_envs = identity.get("num_envs")
    if (
        not _nonnegative_int(num_envs)
        or num_envs <= 0
        or identity.get("action_ball_enabled") is not True
        or identity.get("encoding")
        != (
            "initial_full_plus_sparse_reset_birth_changes_and_dense_"
            "episode_length_swing_generation"
        )
    ):
        raise AuditInputError("joint-safety compact identity header is invalid")
    fields = ("action_episode_sequence", "action_uid", "birth_generation")
    initial = identity.get("initial")
    changes = identity.get("changes")
    if (
        not isinstance(initial, dict)
        or set(initial) != set(fields)
        or not isinstance(changes, dict)
        or set(changes) != set(fields)
    ):
        raise AuditInputError("joint-safety initial/change identity fields drifted")
    current = {
        field: _plain_vector(
            initial[field],
            length=num_envs,
            label="identity.initial.{}".format(field),
            integer=True,
        )
        for field in fields
    }
    allowed_uids = (
        set()
        if manifest is None
        else {int(value) for value in manifest["uids"].values()}
    )
    if manifest is None or any(
        value not in allowed_uids for value in current["action_uid"]
    ):
        raise AuditInputError(
            "joint-safety action_uid is not bound to the exact manifest"
        )
    normalized_changes = {field: {} for field in fields}
    for field in fields:
        change = changes[field]
        if not isinstance(change, dict) or set(change) != {"index", "value"}:
            raise AuditInputError(
                "joint-safety identity change {!r} schema drift".format(field)
            )
        indices = _plain_value(change["index"])
        values = _plain_value(change["value"])
        if (
            not isinstance(indices, (list, tuple))
            or not isinstance(values, (list, tuple))
            or len(indices) != len(values)
        ):
            raise AuditInputError(
                "joint-safety identity change {!r} is malformed".format(field)
            )
        for coordinates, value in zip(indices, values):
            if (
                not isinstance(coordinates, (list, tuple))
                or len(coordinates) != 2
                or not _nonnegative_int(coordinates[0])
                or not _nonnegative_int(coordinates[1])
                or coordinates[0] <= 0
                or coordinates[0] >= policy_steps
                or coordinates[1] >= num_envs
                or not _nonnegative_int(value)
                or tuple(coordinates) in normalized_changes[field]
                or (
                    field == "action_uid"
                    and int(value) not in allowed_uids
                )
            ):
                raise AuditInputError(
                    "joint-safety identity change {!r} is invalid/duplicate".format(
                        field
                    )
                )
            normalized_changes[field][tuple(coordinates)] = int(value)

    receipts = _plain_value(identity.get("initial_birth_receipt_sha256"))
    if (
        not isinstance(receipts, (list, tuple))
        or len(receipts) != num_envs
        or any(not _is_sha256(value) for value in receipts)
    ):
        raise AuditInputError("joint-safety initial birth receipts are invalid")
    receipts = list(receipts)
    receipt_changes = {}
    raw_receipt_changes = _plain_value(identity.get("birth_receipt_changes"))
    if not isinstance(raw_receipt_changes, (list, tuple)):
        raise AuditInputError("joint-safety receipt changes must be a sequence")
    for item in raw_receipt_changes:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not _nonnegative_int(item[0])
            or not _nonnegative_int(item[1])
            or item[0] <= 0
            or item[0] >= policy_steps
            or item[1] >= num_envs
            or not _is_sha256(item[2])
            or (item[0], item[1]) in receipt_changes
        ):
            raise AuditInputError(
                "joint-safety birth receipt change is invalid/duplicate"
            )
        receipt_changes[(item[0], item[1])] = item[2]

    episode_lengths = _plain_matrix(
        identity.get("episode_length_int32"),
        rows=policy_steps,
        columns=num_envs,
        label="identity.episode_length_int32",
        integer=True,
        minimum=-1,
    )
    swing_generations = _plain_matrix(
        identity.get("swing_generation_int32"),
        rows=policy_steps,
        columns=num_envs,
        label="identity.swing_generation_int32",
        integer=True,
    )
    hashes = _plain_value(identity.get("per_step_identity_sha256"))
    if (
        not isinstance(hashes, (list, tuple))
        or len(hashes) != policy_steps
        or any(not _is_sha256(value) for value in hashes)
    ):
        raise AuditInputError("joint-safety per-step identity hashes are invalid")

    rows = []
    for step_index in range(policy_steps):
        for field in fields:
            for env_id in range(num_envs):
                key = (step_index, env_id)
                if key in normalized_changes[field]:
                    current[field][env_id] = normalized_changes[field][key]
        for env_id in range(num_envs):
            key = (step_index, env_id)
            if key in receipt_changes:
                receipts[env_id] = receipt_changes[key]
        if any(value not in allowed_uids for value in current["action_uid"]):
            raise AuditInputError(
                "joint-safety reconstructed action_uid left the manifest"
            )
        rows.append(
            {
                **{field: list(values) for field, values in current.items()},
                "episode_length": episode_lengths[step_index],
                "swing_generation": swing_generations[step_index],
                "birth_receipt_sha256": list(receipts),
                "identity_sha256": hashes[step_index],
            }
        )
        recomputed_identity_sha256 = _runner_identity_sha256(
            rows[-1], action_ball_enabled=True
        )
        if recomputed_identity_sha256 != hashes[step_index]:
            raise AuditInputError(
                "joint-safety reconstructed identity SHA-256 mismatch"
            )
        if step_index:
            previous = rows[step_index - 1]
            current_row = rows[step_index]
            for env_id in range(num_envs):
                delta = (
                    current_row["action_episode_sequence"][env_id]
                    - previous["action_episode_sequence"][env_id]
                )
                if delta not in (0, 1):
                    raise AuditInputError(
                        "joint-safety action episode generation jumped"
                    )
                if delta == 0 and (
                    current_row["action_uid"][env_id]
                    != previous["action_uid"][env_id]
                    or current_row["birth_generation"][env_id]
                    != previous["birth_generation"][env_id]
                    or current_row["birth_receipt_sha256"][env_id]
                    != previous["birth_receipt_sha256"][env_id]
                    or current_row["swing_generation"][env_id]
                    < previous["swing_generation"][env_id]
                ):
                    raise AuditInputError(
                        "joint-safety identity changed illegally inside an episode"
                    )
                if delta == 1 and (
                    current_row["birth_generation"][env_id]
                    <= previous["birth_generation"][env_id]
                    or current_row["birth_receipt_sha256"][env_id]
                    == previous["birth_receipt_sha256"][env_id]
                ):
                    raise AuditInputError(
                        "joint-safety reset reused birth generation/receipt"
                    )
    return {"num_envs": num_envs, "rows": rows}


def _validate_full_forensic_transcript(
    transcript, *, contract, policy_row, env_id
):
    expected_keys = {
        "schema_version",
        "policy_step_sequence",
        "policy_start_timestamp_s",
        "expected_apply_calls",
        "physics_dt_s",
        "apply_call_count",
        "post_readback_count",
        "complete",
        "record_count",
        "record_kind",
        "call_index",
        "timestamp_s",
        "joint_pos_timestamp_s",
        "joint_vel_timestamp_s",
        "env_valid",
        "q",
        "qdot",
        "hard_lower_gap",
        "hard_upper_gap",
        "hard_crossing",
        "actual_hard_edge",
        "qdes_env_latch",
        "crossing_env_latch",
        "qdes_joint_latch",
        "crossing_joint_latch",
        "qdes_joint_count",
        "crossing_joint_count",
        "substep_crossing_joint_latch",
        "substep_actual_joint_latch",
        "substep_crossing_joint_count",
        "substep_actual_joint_count",
        "step_qdes_joint_count",
        "step_policy_crossing_joint_count",
    }
    if not isinstance(transcript, dict) or set(transcript) != expected_keys:
        raise AuditInputError("joint-safety full transcript schema drift")
    apply_calls = contract["expected_apply_calls"]
    record_count = apply_calls + 1
    physics_dt = float(contract["physics_dt_s"])
    start = transcript.get("policy_start_timestamp_s")
    if (
        transcript.get("schema_version") != 1
        or transcript.get("policy_step_sequence")
        != policy_row["policy_step_sequence"]
        or transcript.get("expected_apply_calls") != apply_calls
        or transcript.get("apply_call_count") != apply_calls
        or transcript.get("post_readback_count") != 1
        or transcript.get("complete") is not True
        or transcript.get("record_count") != record_count
        or not _close(transcript.get("physics_dt_s"), physics_dt, rel_tol=0.0)
        or not _finite_number(start)
    ):
        raise AuditInputError("joint-safety full transcript header is incomplete")
    expected_kind = ["apply"] * apply_calls + ["post"]
    if (
        list(transcript.get("record_kind", ())) != expected_kind
        or list(transcript.get("call_index", ())) != list(range(record_count))
    ):
        raise AuditInputError("joint-safety full transcript readback order drift")
    timestamps = []
    for field in (
        "timestamp_s",
        "joint_pos_timestamp_s",
        "joint_vel_timestamp_s",
    ):
        values = _plain_vector(
            transcript.get(field),
            length=record_count,
            label="transcript.{}".format(field),
        )
        timestamps.append(values)
    if timestamps[0] != timestamps[1] or timestamps[0] != timestamps[2]:
        raise AuditInputError("joint-safety lazy-buffer timestamps drift")
    for index, timestamp in enumerate(timestamps[0]):
        if not math.isclose(
            timestamp,
            float(start) + index * physics_dt,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ) or (index and timestamp <= timestamps[0][index - 1]):
            raise AuditInputError("joint-safety transcript timestamp sequence drift")

    joint_count = contract["joint_count"]
    if not all(
        _plain_bool_vector(
            transcript.get("env_valid"),
            length=record_count,
            label="transcript.env_valid",
        )
    ):
        raise AuditInputError("joint-safety transcript contains an invalid row")
    q = _plain_matrix(
        transcript.get("q"),
        rows=record_count,
        columns=joint_count,
        label="transcript.q",
    )
    qdot = _plain_matrix(
        transcript.get("qdot"),
        rows=record_count,
        columns=joint_count,
        label="transcript.qdot",
    )
    lower_gap = _plain_matrix(
        transcript.get("hard_lower_gap"),
        rows=record_count,
        columns=joint_count,
        label="transcript.hard_lower_gap",
    )
    upper_gap = _plain_matrix(
        transcript.get("hard_upper_gap"),
        rows=record_count,
        columns=joint_count,
        label="transcript.hard_upper_gap",
    )
    hard_crossing = _plain_bool_matrix(
        transcript.get("hard_crossing"),
        rows=record_count,
        columns=joint_count,
        label="transcript.hard_crossing",
    )
    actual_hard = _plain_bool_matrix(
        transcript.get("actual_hard_edge"),
        rows=record_count,
        columns=joint_count,
        label="transcript.actual_hard_edge",
    )
    hard_lower = _plain_value(contract["hard_lower"])
    hard_upper = _plain_value(contract["hard_upper"])
    crossing_counts = [0] * joint_count
    actual_counts = [0] * joint_count
    for row_index in range(record_count):
        for joint_id in range(joint_count):
            expected_lower = q[row_index][joint_id] - hard_lower[joint_id]
            expected_upper = hard_upper[joint_id] - q[row_index][joint_id]
            if not _close(
                lower_gap[row_index][joint_id],
                expected_lower,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ) or not _close(
                upper_gap[row_index][joint_id],
                expected_upper,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise AuditInputError(
                    "joint-safety transcript q/hard-gap mismatch"
                )
            expected_actual = expected_lower <= 0.0 or expected_upper <= 0.0
            travel = hard_upper[joint_id] - hard_lower[joint_id]
            inset = (
                float(contract["margin_rad"])
                + float(contract["margin_fraction"]) * travel
            )
            inner_lower = hard_lower[joint_id] + inset
            inner_upper = hard_upper[joint_id] - inset
            ballistic = q[row_index][joint_id] + qdot[row_index][joint_id] * physics_dt
            expected_crossing = (
                q[row_index][joint_id] <= inner_lower
                or q[row_index][joint_id] >= inner_upper
                or ballistic <= inner_lower
                or ballistic >= inner_upper
            )
            if actual_hard[row_index][joint_id] is not expected_actual:
                raise AuditInputError(
                    "joint-safety actual-hard-edge transcript mask is forged"
                )
            if hard_crossing[row_index][joint_id] is not expected_crossing:
                raise AuditInputError(
                    "joint-safety hard-crossing transcript mask is forged"
                )
            crossing_counts[joint_id] += int(expected_crossing)
            actual_counts[joint_id] += int(expected_actual)

    bool_vectors = {
        field: _plain_bool_vector(
            transcript.get(field),
            length=joint_count,
            label="transcript.{}".format(field),
        )
        for field in (
            "qdes_joint_latch",
            "crossing_joint_latch",
            "substep_crossing_joint_latch",
            "substep_actual_joint_latch",
        )
    }
    count_vectors = {
        field: _plain_vector(
            transcript.get(field),
            length=joint_count,
            label="transcript.{}".format(field),
            integer=True,
        )
        for field in (
            "qdes_joint_count",
            "crossing_joint_count",
            "substep_crossing_joint_count",
            "substep_actual_joint_count",
            "step_qdes_joint_count",
            "step_policy_crossing_joint_count",
        )
    }
    qdes_env = _plain_value(transcript.get("qdes_env_latch"))
    crossing_env = _plain_value(transcript.get("crossing_env_latch"))
    if type(qdes_env) is not bool or type(crossing_env) is not bool:
        raise AuditInputError("joint-safety transcript env latch dtype drift")
    for joint_id in range(joint_count):
        if (
            count_vectors["substep_crossing_joint_count"][joint_id]
            != crossing_counts[joint_id]
            or count_vectors["substep_actual_joint_count"][joint_id]
            != actual_counts[joint_id]
            or bool_vectors["substep_crossing_joint_latch"][joint_id]
            != (crossing_counts[joint_id] > 0)
            or bool_vectors["substep_actual_joint_latch"][joint_id]
            != (actual_counts[joint_id] > 0)
            or bool_vectors["qdes_joint_latch"][joint_id]
            != (count_vectors["qdes_joint_count"][joint_id] > 0)
            or bool_vectors["crossing_joint_latch"][joint_id]
            != (
                count_vectors["crossing_joint_count"][joint_id] > 0
                or crossing_counts[joint_id] > 0
                or actual_counts[joint_id] > 0
            )
            or count_vectors["step_qdes_joint_count"][joint_id]
            > count_vectors["qdes_joint_count"][joint_id]
            or count_vectors["step_policy_crossing_joint_count"][joint_id]
            > count_vectors["crossing_joint_count"][joint_id]
        ):
            raise AuditInputError(
                "joint-safety transcript latch/counter conservation failed"
            )
    if qdes_env != any(bool_vectors["qdes_joint_latch"]) or crossing_env != any(
        bool_vectors["crossing_joint_latch"]
    ):
        raise AuditInputError("joint-safety transcript env latch mismatch")
    for field, transcript_field in (
        ("qdes_joint_count", "step_qdes_joint_count"),
        ("policy_crossing_joint_count", "step_policy_crossing_joint_count"),
        ("substep_hard_crossing_joint_count", "substep_crossing_joint_count"),
        ("actual_hard_edge_joint_count", "substep_actual_joint_count"),
    ):
        compact = _sparse_counter_map(policy_row["sparse_counters"][field])
        expected = [
            compact.get((env_id, joint_id), 0)
            for joint_id in range(joint_count)
        ]
        if count_vectors[transcript_field] != expected:
            raise AuditInputError(
                "joint-safety transcript does not bind compact step counters"
            )
    transcript_minimum = min(
        min(row) for row in lower_gap + upper_gap
    )
    if (
        policy_row["minimum_hard_gap_env_id"] == env_id
        and not _close(
            transcript_minimum,
            policy_row["minimum_hard_gap_rad"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise AuditInputError(
            "joint-safety transcript minimum gap does not bind compact step"
        )


def _validate_joint_sidecar_payload(
    payload, *, record, manifest
):
    expected_payload_keys = {
        "event",
        "schema_version",
        "status",
        "rank",
        "ppo_update",
        "contract",
        "sequence",
        "completeness",
        "identity",
        "policy_steps",
        "aggregate_sparse_counters",
        "gaps",
        "fatal_flags",
        "terminal",
        "budgets",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
        raise AuditInputError("joint-safety sidecar top-level schema drift")
    if (
        payload.get("event") != JOINT_SAFETY_EVENT
        or payload.get("schema_version") != JOINT_SAFETY_SCHEMA_VERSION
        or payload.get("status") != "prepared_before_optimizer"
        or payload.get("ppo_update") != record.get("ppo_update")
        or not _nonnegative_int(payload.get("rank"))
    ):
        raise AuditInputError("joint-safety sidecar header is unbound")
    contract = payload.get("contract")
    expected_contract_keys = {
        "expected_apply_calls",
        "physics_dt_s",
        "margin_rad",
        "margin_fraction",
        "num_envs",
        "joint_count",
        "joint_names",
        "hard_lower",
        "hard_upper",
        "sha256",
    }
    if not isinstance(contract, dict) or set(contract) != expected_contract_keys:
        raise AuditInputError("joint-safety hard-envelope contract schema drift")
    num_envs = contract.get("num_envs")
    joint_count = contract.get("joint_count")
    if (
        not _nonnegative_int(num_envs)
        or num_envs <= 0
        or num_envs != record.get("num_envs")
        or not _nonnegative_int(joint_count)
        or joint_count <= 0
        or not _nonnegative_int(contract.get("expected_apply_calls"))
        or contract["expected_apply_calls"] <= 0
        or not _finite_number(contract.get("physics_dt_s"))
        or float(contract["physics_dt_s"]) <= 0.0
        or not _finite_number(contract.get("margin_rad"))
        or float(contract["margin_rad"]) < 0.0
        or not _finite_number(contract.get("margin_fraction"))
        or not 0.0 <= float(contract["margin_fraction"]) < 0.5
        or not _is_sha256(contract.get("sha256"))
    ):
        raise AuditInputError("joint-safety hard-envelope contract is invalid")
    names = _plain_value(contract.get("joint_names"))
    if (
        not isinstance(names, (list, tuple))
        or len(names) != joint_count
        or any(type(value) is not str or not value for value in names)
        or len(set(names)) != len(names)
    ):
        raise AuditInputError("joint-safety joint-name order is invalid")
    lower = _plain_vector(
        contract.get("hard_lower"),
        length=joint_count,
        label="contract.hard_lower",
    )
    upper = _plain_vector(
        contract.get("hard_upper"),
        length=joint_count,
        label="contract.hard_upper",
    )
    if any(not low < high for low, high in zip(lower, upper)):
        raise AuditInputError("joint-safety hard envelope is empty/inverted")
    lower_dtype, lower_shape, lower_bytes = _tensor_metadata(
        contract["hard_lower"], label="contract.hard_lower"
    )
    upper_dtype, upper_shape, upper_bytes = _tensor_metadata(
        contract["hard_upper"], label="contract.hard_upper"
    )
    if (
        lower_dtype != upper_dtype
        or lower_shape != (joint_count,)
        or upper_shape != (joint_count,)
    ):
        raise AuditInputError("joint-safety hard-envelope tensor metadata drift")
    scalar_contract = {
        "expected_apply_calls": contract["expected_apply_calls"],
        "physics_dt_s": float(contract["physics_dt_s"]),
        "margin_rad": float(contract["margin_rad"]),
        "margin_fraction": float(contract["margin_fraction"]),
        "num_envs": num_envs,
        "joint_count": joint_count,
        "joint_names": list(names),
    }
    contract_digest = hashlib.sha256()
    contract_digest.update(
        json.dumps(
            scalar_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    for name, dtype, encoded in (
        ("hard_lower", lower_dtype, lower_bytes),
        ("hard_upper", upper_dtype, upper_bytes),
    ):
        contract_digest.update(name.encode("ascii"))
        contract_digest.update(dtype.encode("ascii"))
        contract_digest.update(encoded)
    if contract_digest.hexdigest() != contract["sha256"]:
        raise AuditInputError(
            "joint-safety hard-envelope contract SHA-256 mismatch"
        )

    steps = record.get("policy_step_count")
    identity = _decode_compact_identity(
        payload.get("identity"), policy_steps=steps, manifest=manifest
    )
    if identity["num_envs"] != num_envs:
        raise AuditInputError("joint-safety identity num_envs drift")

    sequence = payload.get("sequence")
    if (
        not isinstance(sequence, dict)
        or set(sequence)
        != {
            "consume_sequence",
            "first_policy_step_sequence",
            "last_policy_step_sequence",
            "last_archive_sequence",
        }
        or sequence.get("consume_sequence") != record.get("consume_sequence")
        or sequence.get("first_policy_step_sequence")
        != record.get("first_policy_step_sequence")
        or sequence.get("last_policy_step_sequence")
        != record.get("last_policy_step_sequence")
    ):
        raise AuditInputError("joint-safety sidecar sequence receipt drift")
    completeness = payload.get("completeness")
    if (
        not isinstance(completeness, dict)
        or set(completeness)
        != {
            "all_rows_present",
            "all_policy_steps_complete",
            "expected_apply_readbacks",
            "expected_post_readbacks",
            "timestamp_invariant",
        }
        or completeness.get("all_rows_present") is not True
        or completeness.get("all_policy_steps_complete") is not True
        or completeness.get("timestamp_invariant") is not True
        or completeness.get("expected_apply_readbacks")
        != contract["expected_apply_calls"]
        or completeness.get("expected_post_readbacks") != 1
    ):
        raise AuditInputError("joint-safety completeness contract is false")

    policy_rows = payload.get("policy_steps")
    if not isinstance(policy_rows, (list, tuple)) or len(policy_rows) != steps:
        raise AuditInputError("joint-safety sidecar policy-step count drift")
    per_step_totals = {name: 0 for name in JOINT_COUNTERS}
    first_sequence = record["first_policy_step_sequence"]
    stdout_rows = record.get("per_policy_step_sparse_counters")
    if not isinstance(stdout_rows, list) or len(stdout_rows) != steps:
        raise AuditInputError("joint-safety stdout sparse-step rows are incomplete")
    for index, row in enumerate(policy_rows):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "policy_step_sequence",
                "policy_start_timestamp_s",
                "identity_sha256",
                "minimum_hard_gap_rad",
                "minimum_hard_gap_env_id",
                "minimum_hard_gap_joint_id",
                "per_action_minimum_hard_gap",
                "sparse_counters",
            }
            or row.get("policy_step_sequence") != first_sequence + index
            or row.get("identity_sha256")
            != identity["rows"][index]["identity_sha256"]
            or not _finite_number(row.get("policy_start_timestamp_s"))
            or not _finite_number(row.get("minimum_hard_gap_rad"))
            or not _nonnegative_int(row.get("minimum_hard_gap_env_id"))
            or row["minimum_hard_gap_env_id"] >= num_envs
            or not _nonnegative_int(row.get("minimum_hard_gap_joint_id"))
            or row["minimum_hard_gap_joint_id"] >= joint_count
        ):
            raise AuditInputError(
                "joint-safety policy-step row {} is malformed".format(index)
            )
        per_action = row.get("per_action_minimum_hard_gap")
        action_uids = (
            []
            if not isinstance(per_action, dict)
            else _plain_value(per_action.get("action_uid"))
        )
        if (
            not isinstance(per_action, dict)
            or set(per_action) != {"action_uid", "minimum_gap_rad"}
            or not isinstance(action_uids, (list, tuple))
            or not action_uids
            or len(set(action_uids)) != len(action_uids)
            or any(
                not _nonnegative_int(value)
                or value not in set(manifest["uids"].values())
                for value in action_uids
            )
        ):
            raise AuditInputError(
                "joint-safety per-action gap identity is invalid"
            )
        _plain_matrix(
            per_action["minimum_gap_rad"],
            rows=len(action_uids),
            columns=joint_count,
            label="policy_steps[{}].per_action_minimum_gap".format(index),
        )
        sparse = row.get("sparse_counters")
        if not isinstance(sparse, dict) or set(sparse) != set(JOINT_COUNTERS):
            raise AuditInputError(
                "joint-safety policy-step sparse counter set drift"
            )
        stdout_row = stdout_rows[index]
        if (
            not isinstance(stdout_row, dict)
            or stdout_row.get("policy_step_sequence")
            != row["policy_step_sequence"]
            or stdout_row.get("identity_sha256") != row["identity_sha256"]
            or stdout_row.get("complete_env_count") != num_envs
            or stdout_row.get("incomplete_env_count") != 0
            or not _close(
                stdout_row.get("minimum_hard_gap_rad"),
                row["minimum_hard_gap_rad"],
            )
            or not isinstance(stdout_row.get("sparse_counters"), dict)
        ):
            raise AuditInputError(
                "joint-safety stdout/sidecar policy-step binding drift"
            )
        for name in JOINT_COUNTERS:
            event_count = _validate_sparse_counter(
                sparse[name],
                num_envs=num_envs,
                joint_count=joint_count,
                label="policy_steps[{}].{}".format(index, name),
            )
            per_step_totals[name] += event_count
            stdout_counter = stdout_row["sparse_counters"].get(name)
            if (
                not isinstance(stdout_counter, dict)
                or stdout_counter.get("nonzero_cells")
                != sparse[name]["nonzero_cells"]
                or stdout_counter.get("event_count") != event_count
            ):
                raise AuditInputError(
                    "joint-safety stdout sparse counter does not bind sidecar"
                )

    aggregate = payload.get("aggregate_sparse_counters")
    if not isinstance(aggregate, dict) or set(aggregate) != set(JOINT_COUNTERS):
        raise AuditInputError("joint-safety aggregate sparse counter set drift")
    for name in JOINT_COUNTERS:
        total = _validate_sparse_counter(
            aggregate[name],
            num_envs=num_envs,
            joint_count=joint_count,
            label="aggregate.{}".format(name),
        )
        if (
            total != per_step_totals[name]
            or total != record["counter_totals"][name]
        ):
            raise AuditInputError(
                "joint-safety {!r} counter does not close".format(name)
            )

    fatal_flags = payload.get("fatal_flags")
    if (
        not isinstance(fatal_flags, dict)
        or set(fatal_flags)
        != {
            "actual_hard_edge_event_count",
            "nonpositive_physical_hard_gap_cell_count",
        }
        or any(not _nonnegative_int(value) for value in fatal_flags.values())
        or fatal_flags.get("actual_hard_edge_event_count") != 0
        or fatal_flags.get("nonpositive_physical_hard_gap_cell_count") != 0
        or record.get("fatal_flags") != fatal_flags
    ):
        raise AuditInputError(
            "committed joint-safety update carries an invalid fatal flag"
        )
    gaps = payload.get("gaps")
    if (
        not isinstance(gaps, dict)
        or set(gaps)
        != {
            "minimum_lower_gap_by_joint",
            "minimum_lower_gap_env_id_by_joint",
            "minimum_upper_gap_by_joint",
            "minimum_upper_gap_env_id_by_joint",
        }
    ):
        raise AuditInputError("joint-safety aggregate gap schema drift")
    lower_gaps = _plain_vector(
        gaps["minimum_lower_gap_by_joint"],
        length=joint_count,
        label="gaps.minimum_lower_gap_by_joint",
    )
    upper_gaps = _plain_vector(
        gaps["minimum_upper_gap_by_joint"],
        length=joint_count,
        label="gaps.minimum_upper_gap_by_joint",
    )
    lower_gap_envs = _plain_vector(
        gaps["minimum_lower_gap_env_id_by_joint"],
        length=joint_count,
        label="gaps.minimum_lower_gap_env_id_by_joint",
        integer=True,
    )
    upper_gap_envs = _plain_vector(
        gaps["minimum_upper_gap_env_id_by_joint"],
        length=joint_count,
        label="gaps.minimum_upper_gap_env_id_by_joint",
        integer=True,
    )
    if (
        any(value <= 0.0 for value in lower_gaps + upper_gaps)
        or any(value >= num_envs for value in lower_gap_envs + upper_gap_envs)
        or not _close(
            min(lower_gaps + upper_gaps),
            record.get("minimum_hard_gap_rad"),
        )
    ):
        raise AuditInputError(
            "joint-safety aggregate hard gap does not close to stdout"
        )
    if not _finite_number(record.get("minimum_hard_gap_rad")) or float(
        record["minimum_hard_gap_rad"]
    ) <= 0.0:
        raise AuditInputError(
            "committed joint-safety update reached the physical hard edge"
        )

    terminal = payload.get("terminal")
    if (
        not isinstance(terminal, dict)
        or set(terminal) != {"archive_count", "entries"}
        or not _nonnegative_int(terminal.get("archive_count"))
        or not isinstance(terminal.get("entries"), (list, tuple))
        or terminal["archive_count"] != len(terminal["entries"])
        or terminal["archive_count"] != record.get("terminal_archive_count")
    ):
        raise AuditInputError("joint-safety terminal archive count drift")
    reason_counts = {}
    seen_archive_keys = set()
    terminal_archives = {}
    previous_archive_sequence = None
    common_archive_keys = {
        "archive_sequence",
        "env_id",
        "policy_step_sequence",
        "action_episode_sequence",
        "episode_length",
        "episode_length_at_policy_start",
        "episode_length_at_reset_hook",
        "action_ball_enabled",
        "action_uid",
        "birth_generation",
        "swing_generation",
        "birth_receipt_sha256",
        "reasons",
        "reset_hook_observed",
        "termination_status_available",
        "terminated",
        "timed_out",
        "included_in_accumulator",
        "accumulator_consume_sequence",
        "transcript",
    }
    for index, entry in enumerate(terminal["entries"]):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"storage", "archive"}
            or entry.get("storage")
            not in {
                "full_forensic",
                "compact_timeout_or_nonterminated_reset",
            }
            or not isinstance(entry.get("archive"), dict)
        ):
            raise AuditInputError(
                "joint-safety terminal entry {} schema drift".format(index)
            )
        archive = entry["archive"]
        storage = entry["storage"]
        expected_archive_keys = (
            common_archive_keys | {"payload_bytes"}
            if storage == "full_forensic"
            else common_archive_keys | {"source_payload_bytes"}
        )
        archive_sequence = archive.get("archive_sequence")
        if (
            set(archive) != expected_archive_keys
            or not _nonnegative_int(archive_sequence)
            or (
                previous_archive_sequence is not None
                and archive_sequence != previous_archive_sequence + 1
            )
        ):
            raise AuditInputError(
                "joint-safety terminal entry {} archive schema/sequence drift".format(
                    index
                )
            )
        previous_archive_sequence = archive_sequence
        env_id = archive.get("env_id")
        policy_sequence = archive.get("policy_step_sequence")
        step_index = (
            -1
            if not _nonnegative_int(policy_sequence)
            else policy_sequence - first_sequence
        )
        if (
            not _nonnegative_int(env_id)
            or env_id >= num_envs
            or step_index < 0
            or step_index >= steps
            or (policy_sequence, env_id) in seen_archive_keys
        ):
            raise AuditInputError(
                "joint-safety terminal entry {} identity is invalid".format(index)
            )
        seen_archive_keys.add((policy_sequence, env_id))
        identity_row = identity["rows"][step_index]
        if (
            archive.get("action_episode_sequence")
            != identity_row["action_episode_sequence"][env_id]
            or archive.get("episode_length_at_policy_start")
            != identity_row["episode_length"][env_id]
            or archive.get("action_ball_enabled") is not True
            or archive.get("action_uid") != identity_row["action_uid"][env_id]
            or archive.get("birth_generation")
            != identity_row["birth_generation"][env_id]
            or archive.get("swing_generation")
            != identity_row["swing_generation"][env_id]
            or archive.get("birth_receipt_sha256")
            != identity_row["birth_receipt_sha256"][env_id]
        ):
            raise AuditInputError(
                "joint-safety terminal entry {} is not identity-bound".format(index)
            )
        reasons = archive.get("reasons")
        if (
            not isinstance(reasons, (list, tuple))
            or not reasons
            or len(reasons) != len(set(reasons))
            or "reset" not in reasons
            or any(value not in {"unsafe", "reset"} for value in reasons)
            or not isinstance(archive.get("terminated"), bool)
            or not isinstance(archive.get("timed_out"), bool)
            or not isinstance(archive.get("termination_status_available"), bool)
            or archive.get("reset_hook_observed") is not True
            or archive.get("included_in_accumulator") is not True
            or archive.get("accumulator_consume_sequence")
            != sequence["consume_sequence"]
            or archive.get("episode_length")
            != archive.get("episode_length_at_reset_hook")
            or (
                archive["termination_status_available"]
                and not (archive["terminated"] or archive["timed_out"])
            )
            or not isinstance(archive.get("transcript"), dict)
            or archive["transcript"].get("policy_step_sequence")
            != policy_sequence
        ):
            raise AuditInputError(
                "joint-safety terminal entry {} outcome/transcript is invalid".format(
                    index
                )
            )
        if (
            archive["terminated"]
            or "unsafe" in reasons
        ) and storage != "full_forensic":
            raise AuditInputError(
                "terminated/unsafe joint-safety archive lost full forensic data"
            )
        policy_row = policy_rows[step_index]
        if storage == "full_forensic":
            if (
                not _nonnegative_int(archive.get("payload_bytes"))
                or archive["payload_bytes"] != _joint_payload_bytes(archive)
            ):
                raise AuditInputError(
                    "joint-safety full archive payload budget mismatch"
                )
            _validate_full_forensic_transcript(
                archive["transcript"],
                contract=contract,
                policy_row=policy_row,
                env_id=env_id,
            )
        else:
            compact = archive["transcript"]
            if (
                not _nonnegative_int(archive.get("source_payload_bytes"))
                or not isinstance(compact, dict)
                or set(compact)
                != {
                    "storage",
                    "source_sha256",
                    "schema_version",
                    "policy_step_sequence",
                    "policy_start_timestamp_s",
                    "record_count",
                    "record_kind",
                    "call_index",
                    "timestamp_s",
                    "step_qdes_joint_count",
                    "step_policy_crossing_joint_count",
                    "substep_crossing_joint_count",
                    "substep_actual_joint_count",
                }
                or compact.get("storage") != "validated_sha256_compact"
                or not _is_sha256(compact.get("source_sha256"))
                or compact.get("schema_version") != 1
                or compact.get("policy_step_sequence") != policy_sequence
            ):
                raise AuditInputError(
                    "joint-safety compact terminal transcript schema drift"
                )
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        terminal_archives[(policy_sequence, env_id)] = {
            "action_episode_sequence": archive["action_episode_sequence"],
            "action_uid": archive["action_uid"],
            "birth_generation": archive["birth_generation"],
            "swing_generation": archive["swing_generation"],
            "birth_receipt_sha256": archive["birth_receipt_sha256"],
            "reasons": tuple(reasons),
            "terminated": archive["terminated"],
            "timed_out": archive["timed_out"],
            "hard_event": any(
                _sparse_counter_map(policy_row["sparse_counters"][name]).get(
                    (env_id, joint_id), 0
                )
                > 0
                for name in (
                    "policy_crossing_joint_count",
                    "substep_hard_crossing_joint_count",
                    "actual_hard_edge_joint_count",
                )
                for joint_id in range(joint_count)
            ),
        }
    if terminal["entries"] and previous_archive_sequence != sequence[
        "last_archive_sequence"
    ]:
        raise AuditInputError(
            "joint-safety last archive sequence does not bind sidecar"
        )
    if record.get("terminal_reason_counts") != dict(sorted(reason_counts.items())):
        raise AuditInputError(
            "joint-safety terminal reason counts do not close to sidecar"
        )

    budgets = payload.get("budgets")
    expected_budget_keys = {
        "core_payload_bytes",
        "terminal_payload_bytes",
        "total_payload_bytes",
        "core_payload_max_bytes",
        "terminal_payload_max_bytes",
        "normal_serialized_max_bytes",
        "forensic_serialized_max_bytes",
    }
    if (
        not isinstance(budgets, dict)
        or set(budgets) != expected_budget_keys
        or any(not _nonnegative_int(value) for value in budgets.values())
        or budgets["total_payload_bytes"]
        != budgets["core_payload_bytes"] + budgets["terminal_payload_bytes"]
        or budgets["core_payload_bytes"] > budgets["core_payload_max_bytes"]
        or budgets["terminal_payload_bytes"]
        > budgets["terminal_payload_max_bytes"]
    ):
        raise AuditInputError("joint-safety sidecar budget receipt is invalid")
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"terminal", "budgets"}
    }
    core_bytes = _joint_payload_bytes(core)
    terminal_bytes = _joint_payload_bytes(terminal)
    if (
        core_bytes != budgets["core_payload_bytes"]
        or terminal_bytes != budgets["terminal_payload_bytes"]
        or core_bytes + terminal_bytes != budgets["total_payload_bytes"]
    ):
        raise AuditInputError(
            "joint-safety sidecar budget receipt does not recompute"
        )
    return {
        "counter_totals": per_step_totals,
        "identity_rows": identity["rows"],
        "terminal_archives": terminal_archives,
        "rank": payload["rank"],
        "consume_sequence": sequence["consume_sequence"],
    }


def _validate_optimizer_commit(
    *, record, artifact, run_dir, expected_rank
):
    receipt = record.get("optimizer_commit")
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {"format", "schema_version", "path", "sha256", "size_bytes"}
        or receipt.get("format") != "canonical_json"
        or receipt.get("schema_version") != 1
        or not _is_sha256(receipt.get("sha256"))
        or not _nonnegative_int(receipt.get("size_bytes"))
        or receipt["size_bytes"] <= 0
    ):
        raise AuditInputError("joint-safety optimizer-commit receipt is invalid")
    path = _safe_artifact_path(run_dir, receipt.get("path"))
    if (
        path is None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != receipt["size_bytes"]
        or _sha256_file(path) != receipt["sha256"]
    ):
        raise AuditInputError("joint-safety optimizer-commit bytes mismatch")
    marker = _load_json_object(path, "joint-safety optimizer commit")
    if (
        set(marker)
        != {
            "event",
            "schema_version",
            "ppo_update",
            "rank",
            "prepared_artifact_path",
            "prepared_artifact_sha256",
            "consume_sequence",
            "status",
        }
        or marker.get("event") != "hope_joint_safety_optimizer_commit"
        or marker.get("schema_version") != 1
        or marker.get("ppo_update") != record.get("ppo_update")
        or marker.get("prepared_artifact_path") != artifact.get("path")
        or marker.get("prepared_artifact_sha256") != artifact.get("sha256")
        or marker.get("consume_sequence") != record.get("consume_sequence")
        or marker.get("status")
        != "optimizer_succeeded_pending_ledger_ack"
        or marker.get("rank") != expected_rank
    ):
        raise AuditInputError("joint-safety optimizer-commit marker is unbound")


def _validate_joint_safety(
    events,
    activation,
    manifest,
    run_dir,
    audit,
    *,
    sidecar_loader,
):
    records = _group_event(events, JOINT_SAFETY_EVENT)
    if not records:
        audit.check(
            "joint_safety_sidecars",
            False,
            "no hope_joint_safety_update records were supplied",
        )
        return None
    before = len(audit.failures)
    updates = _validate_update_sequence(
        records, field="ppo_update", label="joint_safety", audit=audit
    )
    expected_updates = () if activation is None else activation["updates"]
    if updates != expected_updates:
        audit.fail(
            "joint_safety_update_coverage",
            "joint-safety updates {} differ from activation updates {}".format(
                updates, expected_updates
            ),
        )
    previous_last_sequence = None
    previous_consume_sequence = None
    bound_rank = None
    totals = {name: 0 for name in JOINT_COUNTERS}
    by_update = {}
    for record in records:
        update = record.get("ppo_update")
        if (
            record.get("schema_version") != JOINT_SAFETY_SCHEMA_VERSION
            or record.get("status")
            != "optimizer_committed_and_ledger_acknowledged"
        ):
            audit.fail(
                "joint_safety_schema",
                "joint-safety update {} has unsupported schema/status".format(update),
            )
        num_envs = record.get("num_envs")
        steps = record.get("policy_step_count")
        complete = record.get("complete_env_policy_steps")
        incomplete = record.get("incomplete_env_policy_steps")
        if (
            not _nonnegative_int(num_envs)
            or num_envs <= 0
            or not _nonnegative_int(steps)
            or steps <= 0
            or complete != num_envs * steps
            or incomplete != 0
        ):
            audit.fail(
                "joint_safety_step_conservation",
                "joint-safety update {} is incomplete".format(update),
            )
        first_sequence = record.get("first_policy_step_sequence")
        last_sequence = record.get("last_policy_step_sequence")
        if (
            not _nonnegative_int(first_sequence)
            or not _nonnegative_int(last_sequence)
            or last_sequence - first_sequence + 1 != steps
            or (
                previous_last_sequence is not None
                and first_sequence != previous_last_sequence + 1
            )
        ):
            audit.fail(
                "joint_safety_sequence",
                "joint-safety policy-step sequence is not exact/contiguous",
            )
        if _nonnegative_int(last_sequence):
            previous_last_sequence = last_sequence
        counters = record.get("counter_totals")
        if (
            not isinstance(counters, dict)
            or any(name not in counters for name in JOINT_COUNTERS)
            or any(not _nonnegative_int(counters.get(name)) for name in JOINT_COUNTERS)
        ):
            audit.fail(
                "joint_safety_counters",
                "joint-safety update {} lacks reviewed sparse counters".format(update),
            )
        else:
            for name in JOINT_COUNTERS:
                totals[name] += counters[name]
        if (
            not _nonnegative_int(record.get("terminal_archive_count"))
            or not isinstance(record.get("terminal_reason_counts"), dict)
            or not isinstance(record.get("per_policy_step_sparse_counters"), list)
            or len(record["per_policy_step_sparse_counters"]) != steps
        ):
            audit.fail(
                "joint_safety_terminal_fields",
                "joint-safety update {} lacks terminal/per-step fields".format(update),
            )
        artifact = record.get("artifact")
        if (
            not isinstance(artifact, dict)
            or artifact.get("format") != "torch_save_cpu"
            or artifact.get("schema_version") != JOINT_SAFETY_SCHEMA_VERSION
            or artifact.get("status") != "prepared_before_optimizer"
            or not _is_sha256(artifact.get("sha256"))
            or not _nonnegative_int(artifact.get("size_bytes"))
            or artifact.get("size_bytes") <= 0
        ):
            audit.fail(
                "joint_safety_artifact_receipt",
                "joint-safety update {} has an invalid artifact receipt".format(update),
            )
        elif run_dir is None:
            audit.fail(
                "joint_safety_artifact_unverified",
                "--run-dir is required to hash joint-safety sidecar bytes",
            )
        else:
            candidate = _safe_artifact_path(run_dir, artifact.get("path"))
            if (
                candidate is None
                or not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != artifact["size_bytes"]
                or _sha256_file(candidate) != artifact["sha256"]
            ):
                audit.fail(
                    "joint_safety_artifact_bytes",
                    "joint-safety update {} sidecar bytes do not match receipt".format(
                        update
                    ),
                )
            else:
                try:
                    payload = sidecar_loader(candidate)
                    decoded = _validate_joint_sidecar_payload(
                        payload,
                        record=record,
                        manifest=manifest,
                    )
                    _validate_optimizer_commit(
                        record=record,
                        artifact=artifact,
                        run_dir=run_dir,
                        expected_rank=decoded["rank"],
                    )
                except Exception as exc:
                    audit.fail(
                        "joint_safety_artifact_content",
                        "joint-safety update {} sidecar/commit content rejected: {}: {}".format(
                            update, type(exc).__name__, exc
                        ),
                    )
                else:
                    if bound_rank is None:
                        bound_rank = decoded["rank"]
                    elif decoded["rank"] != bound_rank:
                        audit.fail(
                            "joint_safety_rank_drift",
                            "joint-safety sidecar rank changed within one run",
                        )
                    if (
                        previous_consume_sequence is not None
                        and decoded["consume_sequence"]
                        != previous_consume_sequence + 1
                    ):
                        audit.fail(
                            "joint_safety_consume_sequence",
                            "joint-safety consume sequence is not contiguous",
                        )
                    previous_consume_sequence = decoded["consume_sequence"]
                    if decoded["counter_totals"] != counters:
                        audit.fail(
                            "joint_safety_artifact_counter_binding",
                            "joint-safety update {} decoded totals drifted".format(
                                update
                            ),
                        )
                    by_update[update] = decoded
    passed = len(audit.failures) == before
    audit.check(
        "joint_safety_sidecars",
        passed,
        "{} update sidecar(s) decoded, identity/counter checked, and optimizer-committed".format(
            len(records)
        )
        if passed
        else "joint-safety update receipts/sidecar bytes are incomplete",
    )
    return {"updates": updates, "totals": totals, "by_update": by_update}


def _validate_per_action_activation(events, recipe, manifest, activation, audit):
    records = _group_event(events, PER_ACTION_EVENT)
    if not records:
        audit.check(
            "per_action_reward_accounting",
            False,
            "missing {}: aggregate Reward activation cannot prove per-action behavior".format(
                PER_ACTION_EVENT
            ),
        )
        return None
    before = len(audit.failures)
    updates = _validate_update_sequence(
        records, field="ppo_update", label="per_action_reward", audit=audit
    )
    expected_updates = () if activation is None else activation["updates"]
    if updates != expected_updates:
        audit.fail(
            "per_action_update_coverage",
            "per-action Reward updates {} differ from activation updates {}".format(
                updates, expected_updates
            ),
        )
    by_update = {}
    expected_names = tuple(sorted(recipe["terms"]))
    try:
        expected_taxonomy = (
            REWARD_TAXONOMY.build_action_ball_reward_group_taxonomy(
                [recipe["terms"][name] for name in expected_names]
            )
        )
    except Exception as exc:
        audit.fail(
            "reward_group_taxonomy",
            "effective composed recipe cannot map to authoritative Reward "
            "taxonomy: {}: {}".format(type(exc).__name__, exc),
        )
        return None
    taxonomy_by_name = {
        row["name"]: row for row in expected_taxonomy["active_terms"]
    }
    group_order = tuple(expected_taxonomy["group_order"])
    for record in records:
        update = record.get("ppo_update")
        aggregate = (
            None
            if activation is None
            else activation["records"].get(update)
        )
        if (
            record.get("schema_version") != PER_ACTION_SCHEMA_VERSION
            or record.get("recipe_sha256") != recipe["sha256"]
            or record.get("task_kind") != "action_ball"
            or aggregate is None
            or record.get("reward_group_taxonomy") != expected_taxonomy
            or not _close(
                record.get("step_dt_s"),
                activation["step_dt_s"],
                rel_tol=0.0,
            )
        ):
            audit.fail(
                "per_action_header",
                "per-action Reward update {} has an invalid "
                "recipe/task/dt/taxonomy binding".format(update),
            )
            continue
        if (
            manifest is None
            or tuple(record.get("action_order", ())) != manifest["order"]
            or record.get("manifest_sha256") != manifest["file_sha256"]
        ):
            audit.fail(
                "per_action_order",
                "per-action Reward update {} is not exact-manifest ordered/bound".format(
                    update
                ),
            )
            continue
        rows = record.get("actions")
        if not isinstance(rows, list) or len(rows) != len(manifest["order"]):
            audit.fail(
                "per_action_rows",
                "per-action Reward update {} lacks one row per action".format(update),
            )
            continue
        row_map = {}
        for index, row in enumerate(rows):
            action_id = None if not isinstance(row, dict) else row.get("action_id")
            action_uid = None if not isinstance(row, dict) else row.get("action_uid")
            if (
                action_id != manifest["order"][index]
                or action_uid != manifest["uids"].get(action_id)
                or action_id in row_map
                or not _nonnegative_int(row.get("observed_sample_count"))
            ):
                audit.fail(
                    "per_action_identity",
                    "per-action Reward row {} at update {} has invalid identity/count".format(
                        index, update
                    ),
                )
                continue
            term_rows = row.get("terms")
            if not isinstance(term_rows, list):
                audit.fail(
                    "per_action_terms",
                    "per-action Reward row {!r} has no term list".format(action_id),
                )
                continue
            terms = {}
            for term in term_rows:
                if not isinstance(term, dict) or type(term.get("name")) is not str:
                    audit.fail(
                        "per_action_term_invalid",
                        "per-action Reward row {!r} has malformed term".format(
                            action_id
                        ),
                    )
                    continue
                if term["name"] in terms:
                    audit.fail(
                        "per_action_term_duplicate",
                        "per-action Reward row {!r} duplicates term {!r}".format(
                            action_id, term["name"]
                        ),
                    )
                    continue
                terms[term["name"]] = term
            if tuple(sorted(terms)) != expected_names:
                audit.fail(
                    "per_action_term_set",
                    "per-action Reward row {!r} term set differs from recipe".format(
                        action_id
                    ),
                )
                continue
            for name, term in terms.items():
                observed = row["observed_sample_count"]
                nonzero = term.get("nonzero_sample_count")
                raw_sum = term.get("raw_sum")
                weighted_sum = term.get("weighted_sum")
                if (
                    term.get("observed_sample_count") != observed
                    or not _nonnegative_int(nonzero)
                    or nonzero > observed
                    or not _finite_number(raw_sum)
                    or not _finite_number(weighted_sum)
                ):
                    audit.fail(
                        "per_action_term_fields",
                        "per-action Reward {!r}/{!r} has incomplete fields".format(
                            action_id, name
                        ),
                    )
                    continue
                expected_weighted = (
                    float(raw_sum)
                    * float(recipe["terms"][name]["weight"])
                    * float(activation["step_dt_s"])
                )
                if not _close(weighted_sum, expected_weighted):
                    audit.fail(
                        "per_action_raw_weight_dt",
                        "per-action Reward {!r}/{!r} violates raw*weight*dt".format(
                            action_id, name
                        ),
                    )
            group_rows = row.get("reward_groups")
            if (
                not isinstance(group_rows, list)
                or len(group_rows) != len(group_order)
                or any(
                    not isinstance(group, dict)
                    or set(group) != REWARD_GROUP_FIELDS
                    for group in group_rows
                )
                or tuple(
                    group.get("group")
                    for group in group_rows
                    if isinstance(group, dict)
                )
                != group_order
            ):
                audit.fail(
                    "per_action_reward_groups",
                    "per-action Reward row {!r} lacks exact ordered group "
                    "statistics".format(action_id),
                )
                continue
            group_map = {}
            partition = set()
            total_positive = 0.0
            total_negative = 0.0
            positive_fractions = []
            negative_fractions = []
            groups_valid = True
            for group in group_rows:
                group_name = group["group"]
                expected_objectives = sorted(
                    name
                    for name, taxonomy in taxonomy_by_name.items()
                    if taxonomy["group"] == group_name
                    and taxonomy["role"] == "objective"
                )
                expected_probes = sorted(
                    name
                    for name, taxonomy in taxonomy_by_name.items()
                    if taxonomy["group"] == group_name
                    and taxonomy["role"] == "diagnostic_probe"
                )
                objectives = group["objective_term_names"]
                probes = group["diagnostic_probe_term_names"]
                eligible = group["eligible_sample_count"]
                nonzero = group["nonzero_sample_count"]
                expected_eligible = (
                    row["observed_sample_count"]
                    if expected_objectives
                    else 0
                )
                quantiles = (
                    group["weighted_p5"],
                    group["weighted_p50"],
                    group["weighted_p95"],
                )
                if (
                    objectives != expected_objectives
                    or probes != expected_probes
                    or set(objectives) & set(probes)
                    or group["eligibility"]
                    != "reward_manager_evaluated_active_group_terms"
                    or not _nonnegative_int(eligible)
                    or eligible != expected_eligible
                    or not _nonnegative_int(nonzero)
                    or nonzero > eligible
                    or not _finite_number(group["weighted_sum"])
                    or not _finite_number(group["positive_weighted_sum"])
                    or float(group["positive_weighted_sum"]) < 0.0
                    or not _finite_number(group["negative_weighted_sum"])
                    or float(group["negative_weighted_sum"]) > 0.0
                ):
                    groups_valid = False
                    break
                if eligible:
                    if (
                        any(not _finite_number(value) for value in quantiles)
                        or not (
                            float(quantiles[0])
                            <= float(quantiles[1])
                            <= float(quantiles[2])
                        )
                    ):
                        groups_valid = False
                        break
                elif quantiles != (None, None, None):
                    groups_valid = False
                    break
                expected_weighted = sum(
                    float(terms[name]["weighted_sum"])
                    for name in expected_objectives
                )
                if (
                    not _close(group["weighted_sum"], expected_weighted)
                    or not _close(
                        group["weighted_sum"],
                        float(group["positive_weighted_sum"])
                        + float(group["negative_weighted_sum"]),
                    )
                ):
                    groups_valid = False
                    break
                partition.update(objectives)
                partition.update(probes)
                total_positive += float(group["positive_weighted_sum"])
                total_negative += float(group["negative_weighted_sum"])
                positive_fractions.append(group["positive_return_fraction"])
                negative_fractions.append(group["negative_return_fraction"])
                group_map[group_name] = group
            if (
                not groups_valid
                or partition != set(expected_names)
                or tuple(group_map) != group_order
                or not _finite_number(row.get("positive_weighted_sum"))
                or not _finite_number(row.get("negative_weighted_sum"))
                or not _close(row["positive_weighted_sum"], total_positive)
                or not _close(row["negative_weighted_sum"], total_negative)
            ):
                audit.fail(
                    "per_action_reward_group_closure",
                    "per-action Reward groups for {!r} do not partition terms, "
                    "signed sums, or quantiles".format(action_id),
                )
                continue
            fractions_valid = True
            for total, fractions in (
                (total_positive, positive_fractions),
                (total_negative, negative_fractions),
            ):
                if total == 0.0:
                    fractions_valid &= all(
                        value is None for value in fractions
                    )
                else:
                    fractions_valid &= (
                        all(
                            _finite_number(value)
                            and 0.0 <= float(value) <= 1.0
                            for value in fractions
                        )
                        and _close(sum(float(value) for value in fractions), 1.0)
                    )
            if not fractions_valid:
                audit.fail(
                    "per_action_reward_group_fraction",
                    "per-action positive/negative Reward group fractions for "
                    "{!r} do not close to one".format(action_id),
                )
                continue
            row_map[action_id] = {
                "observed_sample_count": row["observed_sample_count"],
                "terms": terms,
                "positive_weighted_sum": row["positive_weighted_sum"],
                "negative_weighted_sum": row["negative_weighted_sum"],
                "reward_groups": group_map,
            }
        if tuple(row_map) != manifest["order"]:
            continue
        aggregate_observed = aggregate["record"]["observed_sample_count"]
        if sum(row["observed_sample_count"] for row in row_map.values()) != (
            aggregate_observed
        ):
            audit.fail(
                "per_action_sample_union",
                "per-action observed samples do not partition aggregate update {}".format(
                    update
                ),
            )
        for name in expected_names:
            aggregate_term = aggregate["terms"][name]
            for field in ("observed_sample_count", "nonzero_sample_count"):
                total = sum(
                    row["terms"][name][field] for row in row_map.values()
                )
                if total != aggregate_term[field]:
                    audit.fail(
                        "per_action_{}_union".format(field),
                        "per-action {} for {!r} does not close at update {}".format(
                            field, name, update
                        ),
                    )
            for field in ("raw_sum", "weighted_sum"):
                total = sum(
                    float(row["terms"][name][field]) for row in row_map.values()
                )
                if not _close(total, aggregate_term[field]):
                    audit.fail(
                        "per_action_{}_union".format(field),
                        "per-action {} for {!r} does not close at update {}".format(
                            field, name, update
                        ),
                    )
        by_update[update] = row_map
    passed = len(audit.failures) == before and len(by_update) == len(records)
    audit.check(
        "per_action_reward_accounting",
        passed,
        "{} update(s) partition every Reward term across {} actions".format(
            len(records), 0 if manifest is None else len(manifest["order"])
        )
        if passed
        else "per-action Reward rows/groups are missing or do not close to "
        "aggregate activation/taxonomy",
    )
    return {
        "updates": updates,
        "by_update": by_update,
        "reward_group_taxonomy": expected_taxonomy,
    }


def _termination_reason_classes(terms):
    term_set = set(terms)
    classes = []
    if "robot_hit_table" in term_set:
        classes.append("table_hit")
    if term_set & {"base_fell_tilt", "base_too_low"}:
        classes.append("fall")
    if term_set & {"joint_qdes_forbidden", "joint_actual_forbidden"}:
        classes.append("hard_limit")
    if term_set & set(REFERENCE_ENVELOPE_TERMINATION_TERMS):
        classes.append("reference_envelope")
    recognized = set(HARD_SAFETY_TERMINATION_TERMS) | set(
        REFERENCE_ENVELOPE_TERMINATION_TERMS
    )
    if term_set - recognized:
        classes.append("other_termination")
    return classes


def _primary_termination_reason_class(classes):
    for name in (
        "table_hit",
        "fall",
        "hard_limit",
        "reference_envelope",
        "other_termination",
    ):
        if name in classes:
            return name
    return None


def _raw_action_ball_safety_counters(terms):
    """Return every raw episode-sticky safety counter for one closure.

    Closure outcome classification may use a deterministic precedence, but
    formal safety gates may not: a joint-limit event that also hits the table
    must increment both raw counters so table zero-tolerance cannot be hidden.
    """

    term_set = set(terms)
    counters = []
    if "robot_hit_table" in term_set:
        counters.append("U_table")
    if term_set & {"base_fell_tilt", "base_too_low"}:
        counters.append("U_fall")
    if "joint_qdes_forbidden" in term_set:
        counters.append("U_joint_qdes")
    if "joint_actual_forbidden" in term_set:
        counters.append("U_joint_actual")
    return tuple(counters)


def _safety_transition_id(ppo_update, transition):
    identity = {
        "ppo_update": ppo_update,
        "common_step_counter": transition["common_step_counter"],
        "joint_policy_step_sequence": transition[
            "joint_policy_step_sequence"
        ],
        "env_id": transition["env_id"],
        "action_uid": transition["action_uid"],
        "reset_generation": transition["reset_generation"],
        "swing_generation": transition["swing_generation"],
        "birth_receipt_sha256": transition["birth_receipt_sha256"],
        "termination_terms": list(transition["termination_terms"]),
        "rising_termination_terms": list(
            transition["rising_termination_terms"]
        ),
        "pre_terminal_reason_mask": transition[
            "pre_terminal_reason_mask"
        ],
        "post_terminal_reason_mask": transition[
            "post_terminal_reason_mask"
        ],
    }
    return _sha256_bytes(_canonical_json(identity).encode("utf-8"))


def _validate_safety_transitions(
    events,
    recipe,
    manifest,
    activation,
    per_action,
    action_ledger,
    joint_safety,
    audit,
):
    records = _group_event(events, SAFETY_TRANSITION_EVENT)
    if not records:
        audit.check(
            "negative_reward_semantics",
            False,
            "missing {}: aggregate counters cannot prove terminal-event causality or no-double-charge".format(
                SAFETY_TRANSITION_EVENT
            ),
        )
        return None
    before = len(audit.failures)
    soft_terms = {
        name: recipe["terms"].get(name)
        for name in SOFT_LIMIT_TERM_NAMES
    }
    death = recipe["terms"].get("death_penalty")
    for name, term in soft_terms.items():
        params = None if term is None else term.get("params")
        if (
            term is None
            or not _close(
                term["weight"],
                ADOPTED_SOFT_LIMIT_WEIGHT,
                rel_tol=0.0,
            )
            or term["callable"].rsplit(".", 1)[-1]
            != SOFT_LIMIT_CALLABLES[name]
            or not isinstance(params, dict)
        ):
            audit.fail(
                "soft_limit_recipe",
                "recipe lacks adopted {}/{} at weight {}".format(
                    name,
                    SOFT_LIMIT_CALLABLES[name],
                    ADOPTED_SOFT_LIMIT_WEIGHT,
                ),
            )
            continue
        expected_param_keys = (
            {"action_name", "margin_frac", "penalty_floor"}
            if name == "qdes_limit_barrier"
            else {
                "asset_cfg",
                "margin_frac",
                "penalty_floor",
                "expected_joint_count",
            }
        )
        if (
            set(params) != expected_param_keys
            or not _close(
                params.get("margin_frac"),
                ADOPTED_SOFT_LIMIT_MARGIN_FRAC,
                rel_tol=0.0,
            )
            or not _close(
                params.get("penalty_floor"),
                ADOPTED_SOFT_LIMIT_PENALTY_FLOOR,
                rel_tol=0.0,
            )
            or (
                name == "qdes_limit_barrier"
                and params.get("action_name") != "joint_pos"
            )
            or (
                name == "joint_limit"
                and params.get("expected_joint_count")
                != ADOPTED_SOFT_LIMIT_MAX_JOINTS
            )
        ):
            audit.fail(
                "soft_limit_recipe_params",
                "recipe term {!r} does not bind the adopted 0.08-margin/"
                "0.25-floor 31-joint contract".format(name),
            )
    if death is None or not _close(
        death["weight"], ADOPTED_DEATH_WEIGHT, rel_tol=0.0
    ):
        audit.fail(
            "death_recipe",
            "recipe lacks adopted death_penalty weight {}".format(
                ADOPTED_DEATH_WEIGHT
            ),
        )
    generic_terminal_terms = [
        term["name"]
        for term in recipe["terms"].values()
        if term["callable"].rsplit(".", 1)[-1] == "is_terminated"
    ]
    hard_safety_terminal_terms = [
        term["name"]
        for term in recipe["terms"].values()
        if term["callable"].rsplit(".", 1)[-1]
        == "action_ball_safety_terminated"
    ]
    death_params = None if death is None else death.get("params")
    if (
        generic_terminal_terms
        or hard_safety_terminal_terms != ["death_penalty"]
        or not isinstance(death_params, dict)
        or tuple(death_params.get("term_names", ()))
        != HARD_SAFETY_TERMINATION_TERMS
    ):
        audit.fail(
            "generic_death_term_count",
            "death_penalty must be the sole exact hard-safety-union Reward; "
            "generic={} hard_safety={}".format(
                generic_terminal_terms, hard_safety_terminal_terms
            ),
        )
    terminal_specific = [
        term["name"]
        for term in recipe["terms"].values()
        if term["name"] == "table_hit_penalty"
        or term["callable"].rsplit(".", 1)[-1] == "terminated_by_term"
    ]
    if terminal_specific:
        audit.fail(
            "terminal_specific_penalty_active",
            "reason-specific terminal penalties would stack generic death: {}".format(
                terminal_specific
            ),
        )
    if any(term is None for term in soft_terms.values()) or death is None:
        audit.check(
            "negative_reward_semantics",
            False,
            "negative Reward recipe prerequisites are absent",
        )
        return None
    updates = _validate_update_sequence(
        records, field="ppo_update", label="safety_transition", audit=audit
    )
    expected_updates = () if activation is None else activation["updates"]
    if updates != expected_updates:
        audit.fail(
            "safety_transition_update_coverage",
            "safety transition updates {} differ from activation updates {}".format(
                updates, expected_updates
            ),
        )
    coverage = {
        name: 0
        for name in (
            "soft_limit",
            "hard_limit",
            "table_hit",
            "fall",
            "pure_hard_limit",
            "pure_table_hit",
            "pure_fall",
            "reference_envelope",
            "reference_envelope_without_death",
        )
    }
    soft_coverage = {
        name: {
            "active": 0,
            "nonterminal_active": 0,
            "zero_outside_band": 0,
        }
        for name in SOFT_LIMIT_TERM_NAMES
    }
    terminal_by_action = (
        {}
        if manifest is None
        else {
            name: {
                "U_table": 0,
                "U_fall": 0,
                "U_collision": 0,
                "U_joint_qdes": 0,
                "U_joint_actual": 0,
                "closed_unsafe": 0,
            }
            for name in manifest["order"]
        }
    )
    seen_transition_ids = set()
    for record in records:
        update = record.get("ppo_update")
        if (
            record.get("schema_version") != SAFETY_TRANSITION_SCHEMA_VERSION
            or record.get("recipe_sha256") != recipe["sha256"]
            or record.get("coverage") != "complete_update"
            or activation is None
            or per_action is None
            or update not in per_action["by_update"]
            or manifest is None
            or record.get("manifest_sha256") != manifest["file_sha256"]
            or tuple(record.get("action_order", ())) != manifest["order"]
            or tuple(record.get("soft_limit_term_names", ()))
            != SOFT_LIMIT_TERM_NAMES
            or tuple(record.get("hard_safety_termination_term_names", ()))
            != HARD_SAFETY_TERMINATION_TERMS
            or tuple(
                record.get("reference_envelope_termination_term_names", ())
            )
            != REFERENCE_ENVELOPE_TERMINATION_TERMS
            or not isinstance(record.get("termination_term_order"), list)
            or len(record["termination_term_order"])
            != len(set(record["termination_term_order"]))
            or not set(REQUIRED_TERMINATION_TERM_ORDER).issubset(
                set(record["termination_term_order"])
            )
            or not _close(
                record.get("step_dt_s"),
                activation["step_dt_s"],
                rel_tol=0.0,
            )
        ):
            audit.fail(
                "safety_transition_header",
                "safety transition update {} has invalid recipe/dt/coverage binding".format(
                    update
                ),
            )
            continue
        action_rows = per_action["by_update"][update]
        soft_rows = record.get("soft_limit_by_action_term")
        expected_soft_keys = [
            (action_id, term_name)
            for action_id in manifest["order"]
            for term_name in SOFT_LIMIT_TERM_NAMES
        ]
        if (
            not isinstance(soft_rows, list)
            or len(soft_rows) != len(expected_soft_keys)
        ):
            audit.fail(
                "soft_limit_rows",
                "safety update {} lacks exact action-by-two-term soft-limit rows".format(
                    update
                ),
            )
            continue
        for index, row in enumerate(soft_rows):
            action_id = None if not isinstance(row, dict) else row.get("action_id")
            term_name = None if not isinstance(row, dict) else row.get("term_name")
            if (
                (action_id, term_name) != expected_soft_keys[index]
                or row.get("action_uid") != manifest["uids"].get(action_id)
                or not _nonnegative_int(row.get("observed_sample_count"))
                or not _nonnegative_int(row.get("eligible_sample_count"))
                or not _nonnegative_int(row.get("active_sample_count"))
                or not _nonnegative_int(
                    row.get("terminated_active_sample_count")
                )
                or not _finite_number(row.get("raw_sum"))
                or not _finite_number(row.get("weighted_sum"))
                or not _close(
                    row.get("step_dt_s"),
                    activation["step_dt_s"],
                    rel_tol=0.0,
                )
                or row.get("effective") is not True
                or row.get("terminal_reward") is not False
            ):
                audit.fail(
                    "soft_limit_row_invalid",
                    "soft-limit row {} at update {} is incomplete".format(index, update),
                )
                continue
            term_row = action_rows[action_id]["terms"][term_name]
            if (
                row["observed_sample_count"]
                != action_rows[action_id]["observed_sample_count"]
                or row["eligible_sample_count"] != row["observed_sample_count"]
                or row["active_sample_count"]
                != term_row["nonzero_sample_count"]
                or row["terminated_active_sample_count"]
                > row["active_sample_count"]
                or not _close(row["raw_sum"], term_row["raw_sum"])
                or not _close(row["weighted_sum"], term_row["weighted_sum"])
                or float(row["raw_sum"]) < 0.0
                or float(row["weighted_sum"]) > 0.0
                or (
                    row["active_sample_count"] > 0
                    and float(row["raw_sum"])
                    + 1.0e-7
                    < (
                        row["active_sample_count"]
                        * ADOPTED_SOFT_LIMIT_PENALTY_FLOOR
                    )
                )
                or float(row["raw_sum"])
                > (
                    row["active_sample_count"]
                    * ADOPTED_SOFT_LIMIT_MAX_JOINTS
                    + 1.0e-7
                )
            ):
                audit.fail(
                    "soft_limit_activation_binding",
                    "soft-limit row {!r}/{!r} does not bind negative nonterminal activation".format(
                        action_id, term_name
                    ),
                )
            coverage["soft_limit"] += row["active_sample_count"]
            soft_coverage[term_name]["active"] += row[
                "active_sample_count"
            ]
            soft_coverage[term_name]["nonterminal_active"] += (
                row["active_sample_count"]
                - row["terminated_active_sample_count"]
            )
            # The bound v2 kernel is exactly zero iff no joint intrudes into
            # the soft band.  The nonzero count therefore leaves an explicit
            # observed zero-output interior/control denominator.
            soft_coverage[term_name]["zero_outside_band"] += (
                row["observed_sample_count"] - row["active_sample_count"]
            )

        transitions = record.get("terminal_transitions")
        if not isinstance(transitions, list):
            audit.fail(
                "terminal_transition_rows",
                "safety update {} terminal_transitions is not a list".format(update),
            )
            continue
        grouped = {
            action_id: {"count": 0, "raw_sum": 0.0, "weighted_sum": 0.0}
            for action_id in manifest["order"]
        }
        termination_order = record["termination_term_order"]
        for transition in transitions:
            if not isinstance(transition, dict):
                audit.fail(
                    "terminal_transition_invalid",
                    "safety update {} contains a malformed transition".format(update),
                )
                continue
            transition_id = transition.get("transition_id")
            action_id = transition.get("action_id")
            classes = transition.get("reason_classes")
            terms = transition.get("termination_terms")
            rising_terms = transition.get("rising_termination_terms")
            pre_masks = transition.get("pre_terminal_reason_mask")
            post_masks = transition.get("post_terminal_reason_mask")
            death_activation = transition.get("death_activation")
            if (
                type(transition_id) is not str
                or not transition_id
                or transition_id in seen_transition_ids
                or action_id not in manifest["uids"]
                or transition.get("action_uid") != manifest["uids"][action_id]
                or not isinstance(classes, list)
                or not classes
                or len(classes) != len(set(classes))
                or any(value not in TERMINATION_REASON_CLASSES for value in classes)
                or transition.get("primary_reason_class")
                not in TERMINATION_REASON_CLASSES
                or not isinstance(terms, list)
                or not terms
                or any(type(value) is not str or not value for value in terms)
                or terms != sorted(set(terms))
                or not isinstance(rising_terms, list)
                or not rising_terms
                or rising_terms != sorted(set(rising_terms))
                or not set(rising_terms).issubset(set(terms))
                or not isinstance(pre_masks, dict)
                or set(pre_masks) != set(termination_order)
                or any(type(value) is not bool for value in pre_masks.values())
                or not isinstance(post_masks, dict)
                or set(post_masks) != set(termination_order)
                or any(type(value) is not bool for value in post_masks.values())
                or not _nonnegative_int(transition.get("env_id"))
                or transition["env_id"] >= activation["records"][update]["record"][
                    "num_envs"
                ]
                or not _nonnegative_int(transition.get("common_step_counter"))
                or not _nonnegative_int(
                    transition.get("joint_policy_step_sequence")
                )
                or not _nonnegative_int(transition.get("reset_generation"))
                or not _nonnegative_int(transition.get("swing_generation"))
                or not _is_sha256(transition.get("birth_receipt_sha256"))
                or type(transition.get("timed_out_same_step")) is not bool
                or transition.get("reason_specific_penalties") != []
                or not isinstance(death_activation, dict)
                or death_activation.get("term_name") != "death_penalty"
                or death_activation.get("eligible") is not True
                or type(death_activation.get("active")) is not bool
                or not _close(
                    death_activation.get("weighted"),
                    transition.get("death_weighted_contribution"),
                )
                or not _close(
                    death_activation.get("step_dt_s"),
                    activation["step_dt_s"],
                    rel_tol=0.0,
                )
                or death_activation.get("effective") is not True
            ):
                audit.fail(
                    "terminal_transition_fields",
                    "safety update {} has invalid/duplicate/unbound terminal transition".format(
                        update
                    ),
                )
                continue
            true_post_terms = sorted(
                name for name, active in post_masks.items() if active
            )
            expected_rising_terms = sorted(
                name
                for name in true_post_terms
                if not pre_masks[name]
            )
            expected_classes = _termination_reason_classes(terms)
            expected_primary = _primary_termination_reason_class(
                expected_classes
            )
            expected_hard_safety = any(
                name in expected_classes
                for name in ("table_hit", "fall", "hard_limit")
            )
            expected_death_raw = 1.0 if expected_hard_safety else 0.0
            if (
                true_post_terms != terms
                or expected_rising_terms != rising_terms
                or classes != expected_classes
                or transition.get("primary_reason_class")
                != expected_primary
                or death_activation.get("active") is not expected_hard_safety
                or not _close(
                    transition.get("death_raw_value"), expected_death_raw
                )
                or not _close(
                    death_activation.get("raw"), expected_death_raw
                )
            ):
                audit.fail(
                    "termination_reason_mapping",
                    "transition {!r} reason classes/masks/rising edge do not derive from terms".format(
                        transition_id
                    ),
                )
                continue
            expected_transition_id = _safety_transition_id(update, transition)
            if transition_id != expected_transition_id:
                audit.fail(
                    "terminal_transition_id_binding",
                    "transition id does not bind update/env/action/generations/reasons",
                )
                continue
            activation_record = activation["records"][update]["record"]
            common_step_counter = transition["common_step_counter"]
            if not (
                activation_record["common_step_counter_start"]
                < common_step_counter
                <= activation_record["common_step_counter_end"]
            ):
                audit.fail(
                    "terminal_transition_step_window",
                    "transition common_step_counter is outside its activation update",
                )
                continue
            sidecar_update = (
                None
                if joint_safety is None
                else joint_safety.get("by_update", {}).get(update)
            )
            archive = (
                None
                if sidecar_update is None
                else sidecar_update["terminal_archives"].get(
                    (
                        transition["joint_policy_step_sequence"],
                        transition["env_id"],
                    )
                )
            )
            if (
                archive is None
                or archive["action_episode_sequence"]
                != transition["reset_generation"]
                or archive["action_uid"] != transition["action_uid"]
                or archive["swing_generation"]
                != transition["swing_generation"]
                or archive["birth_receipt_sha256"]
                != transition["birth_receipt_sha256"]
                or archive["terminated"] is not True
                or archive["timed_out"]
                is not transition["timed_out_same_step"]
                or (
                    "hard_limit" in expected_classes
                    and (
                        archive["hard_event"] is not True
                        or "unsafe" not in archive["reasons"]
                    )
                )
            ):
                audit.fail(
                    "terminal_transition_joint_archive_binding",
                    "transition does not bind the same joint-safety terminal identity",
                )
                continue
            seen_transition_ids.add(transition_id)
            expected_weighted = (
                float(death["weight"])
                * float(activation["step_dt_s"])
                * expected_death_raw
            )
            if (
                not _close(
                    activation["step_dt_s"],
                    ADOPTED_POLICY_DT_S,
                    rel_tol=0.0,
                )
                or not _close(
                    transition.get("death_weighted_contribution"),
                    expected_weighted,
                )
            ):
                audit.fail(
                    "death_once_per_transition",
                    "transition {!r} hard-safety eligibility does not match "
                    "the adopted {}*{} charge".format(
                        transition_id,
                        ADOPTED_DEATH_WEIGHT,
                        ADOPTED_POLICY_DT_S,
                    ),
                )
            grouped[action_id]["count"] += int(expected_hard_safety)
            grouped[action_id]["raw_sum"] += expected_death_raw
            grouped[action_id]["weighted_sum"] += expected_weighted
            raw_safety_counters = _raw_action_ball_safety_counters(terms)
            terminal_by_action[action_id]["closed_unsafe"] += int(
                expected_hard_safety
            )
            for counter_name in raw_safety_counters:
                terminal_by_action[action_id][counter_name] += 1
            if "table_hit" in classes:
                coverage["table_hit"] += 1
            if "fall" in classes:
                coverage["fall"] += 1
            if "hard_limit" in classes:
                coverage["hard_limit"] += 1
            if "reference_envelope" in classes:
                coverage["reference_envelope"] += 1
                if not expected_hard_safety:
                    coverage["reference_envelope_without_death"] += 1
            if classes == ["hard_limit"]:
                coverage["pure_hard_limit"] += 1
            if classes == ["table_hit"]:
                coverage["pure_table_hit"] += 1
            if classes == ["fall"]:
                coverage["pure_fall"] += 1
        for action_id, grouped_row in grouped.items():
            death_row = action_rows[action_id]["terms"]["death_penalty"]
            if (
                grouped_row["count"] != death_row["nonzero_sample_count"]
                or not _close(grouped_row["raw_sum"], death_row["raw_sum"])
                or not _close(
                    grouped_row["weighted_sum"], death_row["weighted_sum"]
                )
            ):
                audit.fail(
                    "death_transition_activation_binding",
                    "terminal transitions for {!r} do not close to death activation".format(
                        action_id
                    ),
                )

    for term_name, row in soft_coverage.items():
        if row["active"] <= 0:
            audit.fail(
                "{}_trigger_unproven".format(term_name),
                "no nonzero {} activation was observed; zero is not "
                "activation proof".format(term_name),
            )
        if row["nonterminal_active"] <= 0:
            audit.fail(
                "{}_nonterminal_trigger_unproven".format(term_name),
                "{} was never nonzero on a nonterminal sample".format(
                    term_name
                ),
            )
        if row["zero_outside_band"] <= 0:
            audit.fail(
                "{}_interior_zero_unproven".format(term_name),
                "{} has no observed zero-output interior/control sample".format(
                    term_name
                ),
            )
    for name in ("soft_limit", "hard_limit", "table_hit", "fall"):
        if coverage[name] <= 0:
            audit.fail(
                "{}_trigger_unproven".format(name),
                "no event-bound {} trigger was observed; zero is not activation proof".format(
                    name
                ),
            )
    for name in ("hard_limit", "table_hit", "fall"):
        if coverage["pure_{}".format(name)] <= 0:
            audit.fail(
                "{}_isolated_trigger_unproven".format(name),
                "no isolated {} transition was observed; a simultaneous "
                "multi-reason death cannot prove separate accounting".format(
                    name
                ),
            )
    if action_ledger is None or action_ledger["last"] is None:
        audit.fail(
            "safety_episode_binding_missing",
            "terminal transition transcript cannot be checked against ActionBall outcomes",
        )
    elif manifest is not None:
        for action_id in manifest["order"]:
            transcript = terminal_by_action[action_id]
            ledger = action_ledger["last"][action_id]
            if (
                transcript["closed_unsafe"]
                != ledger["C"] - ledger["L"] - ledger["F"]
                or any(
                    transcript[name] != ledger[name]
                    for name in (
                        "U_table",
                        "U_fall",
                        "U_collision",
                        "U_joint_qdes",
                        "U_joint_actual",
                    )
                )
            ):
                audit.fail(
                    "safety_action_outcome_binding",
                    "terminal transcript for {!r} does not close once per "
                    "unsafe attempt while preserving every overlapping raw "
                    "U_* safety signal".format(action_id),
                )
    if joint_safety is None:
        audit.fail(
            "hard_limit_sidecar_binding",
            "hard-limit trigger has no joint-safety sidecar evidence",
        )
    elif coverage["hard_limit"] > 0 and (
        joint_safety["totals"]["policy_crossing_joint_count"]
        + joint_safety["totals"]["substep_hard_crossing_joint_count"]
        + joint_safety["totals"]["actual_hard_edge_joint_count"]
        <= 0
    ):
        audit.fail(
            "hard_limit_counter_zero",
            "hard-limit terminal transcript has no crossing/actual-edge sidecar counter",
        )
    passed = len(audit.failures) == before
    audit.check(
        "negative_reward_semantics",
        passed,
        "soft qbar is nonterminal; hard/table/fall each trigger; death is exactly once; no stacked term"
        if passed
        else "negative Reward triggers lack event binding, coverage, conservation, or no-double-charge proof",
    )
    return {
        "updates": updates,
        "coverage": coverage,
        "soft_limit_coverage": soft_coverage,
    }


def audit_reward_run(
    *,
    recipe_path,
    event_paths,
    manifest_path=None,
    run_dir=None,
    joint_sidecar_loader=None,
):
    audit = Audit()
    try:
        event_paths = tuple(event_paths)
    except Exception as exc:
        audit.check(
            "event_stream_integrity",
            False,
            "event_paths is not an iterable of paths: {}: {}".format(
                type(exc).__name__, exc
            ),
        )
        event_paths = ()
    recipe = _guarded_call(
        audit, "recipe_integrity", _load_recipe, recipe_path, audit
    )
    manifest = _guarded_call(
        audit, "manifest_action_identity", _load_manifest, manifest_path, audit
    )
    if recipe is not None:
        _guarded_call(
            audit,
            "run_recipe_binding",
            _validate_run_recipe_binding,
            recipe_path,
            run_dir,
            recipe,
            audit,
        )
    events = _guarded_call(
        audit, "event_stream_integrity", _event_lines, event_paths, audit
    )
    if events is None:
        events = []
    known_events = {
        ACTIVATION_EVENT,
        PER_ACTION_EVENT,
        SAFETY_TRANSITION_EVENT,
        EPISODE_SEGMENTED_CLOSURE_EVENT,
        JOINT_SAFETY_EVENT,
        ACTION_BALL_LEDGER_EVENT,
        EXACT_BEHAVIOR_EVENT,
    }
    unknown = sorted(
        {
            str(record.get("event"))
            for record in events
            if record.get("event") not in known_events
        }
    )
    if unknown:
        audit.warn(
            "unknown_events_ignored",
            "ignored event kinds: {}".format(", ".join(unknown)),
        )

    activation = None
    episode_closure = None
    action_ledger = None
    joint_safety = None
    per_action = None
    safety = None
    if recipe is not None:
        activation = _guarded_call(
            audit,
            "runtime_activation_integrity",
            _validate_activation,
            events,
            recipe,
            audit,
        )
        episode_closure = _guarded_call(
            audit,
            "episode_segmented_reward_closure",
            _validate_episode_segmented_closure,
            events,
            recipe,
            activation,
            audit,
        )
        action_ledger = _guarded_call(
            audit,
            "curriculum_episode_accounting",
            _validate_action_ledger,
            events,
            manifest,
            activation,
            audit,
        )
        joint_safety = _guarded_call(
            audit,
            "joint_safety_sidecars",
            _validate_joint_safety,
            events,
            activation,
            manifest,
            run_dir,
            audit,
            sidecar_loader=(
                _torch_sidecar_loader
                if joint_sidecar_loader is None
                else joint_sidecar_loader
            ),
        )
        per_action = _guarded_call(
            audit,
            "per_action_reward_accounting",
            _validate_per_action_activation,
            events,
            recipe,
            manifest,
            activation,
            audit,
        )
        safety = _guarded_call(
            audit,
            "negative_reward_semantics",
            _validate_safety_transitions,
            events,
            recipe,
            manifest,
            activation,
            per_action,
            action_ledger,
            joint_safety,
            audit,
        )
    else:
        for name, detail in (
            (
                "runtime_activation_integrity",
                "recipe invalid; runtime activation cannot be bound",
            ),
            (
                "episode_segmented_reward_closure",
                "recipe invalid; episode closure cannot be bound",
            ),
            (
                "curriculum_episode_accounting",
                "recipe invalid; run evidence audit stopped",
            ),
            (
                "joint_safety_sidecars",
                "recipe invalid; run evidence audit stopped",
            ),
            (
                "per_action_reward_accounting",
                "recipe invalid; run evidence audit stopped",
            ),
            (
                "negative_reward_semantics",
                "recipe invalid; run evidence audit stopped",
            ),
        ):
            audit.check(name, False, detail)

    per_action_passed = (
        per_action is not None
        and audit.checks.get("per_action_reward_accounting", {}).get("status")
        == "PASS"
    )
    reward_group_runtime = None
    if per_action_passed:
        taxonomy = per_action["reward_group_taxonomy"]
        group_order = tuple(taxonomy["group_order"])
        reward_group_runtime = {
            "schema_version": 1,
            "reward_group_taxonomy": taxonomy,
            "updates": [
                {
                    "ppo_update": update,
                    "actions": [
                        {
                            "action_id": action_id,
                            "action_uid": manifest["uids"][action_id],
                            "observed_sample_count": row[
                                "observed_sample_count"
                            ],
                            "positive_weighted_sum": row[
                                "positive_weighted_sum"
                            ],
                            "negative_weighted_sum": row[
                                "negative_weighted_sum"
                            ],
                            "reward_groups": [
                                row["reward_groups"][group]
                                for group in group_order
                            ],
                        }
                        for action_id, row in per_action["by_update"][
                            update
                        ].items()
                    ],
                }
                for update in per_action["updates"]
            ],
        }
        reward_group_runtime["sha256"] = _sha256_bytes(
            _canonical_json(reward_group_runtime).encode("utf-8")
        )

    status = "PASS" if not audit.failures else "FAIL_CLOSED"
    isaac_runtime_evidence = (
        status == "PASS"
        and episode_closure is not None
        and episode_closure["all_sources_live"]
        and bool(episode_closure["live_e2_updates"])
        and audit.checks.get(
            "episode_segmented_reward_closure", {}
        ).get("status")
        == "PASS"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "evidence_scope": (
            "offline_validation_of_live_isaac_reward_manager_receipts"
            if isaac_runtime_evidence
            else "offline_artifact_consistency_only"
        ),
        "isaac_runtime_evidence": isaac_runtime_evidence,
        "isaac_runtime_evidence_reason": (
            "validated source-bound RewardManager reset-hook receipts include "
            "at least one completed non-administrative episode segment"
            if isaac_runtime_evidence
            else "no fully valid live RewardManager reset-hook receipt with a "
            "completed non-administrative episode segment was supplied"
        ),
        "inputs": {
            "recipe_path": str(recipe_path),
            "manifest_path": None
            if manifest_path is None
            else str(manifest_path),
            "event_paths": [str(path) for path in event_paths],
            "run_dir": None if run_dir is None else str(run_dir),
        },
        "summary": {
            "recipe_sha256": None if recipe is None else recipe["sha256"],
            "active_reward_term_count": 0
            if recipe is None
            else len(recipe["terms"]),
            "action_count": 0 if manifest is None else len(manifest["order"]),
            "activation_update_count": 0
            if activation is None
            else len(activation["updates"]),
            "episode_closure_update_count": 0
            if episode_closure is None
            else len(episode_closure["updates"]),
            "episode_closure_live_e2_update_count": 0
            if episode_closure is None
            else len(episode_closure["live_e2_updates"]),
            "per_action_available": per_action_passed,
            "negative_semantics_available": safety is not None
            and audit.checks.get("negative_reward_semantics", {}).get("status")
            == "PASS",
        },
        "reward_group_runtime": reward_group_runtime,
        "checks": audit.checks,
        "failures": audit.failures,
        "warnings": audit.warnings,
    }
    report["report_sha256"] = _sha256_bytes(
        _canonical_json(report).encode("utf-8")
    )
    return report


def _write_report(path, report):
    encoded = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        ".{}.{}.tmp".format(output.name, os.getpid())
    )
    if output.exists() or temporary.exists():
        raise AuditInputError(
            "refusing to clobber report path {}".format(output)
        )
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(str(temporary), str(output))
    except FileExistsError as exc:
        raise AuditInputError(
            "refusing to clobber report path {}".format(output)
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        required=True,
        help="params/effective_reward_recipe.json",
    )
    parser.add_argument(
        "--events",
        action="append",
        required=True,
        help="stdout log or JSONL event file; repeatable",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="exact ActionBall manifest used by the run",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="run root used to resolve and hash joint-safety artifact paths",
    )
    parser.add_argument(
        "--output",
        help="new no-clobber JSON report path",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        report = audit_reward_run(
            recipe_path=args.recipe,
            event_paths=args.events,
            manifest_path=args.manifest,
            run_dir=args.run_dir,
        )
        _write_report(args.output, report)
    except Exception as exc:
        print(
            "REWARD_RUN_AUDIT=FAIL_CLOSED {}: {}".format(
                type(exc).__name__, exc
            ),
            file=sys.stderr,
        )
        return 2
    print(
        "REWARD_RUN_AUDIT={status} checks={checks} failures={failures} "
        "evidence_scope={scope} report_sha256={sha}".format(
            status=report["status"],
            checks=len(report["checks"]),
            failures=len(report["failures"]),
            scope=report["evidence_scope"],
            sha=report["report_sha256"],
        )
    )
    if report["failures"]:
        for failure in report["failures"]:
            print(
                "FAIL_CLOSED[{code}] {message}".format(**failure),
                file=sys.stderr,
            )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
