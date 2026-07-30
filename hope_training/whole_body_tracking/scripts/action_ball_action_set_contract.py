#!/usr/bin/env python3
"""Code-owned ActionBall action-set contracts.

The launch spec is deliberately not an authority for action count, order,
stable UIDs, scope, mobility, or manifest identity.  It may name one profile
from :data:`ACTION_SET_CONTRACTS`; this module validates that literal row and
derives the actor-observation and namespace identities from it.

Concrete scientific rows are added only after their exact manifest has been
committed.  In particular, callers must fail closed for an unregistered
profile instead of reconstructing a row from a launch spec.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, Mapping, Union


SCHEMA_VERSION = 1
CONTRACT_KIND = "whole_body_tracking.action_ball.action_set_contract"
REGISTRY_VARIABLE = "ACTION_SET_CONTRACTS"
PROFILE_POLICY_VARIABLE = "ACTION_SET_PROFILE_POLICIES"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ACTOR_OBS_CONTRACT_PREFIX = (
    "action_ball_table_pose_twist_heading_task_n"
)
ACTOR_OBS_BASE_WIDTH = 193

CONTRACT_KEYS = frozenset(
    {
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
    }
)
PROFILE_POLICY_KEYS = frozenset(
    {
        "expected_n",
        "scope",
        "mobility_mode",
        "required_action_ids",
        "retired_action_ids",
    }
)


class ActionSetContractError(ValueError):
    """A malformed, missing, or internally inconsistent action-set contract."""


# Profile-level policy is code-owned independently of a particular manifest.
# The concrete N5 contract row is registered only after the final exact
# manifest bytes and stable UID order have been committed.
ACTION_SET_PROFILE_POLICIES = {
    "fresh_upper_nomove_n5_v3": {
        "expected_n": 5,
        "scope": "upper",
        "mobility_mode": "no_move",
        "required_action_ids": [
            "bh_loop_c",
            "v12_forehand_block",
            "bh_block",
            "s0_highpress",
            "fh_loop_high",
        ],
        "retired_action_ids": ["fh_loop", "fh_block_syn"],
    },
}


# Do not add a row without a committed exact manifest.  N1 and N73 scientific
# identity is intentionally absent until their final manifests are produced.
ACTION_SET_CONTRACTS = {}


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActionSetContractError(
            "action-set contract is not canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by all contract consumers."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def order_uid_digest(
    ordered_action_ids: list[str], ordered_action_uids: list[int]
) -> str:
    """Bind every action index to its stable ID and stable UID."""

    if len(ordered_action_ids) != len(ordered_action_uids):
        raise ActionSetContractError("action ID/UID order lengths differ")
    rows = [
        {"index": index, "action_id": action_id, "action_uid": action_uid}
        for index, (action_id, action_uid) in enumerate(
            zip(ordered_action_ids, ordered_action_uids)
        )
    ]
    return canonical_sha256(
        {"schema_version": SCHEMA_VERSION, "ordered_actions": rows}
    )


def _literal_assignment(source: bytes, variable: str) -> Any:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ActionSetContractError(
            "action-set contract source must be UTF-8"
        ) from exc
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ActionSetContractError(
            "action-set contract source is not valid Python"
        ) from exc
    values = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            values.append(node.value)
    if len(values) != 1:
        raise ActionSetContractError(
            "{} must have exactly one top-level assignment".format(variable)
        )
    try:
        return ast.literal_eval(values[0])
    except (ValueError, SyntaxError) as exc:
        raise ActionSetContractError(
            "{} must be a Python literal".format(variable)
        ) from exc


def _validate_repo_relative_path(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ActionSetContractError("{} must be a non-empty string".format(name))
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ActionSetContractError(
            "{} must be a normalized repo-relative POSIX path".format(name)
        )
    return value


def validate_profile_policies(value: Any) -> Dict[str, Dict[str, Any]]:
    if type(value) is not dict:
        raise ActionSetContractError(
            "{} must be a literal dict".format(PROFILE_POLICY_VARIABLE)
        )
    normalized: Dict[str, Dict[str, Any]] = {}
    for profile_id, raw in value.items():
        if type(profile_id) is not str or not SAFE_PROFILE_RE.fullmatch(profile_id):
            raise ActionSetContractError("profile policy ID is invalid")
        if type(raw) is not dict or set(raw) != PROFILE_POLICY_KEYS:
            raise ActionSetContractError(
                "profile policy {} has wrong keys".format(profile_id)
            )
        expected_n = raw["expected_n"]
        required = raw["required_action_ids"]
        retired = raw["retired_action_ids"]
        if type(expected_n) is not int or isinstance(expected_n, bool) or expected_n < 1:
            raise ActionSetContractError(
                "profile policy {} expected_n is invalid".format(profile_id)
            )
        for name, rows in (
            ("required_action_ids", required),
            ("retired_action_ids", retired),
        ):
            if (
                type(rows) is not list
                or any(type(item) is not str or not item for item in rows)
                or len(rows) != len(set(rows))
            ):
                raise ActionSetContractError(
                    "profile policy {} {} is invalid".format(profile_id, name)
                )
        if required and len(required) != expected_n:
            raise ActionSetContractError(
                "profile policy {} required order length differs from N".format(
                    profile_id
                )
            )
        if set(required).intersection(retired):
            raise ActionSetContractError(
                "profile policy {} requires a retired action".format(profile_id)
            )
        if raw["scope"] not in ("upper", "full"):
            raise ActionSetContractError(
                "profile policy {} scope is invalid".format(profile_id)
            )
        if raw["mobility_mode"] not in ("no_move", "move"):
            raise ActionSetContractError(
                "profile policy {} mobility_mode is invalid".format(profile_id)
            )
        normalized[profile_id] = {
            "expected_n": expected_n,
            "scope": raw["scope"],
            "mobility_mode": raw["mobility_mode"],
            "required_action_ids": list(required),
            "retired_action_ids": list(retired),
        }
    return normalized


def validate_contract(
    value: Any,
    *,
    profile_id: str,
    profile_policies: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate one literal row and return its canonical derived identity."""

    if type(value) is not dict or set(value) != CONTRACT_KEYS:
        raise ActionSetContractError(
            "action-set contract {} has wrong keys".format(profile_id)
        )
    if value["profile_id"] != profile_id or not SAFE_PROFILE_RE.fullmatch(profile_id):
        raise ActionSetContractError("action-set contract profile_id mismatch")
    expected_n = value["expected_n"]
    if type(expected_n) is not int or isinstance(expected_n, bool) or expected_n < 1:
        raise ActionSetContractError("action-set expected_n is invalid")
    action_ids = value["ordered_action_ids"]
    action_uids = value["ordered_action_uids"]
    if (
        type(action_ids) is not list
        or len(action_ids) != expected_n
        or any(type(item) is not str or not item for item in action_ids)
        or len(action_ids) != len(set(action_ids))
    ):
        raise ActionSetContractError("ordered_action_ids is not exact unique N")
    if (
        type(action_uids) is not list
        or len(action_uids) != expected_n
        or any(
            type(item) is not int or isinstance(item, bool) or item < 0
            for item in action_uids
        )
        or len(action_uids) != len(set(action_uids))
    ):
        raise ActionSetContractError("ordered_action_uids is not exact unique N")
    if value["scope"] not in ("upper", "full"):
        raise ActionSetContractError("action-set scope is invalid")
    if value["mobility_mode"] not in ("no_move", "move"):
        raise ActionSetContractError("action-set mobility_mode is invalid")
    digest = value["order_uid_digest_sha256"]
    if (
        type(digest) is not str
        or not SHA256_RE.fullmatch(digest)
        or digest != order_uid_digest(action_ids, action_uids)
    ):
        raise ActionSetContractError("order_uid_digest_sha256 is invalid")
    manifest_path = _validate_repo_relative_path(
        value["manifest_path"], name="manifest_path"
    )
    manifest_sha = value["manifest_sha256"]
    if type(manifest_sha) is not str or not SHA256_RE.fullmatch(manifest_sha):
        raise ActionSetContractError("manifest_sha256 is invalid")
    experiment_name = value["experiment_name"]
    if (
        type(experiment_name) is not str
        or not SAFE_EXPERIMENT_RE.fullmatch(experiment_name)
    ):
        raise ActionSetContractError("experiment_name is invalid")

    policy = profile_policies.get(profile_id)
    if policy is not None:
        if (
            policy["expected_n"] != expected_n
            or policy["scope"] != value["scope"]
            or policy["mobility_mode"] != value["mobility_mode"]
        ):
            raise ActionSetContractError(
                "action-set contract violates profile policy"
            )
        required = list(policy["required_action_ids"])
        if required and action_ids != required:
            raise ActionSetContractError(
                "action-set order violates profile policy"
            )
        stale = sorted(set(action_ids).intersection(policy["retired_action_ids"]))
        if stale:
            raise ActionSetContractError(
                "retired actions are forbidden by profile policy: {}".format(
                    stale
                )
            )

    base = {
        "profile_id": profile_id,
        "expected_n": expected_n,
        "scope": value["scope"],
        "mobility_mode": value["mobility_mode"],
        "ordered_action_ids": list(action_ids),
        "ordered_action_uids": list(action_uids),
        "order_uid_digest_sha256": digest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "experiment_name": experiment_name,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        **base,
        "actor_obs_contract": "{}{}".format(
            ACTOR_OBS_CONTRACT_PREFIX, expected_n
        ),
        "actor_obs_width": ACTOR_OBS_BASE_WIDTH + expected_n,
        "namespace_identity": "n{}-{}".format(expected_n, digest[:12]),
    }
    identity["contract_sha256"] = canonical_sha256(identity)
    return identity


def validate_registry(
    contracts: Any, profile_policies: Any
) -> Dict[str, Dict[str, Any]]:
    if type(contracts) is not dict:
        raise ActionSetContractError(
            "{} must be a literal dict".format(REGISTRY_VARIABLE)
        )
    policies = validate_profile_policies(profile_policies)
    normalized: Dict[str, Dict[str, Any]] = {}
    for profile_id, row in contracts.items():
        if type(profile_id) is not str:
            raise ActionSetContractError("action-set contract key is invalid")
        normalized[profile_id] = validate_contract(
            row, profile_id=profile_id, profile_policies=policies
        )
    return normalized


def load_contract_from_source(
    source: bytes, profile_id: str
) -> Dict[str, Any]:
    """Load exactly one registered profile from literal committed source."""

    if type(profile_id) is not str or not SAFE_PROFILE_RE.fullmatch(profile_id):
        raise ActionSetContractError("requested contract profile is invalid")
    registry = validate_registry(
        _literal_assignment(source, REGISTRY_VARIABLE),
        _literal_assignment(source, PROFILE_POLICY_VARIABLE),
    )
    try:
        return registry[profile_id]
    except KeyError as exc:
        raise ActionSetContractError(
            "unregistered action-set contract profile: {}".format(profile_id)
        ) from exc


def load_registered_contract(profile_id: str) -> Dict[str, Any]:
    """Load one profile from the executing module's literal registry."""

    return validate_registry(
        ACTION_SET_CONTRACTS, ACTION_SET_PROFILE_POLICIES
    )[profile_id]


def verify_manifest_identity(
    contract: Mapping[str, Any], manifest: Any, manifest_bytes: bytes
) -> None:
    """Cross-check exact manifest bytes, action IDs/UIDs, scope and mobility."""

    if hashlib.sha256(manifest_bytes).hexdigest() != contract["manifest_sha256"]:
        raise ActionSetContractError("manifest bytes differ from contract SHA")
    if type(manifest) is not dict or manifest.get("schema_version") != 3:
        raise ActionSetContractError("manifest schema_version must be 3")
    if manifest.get("mobility_mode") != contract["mobility_mode"]:
        raise ActionSetContractError("manifest mobility differs from contract")
    if manifest.get("action_order") != contract["ordered_action_ids"]:
        raise ActionSetContractError("manifest action order differs from contract")
    prototype = manifest.get("prototype")
    if type(prototype) is not dict or prototype.get("scope") != contract["scope"]:
        raise ActionSetContractError("manifest scope differs from contract")
    actions = manifest.get("actions")
    if type(actions) is not list or len(actions) != contract["expected_n"]:
        raise ActionSetContractError("manifest actions do not have exact N")
    ids = [row.get("action_id") if type(row) is dict else None for row in actions]
    uids = [
        row.get("action_uid") if type(row) is dict else None for row in actions
    ]
    if ids != contract["ordered_action_ids"]:
        raise ActionSetContractError("manifest action IDs differ from contract")
    if uids != contract["ordered_action_uids"]:
        raise ActionSetContractError("manifest action UIDs differ from contract")


def load_contract_file(
    path: Union[str, Path], profile_id: str
) -> Dict[str, Any]:
    """Convenience loader for trusted callers that already pinned the file."""

    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ActionSetContractError(
            "cannot read action-set contract source: {}".format(exc)
        ) from exc
    return load_contract_from_source(source, profile_id)


__all__ = [
    "ACTION_SET_CONTRACTS",
    "ACTION_SET_PROFILE_POLICIES",
    "ActionSetContractError",
    "CONTRACT_KIND",
    "REGISTRY_VARIABLE",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "load_contract_file",
    "load_contract_from_source",
    "load_registered_contract",
    "order_uid_digest",
    "validate_contract",
    "validate_profile_policies",
    "validate_registry",
    "verify_manifest_identity",
]
