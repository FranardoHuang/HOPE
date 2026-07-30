from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/action_ball_action_set_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "action_ball_action_set_contract_under_test", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


N5_ORDER = [
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
]


def _source(profile_id, row, policies=None):
    return (
        "ACTION_SET_PROFILE_POLICIES = {!r}\n"
        "ACTION_SET_CONTRACTS = {!r}\n"
    ).format({} if policies is None else policies, {profile_id: row}).encode(
        "utf-8"
    )


def _case(n):
    if n == 5:
        profile_id = "fresh_upper_nomove_n5_v3"
        order = list(N5_ORDER)
        scope = "upper"
        policies = {
            profile_id: {
                "expected_n": 5,
                "scope": "upper",
                "mobility_mode": "no_move",
                "required_action_ids": list(order),
                "retired_action_ids": ["fh_loop", "fh_block_syn"],
            }
        }
    else:
        profile_id = "fixture_full_nomove_n{}_v1".format(n)
        order = ["action_{:03d}".format(index) for index in range(n)]
        scope = "full"
        policies = {}
    uids = [10000 + index for index in range(n)]
    manifest_path = "configs/fixture_n{}.json".format(n)
    manifest = {
        "schema_version": 3,
        "manifest_id": "fixture_n{}".format(n),
        "mobility_mode": "no_move",
        "action_order": order,
        "prototype": {
            "path": "configs/prototype_n{}.json".format(n),
            "sha256": "a" * 64,
            "scope": scope,
        },
        "actions": [
            {"action_id": action_id, "action_uid": uid}
            for action_id, uid in zip(order, uids)
        ],
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    row = {
        "profile_id": profile_id,
        "expected_n": n,
        "scope": scope,
        "mobility_mode": "no_move",
        "ordered_action_ids": order,
        "ordered_action_uids": uids,
        "order_uid_digest_sha256": M.order_uid_digest(order, uids),
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "experiment_name": "fixture_action_ball_n{}".format(n),
    }
    return profile_id, row, policies, manifest, manifest_bytes


@pytest.mark.parametrize("n", [1, 5, 73])
def test_n1_n5_n73_registered_contract_is_exact(n):
    profile_id, row, policies, manifest, manifest_bytes = _case(n)
    contract = M.load_contract_from_source(
        _source(profile_id, row, policies), profile_id
    )
    assert contract["expected_n"] == n
    assert contract["actor_obs_contract"] == (
        "action_ball_table_pose_n{}".format(n)
    )
    assert contract["actor_obs_width"] == 190 + n
    assert contract["namespace_identity"] == "n{}-{}".format(
        n, row["order_uid_digest_sha256"][:12]
    )
    assert contract["contract_sha256"] == M.canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    M.verify_manifest_identity(contract, manifest, manifest_bytes)


def test_unregistered_profile_cannot_be_self_certified():
    profile_id, row, policies, _manifest, _bytes = _case(1)
    with pytest.raises(M.ActionSetContractError, match="unregistered"):
        M.load_contract_from_source(
            _source(profile_id, row, policies), "caller_claimed_profile"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "uid_order",
        "manifest_sha",
        "actor_count",
    ],
)
def test_contract_identity_tamper_is_rejected(mutation):
    profile_id, row, policies, _manifest, _bytes = _case(73)
    row = dict(row)
    if mutation == "digest":
        row["order_uid_digest_sha256"] = "0" * 64
    elif mutation == "uid_order":
        row["ordered_action_uids"] = list(reversed(row["ordered_action_uids"]))
    elif mutation == "manifest_sha":
        row["manifest_sha256"] = "not-a-sha"
    else:
        row["expected_n"] = 72
    with pytest.raises(M.ActionSetContractError):
        M.load_contract_from_source(_source(profile_id, row, policies), profile_id)


def test_n5_retired_action_policy_is_profile_specific():
    profile_id, row, policies, _manifest, _bytes = _case(5)
    row = dict(row)
    order = list(row["ordered_action_ids"])
    order[-1] = "fh_loop"
    row["ordered_action_ids"] = order
    row["order_uid_digest_sha256"] = M.order_uid_digest(
        order, row["ordered_action_uids"]
    )
    policies = dict(policies)
    policies[profile_id] = dict(policies[profile_id])
    policies[profile_id]["required_action_ids"] = order
    with pytest.raises(M.ActionSetContractError, match="retired"):
        M.load_contract_from_source(_source(profile_id, row, policies), profile_id)


def test_manifest_reorder_or_uid_drift_is_rejected():
    profile_id, row, policies, manifest, manifest_bytes = _case(73)
    contract = M.load_contract_from_source(
        _source(profile_id, row, policies), profile_id
    )
    for mutation in ("order", "uid"):
        drifted = json.loads(json.dumps(manifest))
        if mutation == "order":
            drifted["actions"][0], drifted["actions"][1] = (
                drifted["actions"][1],
                drifted["actions"][0],
            )
        else:
            drifted["actions"][0]["action_uid"] += 1
        with pytest.raises(M.ActionSetContractError):
            M.verify_manifest_identity(contract, drifted, manifest_bytes)
