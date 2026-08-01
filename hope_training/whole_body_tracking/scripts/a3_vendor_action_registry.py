#!/usr/bin/env python3
"""Code-owned action registry for the A3-vendor N1 authority chain.

Only reviewed actions may enter the vendor identity/authority launch path.
Stable-v2 source pins are available before live runtime materialization; later
pins remain ``None`` until their exact tracked artifacts have been produced
and reviewed.  Consumers must call :func:`require_materialized_pin` (or check
the corresponding digest) and fail closed rather than accepting operator
supplied replacements.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class ArtifactPin:
    """One planned repo path and its optional reviewed file digest."""

    path: str
    sha256: Optional[str]


@dataclass(frozen=True)
class VendorActionConfig:
    """All action-specific pins across the vendor N1 lineage."""

    action_id: str
    scope: str
    stable_motion: ArtifactPin
    stable_source_manifest: ArtifactPin
    stable_source_prototype: ArtifactPin
    identity_source_commit: Optional[str]
    identity_repin_producer: ArtifactPin
    identity_prototype: ArtifactPin
    identity_repin_receipt: ArtifactPin
    identity_manifest: ArtifactPin
    required_identity_manifest: ArtifactPin
    runtime_contract: ArtifactPin
    runtime_authority_receipt: ArtifactPin
    dynamic_ready_candidate: ArtifactPin
    nominal_hold_receipt: ArtifactPin
    contact_bundle: ArtifactPin
    fixed_domain_initial_receipt: ArtifactPin
    reward_economy_receipt: ArtifactPin


class VendorActionRegistryError(ValueError):
    """Raised when an action or materialized layer is not code-authorized."""


_REWARD_ECONOMY_RECEIPT = ArtifactPin(
    "configs/n1_reward_economy_20260802_r9/reward_economy.v1.json",
    None,
)


_LOOP = VendorActionConfig(
    action_id="bh_loop_c",
    scope="upper",
    stable_motion=ArtifactPin(
        "assets/motions/fivebind_20260727/bh_loop_c_upper_stable_v2.npz",
        "0fa46ad66d57edd006b0a70a7de0542d8d53945ee3ae9802fdbd937555a0c85b",
    ),
    stable_source_manifest=ArtifactPin(
        "configs/n1_contact_20260730_stable_v2/"
        "bh_loop_c.manifest.v3.775f74183e58.json",
        "775f74183e58683df48f5f44084e89320736d1533a4d962f43f455664830d8e5",
    ),
    stable_source_prototype=ArtifactPin(
        "configs/n1_contact_20260730_stable_v2/"
        "bh_loop_c.upper.prototype.v2.1726d7825f1c.json",
        "1726d7825f1ce4d8a5b8e0491cff837c800474a1505bdb5f4ad79116b7a7f88e",
    ),
    identity_source_commit="69b4f12afd6a38c66ab93a91bcba0f08149134d3",
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_identity_manifest.py",
        "b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a",
    ),
    identity_prototype=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_loop_c.vendor_identity.prototype.v2.json",
        "8b81157567f20dc6ea919c2bec01ea6debd7870b10b325318848eb1ca315e68c",
    ),
    identity_repin_receipt=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_loop_c.identity_bootstrap_repin.v1.json",
        "df13d23f8b3013e02fd6e004d4d98e26e637a84f9aeae1a0c5ada1c32c79a262",
    ),
    identity_manifest=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_loop_c.vendor_identity.manifest.v3.json",
        "cdf61d75acec9d471e0cbe2f22e0422e6617d4e730678a861ce8998f9d799f8b",
    ),
    required_identity_manifest=ArtifactPin(
        "configs/a3_vendor_runtime_contract_20260802_r9/"
        "required_identity.bh_loop_c.v1.json",
        None,
    ),
    runtime_contract=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260802_r9/"
        "bh_loop_c.shared_ready.training_contract.json",
        None,
    ),
    runtime_authority_receipt=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260802_r9/"
        "bh_loop_c.vendor_runtime_authority.v1.json",
        None,
    ),
    dynamic_ready_candidate=ArtifactPin(
        "configs/a3_vendor_dynamic_ready_20260802_r9/"
        "bh_loop_c.dynamic_ready.v1.json",
        None,
    ),
    nominal_hold_receipt=ArtifactPin(
        "configs/a3_vendor_dynamic_ready_20260802_r9/"
        "bh_loop_c.nominal_hold.v1.json",
        None,
    ),
    contact_bundle=ArtifactPin(
        "configs/n1_contact_vendor_a3_20260802_r9/bh_loop_c/"
        "bh_loop_c.bundle.v2.pending.json",
        None,
    ),
    fixed_domain_initial_receipt=ArtifactPin(
        "configs/n1_fixed_domain_initial_20260802_r9/"
        "bh_loop_c.fixed_domain_initial.v1.json",
        None,
    ),
    reward_economy_receipt=_REWARD_ECONOMY_RECEIPT,
)

_BLOCK = VendorActionConfig(
    action_id="bh_block",
    scope="upper",
    stable_motion=ArtifactPin(
        "assets/motions/fivebind_20260727/bh_block_upper_stable_v2.npz",
        "cc9bbccd1b5b6207a0ce9677944ba27fa4a062a1eaa61886d802c9d21830caa0",
    ),
    stable_source_manifest=ArtifactPin(
        "configs/n1_contact_20260730_stable_v2/"
        "bh_block.manifest.v3.7b16eef89878.json",
        "7b16eef898780d388e71987ebd7332f5ebbffec72a7513042860d8196b87ddea",
    ),
    stable_source_prototype=ArtifactPin(
        "configs/n1_contact_20260730_stable_v2/"
        "bh_block.upper.prototype.v2.edb3a600e4fc.json",
        "edb3a600e4fcb35a9cb69b3741da5020d733132a3dd3d28b1272a34293481f2d",
    ),
    identity_source_commit="69b4f12afd6a38c66ab93a91bcba0f08149134d3",
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_identity_manifest.py",
        "b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a",
    ),
    identity_prototype=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_block.vendor_identity.prototype.v2.json",
        "58342842063905ecf0eada1864185af414c2d6c245ae4201551ac2b714a74f4c",
    ),
    identity_repin_receipt=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_block.identity_bootstrap_repin.v1.json",
        "7e57afa4b7a1b11b32f6aa9fe16b486508266ec450faa44bde2715ad0fdbcd34",
    ),
    identity_manifest=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260802_r9/"
        "bh_block.vendor_identity.manifest.v3.json",
        "c27026a2205eea8535895dcc2bcc77204f5c8960df0b1f66872b814f3a97f57a",
    ),
    required_identity_manifest=ArtifactPin(
        "configs/a3_vendor_runtime_contract_20260802_r9/"
        "required_identity.bh_block.v1.json",
        None,
    ),
    runtime_contract=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260802_r9/"
        "bh_block.shared_ready.training_contract.json",
        None,
    ),
    runtime_authority_receipt=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260802_r9/"
        "bh_block.vendor_runtime_authority.v1.json",
        None,
    ),
    dynamic_ready_candidate=ArtifactPin(
        "configs/a3_vendor_dynamic_ready_20260802_r9/"
        "bh_block.dynamic_ready.v1.json",
        None,
    ),
    nominal_hold_receipt=ArtifactPin(
        "configs/a3_vendor_dynamic_ready_20260802_r9/"
        "bh_block.nominal_hold.v1.json",
        None,
    ),
    contact_bundle=ArtifactPin(
        "configs/n1_contact_vendor_a3_20260802_r9/bh_block/"
        "bh_block.bundle.v2.pending.json",
        None,
    ),
    fixed_domain_initial_receipt=ArtifactPin(
        "configs/n1_fixed_domain_initial_20260802_r9/"
        "bh_block.fixed_domain_initial.v1.json",
        None,
    ),
    reward_economy_receipt=_REWARD_ECONOMY_RECEIPT,
)

ACTION_CONFIGS: Mapping[str, VendorActionConfig] = MappingProxyType(
    {_LOOP.action_id: _LOOP, _BLOCK.action_id: _BLOCK}
)
ALLOWED_ACTION_IDS = frozenset(ACTION_CONFIGS)
DEFAULT_ACTION_ID = _LOOP.action_id
REGISTRY_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)
FINAL_POLICY_NOISE_STD_OVERRIDE = "algo.policy.noise_std_type=log"


def get_action_config(action_id: object) -> VendorActionConfig:
    """Return one reviewed action or reject unknown/non-string values."""

    if type(action_id) is not str or action_id not in ACTION_CONFIGS:
        raise VendorActionRegistryError(
            "vendor action_id must be one of: "
            + ", ".join(sorted(ALLOWED_ACTION_IDS))
        )
    return ACTION_CONFIGS[action_id]


def require_materialized_pin(
    pin: ArtifactPin, *, action_id: str, layer: str
) -> Mapping[str, str]:
    """Return a complete code pin or fail until that layer is materialized."""

    if not pin.path or pin.sha256 is None:
        raise VendorActionRegistryError(
            f"vendor action {action_id!r} is awaiting code-pinned {layer} materialization"
        )
    return {"path": pin.path, "sha256": pin.sha256}


def stable_pin(pin: ArtifactPin) -> Mapping[str, str]:
    """Return one mandatory stable source pin (registry construction invariant)."""

    if not pin.path or pin.sha256 is None:  # pragma: no cover - programmer error
        raise AssertionError("stable vendor action source pin is incomplete")
    return {"path": pin.path, "sha256": pin.sha256}


def require_identity_source_commit(config: VendorActionConfig) -> str:
    """Return the exact identity-repin source commit or fail before launch."""

    value = config.identity_source_commit
    if value is None or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise VendorActionRegistryError(
            f"vendor action {config.action_id!r} is awaiting code-pinned "
            "identity source commit materialization"
        )
    return value


def action_source_identity(config: VendorActionConfig) -> Mapping[str, object]:
    """Return the immutable producer-facing subset of one action entry.

    Materialized digests, the identity producer pin, and their source commit
    are intentionally excluded: those values are filled in by a later artifact
    commit.  Producer receipts bind this subset so that adding the reviewed
    producer/output digests cannot invalidate the receipt that produced them.
    """

    return {
        "schema_version": 1,
        "action_id": config.action_id,
        "scope": config.scope,
        "stable_motion": dict(stable_pin(config.stable_motion)),
        "stable_source_manifest": dict(
            stable_pin(config.stable_source_manifest)
        ),
        "stable_source_prototype": dict(
            stable_pin(config.stable_source_prototype)
        ),
        "planned_paths": {
            "identity_prototype": config.identity_prototype.path,
            "identity_repin_receipt": config.identity_repin_receipt.path,
            "identity_manifest": config.identity_manifest.path,
            "required_identity_manifest": config.required_identity_manifest.path,
            "runtime_contract": config.runtime_contract.path,
            "runtime_authority_receipt": config.runtime_authority_receipt.path,
            "dynamic_ready_candidate": config.dynamic_ready_candidate.path,
            "nominal_hold_receipt": config.nominal_hold_receipt.path,
            # The contact producer intentionally publishes a content-addressed
            # basename that is unknowable before materialization.  Identity
            # therefore binds the epoch/action directory and naming rule, not
            # the temporary pending basename or later digest basename.
            "contact_bundle": _contact_bundle_output_slot(config),
            "fixed_domain_initial_receipt": (
                config.fixed_domain_initial_receipt.path
            ),
            "reward_economy_receipt": config.reward_economy_receipt.path,
        },
    }


def _contact_bundle_output_slot(config: VendorActionConfig) -> str:
    parent, separator, _ = config.contact_bundle.path.rpartition("/")
    if not separator or not parent:
        raise VendorActionRegistryError(
            f"vendor action {config.action_id!r} contact bundle path lacks a parent"
        )
    return (
        f"{parent}/{config.action_id}.bundle.v2."
        "<content_sha256_12>.json"
    )


def action_source_identity_sha256(config: VendorActionConfig) -> str:
    payload = json.dumps(
        action_source_identity(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def action_source_registry_pin(config: VendorActionConfig) -> Mapping[str, str]:
    return {
        "path": REGISTRY_REPO_PATH,
        "action_id": config.action_id,
        "source_identity_sha256": action_source_identity_sha256(config),
    }
