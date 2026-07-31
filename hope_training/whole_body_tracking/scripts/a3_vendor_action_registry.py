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
    contact_bundle: ArtifactPin


class VendorActionRegistryError(ValueError):
    """Raised when an action or materialized layer is not code-authorized."""


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
    identity_source_commit="a2882d6824cfdb6c05d158797fd5d6a723bc504e",
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_identity_manifest.py",
        "b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a",
    ),
    identity_prototype=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_loop_c.vendor_identity.prototype.v2.json",
        "1fe968823f4349c97e2fbefcfb9c92acdc8a87b4b58caff58acdd110c3f9baec",
    ),
    identity_repin_receipt=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_loop_c.identity_bootstrap_repin.v1.json",
        "ec13d8a3ace8c052dd653c98f69baa5073bf7aa3b9bf6dcf2a65779238d486fb",
    ),
    identity_manifest=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_loop_c.vendor_identity.manifest.v3.json",
        "74c1a1a3e18002d20386180e7637ec92f27bb0a9bc2b5e1ae027c3a47f48c11e",
    ),
    required_identity_manifest=ArtifactPin(
        "configs/a3_vendor_runtime_contract_20260731_r2/"
        "required_identity.bh_loop_c.v1.json",
        "258334281fcba4f19165bb3be170ce1b217e5d213275c2ea4a0ca362fe8abe94",
    ),
    runtime_contract=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260731_r2/"
        "bh_loop_c.shared_ready.training_contract.json",
        "637d1f633cb4937160f231013c36ec4126e72b578fdccc6bcc0e994b9cb619f9",
    ),
    runtime_authority_receipt=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260731_r2/"
        "bh_loop_c.vendor_runtime_authority.v1.json",
        "04418ced9158303745174fd2083ef5ba54a0d38807cf0a61b839cb0b09dbe057",
    ),
    contact_bundle=ArtifactPin(
        "configs/n1_contact_vendor_a3_20260731_r2/bh_loop_c/"
        "bh_loop_c.bundle.v2.bd91b652198f.json",
        "bd91b652198f28f2f5b9499770a8f5b2730da7a1862ad661f047507250a46ecb",
    ),
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
    identity_source_commit="a2882d6824cfdb6c05d158797fd5d6a723bc504e",
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_identity_manifest.py",
        "b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a",
    ),
    identity_prototype=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_block.vendor_identity.prototype.v2.json",
        "e53a509b5fcc9c6daa86360192be343670196b00b6c7a8fcad311870c7b88346",
    ),
    identity_repin_receipt=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_block.identity_bootstrap_repin.v1.json",
        "b29c640b07a940c1e41de9bb0caf70bd2ef36f9ee6da7c2ff2fa0b631a0868c8",
    ),
    identity_manifest=ArtifactPin(
        "configs/a3_vendor_identity_bootstrap_20260731_r2/"
        "bh_block.vendor_identity.manifest.v3.json",
        "6e2f7b15424c664cf01e6e3b52148c8297f1e57e661f41d4aa3c238202a00145",
    ),
    required_identity_manifest=ArtifactPin(
        "configs/a3_vendor_runtime_contract_20260731_r2/"
        "required_identity.bh_block.v1.json",
        "6fead111c51645ffb335d8beea1e3bfa7d549468003e91c315f8f73f803ac1a3",
    ),
    runtime_contract=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260731_r2/"
        "bh_block.shared_ready.training_contract.json",
        "9118719220307a5667a8c51b7349b17fcbe9d3a34e9e03a298833f1df36f2061",
    ),
    runtime_authority_receipt=ArtifactPin(
        "configs/a3_vendor_runtime_authority_20260731_r2/"
        "bh_block.vendor_runtime_authority.v1.json",
        "648182dfc2501fcf551eef7653ee37087cf80236f9c41dd6f95e6eaeb0eef53d",
    ),
    contact_bundle=ArtifactPin(
        "configs/n1_contact_vendor_a3_20260731_r2/bh_block/"
        "bh_block.bundle.v2.bbfb612a7e88.json",
        "bbfb612a7e882b2b5976e3cf41eb194512e05b1b7fecb7b534e4b45ec21af1bf",
    ),
)

ACTION_CONFIGS: Mapping[str, VendorActionConfig] = MappingProxyType(
    {_LOOP.action_id: _LOOP, _BLOCK.action_id: _BLOCK}
)
ALLOWED_ACTION_IDS = frozenset(ACTION_CONFIGS)
DEFAULT_ACTION_ID = _LOOP.action_id
REGISTRY_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)


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
        },
    }


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
