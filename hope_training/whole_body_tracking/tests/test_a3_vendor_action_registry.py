from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import re
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/a3_vendor_action_registry.py"
)
SPEC = importlib.util.spec_from_file_location("a3_vendor_action_registry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


_MATERIALIZED_LAYER_NAMES = (
    "identity_repin_producer",
    "identity_prototype",
    "identity_repin_receipt",
    "identity_manifest",
    "required_identity_manifest",
    "runtime_contract",
    "runtime_authority_receipt",
    "dynamic_ready_candidate",
    "nominal_hold_receipt",
    "contact_bundle",
    "fixed_domain_initial_receipt",
    "reward_economy_receipt",
)

_PLANNED_ARTIFACT_NAMES = tuple(
    name for name in _MATERIALIZED_LAYER_NAMES if name != "identity_repin_producer"
)

_R7_MATERIALIZED_LAYER_NAMES = frozenset(
    {
        "identity_repin_producer",
        "identity_prototype",
        "identity_repin_receipt",
        "identity_manifest",
        "required_identity_manifest",
        "runtime_contract",
        "runtime_authority_receipt",
        "dynamic_ready_candidate",
        "nominal_hold_receipt",
        "contact_bundle",
        "reward_economy_receipt",
    }
)


def test_r7_registry_exposes_only_the_materialized_identity_bootstrap() -> None:
    for action_id in sorted(R.ALLOWED_ACTION_IDS):
        config = R.get_action_config(action_id)

        assert R.require_identity_source_commit(config) == (
            "c75573f37b5d4c11361e1079deb029ae52224f75"
        )
        with pytest.raises(R.VendorActionRegistryError, match="identity source"):
            R.require_identity_source_commit(
                replace(config, identity_source_commit=None)
            )
        assert R.stable_pin(config.stable_motion)["sha256"]
        assert R.stable_pin(config.stable_source_manifest)["sha256"]
        assert R.stable_pin(config.stable_source_prototype)["sha256"]

        for layer_name in _MATERIALIZED_LAYER_NAMES:
            pin = getattr(config, layer_name)
            assert pin.path
            if layer_name in _R7_MATERIALIZED_LAYER_NAMES:
                materialized = R.require_materialized_pin(
                    pin,
                    action_id=action_id,
                    layer=layer_name,
                )
                assert materialized == {"path": pin.path, "sha256": pin.sha256}
            else:
                assert pin.sha256 is None
                with pytest.raises(
                    R.VendorActionRegistryError,
                    match="awaiting code-pinned",
                ):
                    R.require_materialized_pin(
                        pin,
                        action_id=action_id,
                        layer=layer_name,
                    )


def test_r7_planned_paths_are_epoch_scoped_and_action_isolated() -> None:
    loop = R.get_action_config("bh_loop_c")
    block = R.get_action_config("bh_block")

    for name in _PLANNED_ARTIFACT_NAMES:
        loop_path = getattr(loop, name).path
        block_path = getattr(block, name).path
        expected_epoch = "20260801_r7"
        if name == "reward_economy_receipt":
            assert loop.reward_economy_receipt is block.reward_economy_receipt
            assert loop_path == block_path
            assert expected_epoch in loop_path
            assert loop_path.endswith("/reward_economy.v1.json")
            continue
        assert expected_epoch in loop_path
        assert expected_epoch in block_path
        assert "20260801_r5" not in loop_path
        assert "20260801_r5" not in block_path
        assert "bh_loop_c" in loop_path
        assert "bh_block" not in loop_path
        assert "bh_block" in block_path
        assert "bh_loop_c" not in block_path
        assert loop_path != block_path

    loop_identity = R.action_source_identity(loop)
    block_identity = R.action_source_identity(block)
    assert loop_identity["planned_paths"] != block_identity["planned_paths"]
    assert set(loop_identity["planned_paths"]) == {
        "identity_prototype",
        "identity_repin_receipt",
        "identity_manifest",
        "required_identity_manifest",
        "runtime_contract",
        "runtime_authority_receipt",
        "dynamic_ready_candidate",
        "nominal_hold_receipt",
        "contact_bundle",
        "fixed_domain_initial_receipt",
        "reward_economy_receipt",
    }
    loop_sha = R.action_source_identity_sha256(loop)
    block_sha = R.action_source_identity_sha256(block)
    assert re.fullmatch(r"[0-9a-f]{64}", loop_sha)
    assert re.fullmatch(r"[0-9a-f]{64}", block_sha)
    assert loop_sha != block_sha


def test_downstream_output_digest_repins_do_not_invalidate_identity_source() -> None:
    original = R.get_action_config("bh_loop_c")
    original_identity = R.action_source_identity(original)
    original_sha = R.action_source_identity_sha256(original)
    repinned = replace(
        original,
        dynamic_ready_candidate=R.ArtifactPin(
            original.dynamic_ready_candidate.path, "1" * 64
        ),
        nominal_hold_receipt=R.ArtifactPin(
            original.nominal_hold_receipt.path, "2" * 64
        ),
        contact_bundle=R.ArtifactPin(
            original.contact_bundle.path.replace(
                "pending", "0123456789ab"
            ),
            "3" * 64,
        ),
        fixed_domain_initial_receipt=R.ArtifactPin(
            original.fixed_domain_initial_receipt.path, "4" * 64
        ),
        reward_economy_receipt=R.ArtifactPin(
            original.reward_economy_receipt.path, "5" * 64
        ),
    )

    assert R.action_source_identity(repinned) == original_identity
    assert R.action_source_identity_sha256(repinned) == original_sha


@pytest.mark.parametrize(
    "field",
    (
        "dynamic_ready_candidate",
        "nominal_hold_receipt",
        "contact_bundle",
        "fixed_domain_initial_receipt",
        "reward_economy_receipt",
    ),
)
def test_downstream_output_path_change_invalidates_identity_source(
    field: str,
) -> None:
    original = R.get_action_config("bh_loop_c")
    changed = replace(
        original,
        **{field: R.ArtifactPin(f"configs/wrong_epoch/{field}.json", None)},
    )

    assert R.action_source_identity(changed) != R.action_source_identity(original)
    assert R.action_source_identity_sha256(changed) != (
        R.action_source_identity_sha256(original)
    )


@pytest.mark.parametrize(
    ("action_id", "old_r5_path", "old_r5_sha256"),
    (
        (
            "bh_loop_c",
            "configs/n1_contact_vendor_a3_20260801_r5/bh_loop_c/"
            "bh_loop_c.bundle.v2.bf0ae909e108.json",
            "bf0ae909e108ff7d96a9173fe30d69716a23379682e766e98acf615b3a8ac4d5",
        ),
        (
            "bh_block",
            "configs/n1_contact_vendor_a3_20260801_r5/bh_block/"
            "bh_block.bundle.v2.497c4bbd5658.json",
            "497c4bbd5658f32e2e38a7f529207ce303121d36669cb1c3fb654b743437ae8a",
        ),
    ),
)
def test_r7_contact_pin_is_planned_and_old_r5_cannot_authorize_consumer(
    action_id: str, old_r5_path: str, old_r5_sha256: str
) -> None:
    config = R.get_action_config(action_id)
    old_r5_pin = {"path": old_r5_path, "sha256": old_r5_sha256}

    assert "20260801_r7" in config.contact_bundle.path
    assert config.contact_bundle.sha256 is None
    with pytest.raises(R.VendorActionRegistryError, match="awaiting code-pinned"):
        R.require_materialized_pin(
            config.contact_bundle,
            action_id=action_id,
            layer="contact bundle",
        )
    assert old_r5_pin["path"] != config.contact_bundle.path
    assert R.ArtifactPin(old_r5_path, old_r5_sha256) not in {
        candidate.contact_bundle for candidate in R.ACTION_CONFIGS.values()
    }
